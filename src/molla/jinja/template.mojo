"""The public face of the engine: compile once, render many times.

A chat template is compiled when the model loads and rendered on every request,
so the split is the whole point. Compiling is where a template can be rejected,
and rejecting at load time is what stops a model from serving requests that
would misrender. Rendering after that cannot meet an unsupported construct,
because there is nothing left to meet.

The cache is keyed by the digest of the source rather than by the model that
supplied it. Two models shipping the same template share one compiled tree,
which they do more often than not because most Qwen derivatives ship the same
bytes, and a digest is cheap next to a parse.

Values come in as JSON, which is the shape they already have. Messages arrive
in a request body as JSON, tools arrive as JSON, and a tokenizer config holds
the special tokens as JSON. Converting once at the boundary means the evaluator
never has to know where a value came from.
"""

from molla.io.buffer import Buffer
from molla.json.dom import (
    Document,
    JS_ARRAY,
    JS_BOOL,
    JS_DOUBLE,
    JS_INT,
    JS_OBJECT,
    JS_STRING,
    NO_NODE,
    parse,
)
from molla.json.reader import Reader
from molla.jinja.ast import Tree
from molla.jinja.env import Env, Limits
from molla.jinja.eval import builtin_frame, render as render_tree
from molla.jinja.parser import parse_template
from molla.jinja.value import FALSE, NONE, TRUE, dict_set
from molla.sys.mem import keep
from molla.sys.sha256 import sha256_hex

comptime BIND_TEXT = 0
comptime BIND_JSON = 1
comptime BIND_FLAG = 2


struct Binding(Copyable, Movable):
    """One name the template will see, and what to bind it to."""

    var name: String
    var text: String
    var kind: Int

    def __init__(out self, name: String, text: String, kind: Int):
        self.name = name
        self.text = text
        self.kind = kind


def text_binding(name: String, value: String) -> Binding:
    """A plain string, bound as it stands with no escaping to think about."""
    return Binding(name, value, BIND_TEXT)


def json_binding(name: String, value: String) -> Binding:
    """A value written as JSON, which is how messages and tools arrive."""
    return Binding(name, value, BIND_JSON)


def flag_binding(name: String, value: Bool) -> Binding:
    return Binding(name, String("1") if value else String(""), BIND_FLAG)


struct Template(Movable):
    """A parsed template, ready to render as often as it is asked to."""

    var source: String
    var digest: String
    var tree: Tree

    def __init__(out self, source: String) raises:
        self.digest = sha256_hex(source.as_bytes())
        self.tree = parse_template(source)
        self.source = source

    def render(
        self, bindings: List[Binding], limits: Limits, now: Int = 0
    ) raises -> String:
        var env = Env(self.source, limits, 0)
        env.now = now
        var frame = env.push(builtin_frame(env))
        _install(env, frame, bindings)
        render_tree(env, self.tree, frame)
        return env.rendered()

    def render(self, bindings: List[Binding]) raises -> String:
        return self.render(bindings, Limits())

    def render_object(
        self, vars: String, limits: Limits, now: Int = 0
    ) raises -> String:
        """Render with one JSON object, each member of it bound by its name.

        This is the shape the server has anyway. A chat request carries the
        messages and the tools, the tokenizer config carries the special tokens,
        and the two go into one object rather than being taken apart into
        bindings and put back together.

        `now` is what `strftime_now` reads, in seconds since the epoch, and
        zero is the real clock. It exists because the conformance corpus has to
        get the same answer twice.
        """
        var env = Env(self.source, limits, 0)
        env.now = now
        var frame = env.push(builtin_frame(env))
        var doc = Document(0)
        var reader = Reader(0, 4096)
        var body = vars.as_bytes()
        if not parse(doc, reader, body):
            raise Error("the template variables are not valid json")
        if doc.kind(doc.root) != JS_OBJECT:
            raise Error("the template variables have to be a json object")
        var child = doc.first_child(doc.root)
        while child != NO_NODE:
            var name = String(StringSpan(unsafe_from_utf8=doc.key(child)))
            var value = _convert(env, doc, child)
            env.bind(frame, name, value)
            child = doc.next_sibling(child)
        keep(vars)
        render_tree(env, self.tree, frame)
        return env.rendered()

    def render_object(self, vars: String) raises -> String:
        return self.render_object(vars, Limits())


def _install(mut env: Env, frame: Int, bindings: List[Binding]) raises:
    for i in range(len(bindings)):
        var name = bindings[i].name
        var kind = bindings[i].kind
        if kind == BIND_TEXT:
            var made = env.heap.str(bindings[i].text)
            env.bind(frame, name, made)
        elif kind == BIND_FLAG:
            env.bind(frame, name, TRUE if bindings[i].text != "" else FALSE)
        else:
            var made = _from_json(env, bindings[i].text)
            env.bind(frame, name, made)


def _from_json(mut env: Env, text: String) raises -> Int:
    var doc = Document(0)
    var reader = Reader(0, 4096)
    var body = text.as_bytes()
    if not parse(doc, reader, body):
        raise Error("a value given to the template is not valid json: " + text)
    var made = _convert(env, doc, doc.root)
    # The document's strings are spans into `text`, so `text` has to still be
    # alive here even though nothing below reads the name.
    keep(text)
    return made


def _convert(mut env: Env, doc: Document, node: Int) raises -> Int:
    var kind = doc.kind(node)
    if node == NO_NODE:
        return NONE
    if kind == JS_STRING:
        return env.heap.str(String(StringSpan(unsafe_from_utf8=doc.text(node))))
    if kind == JS_INT:
        return env.heap.int(doc.as_int(node))
    if kind == JS_DOUBLE:
        return env.heap.float(doc.as_double(node))
    if kind == JS_BOOL:
        return TRUE if doc.as_bool(node) else FALSE
    if kind == JS_ARRAY:
        var made = env.heap.list()
        var child = doc.first_child(node)
        while child != NO_NODE:
            var element = _convert(env, doc, child)
            env.heap.push(made, element)
            child = doc.next_sibling(child)
        return made
    if kind == JS_OBJECT:
        var made = env.heap.dict()
        var child = doc.first_child(node)
        while child != NO_NODE:
            var key = String(StringSpan(unsafe_from_utf8=doc.key(child)))
            var value = _convert(env, doc, child)
            dict_set(env.heap, made, key, value)
            child = doc.next_sibling(child)
        return made
    return NONE


struct Cache(Movable):
    """Compiled templates, found by the digest of their source.

    A server holds a handful of models and each has one template, so this is a
    short list and a linear scan over it costs less than anything cleverer.
    There is no eviction, because entries are added when a model loads and a
    process that has loaded every model it will ever load is done adding.
    """

    var digests: List[String]
    var templates: List[Template]

    def __init__(out self):
        self.digests = List[String]()
        self.templates = List[Template]()

    def find(self, digest: String) -> Int:
        for i in range(len(self.digests)):
            if self.digests[i] == digest:
                return i
        return -1

    def compile(mut self, source: String) raises -> Int:
        """The index of the compiled template, parsing it only on a miss."""
        var digest = sha256_hex(source.as_bytes())
        var at = self.find(digest)
        if at >= 0:
            return at
        var made = Template(source)
        self.digests.append(digest)
        self.templates.append(made^)
        return len(self.templates) - 1

    def render(
        self, at: Int, bindings: List[Binding], limits: Limits
    ) raises -> String:
        return self.templates[at].render(bindings, limits)

    def render(self, at: Int, bindings: List[Binding]) raises -> String:
        return self.templates[at].render(bindings, Limits())
