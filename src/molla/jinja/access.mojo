"""Attribute access, item access, slicing, and everything callable that is not
a macro.

Jinja's two ways of reaching into a value are not the same way spelled twice,
and templates depend on the difference. `foo.bar` asks Python for an attribute
first and only falls back to a subscript, so on a dict `foo.items` is the bound
method and never the key called `items`. `foo['bar']` asks for the subscript
first and only falls back to an attribute. Both end at Undefined rather than an
error when nothing matches, which is why a template can test `{% if x.tools %}`
against a message that has no tools in it.

A bound method is a value like any other, so `{{ text.split }}` is legal and
prints something rather than calling anything. That falls out of storing the
method as a cell with the receiver in it, which is also what lets `loop.cycle`
carry the loop position and `loop.changed` carry what it last saw.

The globals live here too, because from the evaluator's side `range(3)` and
`text.split(',')` are the same shape: a callable cell, a list of arguments, and
a list of keyword names running alongside them.
"""

from molla.jinja.datefmt import strftime_now as _strftime_now
from molla.jinja.env import Env
from molla.jinja.strop import (
    capitalize_python,
    contains,
    count_of,
    ends_with,
    find,
    lower_ascii,
    lower_full,
    needs_unicode,
    replace,
    rsplit,
    split,
    split_whitespace,
    splitlines,
    starts_with,
    strip,
    title_python,
    upper_ascii,
    upper_full,
)
from molla.jinja.value import (
    Cell,
    FALSE,
    Heap,
    NONE,
    TRUE,
    TUPLE_FLAG,
    UNDEFINED,
    V_BUILTIN,
    V_DICT,
    V_LIST,
    V_NONE,
    V_STRING,
    V_UNDEFINED,
    _char_count,
    as_int,
    dict_get,
    dict_index,
    dict_set,
    equal,
    is_number,
    substring,
    to_string,
)

comptime B_RANGE = 1
comptime B_DICT = 2
comptime B_NAMESPACE = 3
comptime B_STRFTIME_NOW = 4
comptime B_RAISE_EXCEPTION = 5
comptime B_LOOP_CYCLE = 6
comptime B_LOOP_CHANGED = 7

comptime B_S_SPLIT = 20
comptime B_S_RSPLIT = 21
comptime B_S_STRIP = 22
comptime B_S_LSTRIP = 23
comptime B_S_RSTRIP = 24
comptime B_S_REPLACE = 25
comptime B_S_STARTSWITH = 26
comptime B_S_ENDSWITH = 27
comptime B_S_UPPER = 28
comptime B_S_LOWER = 29
comptime B_S_TITLE = 30
comptime B_S_CAPITALIZE = 31
comptime B_S_JOIN = 32
comptime B_S_SPLITLINES = 33
comptime B_S_FIND = 34
comptime B_S_COUNT = 35
comptime B_S_REMOVEPREFIX = 36
comptime B_S_REMOVESUFFIX = 37

comptime B_D_ITEMS = 50
comptime B_D_KEYS = 51
comptime B_D_VALUES = 52
comptime B_D_GET = 53

comptime B_L_INDEX = 64
comptime B_L_COUNT = 65

comptime B_UNSAFE = -1
"""A method the reference sandbox refuses to hand out.

`ImmutableSandboxedEnvironment` blocks anything that would modify a list or a
dict, so `messages.append(x)` raises there. It has to raise here too. A template
that quietly worked in molla and failed in `transformers` would be worse than
one that fails in both, because the prompt it produced would be one nobody else
could reproduce.
"""


def _string_method(name: String) -> Int:
    if name == "split":
        return B_S_SPLIT
    if name == "rsplit":
        return B_S_RSPLIT
    if name == "strip":
        return B_S_STRIP
    if name == "lstrip":
        return B_S_LSTRIP
    if name == "rstrip":
        return B_S_RSTRIP
    if name == "replace":
        return B_S_REPLACE
    if name == "startswith":
        return B_S_STARTSWITH
    if name == "endswith":
        return B_S_ENDSWITH
    if name == "upper":
        return B_S_UPPER
    if name == "lower":
        return B_S_LOWER
    if name == "title":
        return B_S_TITLE
    if name == "capitalize":
        return B_S_CAPITALIZE
    if name == "join":
        return B_S_JOIN
    if name == "splitlines":
        return B_S_SPLITLINES
    if name == "find":
        return B_S_FIND
    if name == "count":
        return B_S_COUNT
    if name == "removeprefix":
        return B_S_REMOVEPREFIX
    if name == "removesuffix":
        return B_S_REMOVESUFFIX
    return 0


def _dict_method(name: String) -> Int:
    if name == "items":
        return B_D_ITEMS
    if name == "keys":
        return B_D_KEYS
    if name == "values":
        return B_D_VALUES
    if name == "get":
        return B_D_GET
    return 0


def _list_method(name: String) -> Int:
    if (
        name == "append"
        or name == "extend"
        or name == "insert"
        or name == "pop"
        or name == "remove"
        or name == "clear"
        or name == "sort"
        or name == "reverse"
    ):
        return B_UNSAFE
    if name == "index":
        return B_L_INDEX
    if name == "count":
        return B_L_COUNT
    return 0


def bind_method(mut heap: Heap, id: Int, name: String, receiver: Int) -> Int:
    """A callable cell holding what it was reached through."""
    var cell = Cell(V_BUILTIN)
    cell.i = id
    cell.s = name
    cell.node = receiver
    return heap.add(cell^)


def global_builtin(mut heap: Heap, id: Int, name: String) -> Int:
    """A callable cell with no receiver, which is what a global is."""
    return bind_method(heap, id, name, -1)


def is_namespace(heap: Heap, v: Int) -> Bool:
    return heap.kind(v) == V_DICT and heap.cells[v].i == 1


def get_attr(mut env: Env, v: Int, name: String, at: Int) raises -> Int:
    """`x.name`, which asks for an attribute before it asks for a key."""
    var k = env.heap.kind(v)
    if k == V_UNDEFINED:
        env.fail(
            at,
            "'" + name + "' was read on a name that has no value",
        )
    if k == V_NONE:
        env.fail(at, "'" + name + "' was read on none")

    if k == V_STRING:
        var id = _string_method(name)
        if id != 0:
            return bind_method(env.heap, id, name, v)
        return UNDEFINED
    if k == V_LIST:
        var id = _list_method(name)
        if id == B_UNSAFE:
            env.fail(
                at,
                "'"
                + name
                + "' would modify a list, which the reference sandbox refuses",
            )
        if id != 0:
            return bind_method(env.heap, id, name, v)
        return UNDEFINED
    if k == V_DICT:
        # A namespace has no methods of its own, so every name on it is a key.
        # An ordinary dict does, and Jinja finds them before it finds keys.
        if not is_namespace(env.heap, v):
            var id = _dict_method(name)
            if id != 0:
                return bind_method(env.heap, id, name, v)
        var got = dict_get(env.heap, v, name)
        return got if got >= 0 else UNDEFINED
    return UNDEFINED


def get_attr_only(mut env: Env, v: Int, name: String, at: Int) raises -> Int:
    """What the `attr` filter does, which is `x.name` with no key fallback.

    The filter exists so that a template can reach a method on a mapping whose
    keys would otherwise be in the way, so skipping the key lookup is the whole
    behaviour rather than an omission.
    """
    var k = env.heap.kind(v)
    if k == V_DICT and not is_namespace(env.heap, v):
        var id = _dict_method(name)
        if id != 0:
            return bind_method(env.heap, id, name, v)
        return UNDEFINED
    return get_attr(env, v, name, at)


def get_item(mut env: Env, v: Int, key: Int, at: Int) raises -> Int:
    """`x[key]`, which asks for a key before it asks for an attribute."""
    var k = env.heap.kind(v)
    if k == V_UNDEFINED:
        env.fail(at, "a subscript was read on a name that has no value")
    if k == V_NONE:
        env.fail(at, "a subscript was read on none")

    if k == V_DICT:
        if env.heap.kind(key) == V_STRING:
            var name = env.heap.cells[key].s
            var got = dict_get(env.heap, v, name)
            if got >= 0:
                return got
            if not is_namespace(env.heap, v):
                var id = _dict_method(name)
                if id != 0:
                    return bind_method(env.heap, id, name, v)
        return UNDEFINED

    if k == V_LIST or k == V_STRING:
        if is_number(env.heap, key):
            var n = len(
                env.heap.cells[v].items
            ) if k == V_LIST else _char_count(env.heap.cells[v].s)
            var index = as_int(env.heap, key)
            if index < 0:
                index += n
            if index < 0 or index >= n:
                return UNDEFINED
            if k == V_LIST:
                return env.heap.cells[v].items[index]
            var one = substring(env.heap.cells[v].s, index, index + 1)
            return env.heap.str(one)
        if env.heap.kind(key) == V_STRING:
            var name = env.heap.cells[key].s
            var id = _list_method(name) if k == V_LIST else _string_method(name)
            if id == B_UNSAFE:
                env.fail(
                    at,
                    "'"
                    + name
                    + "' would modify a list, which the reference sandbox"
                    " refuses",
                )
            if id != 0:
                return bind_method(env.heap, id, name, v)
        return UNDEFINED

    return UNDEFINED


def _clamp(index: Int, n: Int, backwards: Bool) -> Int:
    """Python's slice clamping, which differs at the low end when stepping down.
    """
    var i = index
    if i < 0:
        i += n
        if i < 0:
            return -1 if backwards else 0
    elif i > n:
        return n - 1 if backwards else n
    return i


def get_slice(
    mut env: Env,
    v: Int,
    start: Int,
    stop: Int,
    step: Int,
    has_start: Bool,
    has_stop: Bool,
    has_step: Bool,
    at: Int,
) raises -> Int:
    """`x[a:b:c]` on a list or a string, with Python's rules for the missing
    parts and for a negative step."""
    var k = env.heap.kind(v)
    if k != V_LIST and k != V_STRING:
        env.fail(at, "this value cannot be sliced")

    var by = step if has_step else 1
    if by == 0:
        env.fail(at, "a slice step of zero is not a step")

    var n = len(env.heap.cells[v].items) if k == V_LIST else _char_count(
        env.heap.cells[v].s
    )
    var backwards = by < 0
    var from_ = _clamp(start, n, backwards) if has_start else (
        n - 1 if backwards else 0
    )
    var to = _clamp(stop, n, backwards) if has_stop else (
        -1 if backwards else n
    )

    var picked = List[Int]()
    var i = from_
    while (i > to) if backwards else (i < to):
        picked.append(i)
        i += by

    if k == V_LIST:
        var made = env.heap.list()
        for j in range(len(picked)):
            var element = env.heap.cells[v].items[picked[j]]
            env.heap.push(made, element)
        return made

    if by == 1:
        return env.heap.str(substring(env.heap.cells[v].s, from_, to))
    var out = String("")
    for j in range(len(picked)):
        out += substring(env.heap.cells[v].s, picked[j], picked[j] + 1)
    return env.heap.str(out)


def iterate(mut env: Env, v: Int, at: Int) raises -> List[Int]:
    """The elements a `for` walks, as a list of values.

    A dict iterates its keys, which is Python, and is why every template that
    wants both writes `.items()`. Materialising rather than streaming, because
    the loop object needs the length and `previtem` and `nextitem` up front and
    a chat conversation is tens of elements.
    """
    var k = env.heap.kind(v)
    var out = List[Int]()
    if k == V_LIST:
        for i in range(len(env.heap.cells[v].items)):
            out.append(env.heap.cells[v].items[i])
        return out^
    if k == V_DICT:
        for i in range(len(env.heap.cells[v].keys)):
            var key = env.heap.cells[v].keys[i]
            out.append(env.heap.str(key))
        return out^
    if k == V_STRING:
        var n = _char_count(env.heap.cells[v].s)
        for i in range(n):
            var one = substring(env.heap.cells[v].s, i, i + 1)
            out.append(env.heap.str(one))
        return out^
    if k == V_UNDEFINED:
        env.fail(at, "a loop was run over a name that has no value")
    env.fail(at, "a loop was run over something that cannot be looped over")
    return out^


def _no_keywords(
    mut env: Env, names: List[String], what: String, at: Int
) raises:
    for i in range(len(names)):
        if names[i] != "":
            env.fail(
                at,
                what
                + " does not take keyword arguments, and '"
                + names[i]
                + "' is one",
            )


def _want_count(
    mut env: Env, args: List[Int], low: Int, high: Int, what: String, at: Int
) raises:
    if len(args) < low or len(args) > high:
        env.fail(
            at,
            what
            + " was called with "
            + String(len(args))
            + " arguments, and it takes between "
            + String(low)
            + " and "
            + String(high),
        )


def _want_string(mut env: Env, v: Int, what: String, at: Int) raises -> String:
    if env.heap.kind(v) != V_STRING:
        env.fail(at, what + " takes a string, and this is not one")
    return env.heap.cells[v].s


def _receiver_string(env: Env, fn: Int) -> String:
    return env.heap.cells[env.heap.cells[callee].node].s


def _string_list(mut env: Env, pieces: List[String]) -> Int:
    var made = env.heap.list()
    for i in range(len(pieces)):
        var one = env.heap.str(pieces[i])
        env.heap.push(made, one)
    return made


def call_builtin(
    mut env: Env, callee: Int, args: List[Int], names: List[String], at: Int
) raises -> Int:
    """Everything callable that is not a macro, dispatched on the builtin id."""
    var id = env.heap.cells[callee].i
    var what = env.heap.cells[callee].s + "()"
    var me = env.heap.cells[callee].node

    if id == B_DICT or id == B_NAMESPACE:
        return _make_mapping(env, id, args, names, at)
    if id == B_RANGE:
        return _make_range(env, args, names, at)
    if id == B_STRFTIME_NOW:
        _no_keywords(env, names, what, at)
        _want_count(env, args, 1, 1, what, at)
        var pattern = _want_string(env, args[0], what, at)
        return env.heap.str(_strftime_now(pattern, env.now))
    if id == B_RAISE_EXCEPTION:
        _no_keywords(env, names, what, at)
        _want_count(env, args, 1, 1, what, at)
        raise Error(
            "the template called raise_exception: "
            + to_string(env.heap, args[0])
        )
    if id == B_LOOP_CYCLE:
        return _loop_cycle(env, callee, args, names, at)
    if id == B_LOOP_CHANGED:
        return _loop_changed(env, callee, args, names, at)

    _no_keywords(env, names, what, at)
    if id >= B_S_SPLIT and id <= B_S_REMOVESUFFIX:
        return _string_call(env, id, me, args, what, at)
    if id >= B_D_ITEMS and id <= B_D_GET:
        return _dict_call(env, id, me, args, what, at)
    if id == B_L_INDEX or id == B_L_COUNT:
        return _list_call(env, id, me, args, what, at)

    env.fail(at, "this value is not something that can be called")
    return UNDEFINED


def _make_mapping(
    mut env: Env, id: Int, args: List[Int], names: List[String], at: Int
) raises -> Int:
    """`dict(a=1)` and `namespace(a=1)`, which differ only in a flag.

    `dict()` also takes one mapping positionally, the way Python does, because
    a template copying a message with `dict(message, role='user')` is a thing
    that happens.
    """
    var made = env.heap.dict()
    if id == B_NAMESPACE:
        env.heap.cells[made].i = 1
    for i in range(len(args)):
        if names[i] != "":
            continue
        if env.heap.kind(args[i]) != V_DICT:
            env.fail(
                at,
                (
                    "a positional argument here has to be a mapping, and this"
                    " is not one"
                ),
            )
        for j in range(len(env.heap.cells[args[i]].keys)):
            var key = env.heap.cells[args[i]].keys[j]
            var value = env.heap.cells[args[i]].items[j]
            dict_set(env.heap, made, key, value)
    for i in range(len(args)):
        if names[i] == "":
            continue
        dict_set(env.heap, made, names[i], args[i])
    return made


def _make_range(
    mut env: Env, args: List[Int], names: List[String], at: Int
) raises -> Int:
    _no_keywords(env, names, String("range()"), at)
    _want_count(env, args, 1, 3, String("range()"), at)
    var one = len(args) == 1
    var start = 0 if one else as_int(env.heap, args[0])
    var stop = as_int(env.heap, args[0] if one else args[1])
    var step = as_int(env.heap, args[2]) if len(args) == 3 else 1
    if step == 0:
        env.fail(at, "range() was given a step of zero, which never ends")

    var made = env.heap.list()
    var i = start
    while (i > stop) if step < 0 else (i < stop):
        # Charged against the step budget, so a template asking for a billion
        # numbers stops at the budget instead of at the memory.
        env.step()
        var value = env.heap.int(i)
        env.heap.push(made, value)
        i += step
    return made


def _loop_cycle(
    mut env: Env, callee: Int, args: List[Int], names: List[String], at: Int
) raises -> Int:
    """`loop.cycle(a, b, c)`, which picks by the loop's position.

    The position is in the cell rather than looked up, because the cell is what
    the loop hands the body and updating one integer per iteration is cheaper
    than rebuilding the callable.
    """
    _no_keywords(env, names, String("loop.cycle()"), at)
    if len(args) == 0:
        env.fail(at, "loop.cycle() needs at least one thing to cycle through")
    # `scope` carries the loop position on this cell. It has no receiver, so
    # the field is free, and the loop writes it once per iteration.
    var index = env.heap.cells[callee].scope
    return args[index % len(args)]


def _loop_changed(
    mut env: Env, callee: Int, args: List[Int], names: List[String], at: Int
) raises -> Int:
    """`loop.changed(x)`, true when the arguments differ from last time."""
    _no_keywords(env, names, String("loop.changed()"), at)
    # `node` is the flag for whether anything has been seen yet and `items` is
    # what was seen. Both are free on this cell because it has no receiver, and
    # keeping the state here is what makes it survive across iterations while
    # the loop object around it is rebuilt.
    var seen = env.heap.cells[callee].node == 1
    var same = seen and len(env.heap.cells[callee].items) == len(args)
    if same:
        for i in range(len(args)):
            if not equal(env.heap, env.heap.cells[callee].items[i], args[i]):
                same = False
                break
    env.heap.cells[callee].node = 1
    env.heap.cells[callee].items.clear()
    for i in range(len(args)):
        env.heap.cells[callee].items.append(args[i])
    return FALSE if same else TRUE


def _string_call(
    mut env: Env, id: Int, me: Int, args: List[Int], what: String, at: Int
) raises -> Int:
    var s = env.heap.cells[me].s

    if id == B_S_SPLIT or id == B_S_RSPLIT:
        _want_count(env, args, 0, 2, what, at)
        var limit = as_int(env.heap, args[1]) if len(args) == 2 else -1
        var has_sep = len(args) >= 1 and env.heap.kind(args[0]) != V_NONE
        if not has_sep:
            # `split()` and `rsplit()` agree when there is no separator, right
            # down to which pieces a limit keeps, for every input where the
            # whitespace is single characters. They do not agree in general, and
            # this is the one place the shortcut is taken knowingly.
            return _string_list(env, split_whitespace(s, limit))
        var sep = _want_string(env, args[0], what, at)
        if sep.byte_length() == 0:
            env.fail(at, "split() was given an empty separator")
        if id == B_S_SPLIT:
            return _string_list(env, split(s, sep, limit))
        return _string_list(env, rsplit(s, sep, limit))

    if id == B_S_STRIP or id == B_S_LSTRIP or id == B_S_RSTRIP:
        _want_count(env, args, 0, 1, what, at)
        var chars = String("")
        if len(args) == 1 and env.heap.kind(args[0]) != V_NONE:
            chars = _want_string(env, args[0], what, at)
        var left = id != B_S_RSTRIP
        var right = id != B_S_LSTRIP
        return env.heap.str(strip(s, chars, left, right))

    if id == B_S_REPLACE:
        _want_count(env, args, 2, 3, what, at)
        var old = _want_string(env, args[0], what, at)
        var new = _want_string(env, args[1], what, at)
        var limit = as_int(env.heap, args[2]) if len(args) == 3 else -1
        return env.heap.str(replace(s, old, new, limit))

    if id == B_S_STARTSWITH or id == B_S_ENDSWITH:
        _want_count(env, args, 1, 1, what, at)
        # A tuple of candidates is allowed, because Python allows it and a
        # template checking several prefixes at once reads better that way.
        var candidates = List[String]()
        if env.heap.kind(args[0]) == V_LIST:
            for i in range(len(env.heap.cells[args[0]].items)):
                var one = env.heap.cells[args[0]].items[i]
                candidates.append(_want_string(env, one, what, at))
        else:
            candidates.append(_want_string(env, args[0], what, at))
        for i in range(len(candidates)):
            var hit = starts_with(
                s, candidates[i]
            ) if id == B_S_STARTSWITH else ends_with(s, candidates[i])
            if hit:
                return TRUE
        return FALSE

    if id == B_S_UPPER or id == B_S_LOWER:
        _want_count(env, args, 0, 0, what, at)
        if not needs_unicode(s):
            var folded = upper_ascii(s) if id == B_S_UPPER else lower_ascii(s)
            return env.heap.str(folded)
        var uni = env.unicode()
        var folded = upper_full(s, uni[]) if id == B_S_UPPER else lower_full(
            s, uni[]
        )
        return env.heap.str(folded)

    if id == B_S_TITLE:
        _want_count(env, args, 0, 0, what, at)
        return env.heap.str(title_python(s))

    if id == B_S_CAPITALIZE:
        _want_count(env, args, 0, 0, what, at)
        return env.heap.str(capitalize_python(s))

    if id == B_S_JOIN:
        _want_count(env, args, 1, 1, what, at)
        var pieces = iterate(env, args[0], at)
        var out = String("")
        for i in range(len(pieces)):
            if i > 0:
                out += s
            out += _want_string(env, pieces[i], what, at)
        return env.heap.str(out)

    if id == B_S_SPLITLINES:
        _want_count(env, args, 0, 1, what, at)
        return _string_list(env, splitlines(s))

    if id == B_S_FIND:
        _want_count(env, args, 1, 1, what, at)
        return env.heap.int(find(s, _want_string(env, args[0], what, at)))

    if id == B_S_COUNT:
        _want_count(env, args, 1, 1, what, at)
        return env.heap.int(count_of(s, _want_string(env, args[0], what, at)))

    _want_count(env, args, 1, 1, what, at)
    var fix = _want_string(env, args[0], what, at)
    if id == B_S_REMOVEPREFIX:
        if fix.byte_length() > 0 and starts_with(s, fix):
            return env.heap.str(substring(s, _char_count(fix), _char_count(s)))
        return me
    if fix.byte_length() > 0 and ends_with(s, fix):
        return env.heap.str(substring(s, 0, _char_count(s) - _char_count(fix)))
    return me


def _dict_call(
    mut env: Env, id: Int, me: Int, args: List[Int], what: String, at: Int
) raises -> Int:
    if id == B_D_GET:
        _want_count(env, args, 1, 2, what, at)
        var key = _want_string(env, args[0], what, at)
        var got = dict_get(env.heap, me, key)
        if got >= 0:
            return got
        return args[1] if len(args) == 2 else NONE

    _want_count(env, args, 0, 0, what, at)
    var made = env.heap.list()
    for i in range(len(env.heap.cells[me].keys)):
        if id == B_D_KEYS:
            var name = env.heap.cells[me].keys[i]
            var key = env.heap.str(name)
            env.heap.push(made, key)
            continue
        if id == B_D_VALUES:
            var value = env.heap.cells[me].items[i]
            env.heap.push(made, value)
            continue
        var pair = env.heap.list()
        # A pair out of `items()` is a tuple, and a tuple prints with round
        # brackets, which a template that writes the pair straight out shows.
        env.heap.cells[pair].i = TUPLE_FLAG
        var name = env.heap.cells[me].keys[i]
        var key = env.heap.str(name)
        env.heap.push(pair, key)
        var value = env.heap.cells[me].items[i]
        env.heap.push(pair, value)
        env.heap.push(made, pair)
    return made


def _list_call(
    mut env: Env, id: Int, me: Int, args: List[Int], what: String, at: Int
) raises -> Int:
    if id == B_L_INDEX:
        _want_count(env, args, 1, 1, what, at)
        for i in range(len(env.heap.cells[me].items)):
            if equal(env.heap, env.heap.cells[me].items[i], args[0]):
                return env.heap.int(i)
        env.fail(at, "index() was asked for something the list does not hold")
        return UNDEFINED

    _want_count(env, args, 1, 1, what, at)
    var n = 0
    for i in range(len(env.heap.cells[me].items)):
        if equal(env.heap, env.heap.cells[me].items[i], args[0]):
            n += 1
    return env.heap.int(n)


def value_in(mut env: Env, needle: Int, haystack: Int, at: Int) raises -> Bool:
    """`x in y`, which on a string means substring and elsewhere means element.
    """
    var k = env.heap.kind(haystack)
    if k == V_STRING:
        if env.heap.kind(needle) != V_STRING:
            env.fail(at, "only a string can be looked for inside a string")
        return contains(env.heap.cells[haystack].s, env.heap.cells[needle].s)
    if k == V_DICT:
        if env.heap.kind(needle) != V_STRING:
            return False
        return dict_index(env.heap, haystack, env.heap.cells[needle].s) >= 0
    if k == V_LIST:
        for i in range(len(env.heap.cells[haystack].items)):
            if equal(env.heap, env.heap.cells[haystack].items[i], needle):
                return True
        return False
    if k == V_UNDEFINED:
        # The reference answers False here rather than raising, and templates
        # lean on it. Granite writes `if 'citations' in controls` with no guard
        # around it, and a caller that passed no controls has to get a prompt
        # rather than an error.
        return False
    env.fail(at, "'in' was used on something that holds nothing")
    return False
