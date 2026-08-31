"""DOM mode. A tree, for the documents where a tree is the cheaper answer.

The streaming reader is right for a request body, which is parsed once by code
that knows what it is looking for. It is the wrong shape for a config file, a
model manifest, or a tool schema, all of which get read several times by
different code, in an order nobody decided in advance, and none of which is
larger than a few kilobytes. Writing a hand rolled state machine for each of
those is more code and more bugs than holding the document.

So there are two modes over one scanner, which is what the issue asks for. The
same `Reader` produces the events, and this fills a tree from them, so there is
one place where a document is decided to be well formed and one place where
escapes are decoded.

## Flat, and in order

Nodes live in one list and refer to each other by index. Children are a linked
list, so an object or an array is a first index and a count, and adding a member
never moves the ones already there. That matters because a growing `List` moves
its storage, so a node holding a pointer to its parent would be holding a
dangling one after the next member.

Members come out in the order they were written. Key order is not decoration in
this codebase: a tool call's arguments are a JSON object and a model that was
trained on `{"city": ..., "unit": ...}` is being handed something different when
it arrives sorted. So the tree keeps insertion order and so does the serializer,
and neither has a mode that does anything else.

Duplicate keys are kept rather than merged. `get` returns the first, which is the
behaviour every JSON library that had to pick one picked, and keeping the second
means a caller that wants to reject a duplicated field can see one.

## Strings

A string with no escapes in it is a span into the source buffer, exactly as in
streaming mode. A string that needed decoding is copied into the document's own
buffer. So a document does not own its source and the source has to outlive it,
which is written here and asserted in the tests rather than left to be found.
"""

from molla.io.buffer import Buffer
from molla.json.number import NUM_DOUBLE, NUM_INT
from molla.json.reader import (
    EV_ARRAY_BEGIN,
    EV_ARRAY_END,
    EV_BOOL,
    EV_END,
    EV_ERROR,
    EV_KEY,
    EV_NULL,
    EV_NUMBER,
    EV_OBJECT_BEGIN,
    EV_OBJECT_END,
    EV_STRING,
    JSON_NO_SPACE,
    JSON_OK,
    JSON_SYNTAX,
    Reader,
)
from molla.sys.mem import as_ptr

comptime JS_NULL = 0
comptime JS_BOOL = 1
comptime JS_INT = 2
comptime JS_DOUBLE = 3
comptime JS_STRING = 4
comptime JS_ARRAY = 5
comptime JS_OBJECT = 6

comptime NO_NODE = -1


struct Node(Copyable, ImplicitlyCopyable, Movable):
    """One value. Sixty four bytes, and no allocation of its own."""

    var kind: Int

    var i: Int
    """The integer, the bool as 0 or 1, or the byte offset of a string."""

    var d: Float64

    var length: Int
    """String length in bytes, or the number of members."""

    var first: Int
    """First child, or `NO_NODE`."""

    var last: Int
    """Last child, kept so appending a member is O(1) rather than a walk."""

    var next: Int
    """Next sibling, or `NO_NODE`."""

    var key_at: Int
    var key_len: Int
    """The member's key, for a child of an object. `key_len` is -1 for anything
    that is not one, which is how a stray key is told from an empty one."""

    var key_owned: Bool
    var value_owned: Bool
    """Whether the offset is into the document's own buffer rather than into the
    source. Both are set when a string had to be decoded."""

    def __init__(out self, kind: Int):
        self.kind = kind
        self.i = 0
        self.d = 0.0
        self.length = 0
        self.first = NO_NODE
        self.last = NO_NODE
        self.next = NO_NODE
        self.key_at = 0
        self.key_len = -1
        self.key_owned = False
        self.value_owned = False


struct Document(Movable):
    """The nodes, the bytes the decoded strings live in, and the root."""

    var nodes: List[Node]
    var owned: Buffer
    """Decoded strings. Only strings with an escape in them land here, so an
    ordinary document leaves it empty."""

    var source: Int
    var source_len: Int

    var root: Int
    var error: Int
    var error_at: Int

    def __init__(out self, counter: Int, capacity: Int = 256):
        self.nodes = List[Node]()
        self.owned = Buffer(capacity, counter)
        self.source = 0
        self.source_len = 0
        self.root = NO_NODE
        self.error = JSON_OK
        self.error_at = 0

    def clear(mut self):
        self.nodes.clear()
        self.owned.clear()
        self.source = 0
        self.source_len = 0
        self.root = NO_NODE
        self.error = JSON_OK
        self.error_at = 0

    def ok(self) -> Bool:
        return self.error == JSON_OK and self.root != NO_NODE

    def count(self) -> Int:
        return len(self.nodes)

    def kind(self, node: Int) -> Int:
        if node < 0 or node >= len(self.nodes):
            return JS_NULL
        return self.nodes[node].kind

    def size(self, node: Int) -> Int:
        """Members of a container, or bytes of a string."""
        if node < 0 or node >= len(self.nodes):
            return 0
        return self.nodes[node].length

    def _span(
        self, at: Int, length: Int, owned: Bool
    ) -> Span[UInt8, MutAnyOrigin]:
        var base = self.owned.base() if owned else self.source
        return Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(base + at), length=length
        )

    def text(self, node: Int) -> Span[UInt8, MutAnyOrigin]:
        """The bytes of a string node, empty for anything else."""
        if node < 0 or node >= len(self.nodes):
            return Span[UInt8, MutAnyOrigin](unsafe_ptr=as_ptr(0), length=0)
        var n = self.nodes[node]
        if n.kind != JS_STRING:
            return Span[UInt8, MutAnyOrigin](unsafe_ptr=as_ptr(0), length=0)
        return self._span(n.i, n.length, n.value_owned)

    def key(self, node: Int) -> Span[UInt8, MutAnyOrigin]:
        """The key a member was written under, empty if it is not a member."""
        if node < 0 or node >= len(self.nodes):
            return Span[UInt8, MutAnyOrigin](unsafe_ptr=as_ptr(0), length=0)
        var n = self.nodes[node]
        if n.key_len < 0:
            return Span[UInt8, MutAnyOrigin](unsafe_ptr=as_ptr(0), length=0)
        return self._span(n.key_at, n.key_len, n.key_owned)

    def as_int(self, node: Int, fallback: Int = 0) -> Int:
        if node < 0 or node >= len(self.nodes):
            return fallback
        var n = self.nodes[node]
        if n.kind == JS_INT:
            return n.i
        if n.kind == JS_DOUBLE:
            return Int(n.d)
        return fallback

    def as_double(self, node: Int, fallback: Float64 = 0.0) -> Float64:
        if node < 0 or node >= len(self.nodes):
            return fallback
        var n = self.nodes[node]
        if n.kind == JS_DOUBLE:
            return n.d
        if n.kind == JS_INT:
            return Float64(n.i)
        return fallback

    def as_bool(self, node: Int, fallback: Bool = False) -> Bool:
        if node < 0 or node >= len(self.nodes):
            return fallback
        var n = self.nodes[node]
        if n.kind == JS_BOOL:
            return n.i != 0
        return fallback

    def first_child(self, node: Int) -> Int:
        if node < 0 or node >= len(self.nodes):
            return NO_NODE
        return self.nodes[node].first

    def next_sibling(self, node: Int) -> Int:
        if node < 0 or node >= len(self.nodes):
            return NO_NODE
        return self.nodes[node].next

    def at(self, node: Int, index: Int) -> Int:
        """The nth member, walked rather than indexed.

        A linked list is the right shape for a document that is filled once and
        read in order, and a walk is what a caller iterating an array does
        anyway. `first_child` and `next_sibling` are there for the callers that
        care, and this is here for the ones that would otherwise write it.
        """
        if index < 0:
            return NO_NODE
        var child = self.first_child(node)
        var i = 0
        while child != NO_NODE:
            if i == index:
                return child
            child = self.nodes[child].next
            i += 1
        return NO_NODE

    def get(self, node: Int, name: StringSpan) -> Int:
        """The first member under `name`, or `NO_NODE`."""
        if node < 0 or node >= len(self.nodes):
            return NO_NODE
        if self.nodes[node].kind != JS_OBJECT:
            return NO_NODE
        var want = name.byte_length()
        var p = name.unsafe_ptr()
        var child = self.nodes[node].first
        while child != NO_NODE:
            var n = self.nodes[child]
            if n.key_len == want:
                var text = self._span(n.key_at, n.key_len, n.key_owned)
                var same = True
                for i in range(want):
                    if text[i] != p.unsafe_load(i):
                        same = False
                        break
                if same:
                    return child
            child = n.next
        return NO_NODE

    def get_str(self, node: Int, name: StringSpan) -> Span[UInt8, MutAnyOrigin]:
        return self.text(self.get(node, name))

    def get_int(self, node: Int, name: StringSpan, fallback: Int = 0) -> Int:
        return self.as_int(self.get(node, name), fallback)

    def get_double(
        self, node: Int, name: StringSpan, fallback: Float64 = 0.0
    ) -> Float64:
        return self.as_double(self.get(node, name), fallback)

    def get_bool(
        self, node: Int, name: StringSpan, fallback: Bool = False
    ) -> Bool:
        return self.as_bool(self.get(node, name), fallback)

    def _add(mut self, node: Node) -> Int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def _attach(mut self, parent: Int, child: Int):
        if parent == NO_NODE:
            return
        if self.nodes[parent].first == NO_NODE:
            self.nodes[parent].first = child
        else:
            var last = self.nodes[parent].last
            self.nodes[last].next = child
        self.nodes[parent].last = child
        self.nodes[parent].length += 1


def parse(mut doc: Document, mut reader: Reader, data: Span[UInt8, _]) -> Bool:
    """Fill a document from a buffer.

    The buffer has to outlive the document, because the strings that did not
    need decoding are spans into it. Only the address is kept, so a buffer held
    in a local has to be kept alive explicitly with `keep(buf)` from
    `molla.sys.mem` after the last read of the document.
    """
    doc.clear()
    doc.source = Int(data.unsafe_ptr())
    doc.source_len = len(data)
    reader.begin(data)

    # The container being filled, and the key waiting for its value. A stack
    # rather than recursion, so the depth limit is the reader's one number and
    # not the C stack, which fails differently on every platform.
    var stack = List[Int]()
    var pending_key_at = 0
    var pending_key_len = -1
    var pending_key_owned = False

    while True:
        var e = reader.next()
        if e == EV_END:
            break
        if e == EV_ERROR:
            doc.error = reader.error
            doc.error_at = reader.error_at
            return False

        if e == EV_KEY:
            var text = reader.text()
            pending_key_len = len(text)
            pending_key_owned = reader.str_decoded
            if reader.str_decoded:
                # The reader's scratch is reused by the next string, so a
                # decoded key has to be copied now or not at all.
                pending_key_at = doc.owned.length
                if not doc.owned.append(text):
                    doc.error = JSON_NO_SPACE
                    doc.error_at = reader.at
                    return False
            else:
                pending_key_at = reader.str_at
            continue

        if e == EV_OBJECT_END or e == EV_ARRAY_END:
            _ = stack.pop()
            continue

        var node: Node
        if e == EV_OBJECT_BEGIN:
            node = Node(JS_OBJECT)
        elif e == EV_ARRAY_BEGIN:
            node = Node(JS_ARRAY)
        elif e == EV_STRING:
            node = Node(JS_STRING)
            var text = reader.text()
            node.length = len(text)
            node.value_owned = reader.str_decoded
            if reader.str_decoded:
                node.i = doc.owned.length
                if not doc.owned.append(text):
                    doc.error = JSON_NO_SPACE
                    doc.error_at = reader.at
                    return False
            else:
                node.i = reader.str_at
        elif e == EV_NUMBER:
            if reader.number.kind == NUM_INT:
                node = Node(JS_INT)
                node.i = reader.number.i
            else:
                node = Node(JS_DOUBLE)
                node.d = reader.number.d
        elif e == EV_BOOL:
            node = Node(JS_BOOL)
            node.i = 1 if reader.bool_value else 0
        else:
            node = Node(JS_NULL)

        if pending_key_len >= 0:
            node.key_at = pending_key_at
            node.key_len = pending_key_len
            node.key_owned = pending_key_owned
            pending_key_len = -1

        var index = doc._add(node)
        if len(stack) == 0:
            doc.root = index
        else:
            doc._attach(stack[len(stack) - 1], index)
        if e == EV_OBJECT_BEGIN or e == EV_ARRAY_BEGIN:
            stack.append(index)

    if doc.root == NO_NODE:
        doc.error = reader.error if reader.error != JSON_OK else JSON_SYNTAX
        doc.error_at = reader.error_at
        return False
    return True
