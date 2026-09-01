"""What a template variable is at runtime, and how it prints.

Values live in one arena and are referred to by index, for the same reason the
tree does. It also gives Jinja's reference semantics for free: a list handed to
two names is one cell with two references, so a template that appends through
one name sees it through the other, which is what Python does and therefore what
the reference does.

The first four cells are constants. Undefined, none, false and true are made
once at startup and every place that wants one hands back the same index, so a
loop over a thousand messages testing a flag allocates nothing.

## Printing is not a detail

A rendered template is a prompt, so every difference between how we print a
value and how Python prints it is a difference in what the model reads. `str`
and `repr` are separate here because Python's are separate and because Jinja
uses both: `{{ x }}` uses `str`, and a list printed with `{{ x }}` uses `repr`
on its elements, which is why `{{ ['a'] }}` comes out with quotes around the
`a` and `{{ 'a' }}` does not.

Floats go through Mojo's own `String`, which produces the shortest form that
reads back as the same double, exactly as Python's `repr` does. The two agree on
every value they have been checked against, including the ones where the
exponent form kicks in, and the tests hold that agreement in place.
"""

comptime V_UNDEFINED = 0
comptime V_NONE = 1
comptime V_BOOL = 2
comptime V_INT = 3
comptime V_FLOAT = 4
comptime V_STRING = 5
comptime V_LIST = 6
comptime V_DICT = 7
comptime V_MACRO = 8
comptime V_BUILTIN = 9

comptime TUPLE_FLAG = 1
"""Written into a list cell's `i` to say it prints with parentheses."""

comptime UNDEFINED = 0
comptime NONE = 1
comptime FALSE = 2
comptime TRUE = 3
"""The four constant cells, at fixed indices, made once."""


struct Cell(Copyable, Movable):
    var kind: Int

    var i: Int
    """The integer, the bool as 0 or 1, the builtin id, or, for a dict, one when
    it is a namespace and zero when it is an ordinary mapping."""

    var f: Float64
    var s: String
    """The string, or a macro's name."""

    var items: List[Int]
    """List elements, dict values in key order, or a builtin's captured state."""

    var keys: List[String]
    """Dict keys, in the order they were first written."""

    var node: Int
    """A macro's definition node."""

    var scope: Int
    """A macro's defining frame, which is what makes it a closure."""

    def __init__(out self, kind: Int):
        self.kind = kind
        self.i = 0
        self.f = 0.0
        self.s = String("")
        self.items = List[Int]()
        self.keys = List[String]()
        self.node = -1
        self.scope = -1


struct Heap(Movable):
    var cells: List[Cell]

    def __init__(out self):
        self.cells = List[Cell]()
        self.cells.append(Cell(V_UNDEFINED))
        self.cells.append(Cell(V_NONE))
        var f = Cell(V_BOOL)
        self.cells.append(f^)
        var t = Cell(V_BOOL)
        t.i = 1
        self.cells.append(t^)

    def kind(self, v: Int) -> Int:
        return self.cells[v].kind

    def add(mut self, var cell: Cell) -> Int:
        self.cells.append(cell^)
        return len(self.cells) - 1

    def int(mut self, value: Int) -> Int:
        var cell = Cell(V_INT)
        cell.i = value
        return self.add(cell^)

    def float(mut self, value: Float64) -> Int:
        var cell = Cell(V_FLOAT)
        cell.f = value
        return self.add(cell^)

    def bool(self, value: Bool) -> Int:
        return TRUE if value else FALSE

    def str(mut self, value: String) -> Int:
        var cell = Cell(V_STRING)
        cell.s = value
        return self.add(cell^)

    def list(mut self) -> Int:
        return self.add(Cell(V_LIST))

    def dict(mut self) -> Int:
        return self.add(Cell(V_DICT))

    def push(mut self, into: Int, value: Int):
        """Append to a list cell.

        A method rather than reaching into `cells` at the call site, because
        `heap.cells[x].items.append(heap.int(1))` borrows the heap twice and
        Mojo is right to refuse it. Every append goes through here instead.
        """
        self.cells[into].items.append(value)


def is_number(heap: Heap, v: Int) -> Bool:
    var k = heap.kind(v)
    return k == V_INT or k == V_FLOAT or k == V_BOOL


def as_float(heap: Heap, v: Int) -> Float64:
    var k = heap.kind(v)
    if k == V_FLOAT:
        return heap.cells[v].f
    if k == V_INT or k == V_BOOL:
        return Float64(heap.cells[v].i)
    return 0.0


def as_int(heap: Heap, v: Int) -> Int:
    var k = heap.kind(v)
    if k == V_FLOAT:
        return Int(heap.cells[v].f)
    if k == V_INT or k == V_BOOL:
        return heap.cells[v].i
    return 0


def truthy(heap: Heap, v: Int) -> Bool:
    """Python's rule: empty is false, zero is false, everything else is true."""
    var k = heap.kind(v)
    if k == V_UNDEFINED or k == V_NONE:
        return False
    if k == V_BOOL or k == V_INT:
        return heap.cells[v].i != 0
    if k == V_FLOAT:
        return heap.cells[v].f != 0.0
    if k == V_STRING:
        return heap.cells[v].s.byte_length() > 0
    if k == V_LIST:
        return len(heap.cells[v].items) > 0
    if k == V_DICT:
        return len(heap.cells[v].keys) > 0
    return True


def length(heap: Heap, v: Int) raises -> Int:
    var k = heap.kind(v)
    if k == V_STRING:
        return _char_count(heap.cells[v].s)
    if k == V_LIST:
        return len(heap.cells[v].items)
    if k == V_DICT:
        return len(heap.cells[v].keys)
    raise Error("object of this type has no length")


def _char_count(s: String) -> Int:
    """Characters, not bytes.

    Python counts a string in code points and a template that slices a string
    or asks for its length is written against that. Counting the bytes that are
    not continuation bytes is the whole of it, since the string is already known
    to be valid UTF-8.
    """
    var data = s.as_bytes()
    var n = 0
    for i in range(len(data)):
        if (data[i] & 0xC0) != 0x80:
            n += 1
    return n


def char_offset(s: String, index: Int) -> Int:
    """The byte offset of the nth character, clamped to the ends."""
    var data = s.as_bytes()
    if index <= 0:
        return 0
    var seen = 0
    for i in range(len(data)):
        if (data[i] & 0xC0) != 0x80:
            if seen == index:
                return i
            seen += 1
    return len(data)


def substring(s: String, start: Int, stop: Int) -> String:
    """Characters `start` up to `stop`, both counted in characters."""
    var from_ = char_offset(s, start)
    var to = char_offset(s, stop)
    if to < from_:
        to = from_
    var data = s.as_bytes()
    var raw = List[UInt8]()
    for i in range(from_, to):
        raw.append(data[i])
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def equal(heap: Heap, a: Int, b: Int) -> Bool:
    if a == b:
        return True
    var ka = heap.kind(a)
    var kb = heap.kind(b)
    if ka == V_UNDEFINED or kb == V_UNDEFINED:
        return ka == kb
    if ka == V_NONE or kb == V_NONE:
        return ka == kb
    if is_number(heap, a) and is_number(heap, b):
        # A bool equals an integer in Python, and templates comparing a flag to
        # 1 rely on it. Only when both sides are integral is the comparison done
        # in integers, so the float path never loses a large value's low bits.
        if ka != V_FLOAT and kb != V_FLOAT:
            return heap.cells[a].i == heap.cells[b].i
        return as_float(heap, a) == as_float(heap, b)
    if ka != kb:
        return False
    if ka == V_STRING:
        return heap.cells[a].s == heap.cells[b].s
    if ka == V_LIST:
        if len(heap.cells[a].items) != len(heap.cells[b].items):
            return False
        for i in range(len(heap.cells[a].items)):
            if not equal(heap, heap.cells[a].items[i], heap.cells[b].items[i]):
                return False
        return True
    if ka == V_DICT:
        if len(heap.cells[a].keys) != len(heap.cells[b].keys):
            return False
        for i in range(len(heap.cells[a].keys)):
            var other = dict_get(heap, b, heap.cells[a].keys[i])
            if other < 0:
                return False
            if not equal(heap, heap.cells[a].items[i], other):
                return False
        return True
    return False


def compare(heap: Heap, a: Int, b: Int) raises -> Int:
    """Less than, equal or greater, as -1, 0 or 1.

    Only numbers, strings and lists order, which is what Python does since it
    stopped ordering anything against anything. Comparing what cannot be
    compared raises rather than picking a side, because a template sorting
    mixed data is a template that is about to produce a prompt nobody intended.
    """
    var ka = heap.kind(a)
    var kb = heap.kind(b)
    if is_number(heap, a) and is_number(heap, b):
        if ka != V_FLOAT and kb != V_FLOAT:
            var x = heap.cells[a].i
            var y = heap.cells[b].i
            return -1 if x < y else (1 if x > y else 0)
        var x = as_float(heap, a)
        var y = as_float(heap, b)
        return -1 if x < y else (1 if x > y else 0)
    if ka == V_STRING and kb == V_STRING:
        return _compare_bytes(heap.cells[a].s, heap.cells[b].s)
    if ka == V_LIST and kb == V_LIST:
        var n = len(heap.cells[a].items)
        var m = len(heap.cells[b].items)
        var shorter = n if n < m else m
        for i in range(shorter):
            var c = compare(
                heap, heap.cells[a].items[i], heap.cells[b].items[i]
            )
            if c != 0:
                return c
        return -1 if n < m else (1 if n > m else 0)
    raise Error("these two values cannot be ordered against each other")


def _compare_bytes(a: String, b: String) -> Int:
    """Byte order, which for UTF-8 is code point order.

    UTF-8 is designed so that comparing the encoded bytes gives the same answer
    as comparing the code points, so this is Python's string ordering without
    decoding anything.
    """
    var x = a.as_bytes()
    var y = b.as_bytes()
    var shorter = len(x) if len(x) < len(y) else len(y)
    for i in range(shorter):
        if x[i] != y[i]:
            return -1 if x[i] < y[i] else 1
    if len(x) == len(y):
        return 0
    return -1 if len(x) < len(y) else 1


def dict_index(heap: Heap, d: Int, key: String) -> Int:
    """Which slot a key is in, or -1.

    A linear scan. A chat message has four keys and a tool schema has a dozen,
    the keys have to keep insertion order anyway because `tojson` has to write
    them in it, and a map alongside the order would be two things to keep in
    step for no measurable gain at these sizes.
    """
    if heap.kind(d) != V_DICT:
        return -1
    for i in range(len(heap.cells[d].keys)):
        if heap.cells[d].keys[i] == key:
            return i
    return -1


def dict_get(heap: Heap, d: Int, key: String) -> Int:
    var at = dict_index(heap, d, key)
    if at < 0:
        return -1
    return heap.cells[d].items[at]


def dict_set(mut heap: Heap, d: Int, key: String, value: Int):
    var at = dict_index(heap, d, key)
    if at >= 0:
        heap.cells[d].items[at] = value
        return
    heap.cells[d].keys.append(key)
    heap.cells[d].items.append(value)


def to_string(heap: Heap, v: Int) raises -> String:
    """`str(x)`, which is what `{{ x }}` writes."""
    var k = heap.kind(v)
    if k == V_UNDEFINED:
        return String("")
    if k == V_NONE:
        return String("None")
    if k == V_BOOL:
        return String("True") if heap.cells[v].i != 0 else String("False")
    if k == V_INT:
        return String(heap.cells[v].i)
    if k == V_FLOAT:
        return String(heap.cells[v].f)
    if k == V_STRING:
        return heap.cells[v].s
    if k == V_MACRO:
        return "<macro " + heap.cells[v].s + ">"
    if k == V_BUILTIN:
        return String("<builtin>")
    return repr_of(heap, v)


def repr_of(heap: Heap, v: Int) raises -> String:
    """`repr(x)`, which is what a container writes for its elements."""
    var k = heap.kind(v)
    if k == V_STRING:
        return quote(heap.cells[v].s)
    if k == V_LIST:
        # A tuple is a list with a flag on it. Nothing about a tuple differs
        # from a list here except how it prints, and `dictsort` and `items`
        # print as tuples in the reference, so a template that writes
        # `{{ d | items | list }}` into a prompt has to see the parentheses.
        var tuple = heap.cells[v].i == TUPLE_FLAG
        var out = String("(") if tuple else String("[")
        for i in range(len(heap.cells[v].items)):
            if i > 0:
                out += ", "
            out += repr_of(heap, heap.cells[v].items[i])
        if tuple and len(heap.cells[v].items) == 1:
            # Python writes a one element tuple with a trailing comma, so that
            # it does not read as a parenthesised expression.
            out += ","
        return out + (")" if tuple else "]")
    if k == V_DICT:
        var out = String("{")
        for i in range(len(heap.cells[v].keys)):
            if i > 0:
                out += ", "
            out += quote(heap.cells[v].keys[i])
            out += ": "
            out += repr_of(heap, heap.cells[v].items[i])
        return out + "}"
    return to_string(heap, v)


comptime _HEX = String("0123456789abcdef")


def quote(s: String) -> String:
    """Python's `repr` of a string.

    Single quotes unless the string has one in it and no double quote, which is
    the rule Python uses and the one that makes a printed list of messages look
    like the list somebody typed. Non ASCII passes through untouched, because
    Python 3 stopped escaping it and a chat template full of Chinese would
    otherwise print as a wall of backslashes.
    """
    var data = s.as_bytes()
    var single = False
    var double = False
    for i in range(len(data)):
        if data[i] == 0x27:
            single = True
        elif data[i] == 0x22:
            double = True
    var quote_byte = UInt8(0x27)
    if single and not double:
        quote_byte = 0x22

    var raw = List[UInt8]()
    raw.append(quote_byte)
    var hex = _HEX.as_bytes()
    for i in range(len(data)):
        var c = data[i]
        if c == 0x5C:
            raw.append(0x5C)
            raw.append(0x5C)
        elif c == quote_byte:
            raw.append(0x5C)
            raw.append(c)
        elif c == 0x0A:
            raw.append(0x5C)
            raw.append(0x6E)
        elif c == 0x0D:
            raw.append(0x5C)
            raw.append(0x72)
        elif c == 0x09:
            raw.append(0x5C)
            raw.append(0x74)
        elif c < 0x20 or c == 0x7F:
            raw.append(0x5C)
            raw.append(0x78)
            raw.append(hex[Int(c >> 4)])
            raw.append(hex[Int(c & 0x0F)])
        else:
            raw.append(c)
    raw.append(quote_byte)
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def json_string(s: String) -> String:
    """A JSON string with `ensure_ascii` off, which is what transformers uses.

    `json.dumps` escapes the same seven characters and writes everything below
    a space as `\\u00xx`. Everything above ASCII goes through as itself, which
    is the difference between the filter transformers installs and Jinja's own,
    and it is the difference a template printing a Chinese tool description
    notices.
    """
    var data = s.as_bytes()
    var raw = List[UInt8]()
    var hex = _HEX.as_bytes()
    raw.append(0x22)
    for i in range(len(data)):
        var c = data[i]
        if c == 0x22 or c == 0x5C:
            raw.append(0x5C)
            raw.append(c)
        elif c == 0x0A:
            raw.append(0x5C)
            raw.append(0x6E)
        elif c == 0x0D:
            raw.append(0x5C)
            raw.append(0x72)
        elif c == 0x09:
            raw.append(0x5C)
            raw.append(0x74)
        elif c == 0x08:
            raw.append(0x5C)
            raw.append(0x62)
        elif c == 0x0C:
            raw.append(0x5C)
            raw.append(0x66)
        elif c < 0x20:
            raw.append(0x5C)
            raw.append(0x75)
            raw.append(0x30)
            raw.append(0x30)
            raw.append(hex[Int(c >> 4)])
            raw.append(hex[Int(c & 0x0F)])
        else:
            raw.append(c)
    raw.append(0x22)
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def _sorted_keys(heap: Heap, d: Int) -> List[Int]:
    """Key slots in sorted order, for `tojson(sort_keys=True)`.

    Insertion sort, because the dictionaries a chat template writes have single
    figure key counts and an insertion sort on ten items beats anything with a
    call stack.
    """
    var order = List[Int]()
    for i in range(len(heap.cells[d].keys)):
        var j = len(order)
        while (
            j > 0
            and _compare_bytes(
                heap.cells[d].keys[order[j - 1]], heap.cells[d].keys[i]
            )
            > 0
        ):
            j -= 1
        order.insert(j, i)
    return order^


def to_json(
    heap: Heap, v: Int, indent: Int, sort_keys: Bool, level: Int
) raises -> String:
    """`json.dumps` with the defaults transformers passes.

    The separators matter and are easy to get wrong. With no indent the item
    separator is a comma and a space and the key separator is a colon and a
    space, which is `json.dumps` default and not what most hand written JSON
    writers produce. With an indent the item separator loses its space, because
    the newline is already there.
    """
    var k = heap.kind(v)
    if k == V_NONE or k == V_UNDEFINED:
        return String("null")
    if k == V_BOOL:
        return String("true") if heap.cells[v].i != 0 else String("false")
    if k == V_INT:
        return String(heap.cells[v].i)
    if k == V_FLOAT:
        return String(heap.cells[v].f)
    if k == V_STRING:
        return json_string(heap.cells[v].s)

    var newline = String("")
    var pad = String("")
    var closing = String("")
    if indent > 0:
        newline = "\n"
        for _ in range((level + 1) * indent):
            pad += " "
        for _ in range(level * indent):
            closing += " "
    var comma = String(",") if indent > 0 else String(", ")

    if k == V_LIST:
        if len(heap.cells[v].items) == 0:
            return String("[]")
        var out = String("[")
        for i in range(len(heap.cells[v].items)):
            if i > 0:
                out += comma
            out += newline
            out += pad
            out += to_json(
                heap, heap.cells[v].items[i], indent, sort_keys, level + 1
            )
        out += newline
        out += closing
        return out + "]"

    if k == V_DICT:
        if len(heap.cells[v].keys) == 0:
            return String("{}")
        var order = List[Int]()
        if sort_keys:
            order = _sorted_keys(heap, v)
        else:
            for i in range(len(heap.cells[v].keys)):
                order.append(i)
        var out = String("{")
        for n in range(len(order)):
            if n > 0:
                out += comma
            out += newline
            out += pad
            out += json_string(heap.cells[v].keys[order[n]])
            out += ": "
            out += to_json(
                heap,
                heap.cells[v].items[order[n]],
                indent,
                sort_keys,
                level + 1,
            )
        out += newline
        out += closing
        return out + "}"

    raise Error("this value cannot be written as JSON")
