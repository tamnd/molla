"""The filters and the tests, and the names of both.

Every filter here is one a chat template has been seen to use or one that sits
next to those in the documentation, and every filter that is not here is a load
time error naming itself. That is the deliberate half of the design: a template
using `groupby` is refused with the filter's name and the line it is on, and
that refusal is a line in the backlog. It is not rendered with `groupby` quietly
doing nothing.

Which is why `filter_id` and `test_id` are separate from applying them. The
compiler walks the tree once at load and asks for the id of every filter and
every test name it finds, and a zero is the refusal. Nothing reaches the render
path without having been named already.

Two things about the semantics are worth reading before changing anything here.
Sorting is stable, because Python's is and because a template sorting messages
by role and getting them back in a different order within a role produces a
different prompt. And `case_sensitive` defaults to false on `sort`, `unique`,
`min` and `max`, which is not what anybody expects and is what Jinja does.
"""

from std.math import fma

from molla.jinja.access import (
    get_attr_only,
    get_item,
    iterate,
    value_in,
)
from molla.jinja.env import Env
from molla.jinja.strop import (
    indent_lines,
    lower_ascii,
    lower_full,
    needs_unicode,
    replace,
    split,
    split_whitespace,
    strip,
    title_filter,
    upper_ascii,
    upper_full,
)
from molla.jinja.value import (
    JsonStyle,
    TUPLE_FLAG,
    UNDEFINED,
    V_BOOL,
    V_BUILTIN,
    V_DICT,
    V_FLOAT,
    V_INT,
    V_LIST,
    V_MACRO,
    V_NONE,
    V_STRING,
    V_UNDEFINED,
    as_float,
    as_int,
    compare,
    equal,
    is_number,
    length,
    to_json,
    to_string,
    truthy,
)

comptime F_ABS = 1
comptime F_ATTR = 2
comptime F_BATCH = 3
comptime F_CAPITALIZE = 4
comptime F_DEFAULT = 5
comptime F_DICTSORT = 6
comptime F_ESCAPE = 7
comptime F_FIRST = 8
comptime F_FLOAT = 9
comptime F_INDENT = 10
comptime F_INT = 11
comptime F_ITEMS = 12
comptime F_JOIN = 13
comptime F_LAST = 14
comptime F_LENGTH = 15
comptime F_LIST = 16
comptime F_LOWER = 17
comptime F_MAP = 18
comptime F_MAX = 19
comptime F_MIN = 20
comptime F_REJECT = 21
comptime F_REJECTATTR = 22
comptime F_REPLACE = 23
comptime F_REVERSE = 24
comptime F_ROUND = 25
comptime F_SAFE = 26
comptime F_SELECT = 27
comptime F_SELECTATTR = 28
comptime F_SLICE = 29
comptime F_SORT = 30
comptime F_STRING = 31
comptime F_SUM = 32
comptime F_TITLE = 33
comptime F_TOJSON = 34
comptime F_TRIM = 35
comptime F_UNIQUE = 36
comptime F_UPPER = 37
comptime F_WORDCOUNT = 38


def filter_id(name: String) -> Int:
    """The filter's number, or zero when molla does not have it."""
    if name == "abs":
        return F_ABS
    if name == "attr":
        return F_ATTR
    if name == "batch":
        return F_BATCH
    if name == "capitalize":
        return F_CAPITALIZE
    if name == "default" or name == "d":
        return F_DEFAULT
    if name == "dictsort":
        return F_DICTSORT
    if name == "escape" or name == "e" or name == "forceescape":
        return F_ESCAPE
    if name == "first":
        return F_FIRST
    if name == "float":
        return F_FLOAT
    if name == "indent":
        return F_INDENT
    if name == "int":
        return F_INT
    if name == "items":
        return F_ITEMS
    if name == "join":
        return F_JOIN
    if name == "last":
        return F_LAST
    if name == "length" or name == "count":
        return F_LENGTH
    if name == "list":
        return F_LIST
    if name == "lower":
        return F_LOWER
    if name == "map":
        return F_MAP
    if name == "max":
        return F_MAX
    if name == "min":
        return F_MIN
    if name == "reject":
        return F_REJECT
    if name == "rejectattr":
        return F_REJECTATTR
    if name == "replace":
        return F_REPLACE
    if name == "reverse":
        return F_REVERSE
    if name == "round":
        return F_ROUND
    if name == "safe":
        return F_SAFE
    if name == "select":
        return F_SELECT
    if name == "selectattr":
        return F_SELECTATTR
    if name == "slice":
        return F_SLICE
    if name == "sort":
        return F_SORT
    if name == "string":
        return F_STRING
    if name == "sum":
        return F_SUM
    if name == "title":
        return F_TITLE
    if name == "tojson":
        return F_TOJSON
    if name == "trim":
        return F_TRIM
    if name == "unique":
        return F_UNIQUE
    if name == "upper":
        return F_UPPER
    if name == "wordcount":
        return F_WORDCOUNT
    return 0


comptime T_CALLABLE = 1
comptime T_DEFINED = 2
comptime T_DIVISIBLEBY = 3
comptime T_EQ = 4
comptime T_EVEN = 5
comptime T_FALSE = 6
comptime T_FLOAT = 7
comptime T_GE = 8
comptime T_GT = 9
comptime T_IN = 10
comptime T_INTEGER = 11
comptime T_ITERABLE = 12
comptime T_LE = 13
comptime T_LOWER = 14
comptime T_LT = 15
comptime T_MAPPING = 16
comptime T_NE = 17
comptime T_NONE = 18
comptime T_NUMBER = 19
comptime T_ODD = 20
comptime T_SAMEAS = 21
comptime T_SEQUENCE = 22
comptime T_STRING = 23
comptime T_TRUE = 24
comptime T_UNDEFINED = 25
comptime T_UPPER = 26
comptime T_BOOLEAN = 27


def test_id(name: String) -> Int:
    """The test's number, or zero when molla does not have it."""
    if name == "callable":
        return T_CALLABLE
    if name == "defined":
        return T_DEFINED
    if name == "divisibleby":
        return T_DIVISIBLEBY
    if name == "eq" or name == "equalto" or name == "==":
        return T_EQ
    if name == "even":
        return T_EVEN
    if name == "false":
        return T_FALSE
    if name == "float":
        return T_FLOAT
    if name == "ge" or name == ">=":
        return T_GE
    if name == "gt" or name == "greaterthan" or name == ">":
        return T_GT
    if name == "in":
        return T_IN
    if name == "integer":
        return T_INTEGER
    if name == "iterable":
        return T_ITERABLE
    if name == "le" or name == "<=":
        return T_LE
    if name == "lower":
        return T_LOWER
    if name == "lt" or name == "lessthan" or name == "<":
        return T_LT
    if name == "mapping":
        return T_MAPPING
    if name == "ne" or name == "!=":
        return T_NE
    if name == "none":
        return T_NONE
    if name == "number":
        return T_NUMBER
    if name == "odd":
        return T_ODD
    if name == "sameas":
        return T_SAMEAS
    if name == "sequence":
        return T_SEQUENCE
    if name == "string":
        return T_STRING
    if name == "true":
        return T_TRUE
    if name == "undefined":
        return T_UNDEFINED
    if name == "upper":
        return T_UPPER
    if name == "boolean":
        return T_BOOLEAN
    return 0


def _arg(args: List[Int], names: List[String], index: Int, key: String) -> Int:
    """The argument at position `index`, or the one passed as `key`, or -1.

    One function for both because Python's calling convention is one convention
    and a template writing `sort(reverse=true)` and a template writing
    `sort(true)` are writing the same call.
    """
    var seen = 0
    for i in range(len(args)):
        if names[i] == "":
            if seen == index:
                return args[i]
            seen += 1
        elif names[i] == key:
            return args[i]
    return -1


def _flag(
    env: Env, args: List[Int], names: List[String], index: Int, key: String
) raises -> Bool:
    var got = _arg(args, names, index, key)
    return truthy(env.heap, got) if got >= 0 else False


def _text(mut env: Env, v: Int) raises -> String:
    return to_string(env.heap, v)


def _json_style(
    mut env: Env, args: List[Int], names: List[String], at: Int
) raises -> JsonStyle:
    """What `tojson` was asked for.

    The signature is the one transformers gives the filter, which is
    `ensure_ascii`, `indent`, `separators` and `sort_keys` in that order. The
    order is worth spelling out because the first positional argument is
    `ensure_ascii` and not `indent`, so `tojson(2)` is not an indent of two, and
    every template in the corpus that wants an indent names it.
    """
    var width = 0
    var indent = _arg(args, names, 1, String("indent"))
    if indent >= 0 and env.heap.kind(indent) != V_NONE:
        width = as_int(env.heap, indent)
    var style = JsonStyle(width)
    style.ensure_ascii = _flag(env, args, names, 0, String("ensure_ascii"))
    style.sort_keys = _flag(env, args, names, 3, String("sort_keys"))

    var seps = _arg(args, names, 2, String("separators"))
    if seps < 0 or env.heap.kind(seps) == V_NONE:
        return style^
    if env.heap.kind(seps) != V_LIST or len(env.heap.cells[seps].items) != 2:
        env.fail(at, "`separators` wants a pair, as in `(',', ':')`")
    var item = env.heap.cells[seps].items[0]
    var key = env.heap.cells[seps].items[1]
    style.item_sep = _text(env, item)
    style.key_sep = _text(env, key)
    return style^


def _fold(mut env: Env, s: String, up: Bool) raises -> String:
    if not needs_unicode(s):
        return upper_ascii(s) if up else lower_ascii(s)
    var uni = env.unicode()
    return upper_full(s, uni[]) if up else lower_full(s, uni[])


def _new_list(mut env: Env, values: List[Int]) -> Int:
    var made = env.heap.list()
    for i in range(len(values)):
        env.heap.push(made, values[i])
    return made


def get_path(mut env: Env, v: Int, path: String, at: Int) raises -> Int:
    """`selectattr('a.b')` and friends, where the attribute is a dotted path.

    Each part is a key unless it is all digits, in which case it is an index,
    which is what Jinja's attribute getter does and is how a template reaches
    the second element of a tool call's argument list.
    """
    var parts = split(path, String("."), -1)
    var here = v
    for i in range(len(parts)):
        var digits = parts[i].byte_length() > 0
        var raw = parts[i].as_bytes()
        for j in range(len(raw)):
            if raw[j] < 0x30 or raw[j] > 0x39:
                digits = False
                break
        var key = env.heap.int(Int(parts[i])) if digits else env.heap.str(
            parts[i]
        )
        here = get_item(env, here, key, at)
    return here


def _key_of(
    mut env: Env, v: Int, attribute: Int, case_sensitive: Bool, at: Int
) raises -> Int:
    """What a sort or a comparison actually looks at."""
    var here = v
    if attribute >= 0 and env.heap.kind(attribute) != V_NONE:
        here = get_path(env, v, to_string(env.heap, attribute), at)
    if not case_sensitive and env.heap.kind(here) == V_STRING:
        var text = env.heap.cells[here].s
        var folded = _fold(env, text, False)
        return env.heap.str(folded)
    return here


def _sorted(
    mut env: Env,
    items: List[Int],
    attribute: Int,
    case_sensitive: Bool,
    reverse: Bool,
    at: Int,
) raises -> List[Int]:
    """Insertion sort, which is stable, which is the point.

    Python's `sorted` keeps equal elements in the order it found them, and a
    template sorting a conversation by role would otherwise shuffle the turns
    inside each role. Insertion sort on the tens of elements a chat template
    sorts is also faster than anything with a call stack.
    """
    var keys = List[Int]()
    for i in range(len(items)):
        keys.append(_key_of(env, items[i], attribute, case_sensitive, at))

    var order = List[Int]()
    for i in range(len(items)):
        env.step()
        var j = len(order)
        while j > 0:
            var c = compare(env.heap, keys[order[j - 1]], keys[i])
            var after = c < 0 if reverse else c > 0
            if not after:
                break
            j -= 1
        order.insert(j, i)

    var out = List[Int]()
    for i in range(len(order)):
        out.append(items[order[i]])
    return out^


def _escape(s: String) -> String:
    var data = s.as_bytes()
    var out = String("")
    var start = 0
    for i in range(len(data)):
        var c = data[i]
        if c != 0x26 and c != 0x3C and c != 0x3E and c != 0x27 and c != 0x22:
            continue
        var raw = List[UInt8]()
        for j in range(start, i):
            raw.append(data[j])
        out += String(StringSpan(unsafe_from_utf8=Span(raw)))
        if c == 0x26:
            out += "&amp;"
        elif c == 0x3C:
            out += "&lt;"
        elif c == 0x3E:
            out += "&gt;"
        elif c == 0x27:
            out += "&#39;"
        else:
            out += "&#34;"
        start = i + 1
    var raw = List[UInt8]()
    for j in range(start, len(data)):
        raw.append(data[j])
    return out + String(StringSpan(unsafe_from_utf8=Span(raw)))


def _parse_int(s: String, base: Int) raises -> Tuple[Int, Bool]:
    """A whole string to an integer, and whether it was one.

    Python's `int(s)` allows surrounding whitespace and a sign and nothing else,
    and the `int` filter falls back to its default rather than raising, so the
    caller needs to know which happened without an exception.
    """
    var text = strip(s, String(""), True, True)
    var data = text.as_bytes()
    if len(data) == 0:
        return (0, False)
    var at = 0
    var negative = False
    if data[0] == 0x2B or data[0] == 0x2D:
        negative = data[0] == 0x2D
        at = 1
    if at >= len(data):
        return (0, False)
    var value = 0
    while at < len(data):
        var c = Int(data[at])
        var digit = -1
        if c >= 0x30 and c <= 0x39:
            digit = c - 0x30
        elif c >= 0x61 and c <= 0x7A:
            digit = c - 0x61 + 10
        elif c >= 0x41 and c <= 0x5A:
            digit = c - 0x41 + 10
        if digit < 0 or digit >= base:
            return (0, False)
        value = value * base + digit
        at += 1
    return (-value if negative else value, True)


def _parse_float(s: String) raises -> Tuple[Float64, Bool]:
    var text = strip(s, String(""), True, True)
    if text.byte_length() == 0:
        return (0.0, False)
    try:
        return (Float64(text), True)
    except:
        return (0.0, False)


def _round_half_even(value: Float64, digits: Int) -> Float64:
    """Python's rounding, which goes to the even neighbour on a tie.

    `round(0.5)` is zero and `round(1.5)` is two in Python 3, and a template
    printing a rounded score would otherwise differ by one on every tie.

    Getting the tie right needs more precision than a double has. `2.675` is a
    shade below the halfway point and `2.665` is a shade above it, yet both
    multiply by a hundred to exactly `267.5` and `266.5`, so the remainder on
    its own cannot tell them apart and Python rounds them opposite ways. So the
    product is kept exactly, as a high part and a low part, using a fused
    multiply add to recover the bits the multiply threw away. The remainder is
    then compared against a half with those extra bits still in hand, which is
    the same question Python's decimal path asks.
    """
    var scale = 1.0
    for _ in range(digits if digits > 0 else -digits):
        scale *= 10.0

    var hi = value * scale if digits > 0 else value / scale
    # The exact product is `hi + lo`, and for the divide the same trick recovers
    # the residual by asking what `hi * scale` failed to account for.
    var lo = (
        fma(value, scale, -hi) if digits > 0 else fma(-hi, scale, value) / scale
    )

    var down = Float64(Int(hi))
    if hi < 0.0 and hi != down:
        down -= 1.0
    var rest = (hi - down) + lo
    var picked = down
    if rest > 0.5:
        picked = down + 1.0
    elif rest == 0.5:
        picked = down if Int(down) % 2 == 0 else down + 1.0

    var out = picked / scale if digits > 0 else picked * scale
    # Python keeps the sign of the input, so rounding a small negative number to
    # zero gives negative zero and prints as `-0.0`.
    if out == 0.0 and value < 0.0:
        return -0.0
    return out


def _trunc_to(value: Float64, digits: Int, up: Bool) -> Float64:
    var scale = 1.0
    for _ in range(digits if digits > 0 else -digits):
        scale *= 10.0
    var scaled = value * scale if digits > 0 else value / scale
    var whole = Float64(Int(scaled))
    if up and scaled > whole:
        whole += 1.0
    if not up and scaled < whole:
        whole -= 1.0
    return whole / scale if digits > 0 else whole * scale


def apply_filter(
    mut env: Env,
    id: Int,
    value: Int,
    args: List[Int],
    names: List[String],
    at: Int,
) raises -> Int:
    """One filter, dispatched on its number."""
    env.step()

    if id == F_SAFE:
        # Autoescape is off in the environment transformers builds, so marking
        # something safe changes nothing about how it prints.
        return value

    if id == F_DEFAULT:
        var fallback = _arg(args, names, 0, String("value"))
        var boolean = _flag(env, args, names, 1, String("boolean"))
        var missing = (
            not truthy(env.heap, value) if boolean else env.heap.kind(value)
            == V_UNDEFINED
        )
        if not missing:
            return value
        return fallback if fallback >= 0 else env.heap.str(String(""))

    if id == F_ABS:
        if env.heap.kind(value) == V_FLOAT:
            var f = env.heap.cells[value].f
            return env.heap.float(-f if f < 0.0 else f)
        var n = as_int(env.heap, value)
        return env.heap.int(-n if n < 0 else n)

    if id == F_ATTR:
        # The one accessor that never falls back to a key, which is the whole
        # reason the filter exists next to plain dotted access.
        var name = to_string(env.heap, _arg(args, names, 0, String("name")))
        return get_attr_only(env, value, name, at)

    if id == F_STRING:
        return value if env.heap.kind(value) == V_STRING else env.heap.str(
            _text(env, value)
        )

    if id == F_ESCAPE:
        return env.heap.str(_escape(_text(env, value)))

    if id == F_LOWER or id == F_UPPER:
        return env.heap.str(_fold(env, _text(env, value), id == F_UPPER))

    if id == F_CAPITALIZE:
        var s = _text(env, value)
        if s.byte_length() == 0:
            return env.heap.str(s)
        var head = _fold(env, _first_char(s), True)
        return env.heap.str(head + _fold(env, _rest_chars(s), False))

    if id == F_TITLE:
        return env.heap.str(title_filter(_text(env, value)))

    if id == F_TRIM:
        var chars = _arg(args, names, 0, String("chars"))
        var set = String("")
        if chars >= 0 and env.heap.kind(chars) != V_NONE:
            set = to_string(env.heap, chars)
        return env.heap.str(strip(_text(env, value), set, True, True))

    if id == F_REPLACE:
        var old = to_string(env.heap, _arg(args, names, 0, String("old")))
        var new = to_string(env.heap, _arg(args, names, 1, String("new")))
        var limit = _arg(args, names, 2, String("count"))
        var n = -1
        if limit >= 0 and env.heap.kind(limit) != V_NONE:
            n = as_int(env.heap, limit)
        return env.heap.str(replace(_text(env, value), old, new, n))

    if id == F_INDENT:
        var width = _arg(args, names, 0, String("width"))
        var prefix = String("    ")
        if width >= 0:
            if env.heap.kind(width) == V_STRING:
                prefix = env.heap.cells[width].s
            else:
                prefix = String("")
                for _ in range(as_int(env.heap, width)):
                    prefix += " "
        var first = _flag(env, args, names, 1, String("first"))
        var blank = _flag(env, args, names, 2, String("blank"))
        return env.heap.str(
            indent_lines(_text(env, value), prefix, first, blank)
        )

    if id == F_WORDCOUNT:
        return env.heap.int(len(split_whitespace(_text(env, value), -1)))

    if id == F_LENGTH:
        return env.heap.int(length(env.heap, value))

    if id == F_INT:
        return _to_int(env, value, args, names)

    if id == F_FLOAT:
        return _to_float(env, value, args, names)

    if id == F_ROUND:
        return _do_round(env, value, args, names, at)

    if id == F_TOJSON:
        return env.heap.str(
            to_json(env.heap, value, _json_style(env, args, names, at), 0)
        )

    if id == F_ITEMS or id == F_DICTSORT:
        return _pairs(env, id, value, args, names, at)

    return _sequence_filter(env, id, value, args, names, at)


def _first_char(s: String) -> String:
    var data = s.as_bytes()
    var n = 1
    while n < len(data) and (data[n] & 0xC0) == 0x80:
        n += 1
    var raw = List[UInt8]()
    for i in range(n):
        raw.append(data[i])
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def _rest_chars(s: String) -> String:
    var data = s.as_bytes()
    var n = 1
    while n < len(data) and (data[n] & 0xC0) == 0x80:
        n += 1
    var raw = List[UInt8]()
    for i in range(n, len(data)):
        raw.append(data[i])
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def _to_int(
    mut env: Env, value: Int, args: List[Int], names: List[String]
) raises -> Int:
    var fallback = _arg(args, names, 0, String("default"))
    var base_arg = _arg(args, names, 1, String("base"))
    var base = as_int(env.heap, base_arg) if base_arg >= 0 else 10
    var k = env.heap.kind(value)
    if k == V_INT or k == V_BOOL:
        return env.heap.int(env.heap.cells[value].i)
    if k == V_FLOAT:
        return env.heap.int(Int(env.heap.cells[value].f))
    if k == V_STRING:
        var parsed = _parse_int(env.heap.cells[value].s, base)
        if parsed[1]:
            return env.heap.int(parsed[0])
        # A string that is not an integer might still be a float, and Python's
        # filter tries that before it gives up, so `'3.7'|int` is three.
        var as_real = _parse_float(env.heap.cells[value].s)
        if as_real[1] and base == 10:
            return env.heap.int(Int(as_real[0]))
    return fallback if fallback >= 0 else env.heap.int(0)


def _to_float(
    mut env: Env, value: Int, args: List[Int], names: List[String]
) raises -> Int:
    var fallback = _arg(args, names, 0, String("default"))
    if is_number(env.heap, value):
        return env.heap.float(as_float(env.heap, value))
    if env.heap.kind(value) == V_STRING:
        var parsed = _parse_float(env.heap.cells[value].s)
        if parsed[1]:
            return env.heap.float(parsed[0])
    return fallback if fallback >= 0 else env.heap.float(0.0)


def _do_round(
    mut env: Env, value: Int, args: List[Int], names: List[String], at: Int
) raises -> Int:
    var digits_arg = _arg(args, names, 0, String("precision"))
    var digits = as_int(env.heap, digits_arg) if digits_arg >= 0 else 0
    var how_arg = _arg(args, names, 1, String("method"))
    var how = to_string(env.heap, how_arg) if how_arg >= 0 else String("common")
    var f = as_float(env.heap, value)
    if how == "common":
        return env.heap.float(_round_half_even(f, digits))
    if how == "ceil":
        return env.heap.float(_trunc_to(f, digits, True))
    if how == "floor":
        return env.heap.float(_trunc_to(f, digits, False))
    env.fail(
        at,
        "round() was given the method '"
        + how
        + "', and the only ones there are are common, ceil and floor",
    )
    return UNDEFINED


def _pairs(
    mut env: Env,
    id: Int,
    value: Int,
    args: List[Int],
    names: List[String],
    at: Int,
) raises -> Int:
    """`items` and `dictsort`, both of which turn a mapping into pairs."""
    if env.heap.kind(value) != V_DICT:
        env.fail(at, "this filter takes a mapping, and this is not one")

    var order = List[Int]()
    for i in range(len(env.heap.cells[value].keys)):
        order.append(i)

    if id == F_DICTSORT:
        var case_sensitive = _flag(
            env, args, names, 0, String("case_sensitive")
        )
        var by_arg = _arg(args, names, 1, String("by"))
        var by_value = to_string(env.heap, by_arg) if by_arg >= 0 else String(
            "key"
        )
        if by_value != "key" and by_value != "value":
            env.fail(
                at,
                "dictsort() can sort by 'key' or by 'value', and '"
                + by_value
                + "' is neither",
            )
        var reverse = _flag(env, args, names, 2, String("reverse"))
        var keys = List[Int]()
        for i in range(len(order)):
            if by_value == "key":
                var name = env.heap.cells[value].keys[i]
                var made = env.heap.str(name)
                keys.append(_key_of(env, made, -1, case_sensitive, at))
            else:
                var held = env.heap.cells[value].items[i]
                keys.append(_key_of(env, held, -1, case_sensitive, at))
        order.clear()
        for i in range(len(keys)):
            env.step()
            var j = len(order)
            while j > 0:
                var c = compare(env.heap, keys[order[j - 1]], keys[i])
                var after = c < 0 if reverse else c > 0
                if not after:
                    break
                j -= 1
            order.insert(j, i)

    var made = env.heap.list()
    for n in range(len(order)):
        var pair = env.heap.list()
        env.heap.cells[pair].i = TUPLE_FLAG
        var key = env.heap.cells[value].keys[order[n]]
        var key_cell = env.heap.str(key)
        env.heap.push(pair, key_cell)
        var held = env.heap.cells[value].items[order[n]]
        env.heap.push(pair, held)
        env.heap.push(made, pair)
    return made


def _sequence_filter(
    mut env: Env,
    id: Int,
    value: Int,
    args: List[Int],
    names: List[String],
    at: Int,
) raises -> Int:
    """Everything that walks a sequence, which is most of what is left."""
    var items = iterate(env, value, at)

    if id == F_LIST:
        return _new_list(env, items)

    if id == F_FIRST or id == F_LAST:
        if len(items) == 0:
            return UNDEFINED
        return items[0] if id == F_FIRST else items[len(items) - 1]

    if id == F_REVERSE:
        var out = List[Int]()
        for i in range(len(items)):
            out.append(items[len(items) - 1 - i])
        if env.heap.kind(value) == V_STRING:
            var text = String("")
            for i in range(len(out)):
                text += to_string(env.heap, out[i])
            return env.heap.str(text)
        return _new_list(env, out)

    if id == F_JOIN:
        var glue_arg = _arg(args, names, 0, String("d"))
        var glue = to_string(env.heap, glue_arg) if glue_arg >= 0 else String(
            ""
        )
        var attribute = _arg(args, names, 1, String("attribute"))
        var out = String("")
        for i in range(len(items)):
            if i > 0:
                out += glue
            var one = items[i]
            if attribute >= 0 and env.heap.kind(attribute) != V_NONE:
                one = get_path(env, one, to_string(env.heap, attribute), at)
            out += to_string(env.heap, one)
        return env.heap.str(out)

    if id == F_SORT:
        var reverse = _flag(env, args, names, 0, String("reverse"))
        var case_sensitive = _flag(
            env, args, names, 1, String("case_sensitive")
        )
        var attribute = _arg(args, names, 2, String("attribute"))
        return _new_list(
            env, _sorted(env, items, attribute, case_sensitive, reverse, at)
        )

    if id == F_MIN or id == F_MAX:
        if len(items) == 0:
            return UNDEFINED
        var case_sensitive = _flag(
            env, args, names, 0, String("case_sensitive")
        )
        var attribute = _arg(args, names, 1, String("attribute"))
        var best = items[0]
        var best_key = _key_of(env, best, attribute, case_sensitive, at)
        for i in range(1, len(items)):
            env.step()
            var key = _key_of(env, items[i], attribute, case_sensitive, at)
            var c = compare(env.heap, key, best_key)
            # Strictly better only, so ties keep the earlier element, which is
            # what Python's min and max do.
            if (c > 0) if id == F_MAX else (c < 0):
                best = items[i]
                best_key = key
        return best

    if id == F_SUM:
        var attribute = _arg(args, names, 0, String("attribute"))
        var start = _arg(args, names, 1, String("start"))
        var whole = not (start >= 0 and env.heap.kind(start) == V_FLOAT)
        var total_int = as_int(env.heap, start) if start >= 0 else 0
        var total = as_float(env.heap, start) if start >= 0 else 0.0
        for i in range(len(items)):
            env.step()
            var one = items[i]
            if attribute >= 0 and env.heap.kind(attribute) != V_NONE:
                one = get_path(env, one, to_string(env.heap, attribute), at)
            if env.heap.kind(one) == V_FLOAT:
                whole = False
            total += as_float(env.heap, one)
            total_int += as_int(env.heap, one)
        return env.heap.int(total_int) if whole else env.heap.float(total)

    if id == F_UNIQUE:
        var case_sensitive = _flag(
            env, args, names, 0, String("case_sensitive")
        )
        var attribute = _arg(args, names, 1, String("attribute"))
        var seen = List[Int]()
        var out = List[Int]()
        for i in range(len(items)):
            env.step()
            var key = _key_of(env, items[i], attribute, case_sensitive, at)
            var already = False
            for j in range(len(seen)):
                if equal(env.heap, seen[j], key):
                    already = True
                    break
            if already:
                continue
            seen.append(key)
            out.append(items[i])
        return _new_list(env, out)

    if id == F_BATCH or id == F_SLICE:
        return _chunk(env, id, items, args, names, at)

    if id == F_MAP:
        return _do_map(env, items, args, names, at)

    return _do_select(env, id, items, args, names, at)


def _chunk(
    mut env: Env,
    id: Int,
    items: List[Int],
    args: List[Int],
    names: List[String],
    at: Int,
) raises -> Int:
    """`batch` takes a size and `slice` takes a count, and they round opposite
    ways, which is the only reason they are two filters."""
    var key = String("linecount") if id == F_BATCH else String("slices")
    var count_arg = _arg(args, names, 0, key)
    var count = as_int(env.heap, count_arg) if count_arg >= 0 else 0
    if count <= 0:
        env.fail(at, "this filter needs a positive number of pieces")
    var fill = _arg(args, names, 1, String("fill_with"))
    var has_fill = fill >= 0 and env.heap.kind(fill) != V_NONE

    var made = env.heap.list()
    if id == F_BATCH:
        var i = 0
        while i < len(items):
            var piece = env.heap.list()
            for j in range(i, i + count):
                if j < len(items):
                    env.heap.push(piece, items[j])
                elif has_fill:
                    env.heap.push(piece, fill)
            env.heap.push(made, piece)
            i += count
        return made

    var per = len(items) // count
    var extra = len(items) % count
    var taken = 0
    for n in range(count):
        var size = per + (1 if n < extra else 0)
        var piece = env.heap.list()
        for j in range(taken, taken + size):
            env.heap.push(piece, items[j])
        if has_fill and n >= extra and extra > 0:
            env.heap.push(piece, fill)
        env.heap.push(made, piece)
        taken += size
    return made


def _do_map(
    mut env: Env,
    items: List[Int],
    args: List[Int],
    names: List[String],
    at: Int,
) raises -> Int:
    """`map('filter', ...)` and `map(attribute='x')`, which are two filters
    wearing one name."""
    var attribute = _arg(args, names, -1, String("attribute"))
    var out = List[Int]()
    if attribute >= 0:
        var path = to_string(env.heap, attribute)
        var fallback = _arg(args, names, -1, String("default"))
        for i in range(len(items)):
            env.step()
            var one = get_path(env, items[i], path, at)
            if env.heap.kind(one) == V_UNDEFINED and fallback >= 0:
                one = fallback
            out.append(one)
        return _new_list(env, out)

    var name_arg = _arg(args, names, 0, String("name"))
    if name_arg < 0:
        env.fail(at, "map() needs either a filter name or an attribute")
    var name = to_string(env.heap, name_arg)
    var id = filter_id(name)
    if id == 0:
        env.fail(
            at,
            "map() was given the filter '"
            + name
            + "', which molla does not have",
        )
    var rest = List[Int]()
    var rest_names = List[String]()
    var seen = 0
    for i in range(len(args)):
        if names[i] == "":
            seen += 1
            if seen == 1:
                continue
        elif names[i] == "name" or names[i] == "attribute":
            continue
        rest.append(args[i])
        rest_names.append(names[i])
    for i in range(len(items)):
        out.append(apply_filter(env, id, items[i], rest, rest_names, at))
    return _new_list(env, out)


def _do_select(
    mut env: Env,
    id: Int,
    items: List[Int],
    args: List[Int],
    names: List[String],
    at: Int,
) raises -> Int:
    """`select`, `reject`, `selectattr` and `rejectattr`.

    All four are one loop with two switches: whether the value or one of its
    attributes is tested, and whether a passing test keeps or drops. With no
    test named at all the check is plain truthiness, which is how
    `messages | selectattr('tool_calls')` picks out the turns that have any.
    """
    var keep = id == F_SELECT or id == F_SELECTATTR
    var on_attribute = id == F_SELECTATTR or id == F_REJECTATTR

    var first = 0
    var path = String("")
    if on_attribute:
        var attr_arg = _arg(args, names, 0, String("attribute"))
        if attr_arg < 0:
            env.fail(
                at, "this filter needs the name of an attribute to look at"
            )
        path = to_string(env.heap, attr_arg)
        first = 1

    var test = 0
    var test_args = List[Int]()
    var test_names = List[String]()
    var name_arg = _arg(args, names, first, String("name"))
    if name_arg >= 0:
        var name = to_string(env.heap, name_arg)
        test = test_id(name)
        if test == 0:
            env.fail(
                at,
                "this filter was given the test '"
                + name
                + "', which molla does not have",
            )
        var seen = 0
        for i in range(len(args)):
            if names[i] == "":
                seen += 1
                if seen <= first + 1:
                    continue
            elif names[i] == "name" or names[i] == "attribute":
                continue
            test_args.append(args[i])
            test_names.append(names[i])

    var out = List[Int]()
    for i in range(len(items)):
        env.step()
        var subject = get_path(
            env, items[i], path, at
        ) if on_attribute else items[i]
        var passed = apply_test(
            env, test, subject, test_args, test_names, at
        ) if test != 0 else truthy(env.heap, subject)
        if passed == keep:
            out.append(items[i])
    return _new_list(env, out)


def apply_test(
    mut env: Env,
    id: Int,
    value: Int,
    args: List[Int],
    names: List[String],
    at: Int,
) raises -> Bool:
    """One test, dispatched on its number."""
    env.step()
    var k = env.heap.kind(value)

    if id == T_DEFINED:
        return k != V_UNDEFINED
    if id == T_UNDEFINED:
        return k == V_UNDEFINED
    if id == T_NONE:
        return k == V_NONE
    if id == T_BOOLEAN:
        return k == V_BOOL
    if id == T_TRUE:
        return k == V_BOOL and env.heap.cells[value].i != 0
    if id == T_FALSE:
        return k == V_BOOL and env.heap.cells[value].i == 0
    if id == T_INTEGER:
        return k == V_INT
    if id == T_FLOAT:
        return k == V_FLOAT
    if id == T_NUMBER:
        return is_number(env.heap, value)
    if id == T_STRING:
        return k == V_STRING
    if id == T_MAPPING:
        return k == V_DICT
    if id == T_SEQUENCE:
        return k == V_LIST or k == V_STRING or k == V_DICT
    if id == T_ITERABLE:
        return k == V_LIST or k == V_STRING or k == V_DICT
    if id == T_CALLABLE:
        return k == V_MACRO or k == V_BUILTIN

    if id == T_ODD or id == T_EVEN:
        var n = as_int(env.heap, value)
        var odd = (n % 2) != 0
        return odd if id == T_ODD else not odd

    if id == T_LOWER or id == T_UPPER:
        # `islower` and `isupper` want at least one cased character, so a string
        # of digits is neither. Folding both ways and asking whether only one of
        # them changed nothing says exactly that without a case table.
        var s = to_string(env.heap, value)
        var up = id == T_UPPER
        return _fold(env, s, up) == s and _fold(env, s, not up) != s

    var other = _arg(args, names, 0, String("other"))
    if other < 0:
        env.fail(at, "this test needs something to compare against")

    if id == T_DIVISIBLEBY:
        var by = as_int(env.heap, other)
        if by == 0:
            env.fail(at, "divisibleby was asked about zero")
        return as_int(env.heap, value) % by == 0
    if id == T_EQ:
        return equal(env.heap, value, other)
    if id == T_NE:
        return not equal(env.heap, value, other)
    if id == T_SAMEAS:
        # Identity, and a value is one cell, so identity is one comparison.
        return value == other
    if id == T_IN:
        return value_in(env, value, other, at)

    var c = compare(env.heap, value, other)
    if id == T_LT:
        return c < 0
    if id == T_LE:
        return c <= 0
    if id == T_GT:
        return c > 0
    return c >= 0
