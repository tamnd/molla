"""Tests for `molla.json`.

Three things are worth saying about what is tested and how.

The number conversions are checked against values chosen because they are the
ones that break converters, not because they are round. `2.2250738585072011e-308`
is the input that used to hang PHP, `7.2057594037927933e16` is the classic tie
that separates a correct rounder from a nearly correct one, and the smallest
subnormal and the largest finite double are the two ends. A converter that gets
`1.5` right tells you nothing.

The round trip test goes through both directions of our own code, which would
hide a bug that is symmetric. So the fixed values above are checked against the
bit patterns rather than against our own printer, and the round trip is there to
catch the asymmetric ones.

The parse errors are checked for which error and at which offset, not only that
something failed. An offset in the error is what turns a 400 into a two minute
look at the body, and an offset that is right for the wrong reason is easy to
write and impossible to notice later.
"""

from std.memory import bitcast

from harness import Suite

from molla.io.buffer import Buffer
from molla.io.bytes import equals_str
from molla.json.decimal import Decimal
from molla.json.dom import (
    JS_ARRAY,
    JS_BOOL,
    JS_DOUBLE,
    JS_INT,
    JS_NULL,
    JS_OBJECT,
    JS_STRING,
    NO_NODE,
    Document,
    parse,
)
from molla.json.number import (
    NUM_DOUBLE,
    NUM_INT,
    Number,
    double_to_decimal,
    parse_number,
)
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
    JSON_ESCAPE,
    JSON_SHORT,
    JSON_SYNTAX,
    JSON_TRAILING,
    JSON_UTF8,
    Reader,
    error_text,
)
from molla.json.scan import (
    STR_CONTROL,
    STR_END,
    STR_ESCAPE,
    STR_SHORT,
    all_ascii,
    scan_string,
    skip_ws,
    validate_utf8,
)
from molla.json.serialize import Writer, write_json_double, write_json_string
from molla.sys.mem import AllocCounter


def _load(mut buf: Buffer, text: StringSpan):
    buf.clear()
    _ = buf.append_str(text)


def _hex_digit(c: UInt8) -> Int:
    if c >= 48 and c <= 57:
        return Int(c) - 48
    if c >= 97 and c <= 102:
        return Int(c) - 87
    return -1


def _load_raw(mut buf: Buffer, text: StringSpan):
    """Load `text`, where `%xx` means one raw byte.

    A Mojo string literal holds text and not bytes, so `\\xc0` in one is the code
    point U+00C0 and comes out as two bytes of UTF-8. The byte sequences that
    have to be refused are the ones that are not valid UTF-8 in the first place,
    which is exactly what a literal cannot hold, so they are spelled out here.
    """
    buf.clear()
    var data = text.as_bytes()
    var at = 0
    while at < len(data):
        if data[at] == 37 and at + 2 < len(data):
            var hi = _hex_digit(data[at + 1])
            var lo = _hex_digit(data[at + 2])
            if hi >= 0 and lo >= 0:
                _ = buf.append_byte(UInt8(hi * 16 + lo))
                at += 3
                continue
        _ = buf.append_byte(data[at])
        at += 1


def _same(data: Span[UInt8, _], text: StringSpan) -> Bool:
    return equals_str(data, text)


def _parse_double(mut buf: Buffer, text: StringSpan) -> Float64:
    _load(buf, text)
    var n = Number()
    var end = parse_number(buf.ptr(), 0, buf.length, n)
    if end < 0:
        return 0.0 / 0.0
    return n.as_double()


def _bits(value: Float64) -> UInt64:
    return bitcast[DType.uint64](value)


def _print(mut buf: Buffer, value: Float64) -> Bool:
    buf.clear()
    return write_json_double(buf, value)


def test_scan(mut suite: Suite, counter: Int):
    suite.group("json.scan finds what a parser has to decide about")
    var buf = Buffer(256, counter)

    _load(buf, "   \t\r\n  {")
    suite.check(skip_ws(buf.ptr(), 0, buf.length) == 8, "whitespace is skipped")

    _load(buf, "\x0b{")
    suite.check(
        skip_ws(buf.ptr(), 0, buf.length) == 0,
        "a vertical tab is not JSON whitespace",
    )

    _load(buf, 'plain text",rest')
    var at = 0
    suite.check(
        scan_string(buf.ptr(), 0, buf.length, at) == STR_END and at == 10,
        "a string with nothing in it ends at the quote",
    )

    _load(buf, 'has\\nan escape"')
    suite.check(
        scan_string(buf.ptr(), 0, buf.length, at) == STR_ESCAPE and at == 3,
        "a backslash stops the scan where it is",
    )

    _load(buf, 'raw\nnewline"')
    suite.check(
        scan_string(buf.ptr(), 0, buf.length, at) == STR_CONTROL and at == 3,
        "a raw control byte inside a string is caught by the same pass",
    )

    _load(buf, "never ends")
    suite.check(
        scan_string(buf.ptr(), 0, buf.length, at) == STR_SHORT,
        "a string that runs off the end says so",
    )

    # Long enough to cross several vectors, so the tail loop is not the only
    # thing being tested.
    var long = String("")
    for _ in range(200):
        long += "abcdefgh"
    _load(buf, long + '"')
    suite.check(
        scan_string(buf.ptr(), 0, buf.length, at) == STR_END and at == 1600,
        "and a long string is found across vectors",
    )

    _load(buf, "plain ascii")
    suite.check(all_ascii(buf.ptr(), 0, buf.length), "ascii is recognised")
    _load(buf, "café")
    suite.check(
        not all_ascii(buf.ptr(), 0, buf.length), "and a high byte is not"
    )
    suite.check(validate_utf8(buf.ptr(), 0, buf.length), "valid UTF-8 passes")

    _load_raw(buf, "%c0%af")
    suite.check(
        not validate_utf8(buf.ptr(), 0, buf.length),
        "an overlong encoding is refused",
    )
    _load_raw(buf, "%ed%a0%80")
    suite.check(
        not validate_utf8(buf.ptr(), 0, buf.length),
        "a surrogate half is refused",
    )
    _load_raw(buf, "%f5%80%80%80")
    suite.check(
        not validate_utf8(buf.ptr(), 0, buf.length),
        "and anything past U+10FFFF is refused",
    )
    _load_raw(buf, "%e2%82")
    suite.check(
        not validate_utf8(buf.ptr(), 0, buf.length),
        "a truncated sequence is refused rather than read past the end",
    )


def test_integers(mut suite: Suite, counter: Int):
    suite.group("json numbers, the integers that stay integers")
    var buf = Buffer(256, counter)
    var n = Number()

    _load(buf, "0")
    suite.check(
        parse_number(buf.ptr(), 0, buf.length, n) == 1
        and n.kind == NUM_INT
        and n.i == 0,
        "zero",
    )

    _load(buf, "1024")
    suite.check(
        parse_number(buf.ptr(), 0, buf.length, n) == 4
        and n.kind == NUM_INT
        and n.i == 1024,
        "a token limit stays an integer rather than becoming 1024.0",
    )

    _load(buf, "9223372036854775807")
    suite.check(
        parse_number(buf.ptr(), 0, buf.length, n) > 0
        and n.kind == NUM_INT
        and n.i == 9223372036854775807,
        "the largest signed 64 bit value",
    )

    _load(buf, "-9223372036854775808")
    suite.check(
        parse_number(buf.ptr(), 0, buf.length, n) > 0
        and n.kind == NUM_INT
        and n.i == -9223372036854775807 - 1,
        "and the smallest, which is the one an off by one gets wrong",
    )

    _load(buf, "9223372036854775808")
    suite.check(
        parse_number(buf.ptr(), 0, buf.length, n) > 0 and n.kind == NUM_DOUBLE,
        "one past it becomes a double rather than wrapping",
    )

    _load(buf, "12345678901234567890")
    suite.check(
        parse_number(buf.ptr(), 0, buf.length, n) == 20
        and n.kind == NUM_DOUBLE,
        "and so does anything longer than the mantissa holds",
    )

    _load(buf, "-0")
    suite.check(
        parse_number(buf.ptr(), 0, buf.length, n) == 2 and n.i == 0,
        "negative zero written as an integer is zero",
    )

    _load(buf, "01")
    suite.check(
        parse_number(buf.ptr(), 0, buf.length, n) < 0,
        "a leading zero is two numbers and not one",
    )
    _load(buf, ".5")
    suite.check(
        parse_number(buf.ptr(), 0, buf.length, n) < 0,
        "a bare fraction is JavaScript and not JSON",
    )
    _load(buf, "1.")
    suite.check(
        parse_number(buf.ptr(), 0, buf.length, n) < 0,
        "and so is a trailing point",
    )
    _load(buf, "+1")
    suite.check(
        parse_number(buf.ptr(), 0, buf.length, n) < 0,
        "a leading plus is refused",
    )
    _load(buf, "1e")
    suite.check(
        parse_number(buf.ptr(), 0, buf.length, n) < 0,
        "an exponent with no digits is refused",
    )
    _load(buf, "1e+")
    suite.check(
        parse_number(buf.ptr(), 0, buf.length, n) < 0,
        "and a sign with no digits after it",
    )


def test_doubles(mut suite: Suite, counter: Int):
    suite.group("json numbers, the doubles that break converters")
    var buf = Buffer(256, counter)

    suite.check(
        _bits(_parse_double(buf, "0.1")) == _bits(0.1),
        "0.1 lands on the same bits the compiler produces",
    )
    suite.check(_bits(_parse_double(buf, "1.5")) == _bits(1.5), "1.5 is exact")
    suite.check(
        _bits(_parse_double(buf, "-1.25e3")) == _bits(-1250.0),
        "an exponent and a sign together",
    )
    suite.check(
        _bits(_parse_double(buf, "3.141592653589793"))
        == _bits(3.141592653589793),
        "pi to seventeen digits",
    )

    # The value that used to hang PHP's strtod, and the classic halfway case.
    suite.check(
        _bits(_parse_double(buf, "2.2250738585072011e-308"))
        == UInt64(0x000FFFFFFFFFFFFF),
        "the largest subnormal, which is the input that used to hang PHP",
    )
    suite.check(
        _bits(_parse_double(buf, "7.2057594037927933e16"))
        == _bits(72057594037927936.0),
        "a halfway case that a nearly correct rounder gets wrong",
    )
    suite.check(
        _bits(_parse_double(buf, "5e-324")) == UInt64(1),
        "the smallest subnormal is one bit and not zero",
    )
    suite.check(
        _bits(_parse_double(buf, "2.4703282292062327e-324")) == UInt64(0),
        "and just under half of it rounds to zero rather than to that bit",
    )
    suite.check(
        _bits(_parse_double(buf, "2.4703282292062328e-324")) == UInt64(1),
        "while just over it rounds up",
    )
    suite.check(
        _bits(_parse_double(buf, "1.7976931348623157e308"))
        == UInt64(0x7FEFFFFFFFFFFFFF),
        "the largest finite double",
    )

    var big = _parse_double(buf, "1e309")
    suite.check(big > 1.7976931348623157e308, "past it is an infinity")
    suite.check(
        _parse_double(buf, "-1e309") < -1.7976931348623157e308,
        "and so is the negative side",
    )
    suite.check(_parse_double(buf, "1e-400") == 0.0, "far under is a zero")

    # A number with more digits than the array holds, so the truncation flag is
    # the thing deciding the last bit.
    var many = String("1.")
    for _ in range(900):
        many += "0"
    many += "1"
    suite.check(
        _parse_double(buf, many) == 1.0,
        "nine hundred digits of padding does not move the answer",
    )

    var wide = String("")
    for _ in range(400):
        wide += "9"
    suite.check(
        _parse_double(buf, wide) > 1.7976931348623157e308,
        "and four hundred nines overflows to an infinity rather than wrapping",
    )

    var near = String("1")
    for _ in range(308):
        near += "0"
    suite.check(
        _parse_double(buf, near) == 1e308,
        "while three hundred and nine digits that do fit come out exactly",
    )


def test_double_printing(mut suite: Suite, counter: Int):
    suite.group("json numbers print as the shortest thing that reads back")
    var buf = Buffer(256, counter)

    _ = _print(buf, 0.1)
    suite.check(
        _same(buf.bytes(), "0.1"),
        "0.1 prints as 0.1 and not as 0.10000000000000001",
    )
    _ = _print(buf, 1.0)
    suite.check(_same(buf.bytes(), "1"), "a whole double drops the point")
    _ = _print(buf, -2.5)
    suite.check(_same(buf.bytes(), "-2.5"), "a sign and a fraction")
    _ = _print(buf, 3.141592653589793)
    suite.check(_same(buf.bytes(), "3.141592653589793"), "pi")
    _ = _print(buf, 1e20)
    suite.check(
        _same(buf.bytes(), "100000000000000000000"),
        "1e20 prints in full, which is where JavaScript puts the cutoff",
    )
    _ = _print(buf, 1e21)
    suite.check(
        _same(buf.bytes(), "1e+21"), "and 1e21 is the first one that does not"
    )
    _ = _print(buf, 1e-6)
    suite.check(_same(buf.bytes(), "0.000001"), "small numbers print in full")
    _ = _print(buf, 1e-7)
    suite.check(_same(buf.bytes(), "1e-7"), "down to the other cutoff")
    _ = _print(buf, 5e-324)
    suite.check(
        _same(buf.bytes(), "5e-324"),
        "the smallest subnormal prints as one digit",
    )
    _ = _print(buf, 1.7976931348623157e308)
    suite.check(
        _same(buf.bytes(), "1.7976931348623157e+308"),
        "and the largest finite double as seventeen",
    )
    _ = _print(buf, -0.0)
    suite.check(
        _same(buf.bytes(), "-0"),
        "negative zero keeps its sign, because it is a different double",
    )
    _ = _print(buf, 0.0)
    suite.check(_same(buf.bytes(), "0"), "and positive zero does not gain one")

    var nan = 0.0
    var inf = 1.0
    _ = _print(buf, nan / 0.0 - nan / 0.0)
    suite.check(
        _same(buf.bytes(), "null"),
        "a NaN has no JSON spelling, so it goes out as null",
    )
    _ = _print(buf, inf / 0.0)
    suite.check(_same(buf.bytes(), "null"), "and so does an infinity")


def test_round_trip(mut suite: Suite, counter: Int):
    suite.group("json numbers survive a round trip")
    var buf = Buffer(256, counter)
    var text = Buffer(256, counter)

    # A spread rather than a list of nice values, built from a cheap generator
    # so the inputs are not the ones the code was written against.
    var state = UInt64(0x2545F4914F6CDD1D)
    var checked = 0
    var wrong = 0
    var first_bad = UInt64(0)
    for _ in range(4000):
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        var bits = state
        var value = bitcast[DType.float64](bits)
        if value != value:
            continue
        if value > 1.7976931348623157e308 or value < -1.7976931348623157e308:
            continue
        text.clear()
        if not write_json_double(text, value):
            continue
        var back = _parse_double(
            buf, StringSlice(unsafe_from_utf8=text.bytes())
        )
        checked += 1
        if _bits(back) != _bits(value):
            if wrong == 0:
                first_bad = bits
            wrong += 1
    suite.check(
        wrong == 0 and checked > 3000,
        String("four thousand random doubles print and read back (checked ")
        + String(checked)
        + ", wrong "
        + String(wrong)
        + ", first "
        + String(first_bad)
        + ")",
    )

    var dec = Decimal()
    double_to_decimal(0.1, dec)
    suite.check(
        dec.nd == 1 and dec.d[0] == 1 and dec.dp == 0,
        "and the shortest form of 0.1 really is one digit",
    )


def test_reader_events(mut suite: Suite, counter: Int):
    suite.group("json streaming mode walks a document")
    var buf = Buffer(4096, counter)
    var reader = Reader(counter)

    _load(buf, '{"a":1,"b":[true,null,"x"],"c":{"d":-2.5}}')
    reader.begin(buf.bytes())

    suite.check(reader.next() == EV_OBJECT_BEGIN, "the object opens")
    suite.check(reader.next() == EV_KEY and reader.key_is("a"), "the first key")
    suite.check(
        reader.next() == EV_NUMBER and reader.number.i == 1, "its value"
    )
    suite.check(reader.next() == EV_KEY and reader.key_is("b"), "the second")
    suite.check(reader.next() == EV_ARRAY_BEGIN, "an array opens")
    suite.check(
        reader.next() == EV_BOOL and reader.bool_value, "true inside it"
    )
    suite.check(reader.next() == EV_NULL, "then a null")
    suite.check(
        reader.next() == EV_STRING and _same(reader.text(), "x"),
        "then a string",
    )
    suite.check(reader.next() == EV_ARRAY_END, "the array closes")
    suite.check(reader.next() == EV_KEY and reader.key_is("c"), "the third key")
    suite.check(reader.next() == EV_OBJECT_BEGIN, "a nested object")
    suite.check(reader.next() == EV_KEY and reader.key_is("d"), "its key")
    suite.check(
        reader.next() == EV_NUMBER and reader.number.d == -2.5, "its value"
    )
    suite.check(reader.next() == EV_OBJECT_END, "the nested object closes")
    suite.check(reader.next() == EV_OBJECT_END, "and so does the outer one")
    suite.check(reader.next() == EV_END, "then the document ends")
    suite.check(reader.next() == EV_END, "and it keeps ending")

    _load(buf, "  [ ]  ")
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_ARRAY_BEGIN
        and reader.next() == EV_ARRAY_END
        and reader.next() == EV_END,
        "an empty array with whitespace around it",
    )

    _load(buf, "{}")
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_OBJECT_BEGIN
        and reader.next() == EV_OBJECT_END
        and reader.next() == EV_END,
        "and an empty object",
    )

    _load(buf, "42")
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_NUMBER and reader.number.i == 42,
        "a bare number is a document on its own",
    )
    suite.check(reader.next() == EV_END, "and then it is over")


def test_reader_strings(mut suite: Suite, counter: Int):
    suite.group("json strings are spans until they cannot be")
    var buf = Buffer(4096, counter)
    var reader = Reader(counter)

    _load(buf, '"a plain string"')
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_STRING and _same(reader.text(), "a plain string"),
        "a string with no escapes comes back whole",
    )
    suite.check(
        not reader.str_decoded,
        "and it is a span into the buffer rather than a copy",
    )

    _load(buf, '"tab\\there\\nand \\"quoted\\" and \\\\ and \\/"')
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_STRING
        and _same(reader.text(), 'tab\there\nand "quoted" and \\ and /'),
        "every short escape decodes",
    )
    suite.check(reader.str_decoded, "and that one did have to be copied")

    _load(buf, '"\\u0041\\u00e9\\u20ac"')
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_STRING and _same(reader.text(), "Aé€"),
        "one, two and three byte code points come out as UTF-8",
    )

    _load(buf, '"\\ud83d\\ude00"')
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_STRING and _same(reader.text(), "😀"),
        "a surrogate pair becomes one four byte code point",
    )

    _load(buf, '"\\ud800"')
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_ERROR and reader.error == JSON_ESCAPE,
        "a lone high surrogate is refused rather than replaced",
    )

    _load(buf, '"\\udc00"')
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_ERROR and reader.error == JSON_ESCAPE,
        "and so is a lone low one",
    )

    _load(buf, '"\\x41"')
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_ERROR and reader.error == JSON_ESCAPE,
        "a JavaScript escape is not a JSON escape",
    )

    _load(buf, '"raw\nnewline"')
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_ERROR and reader.error == JSON_SYNTAX,
        "a raw newline inside a string is an error and not something to allow",
    )

    _load(buf, '"café"')
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_STRING and len(reader.text()) == 5,
        "valid UTF-8 passes through untouched",
    )

    _load_raw(buf, '"bad %ff byte"')
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_ERROR and reader.error == JSON_UTF8,
        "and a byte that is not UTF-8 is refused",
    )

    # A long run with one escape at the end, which is the case the copy loop is
    # written for.
    var long = String('"')
    for _ in range(300):
        long += "abcdefghij"
    long += '\\n"'
    _load(buf, long)
    reader.begin(buf.bytes())
    suite.check(
        reader.next() == EV_STRING and len(reader.text()) == 3001,
        "three thousand bytes and one escape decode in one copy",
    )


def _fails(
    mut suite: Suite,
    mut buf: Buffer,
    mut reader: Reader,
    text: StringSpan,
    code: Int,
    at: Int,
    name: String,
):
    """Parse `text`, expect it to fail with `code` at offset `at`.

    The offset is checked and not only the failure, because an offset is what
    turns a 400 into a two minute look at the body, and one that is right by
    accident is easy to write and impossible to notice later.
    """
    _load(buf, text)
    reader.begin(buf.bytes())
    var saw = EV_END
    while True:
        saw = reader.next()
        if saw == EV_ERROR or saw == EV_END:
            break
    if saw != EV_ERROR:
        suite.check(False, name + " (parsed instead of failing)")
        return
    if reader.error != code:
        suite.check(
            False, name + " (got " + String(error_text(reader.error)) + ")"
        )
        return
    suite.check(reader.error_at == at, name + " at " + String(reader.error_at))


def test_reader_errors(mut suite: Suite, counter: Int):
    suite.group("json refuses what is not JSON, and says where")
    var buf = Buffer(4096, counter)
    var reader = Reader(counter)

    _fails(
        suite,
        buf,
        reader,
        "[1,]",
        JSON_SYNTAX,
        3,
        "a trailing comma in an array",
    )
    _fails(
        suite,
        buf,
        reader,
        '{"a":1,}',
        JSON_SYNTAX,
        7,
        "a trailing comma in an object",
    )
    _fails(suite, buf, reader, '{"a" 1}', JSON_SYNTAX, 5, "a missing colon")
    _fails(suite, buf, reader, "[1 2]", JSON_SYNTAX, 3, "a missing comma")
    _fails(
        suite,
        buf,
        reader,
        "{}extra",
        JSON_TRAILING,
        2,
        "content after the document",
    )
    _fails(suite, buf, reader, "nul", JSON_SHORT, 0, "a truncated literal")
    _fails(
        suite, buf, reader, "[1", JSON_SHORT, 2, "a container that never closes"
    )
    _fails(
        suite,
        buf,
        reader,
        '{"a"',
        JSON_SYNTAX,
        4,
        "a key with nothing after it",
    )
    _fails(
        suite,
        buf,
        reader,
        "[}",
        JSON_SYNTAX,
        1,
        "a bracket closed by the wrong brace",
    )
    _fails(
        suite,
        buf,
        reader,
        "tru3",
        JSON_SYNTAX,
        0,
        "a literal that is nearly right",
    )
    _fails(suite, buf, reader, "", JSON_SHORT, 0, "an empty document")
    _fails(suite, buf, reader, "'single'", JSON_SYNTAX, 0, "single quotes")
    _fails(suite, buf, reader, "{a:1}", JSON_SYNTAX, 1, "an unquoted key")

    # Deep nesting is a memory attack and not a document, so the limit is a
    # refusal with a code rather than a stack that runs out somewhere in the
    # runtime.
    var deep = String("")
    for _ in range(200):
        deep += "["
    _load(buf, deep)
    reader.begin(buf.bytes())
    reader.configure_depth(64)
    var saw_depth = False
    for _ in range(300):
        var e = reader.next()
        if e == EV_ERROR:
            saw_depth = True
            break
        if e == EV_END:
            break
    suite.check(saw_depth, "two hundred open brackets hits the depth limit")


def test_skipping(mut suite: Suite, counter: Int):
    suite.group("json skips a value a caller does not want")
    var buf = Buffer(4096, counter)
    var reader = Reader(counter)

    _load(
        buf,
        '{"junk":{"a":[1,2,{"b":3}],"c":"x"},"wanted":7,"more":[[[]]],"last":1}',
    )
    reader.begin(buf.bytes())
    suite.check(reader.next() == EV_OBJECT_BEGIN, "the object opens")

    var wanted = 0
    var seen_keys = 0
    while True:
        var e = reader.next()
        if e == EV_OBJECT_END or e == EV_END or e == EV_ERROR:
            break
        if e != EV_KEY:
            continue
        seen_keys += 1
        if reader.key_is("wanted"):
            _ = reader.next()
            wanted = reader.number.i
        else:
            _ = reader.skip_next_value()
    suite.check(
        wanted == 7,
        "the one field that mattered is read past the ones that did not",
    )
    suite.check(seen_keys == 4, "and every key at that level was still seen")
    suite.check(reader.finish(), "and the rest of the document still parses")


def test_dom(mut suite: Suite, counter: Int):
    suite.group("json dom mode holds a document")
    var buf = Buffer(4096, counter)
    var reader = Reader(counter)
    var doc = Document(counter)

    _load(
        buf,
        (
            '{"model":"m","n":3,"t":0.5,"on":true,"off":null,'
            '"list":[1,"two",{"deep":[]}]}'
        ),
    )
    suite.check(parse(doc, reader, buf.bytes()), "a document parses")
    suite.check(doc.kind(doc.root) == JS_OBJECT, "the root is an object")
    suite.check(doc.size(doc.root) == 6, "with six members")
    suite.check(_same(doc.get_str(doc.root, "model"), "m"), "a string member")
    suite.check(doc.get_int(doc.root, "n") == 3, "an integer member")
    suite.check(doc.get_double(doc.root, "t") == 0.5, "a double member")
    suite.check(doc.get_bool(doc.root, "on"), "a bool member")
    suite.check(
        doc.kind(doc.get(doc.root, "model")) == JS_STRING
        and doc.kind(doc.get(doc.root, "n")) == JS_INT
        and doc.kind(doc.get(doc.root, "t")) == JS_DOUBLE
        and doc.kind(doc.get(doc.root, "on")) == JS_BOOL,
        "each of which kept the type it was written as",
    )
    suite.check(
        doc.kind(doc.get(doc.root, "off")) == JS_NULL, "and a null member"
    )
    suite.check(
        doc.get(doc.root, "absent") == NO_NODE,
        "a key that is not there is not there",
    )

    var list = doc.get(doc.root, "list")
    suite.check(doc.kind(list) == JS_ARRAY and doc.size(list) == 3, "an array")
    suite.check(doc.as_int(doc.at(list, 0)) == 1, "indexed by position")
    suite.check(_same(doc.text(doc.at(list, 1)), "two"), "and again")
    suite.check(
        doc.size(doc.get(doc.at(list, 2), "deep")) == 0,
        "an empty array nested three deep",
    )
    suite.check(
        doc.at(list, 3) == NO_NODE, "and one past the end is not a node"
    )

    # Key order is the reason this is not a hash map.
    var order = String("")
    var child = doc.first_child(doc.root)
    while child != NO_NODE:
        var k = doc.key(child)
        for i in range(len(k)):
            order += String(chr(Int(k[i])))
        order += ","
        child = doc.next_sibling(child)
    suite.check(
        order == "model,n,t,on,off,list,",
        "members come out in the order they were written",
    )

    _load(buf, '{"a":1,"a":2}')
    suite.check(parse(doc, reader, buf.bytes()), "a duplicated key parses")
    suite.check(
        doc.size(doc.root) == 2 and doc.get_int(doc.root, "a") == 1,
        "both members are kept and the first one wins",
    )

    _load(buf, '{"k\\ny":1}')
    suite.check(parse(doc, reader, buf.bytes()), "a key with an escape parses")
    suite.check(
        doc.get_int(doc.root, "k\ny") == 1,
        "and it is copied out of the scratch before the next string reuses it",
    )

    _load(buf, '["\\na","\\nb","\\nc"]')
    suite.check(parse(doc, reader, buf.bytes()), "three decoded strings parse")
    suite.check(
        _same(doc.text(doc.at(doc.root, 0)), "\na")
        and _same(doc.text(doc.at(doc.root, 1)), "\nb")
        and _same(doc.text(doc.at(doc.root, 2)), "\nc"),
        "and none of them is overwritten by the ones after it",
    )

    _load(buf, "{bad}")
    suite.check(
        not parse(doc, reader, buf.bytes()) and not doc.ok(),
        "a bad document fails and says so rather than half filling",
    )


def test_writer(mut suite: Suite, counter: Int):
    suite.group("json writes what it was given, in order")
    var w = Writer(counter, 1024)

    _ = w.begin_object()
    _ = w.field_str("z", "first")
    _ = w.field_int("a", -7)
    _ = w.key("nested")
    _ = w.begin_array()
    _ = w.int(1)
    _ = w.bool(False)
    _ = w.null()
    _ = w.string("x")
    _ = w.end_array()
    _ = w.field_double("t", 0.25)
    _ = w.end_object()
    suite.check(w.ok() and w.complete(), "a document is written and closed")
    suite.check(
        _same(
            w.bytes(),
            '{"z":"first","a":-7,"nested":[1,false,null,"x"],"t":0.25}',
        ),
        "commas and colons land where they belong and keys keep their order",
    )

    w.reset()
    _ = w.begin_array()
    _ = w.end_array()
    suite.check(_same(w.bytes(), "[]"), "an empty array")

    w.reset()
    _ = w.begin_object()
    _ = w.int(1)
    suite.check(
        not w.ok(), "a value with no key in front of it inside an object fails"
    )

    w.reset()
    _ = w.begin_object()
    _ = w.key("a")
    _ = w.end_object()
    suite.check(not w.ok(), "and so does a key with no value after it")

    w.reset()
    _ = w.begin_object()
    _ = w.end_array()
    suite.check(not w.ok(), "and closing with the wrong brace")

    w.reset()
    _ = w.begin_object()
    suite.check(
        w.ok() and not w.complete(),
        (
            "an unclosed document is not complete, which is worth asking before"
            " sending"
        ),
    )

    w.reset()
    _ = w.begin_array()
    _ = w.raw('{"already":"encoded"}'.as_bytes())
    _ = w.end_array()
    suite.check(
        _same(w.bytes(), '[{"already":"encoded"}]'),
        "a preencoded fragment goes in as a value",
    )

    var buf = Buffer(256, counter)
    _ = write_json_string(
        buf, 'a "quote" and a \\ and a \n and a \x01'.as_bytes()
    )
    suite.check(
        _same(
            buf.bytes(),
            '"a \\"quote\\" and a \\\\ and a \\n and a \\u0001"',
        ),
        "escaping covers the quote, the backslash and the controls",
    )

    buf.clear()
    _ = write_json_string(buf, "a / and café".as_bytes())
    suite.check(
        _same(buf.bytes(), '"a / and café"'),
        "and leaves the slash and the UTF-8 alone",
    )


def test_writer_reader_round_trip(mut suite: Suite, counter: Int):
    suite.group("json written here reads back here")
    var w = Writer(counter, 4096)
    var reader = Reader(counter)
    var doc = Document(counter)

    _ = w.begin_object()
    _ = w.field_str("content", 'He said "hi"\nthen left\tquickly')
    _ = w.field_str("emoji", "😀")
    _ = w.field_double("p", 0.1)
    _ = w.field_double("tiny", 5e-324)
    _ = w.field_int("big", 9223372036854775807)
    _ = w.end_object()
    suite.check(w.complete(), "the document is written")

    suite.check(parse(doc, reader, w.bytes()), "and parses back")
    suite.check(
        _same(
            doc.get_str(doc.root, "content"), 'He said "hi"\nthen left\tquickly'
        ),
        "the escaped string comes back as it went in",
    )
    suite.check(
        _same(doc.get_str(doc.root, "emoji"), "😀"),
        "and so does a four byte code point",
    )
    suite.check(
        _bits(doc.get_double(doc.root, "p")) == _bits(0.1),
        "0.1 comes back on the same bits",
    )
    suite.check(
        _bits(doc.get_double(doc.root, "tiny")) == UInt64(1),
        "and so does the smallest subnormal",
    )
    suite.check(
        doc.get_int(doc.root, "big") == 9223372036854775807,
        "and the largest integer is still an integer",
    )


def test_no_allocations(mut suite: Suite):
    suite.group("json streaming mode allocates nothing per document")
    var counter = AllocCounter()
    var buf = Buffer(8192, counter.raw())
    var reader = Reader(counter.raw(), 1024)

    var body = String('{"model":"m","messages":[')
    for i in range(40):
        if i > 0:
            body += ","
        body += '{"role":"user","content":"a message with a \\"quote\\" in it"}'
    body += '],"temperature":0.7,"max_tokens":512}'
    _load(buf, body)

    # A warmup, so what is measured is a steady state connection and not the
    # first few documents growing the scratch.
    for _ in range(8):
        reader.begin(buf.bytes())
        _ = reader.finish()

    var before = counter.total()
    var parsed = 0
    for _ in range(2000):
        reader.begin(buf.bytes())
        if reader.finish():
            parsed += 1
    var after = counter.total()

    suite.check(parsed == 2000, "two thousand documents parse")
    suite.check(
        after == before,
        String("and allocate nothing at all (")
        + String(after - before)
        + " allocations)",
    )
    suite.check(
        reader.decoded > 0,
        "including the strings that had to be decoded into the scratch",
    )
    counter.close()


def run(mut suite: Suite):
    var counter = AllocCounter()
    test_scan(suite, counter.raw())
    test_integers(suite, counter.raw())
    test_doubles(suite, counter.raw())
    test_double_printing(suite, counter.raw())
    test_round_trip(suite, counter.raw())
    test_reader_events(suite, counter.raw())
    test_reader_strings(suite, counter.raw())
    test_reader_errors(suite, counter.raw())
    test_skipping(suite, counter.raw())
    test_dom(suite, counter.raw())
    test_writer(suite, counter.raw())
    test_writer_reader_round_trip(suite, counter.raw())
    counter.close()
    test_no_allocations(suite)
