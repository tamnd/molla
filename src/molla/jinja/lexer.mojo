"""Template source to a flat list of tokens.

A chat template is a few kilobytes at most and is lexed once per model, so the
whole token list is built up front rather than streamed. That buys the parser
unlimited lookahead for free, which matters more than it sounds: telling `{% for
x in y %}` from `{% for x, y in z %}` and telling a dict literal from the end of
a variable block both need to look further ahead than one token.

## The three delimiters, and the fourth thing that looks like one

`{{ ... }}` writes a value, `{% ... %}` runs a statement, and `{# ... #}` is a
comment that produces nothing. Everything else is text and is copied out
verbatim. The fourth thing is `{% raw %}`, which turns the lexer off until
`{% endraw %}` and hands back what was between them as text, and templates that
print JSON examples use it.

Inside a code block a closing brace is ambiguous, because `{{ {'a': 1} }}` ends
with two of them that are not the end delimiter followed by one that is. So the
lexer counts open brackets and only looks for the end delimiter at depth zero,
which is what Jinja itself does and the only rule that gets this right.

## Whitespace, which is not a detail here

Chat templates are whitespace sensitive in a way ordinary templates are not. A
stray newline between two turns is a different prompt, and it is a difference
the model notices and nobody reading the output does. Three separate mechanisms
control it and all three are on:

`trim_blocks` eats the newline directly after a statement or comment tag, so a
`{% if %}` on its own line does not emit that line's newline. `lstrip_blocks`
eats the spaces and tabs between the start of a line and a statement tag, so an
indented `{% if %}` does not emit its indentation. Both are what
`transformers.apply_chat_template` sets, and matching it is the whole point.

The explicit markers override both. `{%-` strips all whitespace before the tag
including newlines, `-%}` strips all whitespace after it, and `{%+` turns
`lstrip_blocks` off for one tag. The explicit form wins because the template
author wrote it on purpose.
"""

from molla.jinja.diag import fail
from molla.text.utf8 import encode

comptime T_EOF = 0
comptime T_TEXT = 1
comptime T_VAR_BEGIN = 2
comptime T_VAR_END = 3
comptime T_BLOCK_BEGIN = 4
comptime T_BLOCK_END = 5
comptime T_NAME = 6
comptime T_STRING = 7
comptime T_INT = 8
comptime T_FLOAT = 9
comptime T_OP = 10


struct Token(Copyable, ImplicitlyCopyable, Movable):
    """One token. `text` carries the name, the decoded string, or the operator.

    `at` is the byte offset in the source, which is all the diagnostics need,
    since line and column are recomputed from it only when something fails.
    """

    var kind: Int
    var text: String
    var i: Int
    var f: Float64
    var at: Int

    def __init__(out self, kind: Int, at: Int):
        self.kind = kind
        self.text = String("")
        self.i = 0
        self.f = 0.0
        self.at = at


def _is_space(c: UInt8) -> Bool:
    return c == 0x20 or c == 0x09 or c == 0x0A or c == 0x0D or c == 0x0C


def _is_digit(c: UInt8) -> Bool:
    return c >= 0x30 and c <= 0x39


def _is_name_start(c: UInt8) -> Bool:
    # Jinja allows any Unicode letter in an identifier. No chat template in the
    # corpus uses one, and accepting the high bytes would mean accepting them in
    # the middle of a malformed sequence too, so identifiers are ASCII here and
    # anything else is a syntax error with a caret under it.
    return (c >= 0x41 and c <= 0x5A) or (c >= 0x61 and c <= 0x7A) or c == 0x5F


def _is_name(c: UInt8) -> Bool:
    return _is_name_start(c) or _is_digit(c)


def _rstrip(mut text: List[UInt8]):
    while len(text) > 0 and _is_space(text[len(text) - 1]):
        _ = text.pop()


def _rstrip_inline(data: Span[UInt8, _], run_at: Int, mut text: List[UInt8]):
    """Spaces and tabs off the end, stopping at a newline.

    This is `lstrip_blocks`: the indentation in front of a statement tag goes,
    and the newline that ended the line before it stays.

    The run of text being trimmed is not the whole line. `{{ x }} {% if %}` has
    one space between the two tags, and that space is not indentation because
    the line started earlier, so the source has to be consulted rather than just
    the text collected since the last tag.
    """
    var i = len(text)
    while i > 0 and (text[i - 1] == 0x20 or text[i - 1] == 0x09):
        i -= 1
    if i > 0:
        if text[i - 1] != 0x0A:
            return
    elif run_at > 0 and data[run_at - 1] != 0x0A:
        return
    while len(text) > i:
        _ = text.pop()


def _hex_value(c: UInt8) -> Int:
    if c >= 0x30 and c <= 0x39:
        return Int(c) - 0x30
    if c >= 0x61 and c <= 0x66:
        return Int(c) - 0x61 + 10
    if c >= 0x41 and c <= 0x46:
        return Int(c) - 0x41 + 10
    return -1


def _starts(data: Span[UInt8, _], at: Int, a: UInt8, b: UInt8) -> Bool:
    return at + 1 < len(data) and data[at] == a and data[at + 1] == b


def _read_string(
    data: Span[UInt8, _], mut at: Int, mut tok: Token
) raises -> None:
    """A quoted literal, with the escapes Python understands.

    An unknown escape keeps the backslash, which is what Python does and what
    templates writing Windows paths or regular expressions rely on.
    """
    var quote = data[at]
    var start = at
    at += 1
    var raw = List[UInt8]()
    while True:
        if at >= len(data):
            fail(data, start, "unterminated string")
        var c = data[at]
        if c == quote:
            at += 1
            break
        if c != 0x5C:
            raw.append(c)
            at += 1
            continue
        at += 1
        if at >= len(data):
            fail(data, start, "unterminated string")
        var e = data[at]
        at += 1
        if e == 0x6E:
            raw.append(0x0A)
        elif e == 0x74:
            raw.append(0x09)
        elif e == 0x72:
            raw.append(0x0D)
        elif e >= 0x30 and e <= 0x37:
            # Octal, one to three digits, which is where `\0` comes from.
            var value = Int(e) - 0x30
            var digits = 1
            while digits < 3 and at < len(data):
                var d = data[at]
                if d < 0x30 or d > 0x37:
                    break
                value = value * 8 + (Int(d) - 0x30)
                at += 1
                digits += 1
            raw.append(UInt8(value & 0xFF))
        elif e == 0x78 or e == 0x75 or e == 0x55:
            # `\xhh`, `\uhhhh` and `\Uhhhhhhhh`. Python decodes these in a
            # string literal, so a template writing `\u200b` for a zero width
            # space has to get the character and not the six letters.
            var want = 2 if e == 0x78 else (4 if e == 0x75 else 8)
            var value = 0
            for _ in range(want):
                if at >= len(data):
                    fail(data, start, "a short escape at the end of a string")
                var d = _hex_value(data[at])
                if d < 0:
                    fail(data, at, "this is not a hexadecimal digit")
                value = value * 16 + d
                at += 1
            if e == 0x78:
                raw.append(UInt8(value))
            elif value > 0x10FFFF:
                fail(data, start, "an escape names no character")
            else:
                encode(value, raw)
        elif e == 0x4E:
            fail(
                data,
                at - 2,
                "`\\N{...}` needs the Unicode name table and is not supported",
            )
        elif e == 0x0A:
            # A backslash at the end of a line continues the string, so nothing
            # goes into it.
            pass
        elif e == 0x61:
            raw.append(0x07)
        elif e == 0x62:
            raw.append(0x08)
        elif e == 0x66:
            raw.append(0x0C)
        elif e == 0x76:
            raw.append(0x0B)
        elif e == 0x5C or e == 0x27 or e == 0x22:
            raw.append(e)
        else:
            raw.append(0x5C)
            raw.append(e)
    tok.text = String(StringSpan(unsafe_from_utf8=Span(raw)))


def _read_number(data: Span[UInt8, _], mut at: Int, mut tok: Token) raises:
    """An integer or a float, with underscores allowed as digit separators."""
    var digits = List[UInt8]()
    var is_float = False
    while at < len(data) and (_is_digit(data[at]) or data[at] == 0x5F):
        if data[at] != 0x5F:
            digits.append(data[at])
        at += 1
    # A dot is only part of the number when a digit follows it. Otherwise it is
    # attribute access on an integer, which `1.foo` is not but `(1).foo` is, and
    # more usefully it is the `..` nobody writes.
    if at + 1 < len(data) and data[at] == 0x2E and _is_digit(data[at + 1]):
        is_float = True
        digits.append(0x2E)
        at += 1
        while at < len(data) and (_is_digit(data[at]) or data[at] == 0x5F):
            if data[at] != 0x5F:
                digits.append(data[at])
            at += 1
    if at < len(data) and (data[at] == 0x65 or data[at] == 0x45):
        var probe = at + 1
        if probe < len(data) and (data[probe] == 0x2B or data[probe] == 0x2D):
            probe += 1
        if probe < len(data) and _is_digit(data[probe]):
            is_float = True
            digits.append(0x65)
            at += 1
            if data[at] == 0x2B or data[at] == 0x2D:
                digits.append(data[at])
                at += 1
            while at < len(data) and _is_digit(data[at]):
                digits.append(data[at])
                at += 1

    var text = String(StringSpan(unsafe_from_utf8=Span(digits)))
    if is_float:
        tok.kind = T_FLOAT
        tok.f = Float64(text)
    else:
        tok.kind = T_INT
        tok.i = Int(text)


comptime _TWO_CHAR = String("** // == != >= <= ")
"""The two byte operators, looked for before the one byte ones.

Written as one string with the pairs separated by spaces so adding one is a
three character edit rather than a new branch.
"""

comptime _ONE_CHAR = String("+-*/%~()[]{},:.|=<>")


def _read_op(data: Span[UInt8, _], mut at: Int, mut tok: Token) raises:
    var two = _TWO_CHAR.as_bytes()
    var i = 0
    while i + 1 < len(two):
        if _starts(data, at, two[i], two[i + 1]):
            var pair = List[UInt8]()
            pair.append(two[i])
            pair.append(two[i + 1])
            tok.text = String(StringSpan(unsafe_from_utf8=Span(pair)))
            at += 2
            return
        i += 3

    var one = _ONE_CHAR.as_bytes()
    for j in range(len(one)):
        if data[at] == one[j]:
            var single = List[UInt8]()
            single.append(one[j])
            tok.text = String(StringSpan(unsafe_from_utf8=Span(single)))
            at += 1
            return
    fail(data, at, "unexpected character in an expression")


def _skip_raw(data: Span[UInt8, _], mut at: Int, mut raw: List[UInt8]) raises:
    """Everything up to `{% endraw %}`, copied out untouched.

    The end tag is found by scanning for `{%` and reading the name after it,
    which means a `{% endraw %}` written inside a string in the raw block still
    ends it. That is what Jinja does, because a raw block has no expressions in
    it and therefore no strings to be inside.
    """
    var start = at
    while at < len(data):
        if not _starts(data, at, 0x7B, 0x25):
            raw.append(data[at])
            at += 1
            continue
        var probe = at + 2
        if probe < len(data) and (data[probe] == 0x2D or data[probe] == 0x2B):
            probe += 1
        while probe < len(data) and _is_space(data[probe]):
            probe += 1
        var name_at = probe
        while probe < len(data) and _is_name(data[probe]):
            probe += 1
        if probe - name_at == 6:
            var same = True
            var want = String("endraw").as_bytes()
            for k in range(6):
                if data[name_at + k] != want[k]:
                    same = False
                    break
            if same:
                # The strip marker on the closing tag applies to the raw text,
                # exactly as it would to ordinary text before a tag.
                if data[at + 2] == 0x2D:
                    _rstrip(raw)
                while probe < len(data) and _is_space(data[probe]):
                    probe += 1
                if not _starts(data, probe, 0x25, 0x7D):
                    fail(data, at, "a raw block ends with `endraw %}`")
                at = probe + 2
                return
        raw.append(data[at])
        at += 1
    fail(data, start, "unterminated raw block, no `endraw`")


def tokenize(source: String) raises -> List[Token]:
    """The whole template, in one pass, with whitespace control applied."""
    var data = source.as_bytes()
    var out = List[Token]()
    var at = 0
    var text = List[UInt8]()
    var text_at = 0
    var strip_next = False

    while at < len(data):
        if data[at] == 0x0D:
            # Newlines are normalised the way the reference normalises them, so
            # a template saved on Windows renders the same prompt as the same
            # template saved anywhere else. The offsets kept for error messages
            # still point into the original bytes.
            text.append(0x0A)
            at += 1
            if at < len(data) and data[at] == 0x0A:
                at += 1
            continue
        if data[at] != 0x7B or at + 1 >= len(data):
            text.append(data[at])
            at += 1
            continue
        var second = data[at + 1]
        if second != 0x7B and second != 0x25 and second != 0x23:
            text.append(data[at])
            at += 1
            continue

        # A tag starts here. Decide what the pending text keeps first, because
        # the marker that decides it is inside the tag.
        var marker_at = at + 2
        var minus = marker_at < len(data) and data[marker_at] == 0x2D
        var plus = marker_at < len(data) and data[marker_at] == 0x2B
        if minus:
            _rstrip(text)
        elif second != 0x7B and not plus:
            _rstrip_inline(data, text_at, text)

        if strip_next:
            # A `-` on the previous tag's closing delimiter. The stripping has
            # to happen here rather than there because the text had not been
            # read yet.
            var i = 0
            while i < len(text) and _is_space(text[i]):
                i += 1
            var kept = List[UInt8]()
            for j in range(i, len(text)):
                kept.append(text[j])
            text = kept^
            strip_next = False

        if len(text) > 0:
            var tok = Token(T_TEXT, text_at)
            tok.text = String(StringSpan(unsafe_from_utf8=Span(text)))
            out.append(tok)
            text = List[UInt8]()

        at += 2
        if minus or plus:
            at += 1

        if second == 0x23:
            # A comment. Nothing comes out of it and nothing inside it is
            # lexed, which is what lets a template comment out broken markup.
            var end = at
            var closed = False
            while end < len(data):
                if _starts(data, end, 0x23, 0x7D):
                    closed = True
                    break
                if data[end] == 0x2D and _starts(data, end + 1, 0x23, 0x7D):
                    closed = True
                    break
                end += 1
            if not closed:
                fail(data, at - 2, "unterminated comment")
            if data[end] == 0x2D:
                strip_next = True
                end += 1
            at = end + 2
            if not strip_next and at < len(data) and data[at] == 0x0A:
                at += 1
            text_at = at
            continue

        var is_block = second == 0x25
        out.append(Token(T_BLOCK_BEGIN if is_block else T_VAR_BEGIN, at - 2))

        # `{% raw %}` is a statement as far as the delimiters go and not one as
        # far as anything else goes, so it is recognised here rather than in the
        # parser, and the block token just emitted is taken back off.
        if is_block:
            var probe = at
            while probe < len(data) and _is_space(data[probe]):
                probe += 1
            var name_at = probe
            while probe < len(data) and _is_name(data[probe]):
                probe += 1
            if probe - name_at == 3 and _raw_name(data, name_at):
                while probe < len(data) and _is_space(data[probe]):
                    probe += 1
                var trim = probe < len(data) and data[probe] == 0x2D
                if trim:
                    probe += 1
                if not _starts(data, probe, 0x25, 0x7D):
                    fail(data, at, "a raw block opens with `raw %}`")
                _ = out.pop()
                at = probe + 2
                if at < len(data) and data[at] == 0x0A:
                    at += 1
                var body = List[UInt8]()
                text_at = at
                _skip_raw(data, at, body)
                if len(body) > 0:
                    var tok = Token(T_TEXT, text_at)
                    tok.text = String(StringSpan(unsafe_from_utf8=Span(body)))
                    out.append(tok)
                if at < len(data) and data[at] == 0x0A:
                    at += 1
                text_at = at
                continue

        var depth = 0
        while True:
            while at < len(data) and _is_space(data[at]):
                at += 1
            if at >= len(data):
                fail(data, out[len(out) - 1].at, "unterminated tag")

            var c = data[at]
            if depth == 0:
                var close_at = at
                var dash = c == 0x2D and at + 2 < len(data)
                var body_at = at + 1 if dash else at
                var closes = _starts(
                    data, body_at, 0x25, 0x7D
                ) if is_block else _starts(data, body_at, 0x7D, 0x7D)
                if closes:
                    out.append(
                        Token(T_BLOCK_END if is_block else T_VAR_END, close_at)
                    )
                    at = body_at + 2
                    if dash:
                        strip_next = True
                    elif is_block and at < len(data) and data[at] == 0x0A:
                        at += 1
                    break

            var tok = Token(T_OP, at)
            if c == 0x27 or c == 0x22:
                tok.kind = T_STRING
                _read_string(data, at, tok)
            elif _is_digit(c):
                _read_number(data, at, tok)
            elif _is_name_start(c):
                tok.kind = T_NAME
                var name_at = at
                while at < len(data) and _is_name(data[at]):
                    at += 1
                tok.text = _name_text(data, name_at, at)
            else:
                if c == 0x28 or c == 0x5B or c == 0x7B:
                    depth += 1
                elif c == 0x29 or c == 0x5D or c == 0x7D:
                    depth -= 1
                    if depth < 0:
                        fail(data, at, "unbalanced closing bracket")
                _read_op(data, at, tok)
            out.append(tok)
        text_at = at

    if strip_next:
        var i = 0
        while i < len(text) and _is_space(text[i]):
            i += 1
        var kept = List[UInt8]()
        for j in range(i, len(text)):
            kept.append(text[j])
        text = kept^
    # One trailing newline goes, which is `keep_trailing_newline` off. A file
    # that ends with a newline because every text file does would otherwise put
    # that newline in the prompt, and the reference does not.
    if len(text) > 0 and text[len(text) - 1] == 0x0A:
        _ = text.pop()
    if len(text) > 0:
        var tok = Token(T_TEXT, text_at)
        tok.text = String(StringSpan(unsafe_from_utf8=Span(text)))
        out.append(tok)
    out.append(Token(T_EOF, len(data)))
    return out^


def _raw_name(data: Span[UInt8, _], at: Int) -> Bool:
    return data[at] == 0x72 and data[at + 1] == 0x61 and data[at + 2] == 0x77


def _name_text(data: Span[UInt8, _], start: Int, end: Int) -> String:
    var raw = List[UInt8]()
    for i in range(start, end):
        raw.append(data[i])
    return String(StringSpan(unsafe_from_utf8=Span(raw)))
