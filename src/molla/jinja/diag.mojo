"""Where it went wrong, said in a way somebody can act on.

Every error out of this package carries the line, the column, and the line of
template it happened on with a caret under the offending byte. That is not
politeness. A chat template arrives inside a model repository written by
somebody else, it is the only artifact standing between a downloaded model and
a prompt the model was never trained on, and the thing most likely to go wrong
is a construct we deliberately do not support. An error saying `unsupported
construct` and nothing else sends the reader to grep a fifteen kilobyte string
with no newlines in it. An error that quotes the line does not.

The column is counted in bytes, not characters. Templates that fail are almost
always failing on ASCII, and a byte column is the one a text editor's byte
offset agrees with.
"""


def line_col(data: Span[UInt8, _], at: Int) -> Tuple[Int, Int]:
    """The one based line and column of a byte offset."""
    var line = 1
    var col = 1
    var i = 0
    var stop = at if at < len(data) else len(data)
    while i < stop:
        if data[i] == 0x0A:
            line += 1
            col = 1
        else:
            col += 1
        i += 1
    return (line, col)


def slice_text(data: Span[UInt8, _], start: Int, end: Int) -> String:
    """The bytes between two offsets as a string, clamped to the buffer."""
    var lo = start if start > 0 else 0
    var hi = end if end < len(data) else len(data)
    var raw = List[UInt8]()
    for i in range(lo, hi):
        raw.append(data[i])
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def _line_bounds(data: Span[UInt8, _], at: Int) -> Tuple[Int, Int]:
    """The offsets of the line containing `at`, without its newline."""
    var stop = at if at < len(data) else len(data)
    var start = stop
    while start > 0 and data[start - 1] != 0x0A:
        start -= 1
    var end = stop
    while end < len(data) and data[end] != 0x0A:
        end += 1
    return (start, end)


comptime SNIPPET_WIDTH = 96
"""How much of the offending line to quote.

Chat templates are frequently written as one line thousands of bytes long, so
quoting the whole line would be worse than quoting none of it. This is a window
around the caret, and the caret is what makes the window readable.
"""


def snippet(data: Span[UInt8, _], at: Int) -> Tuple[String, Int]:
    """A readable piece of the line around `at`, and where the caret goes.

    Whitespace in the quoted piece is left alone apart from tabs, which become
    single spaces so the caret lands under the byte it is pointing at rather
    than wherever the terminal decided the tab stop was.
    """
    var bounds = _line_bounds(data, at)
    var start = bounds[0]
    var end = bounds[1]
    var stop = at if at < len(data) else len(data)

    var half = SNIPPET_WIDTH // 2
    var from_ = start
    if stop - start > half:
        from_ = stop - half
    var to = end
    if to - from_ > SNIPPET_WIDTH:
        to = from_ + SNIPPET_WIDTH

    var raw = List[UInt8]()
    for i in range(from_, to):
        raw.append(32 if data[i] == 0x09 else data[i])
    var text = String(StringSpan(unsafe_from_utf8=Span(raw)))
    if from_ > start:
        text = "..." + text
    if to < end:
        text += "..."
    var caret = stop - from_
    if from_ > start:
        caret += 3
    return (text, caret)


def fail(data: Span[UInt8, _], at: Int, message: String) raises:
    """Raise with the position and the line it happened on.

    The caller writes the message and nothing else. Everything about where it
    happened is added here, so no call site can forget it and no two call sites
    can disagree about the format.
    """
    var pos = line_col(data, at)
    var quoted = snippet(data, at)
    var out = String("template line ")
    out += String(pos[0])
    out += ", column "
    out += String(pos[1])
    out += ": "
    out += message
    out += "\n  "
    out += quoted[0]
    out += "\n  "
    for _ in range(quoted[1]):
        out += " "
    out += "^"
    raise Error(out)
