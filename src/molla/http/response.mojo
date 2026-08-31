"""Fixed responses, built once and refreshed a field at a time.

The spike serves one body, so the entire response including headers is
assembled at startup and every request is a `memcpy` of it. That is the point.
A server that formats a status line and a header block per request is measuring
its formatter, and we want the number for the socket and parser path.

The only part that cannot be frozen is `Date`, which RFC 9110 says an origin
server should send and which changes once a second. Recomputing it per request
would mean a `time` call and a date conversion on the hot path, so it is written
into the prebuilt buffer at a known offset and refreshed when the second rolls
over. This is what nginx does and it is the reason the benchmark can include a
Date header honestly rather than dropping it and hoping nobody asks.

The date conversion is Howard Hinnant's civil_from_days, which is integer only
and has no table and no branch on leap years. There is no `gmtime` here because
that would be another libc call with a static buffer behind it.
"""

from std.memory import unsafe_memcpy

from molla.sys.clock import unix_time

comptime DATE_LENGTH = 29
"""`Sun, 06 Nov 1994 08:49:37 GMT` is fixed width, which is what makes patching
it in place possible."""

comptime DAY_NAMES = "SunMonTueWedThuFriSat"
comptime MONTH_NAMES = "JanFebMarAprMayJunJulAugSepOctNovDec"


def _two_digits[o: MutOrigin](buf: Pointer[UInt8, o], at: Int, value: Int):
    buf.unsafe_store(at, UInt8(48 + (value // 10) % 10))
    buf.unsafe_store(at + 1, UInt8(48 + value % 10))


def format_http_date[o: MutOrigin](buf: Pointer[UInt8, o], at: Int, now: Int):
    """Write an IMF-fixdate at `at`. Always exactly `DATE_LENGTH` bytes."""
    var days = now // 86400
    var secs = now % 86400
    if secs < 0:
        secs += 86400
        days -= 1

    var z = days + 719468
    var era = (z if z >= 0 else z - 146096) // 146097
    var doe = z - era * 146097
    var yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    var y = yoe + era * 400
    var doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    var mp = (5 * doy + 2) // 153
    var d = doy - (153 * mp + 2) // 5 + 1
    var m = mp + (3 if mp < 10 else -9)
    if m <= 2:
        y += 1

    # 1970-01-01 was a Thursday, so day 0 is index 4 counting from Sunday.
    var weekday = (days + 4) % 7
    if weekday < 0:
        weekday += 7

    var day_names = DAY_NAMES.unsafe_ptr()
    var month_names = MONTH_NAMES.unsafe_ptr()

    for i in range(3):
        buf.unsafe_store(at + i, day_names.unsafe_load(weekday * 3 + i))
    buf.unsafe_store(at + 3, UInt8(44))  # comma
    buf.unsafe_store(at + 4, UInt8(32))
    _two_digits(buf, at + 5, d)
    buf.unsafe_store(at + 7, UInt8(32))
    for i in range(3):
        buf.unsafe_store(at + 8 + i, month_names.unsafe_load((m - 1) * 3 + i))
    buf.unsafe_store(at + 11, UInt8(32))
    buf.unsafe_store(at + 12, UInt8(48 + (y // 1000) % 10))
    buf.unsafe_store(at + 13, UInt8(48 + (y // 100) % 10))
    buf.unsafe_store(at + 14, UInt8(48 + (y // 10) % 10))
    buf.unsafe_store(at + 15, UInt8(48 + y % 10))
    buf.unsafe_store(at + 16, UInt8(32))
    _two_digits(buf, at + 17, secs // 3600)
    buf.unsafe_store(at + 19, UInt8(58))
    _two_digits(buf, at + 20, (secs % 3600) // 60)
    buf.unsafe_store(at + 22, UInt8(58))
    _two_digits(buf, at + 23, secs % 60)
    buf.unsafe_store(at + 25, UInt8(32))
    buf.unsafe_store(at + 26, UInt8(71))  # G
    buf.unsafe_store(at + 27, UInt8(77))  # M
    buf.unsafe_store(at + 28, UInt8(84))  # T


def _append(mut out: List[UInt8], text: StringSpan):
    var p = text.unsafe_ptr()
    for i in range(text.byte_length()):
        out.append(p.unsafe_load(i))


def _append_int(mut out: List[UInt8], value: Int):
    if value == 0:
        out.append(UInt8(48))
        return
    var digits = List[UInt8]()
    var n = value
    while n > 0:
        digits.append(UInt8(48 + n % 10))
        n //= 10
    for i in range(len(digits)):
        out.append(digits[len(digits) - 1 - i])


struct Responder(Movable):
    """The prebuilt responses for the fixed body, one per connection outcome."""

    var keep: List[UInt8]
    """Sent when the connection stays open."""

    var close: List[UInt8]
    """Same body, plus `Connection: close`."""

    var keep_date_at: Int
    var close_date_at: Int
    var last_second: Int
    var body_length: Int

    def __init__(out self, body: StringSpan):
        self.body_length = body.byte_length()
        self.keep = List[UInt8]()
        self.close = List[UInt8]()
        self.keep_date_at = 0
        self.close_date_at = 0
        self.last_second = 0

        self.keep_date_at = Responder._build(self.keep, body, False)
        self.close_date_at = Responder._build(self.close, body, True)
        self.refresh()

    @staticmethod
    def _build(mut out: List[UInt8], body: StringSpan, closing: Bool) -> Int:
        """Assemble one response and return where the date field landed."""
        _append(out, "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n")
        _append(out, "Content-Length: ")
        _append_int(out, body.byte_length())
        _append(out, "\r\nServer: molla\r\n")
        if closing:
            _append(out, "Connection: close\r\n")
        _append(out, "Date: ")
        var date_at = len(out)
        for _ in range(DATE_LENGTH):
            out.append(UInt8(32))
        _append(out, "\r\n\r\n")
        _append(out, body)
        return date_at

    def refresh(mut self):
        """Rewrite both date fields if the second has changed.

        One `time` call per pass of the event loop rather than per request,
        which for a loop handling a batch of events is close to free.
        """
        var now = unix_time()
        if now == self.last_second:
            return
        self.last_second = now
        format_http_date(self.keep.unsafe_ptr(), self.keep_date_at, now)
        format_http_date(self.close.unsafe_ptr(), self.close_date_at, now)


def status_text(status: Int) -> StaticString:
    if status == 400:
        return "Bad Request"
    if status == 413:
        return "Content Too Large"
    if status == 414:
        return "URI Too Long"
    if status == 431:
        return "Request Header Fields Too Large"
    if status == 501:
        return "Not Implemented"
    if status == 505:
        return "HTTP Version Not Supported"
    return "Error"


def build_error(status: Int) -> List[UInt8]:
    """A complete error response. Always closes the connection.

    Not prebuilt because it is not the hot path. A server whose error responses
    need to be fast has a different problem.
    """
    var text = status_text(status)
    var out = List[UInt8]()
    _append(out, "HTTP/1.1 ")
    _append_int(out, status)
    _append(out, " ")
    _append(out, text)
    _append(out, "\r\nContent-Type: text/plain\r\nContent-Length: ")
    _append_int(out, text.byte_length() + 1)
    _append(out, "\r\nConnection: close\r\nServer: molla\r\nDate: ")
    var date_at = len(out)
    for _ in range(DATE_LENGTH):
        out.append(UInt8(32))
    _append(out, "\r\n\r\n")
    _append(out, text)
    out.append(UInt8(10))
    format_http_date(out.unsafe_ptr(), date_at, unix_time())
    return out^
