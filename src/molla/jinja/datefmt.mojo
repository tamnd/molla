"""The calendar, because `strftime_now` is in every Llama template.

Meta shipped a chat template that writes today's date into the system prompt,
everybody who fine tuned on it kept the line, and so a date formatter is now
part of what it takes to render a prompt correctly. The formats actually seen
are short, `%d %b %Y` and `%Y-%m-%d` and a handful more, but the directive is
whatever the model author typed, so this implements the set a person would
reasonably reach for and raises on anything else rather than passing it through.
Passing it through is what C does and it means a template asking for something
we do not know silently writes the directive into the prompt.

The time is local time, because `datetime.now()` is local time and the whole
point is to agree with what Python would have produced on the same machine. It
comes from `localtime_r`, so it respects `TZ` and the system zone the same way
Python does.
"""

from std.ffi import external_call
from std.memory import stack_allocation

from molla.sys.clock import unix_time

comptime _DAY_ABBR = String("SunMonTueWedThuFriSat")
comptime _MONTH_ABBR = String("JanFebMarAprMayJunJulAugSepOctNovDec")

comptime _DAY_NAMES = String(
    "Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|"
)

comptime _MONTH_NAMES = String(
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|"
)


def _nth(table: String, index: Int) -> String:
    """The nth bar separated piece of a table.

    A table rather than a list of strings because a list of strings cannot be
    a compile time constant here, and because a scan over sixty bytes to find
    the third bar is not something worth a data structure.
    """
    var data = table.as_bytes()
    var seen = 0
    var start = 0
    for i in range(len(data)):
        if data[i] != 0x7C:
            continue
        if seen == index:
            var raw = List[UInt8]()
            for j in range(start, i):
                raw.append(data[j])
            return String(StringSpan(unsafe_from_utf8=Span(raw)))
        seen += 1
        start = i + 1
    return String("")


struct Civil(Copyable, ImplicitlyCopyable, Movable):
    """A broken down time, with the same fields C's `struct tm` carries."""

    var year: Int
    var month: Int
    """One to twelve."""

    var day: Int
    var hour: Int
    var minute: Int
    var second: Int
    var weekday: Int
    """Zero is Sunday, which is what C uses and what `%w` prints."""

    var yearday: Int
    """One based, so the first of January is one."""

    def __init__(out self):
        self.year = 1970
        self.month = 1
        self.day = 1
        self.hour = 0
        self.minute = 0
        self.second = 0
        self.weekday = 4
        self.yearday = 1


def _leap(year: Int) -> Bool:
    if year % 4 != 0:
        return False
    if year % 100 != 0:
        return True
    return year % 400 == 0


def _days_before_month(month: Int) -> Int:
    """Days in the year before the first of `month`, ignoring leap years."""
    var before = 0
    var lengths = String("\x1f\x1c\x1f\x1e\x1f\x1e\x1f\x1f\x1e\x1f\x1e\x1f")
    var data = lengths.as_bytes()
    for i in range(month - 1):
        before += Int(data[i])
    return before


def civil_from_unix(now: Int) -> Civil:
    """UTC, from seconds since the epoch.

    Howard Hinnant's days to civil, the same one the HTTP date header uses. It
    shifts the era to start in March so that the leap day lands at the end of a
    year and the month length table stops needing a special case.
    """
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

    var out = Civil()
    out.year = y
    out.month = m
    out.day = d
    out.hour = secs // 3600
    out.minute = (secs % 3600) // 60
    out.second = secs % 60
    # 1970-01-01 was a Thursday, so day zero is index four counting from Sunday.
    var wd = (days + 4) % 7
    out.weekday = wd + 7 if wd < 0 else wd
    out.yearday = _days_before_month(m) + d
    if m > 2 and _leap(y):
        out.yearday += 1
    return out^


comptime _TM_SLOTS = 32
"""Room for a `struct tm` and then some.

Nine `int` fields on every platform, followed by a `long` and a pointer on both
glibc and macOS. Over allocating a stack slot costs nothing and means a libc
that carries extra fields cannot write past the end.
"""


def local_from_unix(now: Int) -> Civil:
    """Local time, from libc, so that it agrees with `datetime.now()`.

    Falls back to UTC if `localtime_r` fails, which it does when there is no
    zone database to read. A prompt with a UTC date in it is wrong by hours; a
    prompt with no date in it is wrong entirely, so the fallback is the better
    of the two.
    """
    var when = stack_allocation[1, Int64]()
    when.unsafe_store(0, Int64(now))
    var slot = stack_allocation[_TM_SLOTS, Int64]()
    for i in range(_TM_SLOTS):
        slot.unsafe_store(i, Int64(0))
    var rc = external_call["localtime_r", Int](when, slot)
    if rc == 0:
        return civil_from_unix(now)

    var fields = slot.unsafe_bitcast[Int32]()
    var out = Civil()
    out.second = Int(fields.unsafe_load(0))
    out.minute = Int(fields.unsafe_load(1))
    out.hour = Int(fields.unsafe_load(2))
    out.day = Int(fields.unsafe_load(3))
    out.month = Int(fields.unsafe_load(4)) + 1
    out.year = Int(fields.unsafe_load(5)) + 1900
    out.weekday = Int(fields.unsafe_load(6))
    out.yearday = Int(fields.unsafe_load(7)) + 1
    return out^


def _pad2(value: Int) -> String:
    if value < 10 and value >= 0:
        return "0" + String(value)
    return String(value)


def _pad3(value: Int) -> String:
    if value < 10 and value >= 0:
        return "00" + String(value)
    if value < 100 and value >= 0:
        return "0" + String(value)
    return String(value)


def _abbr(table: String, index: Int) -> String:
    var data = table.as_bytes()
    var raw = List[UInt8]()
    for i in range(index * 3, index * 3 + 3):
        raw.append(data[i])
    return String(StringSpan(unsafe_from_utf8=Span(raw)))


def _week_number(when: Civil, first_is_monday: Bool) -> Int:
    """`%U` and `%W`, which count from the first such weekday of the year."""
    var wd = (when.weekday + 6) % 7 if first_is_monday else when.weekday
    return (when.yearday + 6 - wd) // 7


def strftime(format: String, when: Civil) raises -> String:
    """One directive at a time, and anything unknown is an error.

    `%-d` and friends are the glibc extension that drops the padding, which
    templates do use, so they are handled here rather than left to fail.
    """
    var data = format.as_bytes()
    var out = String("")
    var at = 0
    while at < len(data):
        if data[at] != 0x25:
            var start = at
            while at < len(data) and data[at] != 0x25:
                at += 1
            var raw = List[UInt8]()
            for i in range(start, at):
                raw.append(data[i])
            out += String(StringSpan(unsafe_from_utf8=Span(raw)))
            continue

        at += 1
        if at >= len(data):
            raise Error("the time format ends with a lone percent sign")
        var bare = False
        if data[at] == 0x2D:
            bare = True
            at += 1
            if at >= len(data):
                raise Error("the time format ends with a lone percent sign")
        var c = data[at]
        at += 1

        if c == 0x25:  # %%
            out += "%"
        elif c == 0x59:  # %Y
            out += String(when.year)
        elif c == 0x79:  # %y
            out += String(when.year % 100) if bare else _pad2(when.year % 100)
        elif c == 0x6D:  # %m
            out += String(when.month) if bare else _pad2(when.month)
        elif c == 0x64:  # %d
            out += String(when.day) if bare else _pad2(when.day)
        elif c == 0x65:  # %e, the space padded day
            out += String(when.day) if when.day >= 10 else " " + String(
                when.day
            )
        elif c == 0x48:  # %H
            out += String(when.hour) if bare else _pad2(when.hour)
        elif c == 0x49:  # %I
            var twelve = when.hour % 12
            if twelve == 0:
                twelve = 12
            out += String(twelve) if bare else _pad2(twelve)
        elif c == 0x4D:  # %M
            out += String(when.minute) if bare else _pad2(when.minute)
        elif c == 0x53:  # %S
            out += String(when.second) if bare else _pad2(when.second)
        elif c == 0x66:  # %f, always six zeros because the clock gives seconds
            out += "000000"
        elif c == 0x70:  # %p
            out += "AM" if when.hour < 12 else "PM"
        elif c == 0x61:  # %a
            out += _abbr(_DAY_ABBR, when.weekday)
        elif c == 0x41:  # %A
            out += _nth(_DAY_NAMES, when.weekday)
        elif c == 0x62 or c == 0x68:  # %b and %h
            out += _abbr(_MONTH_ABBR, when.month - 1)
        elif c == 0x42:  # %B
            out += _nth(_MONTH_NAMES, when.month - 1)
        elif c == 0x6A:  # %j
            out += String(when.yearday) if bare else _pad3(when.yearday)
        elif c == 0x77:  # %w
            out += String(when.weekday)
        elif c == 0x75:  # %u, Monday is one
            out += String(7 if when.weekday == 0 else when.weekday)
        elif c == 0x55:  # %U
            out += _pad2(_week_number(when, False))
        elif c == 0x57:  # %W
            out += _pad2(_week_number(when, True))
        elif c == 0x46:  # %F
            out += String(when.year) + "-" + _pad2(when.month)
            out += "-" + _pad2(when.day)
        elif c == 0x44:  # %D
            out += _pad2(when.month) + "/" + _pad2(when.day)
            out += "/" + _pad2(when.year % 100)
        elif c == 0x54:  # %T
            out += _pad2(when.hour) + ":" + _pad2(when.minute)
            out += ":" + _pad2(when.second)
        elif c == 0x52:  # %R
            out += _pad2(when.hour) + ":" + _pad2(when.minute)
        elif c == 0x63:  # %c
            out += _abbr(_DAY_ABBR, when.weekday) + " "
            out += _abbr(_MONTH_ABBR, when.month - 1) + " "
            out += String(when.day) if when.day >= 10 else " " + String(
                when.day
            )
            out += " " + _pad2(when.hour) + ":" + _pad2(when.minute)
            out += ":" + _pad2(when.second) + " " + String(when.year)
        elif c == 0x78:  # %x
            out += _pad2(when.month) + "/" + _pad2(when.day)
            out += "/" + _pad2(when.year % 100)
        elif c == 0x58:  # %X
            out += _pad2(when.hour) + ":" + _pad2(when.minute)
            out += ":" + _pad2(when.second)
        elif c == 0x7A or c == 0x5A:
            # %z and %Z both write nothing, because `strftime_now` formats what
            # `datetime.now()` returns and that is a naive datetime with no zone
            # attached. Python prints an empty string for both, so we do too,
            # even though the zone is sitting right there in libc.
            pass
        elif c == 0x6E:  # %n
            out += "\n"
        elif c == 0x74:  # %t
            out += "\t"
        elif c == 0x73:  # %s
            out += String(unix_time())
        else:
            var raw = List[UInt8]()
            raw.append(c)
            raise Error(
                "the time format uses '%"
                + String(StringSpan(unsafe_from_utf8=Span(raw)))
                + "', which molla does not implement"
            )
    return out


def strftime_now(format: String, now: Int) raises -> String:
    """What the global of the same name does, which is local time, now.

    A `now` of zero means the real clock. Anything else is a pinned time in
    seconds since the epoch, which is what the conformance run passes so that
    a template stamping today's date into the prompt still has one answer.
    """
    return strftime(format, local_from_unix(unix_time() if now == 0 else now))
