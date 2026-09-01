"""The HTTP soak, in the small, and the client parser it leans on.

Issue #18. The soak that matters runs for an hour and cannot live in a test
suite, so what is here is the two things that would make an hour long run lie.

The first is the client side framing. `Wire` is what tells the soak that an
answer arrived, and a bug in it does not fail the soak, it makes the soak
report numbers that mean nothing. A `Wire` that never finds the end of a
response counts no answers and the run fails loudly, which is fine. A `Wire`
that finds the end too early counts two answers where there was one, and the
run passes while measuring a server that was falling apart. So the framing gets
its own checks, on responses assembled by hand, including the two that are easy
to get wrong: a response split across reads, and two responses arriving in one.

The second is that the whole thing runs at all. A short soak here catches the
kind of breakage that stops the run before it starts, which is worth finding in
a pull request rather than at two in the morning when the nightly fires. It is
a few seconds of five kinds of client against a real server, and every gate the
long run checks is checked here too, on numbers too small to prove anything
about an hour. That is the division: this says the soak works, the nightly says
the server does.
"""

from std.sys.info import CompilationTarget

from harness import Suite

from molla.net.latency import LatencyLog, bucket_of
from molla.net.soak_http import (
    ROLE_ABRUPT,
    ROLE_KEEPALIVE,
    ROLE_OVERSIZED,
    ROLE_SLOW,
    ROLE_STREAM,
    Wire,
    current_rss_kb,
    run_http_soak,
)

comptime SOAK_CONNECTIONS = 16
comptime SOAK_SECONDS = 3
"""Enough for all five kinds of client to have done their thing several times
over, and short enough that nobody minds it being in the suite."""


def _feed(mut wire: Wire, text: StringSpan):
    """Hand a `Wire` some bytes the way a socket read would."""
    var p = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(text.unsafe_ptr())
    )
    wire.feed(p, text.byte_length())


def _check_wire(mut suite: Suite):
    suite.group("net.soak_http framing")

    var length_response = String(
        "HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nok\n"
    )
    var chunked_response = String(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding:"
        " chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n"
    )
    var empty_chunked = String(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
    )
    var oversized = String(
        "HTTP/1.1 413 Payload Too Large\r\nContent-Length: 18\r\nConnection:"
        " close\r\n\r\nPayload Too Large\n"
    )

    var wire = Wire()
    suite.check(wire.next_status() == -1, "an empty wire has nothing to report")

    _feed(wire, length_response)
    suite.check(wire.next_status() == 200, "a Content-Length response frames")
    suite.check(wire.next_status() == -1, "and there is nothing behind it")
    suite.check(wire.pending() == 0, "with no bytes left over")

    _feed(wire, chunked_response)
    suite.check(wire.next_status() == 200, "a chunked response frames")
    suite.check(wire.pending() == 0, "to the last byte of the terminator")

    _feed(wire, empty_chunked)
    suite.check(
        wire.next_status() == 200,
        (
            "and so does one whose body was empty, where the terminator is the"
            " whole body"
        ),
    )

    _feed(wire, oversized)
    suite.check(wire.next_status() == 413, "the status is read, not assumed")

    # Split across reads, which is what a slow reader sees on every response.
    var split = Wire()
    _feed(split, "HTTP/1.1 200 OK\r\nContent-Len")
    suite.check(
        split.next_status() == -1, "half a header block is not a response"
    )
    _feed(split, "gth: 3\r\n\r\no")
    suite.check(
        split.next_status() == -1,
        "nor is a complete header block with a body still arriving",
    )
    _feed(split, "k\n")
    suite.check(
        split.next_status() == 200, "and the last byte of the body completes it"
    )

    # Two answers in one read, which is what a keep alive client sees the
    # moment the server gets ahead of it.
    var pipelined = Wire()
    _feed(pipelined, length_response + chunked_response + length_response)
    suite.check(
        pipelined.next_status() == 200
        and pipelined.next_status() == 200
        and pipelined.next_status() == 200,
        "three answers in one read come back as three",
    )
    suite.check(
        pipelined.next_status() == -1 and pipelined.pending() == 0,
        "and then the wire is empty again",
    )

    var junk = Wire()
    _feed(junk, "this is not a response at all, not even close\r\n\r\n")
    suite.check(
        junk.next_status() == -2,
        "bytes that are not a response are a failure rather than a wait",
    )

    var no_framing = Wire()
    _feed(no_framing, "HTTP/1.1 200 OK\r\nServer: molla\r\n\r\nbody")
    suite.check(
        no_framing.next_status() == -2,
        (
            "and so is a response with no Content-Length and no chunking, since"
            " molla never sends one"
        ),
    )

    suite.check(
        ROLE_KEEPALIVE != ROLE_STREAM
        and ROLE_STREAM != ROLE_SLOW
        and ROLE_SLOW != ROLE_ABRUPT
        and ROLE_ABRUPT != ROLE_OVERSIZED,
        "the five client kinds are five different numbers",
    )


def _check_latency(mut suite: Suite):
    suite.group("net.latency")

    suite.check(bucket_of(0) == 0, "a zero duration lands in the first bucket")
    suite.check(bucket_of(999) == 0, "and so does anything under a microsecond")
    suite.check(bucket_of(1000) == 0, "one microsecond is still bucket zero")
    suite.check(bucket_of(3000) == 2, "three microseconds rounds up to four")
    suite.check(
        bucket_of(1_000_000_000) == 20,
        "and a whole second is a bucket a long way up",
    )

    var log = LatencyLog(4)
    suite.check(log.count(0) == 0, "a new log has no samples")
    suite.check(
        log.quantile_us(0, 99) == 0,
        "and a segment with no samples reports zero rather than guessing",
    )

    for _ in range(98):
        log.record(0, 10_000)
    log.record(0, 10_000_000)
    log.record(0, 10_000_000)
    suite.check(log.count(0) == 100, "every sample lands")
    suite.check(
        log.quantile_us(0, 50) == 16,
        "the median is the upper edge of the bucket it fell in",
    )
    suite.check(
        log.quantile_us(0, 99) > log.quantile_us(0, 50),
        "and two slow samples in a hundred move the p99 and not the p50",
    )
    suite.check(
        log.mean_us(0) > 100 and log.mean_us(0) < 1000,
        (
            "the mean is dragged up by the two slow ones and not all the way to"
            " them"
        ),
    )

    log.record(-1, 5000)
    log.record(4, 5000)
    suite.check(
        log.count(0) == 100,
        (
            "a sample for a segment that does not exist is dropped rather than"
            " landing on a neighbour"
        ),
    )


def _check_rss(mut suite: Suite):
    suite.group("net.soak_http rss")

    var rss = current_rss_kb()
    comptime if CompilationTarget.is_linux():
        suite.check(
            rss > 0,
            "Linux publishes the current resident size and the soak reads it",
        )
        suite.check(
            rss < 64 * 1024 * 1024,
            (
                "and the number is a plausible size for this process rather"
                " than a misread field"
            ),
        )
    else:
        suite.check(
            rss == -1,
            (
                "there is no current resident size to read here, and the soak"
                " says so instead of inventing one"
            ),
        )


def _check_short_soak(mut suite: Suite) raises:
    suite.group("net.soak_http")

    var rc = run_http_soak(SOAK_CONNECTIONS, SOAK_SECONDS)
    suite.check(
        rc == 0,
        (
            "a few seconds of keep alive, streaming, slow reading, abrupt"
            " disconnects and oversized bodies all at once leaves nothing"
            " leaked and nothing queued"
        ),
    )


def run(mut suite: Suite) raises:
    _check_wire(suite)
    _check_latency(suite)
    _check_rss(suite)
    _check_short_soak(suite)
