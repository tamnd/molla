"""The acceptance test for issue #13, as a command rather than a claim.

The issue asks for two numbers. A 100 kB chat body parsed in one pass with no
allocations outside the arena, and at least 1 GB/s on the M4. Both are things
you can only argue about with a machine in front of you, so this builds a body
that looks like the traffic, parses it in a loop, and prints what it measured.

The body is generated rather than checked in, because a fixture that is one long
message is a benchmark of `memcpy` and a fixture that is a thousand tiny numbers
is a benchmark of the number parser. This one has the shape of a real request:
a handful of settings at the top, a list of messages with prose in them, a tool
schema with nesting, and a few floats. The shape is what decides where the time
goes, and a shape that is not the traffic measures the wrong thing.

Allocations are counted, not estimated. The reader is built once outside the
loop, exactly as a connection would hold one, and the counter is read before and
after the timed part. Anything above zero means a request allocated, which is
what issue #17 will assert across the whole path.
"""

from molla.io.buffer import Buffer
from molla.json.dom import Document, parse
from molla.json.reader import (
    EV_END,
    EV_ERROR,
    EV_KEY,
    EV_NUMBER,
    EV_STRING,
    Reader,
    error_text,
)
from molla.json.serialize import Writer
from molla.sys.clock import monotonic_ms
from molla.sys.mem import AllocCounter, keep

comptime _WORDS = StaticString(
    "the quick brown fox jumps over a lazy dog while the server holds one"
    " connection open and streams tokens back to a client that reads them as"
    " fast as it can"
)


def _build_body(mut w: Writer, target: Int) -> Bool:
    """A chat request of roughly `target` bytes, with the shape of a real one.

    Built with the writer in this package, which means the benchmark also
    exercises the encoder and means the input is known to be well formed
    without a fixture file to keep in step with the parser.
    """
    if not w.begin_object():
        return False
    if not w.field_str("model", "molla/llama-3.1-8b-instruct"):
        return False
    if not w.field_double("temperature", 0.7):
        return False
    if not w.field_double("top_p", 0.95):
        return False
    if not w.field_int("max_tokens", 4096):
        return False
    if not w.field_bool("stream", True):
        return False

    if not w.key("messages"):
        return False
    if not w.begin_array():
        return False
    if not w.begin_object():
        return False
    if not w.field_str("role", "system"):
        return False
    if not w.field_str("content", "You are a careful assistant."):
        return False
    if not w.end_object():
        return False

    var turn = 0
    while w.length() < target:
        if not w.begin_object():
            return False
        if not w.field_str("role", "user" if turn % 2 == 0 else "assistant"):
            return False
        if not w.key("content"):
            return False
        # A message with a quote and a newline in it, so the escape path is on
        # the measured route rather than only in the tests.
        if not w.string(
            'Turn text: "'
            + String(_WORDS)
            + '"\nand a second line to make the decoder earn its keep.'
        ):
            return False
        if not w.end_object():
            return False
        turn += 1
    if not w.end_array():
        return False

    if not w.key("tools"):
        return False
    if not w.begin_array():
        return False
    if not w.begin_object():
        return False
    if not w.field_str("type", "function"):
        return False
    if not w.key("function"):
        return False
    if not w.begin_object():
        return False
    if not w.field_str("name", "get_weather"):
        return False
    if not w.field_str("description", "Look up the weather for a city."):
        return False
    if not w.key("parameters"):
        return False
    if not w.begin_object():
        return False
    if not w.field_str("type", "object"):
        return False
    if not w.key("properties"):
        return False
    if not w.begin_object():
        return False
    if not w.key("city"):
        return False
    if not w.begin_object():
        return False
    if not w.field_str("type", "string"):
        return False
    if not w.end_object():
        return False
    if not w.key("unit"):
        return False
    if not w.begin_object():
        return False
    if not w.field_str("type", "string"):
        return False
    if not w.key("enum"):
        return False
    if not w.begin_array():
        return False
    if not w.string("celsius"):
        return False
    if not w.string("fahrenheit"):
        return False
    if not w.end_array():
        return False
    if not w.end_object():
        return False
    if not w.end_object():
        return False
    if not w.key("required"):
        return False
    if not w.begin_array():
        return False
    if not w.string("city"):
        return False
    if not w.end_array():
        return False
    if not w.end_object():
        return False
    if not w.end_object():
        return False
    if not w.end_object():
        return False
    if not w.end_array():
        return False
    return w.end_object()


def run_json_bench(size: Int, rounds: Int) -> Int:
    var counter = AllocCounter()
    var code = _run(counter, size, rounds)
    counter.close()
    return code


def _run(mut counter: AllocCounter, size: Int, rounds: Int) -> Int:
    var writer = Writer(counter.raw(), size + 4096)
    if not _build_body(writer, size) or not writer.complete():
        print("molla jsonbench: could not build the body")
        return 1
    var body = writer.bytes()
    print("body      ", len(body), "bytes")

    # One reader for the whole run, which is what a connection would hold. A
    # reader built per request would allocate per request and the number below
    # would be a measurement of that instead.
    var reader = Reader(counter.raw(), 4096)

    # A warmup, so the scratch buffer has already grown to whatever the longest
    # decoded string needs. Counting the growth of a buffer that is reused for
    # the life of a connection as a per request allocation is the mistake that
    # makes a zero allocation claim either false or meaningless.
    for _ in range(4):
        reader.begin(body)
        if not reader.finish():
            print(
                "molla jsonbench: parse failed,",
                error_text(reader.error),
                "at",
                reader.error_at,
            )
            return 1

    var before_live = counter.live()
    var before_total = counter.total()
    var start = monotonic_ms()
    var events = 0
    for _ in range(rounds):
        reader.begin(body)
        while True:
            var e = reader.next()
            if e == EV_END:
                break
            if e == EV_ERROR:
                print("molla jsonbench: parse failed mid run")
                return 1
            events += 1
    var elapsed = monotonic_ms() - start
    var allocations = counter.total() - before_total
    var leaked = counter.live() - before_live

    var bytes_done = len(body) * rounds
    print("rounds    ", rounds)
    print("events    ", events // rounds, "per parse")
    print("strings   ", reader.strings, "of which", reader.decoded, "decoded")
    print("elapsed   ", elapsed, "ms")
    if elapsed > 0:
        # Bytes per millisecond is bytes per second over a thousand, and MB/s is
        # that over another thousand, so the whole thing is one divide.
        print("throughput", bytes_done // elapsed // 1000, "MB/s")
    print("allocations", allocations, "during the timed part")
    print("leaked    ", leaked)

    # The DOM path, measured separately because it is a different promise. It
    # allocates by design, and the number worth knowing is how much.
    var doc = Document(counter.raw(), 4096)
    var dom_before = counter.total()
    var dom_start = monotonic_ms()
    for _ in range(rounds):
        if not parse(doc, reader, body):
            print("molla jsonbench: dom parse failed")
            return 1
    var dom_elapsed = monotonic_ms() - dom_start
    print()
    print("dom nodes ", doc.count())
    print("dom elapsed", dom_elapsed, "ms")
    if dom_elapsed > 0:
        print("dom throughput", bytes_done // dom_elapsed // 1000, "MB/s")
    print("dom allocations", counter.total() - dom_before)

    # The reader and the document hold the body as an address, so nothing above
    # counts as a use of the writer that owns those bytes. Without this the
    # compiler frees it before the first parse.
    keep(writer)

    if allocations != 0:
        print()
        print("FAIL: the streaming parse allocated")
        return 1
    print()
    print("OK: the streaming parse allocated nothing")
    return 0
