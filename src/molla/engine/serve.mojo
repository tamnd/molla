"""`molla serve`, which is the first command that is a server rather than a
spike.

Everything under it already existed. The reactor has been accepting connections
since M0, the protocol has been framing requests since M1, and the model has
been generating tokens since #28. What this file does is put a loaded model
where the protocol can reach it and then get out of the way, which is about
forty lines of real work and a lot of printing.

## One worker

The context is built with one worker on purpose. There is one model, one
session and one sampler behind these routes, so a second worker would be a
second thread contending for a lock that does not exist. When the scheduler
lands in M3 this becomes a real number and the runner becomes a list, and until
then the honest thing is to run one thread and refuse a second request rather
than to run four and corrupt a cache. See the docstring on
`molla.engine.runner`.

The cost is visible and worth saying out loud: a request that is not streaming
holds the server for its whole generation, so a health check behind a long
completion waits for it. A streaming request hands the reactor back after every
token, so the admin routes stay answerable through one at the cost of about one
token of delay. See `Connection.yield_now` for what makes that a token rather
than eight of them.

## The model is a local

The runner is a local of this function and the protocol holds its address. That
is the same arrangement the logger and the metrics view are in, and it works for
the same reason: the server is started and stopped inside this function, so
nothing holding that address outlives the thing it points at. `keep` at the end
is what stops Mojo from deciding otherwise.
"""

from molla.engine.runner import Runner, address_of
from molla.http.protocol import HttpProtocol
from molla.net.context import ServerContext
from molla.net.listener import ListenAddress
from molla.net.server import Server
from molla.net.supervisor import SignalWatcher, serve_until_signal
from molla.ops.config import LEVEL_INFO
from molla.ops.log import LogPump, LogSink
from molla.ops.metrics import Metrics
from molla.sys.clock import monotonic_ms
from molla.sys.mem import keep
from molla.sys.signal import ignore_sigpipe

comptime IDLE_MS = 300000
"""How long a connection may say nothing before it is closed.

Five minutes, which is far longer than the sixty seconds the soak uses. A chat
client opens a connection, waits for somebody to type, and sends. Timing that
out is a reconnect the user sees as a stall."""

comptime DRAIN_MS = 10000
"""How long a shutdown waits for what it already owes. Longer than the drain
test's five seconds because one of the things it may owe is the rest of a
completion."""

comptime LOG_RING = 262144
comptime SEND_BUFFER = 0
"""Zero means leave the socket's send buffer alone. The drain test shrinks it
deliberately to make a shutdown hard; a real server wants the kernel's own
number, which on a token at a time stream is never the constraint."""


def run_serve(
    model_path: String,
    tokenizer_path: String,
    host: String,
    port: UInt16,
    context: Int,
) raises -> Int:
    """Load a model and answer OpenAI requests against it until a signal."""
    _ = ignore_sigpipe()

    var started = monotonic_ms()
    print("loading", model_path)
    var runner = Runner(model_path, tokenizer_path, model_path, context)
    print("  model         ", runner.describe())
    print("  tokenizer     ", tokenizer_path)
    print("  repack        ", runner.repack())
    print(
        "  chat template ",
        "yes" if runner.has_chat else "no, /v1/chat/completions is off",
    )
    print("  loaded in     ", monotonic_ms() - started, "ms")

    # Armed before the listener opens, so a signal arriving during startup is
    # still a clean shutdown rather than a default kill.
    var watcher = SignalWatcher()
    watcher.arm()

    var settings = ServerContext(1, IDLE_MS, 0, DRAIN_MS, SEND_BUFFER)
    var address = ListenAddress(port) if host == "127.0.0.1" else ListenAddress(
        _host_of(host), port
    )
    var server = Server[HttpProtocol](address, settings)

    var sink = LogSink(server.workers, LOG_RING, LEVEL_INFO)
    var metrics = Metrics(server.workers)
    var engine = address_of(runner)
    for i in range(len(server.reactors)):
        server.reactors[i].proto.configure_ops(
            sink.logger(i), metrics.meter(i), metrics.view(), True
        )
        server.reactors[i].proto.configure_engine(engine)
    var pump = LogPump(Int(Pointer(to=sink)))
    pump.start()

    print()
    print("molla serving on http://" + host + ":" + String(server.port))
    print("  POST /v1/chat/completions")
    print("  POST /v1/completions")
    print("  GET  /v1/models")
    print("  GET  /molla/health, /molla/version, /molla/metrics")
    print()
    print("one sequence at a time. a second request in flight gets a 503.")
    print("ctrl-c to stop.")

    server.start()
    var outcome = serve_until_signal(server, watcher, DRAIN_MS)
    pump.stop()

    print()
    print("shutdown:", outcome.describe())
    print("  requests      ", runner.seq)
    print("  logs dropped  ", sink.dropped())
    runner.close()
    # The reactors held this by address for the life of the server, and Mojo
    # would otherwise be free to drop it at its last mention above.
    keep(runner)
    return 0 if outcome.report.clean else 1


def _host_of(host: String) raises -> UInt32:
    """A dotted quad as the number the socket layer wants.

    Four numbers and three dots, and nothing else. There is no resolver here and
    there is not going to be one on this path: a listen address is a local
    interface, and a name for a local interface is a way to bind the wrong one.
    """
    var parts = host.split(".")
    if len(parts) != 4:
        raise Error("'" + host + "' is not an address like 127.0.0.1")
    var out: UInt32 = 0
    for i in range(4):
        var part = atol(String(parts[i]))
        if part < 0 or part > 255:
            raise Error("'" + host + "' is not an address like 127.0.0.1")
        out = (out << 8) | UInt32(part)
    return out
