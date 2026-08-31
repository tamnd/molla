"""Signals, turned into a shutdown that the rest of the code can see coming.

A signal handler is the worst place in a process to do anything. It runs on
whichever thread the kernel picked, between any two instructions, and the list
of functions it may legally call is about thirty entries long. Nothing in molla
should be written under those rules, so nothing is: the handler in
`molla.sys.signal` writes one byte to a descriptor and returns, and everything
that matters happens here, on an ordinary thread, in ordinary code.

The descriptor is what a poller waits on, which makes a signal the same kind of
event as a socket becoming readable. That is what issue #15 asked for when it
said signalfd or EVFILT_SIGNAL, and it is worth saying why neither of those two
is what molla uses.

signalfd needs the signal blocked in every thread of the process, or the
default action fires on whichever thread has not blocked it and the process
dies before the descriptor is ever read. Mojo 1.0 starts runtime threads before
`main` and gives no way to set their masks, so the precondition cannot be met.
EVFILT_SIGNAL on macOS does work without a mask, but it is macOS only, and a
shutdown path that is a different mechanism on each platform is a shutdown path
that is only ever tested on one of them. The self pipe is the version that is
the same on both, and the property that made signalfd attractive, a signal
arriving as a readable descriptor in a poller, is the property it has.

The supervisor thread is the one that called `serve`. It is not a reactor: it
owns a poller with exactly one descriptor in it and spends its life asleep,
which costs a thread and buys a shutdown that cannot be confused by which
thread the signal landed on.

Arming is separate from waiting, and the order matters more than it looks.
A signal that arrives before the handler is installed does what the default
disposition says, which for SIGTERM is to end the process immediately, and a
process that is still opening its listeners is exactly when a supervisor is
most likely to change its mind. So `arm` goes before `Server.start`, and the
byte for a signal that lands in between waits on the descriptor until the wait
begins. The first version of this armed inside the wait, which passed by hand
and died with exit 143 the first time anything sent the signal quickly.
"""

from molla.net.reactor import Protocol
from molla.net.server import DrainReport, Server
from molla.sys.poll import Poller
from molla.sys.signal import (
    SIGINT,
    SIGQUIT,
    SIGTERM,
    SignalChannel,
    catch_signal,
    open_signal_channel,
    restore_default,
    signal_name,
)
from molla.sys.thread import set_thread_name

comptime SUPERVISOR_TICK_MS = 200
"""How long the supervisor waits in the poller between looks around. The signal
wakes it immediately, so this only bounds how long a caller that stops the
server some other way waits before this notices."""

comptime SUPERVISOR_EVENTS = 8


struct ShutdownSignal(Copyable, ImplicitlyCopyable, Movable):
    """Which signal ended the wait, and what the drain that followed did."""

    var signum: Int
    var report: DrainReport

    def __init__(out self, signum: Int, report: DrainReport):
        self.signum = signum
        self.report = report

    def describe(self) -> String:
        return signal_name(self.signum) + ", " + self.report.describe()


struct SignalWatcher(Movable):
    """The descriptor the signals arrive on, and the poller that waits on it."""

    var channel: SignalChannel
    var poller: Poller
    var armed: Bool

    def __init__(out self) raises:
        self.channel = SignalChannel()
        self.poller = Poller(SUPERVISOR_EVENTS)
        self.armed = False

    def arm(mut self) raises:
        """Catch the three signals and start collecting them.

        Call this before the server starts. Everything after this point is
        ordinary code on an ordinary thread."""
        if self.armed:
            return
        var opened = open_signal_channel(self.channel)
        if not opened.is_ok():
            raise Error(opened.describe("could not open the signal channel"))
        _ = catch_signal(SIGINT)
        _ = catch_signal(SIGTERM)
        _ = catch_signal(SIGQUIT)
        self.poller.add_read(self.channel.read_fd)
        self.armed = True

    def wait[P: Protocol](mut self, mut server: Server[P]) raises -> Int:
        """Sleep until a signal says to stop, and return which one it was.

        SIGQUIT does not stop anything. It prints what every worker is doing
        and the wait carries on, which is what you want when a server is not
        shutting down and nobody can say which thread is holding it.
        """
        if not self.armed:
            raise Error("the signal watcher was not armed before the wait")
        var caught = 0
        while caught == 0:
            var count = self.poller.wait(SUPERVISOR_TICK_MS)
            for i in range(count):
                var ready = self.poller.event(i)
                if ready.fd != self.channel.read_fd:
                    continue
                # Drained rather than read once. Two signals in quick
                # succession are two bytes, and the second would otherwise sit
                # there keeping the descriptor readable forever.
                while True:
                    var got = self.channel.take()
                    if not got.is_ok():
                        break
                    if got.value == SIGQUIT:
                        print(server.dump(), end="")
                        continue
                    caught = got.value
        return caught

    def close(mut self):
        """Put the signals back and give the descriptors up.

        SIGQUIT goes back to its default too, which `SignalChannel.close` does
        not do because it does not know molla asked for it. A handler left
        installed after its descriptor is closed writes a byte into whatever
        opens that number next."""
        if not self.armed:
            return
        _ = restore_default(SIGQUIT)
        self.channel.close()
        self.poller.shutdown()
        self.armed = False


def serve_until_signal[
    P: Protocol
](
    mut server: Server[P], mut watcher: SignalWatcher, deadline_ms: Int = 0
) raises -> ShutdownSignal:
    """Wait for SIGINT or SIGTERM, then drain and say what happened.

    The watcher has to be armed already, and the server has to be started.
    """
    _ = set_thread_name("molla-super")
    var caught = watcher.wait(server)
    print("molla: caught", signal_name(caught) + ", draining")
    watcher.close()
    var report = server.drain(deadline_ms)
    return ShutdownSignal(caught, report)
