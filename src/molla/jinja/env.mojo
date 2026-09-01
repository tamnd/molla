"""The render state, and the four limits that stop a template running away.

A chat template is untrusted input. It arrives inside a model repository that
anybody can publish, it is Turing complete enough to loop forever, and it runs
on the request path. So a render has a budget in four dimensions and every one
of them is checked in the same place.

A step budget, counted on every statement and every expression node, which is
what stops `{% for i in range(10000000) %}`. An output cap, which is what stops
a template that produces a byte per step from producing a gigabyte. A recursion
limit, which is what stops a macro that calls itself. And a wall clock deadline,
which is the backstop for anything the other three fail to catch, because the
other three are counts and a count says nothing about how long a count takes.

The clock is only read every 1024 steps. Reading it on every step showed up in
the profile ahead of the work being timed, and 1024 steps is a few microseconds,
which is finer than any deadline worth setting.

## Scopes

Frames form a tree rather than a stack, and a macro remembers the frame it was
defined in, which is what makes it a closure. That matters because a macro
defined at the top of a template and called from inside a loop has to see the
names that were visible where it was written, not the ones visible where it was
called.

A `{% set %}` writes into the innermost frame. Inside a loop that frame is
rebuilt every iteration, so an assignment made in one iteration is gone by the
next, which is the behaviour that surprises everybody writing a counter in a
loop and is exactly what the reference does. `namespace()` exists precisely
because of it, and a namespace is a dict whose attributes can be assigned, so it
survives by being a value rather than a binding.
"""

from molla.io.buffer import Buffer
from molla.jinja.diag import fail
from molla.jinja.value import Heap
from molla.sys.clock import monotonic_ns
from molla.text.props import Unicode

comptime DEFAULT_STEPS = 4000000
comptime DEFAULT_OUTPUT = 16777216
comptime DEFAULT_DEPTH = 64
comptime DEFAULT_DEADLINE_MS = 2000

comptime CLOCK_EVERY = 1024
"""Steps between two reads of the clock."""

comptime NO_FRAME = -1


struct Limits(Copyable, ImplicitlyCopyable, Movable):
    var steps: Int
    var output: Int
    var depth: Int
    var deadline_ms: Int

    def __init__(out self):
        self.steps = DEFAULT_STEPS
        self.output = DEFAULT_OUTPUT
        self.depth = DEFAULT_DEPTH
        self.deadline_ms = DEFAULT_DEADLINE_MS


struct Frame(Copyable, Movable):
    """One scope. Names and the values they are bound to, in one pair of lists.

    Linear lookup, because a frame holds a loop variable, the `loop` object and
    whatever the body assigned, which is between one and a handful of names, and
    a map over three names costs more to build than it saves to search.
    """

    var names: List[String]
    var values: List[Int]
    var parent: Int
    var captured: Bool
    """Set when a macro was defined here, which means the frame has to outlive
    the statement that made it and cannot be reused by the next iteration."""

    def __init__(out self, parent: Int):
        self.names = List[String]()
        self.values = List[Int]()
        self.parent = parent
        self.captured = False

    def clear(mut self, parent: Int):
        self.names.clear()
        self.values.clear()
        self.parent = parent
        self.captured = False


struct Env(Movable):
    var src: String
    var heap: Heap
    var frames: List[Frame]
    var out: Buffer
    var limits: Limits
    var steps: Int
    var depth: Int
    var started_ms: Int
    var now: Int
    """The wall clock `strftime_now` reads, or 0 for the real one.

    A template that stamps the date into the system prompt renders a different
    string tomorrow, which is correct in production and useless in a corpus.
    Pinning this is how the conformance run compares two implementations rather
    than two days.
    """
    var uni: List[Unicode]
    """The Unicode tables, built on first use and not before.

    Only the case filters need them and only when what they are folding is not
    ASCII, which for a chat template means almost never, so building them at
    startup would be paying for something nothing asked for.
    """

    def __init__(out self, source: String, limits: Limits, counter: Int) raises:
        self.src = source
        self.heap = Heap()
        self.frames = List[Frame]()
        self.out = Buffer(4096, counter)
        if not self.out.is_valid():
            raise Error("could not allocate the render buffer")
        self.limits = limits
        self.steps = 0
        self.depth = 0
        self.started_ms = monotonic_ns() // 1000000
        self.now = 0
        self.uni = List[Unicode]()

    def unicode(mut self) raises -> Pointer[Unicode, origin_of(self.uni[0])]:
        """The tables, built the first time somebody needs them."""
        if len(self.uni) == 0:
            self.uni.append(Unicode())
        return Pointer(to=self.uni[0])

    def step(mut self) raises:
        self.steps += 1
        if self.steps > self.limits.steps:
            raise Error(
                "the template used more than "
                + String(self.limits.steps)
                + " steps, which means it is looping"
            )
        if self.steps % CLOCK_EVERY == 0:
            var elapsed = monotonic_ns() // 1000000 - self.started_ms
            if elapsed > self.limits.deadline_ms:
                raise Error(
                    "the template ran for longer than "
                    + String(self.limits.deadline_ms)
                    + "ms"
                )

    def emit(mut self, text: String) raises:
        if self.out.length + text.byte_length() > self.limits.output:
            raise Error(
                "the template produced more than "
                + String(self.limits.output)
                + " bytes"
            )
        if not self.out.append_str(text):
            raise Error("could not grow the render buffer")

    def fail(self, at: Int, message: String) raises:
        fail(self.src.as_bytes(), at, message)

    def push(mut self, parent: Int) -> Int:
        self.frames.append(Frame(parent))
        return len(self.frames) - 1

    def reuse(mut self, frame: Int, parent: Int) -> Int:
        """The same frame again, cleared, unless something captured it."""
        if self.frames[frame].captured:
            return self.push(parent)
        self.frames[frame].clear(parent)
        return frame

    def bind(mut self, frame: Int, name: String, value: Int):
        for i in range(len(self.frames[frame].names)):
            if self.frames[frame].names[i] == name:
                self.frames[frame].values[i] = value
                return
        self.frames[frame].names.append(name)
        self.frames[frame].values.append(value)

    def lookup(self, frame: Int, name: String) -> Int:
        """The nearest binding, or -1 when there is none anywhere."""
        var at = frame
        while at != NO_FRAME:
            for i in range(len(self.frames[at].names)):
                if self.frames[at].names[i] == name:
                    return self.frames[at].values[i]
            at = self.frames[at].parent
        return -1

    def mark(self) -> Int:
        """Where the output currently ends.

        Paired with `take`, this is how a macro body, a `{% set %}` block and a
        `{% filter %}` block get captured. They render into the same buffer
        everything else does and then the tail is lifted back out, which means
        no second buffer and no copy for the common case where nothing captures.
        """
        return self.out.length

    def take(mut self, mark: Int) -> String:
        """Everything written since `mark`, removed from the output."""
        var data = self.out.bytes()
        var raw = List[UInt8]()
        for i in range(mark, len(data)):
            raw.append(data[i])
        self.out.length = mark
        return String(StringSpan(unsafe_from_utf8=Span(raw)))

    def rendered(self) -> String:
        return String(StringSpan(unsafe_from_utf8=self.out.bytes()))
