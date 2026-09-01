"""A latency histogram cut into segments, for the soaks.

Both soaks ask the same question and the question is not what the latency is.
It is whether the latency at the end of an hour is the latency at the start.
So the run is cut into segments, every round trip lands in the segment that was
current when it finished, and the report compares the last segment against the
first.

The histogram is powers of two microseconds, twenty eight of them, from under a
microsecond to over two minutes. That is coarse, and coarse is what makes it
honest here: quantiles are reported as the upper edge of the bucket they landed
in rather than interpolated, because a bucket this wide cannot support a claim
more precise than the bucket. It is also what makes it free. Recording a sample
is a shift loop and an increment on a fixed array, with no allocation and no
sorting at the end, which matters when the thing doing the recording is the
client half of the test and is competing with the server for the machine.

Extracted from the reactor soak when the HTTP soak needed the same thing. Two
copies of a histogram is two chances for the drift gate to mean something
slightly different in each, and the gate is the entire point of both.
"""

comptime BUCKETS = 28
"""One per power of two microseconds. Bucket 0 is under a microsecond and
bucket 27 is over two minutes, which covers everything between a loopback round
trip and a hang."""


def bucket_of(nanoseconds: Int) -> Int:
    """Which power of two microseconds a duration lands in."""
    var us = nanoseconds // 1000
    if us <= 0:
        return 0
    var index = 0
    var edge = 1
    while edge < us and index < BUCKETS - 1:
        edge = edge << 1
        index += 1
    return index


struct LatencyLog(Movable):
    """Round trip times, bucketed, one histogram per segment of a run."""

    var segments: Int
    var counts: List[Int]
    """Flattened, `segments * BUCKETS`, because a list of lists is a pointer
    chase per sample for no gain when the shape is fixed at construction."""
    var totals: List[Int]
    var sums: List[Int]

    def __init__(out self, segments: Int):
        self.segments = segments if segments > 0 else 1
        self.counts = List[Int](length=self.segments * BUCKETS, fill=0)
        self.totals = List[Int](length=self.segments, fill=0)
        self.sums = List[Int](length=self.segments, fill=0)

    def record(mut self, segment: Int, nanoseconds: Int):
        """Add one sample. Out of range segments are dropped rather than
        clamped, since a caller that computes a segment wrongly should lose the
        sample instead of quietly poisoning a neighbour."""
        if segment < 0 or segment >= self.segments:
            return
        self.counts[segment * BUCKETS + bucket_of(nanoseconds)] += 1
        self.totals[segment] += 1
        self.sums[segment] += nanoseconds

    def count(self, segment: Int) -> Int:
        if segment < 0 or segment >= self.segments:
            return 0
        return self.totals[segment]

    def mean_us(self, segment: Int) -> Int:
        var n = self.count(segment)
        if n == 0:
            return 0
        return self.sums[segment] // n // 1000

    def quantile_us(self, segment: Int, percent: Int) -> Int:
        """Upper edge in microseconds of the bucket holding a quantile.

        Zero when the segment has no samples, which a caller comparing two
        segments has to notice: a segment that recorded nothing is a run that
        stalled, not a run that got infinitely fast.
        """
        var total = self.count(segment)
        if total == 0:
            return 0
        var want = (total * percent) // 100
        if want < 1:
            want = 1
        var base = segment * BUCKETS
        var seen = 0
        for i in range(BUCKETS):
            seen += self.counts[base + i]
            if seen >= want:
                return 1 << i
        return 1 << (BUCKETS - 1)
