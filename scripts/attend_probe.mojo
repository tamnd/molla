"""What decode attention costs, at the shape an 8B decode gives it.

Issue #234. [docs/validation/budget.md](../docs/validation/budget.md) split an
8B token three ways by differencing two context lengths, and the middle term,
attention over the keys and values, came to 4.94 ms of a 12.97 ms token at 1121
tokens of context. That was arrived at by subtraction. This measures it.

`device_attend` at the 8B's decode geometry, which is 32 query heads over 8 key
heads of 128, one token, swept over context. Multiply a row by 32 layers to get
what a token pays. The bytes column is the keys and the values one layer holds
at that context, which is what the kernel has to read, and the rate is that over
the time.

Three tables. The first is the single kernel over the whole context, which is
what this measured before the split landed and is kept so the two can be
compared. The second is the grid depth sweep that said the grid was the problem.
The third is the split against the single kernel at the same contexts, which is
the answer to issue #234.

The keys and values are left as allocated. What is in them does not change what
the kernel reads, and a buffer of zeros scores zero, exponentiates to one and
sums to the key count, which is a normal float throughout with no denormal stall
hiding in it. The numbers are thrown away.

Run it on a machine that is not doing anything else.
"""

from std.sys.info import has_accelerator
from std.time import monotonic

from max.gpu.host import DeviceContext

from molla.nn.attention import AttnSpec
from molla.nn.gpu import DeviceVec
from molla.nn.gpu_ops import attend_partials, device_attend

comptime REPS = 20
"""Launches in one timed run, averaged. More than the projection probe uses,
because at a short context this kernel is a few microseconds and the timer is
not."""

comptime LAYERS = 32
"""What one row has to be multiplied by to be a token's worth. The 8B's block
count, and it is here so the last column does not have to be worked out by
hand."""


def _time(
    ctx: DeviceContext,
    spec: AttnSpec,
    q: DeviceVec,
    keys: DeviceVec,
    values: DeviceVec,
    mut out: DeviceVec,
    mut scores: DeviceVec,
    mut partials: DeviceVec,
    count: Int,
    tokens: Int = 1,
) raises -> Float64:
    """Nanoseconds one call takes, best of five runs of `REPS`.

    Best rather than mean, for the reason every timing loop here takes the best:
    a slow run was sharing the card and a fast one was not, and only one of
    those is a property of the kernel.
    """
    var best = Float64(0)
    for rep in range(5):
        var began = monotonic()
        for _ in range(REPS):
            device_attend(
                ctx,
                spec,
                q,
                keys,
                values,
                count,
                count - 1,
                out,
                scores,
                partials,
                tokens,
            )
        ctx.synchronize()
        var took = Float64(monotonic() - began) / Float64(REPS)
        if rep == 0 or took < best:
            best = took
    return best


def main() raises:
    comptime if not has_accelerator():
        print("attend_probe: no accelerator on this machine")
        return
    else:
        var ctx = DeviceContext()
        print("device:", ctx.name())
        print("")

        # Llama 3.1 8B: 32 query heads, 8 key heads, 128 per head.
        var spec = AttnSpec(32, 8, 128)
        var width = spec.heads * spec.head_dim
        var kv_width = spec.kv_heads * spec.head_dim

        var contexts = List[Int]()
        contexts.append(64)
        contexts.append(256)
        contexts.append(512)
        contexts.append(1185)
        contexts.append(2048)

        var most = contexts[len(contexts) - 1]
        var q = DeviceVec(ctx, width)
        var out = DeviceVec(ctx, width)
        var keys = DeviceVec(ctx, most * kv_width)
        var values = DeviceVec(ctx, most * kv_width)
        var scores = DeviceVec(ctx, spec.heads * most)
        # One float, which is less room than one slice needs, so `device_attend`
        # cuts the keys into one piece and launches the single kernel. This is
        # how the first two tables here measure what the kernel used to do now
        # that the shipped path does something else.
        var unsplit = DeviceVec(ctx, 1)
        var split = DeviceVec(ctx, attend_partials(spec, 1, most))
        ctx.synchronize()

        print("one kernel over the whole context, which is what it used to do")
        print("context   a layer    kv bytes   rate       32 layers")
        for i in range(len(contexts)):
            var count = contexts[i]
            var ns = _time(
                ctx, spec, q, keys, values, out, scores, unsplit, count
            )
            # Keys and values, float32, over the whole context. What one query
            # head re-reads because four of them share a key head is not in
            # this, because it is not what leaves the memory.
            var kv = Float64(2 * count * kv_width * 4)
            var token = ns * Float64(LAYERS)
            print(
                String(count)
                + "\t  "
                + String(Int(ns / 1000.0))
                + " us\t   "
                + String(Int(kv / 1048576.0))
                + " MiB\t      "
                + String(Int(kv / ns))
                + " GB/s\t  "
                + String(Int(token / 1000000.0))
                + "."
                + String(Int(token / 10000.0) % 100)
                + " ms"
            )

        print("")
        print(
            "The last column is what a token spends here. budget.md gets 4.94"
            " ms at 1185 by"
        )
        print("subtracting two decodes, and this arrives at it directly.")
        print("")

        # The same keys and values, read by a taller grid. A decode launches
        # `heads` blocks and a prefill chunk launches `heads * tokens`, and
        # every one of those blocks reads the same keys, so if the kernel were
        # bound by what it reads then eight times the blocks over the same bytes
        # would cost eight times as much. Whatever it does instead is how much
        # of the decode number is the grid being 32 blocks wide.
        print("the same 1185 keys, read by a taller grid")
        var count = 1185
        var deep_q = DeviceVec(ctx, 16 * width)
        var deep_out = DeviceVec(ctx, 16 * width)
        var deep_scores = DeviceVec(ctx, 16 * spec.heads * (count + 16))
        ctx.synchronize()
        var one = Float64(0)
        var depths = List[Int]()
        depths.append(1)
        depths.append(2)
        depths.append(4)
        depths.append(8)
        depths.append(16)
        for i in range(len(depths)):
            var deep = depths[i]
            var ns = _time(
                ctx,
                spec,
                deep_q,
                keys,
                values,
                deep_out,
                deep_scores,
                unsplit,
                count,
                deep,
            )
            if i == 0:
                one = ns
            print(
                String(spec.heads * deep)
                + " blocks\t  "
                + String(Int(ns / 1000.0))
                + " us\t   "
                + String(Int(ns / one))
                + "."
                + String(Int(ns * 10.0 / one) % 10)
                + " times one token's cost for "
                + String(deep)
                + " of them"
            )

        print("")
        print("and what the split does with that, one token over the same keys")
        print("context   one kernel   split      speedup    32 layers")
        for i in range(len(contexts)):
            var ctx_len = contexts[i]
            var before = _time(
                ctx, spec, q, keys, values, out, scores, unsplit, ctx_len
            )
            var after = _time(
                ctx, spec, q, keys, values, out, scores, split, ctx_len
            )
            var token = after * Float64(LAYERS)
            print(
                String(ctx_len)
                + "\t  "
                + String(Int(before / 1000.0))
                + " us\t       "
                + String(Int(after / 1000.0))
                + " us\t      "
                + String(Int(before / after))
                + "."
                + String(Int(before * 10.0 / after) % 10)
                + "x\t      "
                + String(Int(token / 1000000.0))
                + "."
                + String(Int(token / 10000.0) % 100)
                + " ms"
            )
