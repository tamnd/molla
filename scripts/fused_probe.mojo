"""Where a fused token goes, one record at a time.

`docs/validation/budget.md` took the 8B apart by differencing two context
lengths and then timing the projection kernel on its own. Neither of those
works on the path the small models actually take. A fused token is one launch
a layer and the steps inside it are records in a table, so there is nothing to
difference and no separate kernel to time.

What there is instead is `launch_fused`, which takes a record range. So this
launches a prefix of every layer, `starts[i]` through `starts[i] + k`, for
every `k` from one to a whole layer, and differences the totals. The difference
between `k` and `k - 1` is what record `k - 1` costs, averaged over the layers,
and every launch reads the same number of bytes of plan and pays the same one
launch, so both of those cancel.

Across the layers rather than repeating one, because a layer of SmolLM2 is 3.8
MB and a 4090 has 72 MB of L2. Timing one layer thirty times measures the
cache. Timing thirty different layers once each measures the memory, which is
what a token does.

A truncated layer leaves the records after the cut reading whatever the last
whole token wrote, which is why a real prompt runs first. The values are stale
rather than wrong shaped, and a stale float times the same as a fresh one. The
answers are thrown away either way.

    pixi run -- mojo run -I src scripts/fused_probe.mojo MODEL [CONTEXT]

Run it on a machine that is not doing anything else.
"""

from std.sys import argv, exit
from std.sys.info import has_accelerator
from std.time import monotonic

from max.gpu.host import DeviceContext

from molla.engine.backend import Backend, parse_backend, pick
from molla.engine.bind import bind
from molla.engine.device import device_context, load_on_device
from molla.engine.device import open_session
from molla.model.gguf import Gguf
from molla.model.repack import model_key, open_cache
from molla.nn.gpu_fused import (
    OP_ACT,
    OP_ADD,
    OP_ATTEND,
    OP_MATVEC,
    OP_NORM,
    OP_ROPE,
    R_COLS,
    R_N,
    R_OP,
    R_ROWS,
    R_SYNC,
    REC_INTS,
    launch_fused,
)
from molla.sys.device import devices

comptime REPS = 25
"""Timed passes over the model at each prefix length, best taken.

Best rather than median because everything that makes a pass slower than its
floor is something else on the machine, and the floor is what a token gets on
an idle card.

Twenty five because the answer is a difference of two of these and the records
worth reading are a few microseconds inside a pass of two thousand. At five the
big rows repeat to within a tenth and the small ones move by a microsecond
either way, which is enough to put a sub microsecond record below zero. The
sweep is nine thousand launches at this count and takes a few seconds, so there
is no reason to be stingy.
"""

comptime PROMPT = 512
"""Positions in the cache before anything is timed.

The attention record is the one step whose cost moves with how much context
there is, so a probe at position zero would report it as free and mislead about
every model with a long prompt.
"""


def _name(op: Int) -> String:
    if op == OP_NORM:
        return "norm"
    if op == OP_MATVEC:
        return "matvec"
    if op == OP_ROPE:
        return "rope"
    if op == OP_ATTEND:
        return "attend"
    if op == OP_ACT:
        return "act"
    if op == OP_ADD:
        return "add"
    return "op " + String(op)


def _shape(op: Int, cols: Int, rows: Int, n: Int) -> String:
    if op == OP_MATVEC:
        return String(cols) + " by " + String(rows)
    return String(n)


def _us(ns: Float64) -> String:
    var v = ns / 1000.0
    var whole = Int(v * 100.0 + 0.5)
    return String(Float64(whole) / 100.0)


def main() raises:
    var args = argv()
    if len(args) < 2 or len(args) > 3:
        print("usage: fused_probe <model.gguf> [context]")
        exit(2)
    var path = String(args[1])
    var context = 2048
    if len(args) == 3:
        context = Int(String(args[2]))
    if context <= PROMPT:
        print(
            "the context has to be longer than the "
            + String(PROMPT)
            + " position prompt"
        )
        exit(2)

    comptime if not has_accelerator():
        print("no accelerator in this build, nothing to probe")
        return
    else:
        var all = devices()
        var fits = List[Bool]()
        for _ in range(len(all)):
            fits.append(True)
        var backend = pick(parse_backend("auto"), all, fits)
        if not backend.on_device:
            print("no device backend here, nothing to probe")
            return
        print("backend  " + backend.describe())

        var ctx = device_context(backend.device.index)
        var g = Gguf(path)
        var cache = open_cache(path, model_key(g))
        var weights = load_on_device(g, cache, path, backend.device, ctx)
        var host_b = bind(g, cache)
        var b = bind(g, cache, weights.residency())
        var held = open_session(ctx, host_b, b, context)
        var s = held.take()
        if s.fused.records == 0:
            print(
                "this session did not build a fused plan, so there is nothing"
                " to take apart. Set MOLLA_FUSED=1"
            )
            return

        # A real prompt, so that the cache holds real keys and the work vector
        # holds real intermediates rather than whatever an allocation left.
        var prompt = List[Int]()
        for i in range(PROMPT):
            prompt.append(i % 100 + 1)
        s.prefill(prompt)
        s.fetch()
        s.step(1)
        s.fetch()

        var layers = len(s.fused.starts) - 1
        var per = s.fused.starts[1] - s.fused.starts[0]
        for i in range(layers):
            if s.fused.starts[i + 1] - s.fused.starts[i] != per:
                print(
                    "layer "
                    + String(i)
                    + " has a different number of records than layer zero, so"
                    " a prefix does not mean the same thing in both and this"
                    " probe cannot difference them"
                )
                return

        # The table itself, back on the host, so a row can say what it timed
        # rather than just how long it took.
        var table = List[Int64]()
        for _ in range(s.fused.records * REC_INTS):
            table.append(Int64(0))
        ctx.enqueue_copy(
            Pointer[Int64, MutAnyOrigin](
                unsafe_from_address=Int(table.unsafe_ptr())
            ),
            s.fused.ints,
        )
        ctx.synchronize()

        var pos = s.pos
        var slot = pos % context

        # Prefix zero is not a launch at all, so the first row's cost carries
        # the launch with it and every row after it does not. Measured anyway,
        # because the difference between it and prefix one is the launch and
        # that is worth printing.
        var totals = List[Float64]()
        for k in range(per + 1):
            var best = Float64(0)
            for rep in range(REPS):
                ctx.synchronize()
                var at = monotonic()
                if k > 0:
                    for i in range(layers):
                        launch_fused(
                            ctx,
                            s.fused,
                            s.model.layers[i].arena.vec,
                            s.x,
                            s.cache.keys[i],
                            s.cache.values[i],
                            s.fused.starts[i],
                            s.fused.starts[i] + k,
                            pos,
                            slot,
                            slot + 1,
                        )
                ctx.synchronize()
                var took = Float64(monotonic() - at)
                if rep == 0 or took < best:
                    best = took
            totals.append(best)

        print("")
        print("model    " + path)
        print("layers   " + String(layers))
        print(
            "records  "
            + String(per)
            + " a layer, "
            + String(s.fused.records)
            + " in all"
        )
        print("blocks   " + String(s.fused.blocks))
        print("position " + String(pos))
        print("")
        print(
            "| record | op | shape | sync | a layer | over the model | share |"
        )
        print("| --- | --- | --- | --- | --- | --- | --- |")

        var whole = totals[per] - totals[0]
        if whole <= 0:
            print(
                "the whole layer measured as free, which means the clock is not"
                " usable here"
            )
            return
        for k in range(per):
            var rec = s.fused.starts[0] + k
            var op = Int(table[rec * REC_INTS + R_OP])
            var one = (totals[k + 1] - totals[k]) / Float64(layers)
            var over = one * Float64(layers)
            var share = Int(over / whole * 1000.0 + 0.5)
            print(
                "| "
                + String(k)
                + " | "
                + _name(op)
                + " | "
                + _shape(
                    op,
                    Int(table[rec * REC_INTS + R_COLS]),
                    Int(table[rec * REC_INTS + R_ROWS]),
                    Int(table[rec * REC_INTS + R_N]),
                )
                + " | "
                + ("yes" if Int(table[rec * REC_INTS + R_SYNC]) != 0 else "no")
                + " | "
                + _us(one)
                + " us | "
                + _us(over)
                + " us | "
                + String(Float64(share) / 10.0)
                + " per cent |"
            )
        print("")
        print(
            "a launch  "
            + _us((totals[1] - totals[0]) / Float64(layers))
            + " us, in the first row"
        )
        print("the layers  " + _us(whole) + " us")
        _ = s^
        _ = weights^
