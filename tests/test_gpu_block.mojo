"""The device forward pass against the host one, on the same weights.

Every kernel under this already has a test that says it agrees with a host
reference on its own inputs. That is not the same statement as this one. A
kernel that is right about arithmetic and wrong about which buffer it was handed
passes every test that only looks at one kernel, and so does a layer that norms
before it should, or a key that is rotated at the wrong offset in the cache, or a
trace that records the residual stream one operation late.

So this builds a small model twice out of the same bytes, once in host memory and
once in a device pool, runs several tokens through both, and asks two questions
per token. Do the logits agree to the precision two different reduction orders
can be expected to reach, and does greedy sampling pick the same token. The
second one is the one that matters: the milestone is that `molla generate`
produces the same text on both, and a token is the unit that text is made of.

The model is synthetic and it is not small in shape. Grouped query attention with
four query heads over two key heads, because a group size of one would not
notice a key head index that is computed wrong. Two layers, because one layer
cannot tell a residual add that lands in the wrong place from one that lands in
the right one. A vocabulary wider than the residual stream, so the output head is
not square and a transposed read shows up as an error rather than as a shuffle.
"""

from std.memory import bitcast
from std.sys.info import has_accelerator

from max.gpu.host import DeviceContext

from harness import Suite

from molla.engine.device import DeviceKvCache
from molla.model.load import DevicePool
from molla.model.spec import architecture_id
from molla.nn.arch import arch_of
from molla.nn.attention import AttnSpec
from molla.nn.block import BlockSpec, LayerWeights, Scratch
from molla.nn.gpu import MM_GROUPS, SPAN, DeviceVec
from molla.nn.gpu_block import DeviceModel, DeviceScratch, device_forward
from molla.nn.model import ModelWeights, forward
from molla.nn.quant import Q_F32, Q_Q8_0
from molla.nn.repack import LAYOUT_PLANAR, planar_row_bytes
from molla.nn.rope import RopeSpec
from molla.nn.tensor import WHERE_DEVICE, Buffer, Tensor
from molla.sys.device import default_device
from molla.sys.mem import keep
from molla.sys.mmap import RawPtr

comptime WIDTH = 64
comptime HIDDEN = 128
comptime HEADS = 4
comptime KV_HEADS = 2
comptime HEAD_DIM = 16
comptime LAYERS = 2
comptime VOCAB = 96
comptime CONTEXT = 24

comptime MATRICES = 2 + LAYERS * 7
"""The embedding, the head, and seven per layer, in the order `_shapes` lists
them."""


def run(mut suite: Suite) raises:
    """Nothing here runs without a device, so this is the skip.

    The launches are in `run_on_device`, which `main` calls with the one context
    the process owns. A CUDA process gets one `DeviceContext` and hangs on the
    first allocation against a second, so no test module may make its own.
    """
    comptime if not has_accelerator():
        suite.group("device forward pass")
        suite.check(True, "skipped, this build has no device code in it")


def run_on_device(mut suite: Suite, ctx: DeviceContext) raises:
    test_forward(suite, ctx)


def _shapes(mut cols: List[Int], mut rows: List[Int]):
    """Every matrix in the model, in one order both halves read.

    A single list rather than a field per weight, because the whole point is
    that the host tensor and the device tensor for a given matrix come from the
    same bytes, and the cheapest way to hold that is for them to come from the
    same index.
    """
    cols.append(WIDTH)
    rows.append(VOCAB)
    cols.append(WIDTH)
    rows.append(VOCAB)
    for _ in range(LAYERS):
        cols.append(WIDTH)
        rows.append(HEADS * HEAD_DIM)
        cols.append(WIDTH)
        rows.append(KV_HEADS * HEAD_DIM)
        cols.append(WIDTH)
        rows.append(KV_HEADS * HEAD_DIM)
        cols.append(HEADS * HEAD_DIM)
        rows.append(WIDTH)
        cols.append(WIDTH)
        rows.append(HIDDEN)
        cols.append(WIDTH)
        rows.append(HIDDEN)
        cols.append(HIDDEN)
        rows.append(WIDTH)


def _align(n: Int, to: Int) -> Int:
    return ((n + to - 1) // to) * to


def _store_f32(p: RawPtr, at: Int, value: Float32):
    var bits = bitcast[DType.uint32, 1](value)
    for b in range(4):
        p.unsafe_store(at + b, UInt8((bits >> UInt32(b * 8)) & 0xFF))


def _fill(p: RawPtr, base: Int, m: Int, cols: Int, rows: Int) raises:
    """One planar q8_0 matrix, filled from its index so no two are alike.

    The scales are small and the quants cover most of the signed range, which
    keeps a two layer stack in a range where a disagreement is a bug rather than
    a float32 running out of mantissa.
    """
    var stride = planar_row_bytes(Q_Q8_0, cols)
    for r in range(rows):
        var row = base + r * stride
        for i in range(cols):
            var q = ((i * 7 + r * 29 + m * 97) % 241) - 120
            p.unsafe_store(row + i, UInt8(q & 0xFF))
        for gi in range(cols // 32):
            var scale = Float32(0.01) + Float32((gi + m + r) % 7) * 0.002
            _store_f32(p, row + cols + gi * 4, scale)


def _gains(mut out: List[Float32], m: Int):
    """One norm gain, near one and not equal to it.

    A gain of exactly one would let a norm that reads the wrong weight pass, and
    there are three of them per layer to read wrong.
    """
    for i in range(WIDTH):
        out.append(Float32(0.9) + Float32((i + m) % 11) * 0.02)


def test_forward(mut suite: Suite, ctx: DeviceContext) raises:
    suite.group("device forward pass")

    comptime if not has_accelerator():
        return
    else:
        var cols = List[Int]()
        var rows = List[Int]()
        _shapes(cols, rows)

        # Offsets first, then one allocation, because a list that grows while
        # something holds a pointer into it is a pointer into freed memory.
        var offs = List[Int]()
        var total = 0
        for m in range(MATRICES):
            offs.append(total)
            var each = planar_row_bytes(Q_Q8_0, cols[m]) * rows[m]
            total = _align(total + each, 256)

        var blob = List[UInt8]()
        for _ in range(total):
            blob.append(0)
        var bp = RawPtr(unsafe_from_address=Int(blob.unsafe_ptr()))
        for m in range(MATRICES):
            _fill(bp, offs[m], m, cols[m], rows[m])

        # The norm gains stay on the host in both models. The device one
        # uploads them when it binds, which is what `DeviceLayer` is for.
        var gains = List[Float32]()
        for m in range(LAYERS * 2 + 3):
            _gains(gains, m)
        var gp = Int(gains.unsafe_ptr())

        var dev = default_device()
        var pool = DevicePool(dev, total, ctx)
        for m in range(MATRICES):
            var each = planar_row_bytes(Q_Q8_0, cols[m]) * rows[m]
            pool.copy_in(offs[m], Int(blob.unsafe_ptr()) + offs[m], each)
        pool.wait()

        var host_at = Int(blob.unsafe_ptr())
        var dev_at = List[Int]()
        for m in range(MATRICES):
            dev_at.append(pool.slot_address(offs[m]))

        def host_tensor(
            at: Int, offs: List[Int], cols: List[Int], rows: List[Int], m: Int
        ) raises -> Tensor:
            return Tensor(at + offs[m], Q_Q8_0, cols[m], rows[m], LAYOUT_PLANAR)

        def dev_tensor(
            at: List[Int], cols: List[Int], rows: List[Int], m: Int
        ) raises -> Tensor:
            return Tensor(
                at[m], Q_Q8_0, cols[m], rows[m], LAYOUT_PLANAR, WHERE_DEVICE
            )

        def gain_tensor(at: Int, m: Int) raises -> Tensor:
            return Tensor(at + m * WIDTH * 4, Q_F32, WIDTH, 1)

        var arch = arch_of(architecture_id("llama"))
        var attn = AttnSpec(HEADS, KV_HEADS, HEAD_DIM)
        var rope = RopeSpec(HEAD_DIM, 10000.0)
        var specs = List[BlockSpec]()
        for _ in range(LAYERS):
            specs.append(BlockSpec(attn, rope, WIDTH, HIDDEN, 1e-5))

        var host_model = ModelWeights()
        host_model.embedding = host_tensor(host_at, offs, cols, rows, 0)
        host_model.output = host_tensor(host_at, offs, cols, rows, 1)
        host_model.output_norm = gain_tensor(gp, LAYERS * 2)
        var dev_model = ModelWeights()
        dev_model.embedding = dev_tensor(dev_at, cols, rows, 0)
        dev_model.output = dev_tensor(dev_at, cols, rows, 1)
        dev_model.output_norm = gain_tensor(gp, LAYERS * 2)

        var host_layers = List[LayerWeights]()
        var dev_layers = List[LayerWeights]()
        for l in range(LAYERS):
            var at = 2 + l * 7
            var h = LayerWeights()
            h.attn_norm = gain_tensor(gp, l * 2)
            h.ffn_norm = gain_tensor(gp, l * 2 + 1)
            # The first layer carries the two norms Gemma puts after a sublayer
            # and the second does not, so one pass covers both shapes. The
            # post norm path is the one that cannot ride a projection epilogue,
            # and a chunk that is not a whole chunk is where it goes wrong.
            if l == 0:
                h.attn_post_norm = gain_tensor(gp, LAYERS * 2 + 1)
                h.ffn_post_norm = gain_tensor(gp, LAYERS * 2 + 2)
            h.wq = host_tensor(host_at, offs, cols, rows, at)
            h.wk = host_tensor(host_at, offs, cols, rows, at + 1)
            h.wv = host_tensor(host_at, offs, cols, rows, at + 2)
            h.wo = host_tensor(host_at, offs, cols, rows, at + 3)
            h.gate = host_tensor(host_at, offs, cols, rows, at + 4)
            h.up = host_tensor(host_at, offs, cols, rows, at + 5)
            h.down = host_tensor(host_at, offs, cols, rows, at + 6)
            h.check(specs[l])
            host_layers.append(h)

            var d = h
            d.wq = dev_tensor(dev_at, cols, rows, at)
            d.wk = dev_tensor(dev_at, cols, rows, at + 1)
            d.wv = dev_tensor(dev_at, cols, rows, at + 2)
            d.wo = dev_tensor(dev_at, cols, rows, at + 3)
            d.gate = dev_tensor(dev_at, cols, rows, at + 4)
            d.up = dev_tensor(dev_at, cols, rows, at + 5)
            d.down = dev_tensor(dev_at, cols, rows, at + 6)
            dev_layers.append(d)

        var factors = List[Float32]()

        # The host run, which is the reference. Ordinary `forward`, ordinary
        # cache, nothing about it knows a device exists.
        var scratch = Scratch(specs[0], CONTEXT)
        scratch.tracing = True
        var x = Buffer(WIDTH)
        var logits = Buffer(VOCAB)
        var keys = List[List[Float32]]()
        var values = List[List[Float32]]()
        for _ in range(LAYERS):
            var k = List[Float32]()
            var v = List[Float32]()
            for _ in range(CONTEXT * KV_HEADS * HEAD_DIM):
                k.append(0.0)
                v.append(0.0)
            keys.append(k^)
            values.append(v^)

        var model = DeviceModel(
            ctx,
            arch,
            specs,
            host_model,
            dev_model,
            host_layers,
            dev_layers,
            factors,
        )
        var dscratch = DeviceScratch(ctx, specs[0], CONTEXT, VOCAB)
        dscratch.tracing = True
        var dx = DeviceVec(ctx, WIDTH)
        var cache = DeviceKvCache(ctx, LAYERS, CONTEXT, KV_HEADS * HEAD_DIM)
        var got = Buffer(VOCAB)

        # Twenty of them rather than a handful, because a prefill block carries
        # `SPAN` tokens and a run shorter than one block would never exercise a
        # second block, a tail block, or the clamp the tail lanes ride on.
        var tokens = List[Int]()
        tokens.append(3)
        tokens.append(41)
        tokens.append(0)
        tokens.append(VOCAB - 1)
        tokens.append(17)
        for i in range(15):
            tokens.append((i * 13 + 5) % VOCAB)

        var worst = Float32(0)
        var peak = Float32(0)
        var picks = 0
        for step in range(len(tokens)):
            forward(
                arch,
                host_model,
                specs,
                host_layers,
                scratch,
                x,
                tokens[step],
                step,
                step,
                keys,
                values,
                factors,
                logits,
            )
            var one: List[Int] = [tokens[step]]
            device_forward(
                ctx,
                model,
                dscratch,
                dx,
                one,
                step,
                step,
                cache.keys,
                cache.values,
            )
            ctx.synchronize()
            dscratch.logits.download(got)

            var want_top = 0
            var got_top = 0
            for i in range(VOCAB):
                if logits.data[i] > logits.data[want_top]:
                    want_top = i
                if got.data[i] > got.data[got_top]:
                    got_top = i
                var m = logits.data[i]
                if m < 0:
                    m = -m
                if m > peak:
                    peak = m
                var gap = got.data[i] - logits.data[i]
                if gap < 0:
                    gap = -gap
                if gap > worst:
                    worst = gap
            if want_top == got_top:
                picks += 1

        # The prefill path, over the same tokens, into a cache of its own. This
        # is the claim #167 makes and it is not implied by anything above: the
        # decodes ran the matvec, the norms and the attention one token at
        # a time, and one pass over five tokens runs a different kernel for
        # every one of them. What has to survive that is the last token's
        # logits and both cache planes, because a chunk that gets the logits
        # right and the cache wrong answers the prompt and then drifts.
        var batch = DeviceScratch(ctx, specs[0], CONTEXT, VOCAB, len(tokens))
        var bx = DeviceVec(ctx, (len(tokens) + SPAN * MM_GROUPS) * WIDTH)
        var bcache = DeviceKvCache(ctx, LAYERS, CONTEXT, KV_HEADS * HEAD_DIM)
        device_forward(
            ctx, model, batch, bx, tokens, 0, 0, bcache.keys, bcache.values
        )
        ctx.synchronize()
        var batched = Buffer(VOCAB)
        batch.logits.download(batched)

        # And again in two chunks, because a prompt longer than the scratch is
        # the ordinary case and the second chunk is the one that has to find
        # the first chunk's keys where it left them. The split is not on a
        # `SPAN` boundary on purpose.
        var split = 13
        var head_run = List[Int]()
        var tail_run = List[Int]()
        for i in range(len(tokens)):
            if i < split:
                head_run.append(tokens[i])
            else:
                tail_run.append(tokens[i])
        var pair = DeviceScratch(ctx, specs[0], CONTEXT, VOCAB, len(tokens))
        var px = DeviceVec(ctx, (len(tokens) + SPAN * MM_GROUPS) * WIDTH)
        var pcache = DeviceKvCache(ctx, LAYERS, CONTEXT, KV_HEADS * HEAD_DIM)
        device_forward(
            ctx, model, pair, px, head_run, 0, 0, pcache.keys, pcache.values
        )
        device_forward(
            ctx,
            model,
            pair,
            px,
            tail_run,
            split,
            split,
            pcache.keys,
            pcache.values,
        )
        ctx.synchronize()
        var split_out = Buffer(VOCAB)
        pair.logits.download(split_out)
        var split_worst = Float32(0)
        for i in range(VOCAB):
            var gap = split_out.data[i] - logits.data[i]
            if gap < 0:
                gap = -gap
            if gap > split_worst:
                split_worst = gap

        var batch_worst = Float32(0)
        var batch_top = 0
        var want_last = 0
        for i in range(VOCAB):
            if batched.data[i] > batched.data[batch_top]:
                batch_top = i
            if logits.data[i] > logits.data[want_last]:
                want_last = i
            var gap = batched.data[i] - logits.data[i]
            if gap < 0:
                gap = -gap
            if gap > batch_worst:
                batch_worst = gap

        var span = CONTEXT * KV_HEADS * HEAD_DIM
        var live = len(tokens) * KV_HEADS * HEAD_DIM
        var mine = Buffer(span)
        var theirs = Buffer(span)
        var cache_worst = Float32(0)
        var cache_peak = Float32(0)
        for l in range(LAYERS):
            for half in range(2):
                if half == 0:
                    cache.keys[l].download(mine)
                    bcache.keys[l].download(theirs)
                else:
                    cache.values[l].download(mine)
                    bcache.values[l].download(theirs)
                for i in range(live):
                    var m = mine.data[i]
                    if m < 0:
                        m = -m
                    if m > cache_peak:
                        cache_peak = m
                    var gap = theirs.data[i] - mine.data[i]
                    if gap < 0:
                        gap = -gap
                    if gap > cache_worst:
                        cache_worst = gap

        keep(pool)
        keep(blob)
        keep(gains)

        suite.check(peak > 0, "the host reference is not all zeros")
        suite.check(
            worst <= peak * Float32(2e-4),
            "and the device logits agree with it on every token",
        )
        suite.check(
            picks == len(tokens),
            "and greedy sampling picks the same token every step",
        )

        # The trace is the other half of the claim. Same count and same numbers
        # means a divergence can be named by layer rather than only noticed at
        # the end, which is what the logit corpus in #30 asks the device path
        # for.
        var want_snaps = scratch.snapshots(WIDTH)
        var got_snaps = dscratch.snapshots(WIDTH)
        suite.check(
            want_snaps == len(tokens) * (LAYERS + 2),
            "the host trace has a snapshot per layer per token plus two",
        )
        suite.check(
            got_snaps == want_snaps,
            "and the device trace has exactly as many",
        )

        var trace_worst = Float32(0)
        var trace_peak = Float32(0)
        for i in range(len(scratch.trace)):
            var m = scratch.trace[i]
            if m < 0:
                m = -m
            if m > trace_peak:
                trace_peak = m
            var gap = dscratch.trace[i] - scratch.trace[i]
            if gap < 0:
                gap = -gap
            if gap > trace_worst:
                trace_worst = gap
        suite.check(
            trace_worst <= trace_peak * Float32(2e-4),
            (
                "and the residual stream agrees layer by layer, not only at"
                " the end"
            ),
        )

        suite.group("device prefill against device decode")
        suite.check(
            batch_worst <= peak * Float32(2e-4),
            (
                "a chunk leaves the last token's logits where the decodes left"
                " them"
            ),
        )
        suite.check(
            batch_top == want_last,
            "and greedy picks the same token off them",
        )
        suite.check(cache_peak > 0, "the cache the decodes left is not zeros")
        suite.check(
            cache_worst <= cache_peak * Float32(2e-4),
            "and the chunk leaves the same keys and values in it",
        )
        suite.check(
            split_worst <= peak * Float32(2e-4),
            "and a prompt split across two chunks reaches the same logits",
        )
