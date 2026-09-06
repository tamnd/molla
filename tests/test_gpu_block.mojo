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
from molla.nn.gpu import MM_GROUPS, SPAN, DeviceHalf, DeviceVec
from molla.nn.gpu_block import (
    DeviceModel,
    DeviceScratch,
    build_fused_plan,
    device_forward,
    device_forward_fused,
)
from molla.nn.model import ModelWeights, forward
from molla.nn.quant import Q_F32, Q_Q8_0
from molla.nn.repack import (
    CACHE_F16,
    CACHE_Q8,
    LAYOUT_PLANAR,
    SCALE_BYTES,
    cache_row,
    planar_row_bytes,
)
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


struct Rng(Movable):
    """The same small generator the cache tests use, so a layout is a seed.

    A fuzz over pool layouts is only worth running if a failure can be run
    again, and a fixed seed is what makes the twentieth trial of a bad run
    reachable without capturing anything.
    """

    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt64:
        self.state ^= self.state << 13
        self.state ^= self.state >> 7
        self.state ^= self.state << 17
        return self.state

    def upto(mut self, n: Int) -> Int:
        """A number in `[0, n)`, which is all this is ever asked for."""
        return Int(self.next() % UInt64(n))


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
    test_pool(suite, ctx)
    test_forward(suite, ctx)


def test_pool(mut suite: Suite, ctx: DeviceContext) raises:
    """The cache is one allocation and a layer's window is where it says.

    Worth its own check rather than leaning on the forward pass, because the
    forward pass would pass with every window at offset zero as long as it was
    the only sequence in flight: it writes a layer and reads the same layer, and
    two layers landing on top of each other shows up as wrong attention several
    layers later, or not at all on a one layer model. This asks the question
    directly, which is whether a write through layer `i`'s window lands at
    layer `i`'s offset in the pool and nowhere else.
    """
    suite.group("device cache pool")

    comptime if not has_accelerator():
        suite.check(True, "skipped, this build has no device code in it")
        return
    else:
        var cache = DeviceKvCache(ctx, LAYERS, CONTEXT, KV_HEADS * HEAD_DIM)
        var per = CONTEXT * cache.row
        suite.check(
            cache.pool.elements() == 2 * LAYERS * per,
            "the pool holds every layer's keys and then every layer's values",
        )
        suite.check(
            cache.bytes() == cache.pool.elements() * 2,
            "and the size it reports is the pool's, at two bytes a half",
        )
        suite.check(
            len(cache.keys) == LAYERS and len(cache.values) == LAYERS,
            "and there is a window a layer on each side",
        )

        # A different value at the front of every window, written through the
        # window and read back through the pool.
        var one = List[Float32]()
        one.append(0.0)
        for i in range(LAYERS):
            one[0] = Float32(i + 1)
            cache.keys[i].upload_run(one, 0, 1)
            one[0] = Float32(-(i + 1))
            cache.values[i].upload_run(one, 0, 1)

        var placed = True
        for i in range(LAYERS):
            if cache.pool.at(i * per) != Float32(i + 1):
                placed = False
            if cache.pool.at((LAYERS + i) * per) != Float32(-(i + 1)):
                placed = False
        suite.check(placed, "and a write through a window lands at its offset")

        var refused = False
        try:
            _ = DeviceHalf(cache.pool, 2 * LAYERS * per, 1)
        except:
            refused = True
        suite.check(refused, "and a window past the end of the pool is refused")


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


def _store_f16(p: RawPtr, at: Int, value: Float32):
    """A little endian float16, which is the width a planar scale is stored at.

    A float32 store here would be two bytes too wide, and the two bytes past the
    end of a group's scale are the next group's scale and then the next row's
    quants, so a four byte store builds a matrix nothing else in the file thinks
    it built. It read as scales near a hundred rather than near a hundredth,
    which a two layer stack turns into activations in the tens of millions.
    """
    var bits = bitcast[DType.uint16, 1](value.cast[DType.float16]())
    p.unsafe_store(at, UInt8(bits & 0xFF))
    p.unsafe_store(at + 1, UInt8((bits >> 8) & 0xFF))


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
            _store_f16(p, row + cols + gi * SCALE_BYTES, scale)


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
        var decoded = List[Float32]()
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
                cache.paging,
            )
            ctx.synchronize()
            dscratch.logits.download(got)
            for i in range(VOCAB):
                decoded.append(got.data[i])

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

        # The fused path, over the same tokens, into a cache of its own. Every
        # record it walks does the same arithmetic as the kernel it replaces,
        # and everywhere outside the attention it does it in the same order, so
        # the cache below is compared bit for bit and the residual stream to a
        # tolerance. The comparison that catches a barrier in the wrong place is
        # either one: a missing barrier reads a value from before a write, which
        # is a plausible number and not a number one bit off.
        #
        # The first layer of this model has the two post norms and the second
        # does not, so the pass covers both the projection that carries the
        # residual add in its epilogue and the one that cannot.
        var plan = build_fused_plan(ctx, model, CONTEXT, False)
        var fscratch = DeviceScratch(ctx, specs[0], CONTEXT, VOCAB)
        var fx = DeviceVec(ctx, WIDTH)
        var fcache = DeviceKvCache(ctx, LAYERS, CONTEXT, KV_HEADS * HEAD_DIM)
        fscratch.tracing = True
        var fused_out = Buffer(VOCAB)
        var fused_worst = Float32(0)
        for step in range(len(tokens)):
            device_forward_fused(
                ctx,
                model,
                plan,
                fscratch,
                fx,
                tokens[step],
                step,
                step,
                fcache.keys,
                fcache.values,
            )
            ctx.synchronize()
            fscratch.logits.download(fused_out)
            for i in range(VOCAB):
                var gap = fused_out.data[i] - decoded[step * VOCAB + i]
                if gap < 0:
                    gap = -gap
                if gap > fused_worst:
                    fused_worst = gap
        var fused_stuck = plan.stuck()

        # The plan a session builds when it has decided against the path, which
        # is nothing at all. Building one compiles the kernel and that is what
        # is being avoided, so what this checks is that the empty plan is empty
        # and that nothing quietly runs against it.
        var unwanted = build_fused_plan(ctx, model, CONTEXT, False, False)
        var unwanted_raised = False
        try:
            device_forward_fused(
                ctx,
                model,
                unwanted,
                fscratch,
                fx,
                tokens[0],
                0,
                0,
                fcache.keys,
                fcache.values,
            )
            ctx.synchronize()
        except:
            unwanted_raised = True

        # The trace is one residual stream a layer a token, so this compares
        # every layer of every step rather than only what came out the end. It
        # is what says which layer went wrong when one does, and a comparison
        # of the logits alone would have said only that something did.
        #
        # A tolerance and not an equality, because the two attentions do not add
        # their keys up in the same order. The unfused kernel gives a head to a
        # block and walks the whole row, and the fused one cuts the row into as
        # many slices as the grid has blocks to spare and folds the slices back
        # together the way flash decoding does. Both are the same sum and
        # neither is more right than the other, and a float32 sum is not
        # associative, so the last bit of a long row is a coin toss between
        # them. The bound is a hundred times tighter than the one the host
        # comparison carries, because both sides of this one read the same half
        # precision cache and the fold is the only thing left to disagree
        # about. Measured worst is 1.7e-7 of the peak, so there is room. What
        # the bound is worth is the same thing every other bound in this file is
        # worth: a barrier in the wrong place reads a value from before a write
        # rather than a value one bit off, and that is not a rounding
        # difference, it is a different number.
        var fused_trace_worst = Float32(0)
        if len(fscratch.trace) != len(dscratch.trace):
            raise Error("the two traces are not the same length")
        for i in range(len(fscratch.trace)):
            var gap = fscratch.trace[i] - dscratch.trace[i]
            if gap < 0:
                gap = -gap
            if gap > fused_trace_worst:
                fused_trace_worst = gap

        # The cache is the one part that is still exact, and it stays exact.
        # Nothing that writes it reduces across a slice: a projection is a row
        # of a matvec, a norm is one element, and a rotation is a pair. So the
        # fused path and the unfused path run the same additions in the same
        # order here and there is no rounding for them to disagree about.
        var fused_cache_diff = 0
        var fused_mine = Buffer(CONTEXT * KV_HEADS * HEAD_DIM)
        var fused_theirs = Buffer(CONTEXT * KV_HEADS * HEAD_DIM)
        for l in range(LAYERS):
            for half in range(2):
                if half == 0:
                    cache.keys[l].download(fused_mine)
                    fcache.keys[l].download(fused_theirs)
                else:
                    cache.values[l].download(fused_mine)
                    fcache.values[l].download(fused_theirs)
                for i in range(len(tokens) * KV_HEADS * HEAD_DIM):
                    if fused_theirs.data[i] != fused_mine.data[i]:
                        fused_cache_diff += 1

        # The same tokens again with the cache held at q8_0, unfused and fused
        # into caches of their own. Two claims, and they are different claims.
        # The two paths write the cache with different kernels, `_put_block` in
        # the fused one and `store_kv_q8_kernel` in the other, and both round
        # the same way over the same block, so the bytes have to match exactly.
        # The logits only have to stay near the float16 run, because a q8 cache
        # is a coarser cache and is expected to differ.
        var qcache = DeviceKvCache(
            ctx, LAYERS, CONTEXT, KV_HEADS * HEAD_DIM, CACHE_Q8
        )
        var qscratch = DeviceScratch(ctx, specs[0], CONTEXT, VOCAB)
        var qx = DeviceVec(ctx, WIDTH)
        var qplan = build_fused_plan(ctx, model, CONTEXT, False, True, CACHE_Q8)
        var qfcache = DeviceKvCache(
            ctx, LAYERS, CONTEXT, KV_HEADS * HEAD_DIM, CACHE_Q8
        )
        var qfscratch = DeviceScratch(ctx, specs[0], CONTEXT, VOCAB)
        var qfx = DeviceVec(ctx, WIDTH)
        var q8_out = Buffer(VOCAB)
        var q8_worst = Float32(0)
        var q8_fused_worst = Float32(0)
        var q8_picks = 0
        for step in range(len(tokens)):
            var one: List[Int] = [tokens[step]]
            device_forward(
                ctx,
                model,
                qscratch,
                qx,
                one,
                step,
                step,
                qcache.keys,
                qcache.values,
                qcache.paging,
                CACHE_Q8,
            )
            ctx.synchronize()
            qscratch.logits.download(q8_out)
            var q8_top = 0
            for i in range(VOCAB):
                if q8_out.data[i] > q8_out.data[q8_top]:
                    q8_top = i
                var gap = q8_out.data[i] - decoded[step * VOCAB + i]
                if gap < 0:
                    gap = -gap
                if gap > q8_worst:
                    q8_worst = gap
            var half_top = 0
            for i in range(VOCAB):
                if decoded[step * VOCAB + i] > decoded[step * VOCAB + half_top]:
                    half_top = i
            if q8_top == half_top:
                q8_picks += 1

            device_forward_fused(
                ctx,
                model,
                qplan,
                qfscratch,
                qfx,
                tokens[step],
                step,
                step,
                qfcache.keys,
                qfcache.values,
            )
            ctx.synchronize()
            var q8_fused = Buffer(VOCAB)
            qfscratch.logits.download(q8_fused)
            for i in range(VOCAB):
                var gap = q8_fused.data[i] - q8_out.data[i]
                if gap < 0:
                    gap = -gap
                if gap > q8_fused_worst:
                    q8_fused_worst = gap

        var q8_row = cache_row(CACHE_Q8, KV_HEADS * HEAD_DIM)
        var q8_diff = 0
        var q8_mine = List[Int]()
        var q8_theirs = List[Int]()
        for l in range(LAYERS):
            for half in range(2):
                if half == 0:
                    qcache.keys[l].download_bits(q8_mine)
                    qfcache.keys[l].download_bits(q8_theirs)
                else:
                    qcache.values[l].download_bits(q8_mine)
                    qfcache.values[l].download_bits(q8_theirs)
                # The live halves of each row and not the padding. A row is
                # rounded up so the next one starts where a thirty two bit
                # store can own it, and nothing writes the halves that rounding
                # adds, so they hold whatever the allocation came with and the
                # two caches have no reason to agree about them. Bit patterns
                # and not floats, because the quant plane read as float16 is
                # mostly nans and a nan is not equal to itself.
                var live = KV_HEADS * HEAD_DIM // 2 + KV_HEADS * HEAD_DIM // 32
                for r in range(len(tokens)):
                    for i in range(live):
                        var at = r * q8_row + i
                        if q8_theirs[at] != q8_mine[at]:
                            q8_diff += 1

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
            ctx,
            model,
            batch,
            bx,
            tokens,
            0,
            0,
            bcache.keys,
            bcache.values,
            bcache.paging,
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
            ctx,
            model,
            pair,
            px,
            head_run,
            0,
            0,
            pcache.keys,
            pcache.values,
            pcache.paging,
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
            pcache.paging,
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

        # And once more through the cell table, which is the whole prompt again
        # but with the cache addressed by cell rather than by position. It is
        # only worth running if the two disagree about where a token goes, so a
        # decoy sequence takes the front of the pool before this one starts and
        # every token here lands four cells past where its position would have
        # put it. The store has to scatter, the mask has to hide the decoy's
        # four cells, and the logits have to come out where the contiguous run
        # left them.
        var gcache = DeviceKvCache(ctx, LAYERS, CONTEXT, KV_HEADS * HEAD_DIM)
        var decoy = List[Int]()
        gcache.table.alloc_run(1, 0, 4, decoy)
        var grid = DeviceScratch(ctx, specs[0], CONTEXT, VOCAB, len(tokens))
        var gx = DeviceVec(ctx, (len(tokens) + SPAN * MM_GROUPS) * WIDTH)
        var shift = gcache.place(0, len(tokens), True)
        var paged_on = gcache.paging.on
        device_forward(
            ctx,
            model,
            grid,
            gx,
            tokens,
            0,
            shift,
            gcache.keys,
            gcache.values,
            gcache.paging,
        )
        ctx.synchronize()
        var paged_out = Buffer(VOCAB)
        grid.logits.download(paged_out)
        var paged_worst = Float32(0)
        for i in range(VOCAB):
            var gap = paged_out.data[i] - batched.data[i]
            if gap < 0:
                gap = -gap
            if gap > paged_worst:
                paged_worst = gap

        # And the same claim over layouts nobody picked. The pool gets a decoy
        # of a random length that then gives a random half of itself back, so
        # the cells this prompt lands in have holes in them rather than being
        # the same run moved along, and the prompt is cut into chunks of a
        # random length so a step is not always the same size. Both routes run
        # the same chunks over the same tokens and the only thing that differs
        # between them is which cell a position went in.
        #
        # A dozen tokens and not twenty, so that a decoy long enough to matter
        # still leaves room for the prompt in a pool of `CONTEXT`.
        var rng = Rng(0x2E31D0)
        var fuzz_n = 12
        var fuzz_worst = Float32(0)
        var fuzz_paged = 0
        var fuzz_trials = 8
        var paged_out2 = Buffer(VOCAB)
        var plain_out = Buffer(VOCAB)
        # A scratch of its own, because the trace on the one the decodes used
        # is compared by length further down and a fuzz would lengthen it. A
        # chunk of one token norms through a vector the width of the residual
        # stream and a longer chunk norms through the wide one, which is the
        # same choice a session makes and the reason both are here.
        var fone = DeviceScratch(ctx, specs[0], CONTEXT, VOCAB)
        var fonex = DeviceVec(ctx, WIDTH)
        for _ in range(fuzz_trials):
            var sizes = List[Int]()
            var cut = 0
            while cut < fuzz_n:
                var n = 1 + rng.upto(4)
                if cut + n > fuzz_n:
                    n = fuzz_n - cut
                sizes.append(n)
                cut += n
            var last = sizes[len(sizes) - 1]

            var holed = DeviceKvCache(ctx, LAYERS, CONTEXT, KV_HEADS * HEAD_DIM)
            var junk = List[Int]()
            var take = 1 + rng.upto(CONTEXT - fuzz_n)
            holed.table.alloc_run(1, 0, take, junk)
            for i in range(take):
                if rng.upto(2) == 0:
                    _ = holed.table.release(1, i, i + 1)

            var plain = DeviceKvCache(ctx, LAYERS, CONTEXT, KV_HEADS * HEAD_DIM)
            for route in range(2):
                var seen = 0
                for c in range(len(sizes)):
                    var n = sizes[c]
                    var run = List[Int]()
                    for i in range(n):
                        run.append(tokens[seen + i])
                    if route == 0:
                        var slot = holed.place(seen, n, False)
                        if holed.paging.on:
                            fuzz_paged += 1
                        device_forward(
                            ctx,
                            model,
                            batch if n > 1 else fone,
                            bx if n > 1 else fonex,
                            run,
                            seen,
                            slot,
                            holed.keys,
                            holed.values,
                            holed.paging,
                        )
                    else:
                        var slot = plain.place(seen, n, False)
                        device_forward(
                            ctx,
                            model,
                            batch if n > 1 else fone,
                            bx if n > 1 else fonex,
                            run,
                            seen,
                            slot,
                            plain.keys,
                            plain.values,
                            plain.paging,
                        )
                    seen += n
                ctx.synchronize()
                if route == 0:
                    if last > 1:
                        batch.logits.download(paged_out2)
                    else:
                        fone.logits.download(paged_out2)
                else:
                    if last > 1:
                        batch.logits.download(plain_out)
                    else:
                        fone.logits.download(plain_out)
            for i in range(VOCAB):
                var gap = paged_out2.data[i] - plain_out.data[i]
                if gap < 0:
                    gap = -gap
                if gap > fuzz_worst:
                    fuzz_worst = gap

        # The cache the two runs left, row for row, offset by the shift. Bytes
        # and not a tolerance: the store writes the same rows either way and
        # the only thing that changed is which cell it wrote them to, so a
        # difference here is an index and not a rounding.
        var row = KV_HEADS * HEAD_DIM
        var here = Buffer(CONTEXT * row)
        var there = Buffer(CONTEXT * row)
        var paged_diff = 0
        for l in range(LAYERS):
            for half in range(2):
                if half == 0:
                    gcache.keys[l].download(here)
                    bcache.keys[l].download(there)
                else:
                    gcache.values[l].download(here)
                    bcache.values[l].download(there)
                for r in range(len(tokens)):
                    for i in range(row):
                        var mine = here.data[(shift + r) * row + i]
                        if mine != there.data[r * row + i]:
                            paged_diff += 1

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

        # Two sequences in one pass, which is the gate #32 stage three is for.
        # One pool holds both of them. Sequence zero's list of cells sits at the
        # front of the index and sequence one's at the halfway mark, and a token
        # says which list is its own by carrying the offset that list starts at.
        # That offset is the whole descriptor, because the number of entries a
        # token reads is its own position plus one and it already carries its
        # position.
        #
        # Nothing here goes through `DeviceKvCache.place`, and that is on
        # purpose. `place` writes one sequence's list at the front of the index
        # and hands out the cells for one sequence, which is right for the
        # session that owns the cache and is exactly what the scheduler in stage
        # four replaces. What is under test here is the pass, so the cells and
        # the two regions are laid out by hand and `CellTable.route` fills each
        # region the way it was written to.
        var mid = CONTEXT // 2
        var duo = DeviceScratch(ctx, specs[0], CONTEXT, VOCAB, 4, 2)
        var duox = DeviceVec(ctx, (4 + SPAN * MM_GROUPS) * WIDTH)
        var solo = DeviceScratch(ctx, specs[0], CONTEXT, VOCAB, 4)
        var solox = DeviceVec(ctx, (4 + SPAN * MM_GROUPS) * WIDTH)
        var shared = DeviceKvCache(ctx, LAYERS, CONTEXT, KV_HEADS * HEAD_DIM)
        var starts = List[Int]()
        starts.append(0)
        starts.append(mid)
        var spots = List[Int32](length=4, fill=0)
        var firsts = List[Int32](length=4, fill=0)
        var alone = Buffer(2 * VOCAB)
        var together = Buffer(2 * VOCAB)
        var one_out = Buffer(VOCAB)
        var pair_prefill = 0
        for q in range(2):
            # Different lengths and different tokens, so that a batch which
            # quietly gave both sequences the longer one's answer would show up
            # rather than agreeing by accident.
            var n = 4 - q
            var prompt = List[Int]()
            for i in range(n):
                prompt.append(tokens[q * 8 + i])
            var next = List[Int]()
            next.append(tokens[q * 8 + n])

            # What this sequence gets on its own: its own cache, its own pool,
            # the ordinary contiguous path, a prefill and then a decode.
            var own = DeviceKvCache(ctx, LAYERS, CONTEXT, KV_HEADS * HEAD_DIM)
            var at = own.place(0, n, False)
            device_forward(
                ctx,
                model,
                solo,
                solox,
                prompt,
                0,
                at,
                own.keys,
                own.values,
                own.paging,
            )
            at = own.place(n, 1, False)
            device_forward(
                ctx,
                model,
                fone,
                fonex,
                next,
                n,
                at,
                own.keys,
                own.values,
                own.paging,
            )
            ctx.synchronize()
            fone.logits.download(one_out)
            for i in range(VOCAB):
                alone.data[q * VOCAB + i] = one_out.data[i]

            # And the same prompt into the shared pool, in its own region. The
            # index goes up whole rather than a span at a time because the two
            # regions are not next to each other, and a scheduler that appends a
            # token to each of sixteen sequences has sixteen spans to send. That
            # is stage four's problem and it does not change what the kernels
            # read.
            for i in range(n):
                var cell = shared.table.alloc(q, i)
                shared.paging.slots[i] = Int32(cell)
                spots[i] = Int32(i)
                firsts[i] = Int32(starts[q])
            shared.table.route(q, n, shared.paging.order, starts[q])
            shared.paging.on = True
            shared.paging.cells.queue_in(shared.paging.slots)
            shared.paging.index.queue_in(shared.paging.order)
            shared.paging.mixed(spots, firsts, n)
            if shared.paging.ragged:
                pair_prefill += 1
            device_forward(
                ctx,
                model,
                duo,
                duox,
                prompt,
                n - 1,
                0,
                shared.keys,
                shared.values,
                shared.paging,
            )

        # One step carrying a decode of each. The positions are four and three,
        # the list starts are zero and the halfway mark, and `pos` is the deeper
        # of the two positions because all it does now is size the scratch for
        # the token that needs the most of it.
        var both = List[Int]()
        for q in range(2):
            var n = 4 - q
            var cell = shared.table.alloc(q, n)
            shared.paging.slots[q] = Int32(cell)
            shared.paging.order[starts[q] + n] = Int32(cell)
            spots[q] = Int32(n)
            firsts[q] = Int32(starts[q])
            both.append(tokens[q * 8 + n])
        shared.paging.cells.queue_in(shared.paging.slots)
        shared.paging.index.queue_in(shared.paging.order)
        shared.paging.mixed(spots, firsts, 2)
        var want_rows = List[Int]()
        want_rows.append(0)
        want_rows.append(1)
        device_forward(
            ctx,
            model,
            duo,
            duox,
            both,
            4,
            0,
            shared.keys,
            shared.values,
            shared.paging,
            CACHE_F16,
            want_rows,
        )
        ctx.synchronize()
        duo.logits.download(together)

        var pair_worst = Float32(0)
        var pair_picks = 0
        var pair_apart = 0
        for q in range(2):
            var mark = q * VOCAB
            var top = 0
            var want = 0
            for i in range(VOCAB):
                if together.data[mark + i] > together.data[mark + top]:
                    top = i
                if alone.data[mark + i] > alone.data[mark + want]:
                    want = i
                var gap = together.data[mark + i] - alone.data[mark + i]
                if gap < 0:
                    gap = -gap
                if gap > pair_worst:
                    pair_worst = gap
            if top == want:
                pair_picks += 1
        # And the two rows are not the same row. A batch that wrote one
        # sequence's logits twice would pass everything above if the two
        # sequences happened to agree, so this says out loud that they do not.
        for i in range(VOCAB):
            if together.data[i] != together.data[VOCAB + i]:
                pair_apart += 1

        # A batch a scratch has no room for, which is the check that the row
        # list and the sequence count are the same number. The pass runs and
        # then refuses, because the layers do not know how many answers are
        # wanted and there is nothing to gain by teaching them.
        var pair_over = False
        var three = List[Int]()
        three.append(0)
        three.append(1)
        three.append(0)
        try:
            device_forward(
                ctx,
                model,
                solo,
                solox,
                both,
                4,
                0,
                shared.keys,
                shared.values,
                shared.paging,
                CACHE_F16,
                three,
            )
        except:
            pair_over = True

        keep(pool)
        keep(blob)
        keep(gains)

        # Three digits and not four, because the host cache is float32 and the
        # device cache is float16. Half carries eleven bits of mantissa, so a
        # key goes in with up to 2.4e-4 of relative error on it before anything
        # has read it, and what comes out the other side of an attention is a
        # weighted sum of values that were rounded the same way. Measured worst
        # here is 3.6e-4 of the peak logit, which is one rounding and not a
        # drift, and the greedy pick below is unaffected.
        var half_cache = Float32(2e-3)
        suite.check(peak > 0, "the host reference is not all zeros")
        suite.check(
            worst <= peak * half_cache,
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
            trace_worst <= trace_peak * half_cache,
            (
                "and the residual stream agrees layer by layer, not only at"
                " the end"
            ),
        )

        suite.group("fused layer against unfused")
        suite.check(
            plan.records == LAYERS * 14 + 4,
            (
                "a plan holds fourteen records a layer, and two more for each"
                " of the two post norms the first layer has"
            ),
        )
        suite.check(
            fused_stuck == 0,
            "no block ever gave up at a grid barrier",
        )
        suite.check(
            unwanted.records == 0,
            "a session that does not want the path gets a plan with no steps",
        )
        suite.check(
            unwanted_raised,
            "and a launch against one raises rather than running nothing",
        )
        suite.check(
            fused_trace_worst <= trace_peak * Float32(1e-5),
            (
                "every residual stream it leaves behind a layer agrees with the"
                " unfused one"
            ),
        )
        suite.check(
            fused_worst <= peak * Float32(1e-5),
            "and so do the logits",
        )
        suite.check(
            fused_cache_diff == 0,
            "and the keys and values agree in every bit",
        )

        suite.group("a q8 cache against a float16 one")
        # Smaller, and no tighter a claim than that at this width. A kv row here
        # is one q8 block, so the row is 17 halves rounded up to 24 and the
        # padding is a third of what is left. At a real width the padding is
        # nothing and the ratio is the 1.0625 bytes a value that
        # `tests/test_repack.mojo` pins.
        suite.check(
            qcache.bytes() < cache.bytes(),
            "a q8 cache is smaller than the float16 one it replaces",
        )
        # Two per cent, where the float16 cache is held to a fifth of that
        # against the host. Measured worst here is 0.86 per cent of the peak
        # logit, and it should be larger than the float16 number rather than
        # equal to it: eight bits and a factor a block is a coarser cache than
        # eleven bits an element, and the point of the flag is to trade that for
        # the memory. What the bound is worth is the same thing every bound here
        # is worth, which is that a scale plane read at the wrong offset is not
        # a rounding difference.
        var q8_gate = Float32(2e-2)
        suite.check(
            q8_worst <= peak * q8_gate,
            "the logits off a q8 cache stay near the float16 ones",
        )
        if q8_worst > peak * q8_gate:
            suite.fail("q8 logits", "worst " + String(q8_worst / peak))
        # Not every step, and the difference is not a fault. This is a model of
        # random weights with a vocabulary of 96, so the top two logits of a
        # step are often a thousandth apart, and a coarser cache moves the pick
        # at one step in twenty. What would say the path was broken is the pick
        # moving at most of them.
        suite.check(
            q8_picks * 10 >= len(tokens) * 9,
            "and greedy picks the same token at nearly every step",
        )
        if q8_picks * 10 < len(tokens) * 9:
            suite.fail(
                "q8 greedy",
                String(q8_picks) + " of " + String(len(tokens)),
            )
        suite.check(
            q8_fused_worst <= peak * Float32(1e-5),
            "the fused path agrees with the unfused one at q8 too",
        )
        suite.check(
            q8_diff == 0,
            "and the two write the same bytes into the cache",
        )

        suite.group("device prefill against device decode")
        suite.check(
            batch_worst <= peak * Float32(2e-4),
            (
                "a chunk leaves the last token's logits where the decodes left"
                " them"
            ),
        )
        if batch_worst > peak * Float32(2e-4):
            suite.fail("batch logits", "worst " + String(batch_worst / peak))
        suite.check(
            batch_top == want_last,
            "and greedy picks the same token off them",
        )
        suite.check(cache_peak > 0, "the cache the decodes left is not zeros")
        suite.check(
            cache_worst <= cache_peak * Float32(2e-4),
            "and the chunk leaves the same keys and values in it",
        )
        if cache_worst > cache_peak * Float32(2e-4):
            suite.fail(
                "batch cache", "worst " + String(cache_worst / cache_peak)
            )
        suite.check(
            split_worst <= peak * Float32(2e-4),
            "and a prompt split across two chunks reaches the same logits",
        )
        if split_worst > peak * Float32(2e-4):
            suite.fail("split logits", "worst " + String(split_worst / peak))

        suite.group("a paged cache against a contiguous one")
        suite.check(paged_on, "cells a position would not have picked page")
        suite.check(shift == 4, "and the run starts where the decoy left off")
        suite.check(
            paged_worst <= peak * Float32(2e-5),
            "a scattered prompt reaches the logits a contiguous one does",
        )
        if paged_worst > peak * Float32(2e-5):
            suite.fail("paged logits", "worst " + String(paged_worst / peak))
        suite.check(
            paged_diff == 0,
            "and leaves the same rows in the cells it was given",
        )
        suite.check(
            fuzz_paged > 0,
            "a pool with holes in it pages without being asked to",
        )
        suite.check(
            fuzz_worst <= peak * Float32(2e-5),
            "and every layout the fuzz picked reaches the same logits",
        )
        if fuzz_worst > peak * Float32(2e-5):
            suite.fail("fuzz logits", "worst " + String(fuzz_worst / peak))

        suite.group("two sequences in one pass")
        suite.check(
            pair_prefill == 2,
            "a token that carries its own list start makes the pass ragged",
        )
        suite.check(
            pair_worst <= peak * Float32(2e-5),
            "a batch of two gives each sequence what it gets on its own",
        )
        if pair_worst > peak * Float32(2e-5):
            suite.fail("batch pair", "worst " + String(pair_worst / peak))
        suite.check(
            pair_picks == 2,
            "and greedy picks the same token for each of them",
        )
        suite.check(
            pair_apart > 0,
            "and the two answers are two answers and not one written twice",
        )
        suite.check(
            pair_over,
            "a batch wanting more answers than the scratch holds is refused",
        )
