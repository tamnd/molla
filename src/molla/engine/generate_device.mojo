"""The same command as `molla.engine.generate`, with the arithmetic on a card.

A second function rather than a flag inside the first one. The two runs differ
in three places that are not small: what the load is asked for, which session
holds the sequence, and what has to be true of the weights before a token is
computed. Everything they share, which is the tokenizer, the prompt, the sampler
and the two reports, is imported from there rather than written again, so the
part that reads the same really is the same code.

## The two things a device run needs

Every weight has to be on the card, and every weight has to be planar. Neither is
a preference. A device kernel handed a host address reads zeros without faulting,
so a model that half fits produces fluent output from a stack where some layers
saw nothing, and a ggml block layout read by a planar kernel is arithmetic on the
wrong bytes. Both are refusals, and the first one is checked against the plan
before the bytes are read rather than against the tensors afterwards, so a model
that will not fit says so in a second rather than after a minute of copying.

The planar requirement is met rather than refused when it can be. A model with no
repack cache beside it gets one written first, which costs a full read of the
file and says so on the way past. That is a slow first run and then a fast one
forever, which is the same bargain `molla load` already offers, and it beats
telling somebody to go and run a different command.
"""

from std.sys.info import has_accelerator

from max.gpu.host import DeviceContext

from molla.engine.bind import bind
from molla.engine.device import DeviceSession
from molla.engine.generate import (
    DEFAULT_CONTEXT,
    DEFAULT_LIMIT,
    report_header,
    report_timing,
)
from molla.engine.sample import Sampler, SamplerConfig
from molla.model.gguf import Gguf
from molla.model.load import load, plan_load
from molla.model.repack import model_key, open_cache
from molla.model.spec import read_geometry
from molla.sys.clock import monotonic_ms
from molla.sys.device import default_device
from molla.sys.mem import AllocCounter
from molla.tokenizer.tokenizer import DecodeStream, Session, Tokenizer


def run_generate_device(
    model_path: String,
    tokenizer_path: String,
    prompt: String,
    limit: Int,
    context: Int,
    sampling: SamplerConfig = SamplerConfig(),
) raises:
    """Load onto the device, prefill, decode, and print as it goes.

    The output is meant to be the same text the host path prints for the same
    model, prompt and seed. That is the acceptance test for the whole device
    stack and it is a stronger one than any kernel comparison, because a kernel
    that is right on its own inputs and wrong about which layer it was called
    for still passes every test that only looks at one kernel.
    """
    sampling.check()

    comptime if not has_accelerator():
        raise Error(
            "this build has no device code in it, so there is nothing to"
            " generate on. Accelerator support is decided when molla is"
            " compiled, not when it is run"
        )

    comptime if has_accelerator():
        var dev = default_device()
        if not dev.accelerator():
            raise Error(
                "this build has device code and this machine has no"
                " accelerator to run it on, so there is only the host path"
            )

        var started = monotonic_ms()
        var g = Gguf(model_path)

        # The repack first and on its own, because the planar bytes have to
        # exist before the plan decides what to put on the card. A load that
        # writes the cache on the way past writes it from the file it is
        # reading, and the tensors this process already bound would still be
        # the ggml ones, so the device would get the layout its kernels cannot
        # read.
        var cache = open_cache(model_path, model_key(g))
        if not cache.usable:
            print("repack:    writing a planar cache first, " + cache.reason)
            var warm = load(
                g, plan_load(g, dev, 0, cache), 0, False, model_path
            )
            _ = warm^
            cache.close()
            cache = open_cache(model_path, model_key(g))
            if not cache.usable:
                raise Error(
                    "the repack cache was written and still cannot be used: "
                    + cache.reason
                )

        # One context for the process, made here and handed down. A second one
        # on CUDA constructs and then hangs on the first allocation against it,
        # so the load takes this rather than making its own.
        var ctx = DeviceContext(device_id=dev.index)
        var plan = plan_load(g, dev, -1, cache)
        if plan.left_behind > 0:
            raise Error(
                "the device forward pass needs every weight on the card and"
                " this plan leaves "
                + String(plan.left_behind)
                + " of them in the mapping, which would be read as zeros. This"
                " model does not fit on "
                + dev.name
            )
        var weights = load(g, plan^, 0, False, "", ctx)
        var loaded = monotonic_ms()

        # The same file bound twice. Once with the residency, which is what the
        # kernels read, and once without it, which is where the norm gains are
        # readable so they can be uploaded. A `Bound` owns no bytes, so the
        # second one is a list of addresses and not a second copy of anything.
        var b = bind(g, cache, weights.residency())
        var host = bind(g, cache)

        var geometry = read_geometry(g)
        var want = context if context > 0 else DEFAULT_CONTEXT
        if geometry.context_length > 0 and want > geometry.context_length:
            want = geometry.context_length

        var counter = AllocCounter()
        var tokenizer = Tokenizer(tokenizer_path, counter.raw())
        var session = Session()
        var ids = List[Int]()
        tokenizer.encode(prompt, True, session, ids)
        if len(ids) == 0:
            raise Error("the prompt encoded to no tokens")
        if len(ids) >= want:
            raise Error(
                "the prompt is "
                + String(len(ids))
                + " tokens and the context is "
                + String(want)
            )

        var eos = g.uint_or("tokenizer.ggml.eos_token_id", -1)
        var take = limit if limit > 0 else DEFAULT_LIMIT
        if take > want - len(ids):
            take = want - len(ids)

        var decode = DeviceSession(ctx, host, b, want)
        var sampler = Sampler(sampling, b.vocab())
        for i in range(len(ids)):
            sampler.observe(ids[i])
        report_header(
            g,
            b,
            want,
            decode.cache.bytes(),
            len(ids),
            sampling,
            loaded - started,
            cache,
            dev.name,
        )

        var prefill_started = monotonic_ms()
        decode.prefill(ids)
        var prefilled = monotonic_ms()

        var stream = DecodeStream(True)
        var written = 0
        for _ in range(take):
            var next = decode.pick(sampler)
            if next == eos:
                break
            print(stream.step(tokenizer, next), end="")
            written += 1
            decode.step(next)
        print()
        var finished = monotonic_ms()

        report_timing(
            prefilled - prefill_started,
            len(ids),
            finished - prefilled,
            written,
        )
        cache.close()
        g.close()
        _ = decode^
        _ = weights^
