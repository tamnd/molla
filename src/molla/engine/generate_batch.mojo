"""Many streams of the same prompt at once, and what that costs per stream.

`molla generate` answers one prompt and reports how fast it did it. This answers
several at once and reports the numbers a shared server is judged on, which are
not the same numbers. One stream's tokens a second says how fast the card is for
one person. What matters when sixteen people share it is whether the total goes
up as they arrive and whether the slowest of them is being starved by the
fastest, and neither of those can be read off a single stream run.

So this prints three things a single run has no version of. Aggregate decode
tokens a second, which is what the card is actually producing. The spread of
inter token latency across the streams, which is what says whether one of them is
being starved. And time to first token per stream, which is where a long prompt
admitted first shows up as a wait for everyone behind it.

Every stream gets the same prompt on purpose. Different prompts would make the
per stream numbers differ for a reason that is about the prompts and not about
the scheduler, and the question here is the scheduler. Prefix caching, which
would make identical prompts share their work, is #33 and is not in yet, so the
streams really do each compute their own prefill.
"""

from std.sys.info import has_accelerator

from molla.engine.backend import Backend
from molla.engine.batch import DeviceBatch, open_batch
from molla.engine.bind import bind
from molla.engine.device import device_context, load_on_device
from molla.engine.generate import DEFAULT_CONTEXT, DEFAULT_LIMIT, report_header
from molla.engine.sample import SamplerConfig
from molla.model.gguf import Gguf
from molla.model.repack import model_key, open_cache
from molla.model.spec import read_geometry
from molla.nn.gpu import PREFILL_CHUNK
from molla.nn.repack import CACHE_F16, cache_type_name
from molla.sys.clock import monotonic_ms
from molla.sys.mem import AllocCounter
from molla.tokenizer.tokenizer import DecodeStream, Session, Tokenizer


def run_generate_batch(
    model_path: String,
    tokenizer_path: String,
    prompt: String,
    streams: Int,
    limit: Int,
    context: Int,
    cap: Int = PREFILL_CHUNK,
    sampling: SamplerConfig = SamplerConfig(),
    backend: Backend = Backend(),
    form: Int = CACHE_F16,
) raises:
    """Load, admit `streams` copies of the prompt, and step until they are done.

    The pool is sized for all of them together rather than per stream, which is
    the point: a context of four thousand shared by sixteen streams is what a
    server has, and a stream that does not fit is refused at admission rather
    than discovered halfway through.
    """
    sampling.check()

    comptime if not has_accelerator():
        raise Error(
            "this build has no device code in it, so there is nothing to batch"
            " on. Accelerator support is decided when molla is compiled, not"
            " when it is run"
        )

    comptime if has_accelerator():
        var dev = backend.device
        if not dev.accelerator():
            raise Error(
                "this build has device code and this machine has no"
                " accelerator to run it on, so there is only the host path"
            )
        if streams < 1:
            raise Error("a batch run needs at least one stream")

        var started = monotonic_ms()
        var g = Gguf(model_path)
        var cache = open_cache(model_path, model_key(g))
        var ctx = device_context(dev.index)
        var weights = load_on_device(
            g, cache, model_path, dev, ctx, String("repack:   ")
        )
        var loaded = monotonic_ms()

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

        var take = limit if limit > 0 else DEFAULT_LIMIT
        var each = len(ids) + take
        if each * streams > want:
            raise Error(
                "each of the "
                + String(streams)
                + " streams needs "
                + String(each)
                + " positions and the pool holds "
                + String(want)
                + ", so this would be refused at admission. Ask for a larger"
                " context, fewer streams, or fewer tokens"
            )

        var eos = g.uint_or("tokenizer.ggml.eos_token_id", -1)
        var opened = open_batch(ctx, host, b, want, streams, cap, form)
        if not opened:
            raise Error("this build has no device code in it")
        var batch = opened.take()
        report_header(
            g,
            b,
            want,
            batch.cache.bytes(),
            len(ids),
            sampling,
            loaded - started,
            cache,
            backend,
            cache_type_name(form),
        )
        print("streams:  ", streams, "of", take, "tokens each")
        print("cap:      ", batch.cap, "tokens a step")
        print()

        cache.close()
        g.close()

        for _ in range(streams):
            _ = batch.admit(ids.copy(), take, eos, sampling)

        # A stamp per stream per token, taken after the step that produced it.
        # Held rather than summarised as it goes, because the interesting number
        # is a percentile and a percentile needs the samples.
        var seen = List[Int](length=streams, fill=0)
        var first = List[Int](length=streams, fill=0)
        var gaps = List[Int]()
        var last = List[Int](length=streams, fill=0)

        var begun = monotonic_ms()
        for i in range(streams):
            last[i] = begun
        var steps = 0
        var carried = 0
        while True:
            var n = batch.step()
            if n == 0:
                break
            steps += 1
            carried += n
            var now = monotonic_ms()
            for i in range(streams):
                var have = batch.produced(i)
                if have == seen[i]:
                    continue
                if seen[i] == 0:
                    first[i] = now - begun
                else:
                    gaps.append(now - last[i])
                seen[i] = have
                last[i] = now
        var ended = monotonic_ms()

        var total = 0
        for i in range(streams):
            total += seen[i]
        var elapsed = ended - begun
        print("steps:    ", steps, "carrying", carried, "tokens")
        print("wall:     ", elapsed, "ms for", total, "tokens")
        if elapsed > 0:
            print("aggregate:", total * 1000 // elapsed, "tokens/s")

        # Time to first token, which is where a stream that waited behind
        # somebody else's prompt shows up.
        var soonest = first[0]
        var latest = first[0]
        for i in range(streams):
            if first[i] < soonest:
                soonest = first[i]
            if first[i] > latest:
                latest = first[i]
        print("ttft:     ", soonest, "ms fastest,", latest, "ms slowest")

        # And the spread of inter token latency, which is the starvation
        # measure. Sorted rather than averaged, because a mean hides exactly the
        # case this is looking for: one stream served late while fifteen are
        # served on time.
        if len(gaps) > 0:
            sort(gaps)
            var mid = gaps[len(gaps) // 2]
            var p95 = gaps[(len(gaps) * 95) // 100]
            print(
                "latency:  ",
                mid,
                "ms median,",
                p95,
                "ms p95,",
                gaps[len(gaps) - 1],
                "ms worst",
            )

        var short = 0
        for i in range(streams):
            if seen[i] < take:
                short += 1
        if short > 0:
            print("stopped:  ", short, "streams ended before their limit")

        # Every stream had the same prompt and the same greedy settings, so
        # every stream has to have written the same tokens. That is not a
        # property of the model, it is a property of the pool: two sequences
        # whose cells got mixed up would disagree here, and a batch that wrote
        # one stream's logits into another's row would too. So it is checked
        # rather than left to whoever reads the text.
        var agreed = 0
        var alone = batch.output(0)
        for i in range(1, streams):
            var mine = batch.output(i)
            var same = len(mine) == len(alone)
            for k in range(len(mine)):
                if k >= len(alone) or mine[k] != alone[k]:
                    same = False
            if same:
                agreed += 1
        if streams > 1:
            print(
                "agreement:",
                agreed + 1,
                "of",
                streams,
                "streams wrote the same tokens",
            )
        print()
        var text = String("")
        var out = DecodeStream(True)
        for k in range(len(alone)):
            text += out.step(tokenizer, alone[k])
        print(prompt + text)
        _ = batch^
        _ = weights^


def sort(mut xs: List[Int]):
    """Insertion sort, because the list is a few thousand at most.

    Called once at the end of a run to take two percentiles off it, so the
    difference between this and something cleverer is microseconds against a run
    measured in seconds.
    """
    for i in range(1, len(xs)):
        var v = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > v:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = v
