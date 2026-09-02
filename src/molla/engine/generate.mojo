"""The command that turns a file and a prompt into text.

Everything below this is checked against something. The quantization is checked
against llama.cpp's own decode, the kernels against dequantizing and doing it
the slow way, the layer against a second implementation, and the cache against
running the same tokens by the other route. None of that says the whole thing
produces English, because a stack of correct pieces assembled in a plausible
order is exactly the failure this project keeps designing against.

So this exists to be run by a person, on a real file, and read. What it prints
is still judged rather than asserted, but the numbers behind it are not: #30
puts the whole forward pass beside llama.cpp's, layer by layer, in
`scripts/logit_oracle.mojo`.

The tokenizer comes from a `tokenizer.json` beside the model rather than out of
the GGUF. The vocabulary and merges are in the file's metadata and reading them
is a real piece of work with its own failure modes, and doing it here as a side
errand of the decode loop is how a tokenizer ends up with no oracle behind it.
`molla.tokenizer` already has one, and it takes a path.
"""

from molla.engine.bind import bind
from molla.engine.sample import Sampler, SamplerConfig
from molla.engine.session import Session as Decode
from molla.model.gguf import Gguf
from molla.model.load import load, plan_load
from molla.model.repack import model_key, open_cache
from molla.model.spec import read_geometry
from molla.sys.clock import monotonic_ms
from molla.sys.device import default_device
from molla.sys.mem import AllocCounter
from molla.tokenizer.tokenizer import DecodeStream, Session, Tokenizer

comptime DEFAULT_LIMIT = 128
"""Tokens to generate when nobody says. Long enough to tell coherent from
fluent, short enough that a scalar decode finishes while somebody watches."""

comptime DEFAULT_CONTEXT = 2048
"""Positions to make room for. The cache is four bytes per element and a
model's trained context can be a hundred and thirty thousand of them, so
allocating what the file allows would be gigabytes for a prompt of nine
words."""


def describe(c: SamplerConfig) -> String:
    """What the sampler was asked for, in one line.

    Printed with the model and the context because a run that reads oddly is
    the first thing anybody argues about, and the argument is shorter when the
    settings that produced it are in the same output as the text.
    """
    if c.greedy():
        return String("greedy")
    var out = String("temp ") + String(c.temperature)
    if c.top_k > 0:
        out += ", top-k " + String(c.top_k)
    if c.top_p < 1.0:
        out += ", top-p " + String(c.top_p)
    if c.min_p > 0:
        out += ", min-p " + String(c.min_p)
    if c.typical_p < 1.0:
        out += ", typical " + String(c.typical_p)
    if c.penalizing():
        out += ", penalties over " + String(c.repeat_last_n)
    out += ", seed " + String(c.seed)
    return out


def run_generate(
    model_path: String,
    tokenizer_path: String,
    prompt: String,
    limit: Int,
    context: Int,
    sampling: SamplerConfig = SamplerConfig(),
) raises:
    """Load, prefill, decode, and print as it goes.

    Printing per token rather than at the end, and through a `DecodeStream`
    rather than by decoding each id on its own. A token is bytes and one
    character can be spread across three of them, so decoding them one at a
    time prints a replacement character in the middle of any word the tokenizer
    split somewhere unexpected.

    The default sampling is greedy, so running this with no flags gives the
    same tokens every time and a run that reads badly is the model or the
    kernels rather than a draw that went somewhere unlikely.
    """
    # Before the file is opened. The sampler checks its own settings when it is
    # built, but that happens after the weights are mapped, and telling
    # somebody their top-p is out of range only once an 8B has finished loading
    # is a slow way to report a typo.
    sampling.check()

    var started = monotonic_ms()
    var g = Gguf(model_path)
    var dev = default_device()

    # Everything stays in the mapping. The kernels are host kernels, so a
    # tensor copied to a card is a tensor they cannot read, and a budget of
    # zero says so rather than leaving it to a placement heuristic that has no
    # way to know what will read the result. The budget stops being zero when
    # there is a device forward pass to read the result, which is #143.
    #
    # A hit binds to the repacked weights and a miss binds to the file and
    # writes the repack on the way past, so the first run against a model is
    # the slow one and says so. The cache goes to the plan as well as to the
    # binder, so the read stage warms the copy of each weight that the kernels
    # are going to read rather than the one in the file beside it.
    var cache = open_cache(model_path, model_key(g))
    var repack_for = String("") if cache.usable else model_path
    var weights = load(g, plan_load(g, dev, 0, cache), 0, False, repack_for)
    var loaded = monotonic_ms()

    var b = bind(g, cache)
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

    var decode = Decode(b, want)
    var sampler = Sampler(sampling, b.vocab())
    for i in range(len(ids)):
        sampler.observe(ids[i])
    print(
        "model:    ",
        g.architecture(),
        b.block_count(),
        "layers,",
        b.width(),
        "wide",
    )
    print(
        "context:  ",
        want,
        "positions,",
        decode.cache.bytes() // (1 << 20),
        "MiB of cache",
    )
    print("prompt:   ", len(ids), "tokens")
    print("sampling: ", describe(sampling))
    print("load:     ", loaded - started, "ms")
    if cache.usable:
        print(
            "repack:   ",
            cache.count(),
            "tensors from cache,",
            cache.bytes() // (1 << 20),
            "MiB",
        )
    else:
        print("repack:   ", cache.reason)
    print()

    var prefill_started = monotonic_ms()
    decode.prefill(b, ids)
    var prefilled = monotonic_ms()

    var stream = DecodeStream(True)
    var written = 0
    for _ in range(take):
        var next = decode.pick(sampler)
        if next == eos:
            break
        print(stream.step(tokenizer, next), end="")
        written += 1
        decode.step(b, next)
    print()
    var finished = monotonic_ms()

    print()
    print(
        "prefill:  ", prefilled - prefill_started, "ms for", len(ids), "tokens"
    )
    print("decode:   ", finished - prefilled, "ms for", written, "tokens")
    if written > 0:
        # Milliseconds per token rather than tokens per second, because a
        # scalar decode of an 8B is five seconds a token and a rate in whole
        # tokens per second prints zero.
        print("rate:     ", (finished - prefilled) // written, "ms/token")
    cache.close()
    g.close()
    _ = weights^
