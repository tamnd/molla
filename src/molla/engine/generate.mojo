"""The command that turns a file and a prompt into text.

Everything below this is checked against something. The quantization is checked
against llama.cpp's own decode, the kernels against dequantizing and doing it
the slow way, the layer against a second implementation, and the cache against
running the same tokens by the other route. None of that says the whole thing
produces English, because a stack of correct pieces assembled in a plausible
order is exactly the failure this project keeps designing against.

So this exists to be run by a person, on a real file, and read. It is the first
thing in molla whose output is judged rather than asserted, and it stays that
way until #30 puts the logits beside llama.cpp's and compares numbers.

The tokenizer comes from a `tokenizer.json` beside the model rather than out of
the GGUF. The vocabulary and merges are in the file's metadata and reading them
is a real piece of work with its own failure modes, and doing it here as a side
errand of the decode loop is how a tokenizer ends up with no oracle behind it.
`molla.tokenizer` already has one, and it takes a path.
"""

from molla.engine.bind import bind
from molla.engine.session import Session as Decode
from molla.model.gguf import Gguf
from molla.model.load import load, plan_load
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


def run_generate(
    model_path: String,
    tokenizer_path: String,
    prompt: String,
    limit: Int,
    context: Int,
) raises:
    """Load, prefill, decode, and print as it goes.

    Printing per token rather than at the end, and through a `DecodeStream`
    rather than by decoding each id on its own. A token is bytes and one
    character can be spread across three of them, so decoding them one at a
    time prints a replacement character in the middle of any word the tokenizer
    split somewhere unexpected.
    """
    var started = monotonic_ms()
    var g = Gguf(model_path)
    var dev = default_device()

    # Everything stays in the mapping. The kernels are host kernels, so a
    # tensor copied to a card is a tensor they cannot read, and a budget of
    # zero says so rather than leaving it to a placement heuristic that has no
    # way to know what will read the result.
    var weights = load(g, plan_load(g, dev, 0), 0, False)
    var loaded = monotonic_ms()

    var b = bind(g)
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
    print("load:     ", loaded - started, "ms")
    print()

    var prefill_started = monotonic_ms()
    decode.prefill(b, ids)
    var prefilled = monotonic_ms()

    var stream = DecodeStream(True)
    var written = 0
    for _ in range(take):
        var next = decode.pick()
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
    g.close()
    _ = weights^
