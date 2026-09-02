"""Attention over a span of keys and values that somebody else is holding.

This does the arithmetic and owns nothing. It is handed a query, a run of keys
and a run of values, and it writes one vector per head. Where those keys came
from, how long they are kept and when they get evicted is a cache, and a cache
is #27. Separating the two means the arithmetic can be checked on twenty tokens
built by hand instead of only through a model.

Keys and values are laid out with position on the outside, so token `t`, head
`h`, element `d` is at `t * kv_heads * head_dim + h * head_dim + d`. That is the
order a decode step appends in, one contiguous write per token, and it is the
order a decode step reads in, walking forward through positions for each head.
ggml stores K transposed instead, which is better for a batched prefill and
worse for this, and picking the other one later is a repacking question and not
a correctness one.

Grouped query attention is not a separate path. Every model here has `heads`
query heads and `kv_heads` key heads with the first a multiple of the second,
and head `h` reads key head `h / (heads / kv_heads)`. Multi head is that with
the two equal and multi query is that with `kv_heads` at one, so there is one
loop and no modes.

Four things get bolted onto the softmax by one architecture or another and all
four are here because leaving them out means the block code grows a branch that
should have lived in one place:

A sliding window. Mistral, Gemma and Qwen with long context alternate layers
that see everything with layers that see the last few thousand tokens. Zero
means no window.

Sink tokens. A window that cuts off the first few positions costs much more than
it should, because a transformer parks attention it does not need on the first
token or two. Keeping those visible regardless of the window is most of the
difference. Zero means none.

Logit softcapping. Gemma 2 squashes scores through `cap * tanh(score / cap)`
before the softmax, which keeps one enormous score from taking the whole row.
Zero means off, and off is not the same as a very large cap because `tanh` of a
tiny number is not exactly that number.

A scale that is not one over the root of the head dimension. Almost every model
uses that, and the ones that do not are not wrong, they were trained that way.
It is a field rather than a computation for that reason.
"""

from std.math import sqrt, tanh

from molla.nn.kernel import softmax
from molla.nn.tensor import Buffer


struct AttnSpec(Copyable, ImplicitlyCopyable, Movable):
    """The shape of one attention layer."""

    var heads: Int
    """Query heads."""

    var kv_heads: Int
    """Key and value heads. Equal to `heads` for multi head, one for multi
    query, a divisor of it for grouped query."""

    var head_dim: Int
    """Elements per head. Not always the model width over the head count, since
    a few models size their heads independently of their embedding."""

    var scale: Float32
    """What a raw score is multiplied by before the softmax."""

    var window: Int
    """How many positions back a query can see, counting itself. Zero is no
    limit."""

    var sinks: Int
    """How many positions at the very start stay visible whatever the window
    says."""

    var softcap: Float32
    """The cap for `cap * tanh(score / cap)`. Zero is off."""

    def __init__(out self, heads: Int, kv_heads: Int, head_dim: Int) raises:
        """The ordinary case: no window, no sinks, no cap, and the usual
        scale."""
        if heads <= 0 or kv_heads <= 0 or head_dim <= 0:
            raise Error("an attention layer needs positive head counts")
        if heads % kv_heads != 0:
            raise Error(
                "query heads ("
                + String(heads)
                + ") have to be a whole multiple of key heads ("
                + String(kv_heads)
                + ")"
            )
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.scale = Float32(1.0 / sqrt(Float64(head_dim)))
        self.window = 0
        self.sinks = 0
        self.softcap = 0.0

    def group(self) -> Int:
        """How many query heads share one key head."""
        return self.heads // self.kv_heads

    def kv_head_of(self, head: Int) -> Int:
        return head // self.group()

    def visible(self, at: Int, pos: Int) -> Bool:
        """Whether a query at `pos` can see the key at `at`.

        Causality is not checked here. The caller passes the keys that exist,
        and a key for a position after the current one does not exist yet in a
        decode and is masked by the caller in a prefill. Putting the check here
        as well would be a second place to get it wrong.
        """
        if at < self.sinks:
            return True
        if self.window <= 0:
            return True
        return at > pos - self.window


def attend(
    spec: AttnSpec,
    q: Buffer,
    keys: List[Float32],
    values: List[Float32],
    count: Int,
    pos: Int,
    mut out: Buffer,
    mut scores: List[Float32],
) raises:
    """One query against `count` keys, writing `heads * head_dim` values.

    `scores` is scratch, at least `count` long, and is passed in rather than
    allocated because a decode calls this once per layer per token and a
    thousand element allocation per call is a thousand element allocation per
    call. It is left holding the last head's probabilities on the way out, which
    is useful when something looks wrong and is not part of the contract.
    """
    var width = spec.heads * spec.head_dim
    if q.elements() < width:
        raise Error(
            "attention wants a query of "
            + String(width)
            + " but got "
            + String(q.elements())
        )
    if out.elements() < width:
        raise Error(
            "attention wants an output of "
            + String(width)
            + " but got "
            + String(out.elements())
        )
    if count <= 0:
        raise Error("attention needs at least one key to look at")
    var kv_width = spec.kv_heads * spec.head_dim
    if len(keys) < count * kv_width or len(values) < count * kv_width:
        raise Error(
            "attention wants "
            + String(count * kv_width)
            + " keys and values but got "
            + String(len(keys))
            + " and "
            + String(len(values))
        )
    if len(scores) < count:
        raise Error(
            "attention wants scratch for "
            + String(count)
            + " scores but got "
            + String(len(scores))
        )

    for h in range(spec.heads):
        var kvh = spec.kv_head_of(h)
        var qa = h * spec.head_dim
        var seen = 0

        for t in range(count):
            if not spec.visible(t, pos):
                continue
            var ka = t * kv_width + kvh * spec.head_dim
            var s = Float32(0)
            for d in range(spec.head_dim):
                s += q.data[qa + d] * keys[ka + d]
            s *= spec.scale
            if spec.softcap > 0:
                s = spec.softcap * Float32(
                    tanh(Float64(s) / Float64(spec.softcap))
                )
            # Written packed rather than at index t, so the softmax runs over
            # the keys that survived the mask instead of over a row full of
            # negative infinities that all have to be exponentiated anyway.
            scores[seen] = s
            seen += 1

        if seen == 0:
            raise Error(
                "position "
                + String(pos)
                + " can see no keys at all, which a window of "
                + String(spec.window)
                + " with "
                + String(spec.sinks)
                + " sinks should never produce"
            )

        softmax(scores, 0, seen)

        for d in range(spec.head_dim):
            out.data[qa + d] = 0.0
        var slot = 0
        for t in range(count):
            if not spec.visible(t, pos):
                continue
            var p = scores[slot]
            slot += 1
            var va = t * kv_width + kvh * spec.head_dim
            for d in range(spec.head_dim):
                out.data[qa + d] += p * values[va + d]
