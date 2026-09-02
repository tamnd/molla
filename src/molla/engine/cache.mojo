"""The keys and values one sequence has accumulated.

`molla.nn.attention` reads keys with position on the outside, so token `t`, head
`h`, element `d` sits at `t * kv_heads * head_dim + h * head_dim + d`, and
`molla.nn.model.forward` already takes one flat list per layer plus a slot to
write into. So this is not a new layout. It is the thing that owns those lists
and answers the three questions the forward pass deliberately does not: how long
each one is, which slot a position goes in, and what happens when the context
fills.

One sequence, contiguous, no paging. Paging is M3 and it changes the answer to
the middle question and nothing else, which is why that question is a method
here rather than an expression at the call site. Today `slot_for` returns the
position it was given. The day it stops doing that is the day every caller that
had inlined the identity becomes a bug, and there are no such callers because
there is a method.

Everything is float32. The keys and values feed a softmax and the accumulation
into it has to be wider than the weights are, and a cache in float16 is the
known cause of long context degradation that shows up as a model that is fine
for two thousand tokens and vague after eight. Half the memory for a defect
that only appears in the cases people bought the long context for is not a
trade this makes. Quantizing the cache is a real option later and it is a
measured one, not a default.
"""


struct KvCache(Movable):
    """One sequence's keys and values, one list per layer."""

    var keys: List[List[Float32]]
    """`layers` lists of `context * kv_width` floats."""

    var values: List[List[Float32]]
    """The same, and the same length, always."""

    var layers: Int
    var context: Int
    """How many positions there is room for. Not the model's trained context,
    which is an upper bound, but what this session asked for."""

    var kv_width: Int
    """`kv_heads * head_dim`, which is what one position occupies per layer."""

    var filled: Int
    """How many positions have been written. Also the next free slot, until
    something evicts."""

    def __init__(out self, layers: Int, context: Int, kv_width: Int) raises:
        """Allocate the whole thing up front.

        A cache that grows is a cache that reallocates in the middle of a
        decode, and a reallocation of several hundred megabytes between two
        tokens is a stall a user can see. The size is known the moment the
        context length is chosen, so it is taken then.
        """
        if layers <= 0:
            raise Error("a cache needs at least one layer")
        if context <= 0:
            raise Error("a cache needs room for at least one position")
        if kv_width <= 0:
            raise Error("a cache needs a positive key width")
        self.layers = layers
        self.context = context
        self.kv_width = kv_width
        self.filled = 0
        self.keys = List[List[Float32]]()
        self.values = List[List[Float32]]()
        var per = context * kv_width
        for _ in range(layers):
            var k = List[Float32]()
            var v = List[Float32]()
            for _ in range(per):
                k.append(0.0)
                v.append(0.0)
            self.keys.append(k^)
            self.values.append(v^)

    def bytes(self) -> Int:
        """What this occupies, which is worth reporting before allocating it."""
        return 2 * self.layers * self.context * self.kv_width * 4

    def reset(mut self):
        """Forget the sequence without giving back the memory.

        The stale floats are left where they are. Nothing reads past `filled`,
        and zeroing several hundred megabytes to make the unread bytes tidier
        is work done for an invariant that is already held elsewhere.
        """
        self.filled = 0

    def slot_for(self, pos: Int) raises -> Int:
        """Where the key for `pos` goes.

        The identity, for now. See the module docstring for why it is a method.
        """
        if pos < 0:
            raise Error("a position cannot be negative")
        if pos >= self.context:
            raise Error(
                "position "
                + String(pos)
                + " is past the end of a cache with room for "
                + String(self.context)
            )
        return pos

    def room(self) -> Int:
        return self.context - self.filled

    def reserve(mut self, count: Int) raises:
        """Refuse a sequence that will not fit, before any of it is computed.

        A refusal and not a wrap. A cache that fills and starts overwriting slot
        zero gives a model that is still fluent and has forgotten its
        instructions, which is worse than an error because nobody reads it as
        one. What to drop when the context is full is a policy, it belongs with
        paging in M3, and until there is one the honest answer is that this
        sequence does not fit.
        """
        if count < 0:
            raise Error("cannot reserve a negative number of positions")
        if count > self.room():
            raise Error(
                "this sequence needs "
                + String(count)
                + " more positions and the cache has "
                + String(self.room())
                + " left of "
                + String(self.context)
            )

    def advance(mut self, count: Int = 1) raises:
        """Record that `count` more positions have been written."""
        self.reserve(count)
        self.filled += count
