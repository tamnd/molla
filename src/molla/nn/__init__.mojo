"""The numeric layer: blocks in a file, tensors in memory, arithmetic on them.

This sits between `molla.model`, which knows what a file says, and the engine,
which knows what a model does. Nothing here holds state that outlives a call.
It knows how a quantized block is laid out, what a tensor's shape and stride
are, how to multiply two of them, where a position goes, how a query is scored
against the keys before it, what one layer is made of, and what each
architecture does differently. What those keys are kept in and how long they
live is the engine's problem, and so is what to do with the logits that come
out of the top.

The rule for this package is that there is always a slow version and it is
always kept. A fused kernel that reads packed bytes and accumulates in one pass
is the only thing fast enough to run a model, and it is also the easiest place
in the project to be wrong in a way that produces sensible looking output. So
the obvious implementation stays, it is checked against an outside oracle, and
the fast one is checked against it.
"""
