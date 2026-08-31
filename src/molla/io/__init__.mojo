"""Memory. Buffers, spans, rings and arenas.

This is the layer that makes writing the edge in Mojo worth the trouble. A
parsed header is a span into the connection's read buffer, the compiler proves
the span does not outlive the buffer, and nothing is copied to make that safe.
In C the same design works and nothing checks it. In a garbage collected
language the same design does not work at all.

Three tiers of lifetime, and every allocation in molla belongs to exactly one of
them.

An arena is per request. It is reset when the response is finished, in constant
time, no matter how many things were put in it. Anything whose lifetime is one
request goes here.

A pool is per connection. Read and write buffers live as long as the socket and
get reused across every request on it, which is the whole point of keep alive.

The heap is for long lived things: model weights, the KV cache, the routing
table. These are allocated once at startup and never on the request path.

Model weights and caches never use arenas. An arena that holds a gigabyte is not
an arena, it is a leak with a nicer name.

Every type here takes a counter address and records what it allocates against
it, which is what makes "this request allocated nothing" a number rather than a
claim. The counter has to outlive every buffer, ring and arena that holds its
address, and Mojo destroys a value at its last use rather than at the end of the
scope, so the counter is closed by hand with `AllocCounter.close` after the
things it counts are gone. `molla.sys.mem` says more about why.

Nothing in here calls `external_call`. The allocation goes through
`molla.sys.mem`, because D1 says the C ABI lives in three modules and this is
not one of them.
"""
