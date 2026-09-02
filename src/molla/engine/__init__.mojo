"""What a model does, as opposed to what it is.

`molla.model` knows what a file says and `molla.nn` knows the arithmetic. This
package is the part that holds state between tokens: the keys and values a
sequence has accumulated, the position it is at, and the loop that turns a
prompt into more of itself.

The split is worth stating because it is the reason `molla.nn.forward` takes a
slot and a position as two arguments and owns neither. Arithmetic that owns no
state can be checked against weights small enough to work out by hand. A cache
that holds state and does no arithmetic can be checked by asking whether two
routes to the same sequence leave the same bytes behind. Together they are a
model, and apart they are two things that can each be wrong in a way somebody
notices.
"""
