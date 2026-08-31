"""JSON for molla, in two modes over one scanner.

`scan` finds the bytes a parser has to decide about and skips the rest at vector
width. `reader` turns those into events, which is the mode a request body is
parsed in, straight into a typed struct with the strings left as spans into the
read buffer. `dom` builds a tree from the same events, for config and manifests
and anything else read more than once. `number` converts both ways with no
strtod and no locale, and `decimal` is the exact arithmetic underneath it.
`serialize` writes, in the order the caller wrote, because a tool call's
arguments are an object whose key order came from the model.
"""
