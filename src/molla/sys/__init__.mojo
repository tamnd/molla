"""Everything that talks to libc.

D1 in docs/design.md limits foreign code to the C ABI in three modules, and this
is the first of them. If a call to `external_call` appears anywhere outside
`molla.sys`, `molla.tls`, or `molla.device`, that is a design violation and not
a shortcut.
"""
