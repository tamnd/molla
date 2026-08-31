"""Null terminated buffers for the C boundary.

Mojo strings are not null terminated, so every libc call that takes a `const
char *` needs a copy. The copy has to outlive the call, which is why these
return an owning `List` rather than a pointer. Taking `unsafe_ptr` of a
temporary compiles and then dangles.
"""


def c_string(text: StringSpan) -> List[UInt8]:
    """Copy `text` into a null terminated byte buffer."""
    var out = List[UInt8]()
    out.reserve(text.byte_length() + 1)
    var p = text.unsafe_ptr()
    for i in range(text.byte_length()):
        out.append(p.unsafe_load(i))
    out.append(0)
    return out^


def from_c_string(p: Pointer[UInt8, _], limit: Int) -> String:
    """Read a null terminated C string back into a Mojo string.

    `limit` caps how far it will walk. Every caller here is reading a string
    some library allocated, and a missing terminator should be a truncated
    error message rather than a walk off the end of the heap.
    """
    var out = List[UInt8]()
    for i in range(limit):
        var b = p.unsafe_load(i)
        if b == 0:
            break
        out.append(b)
    return String(StringSpan(unsafe_from_utf8=out))
