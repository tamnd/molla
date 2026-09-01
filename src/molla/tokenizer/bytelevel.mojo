"""The byte level alphabet.

GPT-2 had a problem: BPE over raw bytes puts control characters and invalid
UTF-8 in the vocabulary, and a vocabulary is a JSON file full of strings. The
fix was to pick 256 printable code points, one per byte value, so every byte
has a spelling that survives a round trip through JSON and through a text
editor. That is why a Qwen vocabulary is full of things like `Ġthe`: the G with
a stroke is byte 32.

The rule is that the bytes which are already printable ASCII or printable
Latin-1 spell themselves, and the other 68 get consecutive code points starting
at U+0100. Both directions are built once here, because the pre-tokenizer needs
one and the decoder needs the other, and a table that disagrees with itself
turns a space into a mystery character halfway down a stream.
"""


def _is_printable(b: Int) -> Bool:
    if b >= 0x21 and b <= 0x7E:
        return True
    if b >= 0xA1 and b <= 0xAC:
        return True
    return b >= 0xAE and b <= 0xFF


def byte_to_point() -> List[Int]:
    """256 entries, byte value to the code point that spells it."""
    var table = List[Int](length=256, fill=0)
    var extra = 0
    for b in range(256):
        if _is_printable(b):
            table[b] = b
        else:
            table[b] = 256 + extra
            extra += 1
    return table^


def point_to_byte() -> List[Int]:
    """The inverse, as a list long enough to index by code point directly.

    The highest code point in the alphabet is U+0143, so 324 entries covers it,
    and anything outside the alphabet reads back as -1.
    """
    var forward = byte_to_point()
    var table = List[Int](length=324, fill=-1)
    for b in range(256):
        table[forward[b]] = b
    return table^
