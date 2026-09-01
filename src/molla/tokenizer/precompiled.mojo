"""The SentencePiece charsmap, which `tokenizer.json` calls `Precompiled`.

Sixty of the three hundred and thirty eight files in the conformance corpus
carry one, and they are not a long tail. It is T5, mT5, XLM-R, ALBERT,
DeBERTa-v3 and NLLB, which is most of the multilingual encoders anybody runs.

What is in the file is a base64 blob that SentencePiece produced when the model
was trained. Inside it is a little endian length, then a double array trie whose
keys are UTF-8 byte sequences, then a table of replacement strings laid end to
end and separated by NUL bytes. A rule is a key in the trie whose value is an
offset into that table. Normalizing means looking text up in the trie and
writing out what it finds.

The rules are a compiled NFKC variant plus whatever else the training script
asked for, so the result is close to NFKC and not equal to it. Approximating it
with NFKC gives ids that are wrong in a way nothing reports, which is why molla
refused these files rather than loading them without this.

Two details decide whether the answer matches, and neither is obvious.

The first is that lookup happens per grapheme cluster, not per character and
not over the whole string. A cluster shorter than six bytes is looked up whole,
which is what turns a letter and a combining acute into one precomposed letter,
and anything else is looked up one character at a time. Six is the reference
implementation's number and there is no principle behind it.

The second is that the trie search returns the shortest key that is a prefix of
what it was given rather than the longest. SentencePiece itself takes the
longest. The Rust port that Hugging Face `tokenizers` uses takes the first
result the search produced, which is the shortest, and since the corpus is
checked against that port, so does this. It is visible: a subscript i with a
combining breve after it comes back as a plain i, because the rule for the
subscript matched first and the breve went with it.
"""

from molla.text.graphemes import cluster_ends
from molla.text.props import Unicode
from molla.text.utf8 import decode, encode

comptime WHOLE_CLUSTER_LIMIT = 6
"""Clusters this long or longer are normalized one character at a time.

The reference implementation writes `if grapheme.len() < 6`, in bytes, and that
is the whole of the reason. It means the rules that build a flag or a family
emoji out of several characters can never fire, and it means an accent on a
Latin letter always can.
"""


def _base64(text: Span[UInt8, _]) raises -> List[UInt8]:
    """The standard alphabet, padding optional, nothing else allowed.

    A charsmap is a quarter of a megabyte of base64 sitting in the middle of a
    `tokenizer.json`, and a file that has been truncated or edited by hand is a
    file whose trie will index off the end of itself. Rejecting it here is
    cheaper than checking every step of the walk.
    """
    var out = List[UInt8]()
    out.reserve(len(text) // 4 * 3)
    var accumulator = 0
    var bits = 0
    for i in range(len(text)):
        var b = text[i]
        if b == 61:
            break
        var digit: Int
        if b >= 65 and b <= 90:
            digit = Int(b) - 65
        elif b >= 97 and b <= 122:
            digit = Int(b) - 97 + 26
        elif b >= 48 and b <= 57:
            digit = Int(b) - 48 + 52
        elif b == 43:
            digit = 62
        elif b == 47:
            digit = 63
        else:
            raise Error("the charsmap is not base64")
        accumulator = (accumulator << 6) | digit
        bits += 6
        if bits >= 8:
            bits -= 8
            out.append(UInt8((accumulator >> bits) & 0xFF))
    return out^


struct Precompiled(Copyable, Movable):
    """One charsmap, ready to normalize with.

    The trie is a flat array of thirty two bit units and the walk over it is
    the one darts-clone does: exclusive or the node position with the byte,
    read the unit there, check the label, then exclusive or with the offset the
    unit carries. A unit that has a leaf holds the rule's value one step on.
    """

    var units: List[UInt32]
    var replacements: List[UInt8]
    """Every replacement string, NUL separated, indexed by a rule's value."""

    def __init__(out self, encoded: Span[UInt8, _]) raises:
        var blob = _base64(encoded)
        if len(blob) < 4:
            raise Error("the charsmap is too short to hold a trie")
        var trie_bytes = (
            Int(blob[0])
            | (Int(blob[1]) << 8)
            | (Int(blob[2]) << 16)
            | (Int(blob[3]) << 24)
        )
        if trie_bytes < 4 or trie_bytes % 4 != 0:
            raise Error("the charsmap trie has a length that cannot be right")
        if 4 + trie_bytes > len(blob):
            raise Error("the charsmap trie runs past the end of the blob")

        self.units = List[UInt32]()
        self.units.reserve(trie_bytes // 4)
        for i in range(trie_bytes // 4):
            var at = 4 + i * 4
            self.units.append(
                UInt32(blob[at])
                | (UInt32(blob[at + 1]) << 8)
                | (UInt32(blob[at + 2]) << 16)
                | (UInt32(blob[at + 3]) << 24)
            )
        self.replacements = List[UInt8]()
        self.replacements.reserve(len(blob) - 4 - trie_bytes)
        for i in range(4 + trie_bytes, len(blob)):
            self.replacements.append(blob[i])

    def _lookup(self, key: List[UInt8]) -> Int:
        """The value of the shortest rule that is a prefix of `key`, or -1.

        A NUL byte ends the search without matching, because the replacement
        table is NUL separated and a key containing one could not have been
        stored. The bounds check on `at` is not in the reference, which trusts
        its blob. This one does not: the blob came out of a file downloaded
        from a model repository.
        """
        if len(self.units) == 0:
            return -1
        var unit = Int(self.units[0])
        var at = (unit >> 10) << ((unit & 0x200) >> 6)
        for i in range(len(key)):
            var byte = Int(key[i])
            if byte == 0:
                return -1
            at ^= byte
            if at < 0 or at >= len(self.units):
                return -1
            unit = Int(self.units[at])
            if (unit & 0x800000FF) != byte:
                return -1
            at ^= (unit >> 10) << ((unit & 0x200) >> 6)
            if ((unit >> 8) & 1) == 1:
                if at < 0 or at >= len(self.units):
                    return -1
                return Int(self.units[at]) & 0x7FFFFFFF
        return -1

    def _write(self, at: Int, mut out: List[Int]):
        """Append the replacement string that starts at `at`.

        It can be empty, and an empty one means the rule deletes what it
        matched. Several of the format characters in every charsmap do exactly
        that.
        """
        var end = at
        while end < len(self.replacements) and self.replacements[end] != 0:
            end += 1
        var span = Span(self.replacements)[at:end]
        var i = 0
        while i < len(span):
            var one = decode(span, i)
            if one.code < 0:
                i += 1
                continue
            out.append(one.code)
            i += one.width

    def apply(self, tables: Unicode, points: List[Int]) -> List[Int]:
        """The whole string, cluster by cluster."""
        var out = List[Int]()
        out.reserve(len(points))
        var ends = cluster_ends(tables, points)
        var key = List[UInt8]()
        var start = 0
        for c in range(len(ends)):
            var stop = ends[c]
            key.clear()
            for i in range(start, stop):
                encode(points[i], key)
            if len(key) < WHOLE_CLUSTER_LIMIT:
                var whole = self._lookup(key)
                if whole >= 0:
                    self._write(whole, out)
                    start = stop
                    continue
            for i in range(start, stop):
                key.clear()
                encode(points[i], key)
                var one = self._lookup(key)
                if one >= 0:
                    self._write(one, out)
                else:
                    out.append(points[i])
            start = stop
        return out^
