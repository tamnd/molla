"""The decoder stage.

Decoding is not encoding run backwards. The vocabulary maps an id to a piece of
text, but that text has been through the pre-tokenizer, so it might be `Ġthe`
or `##ing` or `<0xE2>`, and turning a list of those back into a sentence takes
the same number of decisions that producing them did.

Every step works on the list of token texts rather than on the joined string,
because most of them care about where the token boundaries are: the metaspace
decoder only drops a leading space on the very first token, and the WordPiece
decoder puts a space in front of every token that is not a continuation. The
list is one arena of bytes with a range per token, and steps rewrite it.
"""

from molla.text.utf8 import encode

from .bytelevel import point_to_byte
from .vocab import Vocab

comptime D_BYTE_LEVEL = 0
comptime D_METASPACE = 1
comptime D_WORDPIECE = 2
comptime D_BPE = 3
comptime D_REPLACE = 4
comptime D_STRIP = 5
comptime D_FUSE = 6
comptime D_BYTE_FALLBACK = 7


def _is_byte_token(data: List[UInt8], at: Int, to: Int) -> Int:
    """The byte a `<0xNN>` token stands for, or -1 if it is not one."""
    if to - at != 6:
        return -1
    if data[at] != UInt8(ord("<")) or data[at + 1] != UInt8(ord("0")):
        return -1
    if data[at + 2] != UInt8(ord("x")) or data[at + 5] != UInt8(ord(">")):
        return -1
    var value = 0
    for i in range(at + 3, at + 5):
        var c = Int(data[i])
        if c >= 48 and c <= 57:
            value = value * 16 + c - 48
        elif c >= 65 and c <= 70:
            value = value * 16 + c - 55
        elif c >= 97 and c <= 102:
            value = value * 16 + c - 87
        else:
            return -1
    return value


struct Chunks(Movable):
    """One token text per range, over one arena of bytes."""

    var bytes: List[UInt8]
    var start: List[Int]
    var end: List[Int]

    def __init__(out self):
        self.bytes = List[UInt8]()
        self.start = List[Int]()
        self.end = List[Int]()

    def open(mut self):
        self.start.append(len(self.bytes))

    def close(mut self):
        self.end.append(len(self.bytes))

    def put(mut self, chunk: List[UInt8]):
        for i in range(len(chunk)):
            self.bytes.append(chunk[i])

    def put_span(mut self, chunk: Span[UInt8, _]):
        for i in range(len(chunk)):
            self.bytes.append(chunk[i])

    def copy_from(mut self, chunks: Chunks, at: Int, to: Int):
        for i in range(at, to):
            self.bytes.append(chunks.bytes[i])

    def joined(self) -> List[UInt8]:
        var out = List[UInt8]()
        out.reserve(len(self.bytes))
        for c in range(len(self.start)):
            for i in range(self.start[c], self.end[c]):
                out.append(self.bytes[i])
        return out^


struct DecodeStep(Copyable, Movable):
    var kind: Int
    var pattern: List[UInt8]
    var content: List[UInt8]
    var cleanup: Bool
    var add_prefix_space: Bool
    var strip_start: Int
    var strip_stop: Int
    var replacement: Int
    var prepend_scheme: Int

    def __init__(out self, kind: Int):
        self.kind = kind
        self.pattern = List[UInt8]()
        self.content = List[UInt8]()
        self.cleanup = True
        self.add_prefix_space = True
        self.strip_start = 0
        self.strip_stop = 0
        self.replacement = 0x2581
        self.prepend_scheme = 2


struct Decoder(Movable):
    """The steps, and the byte level table they might need."""

    var steps: List[DecodeStep]
    var inverse: List[Int]

    def __init__(out self):
        self.steps = List[DecodeStep]()
        self.inverse = point_to_byte()

    def is_empty(self) -> Bool:
        return len(self.steps) == 0

    def apply(self, var chunks: Chunks) raises -> Chunks:
        for i in range(len(self.steps)):
            chunks = self._one(self.steps[i], chunks^)
        return chunks^

    def _one(self, step: DecodeStep, var chunks: Chunks) raises -> Chunks:
        if step.kind == D_BYTE_LEVEL:
            return self._byte_level(chunks^)
        if step.kind == D_BYTE_FALLBACK:
            return self._byte_fallback(chunks^)
        if step.kind == D_FUSE:
            return self._fuse(chunks^)
        if step.kind == D_METASPACE:
            return self._metaspace(step, chunks^)
        if step.kind == D_WORDPIECE:
            return self._wordpiece(step, chunks^)
        if step.kind == D_BPE:
            return self._bpe(step, chunks^)
        if step.kind == D_REPLACE:
            return self._replace(step, chunks^)
        return self._strip(step, chunks^)

    def _byte_level(self, var chunks: Chunks) raises -> Chunks:
        """Every character back to the byte it stands for.

        A character outside the alphabet is passed through as its own UTF-8,
        which only happens for an added token that was never byte encoded, and
        dropping it instead would lose the text of every special token.
        """
        var out = Chunks()
        var raw = List[UInt8]()
        for c in range(len(chunks.start)):
            out.open()
            var i = chunks.start[c]
            while i < chunks.end[c]:
                var b = Int(chunks.bytes[i])
                var point = b
                var wanted = 0
                if b >= 0xF0:
                    point = b & 0x07
                    wanted = 3
                elif b >= 0xE0:
                    point = b & 0x0F
                    wanted = 2
                elif b >= 0xC0:
                    point = b & 0x1F
                    wanted = 1
                elif b >= 0x80:
                    i += 1
                    continue
                while wanted > 0 and i + 1 < chunks.end[c]:
                    i += 1
                    point = (point << 6) | (Int(chunks.bytes[i]) & 0x3F)
                    wanted -= 1
                i += 1
                if point < len(self.inverse) and self.inverse[point] >= 0:
                    out.bytes.append(UInt8(self.inverse[point]))
                else:
                    raw.clear()
                    encode(point, raw)
                    out.put(raw)
            out.close()
        return out^

    def _byte_fallback(self, var chunks: Chunks) raises -> Chunks:
        """Runs of `<0xNN>` tokens back into the character they spell.

        A run that is not valid UTF-8 becomes one replacement character per
        byte rather than being dropped, so a truncated stream shows a visible
        gap instead of silently losing text.
        """
        var out = Chunks()
        var pending = List[UInt8]()
        for c in range(len(chunks.start)):
            var value = _is_byte_token(
                chunks.bytes, chunks.start[c], chunks.end[c]
            )
            if value >= 0:
                pending.append(UInt8(value))
                continue
            self._flush_bytes(pending, out)
            out.open()
            out.copy_from(chunks, chunks.start[c], chunks.end[c])
            out.close()
        self._flush_bytes(pending, out)
        return out^

    def _flush_bytes(self, mut pending: List[UInt8], mut out: Chunks) raises:
        if len(pending) == 0:
            return
        out.open()
        if _valid_utf8(pending):
            out.put(pending)
        else:
            var raw = List[UInt8]()
            encode(0xFFFD, raw)
            for _ in range(len(pending)):
                out.put(raw)
        out.close()
        pending.clear()

    def _fuse(self, var chunks: Chunks) -> Chunks:
        var out = Chunks()
        out.open()
        for c in range(len(chunks.start)):
            out.copy_from(chunks, chunks.start[c], chunks.end[c])
        out.close()
        return out^

    def _metaspace(self, step: DecodeStep, var chunks: Chunks) raises -> Chunks:
        var mark = List[UInt8]()
        encode(step.replacement, mark)
        var out = Chunks()
        for c in range(len(chunks.start)):
            out.open()
            var i = chunks.start[c]
            var first = True
            while i < chunks.end[c]:
                if self._at(chunks, i, chunks.end[c], mark):
                    var drop = c == 0 and first and step.prepend_scheme != 0
                    if not drop:
                        out.bytes.append(UInt8(32))
                    i += len(mark)
                else:
                    out.bytes.append(chunks.bytes[i])
                    i += 1
                first = False
            out.close()
        return out^

    def _wordpiece(self, step: DecodeStep, var chunks: Chunks) -> Chunks:
        """The prefix off a continuation, a space in front of anything else."""
        var out = Chunks()
        for c in range(len(chunks.start)):
            out.open()
            var at = chunks.start[c]
            if c != 0:
                if self._at(chunks, at, chunks.end[c], step.pattern):
                    at += len(step.pattern)
                else:
                    out.bytes.append(UInt8(32))
            out.copy_from(chunks, at, chunks.end[c])
            out.close()
        if step.cleanup:
            return _cleanup(out^)
        return out^

    def _bpe(self, step: DecodeStep, var chunks: Chunks) -> Chunks:
        """The end of word suffix becomes a space, except on the last token."""
        var out = Chunks()
        var last = len(chunks.start) - 1
        for c in range(len(chunks.start)):
            out.open()
            var i = chunks.start[c]
            while i < chunks.end[c]:
                if self._at(chunks, i, chunks.end[c], step.pattern):
                    if c != last:
                        out.bytes.append(UInt8(32))
                    i += len(step.pattern)
                else:
                    out.bytes.append(chunks.bytes[i])
                    i += 1
            out.close()
        return out^

    def _replace(self, step: DecodeStep, var chunks: Chunks) -> Chunks:
        var out = Chunks()
        for c in range(len(chunks.start)):
            out.open()
            var i = chunks.start[c]
            while i < chunks.end[c]:
                if len(step.pattern) > 0 and self._at(
                    chunks, i, chunks.end[c], step.pattern
                ):
                    out.put(step.content)
                    i += len(step.pattern)
                else:
                    out.bytes.append(chunks.bytes[i])
                    i += 1
            out.close()
        return out^

    def _strip(self, step: DecodeStep, var chunks: Chunks) -> Chunks:
        var out = Chunks()
        var mark = step.content[0] if len(step.content) > 0 else UInt8(32)
        for c in range(len(chunks.start)):
            var at = chunks.start[c]
            var to = chunks.end[c]
            var taken = 0
            while (
                taken < step.strip_start
                and at < to
                and chunks.bytes[at] == mark
            ):
                at += 1
                taken += 1
            taken = 0
            while (
                taken < step.strip_stop
                and to > at
                and chunks.bytes[to - 1] == mark
            ):
                to -= 1
                taken += 1
            out.open()
            out.copy_from(chunks, at, to)
            out.close()
        return out^

    def _at(
        self, chunks: Chunks, at: Int, limit: Int, what: List[UInt8]
    ) -> Bool:
        if len(what) == 0 or at + len(what) > limit:
            return False
        for i in range(len(what)):
            if chunks.bytes[at + i] != what[i]:
                return False
        return True


def _valid_utf8(data: List[UInt8]) -> Bool:
    var i = 0
    while i < len(data):
        var b = Int(data[i])
        var wanted: Int
        if b < 0x80:
            wanted = 0
        elif b >= 0xF0 and b <= 0xF4:
            wanted = 3
        elif b >= 0xE0:
            wanted = 2
        elif b >= 0xC2:
            wanted = 1
        else:
            return False
        if i + wanted >= len(data):
            return False
        for j in range(1, wanted + 1):
            if (Int(data[i + j]) & 0xC0) != 0x80:
                return False
        i += wanted + 1
    return True


def _cleanup(var chunks: Chunks) -> Chunks:
    """The tidy up an old WordPiece decoder does to its output.

    It is a list of literal string replacements, it is not principled, and it
    is what the reference implementation does, so a decode that skips it does
    not match. Note that it runs per token, so it only fires when a token
    happens to contain both halves.
    """
    var froms = List[String]()
    var tos = List[String]()
    var pairs: List[String] = [
        " .",
        ".",
        " ?",
        "?",
        " !",
        "!",
        " ,",
        ",",
        " ' ",
        "'",
        " n't",
        "n't",
        " 'm",
        "'m",
        " do not",
        " don't",
        " 's",
        "'s",
        " 've",
        "'ve",
        " 're",
        "'re",
    ]
    for i in range(0, len(pairs), 2):
        froms.append(pairs[i])
        tos.append(pairs[i + 1])

    var out = Chunks()
    for c in range(len(chunks.start)):
        var current = List[UInt8]()
        for i in range(chunks.start[c], chunks.end[c]):
            current.append(chunks.bytes[i])
        for p in range(len(froms)):
            current = _replace_bytes(
                current, froms[p].as_bytes(), tos[p].as_bytes()
            )
        out.open()
        out.put(current)
        out.close()
    return out^


def _replace_bytes(
    data: List[UInt8], pattern: Span[UInt8, _], content: Span[UInt8, _]
) -> List[UInt8]:
    var out = List[UInt8]()
    var i = 0
    while i < len(data):
        var hit = i + len(pattern) <= len(data)
        if hit:
            for j in range(len(pattern)):
                if data[i + j] != pattern[j]:
                    hit = False
                    break
        if hit:
            for j in range(len(content)):
                out.append(content[j])
            i += len(pattern)
        else:
            out.append(data[i])
            i += 1
    return out^
