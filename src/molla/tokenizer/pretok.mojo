"""The pre-tokenizer stage.

Between the normalizer and the model there is a step that cuts the string into
pieces the model is never allowed to merge across. It is the reason a BPE
vocabulary has `Ġthe` and not `theand`, and it is the single place where two
implementations of the same tokenizer most often quietly disagree, because the
cut is described by a regular expression and a four way enum about what to do
with the delimiter.

Pieces live as ranges into one arena of code points. A step that only re-cuts,
which is most of them, rewrites the ranges and leaves the arena alone. A step
that changes the text, which is byte level and metaspace, builds a new arena.
That split is what keeps the common path free of copying.
"""

from molla.text.props import (
    CAT_ND,
    CAT_NO,
    CAT_PC,
    CAT_PO,
    Unicode,
    is_whitespace,
)
from molla.text.regex import Regex, Scratch
from molla.text.utf8 import encode

from .bytelevel import byte_to_point

comptime T_BYTE_LEVEL = 0
comptime T_WHITESPACE = 1
comptime T_WHITESPACE_SPLIT = 2
comptime T_SPLIT_STRING = 3
comptime T_SPLIT_REGEX = 4
comptime T_PUNCTUATION = 5
comptime T_METASPACE = 6
comptime T_DIGITS = 7
comptime T_BERT = 8
comptime T_CHAR_DELIMITER = 9

comptime B_REMOVED = 0
comptime B_ISOLATED = 1
comptime B_MERGED_WITH_PREVIOUS = 2
comptime B_MERGED_WITH_NEXT = 3
comptime B_CONTIGUOUS = 4

comptime PREPEND_NEVER = 0
comptime PREPEND_FIRST = 1
comptime PREPEND_ALWAYS = 2

comptime GPT2_PATTERN = (
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+|"
    " ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
)
"""The pattern every byte level pre-tokenizer uses when `use_regex` is on.

It is written out here rather than read from the file because `ByteLevel` in a
`tokenizer.json` carries only the flag, not the pattern, and this is the
pattern the flag has always meant.
"""

comptime WHITESPACE_PATTERN = "\\w+|[^\\w\\s]+"
"""What the `Whitespace` pre-tokenizer keeps, rather than what it splits on."""


def _is_punctuation(tables: Unicode, cp: Int) -> Bool:
    """ASCII punctuation plus the Unicode punctuation categories.

    The ASCII half is not redundant. Characters like `$`, `+` and `^` are
    symbols to Unicode and punctuation to every tokenizer that has ever been
    trained, because the original BERT said so.
    """
    if cp < 128:
        if cp >= 0x21 and cp <= 0x2F:
            return True
        if cp >= 0x3A and cp <= 0x40:
            return True
        if cp >= 0x5B and cp <= 0x60:
            return True
        return cp >= 0x7B and cp <= 0x7E
    var category = tables.category(cp)
    return category >= CAT_PC and category <= CAT_PO


def _is_numeric(tables: Unicode, cp: Int) -> Bool:
    var category = tables.category(cp)
    return category >= CAT_ND and category <= CAT_NO


struct Cuts(Movable):
    """The working lists one re-cut needs, kept between pieces.

    A step runs over every piece the last step produced, and a piece is a word,
    so these lists are filled and emptied a million times for a megabyte of
    text. Allocating them per piece made the pre-tokenizer the slowest stage in
    the pipeline by a wide margin. They live here instead and are cleared, so
    the memory is asked for once per step rather than once per word.
    """

    var hit_start: List[Int]
    var hit_end: List[Int]
    var cover_start: List[Int]
    var cover_end: List[Int]
    var cover_hit: List[Bool]
    var kept_start: List[Int]
    var kept_end: List[Int]
    var window: List[Int]

    def __init__(out self):
        self.hit_start = List[Int]()
        self.hit_end = List[Int]()
        self.cover_start = List[Int]()
        self.cover_end = List[Int]()
        self.cover_hit = List[Bool]()
        self.kept_start = List[Int]()
        self.kept_end = List[Int]()
        self.window = List[Int]()

    def fold_forward(mut self, behavior: Int):
        self.kept_start.clear()
        self.kept_end.clear()
        var previous = False
        for i in range(len(self.cover_start)):
            var hit = self.cover_hit[i]
            var merge = False
            if behavior == B_REMOVED and hit:
                previous = hit
                continue
            if behavior == B_CONTIGUOUS:
                merge = len(self.kept_start) > 0 and hit == previous
            elif behavior == B_MERGED_WITH_PREVIOUS:
                merge = hit and not previous and len(self.kept_start) > 0
            previous = hit
            if merge:
                self.kept_end[len(self.kept_end) - 1] = self.cover_end[i]
            else:
                self.kept_start.append(self.cover_start[i])
                self.kept_end.append(self.cover_end[i])

    def fold_backward(mut self):
        """Merged with next is merged with previous read right to left."""
        self.kept_start.clear()
        self.kept_end.clear()
        var previous = False
        for j in range(len(self.cover_start) - 1, -1, -1):
            var hit = self.cover_hit[j]
            if hit and not previous and len(self.kept_start) > 0:
                self.kept_start[len(self.kept_start) - 1] = self.cover_start[j]
            else:
                self.kept_start.append(self.cover_start[j])
                self.kept_end.append(self.cover_end[j])
            previous = hit
        var i = 0
        var j = len(self.kept_start) - 1
        while i < j:
            var start = self.kept_start[i]
            self.kept_start[i] = self.kept_start[j]
            self.kept_start[j] = start
            var end = self.kept_end[i]
            self.kept_end[i] = self.kept_end[j]
            self.kept_end[j] = end
            i += 1
            j -= 1


struct Pieces(Movable):
    """An arena of code points and the ranges the pre-tokenizer cut it into."""

    var points: List[Int]
    var start: List[Int]
    var end: List[Int]

    def __init__(out self, var points: List[Int]):
        self.start = List[Int]()
        self.end = List[Int]()
        if len(points) > 0:
            self.start.append(0)
            self.end.append(len(points))
        self.points = points^

    def count(self) -> Int:
        return len(self.start)


struct PreStep(Copyable, Movable):
    """One step, with the fields of every kind on it. Same tagged union shape
    as the normalizer, same reason."""

    var kind: Int
    var behavior: Int
    var pattern: List[Int]
    var expression: Int
    var invert: Bool
    var add_prefix_space: Bool
    var use_regex: Bool
    var individual_digits: Bool
    var trim_offsets: Bool
    var replacement: Int
    var prepend_scheme: Int
    var split: Bool

    def __init__(out self, kind: Int):
        self.kind = kind
        self.behavior = B_ISOLATED
        self.pattern = List[Int]()
        self.expression = -1
        self.invert = False
        self.add_prefix_space = False
        self.use_regex = True
        self.individual_digits = False
        self.trim_offsets = True
        self.replacement = 0x2581
        self.prepend_scheme = PREPEND_ALWAYS
        self.split = True


struct PreTokenizer(Movable):
    """The steps and the patterns they compiled to."""

    var steps: List[PreStep]
    var expressions: List[Regex]
    var alphabet: List[Int]

    def __init__(out self):
        self.steps = List[PreStep]()
        self.expressions = List[Regex]()
        self.alphabet = byte_to_point()

    def is_empty(self) -> Bool:
        return len(self.steps) == 0

    def apply(
        self, tables: Unicode, mut scratch: Scratch, var pieces: Pieces
    ) raises -> Pieces:
        for i in range(len(self.steps)):
            pieces = self._one(tables, scratch, self.steps[i], pieces^)
        return pieces^

    def _one(
        self,
        tables: Unicode,
        mut scratch: Scratch,
        step: PreStep,
        var pieces: Pieces,
    ) raises -> Pieces:
        var kind = step.kind
        if kind == T_BYTE_LEVEL:
            return self._byte_level(tables, scratch, step, pieces^)
        if kind == T_METASPACE:
            return self._metaspace(tables, scratch, step, pieces^)
        if kind == T_BERT:
            var spaces = PreStep(T_WHITESPACE_SPLIT)
            spaces.behavior = B_REMOVED
            var marks = PreStep(T_PUNCTUATION)
            marks.behavior = B_ISOLATED
            var words = self._recut(tables, scratch, spaces, pieces^)
            return self._recut(tables, scratch, marks, words^)
        return self._recut(tables, scratch, step, pieces^)

    def _recut(
        self,
        tables: Unicode,
        mut scratch: Scratch,
        step: PreStep,
        var pieces: Pieces,
    ) raises -> Pieces:
        """Every step that only moves the cuts, over every piece in turn."""
        var start = List[Int]()
        var end = List[Int]()
        var cuts = Cuts()
        for p in range(len(pieces.start)):
            self._cut(
                tables,
                scratch,
                step,
                pieces,
                pieces.start[p],
                pieces.end[p],
                cuts,
                start,
                end,
            )
        pieces.start = start^
        pieces.end = end^
        return pieces^

    def _cut(
        self,
        tables: Unicode,
        mut scratch: Scratch,
        step: PreStep,
        pieces: Pieces,
        from_at: Int,
        to_at: Int,
        mut cuts: Cuts,
        mut start: List[Int],
        mut end: List[Int],
    ) raises:
        """The cover of one piece, then the behaviour, then the empties out.

        The cover is the whole range written as alternating stretches that are
        and are not the delimiter, with no empty stretch invented between two
        adjacent delimiters. Every behaviour is a fold over that, and the folds
        are worth reading slowly, because merged with previous over two spaces
        in a row has to give one piece and not two.
        """
        self._cover(tables, scratch, step, pieces, from_at, to_at, cuts)
        if step.behavior == B_MERGED_WITH_NEXT:
            cuts.fold_backward()
        else:
            cuts.fold_forward(step.behavior)
        for i in range(len(cuts.kept_start)):
            if cuts.kept_end[i] > cuts.kept_start[i]:
                start.append(cuts.kept_start[i])
                end.append(cuts.kept_end[i])

    def _cover(
        self,
        tables: Unicode,
        mut scratch: Scratch,
        step: PreStep,
        pieces: Pieces,
        from_at: Int,
        to_at: Int,
        mut cuts: Cuts,
    ) raises:
        self._delimiters(tables, scratch, step, pieces, from_at, to_at, cuts)
        cuts.cover_start.clear()
        cuts.cover_end.clear()
        cuts.cover_hit.clear()
        var at = from_at
        for i in range(len(cuts.hit_start)):
            if cuts.hit_start[i] > at:
                cuts.cover_start.append(at)
                cuts.cover_end.append(cuts.hit_start[i])
                cuts.cover_hit.append(step.invert)
            cuts.cover_start.append(cuts.hit_start[i])
            cuts.cover_end.append(cuts.hit_end[i])
            cuts.cover_hit.append(not step.invert)
            at = cuts.hit_end[i]
        if at < to_at or len(cuts.cover_start) == 0:
            cuts.cover_start.append(at)
            cuts.cover_end.append(to_at)
            cuts.cover_hit.append(step.invert)

    def _delimiters(
        self,
        tables: Unicode,
        mut scratch: Scratch,
        step: PreStep,
        pieces: Pieces,
        from_at: Int,
        to_at: Int,
        mut cuts: Cuts,
    ) raises:
        """Where the pattern matches inside one piece.

        A character predicate that is going to be merged back together gets one
        range per run rather than one per character, which is the same answer
        and stops a piece of a thousand spaces from building a cover of a
        thousand entries.
        """
        cuts.hit_start.clear()
        cuts.hit_end.clear()
        var kind = step.kind
        if kind == T_SPLIT_REGEX or kind == T_WHITESPACE:
            cuts.window.clear()
            cuts.window.reserve(to_at - from_at)
            for i in range(from_at, to_at):
                cuts.window.append(pieces.points[i])
            var found = self.expressions[step.expression].find_all(
                cuts.window, scratch
            )
            for m in range(len(found)):
                cuts.hit_start.append(from_at + found[m].start)
                cuts.hit_end.append(from_at + found[m].end)
            return

        if kind == T_SPLIT_STRING or kind == T_CHAR_DELIMITER:
            if len(step.pattern) == 0:
                return
            var width = len(step.pattern)
            var i = from_at
            while i + width <= to_at:
                var hit = True
                for j in range(width):
                    if pieces.points[i + j] != step.pattern[j]:
                        hit = False
                        break
                if hit:
                    cuts.hit_start.append(i)
                    cuts.hit_end.append(i + width)
                    i += width
                else:
                    i += 1
            return

        var one_each = step.behavior != B_CONTIGUOUS
        var i = from_at
        while i < to_at:
            if not self._is_delimiter(tables, step, pieces.points[i]):
                i += 1
                continue
            var run = i
            while run < to_at and self._is_delimiter(
                tables, step, pieces.points[run]
            ):
                run += 1
            if one_each:
                for one in range(i, run):
                    cuts.hit_start.append(one)
                    cuts.hit_end.append(one + 1)
            else:
                cuts.hit_start.append(i)
                cuts.hit_end.append(run)
            i = run

    def _is_delimiter(self, tables: Unicode, step: PreStep, cp: Int) -> Bool:
        if step.kind == T_WHITESPACE_SPLIT:
            return is_whitespace(cp)
        if step.kind == T_PUNCTUATION:
            return _is_punctuation(tables, cp)
        if step.kind == T_DIGITS:
            return _is_numeric(tables, cp)
        return False

    def _byte_level(
        self,
        tables: Unicode,
        mut scratch: Scratch,
        step: PreStep,
        var pieces: Pieces,
    ) raises -> Pieces:
        """A space in front if asked, then the pattern, then the alphabet.

        The prefix space goes on before the split, not after, because it is
        meant to become part of the first piece. Doing it the other way round
        leaves a lone space in front of every string, which is one extra token
        and therefore a different id for every token after it.
        """
        if step.add_prefix_space:
            pieces = self._prefix(pieces^, 32)
        if step.use_regex:
            var split = PreStep(T_SPLIT_REGEX)
            split.behavior = B_ISOLATED
            split.expression = step.expression
            pieces = self._recut(tables, scratch, split, pieces^)

        var points = List[Int]()
        points.reserve(len(pieces.points))
        var start = List[Int]()
        var end = List[Int]()
        var bytes = List[UInt8]()
        for p in range(len(pieces.start)):
            start.append(len(points))
            for i in range(pieces.start[p], pieces.end[p]):
                bytes.clear()
                encode(pieces.points[i], bytes)
                for b in range(len(bytes)):
                    points.append(self.alphabet[Int(bytes[b])])
            end.append(len(points))
        pieces.points = points^
        pieces.start = start^
        pieces.end = end^
        return pieces^

    def _prefix(self, var pieces: Pieces, cp: Int) -> Pieces:
        """Put one code point in front of the first piece unless it is already
        there."""
        if len(pieces.start) == 0:
            var only = List[Int]()
            only.append(cp)
            return Pieces(only^)
        if pieces.points[pieces.start[0]] == cp:
            return pieces^
        var points = List[Int]()
        var start = List[Int]()
        var end = List[Int]()
        for p in range(len(pieces.start)):
            start.append(len(points))
            if p == 0:
                points.append(cp)
            for i in range(pieces.start[p], pieces.end[p]):
                points.append(pieces.points[i])
            end.append(len(points))
        pieces.points = points^
        pieces.start = start^
        pieces.end = end^
        return pieces^

    def _metaspace(
        self,
        tables: Unicode,
        mut scratch: Scratch,
        step: PreStep,
        var pieces: Pieces,
    ) raises -> Pieces:
        """Spaces become the replacement, then a cut in front of each one.

        Merged with next is the whole trick. The replacement stays attached to
        the word that follows it, so the model can tell a word at the start of
        a line from the same word after a space without a separate token for
        either.
        """
        var points = List[Int]()
        points.reserve(len(pieces.points) + len(pieces.start))
        var start = List[Int]()
        var end = List[Int]()
        for p in range(len(pieces.start)):
            start.append(len(points))
            var prepend = step.prepend_scheme == PREPEND_ALWAYS or (
                step.prepend_scheme == PREPEND_FIRST and p == 0
            )
            var head = -1
            if pieces.end[p] > pieces.start[p]:
                head = pieces.points[pieces.start[p]]
            if prepend and head != step.replacement and head != 32:
                points.append(step.replacement)
            for i in range(pieces.start[p], pieces.end[p]):
                var cp = pieces.points[i]
                points.append(step.replacement if cp == 32 else cp)
            end.append(len(points))
        pieces.points = points^
        pieces.start = start^
        pieces.end = end^

        if not step.split:
            return pieces^
        var cut = PreStep(T_SPLIT_STRING)
        cut.behavior = B_MERGED_WITH_NEXT
        cut.pattern.append(step.replacement)
        return self._recut(tables, scratch, cut, pieces^)
