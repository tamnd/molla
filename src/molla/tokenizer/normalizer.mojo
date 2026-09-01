"""The normalizer stage.

The first thing that happens to a string, and the one with the least to show
for itself. A normalizer lowercases, or composes accents, or turns a space into
U+2581, and if it does the wrong one nothing fails, the model just sees a
different string than it was trained on. Qwen composes so that text pasted from
a Mac tokenizes the same as text typed anywhere else. BERT lowercases and drops
accents. Gemma replaces every space with U+2581 before the pre-tokenizer ever
sees it.

Everything here works on code points rather than bytes, because every operation
in the list is defined on characters, and it returns a new list rather than
editing in place, because most of the steps change the length.

`Precompiled` is not here. It is a sentencepiece character map compiled into a
blob of trie data, it appears in the Llama 1 and T5 tokenizers, and it needs
its own reader. A `tokenizer.json` that asks for one is refused at load time
rather than silently normalized some other way.
"""

from molla.text.normalize import normalize, strip_marks
from molla.text.props import CAT_CC, CAT_CO, Unicode, is_whitespace
from molla.text.regex import Regex, Scratch

comptime N_NFC = 0
comptime N_NFD = 1
comptime N_NFKC = 2
comptime N_NFKD = 3
comptime N_LOWERCASE = 4
comptime N_STRIP = 5
comptime N_STRIP_ACCENTS = 6
comptime N_REPLACE_STRING = 7
comptime N_REPLACE_REGEX = 8
comptime N_PREPEND = 9
comptime N_BERT = 10
comptime N_NMT = 11


def _is_control(tables: Unicode, cp: Int) -> Bool:
    """What BERT means by a control character.

    Tab, newline and carriage return are not controls here even though their
    category says they are, because the step that follows turns them into
    spaces and a string that lost its newlines before that has lost its word
    boundaries.

    Unassigned code points are not controls either. Their category is Cn, which
    sits with the other C categories, but the reference implementation reads
    its property from a table that only lists assigned characters, so a code
    point it has never heard of comes back as ordinary text. Treating Cn as a
    control instead drops those characters, and a word that lost one of them is
    a different word, so it takes a different id.
    """
    if cp == 0x09 or cp == 0x0A or cp == 0x0D:
        return False
    var category = tables.category(cp)
    return category >= CAT_CC and category <= CAT_CO


def _is_chinese(cp: Int) -> Bool:
    """The CJK blocks BERT puts spaces around.

    This is the list from the original BERT tokenizer, kept as it is rather
    than replaced by a script property, because it is what the model was
    trained with. It leaves out the compatibility ideographs in the BMP that a
    script property would include, and that difference is a different
    tokenization.
    """
    if cp >= 0x4E00 and cp <= 0x9FFF:
        return True
    if cp >= 0x3400 and cp <= 0x4DBF:
        return True
    if cp >= 0x20000 and cp <= 0x2A6DF:
        return True
    if cp >= 0x2A700 and cp <= 0x2B73F:
        return True
    if cp >= 0x2B740 and cp <= 0x2B81F:
        return True
    if cp >= 0x2B820 and cp <= 0x2CEAF:
        return True
    if cp >= 0xF900 and cp <= 0xFAFF:
        return True
    return cp >= 0x2F800 and cp <= 0x2FA1F


struct NormStep(Copyable, Movable):
    """One step. The fields mean different things per kind, which is what a
    tagged union looks like in a language without one."""

    var kind: Int
    var pattern: List[Int]
    var content: List[Int]
    var expression: Int
    """Index into the normalizer's compiled patterns, or -1."""

    var lowercase: Bool
    var clean_text: Bool
    var handle_chinese: Bool
    var strip_accents: Bool
    var strip_left: Bool
    var strip_right: Bool

    def __init__(out self, kind: Int):
        self.kind = kind
        self.pattern = List[Int]()
        self.content = List[Int]()
        self.expression = -1
        self.lowercase = False
        self.clean_text = False
        self.handle_chinese = False
        self.strip_accents = False
        self.strip_left = True
        self.strip_right = True


struct Normalizer(Movable):
    """The steps, in order, and the patterns they compiled to."""

    var steps: List[NormStep]
    var expressions: List[Regex]

    def __init__(out self):
        self.steps = List[NormStep]()
        self.expressions = List[Regex]()

    def is_empty(self) -> Bool:
        return len(self.steps) == 0

    def apply(
        self, tables: Unicode, mut scratch: Scratch, points: List[Int]
    ) raises -> List[Int]:
        var current = points.copy()
        for i in range(len(self.steps)):
            current = self._one(tables, scratch, self.steps[i], current)
        return current^

    def _one(
        self,
        tables: Unicode,
        mut scratch: Scratch,
        step: NormStep,
        points: List[Int],
    ) raises -> List[Int]:
        var kind = step.kind
        if kind <= N_NFKD:
            return normalize(tables, points, kind >= N_NFKC, kind % 2 == 0)
        if kind == N_LOWERCASE:
            return self._lowercase(tables, points)
        if kind == N_STRIP:
            return self._strip(points, step.strip_left, step.strip_right)
        if kind == N_STRIP_ACCENTS:
            return strip_marks(tables, points)
        if kind == N_REPLACE_STRING:
            return self._replace_string(points, step.pattern, step.content)
        if kind == N_REPLACE_REGEX:
            return self._replace_regex(scratch, step, points)
        if kind == N_PREPEND:
            var out = step.content.copy()
            for i in range(len(points)):
                out.append(points[i])
            return out^
        if kind == N_BERT:
            return self._bert(tables, step, points)
        if kind == N_NMT:
            return self._nmt(points)
        return points.copy()

    def _lowercase(self, tables: Unicode, points: List[Int]) -> List[Int]:
        """The full mapping rather than the simple one, so the Turkish capital
        with a dot becomes two characters the way Python and Rust both make
        it."""
        var out = List[Int]()
        out.reserve(len(points))
        for i in range(len(points)):
            tables.lowercase(points[i], out)
        return out^

    def _strip(self, points: List[Int], left: Bool, right: Bool) -> List[Int]:
        var start = 0
        var end = len(points)
        if left:
            while start < end and is_whitespace(points[start]):
                start += 1
        if right:
            while end > start and is_whitespace(points[end - 1]):
                end -= 1
        var out = List[Int]()
        for i in range(start, end):
            out.append(points[i])
        return out^

    def _replace_string(
        self, points: List[Int], pattern: List[Int], content: List[Int]
    ) -> List[Int]:
        var out = List[Int]()
        if len(pattern) == 0:
            return points.copy()
        var i = 0
        while i < len(points):
            var hit = i + len(pattern) <= len(points)
            if hit:
                for j in range(len(pattern)):
                    if points[i + j] != pattern[j]:
                        hit = False
                        break
            if hit:
                for j in range(len(content)):
                    out.append(content[j])
                i += len(pattern)
            else:
                out.append(points[i])
                i += 1
        return out^

    def _replace_regex(
        self, mut scratch: Scratch, step: NormStep, points: List[Int]
    ) raises -> List[Int]:
        """Every match replaced by the content, empty matches left alone.

        Leaving them alone is a deliberate narrowing. A pattern that can match
        nothing would otherwise sprinkle the replacement between every pair of
        characters, and no normalizer in a real `tokenizer.json` has one.
        """
        var out = List[Int]()
        var found = self.expressions[step.expression].find_all(points, scratch)
        var at = 0
        for m in range(len(found)):
            if found[m].end == found[m].start:
                continue
            for i in range(at, found[m].start):
                out.append(points[i])
            for j in range(len(step.content)):
                out.append(step.content[j])
            at = found[m].end
        for i in range(at, len(points)):
            out.append(points[i])
        return out^

    def _bert(
        self, tables: Unicode, step: NormStep, points: List[Int]
    ) -> List[Int]:
        """Four optional steps in the order the original does them.

        Cleaning first, then spaces around CJK, then accents, then case. The
        order is not free: dropping accents after lowercasing gives the same
        answer here and does not for every script, and putting the CJK spacing
        after cleaning is what stops a control character from gluing two
        ideographs together.
        """
        var current = List[Int]()
        if step.clean_text:
            for i in range(len(points)):
                var cp = points[i]
                if cp == 0 or cp == 0xFFFD or _is_control(tables, cp):
                    continue
                current.append(32 if is_whitespace(cp) else cp)
        else:
            current = points.copy()

        if step.handle_chinese:
            var spaced = List[Int]()
            for i in range(len(current)):
                if _is_chinese(current[i]):
                    spaced.append(32)
                    spaced.append(current[i])
                    spaced.append(32)
                else:
                    spaced.append(current[i])
            current = spaced^

        if step.strip_accents:
            current = strip_marks(
                tables, normalize(tables, current, False, False)
            )
        if step.lowercase:
            current = self._lowercase(tables, current)
        return current^

    def _nmt(self, points: List[Int]) -> List[Int]:
        """The sentencepiece NMT cleanup: some controls vanish and some odd
        spaces become ordinary ones.

        The two lists are not the same list and neither is the set of
        characters anything else calls whitespace. A tab and a newline become
        spaces here while a vertical tab is deleted, and the four space
        characters just below the zero width space are left exactly as they
        are. Both lists are copied from sentencepiece rather than derived.
        """
        var out = List[Int]()
        for i in range(len(points)):
            var cp = points[i]
            if (
                (cp >= 0x0001 and cp <= 0x0008)
                or cp == 0x000B
                or (cp >= 0x000E and cp <= 0x001F)
                or cp == 0x007F
                or cp == 0x008F
                or cp == 0x009F
            ):
                continue
            if (
                cp == 0x0009
                or cp == 0x000A
                or cp == 0x000C
                or cp == 0x000D
                or cp == 0x1680
                or (cp >= 0x200B and cp <= 0x200F)
                or cp == 0x2028
                or cp == 0x2029
                or cp == 0x2581
                or cp == 0xFEFF
                or cp == 0xFFFD
            ):
                out.append(32)
                continue
            out.append(cp)
        return out^
