"""A backtracking regular expression engine, sized for tokenizer patterns.

A `tokenizer.json` carries regular expressions in it, and they are not
decoration. The pre-tokenizer of every GPT style model is one pattern, and the
split it produces decides which byte sequences the merge table is ever allowed
to see. Get the pattern wrong and nothing crashes, the model just gets a
different sequence of tokens than the one it was trained on.

The patterns are all of one shape, and it is worth saying what that shape is
because it decides everything below. They are alternations of short pieces,
they use Unicode categories rather than named scripts, they use one negative
lookahead, and none of them has a capture group, a back reference or a
conditional in it. So this engine has classes, alternation, the three
quantifiers with lazy and possessive forms, anchors, word boundaries and
lookahead, and it does not have captures. A group is a group for precedence and
nothing more, which is why there is no capture buffer anywhere in here.

Matching is leftmost first, which is what Perl does and what the fancy-regex
crate behind Hugging Face does when a pattern has a lookahead in it. Not
leftmost longest. Given `a|ab` against `ab` this answers `a`, and that is the
answer the tokenizer files were written against.

The engine works on code points rather than bytes. A pattern that says
`\\p{L}` has to ask a question about a character, and a byte is not one.
"""

from molla.text.props import (
    CAT_CN,
    CAT_LO,
    CAT_LU,
    CAT_ME,
    CAT_MN,
    CAT_ND,
    CAT_NO,
    CAT_PC,
    Unicode,
    category_index,
    is_whitespace,
)
from molla.text.utf8 import MAX_CODE_POINT, to_code_points

comptime OP_CLASS = 0
"""Match one code point against the class at `a`."""

comptime OP_SPLIT = 1
"""Try `a` first, and `b` if everything after `a` fails."""

comptime OP_JUMP = 2
comptime OP_MATCH = 3

comptime OP_TEXT_START = 4
comptime OP_TEXT_END = 5
comptime OP_LINE_START = 6
comptime OP_LINE_END = 7

comptime OP_BOUNDARY = 8
"""A word boundary, or the absence of one when `a` is set."""

comptime OP_LOOK = 9
"""Run the program at `a` from here. `b` set means it has to fail."""

comptime OP_MARK = 10
"""Remember how much backtracking is on the stack."""

comptime OP_CUT = 11
"""Throw away everything put on the stack since the matching mark."""

comptime _NODE_CLASS = 0
comptime _NODE_CONCAT = 1
comptime _NODE_ALT = 2
comptime _NODE_REPEAT = 3
comptime _NODE_LOOK = 4
comptime _NODE_ANCHOR = 5
comptime _NODE_EMPTY = 6

comptime _GREEDY = 0
comptime _LAZY = 1
comptime _POSSESSIVE = 2

comptime _ANCHOR_TEXT_START = 0
comptime _ANCHOR_TEXT_END = 1
comptime _ANCHOR_LINE_START = 2
comptime _ANCHOR_LINE_END = 3
comptime _ANCHOR_BOUNDARY = 4
comptime _ANCHOR_NOT_BOUNDARY = 5

comptime STEP_LIMIT = 1 << 22
"""How many instructions one match attempt may run before it is called a loop.

A backtracking engine can be made to take exponential time by a pattern that
nests two quantifiers, and a tokenizer pattern comes out of a model file, which
is to say out of the internet. This is the ceiling. Four million steps is far
more than any real pattern needs against any one piece of a pre-tokenized
string, and hitting it raises rather than hanging a worker thread.
"""


struct _Node(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var a: Int
    var b: Int
    var mode: Int
    var first: Int
    var count: Int

    def __init__(out self, kind: Int):
        self.kind = kind
        self.a = 0
        self.b = 0
        self.mode = _GREEDY
        self.first = 0
        self.count = 0


struct CharClass(Copyable, Movable):
    """A set of code points, held as sorted ranges plus a negation flag.

    Negation is a flag rather than a complemented range list because the
    complement of `\\p{L}` is seven hundred ranges of nothing, and the question
    at match time is a bisection either way.
    """

    var lo: List[Int]
    var hi: List[Int]
    var negated: Bool

    var ascii: InlineArray[Bool, 128]
    """The answer for the first 128 characters, worked out at compile time.

    Text is mostly ASCII even when it is not English, and a bisection over the
    seven hundred ranges of `\\p{L}` to find out that `e` is a letter is ten
    branches to answer a question a byte lookup answers. This is filled in by
    `sort`, which is the last thing done to a class before it is used, and it
    already has the negation folded into it.
    """

    def __init__(out self):
        self.lo = List[Int]()
        self.hi = List[Int]()
        self.negated = False
        self.ascii = InlineArray[Bool, 128](fill=False)

    def add(mut self, lo: Int, hi: Int):
        self.lo.append(lo)
        self.hi.append(hi)

    def add_one(mut self, cp: Int):
        self.add(cp, cp)

    def sort(mut self):
        """Order the ranges and merge the ones that touch.

        Insertion sort. A class built from a category has a few hundred ranges
        and is already in order, which is the case insertion sort is best at,
        and a class written out by hand has five.
        """
        for i in range(1, len(self.lo)):
            var lo = self.lo[i]
            var hi = self.hi[i]
            var j = i
            while j > 0 and self.lo[j - 1] > lo:
                self.lo[j] = self.lo[j - 1]
                self.hi[j] = self.hi[j - 1]
                j -= 1
            self.lo[j] = lo
            self.hi[j] = hi

        var merged_lo = List[Int]()
        var merged_hi = List[Int]()
        for i in range(len(self.lo)):
            if (
                len(merged_lo) > 0
                and self.lo[i] <= merged_hi[len(merged_hi) - 1] + 1
            ):
                if self.hi[i] > merged_hi[len(merged_hi) - 1]:
                    merged_hi[len(merged_hi) - 1] = self.hi[i]
            else:
                merged_lo.append(self.lo[i])
                merged_hi.append(self.hi[i])
        self.lo = merged_lo^
        self.hi = merged_hi^

        for cp in range(128):
            self.ascii[cp] = self._search(cp) != self.negated

    def _search(self, cp: Int) -> Bool:
        var low = 0
        var high = len(self.lo) - 1
        while low <= high:
            var mid = (low + high) >> 1
            if cp < self.lo[mid]:
                high = mid - 1
            elif cp > self.hi[mid]:
                low = mid + 1
            else:
                return True
        return False

    def contains(self, cp: Int) -> Bool:
        if cp < 128:
            return self.ascii[cp]
        return self._search(cp) != self.negated


def category_class(tables: Unicode, first: Int, last: Int) -> CharClass:
    """Every code point whose general category is in `first` to `last`.

    Built by walking the category table once, which is how `\\p{L}` becomes
    something a bisection can answer rather than a table lookup per character.
    """
    var out = CharClass()
    for i in range(len(tables.cat_start)):
        var value = Int(tables.cat_value[i])
        if value >= first and value <= last:
            out.add(tables.cat_start[i], tables.cat_end[i])
    out.sort()
    return out^


def whitespace_class() -> CharClass:
    """The White_Space property, as ranges."""
    var out = CharClass()
    out.add(0x09, 0x0D)
    out.add_one(0x20)
    out.add_one(0x85)
    out.add_one(0xA0)
    out.add_one(0x1680)
    out.add(0x2000, 0x200A)
    out.add(0x2028, 0x2029)
    out.add_one(0x202F)
    out.add_one(0x205F)
    out.add_one(0x3000)
    out.sort()
    return out^


def word_class(tables: Unicode) -> CharClass:
    """What `\\w` means: letters, marks, digits, connectors and the joiners.

    Written the way the Rust regex crate writes it, because that is the engine
    the model files were tested against, and it is wider than the ASCII `\\w`
    most people picture.
    """
    var out = CharClass()
    for i in range(len(tables.cat_start)):
        var value = Int(tables.cat_value[i])
        var wanted = value >= CAT_LU and value <= CAT_LO
        wanted = wanted or (value >= CAT_MN and value <= CAT_ME)
        wanted = wanted or value == CAT_ND or value == CAT_PC
        if wanted:
            out.add(tables.cat_start[i], tables.cat_end[i])
    out.add(0x200C, 0x200D)
    out.sort()
    return out^


struct _Parser(Movable):
    """Pattern text to a tree. One pass, recursive descent, no lookahead.

    The tree exists because quantifiers are easier to compile from one. `a{2,4}`
    is the child compiled twice and then twice more optionally, and doing that
    against a stream of instructions means either copying instructions and
    patching every jump in them or building the tree first.
    """

    var pattern: List[Int]
    var at: Int
    var nodes: List[_Node]
    var children: List[Int]
    var classes: List[CharClass]
    var fold: Bool
    var multiline: Bool

    def __init__(out self, pattern: StringSpan):
        self.pattern = to_code_points(pattern.as_bytes())
        self.at = 0
        self.nodes = List[_Node]()
        self.children = List[Int]()
        self.classes = List[CharClass]()
        self.fold = False
        self.multiline = False

    def _peek(self) -> Int:
        if self.at >= len(self.pattern):
            return -1
        return self.pattern[self.at]

    def _next(mut self) -> Int:
        var cp = self._peek()
        if cp >= 0:
            self.at += 1
        return cp

    def _add(mut self, node: _Node) -> Int:
        self.nodes.append(node)
        return len(self.nodes) - 1

    def _add_class(mut self, var cls: CharClass) -> Int:
        cls.sort()
        self.classes.append(cls^)
        return len(self.classes) - 1

    def parse(mut self, tables: Unicode) raises -> Int:
        var root = self._alternation(tables)
        if self.at < len(self.pattern):
            raise Error(
                "unbalanced ) at character "
                + String(self.at)
                + " of the pattern"
            )
        return root

    def _alternation(mut self, tables: Unicode) raises -> Int:
        var branches = List[Int]()
        branches.append(self._concatenation(tables))
        while self._peek() == 0x7C:
            _ = self._next()
            branches.append(self._concatenation(tables))
        if len(branches) == 1:
            return branches[0]
        var node = _Node(_NODE_ALT)
        node.first = len(self.children)
        node.count = len(branches)
        for i in range(len(branches)):
            self.children.append(branches[i])
        return self._add(node)

    def _concatenation(mut self, tables: Unicode) raises -> Int:
        var parts = List[Int]()
        while True:
            var cp = self._peek()
            if cp < 0 or cp == 0x7C or cp == 0x29:
                break
            parts.append(self._repeat(tables))
        if len(parts) == 0:
            return self._add(_Node(_NODE_EMPTY))
        if len(parts) == 1:
            return parts[0]
        var node = _Node(_NODE_CONCAT)
        node.first = len(self.children)
        node.count = len(parts)
        for i in range(len(parts)):
            self.children.append(parts[i])
        return self._add(node)

    def _repeat(mut self, tables: Unicode) raises -> Int:
        var child = self._atom(tables)
        var cp = self._peek()
        var low = 0
        var high = -1
        if cp == 0x2A:
            _ = self._next()
        elif cp == 0x2B:
            _ = self._next()
            low = 1
        elif cp == 0x3F:
            _ = self._next()
            high = 1
        elif cp == 0x7B and self._is_counted():
            _ = self._next()
            low = self._number()
            high = low
            if self._peek() == 0x2C:
                _ = self._next()
                if self._peek() == 0x7D:
                    high = -1
                else:
                    high = self._number()
            if self._next() != 0x7D:
                raise Error("a counted repeat is missing its closing brace")
        else:
            return child

        var node = _Node(_NODE_REPEAT)
        node.a = low
        node.b = high
        node.first = len(self.children)
        node.count = 1
        self.children.append(child)
        if self._peek() == 0x3F:
            _ = self._next()
            node.mode = _LAZY
        elif self._peek() == 0x2B:
            _ = self._next()
            node.mode = _POSSESSIVE

        # One quantifier per atom. `a**` is an error rather than a repeat of a
        # repeat, because the only thing anyone ever means by it is a typo and
        # the reading that is not a typo is a good way to go exponential.
        if self._is_quantifier():
            raise Error("a quantifier follows a quantifier")
        return self._add(node)

    def _is_quantifier(self) -> Bool:
        var cp = self._peek()
        if cp == 0x2A or cp == 0x2B or cp == 0x3F:
            return True
        return cp == 0x7B and self._is_counted()

    def _is_counted(self) -> Bool:
        """True when `{` here really starts a repeat and is not a literal brace.

        A brace with something other than digits behind it is a brace. Perl and
        the Rust crates both take it that way, and a pattern in the wild does
        contain a literal `{` from time to time.
        """
        var i = self.at + 1
        var digits = 0
        while i < len(self.pattern):
            var cp = self.pattern[i]
            if cp >= 0x30 and cp <= 0x39:
                digits += 1
                i += 1
            elif cp == 0x2C and digits > 0:
                i += 1
            elif cp == 0x7D:
                return digits > 0
            else:
                return False
        return False

    def _number(mut self) -> Int:
        var value = 0
        while True:
            var cp = self._peek()
            if cp < 0x30 or cp > 0x39:
                return value
            value = value * 10 + (cp - 0x30)
            _ = self._next()

    def _atom(mut self, tables: Unicode) raises -> Int:
        var cp = self._next()
        if cp == 0x28:
            return self._group(tables)
        if cp == 0x5B:
            var cls = self._bracket(tables)
            var node = _Node(_NODE_CLASS)
            node.a = self._add_class(cls^)
            return self._add(node)
        if cp == 0x2E:
            # Rust and Perl both leave newline out of dot unless told
            # otherwise, and no tokenizer pattern turns it on.
            var any = CharClass()
            any.add(0, 0x09)
            any.add(0x0B, MAX_CODE_POINT)
            var node = _Node(_NODE_CLASS)
            node.a = self._add_class(any^)
            return self._add(node)
        if cp == 0x5E:
            # Start of the text, not start of a line, unless `(?m)` said
            # otherwise. That is what the Rust crates do and Perl does not, and
            # the tokenizer files were written against the Rust ones.
            var node = _Node(_NODE_ANCHOR)
            node.a = (
                _ANCHOR_LINE_START if self.multiline else _ANCHOR_TEXT_START
            )
            return self._add(node)
        if cp == 0x24:
            # And the end of the text exactly. Python would also match this
            # just before a trailing newline. Rust would not, so neither does
            # this.
            var node = _Node(_NODE_ANCHOR)
            node.a = _ANCHOR_LINE_END if self.multiline else _ANCHOR_TEXT_END
            return self._add(node)
        if cp == 0x5C:
            return self._escape(tables)
        if cp < 0:
            raise Error("the pattern ends where a character was expected")
        if cp == 0x2A or cp == 0x2B or cp == 0x3F:
            raise Error("a quantifier has nothing to repeat")
        var literal = CharClass()
        self._add_literal(tables, literal, cp)
        var node = _Node(_NODE_CLASS)
        node.a = self._add_class(literal^)
        return self._add(node)

    def _add_literal(self, tables: Unicode, mut cls: CharClass, cp: Int):
        """Add one literal character, both cases when folding is on."""
        cls.add_one(cp)
        if self.fold:
            var lower = tables.lowercase_one(cp)
            if lower != cp:
                cls.add_one(lower)
            var upper = tables.uppercase_one(cp)
            if upper != cp:
                cls.add_one(upper)

    def _group(mut self, tables: Unicode) raises -> Int:
        """Everything that starts with an open parenthesis.

        There are no capture groups here. A tokenizer pattern has no use for
        one and pretending otherwise would mean carrying a capture buffer
        through the matcher for nobody.
        """
        var saved_fold = self.fold
        var saved_multiline = self.multiline
        var look = 0
        if self._peek() == 0x3F:
            _ = self._next()
            var cp = self._next()
            if cp == 0x3D:
                look = 1
            elif cp == 0x21:
                look = 2
            elif cp == 0x3A:
                pass
            elif cp == 0x69 or cp == 0x6D:
                # A run of flag letters ending in either `)` or `:`. Closing
                # with a parenthesis sets the flags for the rest of the
                # enclosing group and closing with a colon scopes them to this
                # one, which is close enough to Perl for patterns that only
                # ever write the first form at the very front.
                var flag = cp
                while True:
                    if flag == 0x69:
                        self.fold = True
                    else:
                        self.multiline = True
                    var after = self._next()
                    if after == 0x29:
                        return self._add(_Node(_NODE_EMPTY))
                    if after == 0x3A:
                        break
                    if after != 0x69 and after != 0x6D:
                        raise Error("only i and m are understood after (?")
                    flag = after
            elif cp == 0x3C:
                raise Error("look behind is not supported")
            else:
                raise Error("unknown group opener after (?")

        var inner = self._alternation(tables)
        if self._next() != 0x29:
            raise Error("a group is missing its closing parenthesis")
        self.fold = saved_fold
        self.multiline = saved_multiline

        if look == 0:
            return inner
        var node = _Node(_NODE_LOOK)
        node.a = 1 if look == 2 else 0
        node.first = len(self.children)
        node.count = 1
        self.children.append(inner)
        return self._add(node)

    def _shorthand(mut self, tables: Unicode, cp: Int) raises -> CharClass:
        """The one letter classes, and the `\\p{..}` ones."""
        if cp == 0x64 or cp == 0x44:
            var cls = category_class(tables, CAT_ND, CAT_ND)
            cls.negated = cp == 0x44
            return cls^
        if cp == 0x77 or cp == 0x57:
            var cls = word_class(tables)
            cls.negated = cp == 0x57
            return cls^
        if cp == 0x73 or cp == 0x53:
            var cls = whitespace_class()
            cls.negated = cp == 0x53
            return cls^
        if cp == 0x70 or cp == 0x50:
            var cls = self._property(tables)
            if cp == 0x50:
                cls.negated = not cls.negated
            return cls^
        raise Error("unknown escape in the pattern")

    def _property(mut self, tables: Unicode) raises -> CharClass:
        """`\\p{L}`, `\\p{Nd}`, and the bare one letter form `\\pL`."""
        var name = List[Int]()
        if self._peek() == 0x7B:
            _ = self._next()
            while True:
                var cp = self._next()
                if cp < 0:
                    raise Error("a property name is missing its closing brace")
                if cp == 0x7D:
                    break
                name.append(cp)
        else:
            var cp = self._next()
            if cp < 0:
                raise Error("the pattern ends after a property escape")
            name.append(cp)

        if len(name) == 1:
            var letter = name[0]
            if letter == 0x4C:
                return category_class(tables, CAT_LU, CAT_LO)
            if letter == 0x4D:
                return category_class(tables, CAT_MN, CAT_ME)
            if letter == 0x4E:
                return category_class(tables, CAT_ND, CAT_NO)
            if letter == 0x50:
                return category_class(tables, CAT_PC, 17)
            if letter == 0x53:
                return category_class(tables, 18, 21)
            if letter == 0x5A:
                return category_class(tables, 22, 24)
            if letter == 0x43:
                return category_class(tables, 25, CAT_CN)
            raise Error("unknown one letter category in the pattern")

        var bytes = List[UInt8]()
        for i in range(len(name)):
            bytes.append(UInt8(name[i]))
        var text = String(StringSpan(unsafe_from_utf8=bytes))
        var index = category_index(text)
        if index < 0:
            raise Error("unknown category name in the pattern: " + text)
        return category_class(tables, index, index)

    def _escape(mut self, tables: Unicode) raises -> Int:
        """A backslash outside a bracket, which is a class or an anchor."""
        var cp = self._next()
        if cp < 0:
            raise Error("the pattern ends in a backslash")
        if cp == 0x62:
            var node = _Node(_NODE_ANCHOR)
            node.a = _ANCHOR_BOUNDARY
            return self._add(node)
        if cp == 0x42:
            var node = _Node(_NODE_ANCHOR)
            node.a = _ANCHOR_NOT_BOUNDARY
            return self._add(node)
        if cp == 0x41:
            var node = _Node(_NODE_ANCHOR)
            node.a = _ANCHOR_TEXT_START
            return self._add(node)
        if cp == 0x7A:
            var node = _Node(_NODE_ANCHOR)
            node.a = _ANCHOR_TEXT_END
            return self._add(node)

        var simple = _simple_escape(cp)
        if simple >= 0:
            var literal = CharClass()
            self._add_literal(tables, literal, simple)
            var node = _Node(_NODE_CLASS)
            node.a = self._add_class(literal^)
            return self._add(node)

        if cp == 0x75 or cp == 0x78:
            var value = self._hex_escape(cp == 0x75)
            var literal = CharClass()
            self._add_literal(tables, literal, value)
            var node = _Node(_NODE_CLASS)
            node.a = self._add_class(literal^)
            return self._add(node)

        var cls = self._shorthand(tables, cp)
        var node = _Node(_NODE_CLASS)
        node.a = self._add_class(cls^)
        return self._add(node)

    def _hex_escape(mut self, braced: Bool) raises -> Int:
        """`\\xNN`, `\\uNNNN`, and the `\\u{NNNN}` form."""
        var value = 0
        var digits = 4 if braced else 2
        if self._peek() == 0x7B:
            _ = self._next()
            while True:
                var cp = self._next()
                if cp == 0x7D:
                    return value
                var d = _hex_digit(cp)
                if d < 0:
                    raise Error("a hex escape has a non hex digit in it")
                value = value * 16 + d
        for _ in range(digits):
            var d = _hex_digit(self._next())
            if d < 0:
                raise Error("a hex escape is too short")
            value = value * 16 + d
        return value

    def _bracket(mut self, tables: Unicode) raises -> CharClass:
        """A bracketed class, with ranges, escapes and nested shorthands.

        A shorthand inside a bracket has to be a positive set to be unioned in.
        A negated one, `[\\S]`, cannot be, and rather than pretend, this raises,
        because silently taking `[^\\p{L}\\S]` to mean something plausible is
        how a tokenizer ends up quietly different from the one it copied.
        """
        var out = CharClass()
        if self._peek() == 0x5E:
            _ = self._next()
            out.negated = True
        var first = True
        while True:
            var cp = self._next()
            if cp < 0:
                raise Error("a character class is missing its closing bracket")
            if cp == 0x5D and not first:
                out.sort()
                return out^
            first = False

            var low = cp
            if cp == 0x5C:
                var escaped = self._next()
                if escaped < 0:
                    raise Error("a character class ends in a backslash")
                var simple = _simple_escape(escaped)
                if simple >= 0:
                    low = simple
                elif escaped == 0x75 or escaped == 0x78:
                    low = self._hex_escape(escaped == 0x75)
                else:
                    var nested = self._shorthand(tables, escaped)
                    if nested.negated:
                        raise Error(
                            "a negated shorthand inside a character class is"
                            " not supported"
                        )
                    for i in range(len(nested.lo)):
                        out.add(nested.lo[i], nested.hi[i])
                    continue

            if self._peek() == 0x2D and self.at + 1 < len(self.pattern):
                if self.pattern[self.at + 1] != 0x5D:
                    _ = self._next()
                    var high = self._next()
                    if high == 0x5C:
                        var escaped = self._next()
                        var simple = _simple_escape(escaped)
                        if simple >= 0:
                            high = simple
                        elif escaped == 0x75 or escaped == 0x78:
                            high = self._hex_escape(escaped == 0x75)
                        else:
                            raise Error(
                                "the far end of a range has to be a character"
                            )
                    if high < low:
                        raise Error("a character range runs backwards")
                    out.add(low, high)
                    if self.fold:
                        self._fold_range(tables, out, low, high)
                    continue

            self._add_literal(tables, out, low)

    def _fold_range(
        self, tables: Unicode, mut out: CharClass, low: Int, high: Int
    ):
        """Add the other case of every character in a range, when folding.

        Capped, because a case insensitive class over half of Unicode would
        otherwise walk a million characters at compile time to add nothing. The
        patterns that fold are `(?i:'s|'t)` and its relatives, so the cap is
        never near.
        """
        if high - low > 512:
            return
        for cp in range(low, high + 1):
            var lower = tables.lowercase_one(cp)
            if lower != cp:
                out.add_one(lower)
            var upper = tables.uppercase_one(cp)
            if upper != cp:
                out.add_one(upper)


def _hex_digit(cp: Int) -> Int:
    if cp >= 0x30 and cp <= 0x39:
        return cp - 0x30
    if cp >= 0x61 and cp <= 0x66:
        return cp - 0x61 + 10
    if cp >= 0x41 and cp <= 0x46:
        return cp - 0x41 + 10
    return -1


def _simple_escape(cp: Int) -> Int:
    """The escapes that stand for one specific character."""
    if cp == 0x6E:
        return 0x0A
    if cp == 0x72:
        return 0x0D
    if cp == 0x74:
        return 0x09
    if cp == 0x66:
        return 0x0C
    if cp == 0x76:
        return 0x0B
    if cp == 0x30:
        return 0x00
    if cp == 0x61:
        return 0x07
    if cp == 0x65:
        return 0x1B
    # Anything not a letter or a digit stands for itself, which is how a
    # pattern writes a literal dot, brace, bracket or backslash.
    if (
        (cp >= 0x30 and cp <= 0x39)
        or (cp >= 0x41 and cp <= 0x5A)
        or (cp >= 0x61 and cp <= 0x7A)
    ):
        return -1
    return cp


struct Scratch(Movable):
    """The backtracking stacks, kept between matches rather than rebuilt.

    A match attempt that allocates is a match attempt that costs more in the
    allocator than in the matcher, and a pre-tokenizer runs one attempt per
    character of input. So the four stacks live here, the caller holds one, and
    a match leaves them as long as they have ever needed to be. Nested calls,
    which is to say lookaheads, push on the same stacks above a remembered
    height and cut back to it on the way out.
    """

    var pc: List[Int]
    var sp: List[Int]
    var mark: List[Int]
    var marks: List[Int]

    def __init__(out self):
        self.pc = List[Int]()
        self.sp = List[Int]()
        self.mark = List[Int]()
        self.marks = List[Int]()


struct Match(Copyable, ImplicitlyCopyable, Movable):
    """Where a match started and ended, in code points."""

    var start: Int
    var end: Int

    def __init__(out self, start: Int, end: Int):
        self.start = start
        self.end = end

    def found(self) -> Bool:
        return self.start >= 0


struct Regex(Movable):
    """A compiled pattern.

    Compilation happens once when a tokenizer loads and matching happens
    millions of times after that, so everything that can be decided at compile
    time is: every category is expanded into ranges, every case fold is
    expanded into alternatives, and the word class that `\\b` needs is built
    once and stored, so that matching never touches the Unicode tables at all.
    """

    var op: List[Int]
    var a: List[Int]
    var b: List[Int]

    var class_at: List[Int]
    """Where each class starts in the range arena, with an end sentinel."""

    var class_lo: List[Int]
    var class_hi: List[Int]
    var class_negated: List[Bool]

    var class_ascii: List[Bool]
    """128 answers per class, laid out end to end.

    Flat rather than a list of classes because the matcher asks this question
    once per character of input, and reaching through a list of structs to a
    list inside one of them is three loads where this is one.
    """

    var word: CharClass
    var source: String

    def __init__(out self, pattern: StringSpan, tables: Unicode) raises:
        self.op = List[Int]()
        self.a = List[Int]()
        self.b = List[Int]()
        self.class_at = List[Int]()
        self.class_lo = List[Int]()
        self.class_hi = List[Int]()
        self.class_negated = List[Bool]()
        self.class_ascii = List[Bool]()
        self.word = word_class(tables)
        self.source = String(pattern)

        var parser = _Parser(pattern)
        var root = parser.parse(tables)
        for i in range(len(parser.classes)):
            self.class_at.append(len(self.class_lo))
            self.class_negated.append(parser.classes[i].negated)
            for j in range(len(parser.classes[i].lo)):
                self.class_lo.append(parser.classes[i].lo[j])
                self.class_hi.append(parser.classes[i].hi[j])
            for cp in range(128):
                self.class_ascii.append(parser.classes[i].ascii[cp])
        self.class_at.append(len(self.class_lo))
        self._compile(parser.nodes, parser.children, root)
        _ = self._emit(OP_MATCH, 0, 0)

    def _emit(mut self, op: Int, a: Int, b: Int) -> Int:
        self.op.append(op)
        self.a.append(a)
        self.b.append(b)
        return len(self.op) - 1

    def _compile(
        mut self, nodes: List[_Node], children: List[Int], index: Int
    ) raises:
        var node = nodes[index]
        if node.kind == _NODE_EMPTY:
            return
        if node.kind == _NODE_CLASS:
            _ = self._emit(OP_CLASS, node.a, 0)
            return
        if node.kind == _NODE_ANCHOR:
            if node.a == _ANCHOR_TEXT_START:
                _ = self._emit(OP_TEXT_START, 0, 0)
            elif node.a == _ANCHOR_TEXT_END:
                _ = self._emit(OP_TEXT_END, 0, 0)
            elif node.a == _ANCHOR_LINE_START:
                _ = self._emit(OP_LINE_START, 0, 0)
            elif node.a == _ANCHOR_LINE_END:
                _ = self._emit(OP_LINE_END, 0, 0)
            elif node.a == _ANCHOR_BOUNDARY:
                _ = self._emit(OP_BOUNDARY, 0, 0)
            else:
                _ = self._emit(OP_BOUNDARY, 1, 0)
            return
        if node.kind == _NODE_CONCAT:
            for i in range(node.count):
                self._compile(nodes, children, children[node.first + i])
            return
        if node.kind == _NODE_ALT:
            self._compile_alternation(nodes, children, node)
            return
        if node.kind == _NODE_LOOK:
            self._compile_look(nodes, children, node)
            return
        self._compile_repeat(nodes, children, node)

    def _compile_alternation(
        mut self, nodes: List[_Node], children: List[Int], node: _Node
    ) raises:
        """Branches in order, each one jumping to the end when it matches.

        The order is the whole semantics. A split tries its first target and
        only comes back to the second when everything after the first has
        failed, so the branch written first in the pattern wins, which is what
        leftmost first means.
        """
        var jumps = List[Int]()
        for i in range(node.count):
            var last = i == node.count - 1
            var split = -1
            if not last:
                split = self._emit(OP_SPLIT, 0, 0)
                self.a[split] = len(self.op)
            self._compile(nodes, children, children[node.first + i])
            if not last:
                jumps.append(self._emit(OP_JUMP, 0, 0))
                self.b[split] = len(self.op)
        for i in range(len(jumps)):
            self.a[jumps[i]] = len(self.op)

    def _compile_look(
        mut self, nodes: List[_Node], children: List[Int], node: _Node
    ) raises:
        """A lookahead is a whole little program, jumped over rather than into.
        """
        var look = self._emit(OP_LOOK, 0, node.a)
        var over = self._emit(OP_JUMP, 0, 0)
        self.a[look] = len(self.op)
        self._compile(nodes, children, children[node.first])
        _ = self._emit(OP_MATCH, 0, 0)
        self.a[over] = len(self.op)

    def _compile_repeat(
        mut self, nodes: List[_Node], children: List[Int], node: _Node
    ) raises:
        """Counted repeats are expanded, unbounded ones become a loop.

        Expanding `{2,4}` into four copies is fine because the counts in a
        tokenizer pattern are one to three. A guard stops a pattern from the
        internet expanding to a gigabyte of instructions.
        """
        var child = children[node.first]
        var low = node.a
        var high = node.b
        if low > 1000 or high > 1000:
            raise Error("a counted repeat in the pattern is too large")

        var mark = -1
        if node.mode == _POSSESSIVE:
            mark = self._emit(OP_MARK, 0, 0)

        for _ in range(low):
            self._compile(nodes, children, child)

        if high < 0:
            var top = len(self.op)
            var split = self._emit(OP_SPLIT, 0, 0)
            if node.mode == _LAZY:
                self.b[split] = len(self.op)
            else:
                self.a[split] = len(self.op)
            self._compile(nodes, children, child)
            _ = self._emit(OP_JUMP, top, 0)
            if node.mode == _LAZY:
                self.a[split] = len(self.op)
            else:
                self.b[split] = len(self.op)
        else:
            var splits = List[Int]()
            for _ in range(high - low):
                var split = self._emit(OP_SPLIT, 0, 0)
                splits.append(split)
                if node.mode == _LAZY:
                    self.b[split] = len(self.op)
                else:
                    self.a[split] = len(self.op)
                self._compile(nodes, children, child)
            for i in range(len(splits)):
                if node.mode == _LAZY:
                    self.a[splits[i]] = len(self.op)
                else:
                    self.b[splits[i]] = len(self.op)

        if mark >= 0:
            _ = self._emit(OP_CUT, 0, 0)

    def _in_class(self, index: Int, cp: Int) -> Bool:
        """Is `cp` in class `index`. The hot question, asked per character."""
        if cp < 128:
            return self.class_ascii[(index << 7) + cp]
        var low = self.class_at[index]
        var high = self.class_at[index + 1] - 1
        while low <= high:
            var mid = (low + high) >> 1
            if cp < self.class_lo[mid]:
                high = mid - 1
            elif cp > self.class_hi[mid]:
                low = mid + 1
            else:
                return not self.class_negated[index]
        return self.class_negated[index]

    def _is_word(self, points: List[Int], at: Int) -> Bool:
        if at < 0 or at >= len(points):
            return False
        return self.word.contains(points[at])

    def _run(
        self,
        points: List[Int],
        start: Int,
        from_pc: Int,
        mut scratch: Scratch,
    ) raises -> Int:
        """Run the program from `from_pc` at `start`, and answer where it ended.

        One explicit stack rather than recursion, because a greedy quantifier
        over a long line would otherwise put one stack frame per character on
        the machine stack and fall over on input the server did not choose.
        """
        var pc = from_pc
        var sp = start
        var floor = len(scratch.pc)
        var mark_floor = len(scratch.marks)
        var steps = 0

        while True:
            steps += 1
            if steps > STEP_LIMIT:
                raise Error(
                    "the pattern took more than "
                    + String(STEP_LIMIT)
                    + " steps on one piece of input: "
                    + self.source
                )

            var failed = False
            var op = self.op[pc]

            if op == OP_CLASS:
                if sp < len(points) and self._in_class(self.a[pc], points[sp]):
                    pc += 1
                    sp += 1
                else:
                    failed = True
            elif op == OP_SPLIT:
                scratch.pc.append(self.b[pc])
                scratch.sp.append(sp)
                scratch.mark.append(len(scratch.marks))
                pc = self.a[pc]
            elif op == OP_JUMP:
                pc = self.a[pc]
            elif op == OP_MATCH:
                self._unwind(scratch, floor, mark_floor)
                return sp
            elif op == OP_TEXT_START:
                if sp == 0:
                    pc += 1
                else:
                    failed = True
            elif op == OP_TEXT_END:
                if sp == len(points):
                    pc += 1
                else:
                    failed = True
            elif op == OP_LINE_START:
                if sp == 0 or points[sp - 1] == 0x0A:
                    pc += 1
                else:
                    failed = True
            elif op == OP_LINE_END:
                if sp == len(points) or points[sp] == 0x0A:
                    pc += 1
                else:
                    failed = True
            elif op == OP_BOUNDARY:
                var here = self._is_word(points, sp)
                var before = self._is_word(points, sp - 1)
                var boundary = here != before
                if boundary == (self.a[pc] == 0):
                    pc += 1
                else:
                    failed = True
            elif op == OP_LOOK:
                var got = self._run(points, sp, self.a[pc], scratch)
                var matched = got >= 0
                if matched == (self.b[pc] == 0):
                    pc += 1
                else:
                    failed = True
            elif op == OP_MARK:
                scratch.marks.append(len(scratch.pc))
                pc += 1
            else:
                var height = scratch.marks.pop()
                while len(scratch.pc) > height:
                    _ = scratch.pc.pop()
                    _ = scratch.sp.pop()
                    _ = scratch.mark.pop()
                pc += 1

            if failed:
                if len(scratch.pc) == floor:
                    self._unwind(scratch, floor, mark_floor)
                    return -1
                pc = scratch.pc.pop()
                sp = scratch.sp.pop()
                var height = scratch.mark.pop()
                while len(scratch.marks) > height:
                    _ = scratch.marks.pop()

    @staticmethod
    def _unwind(mut scratch: Scratch, floor: Int, mark_floor: Int):
        """Give the stacks back to whoever was using them before this call."""
        while len(scratch.pc) > floor:
            _ = scratch.pc.pop()
            _ = scratch.sp.pop()
            _ = scratch.mark.pop()
        while len(scratch.marks) > mark_floor:
            _ = scratch.marks.pop()

    def match_at(
        self, points: List[Int], at: Int, mut scratch: Scratch
    ) raises -> Int:
        """Where a match starting exactly at `at` ends, or minus one."""
        return self._run(points, at, 0, scratch)

    def find(
        self, points: List[Int], from_index: Int, mut scratch: Scratch
    ) raises -> Match:
        """The leftmost match at or after `from_index`.

        Leftmost first: the earliest starting position wins, and among the ways
        of matching at that position the one the pattern prefers wins. There is
        no search for a longer match at a later position.
        """
        for start in range(from_index, len(points) + 1):
            var end = self._run(points, start, 0, scratch)
            if end >= 0:
                return Match(start, end)
        return Match(-1, -1)

    def find_all(
        self, points: List[Int], mut scratch: Scratch
    ) raises -> List[Match]:
        """Every non overlapping match, left to right.

        An empty match moves the cursor on by one character, which is what
        stops a pattern that can match nothing from matching nothing forever.
        """
        var out = List[Match]()
        var at = 0
        while at <= len(points):
            var found = self.find(points, at, scratch)
            if not found.found():
                break
            out.append(found)
            if found.end == found.start:
                at = found.start + 1
            else:
                at = found.end
        return out^

    def is_match(self, points: List[Int], mut scratch: Scratch) raises -> Bool:
        return self.find(points, 0, scratch).found()
