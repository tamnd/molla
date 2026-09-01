"""Where one grapheme cluster ends and the next begins.

A grapheme cluster is what a reader would call a character: a base letter plus
whatever combining marks, joiners and variation selectors hang off it. The
letter e followed by a combining acute is two code points and one cluster, and
a flag is two code points and one cluster, and a family emoji is seven.

The rules are UAX #29, extended clusters rather than legacy ones, and all of
them are here including the emoji rule GB11 and the Indic conjunct rule GB9c.
The only caller today is the SentencePiece charsmap normalizer, which only
looks up clusters shorter than six bytes and so can never reach either of those
two rules. They are implemented anyway, because a boundary walker that is right
for accents and wrong for flags would be a trap for whoever calls this next.

Everything works on code points rather than bytes, the way the rest of the text
layer does, and the properties come out of the generated table in props.mojo.
"""

from molla.text.props import (
    GB_CLASS_MASK,
    GB_CONTROL,
    GB_CR,
    GB_EXTEND,
    GB_INCB_SHIFT,
    GB_L,
    GB_LF,
    GB_LV,
    GB_LVT,
    GB_PICTOGRAPHIC,
    GB_PREPEND,
    GB_REGIONAL,
    GB_SPACING_MARK,
    GB_T,
    GB_V,
    GB_ZWJ,
    INCB_CONSONANT,
    INCB_EXTEND,
    INCB_LINKER,
    Unicode,
)


struct _State(Copyable, ImplicitlyCopyable, Movable):
    """What the rules need to remember about the cluster so far.

    Three of the boundary rules look further back than the character in front
    of them. GB12 and GB13 join regional indicators in pairs, so they need the
    count. GB11 joins a pictograph across a ZWJ, but only if the run before the
    ZWJ started with a pictograph. GB9c joins two Indic consonants across a
    linker. All three are answered by carrying a little state forward rather
    than by scanning backwards, which keeps the walk linear.
    """

    var regionals: Int
    """Regional indicators in the unbroken run ending at the last character."""

    var pictographic: Bool
    """The run is a pictograph followed by nothing but Extend so far."""

    var pictographic_zwj: Bool
    """The same run, and the last character was the ZWJ that closed it."""

    var consonant: Bool
    """An Indic consonant, then nothing but conjunct extends and linkers."""

    var linker: Bool
    """And at least one of them was a linker."""

    def __init__(out self):
        self.regionals = 0
        self.pictographic = False
        self.pictographic_zwj = False
        self.consonant = False
        self.linker = False

    def begin(mut self, property: Int):
        """Start a cluster at a character with this packed property."""
        self.regionals = 1 if (property & GB_CLASS_MASK) == GB_REGIONAL else 0
        self.pictographic = (property & GB_PICTOGRAPHIC) != 0
        self.pictographic_zwj = False
        self.consonant = (property >> GB_INCB_SHIFT) == INCB_CONSONANT
        self.linker = False

    def extend(mut self, property: Int):
        """Take one more character into the cluster."""
        var kind = property & GB_CLASS_MASK
        var conjunct = property >> GB_INCB_SHIFT

        if kind == GB_REGIONAL:
            self.regionals += 1
        else:
            self.regionals = 0

        if kind == GB_EXTEND and self.pictographic:
            self.pictographic_zwj = False
        elif kind == GB_ZWJ and self.pictographic:
            self.pictographic = False
            self.pictographic_zwj = True
        else:
            self.pictographic = (property & GB_PICTOGRAPHIC) != 0
            self.pictographic_zwj = False

        if conjunct == INCB_CONSONANT:
            self.consonant = True
            self.linker = False
        elif conjunct == INCB_LINKER and self.consonant:
            self.linker = True
        elif conjunct == INCB_EXTEND and self.consonant:
            pass
        else:
            self.consonant = False
            self.linker = False


def _joins(state: _State, left: Int, right: Int) -> Bool:
    """Whether two adjacent characters are in the same cluster.

    The rules are in the order UAX #29 gives them, because the order is what
    resolves them: GB4 and GB5 have to fire before GB9, or a combining mark
    after a newline would glue itself to the newline.
    """
    var before = left & GB_CLASS_MASK
    var after = right & GB_CLASS_MASK

    if before == GB_CR and after == GB_LF:
        return True
    if before == GB_CR or before == GB_LF or before == GB_CONTROL:
        return False
    if after == GB_CR or after == GB_LF or after == GB_CONTROL:
        return False

    if before == GB_L and (
        after == GB_L or after == GB_V or after == GB_LV or after == GB_LVT
    ):
        return True
    if (before == GB_LV or before == GB_V) and (after == GB_V or after == GB_T):
        return True
    if (before == GB_LVT or before == GB_T) and after == GB_T:
        return True

    if after == GB_EXTEND or after == GB_ZWJ:
        return True
    if after == GB_SPACING_MARK:
        return True
    if before == GB_PREPEND:
        return True

    if state.linker and (right >> GB_INCB_SHIFT) == INCB_CONSONANT:
        return True
    if state.pictographic_zwj and (right & GB_PICTOGRAPHIC) != 0:
        return True
    if (
        before == GB_REGIONAL
        and after == GB_REGIONAL
        and state.regionals % 2 == 1
    ):
        return True
    return False


def cluster_ends(tables: Unicode, points: List[Int]) -> List[Int]:
    """One index per cluster, each of them one past its last code point.

    So the first cluster is `points[0:out[0]]`, the second is
    `points[out[0]:out[1]]`, and the length of the list is the number of
    clusters the string has.
    """
    var out = List[Int]()
    if len(points) == 0:
        return out^

    var state = _State()
    var left = tables.grapheme_break(points[0])
    state.begin(left)
    for i in range(1, len(points)):
        var right = tables.grapheme_break(points[i])
        if _joins(state, left, right):
            state.extend(right)
        else:
            out.append(i)
            state.begin(right)
        left = right
    out.append(len(points))
    return out^
