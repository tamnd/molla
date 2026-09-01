"""The four Unicode normal forms.

Two operations and four names. Decomposition takes every character apart into
its pieces and then puts the pieces in canonical order, and composition puts
them back together as far as the tables allow. NFD is the first, NFC is both,
and the K forms are the same two with the compatibility decompositions turned
on, which is a wider and lossier take apart: a ligature becomes its letters and
a superscript two becomes a two.

Which one matters depends on the model. A BERT normalizer decomposes so it can
drop the accents it just separated. Qwen composes, so that text pasted from a
Mac, where the file system hands you decomposed accents, tokenizes the same as
text typed anywhere else. Getting this wrong does not fail, it just quietly
gives you different tokens for the same sentence.
"""

from molla.text.props import (
    CAT_ME,
    CAT_MN,
    HANGUL_L_BASE,
    HANGUL_N_COUNT,
    HANGUL_S_BASE,
    HANGUL_S_COUNT,
    HANGUL_T_BASE,
    HANGUL_T_COUNT,
    HANGUL_V_BASE,
    Unicode,
)


def _decompose_hangul(cp: Int, mut out: List[Int]) -> Bool:
    """Take a Hangul syllable apart, by arithmetic rather than by table."""
    var index = cp - HANGUL_S_BASE
    if index < 0 or index >= HANGUL_S_COUNT:
        return False
    out.append(HANGUL_L_BASE + index // HANGUL_N_COUNT)
    out.append(HANGUL_V_BASE + (index % HANGUL_N_COUNT) // HANGUL_T_COUNT)
    var trailing = index % HANGUL_T_COUNT
    if trailing != 0:
        out.append(HANGUL_T_BASE + trailing)
    return True


def _decompose_one(
    tables: Unicode, cp: Int, compatibility: Bool, mut out: List[Int]
):
    """Append the full decomposition of one character.

    Full means keep going. A decomposition can name a character that itself
    decomposes, three levels deep in the worst case, and stopping at the first
    level leaves text that is not in normal form and compares unequal to text
    that is.
    """
    if _decompose_hangul(cp, out):
        return
    var index = tables.decomposition(cp, compatibility)
    if index < 0:
        out.append(cp)
        return
    var length = tables.decomposition_length(index)
    for i in range(length):
        _decompose_one(
            tables, tables.decomposition_at(index, i), compatibility, out
        )


def canonical_order(tables: Unicode, mut points: List[Int]):
    """Sort each run of combining marks by combining class, stably.

    Insertion sort on purpose. The runs are short, three characters is a long
    one, and the sort has to be stable because two marks of the same class are
    ordered by the text and swapping them changes what the text says.
    """
    var i = 1
    while i < len(points):
        var this_class = tables.combining(points[i])
        if this_class != 0:
            var cp = points[i]
            var j = i
            while j > 0:
                var previous = tables.combining(points[j - 1])
                if previous == 0 or previous <= this_class:
                    break
                points[j] = points[j - 1]
                j -= 1
            points[j] = cp
        i += 1


def decompose(
    tables: Unicode, points: List[Int], compatibility: Bool
) -> List[Int]:
    """NFD, or NFKD when `compatibility` is set."""
    var out = List[Int]()
    out.reserve(len(points))
    for i in range(len(points)):
        _decompose_one(tables, points[i], compatibility, out)
    canonical_order(tables, out)
    return out^


def compose(tables: Unicode, points: List[Int]) -> List[Int]:
    """Canonical composition, over text that is already decomposed and ordered.

    The rule that does the work is blocking. A starter and a mark compose only
    when nothing between them has a combining class greater than or equal to
    the mark's, because a mark that sits closer to the base owns the position,
    and composing past it would move an accent from one letter to another.
    """
    var out = List[Int]()
    out.reserve(len(points))
    var starter_at = -1
    var previous_class = 0

    for i in range(len(points)):
        var cp = points[i]
        var this_class = tables.combining(cp)

        # Zero means the character before this one is the starter itself, with
        # nothing in between, and that is always composable. Otherwise the mark
        # in between has to belong further out than this one does.
        if starter_at >= 0 and (
            previous_class == 0 or previous_class < this_class
        ):
            var joined = tables.compose(out[starter_at], cp)
            if joined != 0:
                out[starter_at] = joined
                continue

        out.append(cp)
        if this_class == 0:
            starter_at = len(out) - 1
        previous_class = this_class

    return out^


def normalize(
    tables: Unicode, points: List[Int], compatibility: Bool, composed: Bool
) -> List[Int]:
    """One of the four forms, named by its two flags rather than by a string.

    `compatibility` picks the K forms and `composed` picks C over D, so NFC is
    false and true, NFKD is true and false, and there is no fifth combination
    to get wrong.

    Most characters do not need any of this. Every character under the floor is
    already in every normal form and cannot join with anything around it, so a
    run of them is copied across and only the runs in between are taken apart
    and put back together. English text never leaves the first branch, and a
    page of prose with one accent in it does the real work on one word.

    A run of real work starts one character early, because the first character
    of it may be a mark that composes with the ordinary letter in front of it,
    and that letter has to be there for it to compose with. One character back
    is far enough: everything under the floor is a starter, so the segment
    cannot have begun any earlier than that.
    """
    var floor = (
        tables.compatibility_floor if compatibility else tables.canonical_floor
    )
    var out = List[Int]()
    out.reserve(len(points))
    var run = List[Int]()
    var i = 0

    while i < len(points):
        if points[i] < floor:
            out.append(points[i])
            i += 1
            continue

        var start = i
        if len(out) > 0:
            start -= 1
            _ = out.pop()
        var end = i
        while end < len(points) and points[end] >= floor:
            end += 1

        run.clear()
        for j in range(start, end):
            run.append(points[j])
        var done = decompose(tables, run, compatibility)
        if composed:
            done = compose(tables, done)
        for j in range(len(done)):
            out.append(done[j])
        i = end

    return out^


def strip_marks(tables: Unicode, points: List[Int]) -> List[Int]:
    """Drop every non spacing mark.

    This is what strip accents means, and it is only correct after a
    decomposition: e acute is one character with no mark in it, and stripping
    marks from it does nothing at all until it has been taken apart first. The
    BERT normalizer does exactly that pair, in that order, and this is the
    second half.
    """
    var out = List[Int]()
    out.reserve(len(points))
    for i in range(len(points)):
        if tables.category(points[i]) != CAT_MN:
            out.append(points[i])
    return out^


def strip_combining(tables: Unicode, points: List[Int]) -> List[Int]:
    """Drop every mark, spacing and enclosing ones included.

    Wider than `strip_marks` by the two categories a BERT normalizer keeps.
    The two are not interchangeable and the difference is not academic: a
    `StripAccents` step written on its own drops all three categories, while
    the `strip_accents` flag inside a `BertNormalizer` drops only the non
    spacing ones. Both spellings are in the corpus, the reference has one
    function for each, and running one where the other belongs loses the
    enclosing keycap off a digit or the vowel sign off a Devanagari syllable.
    """
    var out = List[Int]()
    out.reserve(len(points))
    for i in range(len(points)):
        var category = tables.category(points[i])
        if category < CAT_MN or category > CAT_ME:
            out.append(points[i])
    return out^
