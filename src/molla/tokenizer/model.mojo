"""The model stage: BPE, WordPiece and Unigram.

This is the part that turns a piece of text into ids, and the three algorithms
have almost nothing in common beyond that sentence.

BPE starts from single characters and repeatedly glues together the adjacent
pair with the lowest rank in the merge table, which is the order the pairs were
learned in. WordPiece walks left to right taking the longest prefix that is in
the vocabulary, marking every piece after the first with a prefix. Unigram runs
a Viterbi pass over every substring that is a token and keeps the segmentation
with the best total log probability.

The one thing they do share is that they are called once per pre-tokenized
piece, and real text repeats its pieces constantly, so a cache in front of all
three is worth more than any tuning inside them.
"""

from .vocab import NO_ID, Merges, Vocab, _hash

comptime M_BPE = 0
comptime M_WORDPIECE = 1
comptime M_UNIGRAM = 2
comptime M_WORDLEVEL = 3

comptime UNK_PENALTY = 10.0
"""How much worse than the worst real token an unknown character is.

Sentencepiece picked this, Hugging Face copied it, and the value matters
because it is what stops Viterbi from covering a whole word in unknowns when
one character in the middle is not in the vocabulary.
"""


def _char_width(lead: UInt8) -> Int:
    """How many bytes the character starting with this one takes.

    A byte that cannot start a character counts as one, because the caller is
    walking bytes that came out of our own encoder and a wrong answer here
    should move the cursor rather than stop it.
    """
    if lead < 0x80:
        return 1
    if lead >= 0xF0:
        return 4
    if lead >= 0xE0:
        return 3
    if lead >= 0xC0:
        return 2
    return 1


def _hex_digit(value: Int) -> UInt8:
    if value < 10:
        return UInt8(48 + value)
    return UInt8(55 + value)


struct Work(Movable):
    """Everything the model needs to scribble on, owned by the caller.

    It is one object rather than a few locals because encode is called once per
    piece and every one of these lists would otherwise be allocated and freed
    per piece. The cache is here for the same reason and does most of the work:
    a page of English is a few thousand pieces and a few hundred distinct ones.
    """

    var sym_id: List[Int]
    var sym_len: List[Int]
    var sym_prev: List[Int]
    var sym_next: List[Int]
    var heap_rank: List[Int]
    var heap_pos: List[Int]
    var heap_new: List[Int]
    var bytes: List[UInt8]
    var score: List[Float64]
    var back: List[Int]
    var pick: List[Int]

    var key: List[Int]
    """Cache slots. Each holds an index into `entry_at`, or -1."""

    var entry_at: List[Int]
    var entry_length: List[Int]
    var entry_ids_at: List[Int]
    var entry_ids_length: List[Int]
    var cache_text: List[UInt8]
    var cache_ids: List[Int]
    var mask: Int

    def __init__(out self):
        self.sym_id = List[Int]()
        self.sym_len = List[Int]()
        self.sym_prev = List[Int]()
        self.sym_next = List[Int]()
        self.heap_rank = List[Int]()
        self.heap_pos = List[Int]()
        self.heap_new = List[Int]()
        self.bytes = List[UInt8]()
        self.score = List[Float64]()
        self.back = List[Int]()
        self.pick = List[Int]()
        self.key = List[Int](length=1 << 14, fill=-1)
        self.entry_at = List[Int]()
        self.entry_length = List[Int]()
        self.entry_ids_at = List[Int]()
        self.entry_ids_length = List[Int]()
        self.cache_text = List[UInt8]()
        self.cache_ids = List[Int]()
        self.mask = (1 << 14) - 1

    def _entry_matches(self, entry: Int, text: Span[UInt8, _]) -> Bool:
        if self.entry_length[entry] != len(text):
            return False
        var at = self.entry_at[entry]
        for i in range(len(text)):
            if self.cache_text[at + i] != text[i]:
                return False
        return True

    def lookup(self, text: Span[UInt8, _], mut out: List[Int]) -> Bool:
        """Append the cached ids for these bytes, if there are any."""
        var slot = _hash(text) & self.mask
        while True:
            var entry = self.key[slot]
            if entry < 0:
                return False
            if self._entry_matches(entry, text):
                var at = self.entry_ids_at[entry]
                for i in range(self.entry_ids_length[entry]):
                    out.append(self.cache_ids[at + i])
                return True
            slot = (slot + 1) & self.mask

    def remember(mut self, text: Span[UInt8, _], ids: List[Int], from_at: Int):
        """Record the ids for these bytes.

        The table never evicts and never grows. When it fills up the tokenizer
        keeps working and stops getting faster, which is the right trade for a
        cache that exists to make repeated words cheap: the words that repeat
        are in it long before it fills.
        """
        if len(self.entry_at) * 4 >= (self.mask + 1) * 3:
            return
        if len(text) > 256:
            return
        var slot = _hash(text) & self.mask
        while self.key[slot] >= 0:
            if self._entry_matches(self.key[slot], text):
                return
            slot = (slot + 1) & self.mask
        self.key[slot] = len(self.entry_at)
        self.entry_at.append(len(self.cache_text))
        self.entry_length.append(len(text))
        for i in range(len(text)):
            self.cache_text.append(text[i])
        self.entry_ids_at.append(len(self.cache_ids))
        self.entry_ids_length.append(len(ids) - from_at)
        for i in range(from_at, len(ids)):
            self.cache_ids.append(ids[i])


struct Model(Movable):
    """The vocabulary plus whichever of the three algorithms this file asked
    for."""

    var kind: Int
    var vocab: Vocab
    var merges: Merges
    var score: List[Float64]
    var unk_id: Int
    var fuse_unk: Bool
    var byte_fallback: Bool
    var ignore_merges: Bool
    var continuing_prefix: List[UInt8]
    var end_suffix: List[UInt8]
    var max_input_chars: Int
    var max_token_bytes: Int
    var min_score: Float64
    var byte_id: List[Int]
    """Id of the token spelled `<0xNN>` for each byte, or -1."""

    def __init__(out self, kind: Int, expected: Int):
        self.kind = kind
        self.vocab = Vocab(expected)
        self.merges = Merges(expected if kind == M_BPE else 16)
        self.score = List[Float64]()
        self.unk_id = NO_ID
        self.fuse_unk = False
        self.byte_fallback = False
        self.ignore_merges = False
        self.continuing_prefix = List[UInt8]()
        self.end_suffix = List[UInt8]()
        self.max_input_chars = 100
        self.max_token_bytes = 1
        self.min_score = 0.0
        self.byte_id = List[Int](length=256, fill=NO_ID)

    def seal(mut self):
        """Work out the things that can only be known once the vocabulary is
        complete: the byte fallback ids, the longest token and the worst
        score."""
        var spelling = List[UInt8]()
        for b in range(256):
            spelling.clear()
            spelling.append(UInt8(ord("<")))
            spelling.append(UInt8(ord("0")))
            spelling.append(UInt8(ord("x")))
            spelling.append(_hex_digit(b >> 4))
            spelling.append(_hex_digit(b & 15))
            spelling.append(UInt8(ord(">")))
            self.byte_id[b] = self.vocab.id_of(spelling)
        var longest = 1
        for id in range(self.vocab.size()):
            var length = len(self.vocab.token(id))
            if length > longest:
                longest = length
        self.max_token_bytes = longest
        if len(self.score) > 0:
            var worst = self.score[0]
            for i in range(len(self.score)):
                if self.score[i] < worst:
                    worst = self.score[i]
            self.min_score = worst

    def tokenize(
        self, text: Span[UInt8, _], mut out: List[Int], mut work: Work
    ) raises:
        """One pre-tokenized piece to ids, appended to `out`."""
        if len(text) == 0:
            return
        if self.kind == M_BPE and self.ignore_merges:
            var whole = self.vocab.id_of(text)
            if whole != NO_ID:
                out.append(whole)
                return
        if work.lookup(text, out):
            return
        var from_at = len(out)
        if self.kind == M_BPE:
            self._bpe(text, out, work)
        elif self.kind == M_WORDPIECE:
            self._wordpiece(text, out)
        elif self.kind == M_WORDLEVEL:
            self._wordlevel(text, out)
        else:
            self._unigram(text, out, work)
        work.remember(text, out, from_at)

    def _wordlevel(self, text: Span[UInt8, _], mut out: List[Int]):
        """The whole piece, or the unknown token, and nothing in between.

        There is no sub-word step here at all. The pre-tokenizer decided what a
        word is and either the vocabulary has it or it does not, which is what
        makes this the one model where an unseen word costs exactly one id.
        """
        var whole = self.vocab.id_of(text)
        if whole != NO_ID:
            out.append(whole)
        elif self.unk_id != NO_ID:
            out.append(self.unk_id)

    def _fallback(self, text: Span[UInt8, _], mut out: List[Int]) -> Bool:
        """These bytes as byte fallback tokens, or nothing if any is missing.

        All or nothing is the rule: a vocabulary that has some of the 256 byte
        tokens and not others would otherwise produce a run of ids with a hole
        in the middle of a character.
        """
        for i in range(len(text)):
            if self.byte_id[Int(text[i])] == NO_ID:
                return False
        for i in range(len(text)):
            out.append(self.byte_id[Int(text[i])])
        return True

    def _spell(
        self,
        text: Span[UInt8, _],
        at: Int,
        width: Int,
        first: Bool,
        last: Bool,
        mut into: List[UInt8],
    ):
        """One character with the prefix or the suffix around it.

        The prefix goes on everything except the first character and the suffix
        goes on the last, which is how a WordPiece style vocabulary tells the
        middle of a word from its start even inside BPE.
        """
        into.clear()
        if not first:
            for i in range(len(self.continuing_prefix)):
                into.append(self.continuing_prefix[i])
        for i in range(width):
            into.append(text[at + i])
        if last:
            for i in range(len(self.end_suffix)):
                into.append(self.end_suffix[i])

    def _bpe(
        self, text: Span[UInt8, _], mut out: List[Int], mut work: Work
    ) raises:
        self._symbols(text, work)
        self._merge_all(work)
        for i in range(len(work.sym_id)):
            if work.sym_len[i] > 0:
                out.append(work.sym_id[i])

    def _symbols(self, text: Span[UInt8, _], mut work: Work) raises:
        """The starting symbols: one per character, or one per byte where a
        character is not in the vocabulary and byte fallback is on."""
        work.sym_id.clear()
        work.sym_len.clear()
        var unk_at = -1
        var unk_length = 0
        var at = 0
        while at < len(text):
            var width = _char_width(text[at])
            if width <= 0 or at + width > len(text):
                width = 1
            var last = at + width >= len(text)
            self._spell(text, at, width, at == 0, last, work.bytes)
            var id = self.vocab.id_of(work.bytes)
            var settled = id != NO_ID
            if not settled and self.byte_fallback:
                settled = self._fallback_symbols(work, text, at, width, unk_at)
            if settled and unk_at >= 0:
                self._flush_unk(work, unk_at, unk_length)
                unk_at = -1
                unk_length = 0
            if id != NO_ID:
                work.sym_id.append(id)
                work.sym_len.append(len(work.bytes))
            elif not settled and self.unk_id != NO_ID:
                if self.fuse_unk:
                    if unk_at < 0:
                        unk_at = at
                        unk_length = 0
                    unk_length += len(work.bytes)
                else:
                    work.sym_id.append(self.unk_id)
                    work.sym_len.append(len(work.bytes))
            at += width
        if unk_at >= 0:
            self._flush_unk(work, unk_at, unk_length)

        var count = len(work.sym_id)
        work.sym_prev.clear()
        work.sym_next.clear()
        for i in range(count):
            work.sym_prev.append(i - 1)
            work.sym_next.append(i + 1 if i + 1 < count else -1)

    def _flush_unk(self, mut work: Work, unk_at: Int, unk_length: Int):
        work.sym_id.append(self.unk_id)
        work.sym_len.append(unk_length)

    def _fallback_symbols(
        self,
        mut work: Work,
        text: Span[UInt8, _],
        at: Int,
        width: Int,
        unk_at: Int,
    ) -> Bool:
        """Try to answer with byte tokens. Nothing is appended unless every
        byte of the character has one, so a partial answer cannot leak out."""
        for i in range(width):
            if self.byte_id[Int(text[at + i])] == NO_ID:
                return False
        if unk_at >= 0:
            return True
        for i in range(width):
            work.sym_id.append(self.byte_id[Int(text[at + i])])
            work.sym_len.append(1)
        return True

    def _merge_all(self, mut work: Work) raises:
        """Glue the best pair until no pair is left.

        The queue holds candidate merges by rank, and an entry goes stale as
        soon as one of its two symbols is merged into something else. Rather
        than find and remove those, a popped entry is checked against the list
        it claims to describe and dropped if it no longer fits, which is what
        the check against the merge table below is doing.
        """
        work.heap_rank.clear()
        work.heap_pos.clear()
        work.heap_new.clear()
        for i in range(len(work.sym_id) - 1):
            self._offer(work, i, i + 1)

        while len(work.heap_rank) > 0:
            var pos = work.heap_pos[0]
            var made = work.heap_new[0]
            self._pop(work)
            if work.sym_len[pos] == 0 or work.sym_next[pos] == -1:
                continue
            var right = work.sym_next[pos]
            if self.merges.merged(work.sym_id[pos], work.sym_id[right]) != made:
                continue
            work.sym_id[pos] = made
            work.sym_len[pos] += work.sym_len[right]
            work.sym_len[right] = 0
            var after = work.sym_next[right]
            work.sym_next[pos] = after
            if after != -1:
                work.sym_prev[after] = pos
            var before = work.sym_prev[pos]
            if before != -1:
                self._offer(work, before, pos)
            if after != -1:
                self._offer(work, pos, after)

    def _offer(self, mut work: Work, left: Int, right: Int):
        var rank = self.merges.find(work.sym_id[left], work.sym_id[right])
        if rank < 0:
            return
        var made = self.merges.merged(work.sym_id[left], work.sym_id[right])
        work.heap_rank.append(rank)
        work.heap_pos.append(left)
        work.heap_new.append(made)
        var child = len(work.heap_rank) - 1
        while child > 0:
            var parent = (child - 1) >> 1
            if not self._before(work, child, parent):
                break
            self._swap(work, child, parent)
            child = parent

    def _before(self, work: Work, a: Int, b: Int) -> Bool:
        """Lower rank wins, and an earlier position breaks the tie.

        The tie break is not cosmetic. Two pairs of the same rank in one word
        happen whenever a character repeats, and taking the left one is what
        Hugging Face does, so taking the right one is a different tokenization
        of `aaa`.
        """
        if work.heap_rank[a] != work.heap_rank[b]:
            return work.heap_rank[a] < work.heap_rank[b]
        return work.heap_pos[a] < work.heap_pos[b]

    def _swap(self, mut work: Work, a: Int, b: Int):
        var rank = work.heap_rank[a]
        work.heap_rank[a] = work.heap_rank[b]
        work.heap_rank[b] = rank
        var pos = work.heap_pos[a]
        work.heap_pos[a] = work.heap_pos[b]
        work.heap_pos[b] = pos
        var made = work.heap_new[a]
        work.heap_new[a] = work.heap_new[b]
        work.heap_new[b] = made

    def _pop(self, mut work: Work):
        var last = len(work.heap_rank) - 1
        self._swap(work, 0, last)
        _ = work.heap_rank.pop()
        _ = work.heap_pos.pop()
        _ = work.heap_new.pop()
        var parent = 0
        while True:
            var left = parent * 2 + 1
            var best = parent
            if left < len(work.heap_rank) and self._before(work, left, best):
                best = left
            var right = left + 1
            if right < len(work.heap_rank) and self._before(work, right, best):
                best = right
            if best == parent:
                return
            self._swap(work, parent, best)
            parent = best

    def _wordpiece(self, text: Span[UInt8, _], mut out: List[Int]) raises:
        """Longest prefix that is a token, then repeat from where it ended.

        A word that cannot be covered this way is one unknown token, not a
        partial cover, which is why the ids go into a local list first and are
        thrown away whole if the walk gets stuck.
        """
        var characters = 0
        var at = 0
        while at < len(text):
            var width = _char_width(text[at])
            if width <= 0 or at + width > len(text):
                width = 1
            characters += 1
            at += width
        if characters > self.max_input_chars:
            if self.unk_id != NO_ID:
                out.append(self.unk_id)
            return

        var found = List[Int]()
        var spelling = List[UInt8]()
        var start = 0
        while start < len(text):
            var end = len(text)
            var id = NO_ID
            while end > start:
                spelling.clear()
                if start > 0:
                    for i in range(len(self.continuing_prefix)):
                        spelling.append(self.continuing_prefix[i])
                for i in range(start, end):
                    spelling.append(text[i])
                id = self.vocab.id_of(spelling)
                if id != NO_ID:
                    break
                end -= 1
                while end > start and (text[end] & 0xC0) == 0x80:
                    end -= 1
            if id == NO_ID:
                if self.unk_id != NO_ID:
                    out.append(self.unk_id)
                return
            found.append(id)
            start = end
        for i in range(len(found)):
            out.append(found[i])

    def _unigram(
        self, text: Span[UInt8, _], mut out: List[Int], mut work: Work
    ) raises:
        """Viterbi over every substring that is a token.

        `score` is the best total log probability of covering the first n
        bytes, `back` is where the token that ends there started, and `pick` is
        which token it was. An id of -1 in `pick` means the step was taken as
        an unknown character, which is only allowed to cover exactly one
        character so that a bad byte cannot swallow a word.
        """
        var size = len(text)
        work.score.clear()
        work.back.clear()
        work.pick.clear()
        for _ in range(size + 1):
            work.score.append(0.0)
            work.back.append(-1)
            work.pick.append(NO_ID)
        var unk_score = self.min_score - UNK_PENALTY

        var spelling = List[UInt8]()
        var at = 0
        while at < size:
            var width = _char_width(text[at])
            if width <= 0 or at + width > size:
                width = 1
            var single = False
            var limit = at + self.max_token_bytes
            if limit > size:
                limit = size
            for end in range(at + 1, limit + 1):
                spelling.clear()
                for i in range(at, end):
                    spelling.append(text[i])
                var id = self.vocab.id_of(spelling)
                if id == NO_ID or id >= len(self.score):
                    continue
                var total = work.score[at] + self.score[id]
                if work.back[end] == -1 or total > work.score[end]:
                    work.score[end] = total
                    work.back[end] = at
                    work.pick[end] = id
                if end - at == width:
                    single = True
            if not single:
                var end = at + width
                var total = work.score[at] + unk_score
                if work.back[end] == -1 or total > work.score[end]:
                    work.score[end] = total
                    work.back[end] = at
                    work.pick[end] = NO_ID
            at += width

        var order = List[Int]()
        var order_start = List[Int]()
        var order_end = List[Int]()
        var ends_at = size
        while ends_at > 0:
            var starts_at = work.back[ends_at]
            if starts_at < 0:
                starts_at = ends_at - 1
            order.append(work.pick[ends_at])
            order_start.append(starts_at)
            order_end.append(ends_at)
            ends_at = starts_at

        var i = len(order) - 1
        while i >= 0:
            if order[i] != NO_ID:
                out.append(order[i])
                i -= 1
                continue
            # `order` runs backwards through the text, so the segment after
            # this one in the text is the one before it here, and gathering a
            # run of unknowns forward means walking the index down and pushing
            # the end of the run out.
            var run_start = order_start[i]
            var run_end = order_end[i]
            while i > 0 and order[i - 1] == NO_ID:
                i -= 1
                run_end = order_end[i]
            i -= 1
            self._unknown(text, run_start, run_end, out)

    def _unknown(
        self, text: Span[UInt8, _], start: Int, end: Int, mut out: List[Int]
    ):
        """A stretch nothing covered: byte fallback if the vocabulary has it,
        one unknown token otherwise."""
        var window = text[start:end]
        if self.byte_fallback and self._fallback(window, out):
            return
        if self.unk_id != NO_ID:
            out.append(self.unk_id)
