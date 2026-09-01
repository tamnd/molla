"""Added tokens, and the trie that finds them.

An added token is a string that has to come out as exactly one id no matter
what the normalizer and the pre-tokenizer would otherwise do to it. It is how
`<|im_start|>` survives a pipeline that would happily split it into six pieces,
and it is why a chat template can be written as text at all.

Finding them is a search for any of a few thousand strings at every position of
the input, which sounds expensive and is not, because almost every byte cannot
start any of them. One lookup in a table of 256 booleans rejects the whole set,
and the trie below is only walked when that lookup says it might be worth it.

Two flags on each token make the match wider than the string. `lstrip` eats the
whitespace before it and `rstrip` eats the whitespace after, which is how a
template that writes a space around a special token gets the same ids as one
that does not. `single_word` makes the match narrower instead: it only counts
when the string is not glued to a word on either side.
"""

from molla.text.props import is_whitespace
from molla.text.utf8 import encode


struct AddedToken(Copyable, Movable):
    var id: Int
    var at: Int
    var length: Int
    var single_word: Bool
    var lstrip: Bool
    var rstrip: Bool
    var normalized: Bool
    var special: Bool

    def __init__(out self, id: Int):
        self.id = id
        self.at = 0
        self.length = 0
        self.single_word = False
        self.lstrip = False
        self.rstrip = False
        self.normalized = False
        self.special = False


struct Trie(Movable):
    """A byte trie whose edges live in one open addressed table.

    A node per byte of every token is tens of thousands of nodes, and a child
    array per node would be a megabyte of mostly nothing, so an edge is a key
    of node and byte in a hash table instead. It costs one probe per byte of a
    candidate match and nothing at all for a byte that cannot start one.
    """

    var first: List[Bool]
    var key: List[Int]
    var value: List[Int]
    var mask: Int
    var count: Int
    var token: List[Int]
    var nodes: Int
    var depth: Int

    def __init__(out self):
        self.first = List[Bool](length=256, fill=False)
        self.key = List[Int](length=1024, fill=-1)
        self.value = List[Int](length=1024, fill=-1)
        self.mask = 1023
        self.count = 0
        self.token = List[Int]()
        self.token.append(-1)
        self.nodes = 1
        self.depth = 0

    def _slot(self, key: Int) -> Int:
        var mixed = (key * 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF
        var slot = (mixed >> 17) & self.mask
        while self.key[slot] != -1 and self.key[slot] != key:
            slot = (slot + 1) & self.mask
        return slot

    def _grow(mut self):
        var size = (self.mask + 1) << 1
        var keys = self.key.copy()
        var values = self.value.copy()
        self.key = List[Int](length=size, fill=-1)
        self.value = List[Int](length=size, fill=-1)
        self.mask = size - 1
        for i in range(len(keys)):
            if keys[i] == -1:
                continue
            var slot = self._slot(keys[i])
            self.key[slot] = keys[i]
            self.value[slot] = values[i]

    def step(self, node: Int, byte: Int) -> Int:
        var slot = self._slot((node << 8) | byte)
        if self.key[slot] == -1:
            return -1
        return self.value[slot]

    def add(mut self, text: Span[UInt8, _], index: Int):
        if len(text) == 0:
            return
        self.first[Int(text[0])] = True
        if len(text) > self.depth:
            self.depth = len(text)
        var node = 0
        for i in range(len(text)):
            if self.count * 4 >= (self.mask + 1) * 3:
                self._grow()
            var edge = (node << 8) | Int(text[i])
            var slot = self._slot(edge)
            if self.key[slot] == -1:
                self.key[slot] = edge
                self.value[slot] = self.nodes
                self.count += 1
                self.token.append(-1)
                self.nodes += 1
            node = self.value[slot]
        self.token[node] = index

    def longest(self, data: List[Int], at: Int) -> Int:
        """The longest token starting at this code point, or -1.

        Longest rather than first, because `<|im_start|>` and `<|im|>` can both
        be in the same file and the shorter one winning would leave the rest of
        the longer one as ordinary text.

        The input is counted in code points and the trie is keyed on bytes, so
        each code point is encoded as it is walked. Nothing is encoded up front
        because the answer for almost every position is the first line, and
        building a byte copy of the whole input plus the two maps between the
        two ways of counting cost more than every other part of this stage put
        together.
        """
        if not self.first[_leading_byte(data[at])]:
            return -1
        var one = List[UInt8]()
        var node = 0
        var best = -1
        var i = at
        while i < len(data):
            one.clear()
            encode(data[i], one)
            var stuck = False
            for j in range(len(one)):
                node = self.step(node, Int(one[j]))
                if node < 0:
                    stuck = True
                    break
            if stuck:
                break
            i += 1
            if self.token[node] >= 0:
                best = self.token[node]
        return best

    def length_of(self, data: List[Int], at: Int, index: Int) -> Int:
        """How many code points the match found by `longest` covered."""
        var one = List[UInt8]()
        var node = 0
        var best = 0
        var i = at
        while i < len(data):
            one.clear()
            encode(data[i], one)
            var stuck = False
            for j in range(len(one)):
                node = self.step(node, Int(one[j]))
                if node < 0:
                    stuck = True
                    break
            if stuck:
                break
            i += 1
            if self.token[node] == index:
                best = i - at
        return best


struct AddedVocabulary(Movable):
    """The added tokens, their text, and one trie per matching stage."""

    var tokens: List[AddedToken]
    var arena: List[UInt8]
    var raw: Trie
    """Tokens matched against the text as it arrived."""

    var cooked: Trie
    """Tokens matched after the normalizer has run."""

    var by_id: List[Int]
    """Index into `tokens` for an id, or -1, sized to the highest added id."""

    def __init__(out self):
        self.tokens = List[AddedToken]()
        self.arena = List[UInt8]()
        self.raw = Trie()
        self.cooked = Trie()
        self.by_id = List[Int]()

    def add(mut self, var token: AddedToken, text: Span[UInt8, _]):
        token.at = len(self.arena)
        token.length = len(text)
        for i in range(len(text)):
            self.arena.append(text[i])
        var index = len(self.tokens)
        if token.normalized:
            self.cooked.add(text, index)
        else:
            self.raw.add(text, index)
        while len(self.by_id) <= token.id:
            self.by_id.append(-1)
        self.by_id[token.id] = index
        self.tokens.append(token^)

    def is_empty(self) -> Bool:
        return len(self.tokens) == 0

    def is_special(self, id: Int) -> Bool:
        if id < 0 or id >= len(self.by_id) or self.by_id[id] < 0:
            return False
        return self.tokens[self.by_id[id]].special

    def split(
        self,
        normalized: Bool,
        data: List[Int],
        mut piece_start: List[Int],
        mut piece_end: List[Int],
        mut piece_token: List[Int],
    ):
        """Cut a run of code points around every added token in it.

        The output covers the whole input. A stretch with a token index of -1
        is ordinary text for the rest of the pipeline, and a stretch with an
        index is one added token that nothing downstream is allowed to touch.
        """
        if normalized:
            self._split_with(
                self.cooked, data, piece_start, piece_end, piece_token
            )
        else:
            self._split_with(
                self.raw, data, piece_start, piece_end, piece_token
            )

    def _split_with(
        self,
        trie: Trie,
        data: List[Int],
        mut piece_start: List[Int],
        mut piece_end: List[Int],
        mut piece_token: List[Int],
    ):
        if trie.depth == 0 or len(data) == 0:
            if len(data) > 0:
                piece_start.append(0)
                piece_end.append(len(data))
                piece_token.append(-1)
            return

        var at = 0
        var plain = 0
        while at < len(data):
            var index = trie.longest(data, at)
            if index < 0:
                at += 1
                continue
            var width = trie.length_of(data, at, index)
            var start = at
            var end = at + width
            if self.tokens[index].single_word and not self._standalone(
                data, start, end
            ):
                at += 1
                continue
            if self.tokens[index].lstrip:
                while start > plain and is_whitespace(data[start - 1]):
                    start -= 1
            var stop = end
            if self.tokens[index].rstrip:
                while stop < len(data) and is_whitespace(data[stop]):
                    stop += 1
            if start > plain:
                piece_start.append(plain)
                piece_end.append(start)
                piece_token.append(-1)
            piece_start.append(start)
            piece_end.append(stop)
            piece_token.append(index)
            plain = stop
            at = stop
        if plain < len(data):
            piece_start.append(plain)
            piece_end.append(len(data))
            piece_token.append(-1)

    def _standalone(self, data: List[Int], start: Int, end: Int) -> Bool:
        if start > 0 and _is_word(data[start - 1]):
            return False
        return not (end < len(data) and _is_word(data[end]))

    def id_at(self, index: Int) -> Int:
        return self.tokens[index].id

    def append_text(self, id: Int, mut into: List[UInt8]) -> Bool:
        """The text of an added id, appended. False when the id is not one.

        Needed because an added token does not have to be in the model
        vocabulary, and for those the only place its spelling exists is here.
        """
        if id < 0 or id >= len(self.by_id) or self.by_id[id] < 0:
            return False
        var token = self.by_id[id]
        var at = self.tokens[token].at
        for i in range(at, at + self.tokens[token].length):
            into.append(self.arena[i])
        return True


def _is_word(cp: Int) -> Bool:
    if cp >= 48 and cp <= 57:
        return True
    if cp >= 65 and cp <= 90:
        return True
    if cp >= 97 and cp <= 122:
        return True
    return cp == 95 or cp > 127


def _leading_byte(cp: Int) -> Int:
    """The first byte this code point encodes to.

    All the trie needs to reject a position, and cheaper than encoding the
    whole character to find out.
    """
    if cp < 0x80:
        return cp
    if cp < 0x800:
        return 0xC0 | (cp >> 6)
    if cp < 0x10000:
        return 0xE0 | (cp >> 12)
    return 0xF0 | (cp >> 18)
