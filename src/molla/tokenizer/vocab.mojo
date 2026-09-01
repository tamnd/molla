"""The vocabulary, and the merge table that sits next to it.

Both are hash tables and both are open addressed, which is worth one paragraph
because the shape of these two decides what a tokenizer costs.

A vocabulary is a quarter of a million short byte strings. Holding them as a
`Dict[String, Int]` means a heap allocation per token at load time and a
`String` built per lookup at encode time, and encode does one lookup per
character of input before it does anything else. So the strings live end to end
in one arena, an id is an offset and a length into it, and the table maps a
hash of the bytes to an id. Nothing allocates per token and a lookup touches
two cache lines.

The merge table is keyed by a pair of ids rather than by a pair of strings.
Hugging Face stores merges as text, but every part of every merge is already a
token, so the table is built once at load time into pairs of ids and the inner
loop of BPE compares integers. A merge also knows what token it produces, which
saves a second lookup on every merge that fires.
"""

from molla.sys.mem import as_ptr

comptime NO_ID = -1
"""What a lookup returns when the bytes are not in the vocabulary."""


def _hash(data: Span[UInt8, _]) -> Int:
    """FNV-1a, 64 bit, then mixed.

    FNV because the keys are short and it has no setup cost, and the extra mix
    at the end because FNV leaves the low bits of a short key doing most of the
    work and an open addressed table takes its slot from the low bits.
    """
    var value = 0xCBF29CE484222325
    for i in range(len(data)):
        value = (value ^ Int(data[i])) * 0x100000001B3
        value &= 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    value = (value * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 29
    return value & 0x7FFFFFFFFFFFFFFF


def _capacity_for(count: Int) -> Int:
    """A power of two with room to spare, because open addressing degrades
    badly past about three quarters full."""
    var size = 16
    while size * 3 < count * 4:
        size <<= 1
    return size


struct Vocab(Movable):
    """Token bytes in an arena, with a map from the bytes back to the id."""

    var arena: List[UInt8]
    var at: List[Int]
    """Where token `id` starts in the arena, or -1 for an id nothing uses."""

    var length: List[Int]
    var slot: List[Int]
    """The table. Each slot holds an id or -1, and collisions walk forward."""

    var mask: Int
    var count: Int

    def __init__(out self, expected: Int):
        self.arena = List[UInt8]()
        self.arena.reserve(expected * 8)
        self.at = List[Int]()
        self.length = List[Int]()
        var size = _capacity_for(expected)
        self.slot = List[Int](length=size, fill=NO_ID)
        self.mask = size - 1
        self.count = 0

    def _grow(mut self):
        var size = (self.mask + 1) << 1
        var moved = List[Int](length=size, fill=NO_ID)
        var mask = size - 1
        for i in range(len(self.slot)):
            var id = self.slot[i]
            if id == NO_ID:
                continue
            var probe = _hash(self.token(id)) & mask
            while moved[probe] != NO_ID:
                probe = (probe + 1) & mask
            moved[probe] = id
        self.slot = moved^
        self.mask = mask

    def _reserve_id(mut self, id: Int):
        while len(self.at) <= id:
            self.at.append(-1)
            self.length.append(0)

    def add(mut self, text: Span[UInt8, _], id: Int):
        """Record that `id` is these bytes.

        A repeated id keeps the first spelling and a repeated spelling keeps the
        first id, because a `tokenizer.json` with either in it is broken in a
        way this cannot fix and picking a side quietly is better than picking
        one loudly on every lookup.
        """
        self._reserve_id(id)
        if self.at[id] >= 0:
            return
        self.at[id] = len(self.arena)
        self.length[id] = len(text)
        for i in range(len(text)):
            self.arena.append(text[i])

        if self.count * 4 >= (self.mask + 1) * 3:
            self._grow()
        var probe = _hash(text) & self.mask
        while self.slot[probe] != NO_ID:
            if self._equals(self.slot[probe], text):
                return
            probe = (probe + 1) & self.mask
        self.slot[probe] = id
        self.count += 1

    def _equals(self, id: Int, text: Span[UInt8, _]) -> Bool:
        if self.length[id] != len(text):
            return False
        var start = self.at[id]
        for i in range(len(text)):
            if self.arena[start + i] != text[i]:
                return False
        return True

    def id_of(self, text: Span[UInt8, _]) -> Int:
        var probe = _hash(text) & self.mask
        while True:
            var id = self.slot[probe]
            if id == NO_ID:
                return NO_ID
            if self._equals(id, text):
                return id
            probe = (probe + 1) & self.mask

    def token(self, id: Int) -> Span[UInt8, MutAnyOrigin]:
        """The bytes of a token. An id nothing uses is an empty span."""
        var base = Int(self.arena.unsafe_ptr())
        if id < 0 or id >= len(self.at) or self.at[id] < 0:
            return Span[UInt8, MutAnyOrigin](unsafe_ptr=as_ptr(base), length=0)
        return Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(base + self.at[id]), length=self.length[id]
        )

    def has(self, id: Int) -> Bool:
        return id >= 0 and id < len(self.at) and self.at[id] >= 0

    def size(self) -> Int:
        """One past the highest id, which is what an array indexed by id
        needs, and not the number of tokens, which is smaller when the ids
        have holes in them."""
        return len(self.at)


struct Merges(Movable):
    """A map from a pair of token ids to the rank and result of merging them."""

    var key: List[Int]
    """The pair packed as `left << 32 | right`, or -1 for an empty slot."""

    var rank: List[Int]
    var result: List[Int]
    var mask: Int
    var count: Int

    def __init__(out self, expected: Int):
        var size = _capacity_for(expected)
        self.key = List[Int](length=size, fill=-1)
        self.rank = List[Int](length=size, fill=0)
        self.result = List[Int](length=size, fill=NO_ID)
        self.mask = size - 1
        self.count = 0

    def _slot(self, key: Int) -> Int:
        var mixed = (key * 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF
        var probe = (mixed >> 17) & self.mask
        while self.key[probe] != -1 and self.key[probe] != key:
            probe = (probe + 1) & self.mask
        return probe

    def _grow(mut self):
        var size = (self.mask + 1) << 1
        var keys = self.key.copy()
        var ranks = self.rank.copy()
        var results = self.result.copy()
        self.key = List[Int](length=size, fill=-1)
        self.rank = List[Int](length=size, fill=0)
        self.result = List[Int](length=size, fill=NO_ID)
        self.mask = size - 1
        for i in range(len(keys)):
            if keys[i] == -1:
                continue
            var probe = self._slot(keys[i])
            self.key[probe] = keys[i]
            self.rank[probe] = ranks[i]
            self.result[probe] = results[i]

    def add(mut self, left: Int, right: Int, rank: Int, result: Int):
        if self.count * 4 >= (self.mask + 1) * 3:
            self._grow()
        var key = (left << 32) | right
        var probe = self._slot(key)
        if self.key[probe] == key:
            return
        self.key[probe] = key
        self.rank[probe] = rank
        self.result[probe] = result
        self.count += 1

    def find(self, left: Int, right: Int) -> Int:
        """The rank of merging this pair, or -1 if they do not merge."""
        var key = (left << 32) | right
        var probe = self._slot(key)
        if self.key[probe] != key:
            return -1
        return self.rank[probe]

    def merged(self, left: Int, right: Int) -> Int:
        """What the pair merges into, or `NO_ID`."""
        var key = (left << 32) | right
        var probe = self._slot(key)
        if self.key[probe] != key:
            return NO_ID
        return self.result[probe]
