"""SHA-256.

Here because content addressed storage is D5 and a blob you did not hash is a
blob you downloaded, not a blob you verified. Mojo 1.0 has hashing for
dictionaries and nothing cryptographic, and OpenSSL is not an option since molla
has to work on a machine without it.

This is the plain FIPS 180-4 version with no SIMD and no unrolling. It runs at
roughly the speed you would expect from that, which is fine for verifying a
manifest and too slow for verifying a twenty gigabyte model file. When that
matters the answer is the CPU's SHA extensions, and this stays as the reference
the fast one gets tested against.

`molla.sys` is the libc boundary and this calls no libc at all, so it is here
only because the alternative is a one file package. If a `molla.hash` grows a
second member it should move.
"""

comptime BLOCK = 64
comptime DIGEST = 32

comptime K: InlineArray[UInt32, 64] = [
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
]
"""The first 32 bits of the fractional parts of the cube roots of the first 64
primes, which is where FIPS 180-4 gets them from."""


def _rotr(x: UInt32, n: UInt32) -> UInt32:
    return (x >> n) | (x << (UInt32(32) - n))


struct Sha256(Movable):
    """A streaming SHA-256. Feed it with `update`, finish with `digest`."""

    var k: InlineArray[UInt32, 64]
    """A runtime copy of the round constants. `K` is comptime and a runtime
    index into a comptime array will not compile, so the table is materialised
    once per hasher rather than once per block."""

    var h: InlineArray[UInt32, 8]
    var buffer: InlineArray[UInt8, BLOCK]
    var buffered: Int
    var total: UInt64

    def __init__(out self):
        self.k = materialize[K]()
        self.h = [
            0x6A09E667,
            0xBB67AE85,
            0x3C6EF372,
            0xA54FF53A,
            0x510E527F,
            0x9B05688C,
            0x1F83D9AB,
            0x5BE0CD19,
        ]
        self.buffer = InlineArray[UInt8, BLOCK](fill=0)
        self.buffered = 0
        self.total = 0

    def _compress(mut self):
        """One 64 byte block, from `self.buffer`."""
        var w = InlineArray[UInt32, 64](fill=0)
        for i in range(16):
            w[i] = (
                (UInt32(self.buffer[i * 4]) << 24)
                | (UInt32(self.buffer[i * 4 + 1]) << 16)
                | (UInt32(self.buffer[i * 4 + 2]) << 8)
                | UInt32(self.buffer[i * 4 + 3])
            )
        for i in range(16, 64):
            var s0 = (
                _rotr(w[i - 15], 7) ^ _rotr(w[i - 15], 18) ^ (w[i - 15] >> 3)
            )
            var s1 = (
                _rotr(w[i - 2], 17) ^ _rotr(w[i - 2], 19) ^ (w[i - 2] >> 10)
            )
            w[i] = w[i - 16] + s0 + w[i - 7] + s1

        var a = self.h[0]
        var b = self.h[1]
        var c = self.h[2]
        var d = self.h[3]
        var e = self.h[4]
        var f = self.h[5]
        var g = self.h[6]
        var hh = self.h[7]

        for i in range(64):
            var s1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
            var ch = (e & f) ^ ((~e) & g)
            var t1 = hh + s1 + ch + self.k[i] + w[i]
            var s0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
            var maj = (a & b) ^ (a & c) ^ (b & c)
            var t2 = s0 + maj
            hh = g
            g = f
            f = e
            e = d + t1
            d = c
            c = b
            b = a
            a = t1 + t2

        self.h[0] += a
        self.h[1] += b
        self.h[2] += c
        self.h[3] += d
        self.h[4] += e
        self.h[5] += f
        self.h[6] += g
        self.h[7] += hh

    def update(mut self, data: Span[UInt8, _]):
        """Absorb some more bytes. Any length, called any number of times."""
        var i = 0
        var n = len(data)
        self.total += UInt64(n)
        while i < n:
            var room = BLOCK - self.buffered
            var take = room if n - i > room else n - i
            for j in range(take):
                self.buffer[self.buffered + j] = data[i + j]
            self.buffered += take
            i += take
            if self.buffered == BLOCK:
                self._compress()
                self.buffered = 0

    def digest(mut self) -> InlineArray[UInt8, DIGEST]:
        """Pad, run the last blocks, and return the 32 byte digest.

        This consumes the state in the sense that calling it twice gives the
        wrong answer the second time. The type does not stop you, because
        enforcing it would mean a consuming method and every caller here hashes
        exactly once.
        """
        var bits = self.total * 8

        self.buffer[self.buffered] = 0x80
        self.buffered += 1
        if self.buffered > BLOCK - 8:
            while self.buffered < BLOCK:
                self.buffer[self.buffered] = 0
                self.buffered += 1
            self._compress()
            self.buffered = 0
        while self.buffered < BLOCK - 8:
            self.buffer[self.buffered] = 0
            self.buffered += 1
        for j in range(8):
            self.buffer[BLOCK - 1 - j] = UInt8((bits >> (UInt64(j) * 8)) & 0xFF)
        self._compress()

        var out = InlineArray[UInt8, DIGEST](fill=0)
        for i in range(8):
            out[i * 4] = UInt8((self.h[i] >> 24) & 0xFF)
            out[i * 4 + 1] = UInt8((self.h[i] >> 16) & 0xFF)
            out[i * 4 + 2] = UInt8((self.h[i] >> 8) & 0xFF)
            out[i * 4 + 3] = UInt8(self.h[i] & 0xFF)
        return out^


comptime HEX = "0123456789abcdef"


def hex_digest(digest: InlineArray[UInt8, DIGEST]) -> String:
    """Lower case hex, which is what an OCI digest looks like after the colon.
    """
    var out = List[UInt8]()
    out.reserve(DIGEST * 2)
    var table = HEX.as_bytes()
    for i in range(DIGEST):
        out.append(table[Int(digest[i] >> 4)])
        out.append(table[Int(digest[i] & 0xF)])
    return String(StringSpan(unsafe_from_utf8=out))


def sha256_hex(data: Span[UInt8, _]) -> String:
    """Hash a whole buffer at once and return it as hex."""
    var state = Sha256()
    state.update(data)
    return hex_digest(state.digest())
