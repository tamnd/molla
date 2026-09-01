"""GGUF metadata, read straight out of a memory map.

Nothing here copies the file. The header, the key value block and the tensor
directory are walked in place, and what gets kept is a list of offsets. A key
is a `Region` into the mapping rather than a `String`, and a value is a type tag
plus the offset its payload starts at. Asking for a value seeks back to that
offset and decodes it there. For a 30522 entry tokenizer vocabulary that is the
difference between a few kilobytes of bookkeeping and materialising the whole
thing to answer a question about the architecture.

Every read is bounds checked against the length of the mapping. That is not
defensive programming for its own sake: a GGUF file is attacker controlled input
the moment molla can be pointed at a downloaded model, and the failure mode for
an unchecked length prefix on a memory map is a segfault rather than an error.
The checks cost nothing here because this runs once per file.

Integers are assembled from bytes rather than loaded through a cast. GGUF is
little endian by specification and both of our platforms are too, so a load
would work today. It would also be an unaligned load of a 64 bit value at
whatever offset a variable length key happened to end on, which is a different
gamble, and assembling bytes is correct on any host.

Arrays are located and measured but their elements are skipped, because nothing
that asks about a model needs to decode a thirty thousand entry vocabulary to
answer. Counting one is a field on the key rather than a walk.

What the values mean is not decided here. This module answers what a key holds
and `molla.model.spec` decides what that says about the model, because the
mapping from `<arch>.attention.head_count_kv` to a head count belongs in one
place and not in every caller.
"""

from std.memory import bitcast

from molla.sys.mmap import Mapping, RawPtr

comptime GGUF_MAGIC = 0x47475546
"""The four bytes are G, G, U, F in file order. Assembled little endian that is
0x46554747, so this is the value you get reading them big endian, which is what
the byte by byte comparison below actually does."""

comptime GGUF_UINT8 = 0
comptime GGUF_INT8 = 1
comptime GGUF_UINT16 = 2
comptime GGUF_INT16 = 3
comptime GGUF_UINT32 = 4
comptime GGUF_INT32 = 5
comptime GGUF_FLOAT32 = 6
comptime GGUF_BOOL = 7
comptime GGUF_STRING = 8
comptime GGUF_ARRAY = 9
comptime GGUF_UINT64 = 10
comptime GGUF_INT64 = 11
comptime GGUF_FLOAT64 = 12

comptime GGUF_MAX_DIMS = 4
"""The format fixes this at four. Storing four fields rather than a list keeps
a tensor directory of a few thousand entries free of per tensor allocation."""

comptime DEFAULT_ALIGNMENT = 32
"""What the specification says to assume when general.alignment is absent."""


@fieldwise_init
struct Region(Copyable, ImplicitlyCopyable, Movable):
    """An offset and a length into the mapping. Never owns anything."""

    var start: Int
    var length: Int


@fieldwise_init
struct KeyValue(Copyable, ImplicitlyCopyable, Movable):
    var key: Region
    var kind: Int
    var at: Int
    """Where the payload starts, past the key and the type tag."""

    var element_kind: Int
    """For an array, the type of its elements. Otherwise -1."""

    var count: Int
    """For an array, how many elements. Otherwise 1."""


@fieldwise_init
struct TensorInfo(Copyable, ImplicitlyCopyable, Movable):
    var name: Region
    var n_dims: Int
    var d0: UInt64
    var d1: UInt64
    var d2: UInt64
    var d3: UInt64
    var kind: Int
    var offset: Int
    """Relative to the start of the tensor data section, not the file."""

    def dim(self, i: Int) -> UInt64:
        if i == 0:
            return self.d0
        if i == 1:
            return self.d1
        if i == 2:
            return self.d2
        return self.d3

    def elements(self) -> UInt64:
        var n = UInt64(1)
        for i in range(self.n_dims):
            n *= self.dim(i)
        return n


struct Cursor(Movable):
    """A bounds checked read head over the mapping."""

    var address: Int
    var length: Int
    var at: Int

    def __init__(out self, address: Int, length: Int):
        self.address = address
        self.length = length
        self.at = 0

    def base(self) -> RawPtr:
        return RawPtr(unsafe_from_address=self.address)

    def _need(self, count: Int) raises:
        if count < 0 or self.at + count > self.length:
            raise Error(
                "gguf truncated: wanted "
                + String(count)
                + " bytes at offset "
                + String(self.at)
                + " of "
                + String(self.length)
            )

    def u8(mut self) raises -> UInt64:
        self._need(1)
        var v = UInt64(self.base().unsafe_load(self.at))
        self.at += 1
        return v

    def _le(mut self, count: Int) raises -> UInt64:
        self._need(count)
        var v = UInt64(0)
        for i in range(count):
            v |= UInt64(self.base().unsafe_load(self.at + i)) << UInt64(8 * i)
        self.at += count
        return v

    def u16(mut self) raises -> UInt64:
        return self._le(2)

    def u32(mut self) raises -> UInt64:
        return self._le(4)

    def u64(mut self) raises -> UInt64:
        return self._le(8)

    def skip(mut self, count: Int) raises:
        self._need(count)
        self.at += count

    def string(mut self) raises -> Region:
        """A length prefixed string. Returns where it is, not what it says."""
        var n = Int(self.u64())
        if n < 0:
            raise Error("gguf string length overflowed")
        self._need(n)
        var region = Region(self.at, n)
        self.at += n
        return region


def escape(text: StringSpan) -> String:
    """Make a value safe to print on one line.

    `tokenizer.chat_template` is a Jinja template with real newlines in it, and
    a dump that emits those raw turns one key into forty lines that nothing can
    parse and nobody can read.

    The backslash is escaped as well, which looks like noise on a template that
    is already full of them. Without it the escaping is not reversible: those
    templates contain the two characters backslash and n as literal text, and
    if a real newline also came out as backslash n then a reader could not tell
    the two apart. Escaping the backslash costs a busier line and buys a dump
    that means exactly one thing.
    """
    var out = List[UInt8]()
    out.reserve(text.byte_length())
    var p = text.unsafe_ptr()
    for i in range(text.byte_length()):
        var c = p.unsafe_load(i)
        # Escaped as bytes rather than by appending to a String, so a multi
        # byte UTF-8 sequence is copied through untouched instead of being
        # sliced at a byte that is not a character boundary.
        if c == 10:
            out.append(92)
            out.append(110)
        elif c == 13:
            out.append(92)
            out.append(114)
        elif c == 9:
            out.append(92)
            out.append(116)
        elif c == 92:
            out.append(92)
            out.append(92)
        else:
            out.append(c)
    return String(StringSpan(unsafe_from_utf8=out))


def scalar_size(kind: Int) raises -> Int:
    """Bytes for a fixed width value type. Raises for STRING and ARRAY."""
    if kind == GGUF_UINT8 or kind == GGUF_INT8 or kind == GGUF_BOOL:
        return 1
    if kind == GGUF_UINT16 or kind == GGUF_INT16:
        return 2
    if kind == GGUF_UINT32 or kind == GGUF_INT32 or kind == GGUF_FLOAT32:
        return 4
    if kind == GGUF_UINT64 or kind == GGUF_INT64 or kind == GGUF_FLOAT64:
        return 8
    raise Error("gguf value type " + String(kind) + " is not fixed width")


def _skip_value(mut cur: Cursor, kind: Int) raises:
    """Step over a value of the given type, whatever it is."""
    if kind == GGUF_STRING:
        _ = cur.string()
        return
    if kind == GGUF_ARRAY:
        var element = Int(cur.u32())
        var count = Int(cur.u64())
        if count < 0:
            raise Error("gguf array count overflowed")
        if element == GGUF_STRING:
            for _ in range(count):
                _ = cur.string()
        elif element == GGUF_ARRAY:
            raise Error("gguf nested arrays are not in the format")
        else:
            # Fixed width elements are one multiply rather than a loop, which
            # matters for a 30522 entry vocabulary.
            cur.skip(scalar_size(element) * count)
        return
    cur.skip(scalar_size(kind))


struct Gguf(Movable):
    """A parsed GGUF header, key value block, and tensor directory."""

    var mapping: Mapping
    var version: Int
    var tensor_count: Int
    var kv_count: Int
    var kvs: List[KeyValue]
    var tensors: List[TensorInfo]
    var alignment: Int
    var data_start: Int
    """File offset where tensor data begins, after alignment padding."""

    def __init__(out self, path: StringSpan) raises:
        self.mapping = Mapping(path)
        self.version = 0
        self.tensor_count = 0
        self.kv_count = 0
        self.kvs = List[KeyValue]()
        self.tensors = List[TensorInfo]()
        self.alignment = DEFAULT_ALIGNMENT
        self.data_start = 0
        try:
            self._parse()
        except e:
            self.mapping.close()
            raise e

    def _parse(mut self) raises:
        var cur = Cursor(self.mapping.address, self.mapping.length)

        # Compared byte by byte so the check does not depend on host endianness
        # and so a big endian GGUF is rejected rather than silently misread.
        if self.mapping.length < 4:
            raise Error("not a gguf file: shorter than the magic")
        var m0 = self.mapping.base().unsafe_load(0)
        var m1 = self.mapping.base().unsafe_load(1)
        var m2 = self.mapping.base().unsafe_load(2)
        var m3 = self.mapping.base().unsafe_load(3)
        if m0 != 71 or m1 != 71 or m2 != 85 or m3 != 70:
            raise Error("not a gguf file: bad magic")
        cur.skip(4)

        self.version = Int(cur.u32())
        if self.version != 2 and self.version != 3:
            raise Error("unsupported gguf version " + String(self.version))

        self.tensor_count = Int(cur.u64())
        self.kv_count = Int(cur.u64())
        if self.tensor_count < 0 or self.kv_count < 0:
            raise Error("gguf counts overflowed")

        self.kvs.reserve(self.kv_count)
        for _ in range(self.kv_count):
            var key = cur.string()
            var kind = Int(cur.u32())
            var element = -1
            var count = 1
            var payload = cur.at
            if kind == GGUF_ARRAY:
                # The element type and count are part of the payload, so the
                # recorded offset points at the elements themselves.
                element = Int(cur.u32())
                count = Int(cur.u64())
                payload = cur.at
                if element == GGUF_STRING:
                    for _ in range(count):
                        _ = cur.string()
                elif element == GGUF_ARRAY:
                    raise Error("gguf nested arrays are not in the format")
                else:
                    cur.skip(scalar_size(element) * count)
            else:
                _skip_value(cur, kind)
            self.kvs.append(KeyValue(key, kind, payload, element, count))

        self.tensors.reserve(self.tensor_count)
        for _ in range(self.tensor_count):
            var name = cur.string()
            var n_dims = Int(cur.u32())
            if n_dims < 1 or n_dims > GGUF_MAX_DIMS:
                raise Error("gguf tensor has " + String(n_dims) + " dimensions")
            var d0 = UInt64(1)
            var d1 = UInt64(1)
            var d2 = UInt64(1)
            var d3 = UInt64(1)
            for i in range(n_dims):
                var d = cur.u64()
                if i == 0:
                    d0 = d
                elif i == 1:
                    d1 = d
                elif i == 2:
                    d2 = d
                else:
                    d3 = d
            var kind = Int(cur.u32())
            var offset = Int(cur.u64())
            self.tensors.append(
                TensorInfo(name, n_dims, d0, d1, d2, d3, kind, offset)
            )

        var found = self.find("general.alignment")
        if found >= 0:
            self.alignment = Int(self.uint(found))
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise Error(
                "gguf alignment "
                + String(self.alignment)
                + " is not a power of two"
            )

        # Tensor data starts at the next alignment boundary after the header.
        var pad = (self.alignment - (cur.at % self.alignment)) % self.alignment
        self.data_start = cur.at + pad
        if self.data_start > self.mapping.length:
            raise Error("gguf header runs past the end of the file")

    def close(mut self):
        self.mapping.close()

    def _region_eq(self, region: Region, text: StringSpan) -> Bool:
        if region.length != text.byte_length():
            return False
        var p = text.unsafe_ptr()
        for i in range(region.length):
            if self.mapping.base().unsafe_load(
                region.start + i
            ) != p.unsafe_load(i):
                return False
        return True

    def find(self, key: StringSpan) -> Int:
        """Index of a key, or -1. Linear because a file has tens of keys."""
        for i in range(len(self.kvs)):
            if self._region_eq(self.kvs[i].key, key):
                return i
        return -1

    def text(self, region: Region) -> String:
        """Copy a region out as a string. The only place anything is copied.

        No terminator goes on the end. A StringSpan carries its own length, so
        appending a zero and handing the whole list over puts the zero inside
        the string rather than after it. That is invisible when the result is
        printed on its own and it broke `context_length`, which builds the key
        `<arch>.context_length` and was looking up `bert\\0.context_length`.
        """
        var out = List[UInt8]()
        out.reserve(region.length)
        for i in range(region.length):
            out.append(self.mapping.base().unsafe_load(region.start + i))
        return String(StringSpan(unsafe_from_utf8=out))

    def key_name(self, i: Int) -> String:
        return self.text(self.kvs[i].key)

    def uint(self, i: Int) raises -> UInt64:
        """Read an unsigned value of whatever width it was stored at."""
        var kv = self.kvs[i]
        if kv.kind == GGUF_ARRAY or kv.kind == GGUF_STRING:
            raise Error("gguf key is not an integer")
        var cur = Cursor(self.mapping.address, self.mapping.length)
        cur.at = kv.at
        if kv.kind == GGUF_UINT8 or kv.kind == GGUF_INT8:
            return cur.u8()
        if kv.kind == GGUF_UINT16 or kv.kind == GGUF_INT16:
            return cur.u16()
        if kv.kind == GGUF_UINT32 or kv.kind == GGUF_INT32:
            return cur.u32()
        if kv.kind == GGUF_UINT64 or kv.kind == GGUF_INT64:
            return cur.u64()
        if kv.kind == GGUF_BOOL:
            return cur.u8()
        raise Error("gguf key is not an integer")

    def string(self, i: Int) raises -> String:
        var kv = self.kvs[i]
        if kv.kind != GGUF_STRING:
            raise Error("gguf key is not a string")
        var cur = Cursor(self.mapping.address, self.mapping.length)
        cur.at = kv.at
        return self.text(cur.string())

    def value_text(self, i: Int) raises -> String:
        """Render a value for a dump. Arrays report their shape, not contents.

        A tokenizer vocabulary is a 30522 entry array and printing it is never
        what anyone wanted, so an array shows its element type and count. That
        is also all the spike needs to check against gguf-dump.
        """
        var kv = self.kvs[i]
        if kv.kind == GGUF_ARRAY:
            return (
                "["
                + String(kv.count)
                + " x "
                + String(value_type_name(kv.element_kind))
                + "]"
            )
        if kv.kind == GGUF_STRING:
            return escape(self.string(i))
        var cur = Cursor(self.mapping.address, self.mapping.length)
        cur.at = kv.at
        if kv.kind == GGUF_FLOAT32:
            return String(bitcast[DType.float32, 1](UInt32(cur.u32())))
        if kv.kind == GGUF_FLOAT64:
            return String(bitcast[DType.float64, 1](cur.u64()))
        if kv.kind == GGUF_BOOL:
            return String("true") if cur.u8() != 0 else String("false")
        # Signed types occupy the same bytes, so the value is read unsigned at
        # its real width and then narrowed to the signed type of that width.
        # Narrowing wraps, which is exactly the two's complement reading. Going
        # through Int directly would not, because the value has already been
        # widened to 64 bits by then and 255 would stay 255 instead of -1.
        if kv.kind == GGUF_INT8:
            return String(UInt8(cur.u8()).cast[DType.int8]())
        if kv.kind == GGUF_INT16:
            return String(UInt16(cur.u16()).cast[DType.int16]())
        if kv.kind == GGUF_INT32:
            return String(UInt32(cur.u32()).cast[DType.int32]())
        if kv.kind == GGUF_INT64:
            return String(cur.u64().cast[DType.int64]())
        return String(self.uint(i))

    def flt(self, i: Int) raises -> Float64:
        """Read a float value at whatever width it was stored at."""
        var kv = self.kvs[i]
        var cur = Cursor(self.mapping.address, self.mapping.length)
        cur.at = kv.at
        if kv.kind == GGUF_FLOAT32:
            return bitcast[DType.float32, 1](UInt32(cur.u32())).cast[
                DType.float64
            ]()
        if kv.kind == GGUF_FLOAT64:
            return bitcast[DType.float64, 1](cur.u64())
        raise Error("gguf key is not a float")

    def string_or(self, key: StringSpan, fallback: StringSpan) -> String:
        var i = self.find(key)
        if i < 0:
            return String(fallback)
        try:
            return self.string(i)
        except:
            return String(fallback)

    def uint_or(self, key: StringSpan, fallback: Int) -> Int:
        var i = self.find(key)
        if i < 0:
            return fallback
        try:
            return Int(self.uint(i))
        except:
            return fallback

    def float_or(self, key: StringSpan, fallback: Float64) -> Float64:
        """A float value, or the fallback.

        An integer stored where a float was expected is accepted, because
        writers do emit `rope.freq_base` as 10000 rather than 10000.0 and
        refusing that would report a model with no rope base at all.
        """
        var i = self.find(key)
        if i < 0:
            return fallback
        try:
            return self.flt(i)
        except:
            pass
        try:
            return Float64(self.uint(i))
        except:
            return fallback

    def bool_or(self, key: StringSpan, fallback: Bool) -> Bool:
        var i = self.find(key)
        if i < 0:
            return fallback
        try:
            return self.uint(i) != 0
        except:
            return fallback

    def has(self, key: StringSpan) -> Bool:
        return self.find(key) >= 0

    def array_count(self, key: StringSpan) -> Int:
        """How many elements an array key holds, without decoding any of them.

        This is how the vocabulary gets counted. The elements are still where
        the file put them and none of them are read.
        """
        var i = self.find(key)
        if i < 0 or self.kvs[i].kind != GGUF_ARRAY:
            return 0
        return self.kvs[i].count

    def tensor_index(self, name: StringSpan) -> Int:
        """Index of a tensor by name, or -1."""
        for i in range(len(self.tensors)):
            if self._region_eq(self.tensors[i].name, name):
                return i
        return -1

    def tensor_prefixed(self, prefix: StringSpan) -> Bool:
        """Whether any tensor name starts with this prefix."""
        var p = prefix.unsafe_ptr()
        var n = prefix.byte_length()
        for i in range(len(self.tensors)):
            var region = self.tensors[i].name
            if region.length < n:
                continue
            var same = True
            for j in range(n):
                if self.mapping.base().unsafe_load(
                    region.start + j
                ) != p.unsafe_load(j):
                    same = False
                    break
            if same:
                return True
        return False

    def architecture(self) -> String:
        return self.string_or("general.architecture", "unknown")

    def context_length(self) -> Int:
        """Architecture prefixed, so it has to be looked up by built key."""
        return self.uint_or(self.architecture() + ".context_length", 0)

    def block_count(self) -> Int:
        return self.uint_or(self.architecture() + ".block_count", 0)

    def embedding_length(self) -> Int:
        return self.uint_or(self.architecture() + ".embedding_length", 0)


def ggml_type_name(kind: Int) -> StaticString:
    """Names as ggml spells them, so a dump can be compared to gguf-dump."""
    if kind == 0:
        return "F32"
    if kind == 1:
        return "F16"
    if kind == 2:
        return "Q4_0"
    if kind == 3:
        return "Q4_1"
    if kind == 6:
        return "Q5_0"
    if kind == 7:
        return "Q5_1"
    if kind == 8:
        return "Q8_0"
    if kind == 9:
        return "Q8_1"
    if kind == 10:
        return "Q2_K"
    if kind == 11:
        return "Q3_K"
    if kind == 12:
        return "Q4_K"
    if kind == 13:
        return "Q5_K"
    if kind == 14:
        return "Q6_K"
    if kind == 15:
        return "Q8_K"
    if kind == 16:
        return "IQ2_XXS"
    if kind == 17:
        return "IQ2_XS"
    if kind == 18:
        return "IQ3_XXS"
    if kind == 19:
        return "IQ1_S"
    if kind == 20:
        return "IQ4_NL"
    if kind == 21:
        return "IQ3_S"
    if kind == 22:
        return "IQ2_S"
    if kind == 23:
        return "IQ4_XS"
    if kind == 24:
        return "I8"
    if kind == 25:
        return "I16"
    if kind == 26:
        return "I32"
    if kind == 27:
        return "I64"
    if kind == 28:
        return "F64"
    if kind == 29:
        return "IQ1_M"
    if kind == 30:
        return "BF16"
    if kind == 34:
        return "TQ1_0"
    if kind == 35:
        return "TQ2_0"
    if kind == 39:
        return "MXFP4"
    if kind == 40:
        return "NVFP4"
    if kind == 41:
        return "Q1_0"
    return "UNKNOWN"


def value_type_name(kind: Int) -> StaticString:
    if kind == GGUF_UINT8:
        return "UINT8"
    if kind == GGUF_INT8:
        return "INT8"
    if kind == GGUF_UINT16:
        return "UINT16"
    if kind == GGUF_INT16:
        return "INT16"
    if kind == GGUF_UINT32:
        return "UINT32"
    if kind == GGUF_INT32:
        return "INT32"
    if kind == GGUF_FLOAT32:
        return "FLOAT32"
    if kind == GGUF_BOOL:
        return "BOOL"
    if kind == GGUF_STRING:
        return "STRING"
    if kind == GGUF_ARRAY:
        return "ARRAY"
    if kind == GGUF_UINT64:
        return "UINT64"
    if kind == GGUF_INT64:
        return "INT64"
    if kind == GGUF_FLOAT64:
        return "FLOAT64"
    return "UNKNOWN"


def file_type_name(kind: Int) -> StaticString:
    """Names from ggml's file type enum, which is a different numbering from
    the tensor type enum above. Type 7 is Q8_0 as a file and Q5_1 as a tensor,
    so the two tables must not be used interchangeably."""
    if kind == 0:
        return "F32"
    if kind == 1:
        return "F16"
    if kind == 2:
        return "Q4_0"
    if kind == 3:
        return "Q4_1"
    if kind == 7:
        return "Q8_0"
    if kind == 8:
        return "Q5_0"
    if kind == 9:
        return "Q5_1"
    if kind == 10:
        return "Q2_K"
    if kind == 11:
        return "Q3_K_S"
    if kind == 12:
        return "Q3_K_M"
    if kind == 13:
        return "Q3_K_L"
    if kind == 14:
        return "Q4_K_S"
    if kind == 15:
        return "Q4_K_M"
    if kind == 16:
        return "Q5_K_S"
    if kind == 17:
        return "Q5_K_M"
    if kind == 18:
        return "Q6_K"
    if kind == 19:
        return "IQ2_XXS"
    if kind == 20:
        return "IQ2_XS"
    if kind == 21:
        return "Q2_K_S"
    if kind == 22:
        return "IQ3_XS"
    if kind == 23:
        return "IQ3_XXS"
    if kind == 24:
        return "IQ1_S"
    if kind == 25:
        return "IQ4_NL"
    if kind == 26:
        return "IQ3_S"
    if kind == 27:
        return "IQ3_M"
    if kind == 28:
        return "IQ2_S"
    if kind == 29:
        return "IQ2_M"
    if kind == 30:
        return "IQ4_XS"
    if kind == 31:
        return "IQ1_M"
    if kind == 32:
        return "BF16"
    if kind == 36:
        return "TQ1_0"
    if kind == 37:
        return "TQ2_0"
    if kind == 38:
        return "MXFP4_MOE"
    if kind == 39:
        return "NVFP4"
    if kind == 40:
        return "Q1_0"
    return "UNKNOWN"


def dominant_type(g: Gguf) -> String:
    """The tensor type that the most tensors use.

    Only a fallback for files with no `general.file_type`, because counting
    tensors is not the same question as what the file was quantized to and it
    gives the wrong answer often. In bge-small-en-v1.5-f16 the 123 F32 norm and
    bias tensors outnumber the 74 F16 weight tensors, so counting says F32 for
    a file everyone calls F16. In qwen2.5-0.5b-instruct-q4_k_m it says Q5_0,
    which is not even one of the names in the file name. Counting by bytes
    would be closer but it would still be a guess at a number the file already
    states.
    """
    var best_kind = -1
    var best_count = 0
    var seen = List[Int]()
    var counts = List[Int]()
    for i in range(len(g.tensors)):
        var kind = g.tensors[i].kind
        var found = -1
        for j in range(len(seen)):
            if seen[j] == kind:
                found = j
                break
        if found < 0:
            seen.append(kind)
            counts.append(1)
            found = len(seen) - 1
        else:
            counts[found] += 1
        if counts[found] > best_count:
            best_count = counts[found]
            best_kind = kind
    if best_kind < 0:
        return String("none")
    return String(ggml_type_name(best_kind))


def quantization(g: Gguf) -> String:
    """What the file says it is stored as.

    `general.file_type` is the file's own answer and it is what gguf-dump
    reports, so it is the one to print. It is optional, and when it is missing
    the dominant tensor type is the best guess available, marked as a guess so
    a dump can never be mistaken for a declaration.
    """
    var i = g.find("general.file_type")
    if i >= 0:
        try:
            return String(file_type_name(Int(g.uint(i))))
        except:
            pass
    return dominant_type(g) + " (guessed, no general.file_type)"


def run_gguf(path: StringSpan) raises:
    """Entry point for `molla gguf`. Prints what the M0 spike asks for."""
    var g = Gguf(path)
    print("file:          " + String(path))
    print("gguf version:  " + String(g.version))
    print("architecture:  " + g.architecture())
    print("context length:" + " " + String(g.context_length()))
    print("quantization:  " + quantization(g))
    print("tensor count:  " + String(g.tensor_count))
    print("kv count:      " + String(g.kv_count))
    print("alignment:     " + String(g.alignment))
    print("data offset:   " + String(g.data_start))
    print("")
    print("metadata:")
    for i in range(len(g.kvs)):
        print(
            "  "
            + g.key_name(i)
            + " "
            + String(value_type_name(g.kvs[i].kind))
            + " = "
            + g.value_text(i)
        )
    print("")
    print("tensor directory:")
    for i in range(len(g.tensors)):
        var t = g.tensors[i]
        var dims = String("")
        for d in range(t.n_dims):
            if d > 0:
                dims += ", "
            dims += String(t.dim(d))
        print(
            "  "
            + g.text(t.name)
            + " ["
            + dims
            + "] "
            + String(ggml_type_name(t.kind))
            + " at "
            + String(t.offset)
        )
    g.close()
