"""Tests for the memory map layer and the GGUF reader.

These build GGUF files byte by byte and read them back. Real models are the
other half of the story and they live in `docs/validation/gguf.md`, because a
67 MB file is not something to put in a git repository and CI cannot download
one on every push. What real models cannot test is the failure paths, since a
model that llama.cpp produced is never truncated and never has bad magic, so
those get a hand built file with exactly one thing wrong with it.

The builder writes little endian because the format is little endian. It does
not go through the reader's own helpers, so a byte order mistake in the reader
cannot cancel itself out.
"""

from std.ffi import c_int, external_call
from std.memory import bitcast

from harness import Suite

from molla.model.gguf import (
    GGUF_ARRAY,
    GGUF_INT32,
    GGUF_STRING,
    Gguf,
    escape,
    file_type_name,
    ggml_type_name,
    quantization,
    value_type_name,
)
from molla.sys.mmap import Mapping


struct Builder(Movable):
    """Assembles a GGUF file in memory."""

    var bytes: List[UInt8]

    def __init__(out self):
        self.bytes = List[UInt8]()

    def u8(mut self, v: UInt8):
        self.bytes.append(v)

    def _le(mut self, v: UInt64, count: Int):
        for i in range(count):
            self.bytes.append(UInt8((v >> UInt64(8 * i)) & 0xFF))

    def u16(mut self, v: UInt64):
        self._le(v, 2)

    def u32(mut self, v: UInt64):
        self._le(v, 4)

    def u64(mut self, v: UInt64):
        self._le(v, 8)

    def f32(mut self, v: Float32):
        self._le(UInt64(bitcast[DType.uint32, 1](v)), 4)

    def f64(mut self, v: Float64):
        self._le(bitcast[DType.uint64, 1](v), 8)

    def raw(mut self, text: StringSpan):
        var p = text.unsafe_ptr()
        for i in range(text.byte_length()):
            self.bytes.append(p.unsafe_load(i))

    def gstring(mut self, text: StringSpan):
        """A length prefixed string, which is how GGUF stores every name."""
        self.u64(UInt64(text.byte_length()))
        self.raw(text)

    def kv_header(mut self, key: StringSpan, kind: Int):
        self.gstring(key)
        self.u32(UInt64(kind))

    def pad_to(mut self, alignment: Int):
        while len(self.bytes) % alignment != 0:
            self.bytes.append(0)

    def finish(deinit self) -> List[UInt8]:
        """Hand the buffer over. Consumes the builder, because moving the list
        out of a builder that is still alive leaves it half destroyed."""
        return self.bytes^


def _c_string(path: StringSpan) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(path.byte_length()):
        out.append(path.unsafe_ptr().unsafe_load(i))
    out.append(0)
    return out^


def _remove(path: StringSpan):
    var buf = _c_string(path)
    _ = external_call["unlink", c_int](buf.unsafe_ptr())


def _write(path: StringSpan, bytes: List[UInt8]) raises:
    with open(String(path), "w") as f:
        f.write_bytes(Span(bytes))


def _temp_path(name: StringSpan) -> String:
    """A path nothing else will collide with.

    CI runs the suite on three platforms at once and a developer can run it
    twice at once, so the pid goes in the name for the same reason every
    listener in test_net binds to port 0.
    """
    var pid = Int(external_call["getpid", Int32]())
    return (
        String("/tmp/molla_test_") + String(name) + "_" + String(pid) + ".gguf"
    )


comptime TENSOR_DATA = 96
"""Bytes of tensor data in the sample: 4 by 8 F16 is 64, then 4 F32 is 16, and
the rest is slack so an offset past the end would still be inside the file."""


def _sample(alignment: Int) -> List[UInt8]:
    """A complete GGUF v3 file covering every value type the reader decodes.

    `alignment` of 0 means write no general.alignment key, so the reader has to
    fall back to 32 the way the specification says.
    """
    var b = Builder()
    b.raw("GGUF")
    b.u32(3)
    b.u64(2)  # tensor count

    var kv_count = 16
    if alignment != 0:
        kv_count += 1
    b.u64(UInt64(kv_count))

    b.kv_header("general.architecture", 8)
    b.gstring("bert")

    # The architecture prefixed key. This is the one that broke when strings
    # came back with a trailing zero inside them, because the lookup was for
    # "bert\0.context_length" and nothing matched.
    b.kv_header("bert.context_length", 4)
    b.u32(512)

    b.kv_header("general.file_type", 4)
    b.u32(1)

    b.kv_header("bert.block_count", 4)
    b.u32(12)

    b.kv_header("bert.embedding_length", 4)
    b.u32(384)

    # Real newline, real tab and a real backslash, all three of which have to
    # survive a round trip through the dump.
    b.kv_header("general.name", 8)
    b.gstring("tiny\nmodel\tx\\y")

    b.kv_header("test.u8", 0)
    b.u8(200)

    b.kv_header("test.i8", 1)
    b.u8(0xFF)  # -1

    b.kv_header("test.u16", 2)
    b.u16(65535)

    b.kv_header("test.i16", 3)
    b.u16(0xFED4)  # -300

    b.kv_header("test.i32", 5)
    b.u32(0xFFFFFFF9)  # -7

    b.kv_header("test.f32", 6)
    b.f32(0.5)

    b.kv_header("test.f64", 12)
    b.f64(0.25)

    b.kv_header("test.bool", 7)
    b.u8(1)

    b.kv_header("test.ints", 9)
    b.u32(UInt64(GGUF_INT32))
    b.u64(3)
    b.u32(1)
    b.u32(2)
    b.u32(3)

    # An array of strings is the one value the reader cannot step over with a
    # multiply, because every element has its own length. A real tokenizer
    # vocabulary is this shape with 30522 entries.
    b.kv_header("test.strs", 9)
    b.u32(UInt64(GGUF_STRING))
    b.u64(2)
    b.gstring("alpha")
    b.gstring("")

    if alignment != 0:
        b.kv_header("general.alignment", 4)
        b.u32(UInt64(alignment))

    b.gstring("token_embd.weight")
    b.u32(2)
    b.u64(4)
    b.u64(8)
    b.u32(1)  # F16
    b.u64(0)

    b.gstring("output_norm.weight")
    b.u32(1)
    b.u64(4)
    b.u32(0)  # F32
    b.u64(64)

    b.pad_to(32 if alignment == 0 else alignment)
    for i in range(TENSOR_DATA):
        b.u8(UInt8(i & 0xFF))
    return b^.finish()


def _truncate(bytes: List[UInt8], length: Int) -> List[UInt8]:
    var out = List[UInt8]()
    for i in range(length):
        out.append(bytes[i])
    return out^


def _raises(path: StringSpan) -> Bool:
    """True when opening the file fails, which is what the bad files want."""
    try:
        var g = Gguf(path)
        g.close()
        return False
    except:
        return True


def run(mut suite: Suite) raises:
    suite.group("mmap")

    var path = _temp_path("sample")
    var sample = _sample(0)
    _write(path, sample)

    var m = Mapping(path)
    suite.check(m.length == len(sample), "maps the whole file")
    suite.check(m.address != 0, "gets an address back")
    suite.check(m.base().unsafe_load(0) == 71, "first byte is G")
    suite.check(
        m.base().unsafe_load(m.length - 1) == UInt8((TENSOR_DATA - 1) & 0xFF),
        "last byte is readable",
    )
    m.close()
    suite.check(not m.mapped, "close is recorded")
    m.close()
    suite.check(True, "closing twice does not fault")

    var missing = _temp_path("does_not_exist")
    var opened = True
    try:
        var gone = Mapping(missing)
        gone.close()
    except:
        opened = False
    suite.check(not opened, "a missing file raises rather than faulting")

    var empty = _temp_path("empty")
    _write(empty, List[UInt8]())
    var mapped_empty = True
    try:
        var e = Mapping(empty)
        e.close()
    except:
        mapped_empty = False
    suite.check(not mapped_empty, "an empty file raises")
    _remove(empty)

    suite.group("gguf header")

    var g = Gguf(path)
    suite.check(g.version == 3, "version")
    suite.check(g.tensor_count == 2, "tensor count")
    suite.check(g.kv_count == 16, "kv count")
    suite.check(len(g.kvs) == 16, "every kv was walked")
    suite.check(len(g.tensors) == 2, "every tensor was walked")
    suite.check(g.alignment == 32, "alignment defaults to 32 when absent")
    suite.check(g.data_start % 32 == 0, "data starts on an alignment boundary")
    suite.check(
        g.data_start + TENSOR_DATA == g.mapping.length,
        "data offset lines up with the end of the file",
    )

    suite.group("gguf values")

    suite.check(g.architecture() == "bert", "architecture")
    suite.check(
        g.architecture().byte_length() == 4,
        "a string has no terminator inside it",
    )
    suite.check(
        g.context_length() == 512, "context length through the prefixed key"
    )
    suite.check(g.block_count() == 12, "block count")
    suite.check(g.embedding_length() == 384, "embedding length")
    suite.check(g.find("general.architecture") == 0, "find hits the first key")
    suite.check(g.find("nothing.at.all") < 0, "find misses cleanly")
    suite.check(g.key_name(2) == "general.file_type", "keys keep file order")

    suite.check(
        g.value_text(g.find("test.u8")) == "200", "uint8 reads as unsigned"
    )
    suite.check(g.value_text(g.find("test.i8")) == "-1", "int8 is signed")
    suite.check(g.value_text(g.find("test.u16")) == "65535", "uint16")
    suite.check(g.value_text(g.find("test.i16")) == "-300", "int16 is signed")
    suite.check(g.value_text(g.find("test.i32")) == "-7", "int32 is signed")
    suite.check(g.value_text(g.find("test.f32")) == "0.5", "float32")
    suite.check(g.value_text(g.find("test.f64")) == "0.25", "float64")
    suite.check(g.value_text(g.find("test.bool")) == "true", "bool")
    suite.check(
        g.value_text(g.find("test.ints")) == "[3 x INT32]",
        "an array reports its shape rather than its contents",
    )
    suite.check(
        g.kvs[g.find("test.ints")].element_kind == GGUF_INT32,
        "array element type",
    )
    suite.check(g.kvs[g.find("test.ints")].count == 3, "array count")
    suite.check(
        g.value_text(g.find("test.strs")) == "[2 x STRING]",
        "an array of strings is stepped over element by element",
    )
    suite.check(
        g.key_name(15) == "test.strs",
        "the key after a string array is still found, so the skip was exact",
    )
    suite.check(
        g.value_text(g.find("general.name")) == "tiny\\nmodel\\tx\\\\y",
        "control characters are escaped in a dump",
    )
    suite.check(
        g.string(g.find("general.name")) == "tiny\nmodel\tx\\y",
        "the string itself is not escaped",
    )

    var not_an_int = False
    try:
        _ = g.uint(g.find("general.architecture"))
    except:
        not_an_int = True
    suite.check(not_an_int, "asking a string for an integer raises")

    suite.check(quantization(g) == "F16", "quantization from general.file_type")

    suite.group("gguf tensors")

    suite.check(g.text(g.tensors[0].name) == "token_embd.weight", "first name")
    suite.check(g.tensors[0].n_dims == 2, "dimension count")
    suite.check(g.tensors[0].dim(0) == 4, "first dimension")
    suite.check(g.tensors[0].dim(1) == 8, "second dimension")
    suite.check(g.tensors[0].elements() == 32, "element count")
    suite.check(g.tensors[0].kind == 1, "tensor type")
    suite.check(g.tensors[0].offset == 0, "first offset")
    suite.check(
        g.text(g.tensors[1].name) == "output_norm.weight", "second name"
    )
    suite.check(g.tensors[1].n_dims == 1, "trailing dimensions are not counted")
    suite.check(g.tensors[1].elements() == 4, "one dimensional element count")
    suite.check(g.tensors[1].offset == 64, "second offset")
    g.close()

    suite.group("gguf alignment")

    var aligned_path = _temp_path("aligned")
    _write(aligned_path, _sample(64))
    var a = Gguf(aligned_path)
    suite.check(a.alignment == 64, "general.alignment is honoured")
    suite.check(a.data_start % 64 == 0, "data respects the stated alignment")
    a.close()
    _remove(aligned_path)

    suite.group("gguf rejects")

    var bad_magic = sample.copy()
    bad_magic[1] = 88
    var bad_magic_path = _temp_path("bad_magic")
    _write(bad_magic_path, bad_magic)
    suite.check(_raises(bad_magic_path), "bad magic")
    _remove(bad_magic_path)

    var bad_version = sample.copy()
    bad_version[4] = 1
    var bad_version_path = _temp_path("bad_version")
    _write(bad_version_path, bad_version)
    suite.check(_raises(bad_version_path), "version 1")
    _remove(bad_version_path)

    # Cut in the middle of the key value block, so the header says there are
    # fourteen keys and the file runs out somewhere around the fourth.
    var short_path = _temp_path("short")
    _write(short_path, _truncate(sample, 120))
    suite.check(_raises(short_path), "truncated inside the metadata")
    _remove(short_path)

    var no_data_path = _temp_path("no_data")
    _write(no_data_path, _truncate(sample, len(sample) - TENSOR_DATA - 1))
    suite.check(_raises(no_data_path), "header runs past the end of the file")
    _remove(no_data_path)

    # A string length prefix of 2^63 is the shape of a hostile file: every
    # arithmetic step after it overflows and the read lands wherever it lands.
    var huge = sample.copy()
    huge[24] = 0
    huge[25] = 0
    huge[26] = 0
    huge[27] = 0
    huge[28] = 0
    huge[29] = 0
    huge[30] = 0
    huge[31] = 0x80
    var huge_path = _temp_path("huge_length")
    _write(huge_path, huge)
    suite.check(_raises(huge_path), "an absurd length prefix is rejected")
    _remove(huge_path)

    suite.group("gguf names")

    suite.check(ggml_type_name(0) == "F32", "ggml type 0")
    suite.check(ggml_type_name(14) == "Q6_K", "ggml type 14")
    suite.check(ggml_type_name(999) == "UNKNOWN", "an unknown ggml type")
    # The two enumerations disagree, which is the whole reason there are two
    # tables. Type 7 is Q5_1 as a tensor and Q8_0 as a file.
    suite.check(ggml_type_name(7) == "Q5_1", "tensor type 7")
    suite.check(file_type_name(7) == "Q8_0", "file type 7")
    suite.check(file_type_name(15) == "Q4_K_M", "file type 15")
    suite.check(value_type_name(GGUF_STRING) == "STRING", "value type 8")
    suite.check(value_type_name(GGUF_ARRAY) == "ARRAY", "value type 9")
    suite.check(escape("a\nb") == "a\\nb", "escape a newline")
    suite.check(escape("a\\b") == "a\\\\b", "escape a backslash")
    suite.check(escape("plain") == "plain", "escape leaves text alone")

    _remove(path)
