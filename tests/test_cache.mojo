"""Tests for the repack cache file.

The layout transform itself is tested in `test_repack.mojo` against pointers
into a buffer. What is left here is everything that happens between one load and
the next: a plan that only names tensors with a planar form, a file written by
the transfer pool and read back the way a later process reads it, and the eight
ways a cache can be wrong that all have to end in a miss rather than a wrong
weight.

The wrong cases are made by writing a good cache and then editing one field of
it, because that is the only way to get a file plausible enough to reach the
check being tested. A file full of noise fails at the magic and proves nothing
about the seven checks after it.
"""

from std.ffi import c_int, external_call
from std.memory import bitcast

from harness import Suite
from test_gguf import Builder

from molla.model.gguf import Gguf
from molla.model.load import load, plan_load
from molla.model.repack import (
    CONTAINER_VERSION,
    HEADER_BYTES,
    cache_path,
    model_key,
    open_cache,
    plan_repack,
)
from molla.model.spec import tensor_bytes
from molla.nn.quant import Q_F32, Q_Q4_0, Q_Q8_0, dequant_run
from molla.nn.repack import LAYOUT_PLANAR, unpack_run
from molla.sys.device import DEV_CPU, Device
from molla.sys.mmap import RawPtr

comptime COLS = 64
comptime ROWS = 4


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


def _read(path: StringSpan) raises -> List[UInt8]:
    with open(String(path), "r") as f:
        return f.read_bytes()


def _copy(bytes: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8](capacity=len(bytes))
    for i in range(len(bytes)):
        out.append(bytes[i])
    return out^


def _temp_path() -> String:
    var pid = Int(external_call["getpid", Int32]())
    return String("/tmp/molla_test_cache_") + String(pid) + ".gguf"


struct Rng(Movable):
    """The same small generator the quant tests use, so a fixture is a seed."""

    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt64:
        self.state ^= self.state << 13
        self.state ^= self.state >> 7
        self.state ^= self.state << 17
        return self.state

    def byte(mut self) -> UInt8:
        return UInt8(self.next() & 0xFF)


def _scale_bits(v: Float32) -> UInt64:
    """A float16 bit pattern for a scale that is a power of two.

    Every scale in these fixtures is exact in float16, so a full conversion is
    not needed and a table of the two cases would be longer than this.
    """
    var bits = bitcast[DType.uint32, 1](v)
    var sign = (bits >> 31) & 1
    var exp = Int((bits >> 23) & 0xFF) - 127 + 15
    var mant = (bits >> 13) & 0x3FF
    return UInt64((sign << 15) | (UInt32(exp) << 10) | mant)


def _names() -> List[String]:
    return [
        String("token_embd.weight"),
        String("blk.0.attn_q.weight"),
        String("output_norm.weight"),
    ]


def _kinds() -> List[Int]:
    return [Q_Q8_0, Q_Q4_0, Q_F32]


def _sizes() -> List[Int]:
    """Bytes each tensor takes. Two are quantized and the third is not."""
    return [ROWS * (COLS // 32) * 34, ROWS * (COLS // 32) * 18, COLS * 4]


def _model() -> List[UInt8]:
    """A GGUF file with one Q8_0 tensor, one Q4_0 tensor and one F32 tensor.

    The F32 one is as much the point of the fixture as the other two. It has no
    planar form, so it has to be absent from the plan and absent from the cache
    and still bind, and a fixture where everything was repackable would not
    notice if it were quietly repacked into nonsense.
    """
    var names = _names()
    var kinds = _kinds()
    var sizes = _sizes()

    var b = Builder()
    b.raw("GGUF")
    b.u32(3)
    b.u64(UInt64(len(names)))
    b.u64(1)

    b.kv_header("general.architecture", 8)
    b.gstring("llama")

    var at = 0
    for i in range(len(names)):
        b.gstring(names[i])
        b.u32(2)
        b.u64(UInt64(COLS))
        b.u64(UInt64(ROWS) if kinds[i] != Q_F32 else UInt64(1))
        b.u32(UInt64(kinds[i]))
        b.u64(UInt64(at))
        at += sizes[i]
        at += (32 - (at % 32)) % 32

    b.pad_to(32)

    var rng = Rng(0x9E3779B97F4A7C15)
    # Q8_0 is two bytes of scale and then thirty two signed quants.
    for _ in range(ROWS * (COLS // 32)):
        b.u16(_scale_bits(0.0625))
        for _ in range(32):
            b.u8(rng.byte())
    # Q4_0 is two bytes of scale and then sixteen bytes of packed nibbles.
    for _ in range(ROWS * (COLS // 32)):
        b.u16(_scale_bits(0.25))
        for _ in range(16):
            b.u8(rng.byte())
    for i in range(COLS):
        b.f32(Float32(i) * 0.5 - 8.0)
    while len(b.bytes) % 32 != 0:
        b.u8(0)
    return b^.finish()


def _host() -> Device:
    return Device(DEV_CPU, 0, String("host"), String("test host"))


def run(mut suite: Suite) raises:
    var path = _temp_path()
    var cache = cache_path(path)
    _remove(cache)
    _remove(path)
    _write(path, _model())
    try:
        test_plan(suite, path)
        test_hit(suite, path)
        test_sources(suite, path)
        test_rejects(suite, path)
    finally:
        _remove(cache)
        _remove(path)


def test_plan(mut suite: Suite, path: String) raises:
    var g = Gguf(path)
    var plan = plan_repack(g)
    var names = _names()
    suite.check(plan.count() == 2, "repack plan skips the f32 tensor")
    suite.check(plan.names[0] == names[0], "repack plan is in file order")
    suite.check(plan.names[1] == names[1], "repack plan keeps the second")
    suite.check(
        plan.data_at >= HEADER_BYTES,
        "repack plan puts the data past the directory",
    )
    suite.check(
        plan.total > plan.source_bytes,
        "repack plan grows the bytes it covers",
    )
    suite.check(
        plan.dst_off[1] >= plan.dst_off[0] + plan.bytes[0],
        "repack plan destinations do not overlap",
    )
    suite.check(plan.key == model_key(g), "repack plan carries the model key")
    g.close()


def test_hit(mut suite: Suite, path: String) raises:
    """Write a cache on one load and read it back the way the next one does."""
    var cache = cache_path(path)
    _remove(cache)

    var g = Gguf(path)
    var key = model_key(g)
    var miss = open_cache(path, key)
    suite.check(not miss.usable, "a model with no cache beside it misses")
    suite.check(
        miss.reason == "no cache file beside the model",
        "a missing cache says so",
    )
    miss.close()

    var weights = load(g, plan_load(g, _host(), 0), 1, False, path)
    suite.check(
        weights.report.repacked == 2,
        "the load repacked both quantized tensors",
    )
    suite.check(
        weights.report.repack_note == "repack cache written to " + cache,
        "the load says where the cache went",
    )
    _ = weights^

    var hit = open_cache(path, key)
    suite.check(hit.usable, "the cache the load wrote is usable")
    suite.check(hit.reason == "", "a usable cache has no reason")
    suite.check(hit.count() == 2, "the cache holds both tensors")
    suite.check(
        hit.find("blk.0.attn_q.weight") == 1, "the cache finds a name it has"
    )
    suite.check(
        hit.find("output_norm.weight") == -1,
        "the cache does not find one it has not",
    )

    # The claim the whole cache rests on. A row read out of the cache has to
    # produce the same float32 values as the same row read out of the model
    # file, with no tolerance, or a cached load is a different model.
    var names = _names()
    var kinds = _kinds()
    for i in range(2):
        var t = hit.tensor(hit.find(names[i]))
        suite.check(
            t.layout == LAYOUT_PLANAR, "a cached tensor is planar " + names[i]
        )
        suite.check(
            t.kind == kinds[i], "a cached tensor keeps its type " + names[i]
        )
        suite.check(
            t.cols == COLS, "a cached tensor keeps its width " + names[i]
        )
        suite.check(
            t.rows == ROWS, "a cached tensor keeps its height " + names[i]
        )

        var info = g.tensors[g.tensor_index(names[i])]
        var src = RawPtr(
            unsafe_from_address=g.mapping.address
            + g.data_start
            + Int(info.offset)
        )
        var want = List[Float32](length=COLS * ROWS, fill=0)
        var got = List[Float32](length=COLS * ROWS, fill=0)
        dequant_run(kinds[i], src, 0, COLS * ROWS, want, 0)
        for r in range(ROWS):
            unpack_run(
                t.kind,
                t.layout,
                t.base(),
                r * t.row_bytes(),
                COLS,
                got,
                r * COLS,
            )
        var bad = -1
        for k in range(COLS * ROWS):
            if want[k] != got[k]:
                bad = k
                break
        suite.check(bad == -1, "a cached tensor decodes the same " + names[i])

    hit.close()
    g.close()


def test_sources(mut suite: Suite, path: String) raises:
    """A plan made against a cache reads the cache and not the file.

    This is what makes a device load hold the layout the device kernels want.
    The pool is filled from whatever the plan called each tensor's source, so a
    plan that still points at the model file is a card full of ggml blocks and
    a set of kernels that cannot read them.
    """
    suite.group("load sources")

    var g = Gguf(path)
    var hit = open_cache(path, model_key(g))
    suite.check(hit.usable, "the cache from the last test is still there")

    var plan = plan_load(g, _host(), 0, hit)
    var names = _names()
    var cache_lo = hit.mapping.address
    var cache_hi = cache_lo + hit.mapping.length
    var file_lo = g.mapping.address
    var file_hi = file_lo + g.mapping.length

    var grew = 0
    for i in range(2):
        var at = g.tensor_index(names[i])
        var one = plan.placements[at]
        suite.check(
            one.source >= cache_lo and one.source < cache_hi,
            "a repacked tensor is read out of the cache " + names[i],
        )
        var t = hit.tensor(hit.find(names[i]))
        suite.check(
            one.length == t.bytes(),
            "and its length is the planar length " + names[i],
        )
        suite.check(
            one.bytes == tensor_bytes(g.tensors[at]),
            "while the file size stays on record " + names[i],
        )
        grew += one.length - one.bytes
    # Zero, and that is the point rather than a coincidence. A q8_0 block is 32
    # bytes of quant and a float16 scale and a q4_0 block is 16 and the same
    # scale, and the planar row for both is now exactly those bytes in two
    # planes instead of one block. The layout costs nothing over the file for
    # these two types, which was not true of any version of it before #182.
    suite.check(
        grew == 0, "the planar form of q4_0 and q8_0 is the size of the file"
    )

    # The f32 tensor has no planar form, is not in the cache, and has to still
    # come from the file. A plan that sent every tensor to the cache would be
    # reading past the end of one.
    var norm = g.tensor_index("output_norm.weight")
    suite.check(
        plan.placements[norm].source >= file_lo
        and plan.placements[norm].source < file_hi,
        "a tensor with no planar form is still read out of the file",
    )

    var summed = 0
    for i in range(plan.count()):
        summed += plan.placements[i].length
    suite.check(
        plan.host_bytes == summed,
        "the host figure counts the bytes that get read, not the file's",
    )

    hit.close()
    g.close()


def test_rejects(mut suite: Suite, path: String) raises:
    """Every way a cache can be wrong ends in a miss with a reason.

    The good cache the previous test wrote is read once and edited eight ways,
    so each case differs from a file that works by exactly the field it is
    about.
    """
    var cache = cache_path(path)
    var g = Gguf(path)
    var key = model_key(g)
    var good = _read(cache)
    suite.check(len(good) > HEADER_BYTES, "the cache on disk has a header")

    var short = List[UInt8]()
    for i in range(HEADER_BYTES - 8):
        short.append(good[i])
    _refused(
        suite,
        path,
        key,
        short,
        "a header that was cut off",
        "the cache file is too short to have a header",
    )

    var magic = _copy(good)
    magic[3] = 88
    _refused(
        suite,
        path,
        key,
        magic,
        "the wrong magic",
        "the cache file is not a molla repack",
    )

    var container = _copy(good)
    container[8] = UInt8(CONTAINER_VERSION + 1)
    _refused(
        suite,
        path,
        key,
        container,
        "a container from another molla",
        "the cache was written by a different molla",
    )

    var layout = _copy(good)
    layout[12] = 0
    _refused(
        suite,
        path,
        key,
        layout,
        "an older weight layout",
        "the cache is an older weight layout",
    )

    var target = _copy(good)
    target[16] = 7
    _refused(
        suite,
        path,
        key,
        target,
        "another target",
        "the cache was repacked for a different target",
    )

    var cut = List[UInt8]()
    for i in range(len(good) - 64):
        cut.append(good[i])
    _refused(
        suite,
        path,
        key,
        cut,
        "a file that lost its tail",
        "the cache is truncated",
    )

    var other = _copy(good)
    other[48] = other[48] ^ 0xFF
    _refused(
        suite,
        path,
        key,
        other,
        "a key from another model",
        "the cache belongs to a different model",
    )

    # The key covers the model and not the cache, so a cache whose directory
    # points past its own data still has a key that matches and has to be caught
    # by the bounds check rather than by the digest.
    var far = _copy(good)
    for i in range(8):
        far[HEADER_BYTES + 20 + i] = UInt8((len(good) >> (i * 8)) & 0xFF)
    _refused(
        suite,
        path,
        key,
        far,
        "an entry pointing past the end",
        "a cache entry points outside the file",
    )

    _write(cache, good)
    var again = open_cache(path, key)
    suite.check(again.usable, "the untouched cache still opens")
    again.close()
    g.close()


def _refused(
    mut suite: Suite,
    path: String,
    key: String,
    bytes: List[UInt8],
    what: String,
    reason: String,
) raises:
    _write(cache_path(path), bytes)
    var out = open_cache(path, key)
    suite.check(not out.usable, "a cache with " + what + " is refused")
    suite.check(out.reason == reason, "a cache with " + what + " says why")
    out.close()
