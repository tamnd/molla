"""The repacked weights on disk, and what makes one safe to reuse.

`molla.nn.repack` says what the planar layout is. This says where a repacked
copy of a model lives, how a second load knows the one it found belongs to the
model it is loading, and how the repack gets written without a second pass over
four gigabytes.

The file sits beside the model as `<model>.molla-repack`. Beside rather than in
a shared cache directory because a repack is roughly the size of the model it
came from, and a directory of things that big somewhere a user did not choose is
the kind of thing that fills a disk and gets molla blamed for it. Beside the
model, the answer to where it went is obvious and deleting it is obvious, which
matters because deleting it has to always be safe. Nothing here is authoritative:
a missing cache, a truncated one, one written by an older layout or one that
belongs to a different model are all the same thing, a miss, and a miss just
means this load repacks again.

What the file is keyed on is the part worth arguing about. The honest key is a
digest of the whole model, and the honest key costs forty five seconds on a
4 GB file, which would make the cache slower than the thing it is caching. So
the key is a sha256 over the GGUF header and the tensor directory together with
the file's length, which is every byte that says what tensors this file has,
what type each one is, what shape it is and where it starts. Change a tensor,
requantize the model, or truncate the file and the key changes. What it does not
catch is somebody rewriting tensor data in place without changing the file
length, and that is a thing no tool does and no download produces. It is the
same bargain every build cache makes and it is written down here rather than
being a surprise later.

Three more things go in the key, and the first two are the ones that would be
wrong silently. The layout version, so a change to what a planar byte means is a
miss and not a misreading. The ggml type of every tensor, which is already in
the directory the digest covers. And the target, which today is the host and is
recorded so that the device repack M3 wants cannot come along and reuse a host
cache.

The write is a temporary file renamed into place at the end. A repack that is
killed halfway leaves a `.molla-repack.tmp` and no cache, rather than a
plausible looking file that the next load trusts and reads garbage out of.
"""

from molla.model.gguf import Gguf
from molla.nn.repack import (
    LAYOUT_VERSION,
    planar_row_bytes,
    repack_row,
    repackable,
)
from molla.nn.tensor import Tensor
from molla.sys.errno import errno_name
from molla.sys.file import (
    close_fd,
    create,
    fsync,
    pwrite_all,
    rename,
    unlink,
)
from molla.sys.mmap import Mapping, RawPtr
from molla.sys.sha256 import DIGEST, Sha256, hex_digest

comptime CACHE_SUFFIX = ".molla-repack"
comptime TEMP_SUFFIX = ".molla-repack.tmp"

comptime MAGIC = "MOLLARPK"
"""Eight bytes at the front, so a file that is not this is rejected on the
first read rather than on the first weight."""

comptime CONTAINER_VERSION = 1
"""The shape of the header and the directory, which can change without the
meaning of a planar byte changing. Kept apart from `LAYOUT_VERSION` for that
reason: they are two things that go out of date for two different causes."""

comptime TARGET_HOST = 0
"""Where the repacked bytes are meant to be read. Only the host today. It is in
the header because a device repack is a different set of bytes and reusing a
host cache for one would be the failure this whole file exists to prevent."""

comptime HEADER_BYTES = 96
"""
    0   8   magic
    8   4   container version
    12  4   layout version
    16  4   target
    20  4   entry count
    24  8   directory offset
    32  8   data offset
    40  8   file length
    48  32  model key
    80  16  reserved, zero
"""

comptime DIR_ALIGN = 4096
"""The data section starts on a page, so the mapping's tensor addresses are the
same distance from a page boundary every time and a load that reads them in
order reads whole pages."""

comptime TENSOR_ALIGN = 64
"""Every tensor starts on a cache line. Rows within a tensor are packed back to
back and are already a multiple of four bytes wide, so this is the only
alignment the layout needs."""

comptime SCRATCH_BYTES = 4 * 1024 * 1024
"""How much one worker repacks before it writes.

Big enough that a write is a real write and not a syscall per row, small enough
that eight workers holding one each is 32 MB and not a fraction of the model.
One buffer per worker for the whole run, allocated before any thread starts,
because a transfer worker reaches its job through a raw address and has nothing
it could safely allocate through.
"""


def cache_path(model: StringSpan) -> String:
    return String(model) + CACHE_SUFFIX


def temp_path(model: StringSpan) -> String:
    return String(model) + TEMP_SUFFIX


def model_key(g: Gguf) -> String:
    """A digest of everything the file says about itself, as hex.

    The header, the metadata and the tensor directory, which is bytes zero to
    `data_start`, plus the file length as eight bytes on the end. The length is
    in there because the directory alone would be identical for a file that had
    been truncated after it, and a truncated model is exactly the case where a
    cache that claimed to match would be read past the end of.
    """
    var h = Sha256()
    h.update(
        Span[UInt8, MutAnyOrigin](
            unsafe_ptr=g.mapping.base(), length=g.data_start
        )
    )
    var tail = List[UInt8]()
    var length = g.mapping.length
    for i in range(8):
        tail.append(UInt8((length >> (i * 8)) & 0xFF))
    h.update(Span(tail))
    return hex_digest(h.digest())


def _put(mut b: List[UInt8], at: Int, value: Int, width: Int):
    for i in range(width):
        b[at + i] = UInt8((value >> (i * 8)) & 0xFF)


def _get(p: RawPtr, at: Int, width: Int) -> Int:
    var out = 0
    for i in range(width):
        out |= Int(p.unsafe_load(at + i)) << (i * 8)
    return out


def _align(n: Int, to: Int) -> Int:
    return ((n + to - 1) // to) * to


def _hex_bytes(text: String) -> List[UInt8]:
    """A hex digest back to the bytes it came from.

    The digest travels as hex because that is what a log line and an error
    message want, and it is stored in the header as bytes because thirty two
    bytes is thirty two bytes. Anything that is not a hex digit reads as zero,
    which cannot happen here and would be a mismatch rather than a wrong answer
    if it did.
    """
    var out = List[UInt8]()
    var n = text.byte_length() // 2
    var raw = text.as_bytes()
    for i in range(n):
        var hi = _nibble(Int(raw[i * 2]))
        var lo = _nibble(Int(raw[i * 2 + 1]))
        out.append(UInt8((hi << 4) | lo))
    return out^


def _nibble(c: Int) -> Int:
    if c >= 48 and c <= 57:
        return c - 48
    if c >= 97 and c <= 102:
        return c - 87
    if c >= 65 and c <= 70:
        return c - 55
    return 0


@fieldwise_init
struct CacheEntry(Copyable, ImplicitlyCopyable, Movable):
    """One repacked tensor in the cache file."""

    var name: String
    var kind: Int
    var cols: Int
    var rows: Int
    var offset: Int
    """From the start of the cache file, not from the data section."""

    var bytes: Int


struct RepackPlan(Movable):
    """Which tensors get repacked, how big each one is, and where it goes.

    Decided from the directory alone, before a byte of tensor data is touched,
    for the same reason `plan_load` is: a machine can be told what a repack
    would cost without having to do it, and the workers that do it need nothing
    from each other because every destination offset was fixed here.
    """

    var index: List[Int]
    """Position in the GGUF directory, so a caller can match a placement to a
    repack without matching on names."""

    var kind: List[Int]
    var cols: List[Int]
    var rows: List[Int]
    var src_off: List[Int]
    """Absolute byte offset in the model file."""

    var dst_off: List[Int]
    """Absolute byte offset in the cache file."""

    var bytes: List[Int]
    var names: List[String]
    var head: List[UInt8]
    """The header and directory, ready to be written at offset zero."""

    var data_at: Int
    var total: Int
    var source_bytes: Int
    """What the same tensors occupy in the model file, so the growth can be
    reported rather than discovered when the disk fills."""

    var key: String

    def __init__(out self, key: String):
        self.index = List[Int]()
        self.kind = List[Int]()
        self.cols = List[Int]()
        self.rows = List[Int]()
        self.src_off = List[Int]()
        self.dst_off = List[Int]()
        self.bytes = List[Int]()
        self.names = List[String]()
        self.head = List[UInt8]()
        self.data_at = 0
        self.total = 0
        self.source_bytes = 0
        self.key = key

    def count(self) -> Int:
        return len(self.index)

    def widest_row(self) raises -> Int:
        var out = 0
        for i in range(self.count()):
            var one = planar_row_bytes(self.kind[i], self.cols[i])
            if one > out:
                out = one
        return out


def plan_repack(g: Gguf) raises -> RepackPlan:
    """Every repackable tensor in the file, laid out in a cache.

    Directory order, which is file order, which is the order the transfer pool
    will fault the source pages in. Writing the destination in the same order
    means the cache file is written front to back rather than seeked around,
    which is what the page cache and the device both want.
    """
    var plan = RepackPlan(model_key(g))
    var entries = List[CacheEntry]()
    for i in range(len(g.tensors)):
        var t = g.tensors[i]
        if not repackable(t.kind):
            continue
        var cols = Int(t.d0)
        var rows = Int(t.d1) if t.n_dims > 1 else 1
        if t.n_dims > 2:
            rows = Int(t.elements()) // cols
        var row = planar_row_bytes(t.kind, cols)
        if row > SCRATCH_BYTES:
            # A row wider than the scratch buffer cannot be written a whole row
            # at a time, and a partial row write is a special case for a shape
            # no model has. Leaving it in the ggml layout is correct and costs
            # one tensor's worth of unpacking.
            continue
        plan.index.append(i)
        plan.kind.append(t.kind)
        plan.cols.append(cols)
        plan.rows.append(rows)
        plan.src_off.append(g.data_start + t.offset)
        plan.bytes.append(row * rows)
        plan.names.append(g.text(t.name))
        entries.append(
            CacheEntry(g.text(t.name), t.kind, cols, rows, 0, row * rows)
        )

    var dir_bytes = 0
    for i in range(len(entries)):
        dir_bytes += _entry_bytes(entries[i].name)
    plan.data_at = _align(HEADER_BYTES + dir_bytes, DIR_ALIGN)

    # The padding goes between tensors and not after the last one. Nothing is
    # written past the end of the final tensor, so a total that counted the pad
    # would be a length no file ever has, and the length is the first thing a
    # reader checks.
    var at = plan.data_at
    var end = plan.data_at
    for i in range(len(entries)):
        entries[i].offset = at
        plan.dst_off.append(at)
        end = at + entries[i].bytes
        at = _align(end, TENSOR_ALIGN)
    plan.total = end

    plan.head = _write_head(plan.key, entries, plan.data_at, plan.total)
    return plan^


def _entry_bytes(name: String) -> Int:
    """One directory entry, padded so the next one starts eight aligned."""
    return _align(40 + name.byte_length(), 8)


def _write_head(
    key: String, entries: List[CacheEntry], data_at: Int, total: Int
) -> List[UInt8]:
    var out = List[UInt8]()
    for _ in range(data_at):
        out.append(0)

    var magic = MAGIC.as_bytes()
    for i in range(8):
        out[i] = magic[i]
    _put(out, 8, CONTAINER_VERSION, 4)
    _put(out, 12, LAYOUT_VERSION, 4)
    _put(out, 16, TARGET_HOST, 4)
    _put(out, 20, len(entries), 4)
    _put(out, 24, HEADER_BYTES, 8)
    _put(out, 32, data_at, 8)
    _put(out, 40, total, 8)
    var raw = _hex_bytes(key)
    for i in range(DIGEST):
        out[48 + i] = raw[i] if i < len(raw) else UInt8(0)

    var at = HEADER_BYTES
    for i in range(len(entries)):
        var one = entries[i]
        _put(out, at, one.kind, 4)
        _put(out, at + 4, one.cols, 8)
        _put(out, at + 12, one.rows, 8)
        _put(out, at + 20, one.offset, 8)
        _put(out, at + 28, one.bytes, 8)
        _put(out, at + 36, one.name.byte_length(), 4)
        var name = one.name.as_bytes()
        for j in range(len(name)):
            out[at + 40 + j] = name[j]
        at += _entry_bytes(one.name)
    return out^


def open_cache_file(path: StringSpan) raises -> Int:
    """Create the temporary file the repack writes into."""
    var fd = create(path)
    if fd.is_err():
        raise Error(
            "could not create " + String(path) + ": " + errno_name(fd.err)
        )
    return fd.value


def write_head(fd: Int, plan: RepackPlan) raises:
    var wrote = pwrite_all(
        fd,
        Pointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(plan.head.unsafe_ptr())
        ),
        len(plan.head),
        0,
    )
    if wrote.is_err() or wrote.value != len(plan.head):
        raise Error("could not write the repack header")


def repack_tensor(
    src_base: Int,
    src_off: Int,
    kind: Int,
    cols: Int,
    rows: Int,
    fd: Int,
    dst_off: Int,
    scratch: Int,
    scratch_bytes: Int,
) -> Int:
    """Repack one tensor from the mapping into the cache file.

    Rows at a time into a fixed buffer and then one write per buffer, so the
    memory a repack needs does not depend on how big the biggest tensor is. The
    return is an errno rather than an exception because this is called from a
    transfer worker, and a worker that raises has nowhere to raise to.

    Called on the thread that just faulted this tensor's pages in, on purpose.
    The source bytes are in cache at that moment and they are never going to be
    warmer, so the repack costs the arithmetic and not the memory traffic.
    """
    var row: Int
    try:
        row = planar_row_bytes(kind, cols)
    except:
        return -1
    if row <= 0 or row > scratch_bytes:
        return -1
    var per_chunk = scratch_bytes // row
    var src = RawPtr(unsafe_from_address=src_base + src_off)
    var dst = RawPtr(unsafe_from_address=scratch)
    var src_row: Int
    try:
        src_row = _ggml_row_bytes(kind, cols)
    except:
        return -1

    var at = 0
    while at < rows:
        var take = per_chunk
        if at + take > rows:
            take = rows - at
        for r in range(take):
            try:
                repack_row(kind, src, (at + r) * src_row, cols, dst, r * row)
            except:
                return -1
        var wrote = pwrite_all(fd, dst, take * row, dst_off + at * row)
        if wrote.is_err():
            return wrote.err
        if wrote.value != take * row:
            return -1
        at += take
    return 0


def _ggml_row_bytes(kind: Int, cols: Int) raises -> Int:
    var t = Tensor(1, kind, cols, 1)
    return t.row_bytes()


def commit_cache(fd: Int, temp: StringSpan, final: StringSpan) raises:
    """Flush, close and rename, in that order and not a different one.

    The rename is what publishes the cache, so everything the cache claims to
    contain has to be on the device before it happens. Renaming first and
    syncing after would leave a window where a crash produces a file that is
    named as complete and is not, which is the one outcome this whole file is
    built to avoid.
    """
    var synced = fsync(fd)
    _ = close_fd(fd)
    if synced.is_err():
        _ = unlink(temp)
        raise Error("could not flush the repack: " + errno_name(synced.err))
    var moved = rename(temp, final)
    if moved.is_err():
        _ = unlink(temp)
        raise Error(
            "could not rename the repack into place: " + errno_name(moved.err)
        )


def abandon_cache(fd: Int, temp: StringSpan):
    """Give up on a repack without leaving anything behind."""
    _ = close_fd(fd)
    _ = unlink(temp)


struct RepackCache(Movable):
    """A cache file that has been opened and believed, or one that has not.

    An unopened cache is not an error state. It is what every first load has and
    what every load with a stale or missing file falls back to, so `usable` is
    checked rather than caught, and everything that reads a weight works the
    same either way.
    """

    var mapping: Mapping
    var entries: List[CacheEntry]
    var usable: Bool
    var reason: String
    """Why the cache was not used, in the words a person would want. Empty when
    it was."""

    def __init__(out self):
        self.mapping = Mapping()
        self.entries = List[CacheEntry]()
        self.usable = False
        self.reason = String("no repack cache was opened")

    def count(self) -> Int:
        return len(self.entries)

    def find(self, name: StringSpan) -> Int:
        if not self.usable:
            return -1
        for i in range(len(self.entries)):
            if self.entries[i].name == name:
                return i
        return -1

    def tensor(self, at: Int) raises -> Tensor:
        """The cached tensor as a planar view into the mapping."""
        if at < 0 or at >= len(self.entries):
            raise Error("repack cache entry " + String(at) + " does not exist")
        var one = self.entries[at]
        var base = Tensor(0, one.kind, one.cols, one.rows)
        return base.as_planar(self.mapping.address + one.offset)

    def bytes(self) -> Int:
        return self.mapping.length

    def close(mut self):
        self.mapping.close()
        self.usable = False


def open_cache(model: StringSpan, key: String) -> RepackCache:
    """Open the cache beside this model, if there is one worth using.

    Never raises. Every reason a cache cannot be used is the same reason as far
    as the caller is concerned, which is that this load repacks, and turning
    each of them into an exception would mean a try around a thing that is not
    exceptional. The reason is recorded so the load can say it out loud, because
    a repack that silently reruns every time is the failure this is here to make
    visible.
    """
    var out = RepackCache()
    var path = cache_path(model)
    var mapping: Mapping
    try:
        mapping = Mapping(path)
    except:
        out.reason = String("no cache file beside the model")
        return out^

    var p = mapping.base()
    if mapping.length < HEADER_BYTES:
        mapping.close()
        out.reason = String("the cache file is too short to have a header")
        return out^

    var magic = MAGIC.as_bytes()
    for i in range(8):
        if p.unsafe_load(i) != magic[i]:
            mapping.close()
            out.reason = String("the cache file is not a molla repack")
            return out^

    if _get(p, 8, 4) != CONTAINER_VERSION:
        mapping.close()
        out.reason = String("the cache was written by a different molla")
        return out^
    if _get(p, 12, 4) != LAYOUT_VERSION:
        mapping.close()
        out.reason = String("the cache is an older weight layout")
        return out^
    if _get(p, 16, 4) != TARGET_HOST:
        mapping.close()
        out.reason = String("the cache was repacked for a different target")
        return out^
    if _get(p, 40, 8) != mapping.length:
        mapping.close()
        out.reason = String("the cache is truncated")
        return out^

    var want = _hex_bytes(key)
    for i in range(DIGEST):
        var have = Int(p.unsafe_load(48 + i))
        if have != Int(want[i] if i < len(want) else UInt8(0)):
            mapping.close()
            out.reason = String("the cache belongs to a different model")
            return out^

    var count = _get(p, 20, 4)
    var at = _get(p, 24, 8)
    var data_at = _get(p, 32, 8)
    var entries = List[CacheEntry]()
    for _ in range(count):
        if at + 40 > data_at:
            mapping.close()
            out.reason = String("the cache directory runs past its data")
            return out^
        var kind = _get(p, at, 4)
        var cols = _get(p, at + 4, 8)
        var rows = _get(p, at + 12, 8)
        var offset = _get(p, at + 20, 8)
        var bytes = _get(p, at + 28, 8)
        var name_len = _get(p, at + 36, 4)
        if at + 40 + name_len > data_at:
            mapping.close()
            out.reason = String("a cache entry names more than it has room for")
            return out^
        if offset < data_at or offset + bytes > mapping.length:
            mapping.close()
            out.reason = String("a cache entry points outside the file")
            return out^
        var name = List[UInt8]()
        for j in range(name_len):
            name.append(p.unsafe_load(at + 40 + j))
        entries.append(
            CacheEntry(
                String(StringSpan(unsafe_from_utf8=Span(name))),
                kind,
                cols,
                rows,
                offset,
                bytes,
            )
        )
        at = _align(at + 40 + name_len, 8)

    out.mapping = mapping^
    out.entries = entries^
    out.usable = True
    out.reason = String("")
    return out^
