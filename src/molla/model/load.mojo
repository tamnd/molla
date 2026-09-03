"""Getting weights out of a file and into the place a kernel will read them.

A model load is not one operation, it is two that want to happen at the same
time. One is io: four or five gigabytes have to come off a disk, and a single
thread asking for one page at a time will get a fraction of what the device can
do. The other is the host to device copy, which on a discrete card is the
slowest link in the chain and is idle for as long as the io is serial. Doing
them one after another takes the sum of the two, and doing them together takes
the larger. That is the whole design: a pool of transfer threads pulls tensors
off the mapping in file order while the thread that owns the device drains a
queue of finished tensors and enqueues their copies.

Where a tensor ends up is a placement decision made once, in `plan_load`, and
not a branch scattered through the copy loop. There are three answers and they
are not the same:

Host. The tensor stays in the mapping. There is no copy and no allocation, the
kernel reads the file's own pages, and the only work is faulting them in.

Unified. The same thing, on a machine whose accelerator draws from the same
physical memory. There is no copy and no allocation, and what that buys is a
budget rather than an address: the bytes are already in the memory the device
uses, but not at a number a device kernel can follow. This is the placement for
a load whose kernels are going to run on the host anyway.

Device. The tensor is copied into a slot in one large device buffer. One buffer
rather than one per tensor, because a few hundred separate allocations on a card
fragment the address space and cost a driver round trip each, and because the
engine wants one base pointer and a table of offsets. A unified machine takes
this path too when its kernels are going to run on the device. That is not what
this file used to do, and the reason it does now is #152: a Metal kernel handed
a host address reads zeros rather than faulting, so sharing a pool of memory is
not sharing an address space, and a weight a device is going to read has to be
in a pool wherever it runs. The copy itself is the same `enqueue_copy` on both,
which was measured rather than assumed.

The repack rides along on the same threads. A worker that has just faulted a
tensor's pages in is holding the warmest copy of those bytes that will ever
exist, so if a repack is wanted that is the moment to do it, and the layout
transform costs the arithmetic rather than the memory traffic. The alternative,
a second pass after the load, reads four gigabytes twice. What a worker does
with the result is one `pwrite` per few megabytes into a temporary file at an
offset that was decided before any thread started, so the workers never talk to
each other and never talk to this thread. See `molla.model.repack` for the cache
that comes out of it and for what makes one safe to reuse.

Two things about the numbers that took a machine to find out. Free memory
reported by a device is not usable as a budget, because under WSL2 a 4090
reports free equal to total and both below the physical size. So the budget
comes off the total with a reserve held back, and the reserve is not small: a
card that is full is a card that fails the first allocation the kernels make.
And the reserve is a fraction rather than a constant, because ten per cent of a
24 GB card and ten per cent of a 8 GB one are different amounts of the same
thing.
"""

from std.sys.info import has_accelerator

from max.gpu.host import DeviceBuffer, DeviceContext

from molla.model.gguf import Gguf
from molla.model.repack import (
    RepackCache,
    RepackPlan,
    SCRATCH_BYTES,
    abandon_cache,
    cache_path,
    commit_cache,
    model_key,
    open_cache,
    open_cache_file,
    plan_repack,
    repack_tensor,
    temp_path,
    write_head,
)
from molla.model.spec import tensor_bytes
from molla.nn.tensor import (
    WHERE_DEVICE,
    WHERE_HOST,
    WHERE_UNIFIED,
    place_name,
)
from molla.sys.atomic import AtomicBlock
from molla.sys.clock import monotonic_ms
from molla.sys.device import Device, default_device
from molla.sys.mem import keep
from molla.sys.mmap import RawPtr, page_size, will_need
from molla.sys.queue import MpscQueue, round_up_pow2
from molla.sys.thread import (
    Thread,
    ThreadFunc,
    cpu_count,
    set_thread_name,
    sleep_ms,
    spawn,
)

comptime STAGE_PLAN = 0
comptime STAGE_READ = 1
comptime STAGE_COPY = 2
comptime STAGE_REPACK = 3
comptime STAGE_DONE = 4

comptime SLOT_ALIGN = 256
"""Device buffers want their offsets aligned. 256 is the coarsest alignment any
current backend asks for, so one number covers all of them."""

comptime MIB = 1024 * 1024

comptime MIN_RESERVE = 512 * MIB
"""Held back from the device however small it is. A card with nothing left is a
card where the first kv cache allocation fails, which is a worse failure than
one tensor staying on the host."""

comptime RESERVE_SHARE = 10
"""And at least a tenth of the card, because ten per cent of 24 GB and ten per
cent of 8 GB are different amounts of the same thing."""

comptime UNIFIED_RESERVE_SHARE = 4
"""A quarter of a unified machine instead of a tenth.

The reserve on a card is holding back memory nothing else wants. The reserve on
a unified machine is holding back memory the operating system, the page cache
and every other process on the box are drawing from, and a tenth of sixteen
gigabytes is not enough left to run in. A quarter lands close to what Metal
itself recommends as a working set."""

comptime MAX_WORKERS = 8
"""More transfer threads than this stops helping. The limit is the device, and
past a handful of readers the queue depth is deep enough to keep it busy."""


def stage_name(stage: Int) -> String:
    if stage == STAGE_PLAN:
        return String("plan")
    if stage == STAGE_READ:
        return String("read")
    if stage == STAGE_COPY:
        return String("copy")
    if stage == STAGE_REPACK:
        return String("repack")
    return String("done")


@fieldwise_init
struct Placement(Copyable, ImplicitlyCopyable, Movable):
    """Where one tensor goes, decided before anything is read."""

    var index: Int
    """Position in the file's tensor directory, which is also read order."""

    var offset: Int
    """Absolute byte offset in the file, not the offset the directory holds.
    The repack reads from here, so it stays the file's own number even for a
    tensor whose bytes are going to be taken from the cache instead."""

    var bytes: Int
    """What this tensor occupies in the model file."""

    var source: Int
    """Absolute host address of the bytes that will actually be read.

    In the model file's mapping, or in the repack cache's when the cache has a
    copy of this tensor. Which one it is does not need recording, because
    nothing downstream asks: the read stage warms this address and the copy
    stage copies from it. The plan holds addresses into two mappings and is
    only valid while both are open."""

    var length: Int
    """How many bytes that source is. The planar form of a tensor is not the
    same size as the file's, so this is not `bytes` on a cache hit."""

    var place: Int
    var slot: Int
    """Byte offset in the device pool, or -1 for a tensor that is not copied."""


struct Plan(Movable):
    """A whole file's worth of placements, and what they add up to.

    Holds raw addresses into the model file's mapping and, when there was a
    cache to plan against, into the cache's. Both have to stay open for as long
    as the plan is worth anything, which is the same rule a `Bound` follows for
    the same reason.
    """

    var placements: List[Placement]
    var device: Device
    var total_bytes: Int
    var host_bytes: Int
    var device_bytes: Int
    var budget: Int
    var left_behind: Int
    """Tensors that wanted the device and did not fit in the budget. Zero is the
    normal case and a report says so when it is not, because a model that half
    fits runs at a speed nobody expects and should not have to be guessed at."""

    def __init__(out self, device: Device):
        self.placements = List[Placement]()
        self.device = device
        self.total_bytes = 0
        self.host_bytes = 0
        self.device_bytes = 0
        self.budget = 0
        self.left_behind = 0

    def count(self) -> Int:
        return len(self.placements)

    def uses_device(self) -> Bool:
        return self.device_bytes > 0


def device_budget(dev: Device) -> Int:
    """How many bytes of this device molla is willing to fill.

    Off the total rather than off the free figure, on purpose. A 4090 under WSL2
    reports free equal to total and both below the physical size of the card, so
    the free number is not tracking anything there and sizing an allocation from
    it would be sizing it from a constant that happens to look like a
    measurement. The total is wrong in a knowable direction and the reserve
    covers it.

    A unified device gets an answer here and used to get zero. Zero was written
    when a mapped file was believed to be a device visible buffer, which made a
    pool on a unified machine a copy bought for nothing. It is not one. A device
    kernel cannot read the mapping, which is what #152 found, so a weight a
    Metal kernel is going to read needs a pool exactly as much as a weight a
    CUDA kernel is going to read. What differs between them is the reserve and
    not the shape of the answer.
    """
    if not dev.accelerator():
        return 0
    var share = UNIFIED_RESERVE_SHARE if dev.unified() else RESERVE_SHARE
    var reserve = dev.total // share
    if reserve < MIN_RESERVE:
        reserve = MIN_RESERVE
    if dev.total <= reserve:
        return 0
    return dev.total - reserve


def _lookup_only(name: String) -> Bool:
    """Whether this tensor is read one row at a time rather than in full.

    The token embedding is the biggest tensor in most quantized models and the
    least worth having on a card: a forward pass gathers one row per token from
    it and touches nothing else. So when the budget runs short it is the first
    thing to leave behind, which is what llama.cpp does too.
    """
    return name.startswith("token_embd") or name.startswith("tok_embeddings")


def plan_load(g: Gguf, dev: Device, budget: Int) raises -> Plan:
    """Decide where every tensor in this file goes, with no repack cache.

    Every tensor is read from the model file in the layout the file has, which
    is what a first load against a model does and what a load on a machine with
    nowhere to write a cache does forever.
    """
    return plan_load(g, dev, budget, RepackCache())


def plan_load(
    g: Gguf, dev: Device, budget: Int, cache: RepackCache
) raises -> Plan:
    """Decide where every tensor in this file goes.

    Reads the two directories and nothing else, so a machine can be told what a
    load would do without having the memory to do it.

    The cache is what decides where a tensor's bytes come from, and it decides
    it for the host and the device the same way. A tensor the cache has is read
    from the cache in the planar layout, because that is what `bind` is going
    to point the kernels at and warming the file's copy of it would be warming
    pages nothing will touch. A tensor the cache does not have is read from the
    file. So a device pool holds planar bytes on a machine that has a cache and
    ggml bytes on one that does not, and either way the tensor says which.
    """
    var plan = Plan(dev)
    plan.budget = device_budget(dev) if budget < 0 else budget

    var count = len(g.tensors)
    var sizes = List[Int](capacity=count)
    var sources = List[Int](capacity=count)
    var lengths = List[Int](capacity=count)
    var chosen = List[Bool](capacity=count)
    for i in range(count):
        var size = tensor_bytes(g.tensors[i])
        sizes.append(size)
        chosen.append(False)
        plan.total_bytes += size

        var at = g.mapping.address + g.data_start + g.tensors[i].offset
        var length = size
        var hit = cache.find(g.text(g.tensors[i].name))
        if hit >= 0:
            # The same three comparisons `bind` makes, so the bytes this warms
            # are the bytes the kernels will read. A cache entry that does not
            # match the file is one `bind` will skip, and warming it would warm
            # the wrong pages and leave the right ones cold.
            var cached = cache.tensor(hit)
            var rows = Int(g.tensors[i].d1) if g.tensors[i].n_dims > 1 else 1
            if (
                cached.kind == g.tensors[i].kind
                and cached.cols == Int(g.tensors[i].d0)
                and cached.rows == rows
            ):
                at = cached.address
                length = cached.bytes()
        sources.append(at)
        lengths.append(length)

    # Two passes so the tensors that are read every token get the device first,
    # and the embedding gets whatever is left. Within a pass it is directory
    # order, which is file order, which is the order the reads will happen in.
    # Unified devices go through here too. They did not, on the grounds that an
    # accelerator sharing the memory could read the mapping where it lies, and
    # #152 found that it cannot. A pool is how a weight gets an address a kernel
    # can follow, and that is as true on an M4 as on a 4090.
    if dev.accelerator() and plan.budget > 0:
        var used = 0
        for pass_id in range(2):
            for i in range(count):
                var name = g.text(g.tensors[i].name)
                if _lookup_only(name) != (pass_id == 1):
                    continue
                var need = _align(lengths[i], SLOT_ALIGN)
                if used + need > plan.budget:
                    plan.left_behind += 1
                    continue
                used += need
                chosen[i] = True

    var slot = 0
    for i in range(count):
        var place = WHERE_HOST
        var at = -1
        if chosen[i]:
            place = WHERE_DEVICE
            at = slot
            slot += _align(lengths[i], SLOT_ALIGN)
            plan.device_bytes += lengths[i]
        else:
            if dev.accelerator() and dev.unified():
                place = WHERE_UNIFIED
            plan.host_bytes += lengths[i]
        plan.placements.append(
            Placement(
                i,
                g.data_start + g.tensors[i].offset,
                sizes[i],
                sources[i],
                lengths[i],
                place,
                at,
            )
        )
    return plan^


def _align(n: Int, to: Int) -> Int:
    return ((n + to - 1) // to) * to


comptime SLOT_NEXT = 0
"""Next tensor index to claim. One atomic add is the whole scheduler."""

comptime SLOT_BYTES = 1
"""Bytes faulted in so far, which is what the progress line reports."""

comptime SLOT_SINK = 2
"""Where the touched bytes are added up, so the loop cannot be optimised away.
Nothing reads it for meaning."""

comptime SLOT_REPACK_ERR = 3
"""The first errno a repack worker hit, or zero. One slot for all of them,
because the answer to any of them is the same: abandon the cache and load
without one, and say which errno it was."""

comptime SLOT_COUNT = 4


@fieldwise_init
struct WorkerArg(Copyable, ImplicitlyCopyable, Movable):
    """What one transfer thread is handed.

    The job and which worker this is, and the second one only exists because a
    repack needs a scratch buffer it does not share. A thread entry point takes
    one integer, so the two travel as the address of one of these.
    """

    var job: Int
    var index: Int


struct LoadJob(Movable):
    """What the transfer threads share.

    Deliberately flat. A worker reaches this through a raw address handed to a
    thin function entry point, so everything it touches has to be reachable
    without a borrow, and the two lists are read only for the whole run.
    """

    var base: Int
    """Address of the model file's mapping, so a worker needs no reference to
    it. The repack reads from here and only from here."""

    var page: Int
    var count: Int
    var offsets: List[Int]
    """Per tensor, the file offset the repack reads from."""

    var warm: List[Int]
    """Per tensor, the absolute address the read stage faults in. The file for
    most loads and the repack cache for a tensor the cache already has, because
    warming a copy of a weight that nothing is going to read is the one kind of
    io a load can do that is pure waste."""

    var lengths: List[Int]
    """How long each of those is. Not the file size when the source is the
    cache."""
    var cursor: AtomicBlock
    var ready: MpscQueue

    var repack_fd: Int
    """The cache being written, or -1 when this load is not repacking."""

    var repack_kind: List[Int]
    """Per tensor, the ggml type to repack it as, or -1 to leave it alone. One
    entry per placement rather than a list of the repacked ones, so a worker
    that has claimed tensor `i` needs no search to find out what to do with
    it."""

    var repack_cols: List[Int]
    var repack_rows: List[Int]
    var repack_off: List[Int]
    var scratch: List[Int]
    """One buffer address per worker, allocated before any thread starts."""

    var scratch_hold: List[List[UInt8]]
    """What those addresses point into. Held here so the buffers outlive the
    threads reading them, which a list of raw addresses cannot say on its
    own."""

    var scratch_bytes: Int

    def __init__(out self, mut plan: Plan, base: Int) raises:
        self.base = base
        self.page = page_size()
        self.count = plan.count()
        self.offsets = List[Int](capacity=self.count)
        self.warm = List[Int](capacity=self.count)
        self.lengths = List[Int](capacity=self.count)
        for i in range(self.count):
            self.offsets.append(plan.placements[i].offset)
            self.warm.append(plan.placements[i].source)
            self.lengths.append(plan.placements[i].length)
        self.cursor = AtomicBlock(SLOT_COUNT)
        self.ready = MpscQueue(round_up_pow2(self.count + 2))
        if not self.cursor.is_valid() or not self.ready.is_valid():
            raise Error("could not allocate the transfer queue")
        self.repack_fd = -1
        self.repack_kind = List[Int]()
        self.repack_cols = List[Int]()
        self.repack_rows = List[Int]()
        self.repack_off = List[Int]()
        self.scratch = List[Int]()
        self.scratch_hold = List[List[UInt8]]()
        self.scratch_bytes = 0

    def attach_repack(mut self, rp: RepackPlan, fd: Int, workers: Int) raises:
        """Say that the workers should also repack, and give them room to.

        The per tensor arrays are indexed by placement, which is directory
        order, which is what a worker has after it claims one. The scratch is
        allocated here, on this thread, before anything is spawned, because a
        transfer worker reaches everything it touches through a raw address and
        allocating inside one is the kind of thing that works until the day two
        of them do it at once.
        """
        self.repack_fd = fd
        self.scratch_bytes = SCRATCH_BYTES
        for _ in range(self.count):
            self.repack_kind.append(-1)
            self.repack_cols.append(0)
            self.repack_rows.append(0)
            self.repack_off.append(0)
        for j in range(rp.count()):
            var i = rp.index[j]
            if i < 0 or i >= self.count:
                continue
            self.repack_kind[i] = rp.kind[j]
            self.repack_cols[i] = rp.cols[j]
            self.repack_rows[i] = rp.rows[j]
            self.repack_off[i] = rp.dst_off[j]
        for _ in range(workers):
            var one = List[UInt8](capacity=self.scratch_bytes)
            for _ in range(self.scratch_bytes):
                one.append(0)
            self.scratch_hold.append(one^)
        for k in range(workers):
            self.scratch.append(Int(self.scratch_hold[k].unsafe_ptr()))

    def repacking(self) -> Bool:
        return self.repack_fd >= 0


def _transfer(arg: Int) abi("C") -> Int:
    """One transfer thread. Claims tensors, faults their pages in, repacks.

    The touch loop is what actually reads the file. `madvise` starts the reads
    and the loop waits for them, one byte per page, and the byte goes into a
    running sum that is stored at the end so nothing can decide the loop has no
    effect and delete it.

    The repack happens between the touch and the push, which is the whole reason
    it is here rather than in a pass of its own. The pages this tensor lives in
    were pulled in a microsecond ago by this thread, so the transform reads them
    out of cache. Doing it after the load instead would mean reading the model a
    second time from a page cache that a four gigabyte load has already put
    under pressure.

    A repack that fails does not fail the load. The errno goes in a slot, this
    thread carries on faulting pages, and the caller throws the half written
    cache away and runs on the ggml layout. That is the correct order of
    priorities: a model that loads slowly is a model that loads.
    """
    var me = Pointer[WorkerArg, MutAnyOrigin](unsafe_from_address=arg)
    var job = Pointer[LoadJob, MutAnyOrigin](unsafe_from_address=me[].job)
    var slot = me[].index
    _ = set_thread_name("molla-load")
    var next = job[].cursor.slot(SLOT_NEXT)
    var warmed = job[].cursor.slot(SLOT_BYTES)
    var failed = job[].cursor.slot(SLOT_REPACK_ERR)
    var page = job[].page
    var repacking = job[].repack_fd >= 0 and slot < len(job[].scratch)
    var sink = UInt64(0)
    while True:
        var i = next.add(1)
        if i >= job[].count:
            break
        var start = job[].warm[i]
        var length = job[].lengths[i]
        _ = will_need(start, length)
        var p = RawPtr(unsafe_from_address=start)
        var at = 0
        while at < length:
            sink += UInt64(p.unsafe_load(at))
            at += page
        if length > 0:
            sink += UInt64(p.unsafe_load(length - 1))
        _ = warmed.add(length)
        if repacking and job[].repack_kind[i] >= 0 and failed.load() == 0:
            var rc = repack_tensor(
                job[].base,
                job[].offsets[i],
                job[].repack_kind[i],
                job[].repack_cols[i],
                job[].repack_rows[i],
                job[].repack_fd,
                job[].repack_off[i],
                job[].scratch[slot],
                job[].scratch_bytes,
            )
            if rc != 0:
                _ = failed.add(rc if rc > 0 else 1)
        while not job[].ready.push(i):
            _ = sleep_ms(1)
    _ = job[].cursor.slot(SLOT_SINK).add(Int(sink & 0xFFFF))
    return 0


struct DevicePool(Copyable, ImplicitlyCopyable, Movable):
    """One buffer on the device, and the slots cut out of it.

    Never constructed on a machine with no accelerator. `max-core` resolves the
    device architecture at compile time and refuses to compile a build for a
    machine it cannot name one for, so every call that reaches this is behind a
    `comptime if has_accelerator()`. Declaring the struct is free; instantiating
    its constructor is what would break the CPU boxes.
    """

    var ctx: DeviceContext
    """The context the pool allocates and copies against.

    There are two constructors because there are two callers and only one of
    them has a context already. A CUDA process gets one `DeviceContext` and
    hangs on the first allocation against a second, with the GPU idle and every
    thread asleep on a futex, so a caller that owns one has to be able to hand
    it over rather than watch a second appear. `load` still makes its own, which
    holds while a load is the only thing in the process talking to the device,
    and #143 is where the engine starts owning one and passes it in."""

    var pool: DeviceBuffer[DType.uint8]
    var bytes: Int

    var device: Device
    """Which device this came off. Carried rather than looked up again, because
    a pool and a device that disagree is a copy onto a card nothing is going to
    run a kernel on, and there is no way to notice that from the numbers."""

    def __init__(out self, device: Device, bytes: Int) raises:
        self.ctx = DeviceContext(device_id=device.index)
        self.pool = self.ctx.enqueue_create_buffer[DType.uint8](bytes)
        self.bytes = bytes
        self.device = device

    def __init__(
        out self, device: Device, bytes: Int, ctx: DeviceContext
    ) raises:
        """The same pool, against a context somebody else already owns."""
        self.ctx = ctx
        self.pool = self.ctx.enqueue_create_buffer[DType.uint8](bytes)
        self.bytes = bytes
        self.device = device

    def base(self) -> Int:
        """Where the pool starts, in the device's own addresses."""
        return Int(self.pool.unsafe_ptr())

    def slot_address(self, slot: Int) -> Int:
        """Where one tensor starts, for a kernel argument."""
        return self.base() + slot

    def copy_in(mut self, slot: Int, host: Int, length: Int) raises:
        """Queue one tensor's bytes into its slot.

        Asynchronous, which is the point. The call returns once the copy is on
        the stream, so the thread that issues it goes straight back to draining
        the ready queue while the card is still moving the last one.
        """
        var view = self.pool.create_sub_buffer[DType.uint8](slot, length)
        self.ctx.enqueue_copy(view, RawPtr(unsafe_from_address=host))

    def wait(mut self) raises:
        self.ctx.synchronize()


@fieldwise_init
struct LoadReport(Copyable, ImplicitlyCopyable, Movable):
    """What a load did, in the terms a slow one has to be explained in."""

    var tensors: Int
    var host_bytes: Int
    var device_bytes: Int
    var warmed_bytes: Int
    """What the transfer threads actually faulted in. It should equal the two
    above added together and it is recorded separately so that it can be
    checked, because a claim loop that skips a tensor or takes one twice is
    otherwise invisible: the load still finishes and the weights are still
    wrong."""

    var workers: Int
    var read_ms: Int
    var copy_ms: Int
    var total_ms: Int
    var device: String
    var left_behind: Int

    var resident: Int
    """Tensors that ended up in the device pool."""

    var pool_bytes: Int
    """What the pool allocation actually is, which is the slots plus the
    alignment between them. Reported next to `device_bytes` because a gap
    between the two that is bigger than a few hundred bytes a tensor means the
    slots were sized from something other than what got copied into them, and
    that reads as a model that loads and then produces noise."""

    var repacked: Int
    """Tensors written into the repack cache on this load. Zero on a hit and
    zero when there is no cache, and the note says which."""

    var repack_bytes: Int
    var repack_note: String
    """What happened to the cache, in one line. Always said out loud, because a
    repack that silently reruns on every load is the thing the cache exists to
    prevent and the only way to notice it is to be told."""

    def read_mib_s(self) -> Int:
        if self.read_ms <= 0:
            return 0
        var bytes = self.host_bytes + self.device_bytes
        return (bytes // MIB) * 1000 // self.read_ms

    def copy_mib_s(self) -> Int:
        if self.copy_ms <= 0:
            return 0
        return (self.device_bytes // MIB) * 1000 // self.copy_ms


struct Residency(Copyable, Movable):
    """Where each tensor ended up, indexed by position in the directory.

    Directory index and not name, because that is what the binder already has
    after it looks a weight up and it is what a placement already carries, so
    the two meet without either of them growing a table of strings.

    An empty one answers host for everything, which is what a caller that never
    ran a device load should get and what every test that does not care about
    placement gets by writing nothing.
    """

    var place: List[Int]
    var address: List[Int]
    """Where the tensor is on the device, or zero for one that did not move."""

    def __init__(out self):
        self.place = List[Int]()
        self.address = List[Int]()

    def count(self) -> Int:
        return len(self.place)

    def place_of(self, index: Int) -> Int:
        if index < 0 or index >= len(self.place):
            return WHERE_HOST
        return self.place[index]

    def address_of(self, index: Int) -> Int:
        if index < 0 or index >= len(self.address):
            return 0
        return self.address[index]


struct Weights(Movable):
    """A loaded model: the plan that placed it and the pool that holds it.

    The pool is optional because most of the fleet has nowhere to put one, and
    because a load whose kernels run on the host does not want one on any
    machine. A `Weights` with no pool is not a failed load, it is a model that
    lives in the file's own pages.
    """

    var plan: Plan
    var pool: Optional[DevicePool]
    var report: LoadReport

    def __init__(out self, var plan: Plan, report: LoadReport):
        self.plan = plan^
        self.pool = None
        self.report = report

    def device_base(self) raises -> Int:
        """Address of the pool on the device, or zero when there is not one."""
        if not self.pool:
            return 0
        return self.pool.value().base()

    def residency(self) raises -> Residency:
        """Where every tensor in the file ended up, by directory index.

        The one thing `bind` needs out of a load. Everything else in here is
        either accounting or the allocation itself, and handing the binder a
        whole `Weights` would let it reach the pool, the plan and the report
        when the only question it has is where one weight is.
        """
        var out = Residency()
        var at = self.device_base()
        for i in range(self.plan.count()):
            var one = self.plan.placements[i]
            var address = 0
            if one.place == WHERE_DEVICE:
                if at == 0:
                    raise Error(
                        "the plan put tensor "
                        + String(one.index)
                        + " on "
                        + self.plan.device.name
                        + " and this load has no pool to put it in"
                    )
                address = at + one.slot
            out.place.append(one.place)
            out.address.append(address)
        return out^


def worker_count(requested: Int) -> Int:
    """How many transfer threads to start.

    Half the cores, capped, because the other half of a load is the kernel
    faulting pages in and it wants somewhere to run. One on a single core box,
    which is not a special case so much as the same formula's floor.
    """
    if requested > 0:
        return requested
    var half = cpu_count() // 2
    if half < 1:
        return 1
    if half > MAX_WORKERS:
        return MAX_WORKERS
    return half


def load(
    g: Gguf,
    var plan: Plan,
    workers: Int = 0,
    stream: Bool = True,
    repack_for: StringSpan = "",
    ctx: Optional[DeviceContext] = None,
) raises -> Weights:
    """Run the plan. Reads on a thread pool, copies on this thread.

    The two stages overlap. Workers claim tensors with one atomic add, fault
    their pages in, and push the index onto a queue. This thread pops indices
    and enqueues the device copy for each one as it arrives, so the card starts
    moving the first tensor while the pool is still reading the second. On a
    host or unified plan there is nothing to enqueue and this thread does
    nothing but report progress.

    Progress goes out during the load and not after it, in tenths, because a
    load that takes thirty seconds and says nothing is indistinguishable from a
    hang and gets killed by someone who was right to kill it.

    `repack_for` is the model's own path and turns the repack on. Empty means
    the caller either found a usable cache already or does not want one, and
    this function does not go looking: deciding whether a cache is worth
    trusting needs the model key, and the caller has already computed it to ask
    that question.

    `ctx` is a device context the caller already owns. A CUDA process gets one
    of them and hangs on the first allocation against a second, so a caller that
    is going to run kernels against these weights has to hand its own over
    rather than let this make a second one behind it. Nothing means this makes
    its own, which is right for a load that is the only thing in the process
    talking to the device, and that is what `molla load` is.
    """
    var started = monotonic_ms()
    var count = plan.count()
    var pool_bytes = 0
    for i in range(count):
        if plan.placements[i].place == WHERE_DEVICE:
            pool_bytes = _align(
                plan.placements[i].slot + plan.placements[i].length, SLOT_ALIGN
            )

    if stream:
        _report_plan(plan, pool_bytes)
        _report_placement(g, plan)

    var threads = worker_count(workers)
    if threads > count:
        threads = count if count > 0 else 1

    var job = LoadJob(plan, g.mapping.address)
    var report = LoadReport(
        tensors=count,
        host_bytes=plan.host_bytes,
        device_bytes=plan.device_bytes,
        warmed_bytes=0,
        workers=threads,
        read_ms=0,
        copy_ms=0,
        total_ms=0,
        device=plan.device.name,
        left_behind=plan.left_behind,
        resident=0,
        pool_bytes=pool_bytes,
        repacked=0,
        repack_bytes=0,
        repack_note=String("nothing was repacked on this load"),
    )

    var temp = String("")
    var final = String("")
    if repack_for.byte_length() > 0:
        var rp = plan_repack(g)
        if rp.count() == 0:
            report.repack_note = String(
                "nothing in this file has a repacked form"
            )
        else:
            temp = temp_path(repack_for)
            final = cache_path(repack_for)
            try:
                var fd = open_cache_file(temp)
                write_head(fd, rp)
                job.attach_repack(rp, fd, threads)
                report.repacked = rp.count()
                report.repack_bytes = rp.total
            except e:
                # A model directory that cannot be written to is a normal thing
                # to run against and not a reason to refuse to load, so this
                # says what happened and carries on without a cache.
                temp = String("")
                report.repacked = 0
                report.repack_note = String("no cache written: ") + String(e)

    var out = Weights(plan^, report)

    comptime if has_accelerator():
        if pool_bytes > 0:
            if ctx:
                out.pool = DevicePool(out.plan.device, pool_bytes, ctx.value())
            else:
                out.pool = DevicePool(out.plan.device, pool_bytes)

    var read_started = monotonic_ms()
    var pool_threads = List[Thread]()
    var entry: ThreadFunc = _transfer
    # Filled before any address is taken. Appending to a list moves what is
    # already in it, so a thread started against element zero and then handed
    # a grown list is a thread reading freed memory.
    var args = List[WorkerArg](capacity=threads)
    for k in range(threads):
        args.append(WorkerArg(Int(Pointer(to=job)), k))
    for k in range(threads):
        var one = Thread()
        var rc = spawn(entry, Int(Pointer(to=args[k])), one)
        if not rc.is_ok():
            raise Error(rc.describe("could not start a transfer thread"))
        pool_threads.append(one^)

    var drained = 0
    var reported = -1
    var copy_ms = 0
    var copied = 0
    while drained < count:
        var index = 0
        if not job.ready.pop(index):
            _ = sleep_ms(1)
            continue
        var one = out.plan.placements[index]
        if one.place == WHERE_DEVICE:
            var at = monotonic_ms()
            comptime if has_accelerator():
                if out.pool:
                    out.pool.value().copy_in(one.slot, one.source, one.length)
                    out.report.resident += 1
                    copied += one.length
            copy_ms += monotonic_ms() - at
        drained += 1
        if stream:
            var tenth = drained * 10 // count
            if tenth != reported and tenth > 0:
                reported = tenth
                _report_tick(
                    tenth, job.cursor.slot(SLOT_BYTES).load(), read_started
                )

    for i in range(len(pool_threads)):
        _ = pool_threads[i].join()
    # The drain loop ends when the last tensor is popped, which is before the
    # worker that pushed it has finished its own loop. Mojo destroys a local at
    # its last use and handing out an address is not a use it can see, so
    # without this the queue and the counters are freed while a live thread is
    # still incrementing them. It shows up as a claim index that is a stale heap
    # pointer, which is a bounds error a long way from the cause. The worker
    # arguments are held for the same reason and the scratch buffers ride along
    # inside the job.
    out.report.warmed_bytes = job.cursor.slot(SLOT_BYTES).load()
    out.report.read_ms = monotonic_ms() - read_started

    if temp.byte_length() > 0:
        var failed = job.cursor.slot(SLOT_REPACK_ERR).load()
        if failed != 0:
            abandon_cache(job.repack_fd, temp)
            out.report.repacked = 0
            out.report.repack_bytes = 0
            out.report.repack_note = String(
                "the repack failed and was thrown away, errno "
            ) + String(failed)
        else:
            try:
                commit_cache(job.repack_fd, temp, final)
                out.report.repack_note = (
                    String("repack cache written to ") + final
                )
            except e:
                out.report.repacked = 0
                out.report.repack_bytes = 0
                out.report.repack_note = String("no cache written: ") + String(
                    e
                )
        job.repack_fd = -1
    keep(job)
    keep(args)

    comptime if has_accelerator():
        if out.pool:
            var at = monotonic_ms()
            out.pool.value().wait()
            copy_ms += monotonic_ms() - at
            # The plan said how many bytes would be on the card and this counts
            # what was put there. They are computed from the same placements a
            # few dozen lines apart, so the only way they disagree is a drain
            # loop that skipped a tensor or took one twice, and a card that is
            # missing one weight out of three hundred generates text that is
            # almost right, which is the worst failure in the file.
            if copied != out.plan.device_bytes:
                raise Error(
                    "the plan placed "
                    + String(out.plan.device_bytes)
                    + " bytes on "
                    + out.plan.device.name
                    + " and the load copied "
                    + String(copied)
                )
    out.report.copy_ms = copy_ms
    out.report.total_ms = monotonic_ms() - started

    if stream:
        _report_done(out.report)
    return out^


def _mib(bytes: Int) -> String:
    return String(bytes // MIB) + " MiB"


def _small(bytes: Int) -> String:
    """Megabytes, or kilobytes when megabytes would round to nothing.

    The norms of a small model come to a few hundred kilobytes, and a table
    that reports them as `0 MiB` reads like they were not placed at all.
    """
    if bytes < MIB:
        return String(bytes // 1024) + " KiB"
    return _mib(bytes)


def _report_plan(plan: Plan, pool_bytes: Int):
    print(
        stage_name(STAGE_PLAN)
        + "   "
        + String(plan.count())
        + " tensors, "
        + _mib(plan.total_bytes)
        + ", "
        + plan.device.api
        + " "
        + plan.device.name
    )
    if pool_bytes > 0:
        print(
            stage_name(STAGE_PLAN)
            + "   "
            + _mib(plan.device_bytes)
            + " to the device pool, "
            + _mib(plan.host_bytes)
            + " left on the host, budget "
            + _mib(plan.budget)
        )
    elif plan.device.unified() and plan.device.accelerator():
        print(
            stage_name(STAGE_PLAN)
            + "   "
            + _mib(plan.host_bytes)
            + " unified, in the memory the device draws from but not at an"
            " address it can read"
        )
    else:
        print(
            stage_name(STAGE_PLAN)
            + "   "
            + _mib(plan.host_bytes)
            + " on the host, no device pool"
        )
    if plan.left_behind > 0:
        print(
            stage_name(STAGE_PLAN)
            + "   "
            + String(plan.left_behind)
            + " tensors did not fit the budget and stay on the host"
        )


comptime CLASS_EMBEDDING = 0
comptime CLASS_ATTENTION = 1
comptime CLASS_FEEDFORWARD = 2
comptime CLASS_NORM = 3
comptime CLASS_OUTPUT = 4
comptime CLASS_OTHER = 5
comptime CLASS_COUNT = 6


def class_name(id: Int) -> String:
    if id == CLASS_EMBEDDING:
        return String("embedding")
    if id == CLASS_ATTENTION:
        return String("attention")
    if id == CLASS_FEEDFORWARD:
        return String("feed forward")
    if id == CLASS_NORM:
        return String("norms")
    if id == CLASS_OUTPUT:
        return String("output head")
    return String("other")


def class_of(name: String) -> Int:
    """Which group of weights a tensor belongs to, by its name.

    Coarse on purpose. The question this answers is which part of the model
    landed where, and the parts that can land in different places are the
    embedding, the per layer matrices, and the output head. Splitting the
    attention block into four would make the table longer without making any
    placement decision visible that is not visible already.

    Names rather than shapes, because the shapes of a key projection and a value
    projection are the same in most architectures and the names never are. The
    two embedding spellings are the two the architecture table already knows
    about.
    """
    if name.startswith("token_embd") or name.startswith("tok_embeddings"):
        return CLASS_EMBEDDING
    if name.find("attn") >= 0 or name.find("attention") >= 0:
        return CLASS_NORM if name.find("norm") >= 0 else CLASS_ATTENTION
    if name.find("ffn") >= 0 or name.find("feed_forward") >= 0:
        return CLASS_NORM if name.find("norm") >= 0 else CLASS_FEEDFORWARD
    if name.find("norm") >= 0:
        return CLASS_NORM
    if name.startswith("output") or name.startswith("lm_head"):
        return CLASS_OUTPUT
    return CLASS_OTHER


def _report_placement(g: Gguf, plan: Plan) raises:
    """Where each class of weight ended up, in tensors and megabytes.

    Printed because a model that quietly landed on the host is visible in
    nothing else. The total bytes are the same either way, the load finishes
    either way, and the only symptom is a token rate somebody has to already
    know the right value of. A load that put the attention matrices on the card
    and left the feed forward ones behind is the interesting case and it is one
    line here.
    """
    var host_n = List[Int]()
    var host_b = List[Int]()
    var dev_n = List[Int]()
    var dev_b = List[Int]()
    for _ in range(CLASS_COUNT):
        host_n.append(0)
        host_b.append(0)
        dev_n.append(0)
        dev_b.append(0)

    for i in range(plan.count()):
        var one = plan.placements[i]
        var id = class_of(g.text(g.tensors[one.index].name))
        if one.place == WHERE_DEVICE:
            dev_n[id] += 1
            dev_b[id] += one.length
        else:
            host_n[id] += 1
            host_b[id] += one.length

    for id in range(CLASS_COUNT):
        if host_n[id] == 0 and dev_n[id] == 0:
            continue
        var line = stage_name(STAGE_PLAN) + "   " + class_name(id)
        while line.byte_length() < 22:
            line += " "
        if dev_n[id] > 0:
            line += String(dev_n[id]) + " on the device, " + _small(dev_b[id])
            if host_n[id] > 0:
                line += ", "
        if host_n[id] > 0:
            line += String(host_n[id]) + " on the host, " + _small(host_b[id])
        print(line)


def _report_tick(tenth: Int, bytes: Int, started: Int):
    var elapsed = monotonic_ms() - started
    var rate = 0
    if elapsed > 0:
        rate = (bytes // MIB) * 1000 // elapsed
    print(
        stage_name(STAGE_READ)
        + "   "
        + String(tenth * 10)
        + "%  "
        + _mib(bytes)
        + "  "
        + String(rate)
        + " MiB/s"
    )


def _report_done(report: LoadReport):
    print(
        stage_name(STAGE_READ)
        + "   "
        + _mib(report.host_bytes + report.device_bytes)
        + " on "
        + String(report.workers)
        + " threads in "
        + String(report.read_ms)
        + " ms, "
        + String(report.read_mib_s())
        + " MiB/s"
    )
    if report.device_bytes > 0:
        print(
            stage_name(STAGE_COPY)
            + "   "
            + _mib(report.device_bytes)
            + " to "
            + report.device
            + " in "
            + String(report.copy_ms)
            + " ms, "
            + String(report.copy_mib_s())
            + " MiB/s"
        )
        print(
            stage_name(STAGE_COPY)
            + "   "
            + String(report.resident)
            + " tensors resident in a "
            + _mib(report.pool_bytes)
            + " pool"
        )
    else:
        print(
            stage_name(STAGE_COPY)
            + "   nothing to copy, the weights are read where they lie"
        )
    if report.repacked > 0:
        print(
            stage_name(STAGE_REPACK)
            + " "
            + String(report.repacked)
            + " tensors, "
            + _mib(report.repack_bytes)
            + ", "
            + report.repack_note
        )
    else:
        print(stage_name(STAGE_REPACK) + " " + report.repack_note)
    print(
        stage_name(STAGE_DONE)
        + "   "
        + String(report.tensors)
        + " tensors in "
        + String(report.total_ms)
        + " ms"
    )


def run_load(
    path: StringSpan,
    workers: Int = 0,
    repack: Bool = True,
    host_only: Bool = False,
    place: Optional[Device] = None,
) raises:
    """Entry point for `molla load` on a GGUF file.

    Reports a hit or a miss either way. A load that already has a cache says so
    and does not write one, and a load that does not says why, so the question
    of whether the repack is being redone every time has an answer on the
    screen rather than in a stopwatch.

    `host_only` is what `--host` sets and it is there because a machine with an
    accelerator has two loads worth timing, not one. The default fills the
    device, which is what a load for device kernels costs. `--host` leaves
    everything in the mapping, which is what the engine does today and what a
    load for host kernels costs. Neither is the fallback of the other and the
    difference between the two numbers is the price of a device address.

    `place` is what `--device=` resolved to. Nothing means the first
    accelerator, which is what this command did before there was a way to say.
    It arrives as a `Device` rather than as the flag itself because resolving a
    flag needs the model file and the fit, and asking that question from in here
    would make this module import the engine that imports it.
    """
    var g = Gguf(path)
    var dev = place.value() if place else default_device()

    # The cache is opened before the plan and not after it, because the plan is
    # what decides which copy of each weight gets read and a cache that turns up
    # afterwards is a cache the plan could not use.
    var want = String("")
    var cache = open_cache(path, model_key(g))
    var plan = plan_load(g, dev, 0 if host_only else -1, cache)
    if cache.usable:
        print(
            stage_name(STAGE_REPACK)
            + " hit, "
            + String(cache.count())
            + " tensors, "
            + _mib(cache.bytes())
            + " already repacked"
        )
    elif repack:
        print(stage_name(STAGE_REPACK) + " miss, " + cache.reason)
        want = String(path)

    var weights = load(g, plan^, workers, True, want)
    _ = weights.report.tensors
    cache.close()
    g.close()
