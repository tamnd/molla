"""Tests for placement and for the transfer pool.

Placement is arithmetic and it is tested against made up devices, because the
interesting cases are a card that is too small and a card that is unified, and
no machine in the fleet is both. A `Device` is a plain struct with public
fields, so a 4 GB discrete card and a 24 GB one are two lines of setup rather
than two machines.

The load itself runs against a real file on a real thread pool. What it can
check is conservation: every tensor comes back exactly once, every byte in the
directory gets faulted in, and the bytes the workers counted add up to the bytes
the plan said were there. A load that drops a tensor or reads one twice fails
here rather than in a kernel that gets the wrong weights and still produces
words.

There is no device copy in these tests. The copy needs an accelerator and three
of the five machines do not have one, so what proves that half is the pool group
in `tests/test_gpu.mojo`, which runs where there is a device, and a real model on
a real card, which is written down in `docs/validation/load.md`.
"""

from std.ffi import c_int, external_call

from harness import Suite
from test_gguf import Builder

from molla.model.gguf import Gguf
from molla.model.load import (
    MIN_RESERVE,
    WHERE_DEVICE,
    WHERE_HOST,
    WHERE_UNIFIED,
    Residency,
    device_budget,
    load,
    place_name,
    plan_load,
    stage_name,
    worker_count,
)
from molla.sys.device import DEV_CPU, DEV_DISCRETE, DEV_UNIFIED, Device
from molla.sys.mmap import Mapping, page_size, will_need

comptime GIB = 1024 * 1024 * 1024
comptime MIB = 1024 * 1024


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


def _temp_path() -> String:
    var pid = Int(external_call["getpid", Int32]())
    return String("/tmp/molla_test_load_") + String(pid) + ".gguf"


def _names() -> List[String]:
    """Six tensors with the names a real file uses, the embedding first.

    The embedding goes first on purpose. It is the one the planner has to move
    to the back of the queue, and a fixture that already had it last would pass
    whether the planner did anything or not.
    """
    return [
        String("token_embd.weight"),
        String("blk.0.attn_q.weight"),
        String("blk.0.attn_k.weight"),
        String("blk.0.ffn_up.weight"),
        String("blk.1.attn_q.weight"),
        String("output_norm.weight"),
    ]


def _elements() -> List[Int]:
    return [8192, 4096, 4096, 4096, 4096, 256]


def _sizes() -> List[Int]:
    """Bytes each tensor takes, which is four per element for F32."""
    var counts = _elements()
    var out = List[Int]()
    for i in range(len(counts)):
        out.append(counts[i] * 4)
    return out^


def _total() -> Int:
    var sizes = _sizes()
    var out = 0
    for i in range(len(sizes)):
        out += sizes[i]
    return out


def _model() -> List[UInt8]:
    """A GGUF file whose tensors are F32 and whose data is a known pattern."""
    var names = _names()
    var counts = _elements()
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
        b.u32(1)
        b.u64(UInt64(counts[i]))
        b.u32(0)  # F32
        b.u64(UInt64(at))
        at += sizes[i]
        at += (32 - (at % 32)) % 32

    b.pad_to(32)
    for i in range(_total()):
        b.u8(UInt8(i & 0xFF))
    # The directory pads between tensors, so the data section is as long as the
    # last offset says it is and not as long as the sizes add up to.
    while len(b.bytes) % 32 != 0:
        b.u8(0)
    return b^.finish()


def _fake(kind: Int, api: String, total: Int) -> Device:
    var one = Device(kind, 0, api, String("test device"))
    one.total = total
    one.free = total
    return one^


def run(mut suite: Suite) raises:
    var path = _temp_path()
    _remove(path)
    _write(path, _model())

    _check_budget(suite)
    _check_plan(suite, path)
    _check_load(suite, path)
    _check_residency(suite, path)
    _check_names(suite)

    _remove(path)


def _check_budget(mut suite: Suite) raises:
    suite.group("load budget")

    suite.check(
        device_budget(_fake(DEV_CPU, String("cpu"), 0)) == 0,
        "the host has no device budget",
    )
    # A unified device gets a budget, and used to get zero on the grounds that
    # an accelerator sharing the memory could read the mapping. It cannot, which
    # is #152, so a pool is how a weight there gets a readable address too. The
    # reserve is a quarter rather than a tenth because the total is the whole
    # machine's memory.
    var shared = _fake(DEV_UNIFIED, String("metal"), 16 * GIB)
    suite.check(
        device_budget(shared) == 16 * GIB - (16 * GIB) // 4,
        "a unified device keeps a quarter of the machine back",
    )
    suite.check(
        device_budget(shared)
        < device_budget(_fake(DEV_DISCRETE, String("cuda"), 16 * GIB)),
        "which is more than a card of the same size holds back",
    )

    # A tenth of 24 GB is more than the floor, so the fraction is what applies.
    var big = _fake(DEV_DISCRETE, String("cuda"), 24 * GIB)
    suite.check(
        device_budget(big) == 24 * GIB - (24 * GIB) // 10,
        "a large card keeps a tenth of itself back",
    )

    # A tenth of 2 GB is less than the floor, so the floor is what applies.
    var small = _fake(DEV_DISCRETE, String("cuda"), 2 * GIB)
    suite.check(
        device_budget(small) == 2 * GIB - MIN_RESERVE,
        "and a small one keeps the floor back instead",
    )

    var tiny = _fake(DEV_DISCRETE, String("cuda"), 256 * MIB)
    suite.check(
        device_budget(tiny) == 0,
        (
            "a card smaller than the reserve gets nothing, rather than a"
            " negative budget"
        ),
    )


def _check_plan(mut suite: Suite, path: String) raises:
    suite.group("load plan")

    var g = Gguf(path)
    var sizes = _sizes()

    var host = plan_load(g, _fake(DEV_CPU, String("cpu"), 0), -1)
    suite.check(host.count() == len(sizes), "every tensor is placed")
    suite.check(host.total_bytes == _total(), "and the total is the file's")
    suite.check(
        host.host_bytes == _total(), "on the host, all of it stays in the map"
    )
    suite.check(host.device_bytes == 0, "and none of it is copied anywhere")
    var all_host = True
    for i in range(host.count()):
        if host.placements[i].place != WHERE_HOST:
            all_host = False
    suite.check(all_host, "and every placement says host")

    # A unified device with room takes the pool, the same as a card does. The
    # placement follows where the kernels are going to read from and not what
    # kind of memory the machine has, because #152 found that an accelerator
    # sharing the memory still cannot follow a host address.
    var metal = _fake(DEV_UNIFIED, String("metal"), 16 * GIB)
    var unified = plan_load(g, metal, -1)
    var all_device = True
    for i in range(unified.count()):
        if unified.placements[i].place != WHERE_DEVICE:
            all_device = False
    suite.check(all_device, "a unified device with room takes every tensor")
    suite.check(
        unified.device_bytes == _total(),
        "and the bytes go to a pool rather than staying in the mapping",
    )

    # And a budget of nothing is how a load for host kernels is asked for, which
    # is the placement unified still names.
    var shared = plan_load(g, metal, 0)
    var all_unified = True
    for i in range(shared.count()):
        if shared.placements[i].place != WHERE_UNIFIED:
            all_unified = False
    suite.check(
        all_unified, "with no budget every placement on it says unified"
    )
    suite.check(
        shared.device_bytes == 0,
        "which is a placement and not a copy, so the pool stays empty",
    )

    var card = _fake(DEV_DISCRETE, String("cuda"), 24 * GIB)
    var whole = plan_load(g, card, -1)
    suite.check(
        whole.device_bytes == _total(),
        "a card with room takes every tensor",
    )
    suite.check(whole.left_behind == 0, "and leaves nothing behind")

    # Slots have to be aligned, in order, and not overlap. Getting any of these
    # wrong writes one tensor over another and the model still loads.
    var ordered = True
    var aligned = True
    var reach = 0
    for i in range(whole.count()):
        var one = whole.placements[i]
        if one.slot % 256 != 0:
            aligned = False
        if one.slot < reach:
            ordered = False
        reach = one.slot + one.bytes
    suite.check(aligned, "every slot is aligned to 256 bytes")
    suite.check(ordered, "and no slot starts before the last one ended")

    # Room for everything except the embedding, which is the case the two pass
    # order exists for.
    var budget = _total() - sizes[0] + 256
    var partial = plan_load(g, card, budget)
    suite.check(
        partial.left_behind == 1, "a budget one tensor short leaves one behind"
    )
    suite.check(
        partial.placements[0].place == WHERE_HOST,
        "and the one it leaves is the embedding, not the first in the file",
    )
    var rest_on_card = True
    for i in range(1, partial.count()):
        if partial.placements[i].place != WHERE_DEVICE:
            rest_on_card = False
    suite.check(
        rest_on_card, "everything read every token is still on the card"
    )

    var nothing = plan_load(g, card, 0)
    suite.check(
        nothing.device_bytes == 0 and nothing.host_bytes == _total(),
        "a budget of zero puts the whole model on the host",
    )

    g.close()


def _check_load(mut suite: Suite, path: String) raises:
    suite.group("load transfer")

    var g = Gguf(path)
    var plan = plan_load(g, _fake(DEV_CPU, String("cpu"), 0), -1)
    var weights = load(g, plan^, 4, False)
    var report = weights.report

    suite.check(
        report.tensors == len(_sizes()), "every tensor is accounted for"
    )
    suite.check(
        report.host_bytes == _total(), "and every byte of it stayed on the host"
    )
    suite.check(report.device_bytes == 0, "with nothing copied to a device")
    suite.check(
        report.warmed_bytes == report.host_bytes,
        "and the threads faulted in exactly the bytes the plan named",
    )
    suite.check(
        report.workers == 4, "the pool ran the threads it was asked for"
    )
    suite.check(report.copy_ms == 0, "and the copy stage did no work")
    suite.check(
        report.total_ms >= report.read_ms,
        "the total covers the read rather than being measured beside it",
    )
    suite.check(not weights.pool, "a host plan allocates no device pool at all")
    suite.check(
        weights.device_base() == 0, "so there is no device address to hand out"
    )

    # One worker rather than four, which is the same work through a queue that
    # never has two producers. It has caught a claim loop that only worked when
    # something else was racing it.
    var again = plan_load(g, _fake(DEV_CPU, String("cpu"), 0), -1)
    var solo = load(g, again^, 1, False)
    suite.check(
        solo.report.tensors == report.tensors,
        "one thread loads the same tensors as four",
    )
    suite.check(
        solo.report.host_bytes == report.host_bytes,
        "and the same bytes",
    )
    suite.check(
        solo.report.warmed_bytes == report.warmed_bytes,
        "and faults in the same bytes, so no tensor was claimed twice",
    )

    suite.check(
        worker_count(3) == 3, "a worker count that is asked for is used"
    )
    suite.check(worker_count(0) >= 1, "and one that is not is at least one")

    g.close()


def _check_residency(mut suite: Suite, path: String) raises:
    """What a load hands the binder, and what an empty one answers.

    An empty residency is not a placeholder. It is what every caller that loads
    with a device budget of zero gets, which today is every caller that
    generates a token, so its answers matter as much as a real one's.
    """
    suite.group("load residency")

    var empty = Residency()
    suite.check(empty.count() == 0, "an empty residency covers no tensors")
    suite.check(
        empty.place_of(0) == WHERE_HOST, "and answers host for all of them"
    )
    suite.check(
        empty.address_of(0) == 0, "with no device address to go with it"
    )
    suite.check(
        empty.place_of(-1) == WHERE_HOST and empty.place_of(9999) == WHERE_HOST,
        "including for an index it has never heard of",
    )

    var g = Gguf(path)
    var plan = plan_load(g, _fake(DEV_CPU, String("cpu"), 0), -1)
    var weights = load(g, plan^, 1, False)
    var res = weights.residency()
    suite.check(
        res.count() == len(_sizes()), "a load reports on every tensor it placed"
    )
    var all_host = True
    var no_address = True
    for i in range(res.count()):
        if res.place_of(i) != WHERE_HOST:
            all_host = False
        if res.address_of(i) != 0:
            no_address = False
    suite.check(all_host, "and a host load says host for each of them")
    suite.check(no_address, "and hands out no device addresses")

    # A unified plan with no budget, which is the load the engine does today and
    # what `molla load --host` reports. Nothing is copied and nothing is
    # allocated, and that saving is what unified names. It is not an address: a
    # weight placed this way has none a device kernel can follow, which is why
    # the pool exists on this machine too.
    var shared = plan_load(g, _fake(DEV_UNIFIED, String("metal"), 16 * GIB), 0)
    var on_metal = load(g, shared^, 1, False)
    suite.check(
        not on_metal.pool, "a unified load allocates nothing on the device"
    )
    var res2 = on_metal.residency()
    var all_unified = True
    for i in range(res2.count()):
        if res2.place_of(i) != WHERE_UNIFIED:
            all_unified = False
        if res2.address_of(i) != 0:
            no_address = False
    suite.check(all_unified, "and every weight comes back unified")
    suite.check(
        no_address,
        "with no address of its own, which is the thing a pool would give it",
    )

    g.close()


def _check_names(mut suite: Suite) raises:
    suite.group("load names")

    suite.check(stage_name(0) == "plan", "the stages are named in order")
    suite.check(stage_name(1) == "read", "read is the second")
    suite.check(stage_name(2) == "copy", "copy is the third")
    suite.check(stage_name(3) == "repack", "repack is the fourth")
    suite.check(stage_name(4) == "done", "and done is the last")
    suite.check(place_name(WHERE_HOST) == "host", "host is named")
    suite.check(place_name(WHERE_UNIFIED) == "unified", "unified is named")
    suite.check(place_name(WHERE_DEVICE) == "device", "and device is named")

    var page = page_size()
    suite.check(page >= 4096, "a page is at least four kilobytes")
    suite.check(page & (page - 1) == 0, "and a power of two")
    suite.check(
        not will_need(0, 0), "advising an empty range does nothing and says so"
    )
