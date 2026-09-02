"""What accelerators this machine has, and how much memory each of them has.

Everything above this asks the same three questions before it places a tensor:
what is here, how big is it, and does the host share memory with it. The last
one is not a detail. On an Apple machine there is no host and device split, so
a mapped file is already visible to the GPU and copying it anywhere would be
work done for nothing. On a discrete card the same mapping has to be copied
across a bus that is the slowest thing in the load, and the copy is worth
overlapping with the read that feeds it. One flag decides which of those two
programs runs, and it is better read off the device than guessed from the
platform.

The CPU is always in the list and is always first. It is not a fallback for
when the enumeration fails, it is a real placement: a tensor that stays in the
mapping is on the CPU, and on a machine with no accelerator at all that is
every tensor and molla still runs.

`max-core` supplies the enumeration, which is the whole reason it is a
dependency. See docs/adr/0002-accept-max-core.md for what that costs.
"""

from std.sys.info import _accelerator_arch, has_accelerator

from max.gpu.host import DeviceContext

comptime DEV_CPU = 0
"""Host memory. A tensor here is the mapping itself, with no copy at all."""

comptime DEV_UNIFIED = 1
"""An accelerator that shares memory with the host, which is Apple silicon."""

comptime DEV_DISCRETE = 2
"""An accelerator with its own memory, reached over a bus."""


struct Device(Copyable, ImplicitlyCopyable, Movable):
    """One place a tensor can live.

    `total` and `free` are bytes and both are zero for the CPU entry, because
    the number that matters for host placement is how much the operating system
    will let this process keep resident, and nothing here can answer that
    honestly. A placement decision about host memory is a decision about the
    file size and the page cache, not about a number read from a driver.
    """

    var kind: Int
    var index: Int
    """Which device of its kind, and always zero for the CPU."""

    var api: String
    """What the runtime calls the backend: `cpu`, `metal` or `cuda`."""

    var name: String
    var total: Int
    var free: Int

    def __init__(out self, kind: Int, index: Int, api: String, name: String):
        self.kind = kind
        self.index = index
        self.api = api
        self.name = name
        self.total = 0
        self.free = 0

    def unified(self) -> Bool:
        """Whether a host pointer is already visible to this device."""
        return self.kind != DEV_DISCRETE

    def accelerator(self) -> Bool:
        return self.kind != DEV_CPU


def _kind_of(api: String) -> Int:
    """Metal is unified and everything else with its own name is not.

    Read off the api rather than off the build target, because a Linux box with
    an integrated GPU and a Mac with a discrete one are both possible and the
    thing that decides the copy is the memory, not the operating system.
    """
    if api == "metal":
        return DEV_UNIFIED
    return DEV_DISCRETE


def build_targets_gpu() -> Bool:
    """Whether this build can talk to an accelerator at all.

    This is a property of the machine that ran the compiler, not of the machine
    running the binary. `max-core` bakes the device architecture in at compile
    time, so a build made on a box with no GPU has no device code in it and
    will report no accelerators even if you carry it to a box that has one.
    That is worth knowing before trusting a binary you did not build here.
    """
    return has_accelerator()


def build_target_arch() -> String:
    """What architecture this build was compiled for, or `none`."""
    comptime if has_accelerator():
        return String(_accelerator_arch())
    return String("none")


def host_device() -> Device:
    """The CPU entry, which is present on every machine and needs no runtime."""
    return Device(DEV_CPU, 0, String("cpu"), String("host"))


def devices() raises -> List[Device]:
    """Everything this machine can put a tensor on, the CPU first.

    A machine with no accelerator returns a list of one rather than an empty
    list or an error, because "there is nowhere to put this" is never the
    answer: host memory is always somewhere to put it.

    The accelerator half of this is behind `comptime if` because `max-core`
    resolves the device architecture at compile time and refuses to compile at
    all on a machine it cannot name one for. Three of our five boxes have no
    GPU, so leaving the call unguarded would mean molla does not build on the
    machines that most need the CPU path to work. See `build_targets_gpu`.
    """
    var out = List[Device]()
    out.append(host_device())

    comptime if has_accelerator():
        var count = DeviceContext.number_of_devices()
        for i in range(count):
            var ctx = DeviceContext(device_id=i)
            var api = String(ctx.api())
            var one = Device(_kind_of(api), i, api, String(ctx.name()))
            var memory = ctx.get_memory_info()
            one.free = Int(memory[0])
            one.total = Int(memory[1])
            out.append(one^)
    return out^


def default_device() raises -> Device:
    """The accelerator to use when nobody said which, or the CPU if there is
    none.

    The first accelerator rather than the largest. A machine with two cards
    wants a placement decision made by whoever knows the model, and picking the
    bigger one here would be that decision made in the wrong place with less
    information.
    """
    var all = devices()
    for i in range(len(all)):
        if all[i].accelerator():
            return all[i]
    return all[0]
