"""Which device the arithmetic runs on, decided once and said out loud.

Until #143 there was one backend and nothing to choose. Now there are two, they
produce the same text at very different speeds, and the difference between them
is invisible in the output. That is the whole reason this file exists: a
benchmark run against the wrong backend is worse than one that refused to start,
and a server that quietly fell back to the host looks exactly like a slow card.

## An explicit ask is a refusal, an automatic one is a fallback

`--device=cuda` on a machine with no CUDA is an error, before anything is
loaded. Somebody who names a backend has a reason to name it, and the useful
answer to "run this on the card" when there is no card is not thirty seconds of
loading followed by host speed.

`--device=auto` is the opposite and it says why. It takes the accelerator with
the most room the model fits in, and when nothing qualifies it stays on the host
and carries the reason with it, so the line under the model is `no accelerator
on this machine` rather than silence.

## Fitting is asked of the planner, not estimated

Whether a model fits is `plan_load` with a real budget and `left_behind` at
zero, which is the same function that will place the tensors a minute later. An
estimate here would be a second implementation of the reserve arithmetic, and
the two would agree until the day they did not. Planning reads the two
directories and touches no weights, so asking properly costs a few hundred
microseconds.

The device forward pass wants every weight on the card, so a model that half
fits does not count as fitting. That is stricter than what `molla load` allows,
and it is the rule the kernels actually impose: a device kernel handed a host
address reads zeros without faulting.
"""

from std.sys.info import has_accelerator

from molla.model.gguf import Gguf
from molla.model.load import plan_load
from molla.model.repack import RepackCache, model_key, open_cache
from molla.sys.device import Device, build_targets_gpu, devices, host_device

comptime WANT_AUTO = 0
"""Take the best accelerator the model fits on, or the host, and say which."""

comptime WANT_CPU = 1
"""Stay on the host whatever is attached to this machine."""

comptime WANT_API = 2
"""A backend named by its api, optionally with an index after a colon."""


struct Request(Copyable, ImplicitlyCopyable, Movable):
    """What a `--device=` flag asked for, before any machine is consulted.

    Parsing and resolving are apart because they fail for different reasons and
    at different times. A spelling mistake is worth catching before a four
    gigabyte file is opened, and whether a card is present is not knowable until
    it is.
    """

    var mode: Int
    var api: String
    """`metal` or `cuda`, and empty unless the mode is `WANT_API`."""

    var index: Int
    """Which device of that api, or -1 when the flag did not say."""

    def __init__(out self, mode: Int, api: String, index: Int):
        self.mode = mode
        self.api = api
        self.index = index

    def __init__(out self):
        self.mode = WANT_AUTO
        self.api = String("")
        self.index = -1

    def named(self) -> String:
        """The flag value that would produce this request, for a message."""
        if self.mode == WANT_CPU:
            return String("cpu")
        if self.mode != WANT_API:
            return String("auto")
        if self.index < 0:
            return self.api
        return self.api + ":" + String(self.index)


struct Backend(Copyable, ImplicitlyCopyable, Movable):
    """Where this run is going to compute, and why it is that one."""

    var device: Device
    var on_device: Bool
    """Whether the kernels are device kernels. False for the host, and false for
    a unified accelerator that is only being used as a budget."""

    var note: String
    """Why this is the answer, when the answer is not what was asked for.

    Empty when the run landed where it was told to. Non empty only after `auto`
    stayed on the host, because that is the case somebody reading a slow run
    needs explained.
    """

    def __init__(out self, device: Device, on_device: Bool, note: String):
        self.device = device
        self.on_device = on_device
        self.note = note

    def __init__(out self):
        """The host, with nothing to explain.

        Every entry point that takes a backend has this as its default, because
        a caller that did not think about backends is a caller whose kernels are
        host kernels. It does not raise, so it can stand as a default argument.
        """
        self.device = host_device()
        self.on_device = False
        self.note = String("")

    def describe(self) -> String:
        """The one line that goes under the model, in every command."""
        return self.device.api + " " + self.device.name


def parse_backend(text: String) raises -> Request:
    """A `--device=` value, or an error naming the ones there are.

    `auto`, `cpu`, an api, or an api and an index after a colon. The api names
    are the ones `molla devices` prints, because a flag that takes a different
    vocabulary from the command that lists the options is a flag people get
    wrong twice.
    """
    var value = String(text.strip())
    if value.byte_length() == 0 or value == "auto":
        return Request(WANT_AUTO, String(""), -1)
    if value == "cpu":
        return Request(WANT_CPU, String("cpu"), -1)

    var api = value
    var index = -1
    var colon = value.find(":")
    if colon >= 0:
        api = String(value[byte=0:colon])
        var rest = String(value[byte = colon + 1 : value.byte_length()])
        try:
            index = atol(rest)
        except:
            raise Error(
                "'"
                + rest
                + "' is not a device index. --device takes auto, cpu, or an api"
                " like metal or cuda, and an api may be followed by a colon and"
                " a number"
            )
        if index < 0:
            raise Error("a device index cannot be negative")
    if api != "metal" and api != "cuda":
        raise Error(
            "'"
            + api
            + "' is not a backend molla knows. It takes auto, cpu, metal or"
            " cuda, and 'molla devices' lists what this machine has"
        )
    return Request(WANT_API, api, index)


def fits_on(g: Gguf, cache: RepackCache, dev: Device) raises -> Bool:
    """Whether every weight of this model would land on this device.

    Every weight and not most of them. The device forward pass reads all of
    them through device pointers, and a tensor left in the mapping is read as
    zeros rather than as an error, so a model that half fits is a model that
    answers confidently out of layers that saw nothing.
    """
    if not dev.accelerator():
        return True
    return plan_load(g, dev, -1, cache).left_behind == 0


def _yes(all: List[Device]) -> List[Bool]:
    """A fit list that says yes to everything, for the questions fit does not
    enter into."""
    var out = List[Bool]()
    for _ in range(len(all)):
        out.append(True)
    return out^


def _api_list(all: List[Device]) -> String:
    """The apis this machine has, for a message about one it does not."""
    var out = String("")
    for i in range(len(all)):
        if out.byte_length() > 0:
            out += ", "
        out += all[i].api
    return out


def pick(want: Request, all: List[Device], fits: List[Bool]) raises -> Backend:
    """Resolve a request against a machine.

    Separated from the machine and from the model file so it can be tested
    against device lists this fleet does not contain. `all` is what
    `molla.sys.device.devices` returns, the host first, and `fits` says per
    entry whether the model would fit, which the host answers yes to because a
    weight in the mapping is already where it is going to be read.
    """
    if len(all) == 0 or len(fits) != len(all):
        raise Error("the device list and the fit list are different lengths")

    if want.mode == WANT_CPU:
        return Backend(all[0], False, String(""))

    if want.mode == WANT_API:
        if not build_targets_gpu():
            raise Error(
                "this build has no device code in it, so there is no "
                + want.api
                + " to run on. Accelerator support is decided when molla is"
                " compiled, not when it is run"
            )
        var seen = 0
        var highest = -1
        for i in range(len(all)):
            if not all[i].accelerator() or all[i].api != want.api:
                continue
            seen += 1
            if all[i].index > highest:
                highest = all[i].index
            if want.index >= 0 and all[i].index != want.index:
                continue
            if not fits[i]:
                raise Error(
                    "this model does not fit on "
                    + all[i].name
                    + ", which has "
                    + String(all[i].total // (1 << 20))
                    + " MiB. Every weight has to be on the device for the"
                    " device kernels to read it, so run without --device to"
                    " stay on the host"
                )
            return Backend(all[i], True, String(""))
        if seen == 0:
            raise Error(
                "this machine has no "
                + want.api
                + " device. It has: "
                + _api_list(all)
            )
        raise Error(
            "this machine has no "
            + want.api
            + " device with index "
            + String(want.index)
            + ". The ones it has go up to "
            + String(highest)
        )

    # auto. The most room the model fits in, and the host with a reason when
    # nothing qualifies.
    var best = -1
    var accelerators = 0
    var biggest = -1
    for i in range(len(all)):
        if not all[i].accelerator():
            continue
        accelerators += 1
        if biggest < 0 or all[i].total > all[biggest].total:
            biggest = i
        if not fits[i]:
            continue
        if best < 0 or all[i].total > all[best].total:
            best = i
    if best >= 0:
        return Backend(all[best], True, String(""))

    # Most specific reason first. A card in the list is a card whatever the
    # build can talk to, so "does not fit" outranks both of the others, and
    # "no accelerator on this machine" is only honest on a build that would
    # have been able to see one.
    var note = String("this build has no device code in it")
    if build_targets_gpu():
        note = String("no accelerator on this machine")
    if accelerators > 0:
        note = (
            String("the model does not fit on ")
            + all[biggest].name
            + ", which has "
            + String(all[biggest].total // (1 << 20))
            + " MiB"
        )
    return Backend(all[0], False, note^)


def choose_backend(model_path: String, want: Request) raises -> Backend:
    """Resolve a request against this machine and this model.

    Opens the file and the repack cache for as long as it takes to plan against
    each device, which is a header parse and two directory reads. It is a second
    open of both, since the run that follows opens them again, and that is worth
    it to keep the decision in one place instead of threading a half loaded
    state through two entry points that already differ in every other respect.
    """
    if want.mode == WANT_CPU:
        return Backend(host_device(), False, String(""))

    var all = devices()
    if want.mode == WANT_API:
        # Everything about a named backend except whether the model fits is
        # answerable without the file, and asking now means `--device=cuda` on
        # a machine with no CUDA says so rather than reporting whatever is
        # wrong with the path first. Nothing can come back from this call but a
        # refusal, because every entry is marked as fitting.
        _ = pick(want, all, _yes(all))

    var g = Gguf(model_path)
    var cache = open_cache(model_path, model_key(g))
    var fits = List[Bool]()
    for i in range(len(all)):
        fits.append(fits_on(g, cache, all[i]))
    cache.close()
    g.close()
    return pick(want, all, fits)
