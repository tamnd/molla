"""Tests for the backend decision.

The point of `pick` being a separate function from `choose_backend` is that it
can be asked about machines this fleet does not have. Two cards of different
sizes, a card the model does not fit on, a machine with an api nobody asked
for: none of those exist here, and all of them are one list literal away. So
every check below builds the machine it wants to ask about rather than
describing the one it is running on.

The checks that need a card are the ones that resolve `--device=metal` to a
real placement, and those are split on `build_targets_gpu` because a build with
no device code refuses them by design. On the boxes with no GPU the refusal
itself is what gets checked, which is the behaviour that matters there.
"""

from harness import Suite

from molla.engine.backend import (
    Request,
    WANT_API,
    WANT_AUTO,
    WANT_CPU,
    parse_backend,
    pick,
)
from molla.model.load import (
    CLASS_ATTENTION,
    CLASS_EMBEDDING,
    CLASS_FEEDFORWARD,
    CLASS_NORM,
    CLASS_OTHER,
    CLASS_OUTPUT,
    class_of,
)
from molla.sys.device import (
    DEV_CPU,
    DEV_DISCRETE,
    DEV_UNIFIED,
    Device,
    build_targets_gpu,
    host_device,
)

comptime GIB = 1 << 30


def _card(api: String, index: Int, name: String, gib: Int) -> Device:
    """An accelerator with a size, which is all `pick` reads off one."""
    var kind = DEV_UNIFIED if api == "metal" else DEV_DISCRETE
    var one = Device(kind, index, api, name)
    one.total = gib * GIB
    one.free = gib * GIB
    return one^


def _machine(cards: List[Device]) -> List[Device]:
    """A device list in the shape `devices` returns it, the host first."""
    var out = List[Device]()
    out.append(host_device())
    for i in range(len(cards)):
        out.append(cards[i])
    return out^


def _all_fit(all: List[Device]) -> List[Bool]:
    var out = List[Bool]()
    for _ in range(len(all)):
        out.append(True)
    return out^


def run(mut suite: Suite) raises:
    _check_parse(suite)
    _check_auto(suite)
    _check_named(suite)
    _check_classes(suite)


def _check_parse(mut suite: Suite) raises:
    """The flag values, before any machine is consulted."""
    suite.group("backend.parse")

    suite.check(parse_backend("").mode == WANT_AUTO, "an empty value is auto")
    suite.check(parse_backend("auto").mode == WANT_AUTO, "and so is auto")
    suite.check(parse_backend("  auto ").mode == WANT_AUTO, "with any spacing")
    suite.check(parse_backend("cpu").mode == WANT_CPU, "cpu is the host")

    var metal = parse_backend("metal")
    suite.check(metal.mode == WANT_API, "an api names a backend")
    suite.check(metal.api == "metal", "and is carried through as written")
    suite.check(metal.index == -1, "with no index when none was given")

    var second = parse_backend("cuda:1")
    suite.check(second.api == "cuda", "a colon splits the api from the index")
    suite.check(second.index == 1, "and the index is the number after it")
    suite.check(parse_backend("cuda:0").index == 0, "index zero is an index")

    suite.check(
        parse_backend("cuda:2").named() == "cuda:2", "named round trips an api"
    )
    suite.check(parse_backend("cpu").named() == "cpu", "and the host")
    suite.check(parse_backend("").named() == "auto", "and auto")

    var raised = False
    try:
        _ = parse_backend("vulkan")
    except:
        raised = True
    suite.check(raised, "an api molla has no kernels for is refused")

    raised = False
    try:
        _ = parse_backend("cuda:x")
    except:
        raised = True
    suite.check(raised, "an index that is not a number is refused")

    raised = False
    try:
        _ = parse_backend("cuda:-1")
    except:
        raised = True
    suite.check(raised, "and so is a negative one")


def _check_auto(mut suite: Suite) raises:
    """Auto never fails, so every case here is about what it lands on."""
    suite.group("backend.auto")

    var want = Request(WANT_AUTO, String(""), -1)

    var bare = _machine(List[Device]())
    var host_only = pick(want, bare, _all_fit(bare))
    suite.check(not host_only.on_device, "auto stays on a machine with no card")
    suite.check(
        host_only.note.byte_length() > 0, "and says why it is on the host"
    )

    var cards = List[Device]()
    cards.append(_card("cuda", 0, String("small card"), 8))
    cards.append(_card("cuda", 1, String("big card"), 24))
    var two = _machine(cards)

    var best = pick(want, two, _all_fit(two))
    suite.check(best.on_device, "auto takes a card when the model fits")
    suite.check(best.device.name == "big card", "and takes the largest of them")
    suite.check(best.note.byte_length() == 0, "with nothing to explain")

    var fits = _all_fit(two)
    fits[2] = False
    var smaller = pick(want, two, fits)
    suite.check(
        smaller.device.name == "small card",
        "and skips a card the model does not fit on",
    )

    fits[1] = False
    var neither = pick(want, two, fits)
    suite.check(not neither.on_device, "no card fitting means the host")
    suite.check(
        neither.note.find("does not fit") >= 0,
        "and the note names the size rather than blaming the machine",
    )

    var cpu = pick(Request(WANT_CPU, String("cpu"), -1), two, _all_fit(two))
    suite.check(not cpu.on_device, "cpu stays on the host with cards present")
    suite.check(cpu.device.kind == DEV_CPU, "on the host entry itself")
    suite.check(cpu.note.byte_length() == 0, "and owes no explanation")


def _check_named(mut suite: Suite) raises:
    """An api that was asked for by name, which is refused rather than moved."""
    suite.group("backend.named")

    var cards = List[Device]()
    cards.append(_card("metal", 0, String("Apple M4"), 12))
    var mac = _machine(cards)
    var fit = _all_fit(mac)

    var raised = False
    try:
        _ = pick(Request(WANT_API, String("cuda"), -1), mac, fit)
    except:
        raised = True
    suite.check(raised, "an api this machine does not have is an error")

    raised = False
    try:
        _ = pick(Request(WANT_API, String("metal"), 3), mac, fit)
    except:
        raised = True
    suite.check(raised, "and so is an index it does not have")

    raised = False
    try:
        _ = pick(Request(WANT_AUTO, String(""), -1), mac, List[Bool]())
    except:
        raised = True
    suite.check(raised, "a fit list of the wrong length is a caller bug")

    if not build_targets_gpu():
        raised = False
        try:
            _ = pick(Request(WANT_API, String("metal"), -1), mac, fit)
        except:
            raised = True
        suite.check(
            raised, "a build with no device code refuses a named backend"
        )
        return

    var named = pick(Request(WANT_API, String("metal"), -1), mac, fit)
    suite.check(named.on_device, "an api that is here resolves to the card")
    suite.check(named.device.index == 0, "taking the first of that api")

    cards.append(_card("metal", 1, String("second"), 4))
    var pair = _machine(cards)
    var both = _all_fit(pair)
    var one = pick(Request(WANT_API, String("metal"), 1), pair, both)
    suite.check(one.device.index == 1, "an index picks that device and not the")
    suite.check(one.device.name == "second", "largest or the first")

    both[1] = False
    raised = False
    try:
        _ = pick(Request(WANT_API, String("metal"), 0), pair, both)
    except:
        raised = True
    suite.check(
        raised, "a named card the model does not fit on is an error too"
    )


def _check_classes(mut suite: Suite) raises:
    """The tensor classes the load report groups by.

    Names taken from the files this repo actually reads, both the llama.cpp
    convention and the older one, because the report is worthless if half the
    tensors of a model land in `other`.
    """
    suite.group("backend.classes")

    suite.check(
        class_of("token_embd.weight") == CLASS_EMBEDDING,
        "token_embd is the embedding",
    )
    suite.check(
        class_of("tok_embeddings.weight") == CLASS_EMBEDDING,
        "and so is the older spelling",
    )
    suite.check(
        class_of("blk.0.attn_q.weight") == CLASS_ATTENTION,
        "an attention projection is attention",
    )
    suite.check(
        class_of("layers.0.attention.wv.weight") == CLASS_ATTENTION,
        "under either spelling",
    )
    suite.check(
        class_of("blk.0.ffn_gate.weight") == CLASS_FEEDFORWARD,
        "a gate is feed forward",
    )
    suite.check(
        class_of("layers.0.feed_forward.w2.weight") == CLASS_FEEDFORWARD,
        "under either spelling",
    )
    suite.check(
        class_of("blk.0.attn_norm.weight") == CLASS_NORM,
        "an attention norm is a norm and not attention",
    )
    suite.check(
        class_of("blk.0.ffn_norm.weight") == CLASS_NORM,
        "and a feed forward norm is a norm",
    )
    suite.check(
        class_of("output_norm.weight") == CLASS_NORM,
        "the final norm is a norm before it is an output",
    )
    suite.check(
        class_of("output.weight") == CLASS_OUTPUT, "the head is the output"
    )
    suite.check(
        class_of("lm_head.weight") == CLASS_OUTPUT,
        "under either spelling",
    )
    suite.check(
        class_of("rope_freqs.weight") == CLASS_OTHER,
        "and anything else says so rather than guessing",
    )
