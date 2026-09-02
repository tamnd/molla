"""The weight view and the activation buffer.

Almost all of this is arithmetic on four integers, which is exactly why it gets
tests. A row stride that is off by one block is a model that reads coherently
for the first row of the first layer and then walks off into the next row, and
the symptom of that is output that degrades over a few tokens rather than
anything that looks like a crash.
"""

from harness import Suite

from molla.nn.quant import Q_F16, Q_F32, Q_Q4_K, Q_Q6_K, Q_Q8_0
from molla.nn.repack import LAYOUT_PLANAR
from molla.nn.tensor import (
    WHERE_DEVICE,
    WHERE_HOST,
    WHERE_UNIFIED,
    Buffer,
    Tensor,
    place_name,
)


def run(mut suite: Suite) raises:
    test_geometry(suite)
    test_rows(suite)
    test_residency(suite)
    test_buffer(suite)


def test_geometry(mut suite: Suite) raises:
    suite.group("tensor geometry")

    # The shape an 8B Llama prints for an mlp down projection. In ggml order
    # that is 14336 rows of 4096, so a matvec against it takes 4096 in and
    # gives 14336 out, and reading it the other way is off by a factor of 3.5.
    var down = Tensor(0, Q_Q4_K, 4096, 14336)
    suite.check(
        down.cols == 4096 and down.rows == 14336, "dims[0] is the fast axis"
    )
    suite.check(
        down.elements() == 4096 * 14336, "and the element count is the product"
    )
    suite.check(
        down.row_bytes() == (4096 // 256) * 144,
        "a q4_k row is one block per 256 values",
    )
    suite.check(
        down.bytes() == (4096 // 256) * 144 * 14336,
        "and a tensor is nothing but its rows end to end",
    )

    var out_head = Tensor(0, Q_Q6_K, 4096, 128256)
    suite.check(
        out_head.row_bytes() == (4096 // 256) * 210,
        "and a q6_k row is 210 bytes per block",
    )

    suite.check(
        Tensor(0, Q_F32, 4096, 1).row_bytes() == 4096 * 4,
        "an f32 row is four bytes a value",
    )
    suite.check(
        Tensor(0, Q_F16, 4096, 1).row_bytes() == 4096 * 2,
        "and an f16 row is two",
    )
    suite.check(
        Tensor(0, Q_Q8_0, 4096, 1).row_bytes() == (4096 // 32) * 34,
        "and a q8_0 row is 34 bytes per 32 values",
    )

    var raised = False
    try:
        _ = Tensor(0, Q_Q4_K, 100, 1).row_bytes()
    except:
        raised = True
    suite.check(
        raised,
        "a row that is not a whole number of blocks is not a row ggml wrote",
    )

    raised = False
    try:
        _ = Tensor(0, 99, 256, 1).row_bytes()
    except:
        raised = True
    suite.check(raised, "and a type with no known block size has no row size")


def test_rows(mut suite: Suite) raises:
    suite.group("tensor rows")

    var w = Tensor(4096, Q_Q4_K, 512, 8)
    suite.check(w.row(0) == 4096, "the first row starts where the tensor does")
    suite.check(
        w.row(1) == 4096 + 2 * 144, "and each one after is a row stride along"
    )
    suite.check(w.row(7) == 4096 + 7 * 2 * 144, "including the last")

    var raised = False
    try:
        _ = w.row(8)
    except:
        raised = True
    suite.check(raised, "one past the end is an error and not an address")

    raised = False
    try:
        _ = w.row(-1)
    except:
        raised = True
    suite.check(raised, "and so is a negative row")


def test_residency(mut suite: Suite) raises:
    """Which memory a weight is in, and who is allowed to read it.

    The address is a plausible looking number rather than a real allocation on
    purpose. What is being tested is that the wrong side is refused before it
    dereferences anything, so a test that needed a real device pointer to prove
    it would be a test that only runs on two of the five machines.
    """
    suite.group("tensor residency")

    var host = Tensor(0x1000, Q_Q8_0, 64, 4)
    suite.check(host.place == WHERE_HOST, "a weight is on the host by default")
    suite.check(not host.on_device(), "so nothing has to be kept away from it")
    suite.check(
        Int(host.base()) == 0x1000, "and a host kernel gets the address back"
    )

    var raised = False
    try:
        _ = host.device_address()
    except:
        raised = True
    suite.check(raised, "a device kernel cannot read a host weight")

    var card = host.resident(0x7F00, WHERE_DEVICE)
    suite.check(card.on_device(), "a weight moved to a pool says so")
    suite.check(
        card.device_address() == 0x7F00, "and a device kernel gets the pool"
    )
    suite.check(
        card.kind == host.kind and card.cols == 64 and card.rows == 4,
        "moving a weight does not change what it is",
    )
    suite.check(
        card.layout == host.layout,
        "and a copy of bytes is not a transform of them, so the layout rides",
    )

    raised = False
    var message = String("")
    try:
        _ = card.base()
    except e:
        raised = True
        message = String(e)
    suite.check(raised, "a host kernel cannot read a device weight")
    suite.check(
        message
        == "a 64 by 4 weight is on the device and a host kernel cannot read it",
        "and the error says which weight it was, in the shape it was bound as",
    )

    # Unified used to be the case that both sides read, and it is not. One pool
    # of memory turned out not to mean one address: a device buffer's own
    # pointer segfaults when read from the host, `map_to_host` hands back a
    # different address for the same bytes, and a kernel given the host one
    # reads zeros without faulting. So the saving unified names is real, the
    # load allocates and copies nothing, and the address it carries is a host
    # address and nothing more. Issue #152 is where it gets a device one.
    var shared = host.resident(0x1000, WHERE_UNIFIED)
    suite.check(
        not shared.on_device(), "a unified weight is not out of a host reach"
    )
    suite.check(Int(shared.base()) == 0x1000, "so a host kernel still reads it")
    raised = False
    message = String("")
    try:
        _ = shared.device_address()
    except e:
        raised = True
        message = String(e)
    suite.check(raised, "and a device kernel does not, sharing a pool or not")
    suite.check(
        "unified resident" in message,
        "and the error says which of the three places it was in",
    )
    suite.check(
        "zeros" in message,
        "and what handing the address over anyway would have done",
    )

    var planar = card.as_planar(0x7F00)
    suite.check(
        planar.layout == LAYOUT_PLANAR and planar.on_device(),
        "repacking a resident weight leaves it where it is",
    )

    suite.check(Tensor.none().place == WHERE_HOST, "an absent weight is host")
    suite.check(
        place_name(WHERE_HOST) == "host"
        and place_name(WHERE_UNIFIED) == "unified"
        and place_name(WHERE_DEVICE) == "device",
        "every place has a name a report can print",
    )


def test_buffer(mut suite: Suite) raises:
    suite.group("activation buffer")

    var b = Buffer(4)
    suite.check(b.elements() == 4, "a buffer knows how many values it holds")
    suite.check(b.at(0) == 0.0 and b.at(3) == 0.0, "and it starts at zero")

    b.fill(2.5)
    suite.check(b.at(2) == 2.5, "fill writes every slot")
    b.zero()
    suite.check(b.at(2) == 0.0, "and zero puts it back")

    var wide = Buffer(3, 2)
    suite.check(
        wide.elements() == 6,
        "a buffer can have rows, and then it holds cols times rows",
    )

    var src = Buffer(4)
    src.fill(1.5)
    b.copy_from(src)
    suite.check(b.at(1) == 1.5, "and one buffer can be copied into another")

    var raised = False
    try:
        b.copy_from(wide)
    except:
        raised = True
    suite.check(raised, "copying a different size in is an error")
