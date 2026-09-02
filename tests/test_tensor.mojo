"""The weight view and the activation buffer.

Almost all of this is arithmetic on four integers, which is exactly why it gets
tests. A row stride that is off by one block is a model that reads coherently
for the first row of the first layer and then walks off into the next row, and
the symptom of that is output that degrades over a few tokens rather than
anything that looks like a crash.
"""

from harness import Suite

from molla.nn.quant import Q_F16, Q_F32, Q_Q4_K, Q_Q6_K, Q_Q8_0
from molla.nn.tensor import Buffer, Tensor


def run(mut suite: Suite) raises:
    test_geometry(suite)
    test_rows(suite)
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
