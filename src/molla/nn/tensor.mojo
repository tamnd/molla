"""A weight where it lies, and a buffer for the numbers passing through it.

Two types, and the difference between them is the whole point.

`Tensor` is a view. It owns nothing, it is four integers, and it points at bytes
inside a mapped file or a device pool. Weights are read a few hundred times per
token and never written, so copying one anywhere is a cost with no benefit, and
a view that owns nothing cannot be freed while a kernel is reading it.

`Buffer` owns float32 and is what activations live in. Activations are written
every token, they are small next to the weights, and they are always float32
here because the accumulation has to be wider than the weights are and float32
is the widest thing that is free on both a CPU and a GPU.

Shape is ggml's and not numpy's. `dims[0]` is the fast axis, so a weight the
GGUF directory prints as `[4096, 14336]` has 14336 rows of 4096, and a matvec
against it takes a vector of 4096 and returns one of 14336. Getting that
backwards produces a shape error rather than wrong numbers, which is the one
mercy in the whole layout.
"""

from molla.nn.quant import block_bytes, block_elements, supported
from molla.sys.mmap import RawPtr


struct Tensor(Copyable, ImplicitlyCopyable, Movable):
    """A two dimensional weight at a fixed address.

    Two dimensions and not four. Every weight in a dense transformer is a
    matrix or a vector, the four dimensional case in ggml is for convolutions
    and for tensors that have been reshaped for a kernel that wants them that
    way, and neither happens here. A model that needs more than this gets more
    than this, rather than everything carrying four dimensions today for a case
    that has not arrived.
    """

    var address: Int
    """Where the first byte is. An integer rather than a pointer for the same
    reason `Mapping` holds one: a struct field may not carry AnyOrigin, and the
    bytes are in a mapping or a device pool that has no origin to borrow."""

    var kind: Int
    """The ggml type number."""

    var cols: Int
    """Elements along the fast axis, which is `dims[0]` in the directory."""

    var rows: Int
    """How many of those there are. One for a vector."""

    def __init__(out self, address: Int, kind: Int, cols: Int, rows: Int):
        self.address = address
        self.kind = kind
        self.cols = cols
        self.rows = rows

    @staticmethod
    def none() -> Self:
        """A weight the model does not have.

        Address zero, which is not a place any mapping can start, so a tensor
        that is absent and a tensor that is present are told apart by the one
        field that cannot lie about it. The alternative is a boolean beside
        every weight, and two things that have to agree are one thing that can
        disagree.
        """
        return Self(0, 0, 0, 0)

    def present(self) -> Bool:
        return self.address != 0

    def base(self) -> RawPtr:
        return RawPtr(unsafe_from_address=self.address)

    def elements(self) -> Int:
        return self.cols * self.rows

    def row_bytes(self) raises -> Int:
        """Bytes one row occupies.

        Raises when a row is not a whole number of blocks. ggml quantizes along
        the fast axis, so every row starts on a block boundary and a row length
        that is not a multiple of the block size is not a tensor ggml could have
        written. Rounding here would put every row after the first at an offset
        that drifts further from the truth, which reads as a model that is
        coherent for the first layer and noise after it.
        """
        if not supported(self.kind):
            raise Error(
                "ggml type " + String(self.kind) + " has no known block size"
            )
        var per = block_elements(self.kind)
        if self.cols % per != 0:
            raise Error(
                "a row of "
                + String(self.cols)
                + " is not a whole number of "
                + String(per)
                + " element blocks"
            )
        return (self.cols // per) * block_bytes(self.kind)

    def row(self, index: Int) raises -> Int:
        """The address of one row."""
        if index < 0 or index >= self.rows:
            raise Error(
                "row "
                + String(index)
                + " is out of range for a tensor with "
                + String(self.rows)
                + " rows"
            )
        return self.address + index * self.row_bytes()

    def bytes(self) raises -> Int:
        return self.row_bytes() * self.rows


struct Buffer(Movable):
    """Float32 activations, owned.

    A thin wrapper over `List[Float32]` rather than a new allocation strategy.
    The engine will want an arena so a token's worth of activations is one
    allocation reused forever, and `molla.io.arena` already exists for exactly
    that, but wiring it in before there is a forward pass to size it against
    would be guessing at the sizes. What this gives today is a shape next to the
    numbers, which is what stops a kernel writing a 4096 vector into a 1024 one.
    """

    var data: List[Float32]
    var cols: Int
    var rows: Int

    def __init__(out self, cols: Int, rows: Int = 1):
        self.data = List[Float32]()
        self.cols = cols
        self.rows = rows
        for _ in range(cols * rows):
            self.data.append(0.0)

    def elements(self) -> Int:
        return self.cols * self.rows

    def at(self, index: Int) -> Float32:
        return self.data[index]

    def zero(mut self):
        for i in range(len(self.data)):
            self.data[i] = 0.0

    def fill(mut self, value: Float32):
        for i in range(len(self.data)):
            self.data[i] = value

    def copy_from(mut self, other: Buffer) raises:
        if other.elements() != self.elements():
            raise Error(
                "cannot copy "
                + String(other.elements())
                + " values into a buffer of "
                + String(self.elements())
            )
        for i in range(len(self.data)):
            self.data[i] = other.data[i]
