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

A `Tensor` also says which memory its address is in. That is one integer and it
exists because the alternative is a segfault: a device pointer is a number the
driver understands and a load instruction does not, so a host kernel handed one
does not fail where the mistake was made. `base` is the single call every host
kernel makes to get at the bytes, so the check goes there and costs an integer
compare against a matvec.
"""

from molla.nn.quant import block_bytes, block_elements, supported
from molla.nn.repack import LAYOUT_GGML, LAYOUT_PLANAR, planar_row_bytes
from molla.sys.mmap import RawPtr

comptime WHERE_HOST = 0
"""In host memory. The model file's mapping, the repack cache, or a buffer."""

comptime WHERE_UNIFIED = 1
"""In host memory on a machine whose accelerator shares the same physical pool.
Host kernels read it exactly as they read `WHERE_HOST`, and the saving is real:
a unified plan allocates nothing and copies nothing, which is measured in
`docs/validation/load.md`.

What it does not mean is that a device kernel can be handed this address. It
cannot. Sharing a pool is not sharing an address space here: a device buffer's
own pointer segfaults when read from the host, `map_to_host` returns a different
address for the same bytes with no copy between them, and a Metal kernel given a
host allocation reads zeros without faulting rather than reporting anything.
This place is host memory with a note about what it did not cost, and issue #152
is where it earns a device address."""

comptime WHERE_DEVICE = 2
"""In a device pool. A host kernel cannot follow this address."""


def place_name(place: Int) -> String:
    if place == WHERE_HOST:
        return String("host")
    if place == WHERE_UNIFIED:
        return String("unified")
    return String("device")


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

    var layout: Int
    """Whether the bytes are the file's or the repack's.

    `LAYOUT_GGML` means blocks exactly as the GGUF has them. `LAYOUT_PLANAR`
    means `molla.nn.repack` has been over them and the address points into a
    cache rather than into the model file. The type number is unchanged either
    way, because the planar layout keeps the group size and the presence of a
    minimum from the type it came from and needs to be told which one that was.
    """

    var place: Int
    """Which memory the address is in. One of `WHERE_HOST`, `WHERE_UNIFIED` or
    `WHERE_DEVICE`. Independent of the layout: a device pool holds planar bytes
    on a machine that has a repack cache and ggml bytes on one that does not,
    and those are two questions with different answers about the same weight."""

    def __init__(
        out self,
        address: Int,
        kind: Int,
        cols: Int,
        rows: Int,
        layout: Int = LAYOUT_GGML,
        place: Int = WHERE_HOST,
    ):
        self.address = address
        self.kind = kind
        self.cols = cols
        self.rows = rows
        self.layout = layout
        self.place = place

    @staticmethod
    def none() -> Self:
        """A weight the model does not have.

        Address zero, which is not a place any mapping can start, so a tensor
        that is absent and a tensor that is present are told apart by the one
        field that cannot lie about it. The alternative is a boolean beside
        every weight, and two things that have to agree are one thing that can
        disagree.
        """
        return Self(0, 0, 0, 0, LAYOUT_GGML, WHERE_HOST)

    def present(self) -> Bool:
        return self.address != 0

    def on_device(self) -> Bool:
        """Whether a host kernel has to be kept away from these bytes.

        Unified is not on the device by this question's meaning, and it is the
        only place where that is the interesting half of the answer: a unified
        weight is host memory a host kernel reads normally. It is not a device
        address either, which is the other half and is `device_address`.
        """
        return self.place == WHERE_DEVICE

    def base(self) raises -> RawPtr:
        """The bytes, for a host kernel to read.

        Raises for a weight that lives in a device pool. That address means
        something to the driver and nothing to a load instruction, so following
        one is a fault somewhere far from whatever placed the tensor, on a
        thread that has no idea a placement happened. Every host kernel goes
        through here, so this is the one place the question has to be asked,
        and it is asked once per matvec rather than once per row.
        """
        if self.place == WHERE_DEVICE:
            raise Error(
                "a "
                + String(self.cols)
                + " by "
                + String(self.rows)
                + " weight is on the device and a host kernel cannot read it"
            )
        return RawPtr(unsafe_from_address=self.address)

    def device_address(self) raises -> Int:
        """The address, for a device kernel.

        The mirror of `base`, and it refuses the host case for the same reason
        in the other direction, except that the fault is worse in this
        direction. A host kernel following a device address crashes. A device
        kernel following a host address reads zeros and reports nothing, so a
        model bound that way runs at full speed and answers with noise.

        Unified is refused too, which is a correction. This used to let it
        through on the grounds that one pool of memory means one address, and
        that turned out to be false on the machine it was written for: a device
        buffer's own pointer segfaults when read from the host, `map_to_host`
        hands back a different address for the same bytes, and a kernel given
        the host one reads zeros. Making a unified weight carry an address a
        device can use is issue #152, and until then the honest answer is that
        it has not got one.
        """
        if self.place != WHERE_DEVICE:
            raise Error(
                "a "
                + String(self.cols)
                + " by "
                + String(self.rows)
                + " weight is "
                + place_name(self.place)
                + " resident and has no address a device kernel can read. A"
                " device kernel reads a host address as zeros rather than as an"
                " error, so this refuses rather than returning one"
            )
        return self.address

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
        if self.layout == LAYOUT_PLANAR:
            return planar_row_bytes(self.kind, self.cols)
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

    def as_planar(self, address: Int) -> Self:
        """The same weight and shape, pointing at its repacked bytes.

        The address changes because a repack is a second copy in a cache and not
        an edit of the mapped file, so a tensor that has been repacked and one
        that has not are two views and never the same one mutated.
        """
        return Self(
            address, self.kind, self.cols, self.rows, LAYOUT_PLANAR, self.place
        )

    def resident(self, address: Int, place: Int) -> Self:
        """The same weight and layout, in another memory.

        The layout rides along because a copy to a device is a copy of bytes
        and not a transform of them. A planar weight uploaded to a card is
        still planar and a ggml one still needs unpacking, and the kernel on
        the other side has to be told which without going back to the file.
        """
        return Self(
            address, self.kind, self.cols, self.rows, self.layout, place
        )


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
