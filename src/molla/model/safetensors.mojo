"""The safetensors container, read out of a memory map.

A safetensors file is eight bytes of length, a JSON header of that many bytes,
and then the tensor data with nothing between the tensors. The header names
every tensor and gives its dtype, its shape, and the byte range it occupies in
the data section. That is the whole format, and the two things worth saying
about reading it are about trust and about size.

The byte ranges are absolute and they come out of a downloaded file, so every
one of them is checked against the length of the mapping before it is kept. An
unchecked range here is a read at whatever offset the file asked for, and the
failure mode is a segfault rather than an error. The check costs one comparison
per tensor and it runs once.

The declared range is also checked against the dtype and the shape. A BF16
tensor of shape [10, 10] occupies two hundred bytes and nothing else, so a
header that claims a different number is describing a file that no writer
produced, and refusing it is cheaper than discovering it three layers up. This
is the safetensors equivalent of the block geometry check on the GGUF side, and
it does the same job: it makes a directory checkable rather than merely
readable.

Sharded repositories are resolved here too. `model.safetensors.index.json` maps
every tensor name to the file it lives in, and a shard is a mapping like any
other, so the tensor list this hands back covers the whole model with a shard
index on each entry. Whether the index agrees with the shards is recorded
rather than assumed: a name in the map with no tensor behind it, and a tensor in
a shard that the map never mentions, are both counted.

Rank is capped at four dimensions. That is not a safetensors limit, it is the
GGUF limit, and molla holds both formats in the same shape so the layers above
do not have to ask which one a model came from. No real model carries a rank
five tensor. One that did would be refused here with a message saying so rather
than flattened into something that looks fine.
"""

from molla.io.buffer import Buffer
from molla.json.dom import (
    JS_ARRAY,
    JS_OBJECT,
    JS_STRING,
    NO_NODE,
    Document,
    parse,
)
from molla.json.reader import Reader
from molla.sys.file import DirEntry, exists, read_dir
from molla.sys.mmap import Mapping

comptime ST_MAX_DIMS = 4
"""Matches the GGUF limit, so one tensor struct describes either format."""

comptime ST_MAX_HEADER = 128 * 1024 * 1024
"""A header is tens of kilobytes on the largest model anyone ships. A file
claiming more than this is refused before any of it is parsed, because the
length prefix is attacker controlled and the parse is the expensive part."""

comptime ST_MAX_ELEMENTS = 1 << 48
"""A shape whose product runs past this is refused. The point is not the number,
it is that multiplying four dimensions out of a file can overflow and land on a
byte count that happens to match the declared one."""

comptime INDEX_NAME = "model.safetensors.index.json"
comptime SINGLE_NAME = "model.safetensors"


def dtype_bytes(name: StringSpan) -> Int:
    """Bytes one element of this dtype occupies. Zero for a name we do not know.

    Zero rather than an error, because a file using a dtype this build has never
    heard of is still a readable directory, and saying which tensor stopped us
    is more use than refusing the whole file.
    """
    if name == "F64" or name == "I64" or name == "U64":
        return 8
    if name == "F32" or name == "I32" or name == "U32":
        return 4
    if name == "F16" or name == "BF16" or name == "I16" or name == "U16":
        return 2
    if name == "I8" or name == "U8" or name == "BOOL":
        return 1
    if name == "F8_E4M3" or name == "F8_E5M2":
        return 1
    return 0


def dtype_narrow(name: StringSpan) -> Bool:
    """Whether this dtype is a compressed float rather than a stored one.

    The eight bit float types are the only thing in safetensors that is doing
    what a GGUF quantization does. Half precision is not: it is the width the
    weights were trained and saved at, and calling it quantized would report
    every fp16 repository on the Hub as compressed.
    """
    return name == "F8_E4M3" or name == "F8_E5M2"


def _text(bytes: Span[UInt8, MutAnyOrigin]) -> String:
    var out = List[UInt8]()
    out.reserve(len(bytes))
    for i in range(len(bytes)):
        out.append(bytes[i])
    return String(StringSpan(unsafe_from_utf8=out))


def _mul(a: Int, b: Int) raises -> Int:
    """Multiply, refusing an overflow rather than wrapping into a plausible
    number. A wrapped product that lands on the declared byte count is the one
    way a bad shape gets past the size check below."""
    if a < 0 or b < 0:
        raise Error("safetensors shape has a negative dimension")
    if b != 0 and a > ST_MAX_ELEMENTS // b:
        raise Error("safetensors shape has more elements than a file can hold")
    return a * b


@fieldwise_init
struct StTensor(Copyable, ImplicitlyCopyable, Movable):
    """One entry from a header, with the shard it was found in."""

    var name: String
    var dtype: String
    var n_dims: Int
    var d0: Int
    var d1: Int
    var d2: Int
    var d3: Int
    var begin: Int
    """Start of the tensor inside the data section of its shard."""

    var end: Int
    var shard: Int

    def dim(self, i: Int) -> Int:
        if i == 0:
            return self.d0
        if i == 1:
            return self.d1
        if i == 2:
            return self.d2
        return self.d3

    def elements(self) raises -> Int:
        var n = 1
        for i in range(self.n_dims):
            n = _mul(n, self.dim(i))
        return n

    def bytes(self) -> Int:
        return self.end - self.begin

    def shape_text(self) -> String:
        var out = String("")
        for i in range(self.n_dims):
            if i > 0:
                out += ", "
            out += String(self.dim(i))
        return out


struct Shard(Movable):
    """One mapped file, and where its data section starts."""

    var name: String
    var mapping: Mapping
    var header_bytes: Int
    var data_start: Int
    var tensors: Int

    def __init__(out self, var name: String, var mapping: Mapping):
        self.name = name^
        self.mapping = mapping^
        self.header_bytes = 0
        self.data_start = 0
        self.tensors = 0

    def close(mut self):
        self.mapping.close()


struct SafeTensors(Movable):
    """Every tensor in a model, across however many files it was written to."""

    var dir: String
    var shards: List[Shard]
    var tensors: List[StTensor]
    var format: String
    """`__metadata__.format`, which every writer sets to pt. Reported rather
    than enforced, because the format field says which framework saved the file
    and not how to read it."""

    var sharded: Bool
    var total_size: Int
    """`metadata.total_size` from the index, or -1 when there is no index."""

    var mapped_names: Int
    """Entries in the weight map. Zero when there is no index."""

    var missing: Int
    """Named in the index, absent from the shard the index named."""

    var unmapped: Int
    """Present in a shard, never named by the index."""

    def __init__(out self, path: StringSpan, counter: Int) raises:
        self.dir = String(path)
        # A trailing slash is what tab completion produces and it turns every
        # joined path into a double slash, which opens on both platforms and
        # then prints back wrong.
        while self.dir.byte_length() > 1 and self.dir.endswith("/"):
            var trimmed = String(
                self.dir[byte = 0 : self.dir.byte_length() - 1]
            )
            self.dir = trimmed^
        self.shards = List[Shard]()
        self.tensors = List[StTensor]()
        self.format = String("unstated")
        self.sharded = False
        self.total_size = -1
        self.mapped_names = 0
        self.missing = 0
        self.unmapped = 0
        try:
            self._open(counter)
        except e:
            self.close()
            raise e

    def _open(mut self, counter: Int) raises:
        if self.dir.endswith(".safetensors"):
            # A file was named rather than a directory. The companion JSON still
            # lives beside it, so the directory is where it sits.
            var file = self.dir
            var cut = file.rfind("/")
            self.dir = String(file[byte=0:cut]) if cut > 0 else String(".")
            self._read_shard(file[byte = cut + 1 :], counter)
            return

        if exists(self.dir + "/" + INDEX_NAME):
            self.sharded = True
            self._read_index(counter)
            return

        if exists(self.dir + "/" + SINGLE_NAME):
            self._read_shard(SINGLE_NAME, counter)
            return

        # Neither of the two names the ecosystem uses. A repository with one
        # differently named file is common enough to accept, and more than one
        # with no index is a repository we cannot order.
        var found = List[String]()
        var entries = List[DirEntry]()
        var listed = read_dir(self.dir, entries)
        if listed.is_err():
            raise Error("cannot read " + self.dir + " as a directory")
        for i in range(len(entries)):
            if entries[i].name.endswith(".safetensors"):
                found.append(entries[i].name)
        if len(found) == 1:
            self._read_shard(found[0], counter)
            return
        if len(found) == 0:
            raise Error("no safetensors file in " + self.dir)
        raise Error(
            String(len(found))
            + " safetensors files in "
            + self.dir
            + " and no "
            + INDEX_NAME
        )

    def _read_index(mut self, counter: Int) raises:
        var path = self.dir + "/" + INDEX_NAME
        var mapping = Mapping(path)
        var doc = Document(counter)
        var reader = Reader(counter, 4096)
        var body = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=mapping.base(), length=mapping.length
        )
        if not parse(doc, reader, body):
            mapping.close()
            raise Error(INDEX_NAME + " is not valid json")

        var meta = doc.get(doc.root, "metadata")
        if meta != NO_NODE:
            self.total_size = doc.get_int(meta, "total_size", -1)

        var weights = doc.get(doc.root, "weight_map")
        if weights == NO_NODE or doc.kind(weights) != JS_OBJECT:
            mapping.close()
            raise Error(INDEX_NAME + " has no weight_map")

        # The shards, in the order the map first mentions them, so a report
        # lists them the way the file does.
        var names = List[String]()
        var child = doc.first_child(weights)
        while child != NO_NODE:
            self.mapped_names += 1
            var shard = _text(doc.text(child))
            var known = False
            for i in range(len(names)):
                if names[i] == shard:
                    known = True
                    break
            if not known:
                names.append(shard)
            child = doc.next_sibling(child)

        for i in range(len(names)):
            self._read_shard(names[i], counter)

        # Every name the map promised has to be behind one of those shards, and
        # every tensor in a shard has to have been promised. Both directions
        # matter: the first is a load that fails at the first missing weight and
        # the second is a weight nothing will ever look for.
        child = doc.first_child(weights)
        while child != NO_NODE:
            var name = _text(doc.key(child))
            var shard = _text(doc.text(child))
            var at = self.find(name)
            if at < 0 or self.shards[self.tensors[at].shard].name != shard:
                self.missing += 1
            child = doc.next_sibling(child)

        for i in range(len(self.tensors)):
            if doc.get(weights, self.tensors[i].name) == NO_NODE:
                self.unmapped += 1

        mapping.close()

    def _read_shard(mut self, name: StringSpan, counter: Int) raises:
        var path = self.dir + "/" + String(name)
        var shard = Shard(String(name), Mapping(path))
        try:
            self._parse_header(shard, counter)
        except e:
            shard.close()
            raise e
        self.shards.append(shard^)

    def _parse_header(mut self, mut shard: Shard, counter: Int) raises:
        var mapping_length = shard.mapping.length
        if mapping_length < 8:
            raise Error(shard.name + " is shorter than a safetensors header")

        var header_bytes = 0
        for i in range(8):
            header_bytes |= Int(shard.mapping.base().unsafe_load(i)) << (8 * i)
        if header_bytes < 0 or header_bytes > ST_MAX_HEADER:
            raise Error(
                shard.name
                + " declares a "
                + String(header_bytes)
                + " byte header, which is not a header"
            )
        if 8 + header_bytes > mapping_length:
            raise Error(shard.name + " header runs past the end of the file")

        shard.header_bytes = header_bytes
        shard.data_start = 8 + header_bytes
        var data_length = mapping_length - shard.data_start

        var doc = Document(counter)
        var reader = Reader(counter, 4096)
        var body = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=shard.mapping.base().unsafe_offset(8),
            length=header_bytes,
        )
        if not parse(doc, reader, body):
            raise Error(shard.name + " has a header that is not valid json")
        if doc.kind(doc.root) != JS_OBJECT:
            raise Error(shard.name + " has a header that is not an object")

        var index = len(self.shards)
        var child = doc.first_child(doc.root)
        while child != NO_NODE:
            var key = _text(doc.key(child))
            if key == "__metadata__":
                var f = doc.get(child, "format")
                if f != NO_NODE and doc.kind(f) == JS_STRING:
                    self.format = _text(doc.text(f))
                child = doc.next_sibling(child)
                continue
            if doc.kind(child) != JS_OBJECT:
                raise Error(shard.name + ": " + key + " is not a tensor entry")

            var dtype_node = doc.get(child, "dtype")
            if dtype_node == NO_NODE or doc.kind(dtype_node) != JS_STRING:
                raise Error(shard.name + ": " + key + " has no dtype")
            var dtype = _text(doc.text(dtype_node))

            var shape = doc.get(child, "shape")
            if shape == NO_NODE or doc.kind(shape) != JS_ARRAY:
                raise Error(shard.name + ": " + key + " has no shape")
            var rank = doc.size(shape)
            if rank > ST_MAX_DIMS:
                raise Error(
                    shard.name
                    + ": "
                    + key
                    + " has "
                    + String(rank)
                    + " dimensions and molla holds at most "
                    + String(ST_MAX_DIMS)
                )
            var d0 = 1
            var d1 = 1
            var d2 = 1
            var d3 = 1
            var dim = doc.first_child(shape)
            var at = 0
            while dim != NO_NODE:
                var value = doc.as_int(dim, -1)
                if value < 0:
                    raise Error(shard.name + ": " + key + " has a bad shape")
                if at == 0:
                    d0 = value
                elif at == 1:
                    d1 = value
                elif at == 2:
                    d2 = value
                else:
                    d3 = value
                at += 1
                dim = doc.next_sibling(dim)

            var offsets = doc.get(child, "data_offsets")
            if offsets == NO_NODE or doc.size(offsets) != 2:
                raise Error(
                    shard.name + ": " + key + " has no pair of data offsets"
                )
            var begin = doc.as_int(doc.at(offsets, 0), -1)
            var end = doc.as_int(doc.at(offsets, 1), -1)
            if begin < 0 or end < begin or end > data_length:
                raise Error(
                    shard.name
                    + ": "
                    + key
                    + " occupies bytes "
                    + String(begin)
                    + " to "
                    + String(end)
                    + " of a "
                    + String(data_length)
                    + " byte data section"
                )

            var tensor = StTensor(
                name=key,
                dtype=dtype,
                n_dims=rank,
                d0=d0,
                d1=d1,
                d2=d2,
                d3=d3,
                begin=begin,
                end=end,
                shard=index,
            )

            # The size check. A known dtype and a shape give exactly one byte
            # count, and a header that disagrees is describing a file that
            # nothing wrote.
            var width = dtype_bytes(dtype)
            if width > 0:
                var want = _mul(tensor.elements(), width)
                if want != tensor.bytes():
                    raise Error(
                        shard.name
                        + ": "
                        + key
                        + " is "
                        + dtype
                        + " of ["
                        + tensor.shape_text()
                        + "], which is "
                        + String(want)
                        + " bytes, and the header gives it "
                        + String(tensor.bytes())
                    )

            self.tensors.append(tensor)
            shard.tensors += 1
            child = doc.next_sibling(child)

    def close(mut self):
        for i in range(len(self.shards)):
            self.shards[i].close()

    def bytes(self) -> Int:
        var total = 0
        for i in range(len(self.tensors)):
            total += self.tensors[i].bytes()
        return total

    def find(self, name: StringSpan) -> Int:
        for i in range(len(self.tensors)):
            if self.tensors[i].name == name:
                return i
        return -1

    def prefixed(self, prefix: StringSpan) -> Bool:
        """Whether any tensor name starts with this prefix."""
        for i in range(len(self.tensors)):
            if self.tensors[i].name.startswith(prefix):
                return True
        return False

    def dominant_dtype(self) -> String:
        """The dtype the most bytes are stored in.

        By bytes rather than by tensor count, because the count says F32 for a
        half precision model: the norms and the biases are stored at full width
        and there are more of them than there are weight matrices.
        """
        var names = List[String]()
        var totals = List[Int]()
        for i in range(len(self.tensors)):
            var t = self.tensors[i]
            var at = -1
            for j in range(len(names)):
                if names[j] == t.dtype:
                    at = j
                    break
            if at < 0:
                names.append(t.dtype)
                totals.append(0)
                at = len(names) - 1
            totals[at] += t.bytes()
        var best = -1
        for i in range(len(totals)):
            if best < 0 or totals[i] > totals[best]:
                best = i
        if best < 0:
            return String("none")
        return names[best]

    def dtype_count(self) -> Int:
        var names = List[String]()
        for i in range(len(self.tensors)):
            var known = False
            for j in range(len(names)):
                if names[j] == self.tensors[i].dtype:
                    known = True
                    break
            if not known:
                names.append(self.tensors[i].dtype)
        return len(names)


def run_safetensors(path: StringSpan, counter: Int) raises:
    """Entry point for `molla safetensors`. The directory, and nothing else."""
    var st = SafeTensors(path, counter)
    print("path:          " + String(path))
    print("shards:        " + String(len(st.shards)))
    for i in range(len(st.shards)):
        print(
            "  "
            + st.shards[i].name
            + " header "
            + String(st.shards[i].header_bytes)
            + " bytes, data at "
            + String(st.shards[i].data_start)
            + ", "
            + String(st.shards[i].tensors)
            + " tensors"
        )
    print("format:        " + st.format)
    print("tensor count:  " + String(len(st.tensors)))
    print("weight bytes:  " + String(st.bytes()))
    if st.sharded:
        print("index size:    " + String(st.total_size))
        print("index names:   " + String(st.mapped_names))
        print("missing:       " + String(st.missing))
        print("unmapped:      " + String(st.unmapped))
    print("")
    print("tensor directory:")
    for i in range(len(st.tensors)):
        var t = st.tensors[i]
        print(
            "  "
            + t.name
            + " ["
            + t.shape_text()
            + "] "
            + t.dtype
            + " at "
            + String(t.begin)
            + " in "
            + st.shards[t.shard].name
        )
    st.close()
