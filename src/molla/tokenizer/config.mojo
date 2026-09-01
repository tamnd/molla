"""Reading a `tokenizer.json`.

The file is a pipeline written down as JSON, and the awkward part is its size.
A Qwen file is 17 MB and almost all of it is two things: an object with a
hundred and fifty thousand keys, and an array with a hundred and fifty thousand
merges. Building a document tree for that is hundreds of megabytes and several
seconds, so this reads the file as a stream of events and puts each vocabulary
entry straight into the arena it will live in.

The five stages are read the same way. Each one is an object with a `type` and
some fields, the fields differ per type, and a `Sequence` holds a list of more
of them. Rather than parse each type separately and depend on `type` arriving
before the fields it selects, every component is read into one bag of every
field any component has, and the bag is turned into a step once the object is
closed and the type is known for certain.
"""

from molla.json.reader import (
    EV_ARRAY_BEGIN,
    EV_ARRAY_END,
    EV_BOOL,
    EV_END,
    EV_ERROR,
    EV_KEY,
    EV_NULL,
    EV_NUMBER,
    EV_OBJECT_BEGIN,
    EV_OBJECT_END,
    EV_STRING,
    Reader,
)
from molla.sys.mem import as_ptr
from molla.sys.mmap import Mapping
from molla.text.props import Unicode
from molla.text.regex import Regex
from molla.text.utf8 import to_code_points

from .added import AddedToken, AddedVocabulary
from .decoder import (
    D_BPE,
    D_BYTE_FALLBACK,
    D_BYTE_LEVEL,
    D_FUSE,
    D_METASPACE,
    D_REPLACE,
    D_STRIP,
    D_WORDPIECE,
    DecodeStep,
    Decoder,
)
from .model import M_BPE, M_UNIGRAM, M_WORDLEVEL, M_WORDPIECE, Model
from .normalizer import (
    N_BERT,
    N_LOWERCASE,
    N_NFC,
    N_NFD,
    N_NFKC,
    N_NFKD,
    N_NMT,
    N_PREPEND,
    N_REPLACE_REGEX,
    N_REPLACE_STRING,
    N_STRIP,
    N_STRIP_ACCENTS,
    NormStep,
    Normalizer,
)
from .post import I_SEQUENCE_A, I_SEQUENCE_B, I_SPECIAL, PostItem, PostProcessor
from .pretok import (
    B_CONTIGUOUS,
    B_ISOLATED,
    B_MERGED_WITH_NEXT,
    B_MERGED_WITH_PREVIOUS,
    B_REMOVED,
    GPT2_PATTERN,
    PREPEND_ALWAYS,
    PREPEND_FIRST,
    PREPEND_NEVER,
    T_BERT,
    T_BYTE_LEVEL,
    T_CHAR_DELIMITER,
    T_DIGITS,
    T_METASPACE,
    T_PUNCTUATION,
    T_SPLIT_REGEX,
    T_SPLIT_STRING,
    T_WHITESPACE,
    T_WHITESPACE_SPLIT,
    WHITESPACE_PATTERN,
    PreStep,
    PreTokenizer,
)
from .vocab import NO_ID


def _text_of(bytes: List[UInt8]) -> String:
    return String(StringSpan(unsafe_from_utf8=bytes))


def _first_point(bytes: List[UInt8]) -> Int:
    """The first code point of a string, or a space when it is empty.

    Metaspace and its decoder carry their replacement as a one character
    string, and a missing one means the default, which everything that uses it
    writes as U+2581.
    """
    var points = to_code_points(bytes)
    if len(points) == 0:
        return 0x2581
    return points[0]


struct Spec(Copyable, Movable):
    """Every field any component can carry, plus its children.

    One struct for all of them is not elegant and it is the shape the file
    wants: the reader cannot know which fields matter until it has seen the
    type, and the type is not guaranteed to come first.
    """

    var kind: List[UInt8]
    var children: List[Int]
    """Indices into the pool that holds this spec, since a struct cannot
    contain itself."""

    var pattern: List[UInt8]
    var pattern_is_regex: Bool
    var content: List[UInt8]
    var prepend: List[UInt8]
    var prefix: List[UInt8]
    var suffix: List[UInt8]
    var replacement: List[UInt8]
    var behavior: List[UInt8]
    var prepend_scheme: List[UInt8]
    var sep_token: List[UInt8]
    var cls_token: List[UInt8]
    var sep_id: Int
    var cls_id: Int
    var start: Int
    var stop: Int
    var max_input_chars: Int
    var invert: Bool
    var add_prefix_space: Bool
    var trim_offsets: Bool
    var use_regex: Bool
    var individual_digits: Bool
    var lowercase: Bool
    var clean_text: Bool
    var handle_chinese: Bool
    var strip_accents: Bool
    var strip_accents_set: Bool
    var strip_left: Bool
    var strip_right: Bool
    var cleanup: Bool
    var split: Bool

    var template_kind: List[Int]
    var template_type: List[Int]
    var template_name_at: List[Int]
    var template_name_length: List[Int]
    var template_names: List[UInt8]
    var single_count: Int

    var special_name_at: List[Int]
    var special_name_length: List[Int]
    var special_ids_at: List[Int]
    var special_ids_length: List[Int]
    var special_names: List[UInt8]
    var special_ids: List[Int]

    def __init__(out self):
        self.kind = List[UInt8]()
        self.children = List[Int]()
        self.pattern = List[UInt8]()
        self.pattern_is_regex = False
        self.content = List[UInt8]()
        self.prepend = List[UInt8]()
        self.prefix = List[UInt8]()
        self.suffix = List[UInt8]()
        self.replacement = List[UInt8]()
        self.behavior = List[UInt8]()
        self.prepend_scheme = List[UInt8]()
        self.sep_token = List[UInt8]()
        self.cls_token = List[UInt8]()
        self.sep_id = NO_ID
        self.cls_id = NO_ID
        self.start = 0
        self.stop = 0
        self.max_input_chars = 100
        self.invert = False
        self.add_prefix_space = True
        self.trim_offsets = True
        self.use_regex = True
        self.individual_digits = False
        self.lowercase = True
        self.clean_text = True
        self.handle_chinese = True
        self.strip_accents = False
        self.strip_accents_set = False
        self.strip_left = True
        self.strip_right = True
        self.cleanup = True
        self.split = True
        self.template_kind = List[Int]()
        self.template_type = List[Int]()
        self.template_name_at = List[Int]()
        self.template_name_length = List[Int]()
        self.template_names = List[UInt8]()
        self.single_count = 0
        self.special_name_at = List[Int]()
        self.special_name_length = List[Int]()
        self.special_ids_at = List[Int]()
        self.special_ids_length = List[Int]()
        self.special_names = List[UInt8]()
        self.special_ids = List[Int]()

    def is_a(self, name: StringSpan) -> Bool:
        var wanted = name.as_bytes()
        if len(self.kind) != len(wanted):
            return False
        for i in range(len(self.kind)):
            if self.kind[i] != wanted[i]:
                return False
        return True

    def is_empty(self) -> Bool:
        return len(self.kind) == 0 and len(self.children) == 0


def _copy_text(mut reader: Reader) -> List[UInt8]:
    var out = List[UInt8]()
    var span = reader.text()
    for i in range(len(span)):
        out.append(span[i])
    return out^


def _read_string(mut reader: Reader) raises -> List[UInt8]:
    """The next value as bytes. Null and anything else come back empty."""
    var event = reader.next()
    if event == EV_STRING:
        return _copy_text(reader)
    if event == EV_NULL or event == EV_BOOL or event == EV_NUMBER:
        return List[UInt8]()
    if event == EV_OBJECT_BEGIN or event == EV_ARRAY_BEGIN:
        _ = reader.skip_value()
    return List[UInt8]()


def _read_bool(mut reader: Reader, fallback: Bool) raises -> Bool:
    var event = reader.next()
    if event == EV_BOOL:
        return reader.bool_value
    if event == EV_OBJECT_BEGIN or event == EV_ARRAY_BEGIN:
        _ = reader.skip_value()
    return fallback


def _read_int(mut reader: Reader, fallback: Int) raises -> Int:
    var event = reader.next()
    if event == EV_NUMBER:
        return reader.number.as_int()
    if event == EV_OBJECT_BEGIN or event == EV_ARRAY_BEGIN:
        _ = reader.skip_value()
    return fallback


def _read_pattern(mut reader: Reader, mut spec: Spec) raises:
    """A pattern is either `{"String": "x"}` or `{"Regex": "x"}`.

    Which one it is decides whether the step below compiles an expression or
    compares bytes, and getting it backwards means a pattern of `.` matching
    every character instead of one full stop.
    """
    var event = reader.next()
    if event != EV_OBJECT_BEGIN:
        if event == EV_STRING:
            spec.pattern = _copy_text(reader)
        elif event == EV_ARRAY_BEGIN:
            _ = reader.skip_value()
        return
    while True:
        var member = reader.next()
        if member == EV_OBJECT_END or member == EV_END or member == EV_ERROR:
            return
        if member != EV_KEY:
            continue
        if reader.key_is("Regex"):
            spec.pattern_is_regex = True
            spec.pattern = _read_string(reader)
        elif reader.key_is("String"):
            spec.pattern_is_regex = False
            spec.pattern = _read_string(reader)
        else:
            _ = reader.skip_next_value()


def _read_template(mut reader: Reader, mut spec: Spec, single: Bool) raises:
    """One side of a TemplateProcessing, as items appended to the spec."""
    var event = reader.next()
    if event != EV_ARRAY_BEGIN:
        if event == EV_OBJECT_BEGIN or event == EV_STRING:
            _ = reader.skip_value()
        return
    while True:
        var item = reader.next()
        if item == EV_ARRAY_END or item == EV_END or item == EV_ERROR:
            break
        if item != EV_OBJECT_BEGIN:
            _ = reader.skip_value()
            continue
        var kind = I_SPECIAL
        var type_id = 0
        var name = List[UInt8]()
        while True:
            var member = reader.next()
            if member == EV_OBJECT_END or member == EV_END:
                break
            if member != EV_KEY:
                continue
            var sequence = reader.key_is("Sequence")
            if not sequence and not reader.key_is("SpecialToken"):
                _ = reader.skip_next_value()
                continue
            kind = I_SEQUENCE_A if sequence else I_SPECIAL
            if reader.next() != EV_OBJECT_BEGIN:
                continue
            while True:
                var field = reader.next()
                if field == EV_OBJECT_END or field == EV_END:
                    break
                if field != EV_KEY:
                    continue
                if reader.key_is("id"):
                    name = _read_string(reader)
                elif reader.key_is("type_id"):
                    type_id = _read_int(reader, 0)
                else:
                    _ = reader.skip_next_value()
            if sequence:
                kind = I_SEQUENCE_A
                if len(name) == 1 and name[0] == UInt8(ord("B")):
                    kind = I_SEQUENCE_B
                name.clear()
        spec.template_kind.append(kind)
        spec.template_type.append(type_id)
        spec.template_name_at.append(len(spec.template_names))
        spec.template_name_length.append(len(name))
        for i in range(len(name)):
            spec.template_names.append(name[i])
    if single:
        spec.single_count = len(spec.template_kind)


def _read_special_tokens(mut reader: Reader, mut spec: Spec) raises:
    """The map from a template name to the ids it stands for."""
    if reader.next() != EV_OBJECT_BEGIN:
        return
    while True:
        var member = reader.next()
        if member == EV_OBJECT_END or member == EV_END or member == EV_ERROR:
            return
        if member != EV_KEY:
            continue
        var name = _copy_text(reader)
        if reader.next() != EV_OBJECT_BEGIN:
            continue
        var ids = List[Int]()
        while True:
            var field = reader.next()
            if field == EV_OBJECT_END or field == EV_END:
                break
            if field != EV_KEY:
                continue
            if reader.key_is("ids"):
                if reader.next() != EV_ARRAY_BEGIN:
                    continue
                while True:
                    var one = reader.next()
                    if one == EV_ARRAY_END or one == EV_END:
                        break
                    if one == EV_NUMBER:
                        ids.append(reader.number.as_int())
            else:
                _ = reader.skip_next_value()
        spec.special_name_at.append(len(spec.special_names))
        spec.special_name_length.append(len(name))
        for i in range(len(name)):
            spec.special_names.append(name[i])
        spec.special_ids_at.append(len(spec.special_ids))
        spec.special_ids_length.append(len(ids))
        for i in range(len(ids)):
            spec.special_ids.append(ids[i])


struct Specs(Movable):
    """A component and everything nested inside it, flattened.

    A Sequence holds other components, so the natural shape is a tree, and a
    Mojo struct cannot contain a list of itself. So the nodes live in one list
    and a parent keeps the indices of its children.
    """

    var nodes: List[Spec]
    var root: Int

    def __init__(out self):
        self.nodes = List[Spec]()
        self.root = -1

    def at(self, index: Int) -> Spec:
        return self.nodes[index].copy()


def read_spec(mut reader: Reader) raises -> Specs:
    """The next value as a component, or an empty pool if it is null."""
    var specs = Specs()
    var event = reader.next()
    if event != EV_OBJECT_BEGIN:
        if event == EV_ARRAY_BEGIN:
            _ = reader.skip_value()
        return specs^
    var root = _read_node(reader, specs)
    specs.root = len(specs.nodes)
    specs.nodes.append(root^)
    return specs^


def _read_node(mut reader: Reader, mut specs: Specs) raises -> Spec:
    """The members of a component whose opening brace has already been read."""
    var spec = Spec()
    while True:
        var member = reader.next()
        if member == EV_OBJECT_END or member == EV_END or member == EV_ERROR:
            break
        if member != EV_KEY:
            continue
        if reader.key_is("type"):
            spec.kind = _read_string(reader)
        elif (
            reader.key_is("normalizers")
            or reader.key_is("pretokenizers")
            or reader.key_is("decoders")
            or reader.key_is("processors")
        ):
            if reader.next() == EV_ARRAY_BEGIN:
                while True:
                    var item = reader.next()
                    if item == EV_ARRAY_END or item == EV_END:
                        break
                    if item != EV_OBJECT_BEGIN:
                        _ = reader.skip_value()
                        continue
                    var child = _read_node(reader, specs)
                    spec.children.append(len(specs.nodes))
                    specs.nodes.append(child^)
        elif reader.key_is("pattern"):
            _read_pattern(reader, spec)
        elif reader.key_is("content"):
            spec.content = _read_string(reader)
        elif reader.key_is("prepend"):
            spec.prepend = _read_string(reader)
        elif reader.key_is("prefix"):
            spec.prefix = _read_string(reader)
        elif reader.key_is("suffix"):
            spec.suffix = _read_string(reader)
        elif reader.key_is("replacement") or reader.key_is("delimiter"):
            spec.replacement = _read_string(reader)
        elif reader.key_is("behavior"):
            spec.behavior = _read_string(reader)
        elif reader.key_is("prepend_scheme"):
            spec.prepend_scheme = _read_string(reader)
        elif reader.key_is("sep"):
            _read_pair(reader, spec, True)
        elif reader.key_is("cls"):
            _read_pair(reader, spec, False)
        elif reader.key_is("start"):
            spec.start = _read_int(reader, 0)
        elif reader.key_is("stop"):
            spec.stop = _read_int(reader, 0)
        elif reader.key_is("invert"):
            spec.invert = _read_bool(reader, False)
        elif reader.key_is("add_prefix_space"):
            spec.add_prefix_space = _read_bool(reader, True)
        elif reader.key_is("trim_offsets"):
            spec.trim_offsets = _read_bool(reader, True)
        elif reader.key_is("use_regex"):
            spec.use_regex = _read_bool(reader, True)
        elif reader.key_is("individual_digits"):
            spec.individual_digits = _read_bool(reader, False)
        elif reader.key_is("lowercase"):
            spec.lowercase = _read_bool(reader, True)
        elif reader.key_is("clean_text"):
            spec.clean_text = _read_bool(reader, True)
        elif reader.key_is("handle_chinese_chars"):
            spec.handle_chinese = _read_bool(reader, True)
        elif reader.key_is("strip_accents"):
            var event_kind = reader.next()
            if event_kind == EV_BOOL:
                spec.strip_accents = reader.bool_value
                spec.strip_accents_set = True
        elif reader.key_is("strip_left"):
            spec.strip_left = _read_bool(reader, True)
        elif reader.key_is("strip_right"):
            spec.strip_right = _read_bool(reader, True)
        elif reader.key_is("cleanup"):
            spec.cleanup = _read_bool(reader, True)
        elif reader.key_is("split"):
            spec.split = _read_bool(reader, True)
        elif reader.key_is("single"):
            _read_template(reader, spec, True)
        elif reader.key_is("pair"):
            _read_template(reader, spec, False)
        elif reader.key_is("special_tokens"):
            _read_special_tokens(reader, spec)
        else:
            _ = reader.skip_next_value()
    return spec^


def _read_pair(mut reader: Reader, mut spec: Spec, is_sep: Bool) raises:
    """The `["[SEP]", 102]` pair a BERT or Roberta processor carries."""
    var event = reader.next()
    if event != EV_ARRAY_BEGIN:
        if event == EV_OBJECT_BEGIN:
            _ = reader.skip_value()
        return
    var text = List[UInt8]()
    var id = NO_ID
    while True:
        var item = reader.next()
        if item == EV_ARRAY_END or item == EV_END or item == EV_ERROR:
            break
        if item == EV_STRING:
            text = _copy_text(reader)
        elif item == EV_NUMBER:
            id = reader.number.as_int()
    if is_sep:
        spec.sep_token = text^
        spec.sep_id = id
    else:
        spec.cls_token = text^
        spec.cls_id = id


def _behavior_of(name: List[UInt8]) raises -> Int:
    """The four ways a split can treat its delimiter, plus contiguous."""
    var text = _text_of(name)
    if text == "Removed":
        return B_REMOVED
    if text == "Isolated":
        return B_ISOLATED
    if text == "MergedWithPrevious":
        return B_MERGED_WITH_PREVIOUS
    if text == "MergedWithNext":
        return B_MERGED_WITH_NEXT
    if text == "Contiguous":
        return B_CONTIGUOUS
    if len(name) == 0:
        return B_ISOLATED
    raise Error("unknown split behavior " + text)


def _prepend_of(name: List[UInt8]) -> Int:
    var text = _text_of(name)
    if text == "never":
        return PREPEND_NEVER
    if text == "first":
        return PREPEND_FIRST
    return PREPEND_ALWAYS


def build_normalizer(
    tables: Unicode, specs: Specs, at: Int, mut into: Normalizer
) raises:
    """One spec into zero or more steps, flattening sequences as it goes."""
    if at < 0:
        return
    var spec = specs.at(at)
    if len(spec.kind) == 0:
        return
    if spec.is_a("Sequence"):
        for i in range(len(spec.children)):
            build_normalizer(tables, specs, spec.children[i], into)
        return

    if spec.is_a("NFC"):
        into.steps.append(NormStep(N_NFC))
    elif spec.is_a("NFD"):
        into.steps.append(NormStep(N_NFD))
    elif spec.is_a("NFKC"):
        into.steps.append(NormStep(N_NFKC))
    elif spec.is_a("NFKD"):
        into.steps.append(NormStep(N_NFKD))
    elif spec.is_a("Lowercase"):
        into.steps.append(NormStep(N_LOWERCASE))
    elif spec.is_a("StripAccents"):
        into.steps.append(NormStep(N_STRIP_ACCENTS))
    elif spec.is_a("Nmt"):
        into.steps.append(NormStep(N_NMT))
    elif spec.is_a("Strip"):
        var step = NormStep(N_STRIP)
        step.strip_left = spec.strip_left
        step.strip_right = spec.strip_right
        into.steps.append(step^)
    elif spec.is_a("Prepend"):
        var step = NormStep(N_PREPEND)
        step.content = to_code_points(spec.prepend)
        into.steps.append(step^)
    elif spec.is_a("Replace"):
        if spec.pattern_is_regex:
            var step = NormStep(N_REPLACE_REGEX)
            step.expression = len(into.expressions)
            step.content = to_code_points(spec.content)
            into.expressions.append(Regex(_text_of(spec.pattern), tables))
            into.steps.append(step^)
        else:
            var step = NormStep(N_REPLACE_STRING)
            step.pattern = to_code_points(spec.pattern)
            step.content = to_code_points(spec.content)
            into.steps.append(step^)
    elif spec.is_a("BertNormalizer"):
        var step = NormStep(N_BERT)
        step.clean_text = spec.clean_text
        step.handle_chinese = spec.handle_chinese
        step.lowercase = spec.lowercase
        step.strip_accents = (
            spec.strip_accents if spec.strip_accents_set else spec.lowercase
        )
        into.steps.append(step^)
    else:
        raise Error("unsupported normalizer " + _text_of(spec.kind))


def build_pretokenizer(
    tables: Unicode, specs: Specs, at: Int, mut into: PreTokenizer
) raises:
    if at < 0:
        return
    var spec = specs.at(at)
    if len(spec.kind) == 0:
        return
    if spec.is_a("Sequence"):
        for i in range(len(spec.children)):
            build_pretokenizer(tables, specs, spec.children[i], into)
        return

    if spec.is_a("ByteLevel"):
        var step = PreStep(T_BYTE_LEVEL)
        step.add_prefix_space = spec.add_prefix_space
        step.use_regex = spec.use_regex
        step.trim_offsets = spec.trim_offsets
        if spec.use_regex:
            step.expression = len(into.expressions)
            into.expressions.append(Regex(GPT2_PATTERN, tables))
        into.steps.append(step^)
    elif spec.is_a("Whitespace"):
        var step = PreStep(T_WHITESPACE)
        step.behavior = B_REMOVED
        step.invert = True
        step.expression = len(into.expressions)
        into.expressions.append(Regex(WHITESPACE_PATTERN, tables))
        into.steps.append(step^)
    elif spec.is_a("WhitespaceSplit"):
        var step = PreStep(T_WHITESPACE_SPLIT)
        step.behavior = B_REMOVED
        into.steps.append(step^)
    elif spec.is_a("BertPreTokenizer"):
        into.steps.append(PreStep(T_BERT))
    elif spec.is_a("Punctuation"):
        var step = PreStep(T_PUNCTUATION)
        step.behavior = _behavior_of(spec.behavior)
        into.steps.append(step^)
    elif spec.is_a("Digits"):
        var step = PreStep(T_DIGITS)
        step.individual_digits = spec.individual_digits
        step.behavior = B_ISOLATED if spec.individual_digits else B_CONTIGUOUS
        into.steps.append(step^)
    elif spec.is_a("CharDelimiterSplit"):
        var step = PreStep(T_CHAR_DELIMITER)
        step.behavior = B_REMOVED
        step.pattern = to_code_points(spec.replacement)
        into.steps.append(step^)
    elif spec.is_a("Metaspace"):
        var step = PreStep(T_METASPACE)
        step.replacement = _first_point(spec.replacement)
        step.prepend_scheme = _prepend_of(spec.prepend_scheme)
        step.split = spec.split
        into.steps.append(step^)
    elif spec.is_a("Split"):
        if spec.pattern_is_regex:
            var step = PreStep(T_SPLIT_REGEX)
            step.behavior = _behavior_of(spec.behavior)
            step.invert = spec.invert
            step.expression = len(into.expressions)
            into.expressions.append(Regex(_text_of(spec.pattern), tables))
            into.steps.append(step^)
        else:
            var step = PreStep(T_SPLIT_STRING)
            step.behavior = _behavior_of(spec.behavior)
            step.invert = spec.invert
            step.pattern = to_code_points(spec.pattern)
            into.steps.append(step^)
    else:
        raise Error("unsupported pre-tokenizer " + _text_of(spec.kind))


def build_decoder(specs: Specs, at: Int, mut into: Decoder) raises:
    if at < 0:
        return
    var spec = specs.at(at)
    if len(spec.kind) == 0:
        return
    if spec.is_a("Sequence"):
        for i in range(len(spec.children)):
            build_decoder(specs, spec.children[i], into)
        return

    if spec.is_a("ByteLevel"):
        into.steps.append(DecodeStep(D_BYTE_LEVEL))
    elif spec.is_a("ByteFallback"):
        into.steps.append(DecodeStep(D_BYTE_FALLBACK))
    elif spec.is_a("Fuse"):
        into.steps.append(DecodeStep(D_FUSE))
    elif spec.is_a("Metaspace"):
        var step = DecodeStep(D_METASPACE)
        step.replacement = _first_point(spec.replacement)
        step.prepend_scheme = _prepend_of(spec.prepend_scheme)
        into.steps.append(step^)
    elif spec.is_a("WordPiece"):
        var step = DecodeStep(D_WORDPIECE)
        step.pattern = spec.prefix.copy()
        if len(step.pattern) == 0:
            step.pattern.append(UInt8(ord("#")))
            step.pattern.append(UInt8(ord("#")))
        step.cleanup = spec.cleanup
        into.steps.append(step^)
    elif spec.is_a("BPEDecoder"):
        var step = DecodeStep(D_BPE)
        step.pattern = spec.suffix.copy()
        if len(step.pattern) == 0:
            var fallback = StaticString("</w>").as_bytes()
            for i in range(len(fallback)):
                step.pattern.append(fallback[i])
        into.steps.append(step^)
    elif spec.is_a("Replace"):
        if spec.pattern_is_regex:
            raise Error("a decoder with a regex replacement is not supported")
        var step = DecodeStep(D_REPLACE)
        step.pattern = spec.pattern.copy()
        step.content = spec.content.copy()
        into.steps.append(step^)
    elif spec.is_a("Strip"):
        var step = DecodeStep(D_STRIP)
        step.content = spec.content.copy()
        step.strip_start = spec.start
        step.strip_stop = spec.stop
        into.steps.append(step^)
    else:
        raise Error("unsupported decoder " + _text_of(spec.kind))


def _special_ids(spec: Spec, name_at: Int, name_length: Int) -> List[Int]:
    for i in range(len(spec.special_name_at)):
        if spec.special_name_length[i] != name_length:
            continue
        var same = True
        for j in range(name_length):
            if (
                spec.special_names[spec.special_name_at[i] + j]
                != spec.template_names[name_at + j]
            ):
                same = False
                break
        if not same:
            continue
        var out = List[Int]()
        for j in range(spec.special_ids_length[i]):
            out.append(spec.special_ids[spec.special_ids_at[i] + j])
        return out^
    return List[Int]()


def build_post(specs: Specs, at: Int, mut into: PostProcessor) raises:
    if at < 0:
        return
    var spec = specs.at(at)
    if len(spec.kind) == 0:
        return
    if spec.is_a("Sequence"):
        for i in range(len(spec.children)):
            build_post(specs, spec.children[i], into)
        return
    if spec.is_a("ByteLevel"):
        return

    if spec.is_a("TemplateProcessing"):
        into.present = True
        for i in range(len(spec.template_kind)):
            var item = PostItem(spec.template_kind[i], spec.template_type[i])
            if item.kind == I_SPECIAL:
                item.ids = _special_ids(
                    spec,
                    spec.template_name_at[i],
                    spec.template_name_length[i],
                )
            if i < spec.single_count:
                into.single.append(item^)
            else:
                into.pair.append(item^)
        return

    if spec.is_a("BertProcessing") or spec.is_a("RobertaProcessing"):
        var roberta = spec.is_a("RobertaProcessing")
        into.present = True
        var cls = PostItem(I_SPECIAL, 0)
        cls.ids.append(spec.cls_id)
        var sep = PostItem(I_SPECIAL, 0)
        sep.ids.append(spec.sep_id)
        into.single.append(cls.copy())
        into.single.append(PostItem(I_SEQUENCE_A, 0))
        into.single.append(sep.copy())
        into.pair.append(cls.copy())
        into.pair.append(PostItem(I_SEQUENCE_A, 0))
        into.pair.append(sep.copy())
        if roberta:
            into.pair.append(sep.copy())
        into.pair.append(PostItem(I_SEQUENCE_B, 0 if roberta else 1))
        var closing = PostItem(I_SPECIAL, 0 if roberta else 1)
        closing.ids.append(spec.sep_id)
        into.pair.append(closing^)
        return

    raise Error("unsupported post-processor " + _text_of(spec.kind))


def _read_vocab_object(mut reader: Reader, mut model: Model) raises:
    """The big one: a token to id map with up to a quarter of a million keys.

    The key bytes have to be copied out of the reader before the number that
    follows is read, because a key that contained an escape lives in the
    reader's scratch space and the next event overwrites it.
    """
    if reader.next() != EV_OBJECT_BEGIN:
        raise Error("the vocabulary is not an object")
    var key = List[UInt8]()
    while True:
        var member = reader.next()
        if member == EV_OBJECT_END or member == EV_END:
            return
        if member == EV_ERROR:
            raise Error("the vocabulary is not valid json")
        if member != EV_KEY:
            continue
        key.clear()
        var span = reader.text()
        for i in range(len(span)):
            key.append(span[i])
        if reader.next() != EV_NUMBER:
            raise Error("a vocabulary entry is not a number")
        model.vocab.add(key, reader.number.as_int())


def _read_vocab_array(mut reader: Reader, mut model: Model) raises:
    """The Unigram form: a list of pairs, where the id is the position."""
    if reader.next() != EV_ARRAY_BEGIN:
        raise Error("the vocabulary is not an array")
    var key = List[UInt8]()
    var id = 0
    while True:
        var item = reader.next()
        if item == EV_ARRAY_END or item == EV_END:
            return
        if item != EV_ARRAY_BEGIN:
            raise Error("a unigram vocabulary entry is not a pair")
        key.clear()
        var score = 0.0
        var seen = 0
        while True:
            var part = reader.next()
            if part == EV_ARRAY_END or part == EV_END:
                break
            if part == EV_STRING and seen == 0:
                var span = reader.text()
                for i in range(len(span)):
                    key.append(span[i])
                seen = 1
            elif part == EV_NUMBER:
                score = reader.number.as_double()
        model.vocab.add(key, id)
        model.score.append(score)
        id += 1


def _add_merge(
    mut model: Model, left: List[UInt8], right: List[UInt8], rank: Int
) raises:
    """One merge, with both parts and the result resolved to ids now.

    A merge whose parts are not in the vocabulary is skipped rather than
    refused. Real files have a few, they are unreachable because nothing can
    ever produce the left part, and refusing the file over one would mean
    refusing models that Hugging Face itself loads.
    """
    var first = model.vocab.id_of(left)
    var second = model.vocab.id_of(right)
    if first == NO_ID or second == NO_ID:
        return
    var joined = left.copy()
    for i in range(len(right)):
        joined.append(right[i])
    var result = model.vocab.id_of(joined)
    if result == NO_ID:
        return
    model.merges.add(first, second, rank, result)


def _read_merges(mut reader: Reader, mut model: Model) raises:
    """Merges arrive either as `"a b"` or as `["a", "b"]`.

    Both spellings are in the wild, sometimes for the same architecture, and
    the string form splits on the first space because a byte level vocabulary
    never has a literal space in a token.
    """
    if reader.next() != EV_ARRAY_BEGIN:
        raise Error("the merge list is not an array")
    var rank = 0
    var left = List[UInt8]()
    var right = List[UInt8]()
    while True:
        var item = reader.next()
        if item == EV_ARRAY_END or item == EV_END:
            return
        left.clear()
        right.clear()
        if item == EV_STRING:
            var span = reader.text()
            var cut = -1
            for i in range(len(span)):
                if span[i] == UInt8(32):
                    cut = i
                    break
            if cut < 0:
                continue
            for i in range(cut):
                left.append(span[i])
            for i in range(cut + 1, len(span)):
                right.append(span[i])
        elif item == EV_ARRAY_BEGIN:
            var seen = 0
            while True:
                var part = reader.next()
                if part == EV_ARRAY_END or part == EV_END:
                    break
                if part != EV_STRING:
                    continue
                var span = reader.text()
                for i in range(len(span)):
                    if seen == 0:
                        left.append(span[i])
                    else:
                        right.append(span[i])
                seen += 1
        else:
            continue
        _add_merge(model, left, right, rank)
        rank += 1


def _over(base: Int, start: Int, end: Int) -> Reader:
    """A second reader over one member of the object the first one is in.

    The bytes are the same bytes, not a copy, so this costs a scratch buffer
    and nothing else.
    """
    var out = Reader(0, 1024)
    out.begin(
        Span[UInt8, MutAnyOrigin](
            unsafe_ptr=as_ptr(base + start), length=end - start
        )
    )
    return out^


def _guess_kind(
    base: Int, vocab_at: Int, vocab_end: Int, has_merges: Bool, capped: Bool
) raises -> Int:
    """Which model a `model` object with no `type` in it is.

    A Unigram vocabulary is an array and the other three are objects, so one
    character settles that. After it, `merges` means BPE, a word length limit
    means WordPiece, and what is left is a plain word level vocabulary.
    """
    if vocab_end > vocab_at:
        var p = as_ptr(base)
        for i in range(vocab_at, vocab_end):
            var b = p.unsafe_load(i)
            if (
                b == UInt8(32)
                or b == UInt8(9)
                or b == UInt8(10)
                or b == UInt8(13)
            ):
                continue
            if b == UInt8(ord("[")):
                return M_UNIGRAM
            break
    if has_merges:
        return M_BPE
    if capped:
        return M_WORDPIECE
    if vocab_end > vocab_at:
        return M_WORDLEVEL
    raise Error("the model has no type")


def read_model(mut reader: Reader) raises -> Model:
    """The `model` object, into a `Model` that is ready to tokenize.

    The two big members are read last whatever order the file writes them in.
    A vocabulary cannot be read before the type, because the type says whether
    it is an object or an array, and a merge cannot be resolved before the
    vocabulary, because a merge is a pair of ids. Files disagree about the
    order: most write the type first, and enough of them write `unk_token` or
    `dropout` or the vocabulary itself ahead of it that reading in file order
    means refusing files the reference implementation loads. So the spans of
    `vocab` and `merges` are noted and skipped on the way past, and read once
    the object has closed and the type is known.

    The type can also be missing, which the reference implementation answers by
    trying each model in turn until one of them accepts the other members. What
    the members actually say is unambiguous, so it is read off them directly:
    an array of pairs is a Unigram vocabulary, a `merges` list means BPE, and a
    word length limit means WordPiece.
    """
    if reader.next() != EV_OBJECT_BEGIN:
        raise Error("the model is not an object")
    var kind = -1
    var continuing_prefix = List[UInt8]()
    var end_suffix = List[UInt8]()
    var fuse_unk = False
    var byte_fallback = False
    var ignore_merges = False
    var max_input_chars = 100
    var saw_max_input_chars = False
    var unk_text = List[UInt8]()
    var unk_id = NO_ID
    var base = reader.base
    var vocab_at = 0
    var vocab_end = 0
    var merges_at = 0
    var merges_end = 0
    while True:
        var member = reader.next()
        if member == EV_OBJECT_END or member == EV_END:
            break
        if member == EV_ERROR:
            raise Error("the model is not valid json")
        if member != EV_KEY:
            continue
        if reader.key_is("type"):
            var text = _text_of(_read_string(reader))
            if text == "BPE":
                kind = M_BPE
            elif text == "WordPiece":
                kind = M_WORDPIECE
            elif text == "Unigram":
                kind = M_UNIGRAM
            elif text == "WordLevel":
                kind = M_WORDLEVEL
            else:
                raise Error("unsupported tokenizer model " + text)
        elif reader.key_is("vocab"):
            vocab_at = reader.at
            _ = reader.skip_next_value()
            vocab_end = reader.at
        elif reader.key_is("merges"):
            merges_at = reader.at
            _ = reader.skip_next_value()
            merges_end = reader.at
        elif reader.key_is("unk_token"):
            unk_text = _read_string(reader)
        elif reader.key_is("unk_id"):
            unk_id = _read_int(reader, NO_ID)
        elif reader.key_is("continuing_subword_prefix"):
            continuing_prefix = _read_string(reader)
        elif reader.key_is("end_of_word_suffix"):
            end_suffix = _read_string(reader)
        elif reader.key_is("fuse_unk"):
            fuse_unk = _read_bool(reader, False)
        elif reader.key_is("byte_fallback"):
            byte_fallback = _read_bool(reader, False)
        elif reader.key_is("ignore_merges"):
            ignore_merges = _read_bool(reader, False)
        elif reader.key_is("max_input_chars_per_word"):
            max_input_chars = _read_int(reader, 100)
            saw_max_input_chars = True
        else:
            _ = reader.skip_next_value()

    if kind < 0:
        kind = _guess_kind(
            base,
            vocab_at,
            vocab_end,
            merges_end > merges_at,
            saw_max_input_chars,
        )
    var expected = 65536 if kind == M_BPE else 32768
    var model = Model(kind, expected)
    model.continuing_prefix = continuing_prefix^
    model.end_suffix = end_suffix^
    model.fuse_unk = fuse_unk
    model.byte_fallback = byte_fallback
    model.ignore_merges = ignore_merges
    model.max_input_chars = max_input_chars
    if vocab_end > vocab_at:
        var sub = _over(base, vocab_at, vocab_end)
        if model.kind == M_UNIGRAM:
            _read_vocab_array(sub, model)
        else:
            _read_vocab_object(sub, model)
    if merges_end > merges_at:
        var sub = _over(base, merges_at, merges_end)
        _read_merges(sub, model)
    if len(unk_text) > 0:
        model.unk_id = model.vocab.id_of(unk_text)
    elif unk_id != NO_ID:
        model.unk_id = unk_id
    model.seal()
    return model^


def read_added(mut reader: Reader, mut added: AddedVocabulary) raises:
    """The `added_tokens` array."""
    if reader.next() != EV_ARRAY_BEGIN:
        return
    while True:
        var item = reader.next()
        if item == EV_ARRAY_END or item == EV_END:
            return
        if item != EV_OBJECT_BEGIN:
            _ = reader.skip_value()
            continue
        var token = AddedToken(NO_ID)
        var content = List[UInt8]()
        while True:
            var field = reader.next()
            if field == EV_OBJECT_END or field == EV_END:
                break
            if field != EV_KEY:
                continue
            if reader.key_is("id"):
                token.id = _read_int(reader, NO_ID)
            elif reader.key_is("content"):
                content = _read_string(reader)
            elif reader.key_is("single_word"):
                token.single_word = _read_bool(reader, False)
            elif reader.key_is("lstrip"):
                token.lstrip = _read_bool(reader, False)
            elif reader.key_is("rstrip"):
                token.rstrip = _read_bool(reader, False)
            elif reader.key_is("normalized"):
                token.normalized = _read_bool(reader, False)
            elif reader.key_is("special"):
                token.special = _read_bool(reader, False)
            else:
                _ = reader.skip_next_value()
        if token.id != NO_ID and len(content) > 0:
            added.add(token^, content)
