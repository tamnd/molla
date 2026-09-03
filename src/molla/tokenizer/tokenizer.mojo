"""The whole pipeline, and the file it is loaded from.

Encoding runs five stages in a fixed order. Added tokens are found first, in
the text as it arrived, because they have to survive everything that follows.
What is left between them is normalized, searched for added tokens again in
case one of them is only spelled correctly after normalization, cut into pieces
by the pre-tokenizer, turned into ids by the model, and wrapped by the post
processor.

Decoding is the same list read upside down, and it is not symmetric: the
decoder is its own set of steps in the file, and a tokenizer whose decoder
disagrees with its pre-tokenizer will encode correctly and print nonsense.
"""

from molla.json.reader import (
    EV_END,
    EV_ERROR,
    EV_KEY,
    EV_OBJECT_BEGIN,
    EV_OBJECT_END,
    Reader,
)
from molla.sys.mem import AllocCounter
from molla.sys.mmap import Mapping
from molla.text.props import Unicode
from molla.text.regex import Scratch
from molla.text.utf8 import encode, to_code_points

from .added import AddedVocabulary
from .config import (
    build_decoder,
    build_normalizer,
    build_post,
    build_pretokenizer,
    read_added,
    read_model,
    read_spec,
)
from .decoder import Chunks, Decoder
from .model import M_BPE, Model, Work
from .normalizer import Normalizer
from .post import PostProcessor
from .pretok import PreTokenizer, Pieces
from .vocab import NO_ID


struct Session(Movable):
    """The scratch space one thread needs to encode.

    Held by the caller rather than by the tokenizer so that the tokenizer can
    be shared and read only, which is what lets one loaded file serve every
    request without a lock.

    It belongs to one tokenizer. The word cache inside it maps bytes to ids and
    those ids only mean anything in the vocabulary they came from, so a session
    handed to a second tokenizer will answer with the first one's ids and will
    not say so. One session per thread per loaded file.
    """

    var scratch: Scratch
    var work: Work

    def __init__(out self):
        self.scratch = Scratch()
        self.work = Work()


struct Tokenizer(Movable):
    """A loaded `tokenizer.json`."""

    var tables: Unicode
    var normalizer: Normalizer
    var pretokenizer: PreTokenizer
    var model: Model
    var post: PostProcessor
    var decoder: Decoder
    var added: AddedVocabulary

    def __init__(out self, path: StringSpan, counter: Int) raises:
        self.tables = Unicode()
        self.normalizer = Normalizer()
        self.pretokenizer = PreTokenizer()
        self.model = Model(M_BPE, 16)
        self.post = PostProcessor()
        self.decoder = Decoder()
        self.added = AddedVocabulary()

        var mapping = Mapping(path)
        var reader = Reader(counter, 1 << 16)
        var body = Span[UInt8, MutAnyOrigin](
            unsafe_ptr=mapping.base(), length=mapping.length
        )
        reader.begin(body)
        try:
            self._read(reader)
        except e:
            mapping.close()
            raise e
        mapping.close()

    def _read(mut self, mut reader: Reader) raises:
        if reader.next() != EV_OBJECT_BEGIN:
            raise Error("tokenizer.json is not an object")
        while True:
            var member = reader.next()
            if member == EV_OBJECT_END or member == EV_END:
                break
            if member == EV_ERROR:
                raise Error("tokenizer.json is not valid json")
            if member != EV_KEY:
                continue
            if reader.key_is("model"):
                self.model = read_model(reader)
            elif reader.key_is("added_tokens"):
                read_added(reader, self.added)
            elif reader.key_is("normalizer"):
                var specs = read_spec(reader)
                build_normalizer(
                    self.tables, specs, specs.root, self.normalizer
                )
            elif reader.key_is("pre_tokenizer"):
                var specs = read_spec(reader)
                build_pretokenizer(
                    self.tables, specs, specs.root, self.pretokenizer
                )
            elif reader.key_is("post_processor"):
                var specs = read_spec(reader)
                build_post(specs, specs.root, self.post)
            elif reader.key_is("decoder"):
                var specs = read_spec(reader)
                build_decoder(specs, specs.root, self.decoder)
            else:
                _ = reader.skip_next_value()

    def encode(
        self,
        text: StringSpan,
        add_special: Bool,
        mut session: Session,
        mut out: List[Int],
    ) raises:
        """Text to ids, exactly what the reference implementation does."""
        var types = List[Int]()
        self.encode_pair(
            text, "", False, add_special, False, session, out, types
        )

    def encode_rendered(
        self, text: StringSpan, mut session: Session, mut out: List[Int]
    ) raises:
        """The same, for text a chat template produced.

        The one difference is that the template has already written the
        beginning of text token into the text, so the post-processor does not
        write a second one. See `PostProcessor._write` for why that matters.
        """
        var types = List[Int]()
        self.encode_pair(text, "", False, True, True, session, out, types)

    def encode_pair(
        self,
        first: StringSpan,
        second: StringSpan,
        has_second: Bool,
        add_special: Bool,
        reconcile_bos: Bool,
        mut session: Session,
        mut out: List[Int],
        mut types: List[Int],
    ) raises:
        var left = List[Int]()
        self._encode_one(first, session, left)
        var right = List[Int]()
        if has_second:
            self._encode_one(second, session, right)
        if not add_special:
            for i in range(len(left)):
                out.append(left[i])
                types.append(0)
            for i in range(len(right)):
                out.append(right[i])
                types.append(1)
            return
        self.post.apply(left, right, has_second, reconcile_bos, out, types)

    def _encode_one(
        self, text: StringSpan, mut session: Session, mut out: List[Int]
    ) raises:
        var points = to_code_points(text.as_bytes())
        var start = List[Int]()
        var end = List[Int]()
        var token = List[Int]()
        self.added.split(False, points, start, end, token)
        for p in range(len(start)):
            if token[p] >= 0:
                out.append(self.added.id_at(token[p]))
                continue
            var window = List[Int]()
            for i in range(start[p], end[p]):
                window.append(points[i])
            self._encode_raw(window, session, out)

    def _encode_raw(
        self, points: List[Int], mut session: Session, mut out: List[Int]
    ) raises:
        """Normalize, look for added tokens again, then the model."""
        var normalized = self.normalizer.apply(
            self.tables, session.scratch, points
        )
        var start = List[Int]()
        var end = List[Int]()
        var token = List[Int]()
        self.added.split(True, normalized, start, end, token)
        for p in range(len(start)):
            if token[p] >= 0:
                out.append(self.added.id_at(token[p]))
                continue
            var window = List[Int]()
            for i in range(start[p], end[p]):
                window.append(normalized[i])
            self._encode_pieces(window^, session, out)

    def _encode_pieces(
        self, var points: List[Int], mut session: Session, mut out: List[Int]
    ) raises:
        var pieces = self.pretokenizer.apply(
            self.tables, session.scratch, Pieces(points^)
        )
        var bytes = List[UInt8]()
        for p in range(len(pieces.start)):
            bytes.clear()
            _encode_points(pieces.points, pieces.start[p], pieces.end[p], bytes)
            self.model.tokenize(bytes, out, session.work)

    def token_bytes(self, id: Int, mut into: List[UInt8]) -> Bool:
        """The text of one id, appended. False when nothing has that id."""
        var known = self.model.vocab.token(id)
        if len(known) > 0:
            for i in range(len(known)):
                into.append(known[i])
            return True
        return self.added.append_text(id, into)

    def decode(self, ids: List[Int], skip_special: Bool) raises -> String:
        var bytes = self.decode_bytes(ids, skip_special)
        return String(StringSpan(unsafe_from_utf8=bytes))

    def decode_bytes(
        self, ids: List[Int], skip_special: Bool
    ) raises -> List[UInt8]:
        var chunks = Chunks()
        var any = False
        var one = List[UInt8]()
        for i in range(len(ids)):
            if skip_special and self.added.is_special(ids[i]):
                continue
            one.clear()
            if not self.token_bytes(ids[i], one):
                continue
            chunks.open()
            chunks.put(one)
            chunks.close()
            any = True
        if not any:
            return List[UInt8]()
        if self.decoder.is_empty():
            return _join_with_spaces(chunks)
        var done = self.decoder.apply(chunks^)
        return done.joined()


def _join_with_spaces(chunks: Chunks) -> List[UInt8]:
    """What a file with no decoder means, which is one space between tokens."""
    var out = List[UInt8]()
    for c in range(len(chunks.start)):
        if c > 0:
            out.append(UInt8(32))
        for i in range(chunks.start[c], chunks.end[c]):
            out.append(chunks.bytes[i])
    return out^


def _encode_points(
    points: List[Int], start: Int, end: Int, mut out: List[UInt8]
) raises:
    for i in range(start, end):
        encode(points[i], out)


struct DecodeStream(Movable):
    """Decoding one id at a time without ever printing half a character.

    A token is bytes, not text, and one character can be spread over three of
    them, so a decoder that prints each token as it arrives prints replacement
    characters in the middle of any word it did not expect. This keeps the ids
    it has been given, decodes all of them each step, and hands back only the
    part of the answer that has not been handed back already and that ends on a
    character boundary.

    The list of ids is trimmed when it can be done without changing the answer,
    which is checked rather than assumed, because a metaspace decoder treats
    its first token differently from the rest and a trim in the wrong place
    would silently drop a space.
    """

    var ids: List[Int]
    var emitted: Int
    var skip_special: Bool

    def __init__(out self, skip_special: Bool):
        self.ids = List[Int]()
        self.emitted = 0
        self.skip_special = skip_special

    def step(mut self, tokenizer: Tokenizer, id: Int) raises -> String:
        self.ids.append(id)
        var text = tokenizer.decode_bytes(self.ids, self.skip_special)
        var cut = _last_boundary(text)
        if cut <= self.emitted:
            return String("")
        var out = List[UInt8]()
        for i in range(self.emitted, cut):
            out.append(text[i])
        self.emitted = cut
        self._trim(tokenizer, text)
        return String(StringSpan(unsafe_from_utf8=out))

    def _trim(mut self, tokenizer: Tokenizer, whole: List[UInt8]) raises:
        """Drop the ids at the front whose text is already out and settled.

        The check is the point. The head and the tail are decoded on their own
        and only accepted when they join back into exactly what the whole list
        decoded to, so a decoder that treats position specially just declines
        the trim and keeps working.
        """
        if len(self.ids) < 64:
            return
        var cut = len(self.ids) - 16
        var head = List[Int]()
        for i in range(cut):
            head.append(self.ids[i])
        var tail = List[Int]()
        for i in range(cut, len(self.ids)):
            tail.append(self.ids[i])
        var head_text = tokenizer.decode_bytes(head, self.skip_special)
        var tail_text = tokenizer.decode_bytes(tail, self.skip_special)
        if len(head_text) + len(tail_text) != len(whole):
            return
        if len(head_text) > self.emitted:
            return
        for i in range(len(head_text)):
            if head_text[i] != whole[i]:
                return
        for i in range(len(tail_text)):
            if tail_text[i] != whole[len(head_text) + i]:
                return
        self.ids = tail^
        self.emitted -= len(head_text)


def _last_boundary(text: List[UInt8]) -> Int:
    """The end of the last complete character in these bytes.

    Walk back from the end over continuation bytes until the character they
    belong to is found. If all of it is here the answer is the end of the text,
    and if it is not the answer is where that character starts, so the caller
    holds it back and asks again when the rest of it has arrived.
    """
    var at = len(text)
    var back = 0
    while at > 0 and back < 4:
        var b = Int(text[at - 1])
        if b < 0x80:
            return at
        if b >= 0xC0:
            var wanted = 2
            if b >= 0xF0:
                wanted = 4
            elif b >= 0xE0:
                wanted = 3
            if back + 1 == wanted:
                return at - 1 + wanted
            return at - 1
        at -= 1
        back += 1
    return len(text)


def run_tokenize(tokenizer_path: String, prompt: String, ids: Bool) raises:
    """Print how many tokens a prompt is, and the ids when asked.

    The one thing a benchmark harness needs before it starts: three engines
    have to be given the same amount of work, and a prompt built to a target
    length is only that length once somebody has counted it. There is no model
    file in this because encoding does not need one, so counting a 512 token
    prompt costs the tokenizer file rather than eight gigabytes of weights.
    """
    var counter = AllocCounter()
    var tokenizer = Tokenizer(tokenizer_path, counter.raw())
    var session = Session()
    var out = List[Int]()
    tokenizer.encode(prompt, True, session, out)

    if not ids:
        print(len(out))
        return
    var line = String("")
    for i in range(len(out)):
        if i > 0:
            line += " "
        line += String(out[i])
    print(len(out))
    print(line)
