"""The post-processor stage.

After the model has produced ids there is one more step that wraps them in
whatever the model was trained to see around a sequence: a beginning of text
token, a classifier token, a separator between two sequences. Hugging Face has
four processors for this and three of them are special cases of the fourth, so
all four are compiled into one template here.

A template is a list of items. An item is either the sequence being encoded or
a fixed run of ids, and it carries the type id that goes with it, which is the
zero or one a BERT style model reads as "first sentence" or "second sentence".
"""

comptime I_SEQUENCE_A = 0
comptime I_SEQUENCE_B = 1
comptime I_SPECIAL = 2


struct PostItem(Copyable, Movable):
    var kind: Int
    var type_id: Int
    var ids: List[Int]

    def __init__(out self, kind: Int, type_id: Int):
        self.kind = kind
        self.type_id = type_id
        self.ids = List[Int]()


struct PostProcessor(Movable):
    """The template for one sequence and the template for two."""

    var single: List[PostItem]
    var pair: List[PostItem]
    var present: Bool

    def __init__(out self):
        self.single = List[PostItem]()
        self.pair = List[PostItem]()
        self.present = False

    def apply(
        self,
        first: List[Int],
        second: List[Int],
        has_second: Bool,
        reconcile: Bool,
        mut out: List[Int],
        mut types: List[Int],
    ):
        """Wrap the ids, writing the sequence and its type ids together.

        With no processor in the file the ids come through untouched, which is
        what a plain causal model wants and what every generative
        `tokenizer.json` without a processor means. The two sequences still get
        their own type ids, because that is the one thing joining them says
        even when nothing is written around them.
        """
        if not self.present:
            for i in range(len(first)):
                out.append(first[i])
                types.append(0)
            if has_second:
                for i in range(len(second)):
                    out.append(second[i])
                    types.append(1)
            return

        if has_second:
            self._write(self.pair, first, second, reconcile, out, types)
        else:
            self._write(self.single, first, second, reconcile, out, types)

    def _write(
        self,
        template: List[PostItem],
        first: List[Int],
        second: List[Int],
        reconcile: Bool,
        mut out: List[Int],
        mut types: List[Int],
    ):
        # A chat template writes the beginning of text token into the text it
        # renders, the added token matcher turns that back into an id, and then
        # this template writes another one in front of it. Two of them is not
        # what the model was trained on, and a double BOS does not fail, it
        # just makes the answers worse in a way nobody traces back to here. So
        # when the caller says the text came out of a template, and the
        # sequence already opens with the id this template is about to write,
        # the template's copy is dropped. Only the caller can know that, which
        # is why it is a flag rather than something decided here.
        var drop = -1
        if (
            reconcile
            and len(template) > 1
            and template[0].kind == I_SPECIAL
            and template[1].kind == I_SEQUENCE_A
            and len(template[0].ids) > 0
            and len(first) > 0
            and template[0].ids[len(template[0].ids) - 1] == first[0]
        ):
            drop = len(template[0].ids) - 1

        for i in range(len(template)):
            var kind = template[i].kind
            var type_id = template[i].type_id
            if kind == I_SEQUENCE_A:
                for j in range(len(first)):
                    out.append(first[j])
                    types.append(type_id)
            elif kind == I_SEQUENCE_B:
                for j in range(len(second)):
                    out.append(second[j])
                    types.append(type_id)
            else:
                for j in range(len(template[i].ids)):
                    if i == 0 and j == drop:
                        continue
                    out.append(template[i].ids[j])
                    types.append(type_id)
