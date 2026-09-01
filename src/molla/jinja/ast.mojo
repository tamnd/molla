"""The tree, flat, in one list, referred to by index.

Same shape as the JSON document for the same reason: a growing `List` moves its
storage, so a node holding a pointer to another node would be holding a
dangling one after the next node was added. Indices do not move.

Every node carries the byte offset it was parsed at, because every error the
evaluator can raise is about a piece of template somebody wrote, and an error
that cannot say where is an error somebody has to bisect for.

Children are a `List[Int]` on the node rather than a linked list. A template
body is walked front to back many times, once per loop iteration, and an array
is the right shape for that where the JSON document's write once read once
linked list was the right shape for the other job.
"""

comptime NO_NODE = -1

comptime N_BLOCK = 0
"""A run of statements. A body, a branch, a macro's contents."""

comptime N_TEXT = 1
comptime N_OUTPUT = 2
comptime N_IF = 3
comptime N_FOR = 4
comptime N_SET = 5
comptime N_SET_BLOCK = 6
comptime N_MACRO = 7
comptime N_CALL_BLOCK = 8
comptime N_FILTER_BLOCK = 9

comptime N_NONE = 20
comptime N_BOOL = 21
comptime N_INT = 22
comptime N_FLOAT = 23
comptime N_STR = 24
comptime N_NAME = 25
comptime N_LIST = 26
comptime N_TUPLE = 27
comptime N_DICT = 28
comptime N_GETATTR = 29
comptime N_GETITEM = 30
comptime N_SLICE = 31
comptime N_CALL = 32
comptime N_FILTER = 33
comptime N_TEST = 34
comptime N_BINOP = 35
comptime N_UNARY = 36
comptime N_AND = 37
comptime N_OR = 38
comptime N_NOT = 39
comptime N_COND = 40

comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_DIV = 3
comptime OP_FLOORDIV = 4
comptime OP_MOD = 5
comptime OP_POW = 6
comptime OP_CONCAT = 7
comptime OP_EQ = 8
comptime OP_NE = 9
comptime OP_LT = 10
comptime OP_LE = 11
comptime OP_GT = 12
comptime OP_GE = 13
comptime OP_IN = 14
comptime OP_NOT_IN = 15
comptime OP_NEG = 16
comptime OP_POS = 17


struct Node(Copyable, Movable):
    """One node. The four integer slots mean different things per kind and
    every meaning is written down at the place that fills them."""

    var kind: Int
    var a: Int
    var b: Int
    var c: Int
    var d: Int
    var i: Int
    var f: Float64
    var flag: Bool
    var text: String
    var items: List[Int]
    var names: List[String]
    var at: Int

    def __init__(out self, kind: Int, at: Int):
        self.kind = kind
        self.a = NO_NODE
        self.b = NO_NODE
        self.c = NO_NODE
        self.d = NO_NODE
        self.i = 0
        self.f = 0.0
        self.flag = False
        self.text = String("")
        self.items = List[Int]()
        self.names = List[String]()
        self.at = at


struct Tree(Movable):
    """The nodes and the root block."""

    var nodes: List[Node]
    var root: Int

    def __init__(out self):
        self.nodes = List[Node]()
        self.root = NO_NODE

    def add(mut self, kind: Int, at: Int) -> Int:
        self.nodes.append(Node(kind, at))
        return len(self.nodes) - 1
