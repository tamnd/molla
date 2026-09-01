"""Tokens to a tree, and the one place a template is allowed to be rejected.

The subset is decided here and nowhere else. Every construct Jinja has that we
do not implement is a branch in this file that raises with the construct named,
the line, the column and the line of template quoted, and there is no path that
parses something into a shape the evaluator will later have to guess about. That
is the property the design notes ask for: misrendering is impossible, failing is
loud, and failing happens when the model is loaded rather than in the middle of
somebody's conversation.

## Precedence, copied rather than invented

The expression grammar is Jinja's, in Jinja's order, including the two places
that surprise people. Filters and tests bind tighter than arithmetic, so
`1 + x|length` adds one to the length. And a unary minus binds tighter than a
filter applied after it, so `-x|abs` takes the absolute value of the negation.
Both are what the reference does, and a template that relies on either would
otherwise render differently here with nothing to show for it.

## What is not here

`include`, `extends`, `import` and `from` need a template loader, and a chat
template is one self contained string, so there is nothing for a loader to load.
`do` and `autoescape` are Jinja extensions that `transformers` does not enable,
so a template using them does not work with the reference either. `break` and
`continue` are an extension that `transformers` does enable, so they are here.
"""

from molla.jinja.ast import (
    N_AND,
    N_BINOP,
    N_BLOCK,
    N_BOOL,
    N_CALL,
    N_CALL_BLOCK,
    N_COND,
    N_DICT,
    N_FILTER,
    N_FILTER_BLOCK,
    N_FLOAT,
    N_FOR,
    N_GETATTR,
    N_GETITEM,
    N_IF,
    N_INT,
    N_LIST,
    N_MACRO,
    N_NAME,
    N_NONE,
    N_NOT,
    N_OR,
    N_OUTPUT,
    N_SET,
    N_SET_BLOCK,
    N_SLICE,
    N_STR,
    N_TEST,
    N_TEXT,
    N_TUPLE,
    N_UNARY,
    NO_NODE,
    OP_ADD,
    OP_CONCAT,
    OP_DIV,
    OP_EQ,
    OP_FLOORDIV,
    OP_GE,
    OP_GT,
    OP_IN,
    OP_LE,
    OP_LT,
    OP_MOD,
    OP_MUL,
    OP_NE,
    OP_NEG,
    OP_NOT_IN,
    OP_POS,
    OP_POW,
    OP_SUB,
    Tree,
)
from molla.jinja.diag import fail
from molla.jinja.lexer import (
    T_BLOCK_BEGIN,
    T_BLOCK_END,
    T_EOF,
    T_FLOAT,
    T_INT,
    T_NAME,
    T_OP,
    T_STRING,
    T_TEXT,
    T_VAR_BEGIN,
    T_VAR_END,
    Token,
    tokenize,
)

comptime BREAK_LOOP = -2
comptime CONTINUE_LOOP = -3
"""Sentinel statement kinds, stored as the `a` slot of a text node with no text.

`break` and `continue` carry nothing, so rather than two node kinds that hold
nothing they are one kind with a marker. See `_statement`.
"""

comptime N_LOOP_CONTROL = 10


struct Parser(Movable):
    var src: String
    var toks: List[Token]
    var pos: Int
    var tree: Tree
    var depth: Int

    def __init__(out self, source: String) raises:
        self.src = source
        self.toks = tokenize(source)
        self.pos = 0
        self.tree = Tree()
        self.depth = 0

    def _fail(self, at: Int, message: String) raises:
        fail(self.src.as_bytes(), at, message)

    def _kind(self) -> Int:
        return self.toks[self.pos].kind

    def _at(self) -> Int:
        return self.toks[self.pos].at

    def _is_op(self, text: StringSpan) -> Bool:
        return (
            self.toks[self.pos].kind == T_OP
            and self.toks[self.pos].text == text
        )

    def _is_name(self, text: StringSpan) -> Bool:
        return (
            self.toks[self.pos].kind == T_NAME
            and self.toks[self.pos].text == text
        )

    def _skip_op(mut self, text: StringSpan) -> Bool:
        if self._is_op(text):
            self.pos += 1
            return True
        return False

    def _skip_name(mut self, text: StringSpan) -> Bool:
        if self._is_name(text):
            self.pos += 1
            return True
        return False

    def _want_op(mut self, text: StringSpan) raises:
        if not self._skip_op(text):
            self._fail(self._at(), "expected `" + String(text) + "`")

    def _want_name(mut self, text: StringSpan) raises:
        if not self._skip_name(text):
            self._fail(self._at(), "expected `" + String(text) + "`")

    def _want_identifier(mut self) raises -> String:
        if self._kind() != T_NAME:
            self._fail(self._at(), "expected a name")
        var name = self.toks[self.pos].text
        if _is_reserved(name):
            self._fail(self._at(), "`" + name + "` cannot be used as a name")
        self.pos += 1
        return name

    # Statements.

    def parse(mut self) raises:
        var body = self._block(List[String]())
        if self._kind() != T_EOF:
            self._fail(
                self._at(),
                "unexpected `" + _describe(self.toks[self.pos]) + "`",
            )
        self.tree.root = body

    def into_tree(deinit self) -> Tree:
        """Hand the tree over and leave the parser behind.

        Consuming rather than borrowing because the tree outlives the token
        list and the source it was cut from, and copying a few thousand nodes
        to say so would be a waste.
        """
        return self.tree^

    def _at_end_tag(self, stops: List[String]) -> Bool:
        if self._kind() != T_BLOCK_BEGIN:
            return False
        var next = self.toks[self.pos + 1]
        if next.kind != T_NAME:
            return False
        for i in range(len(stops)):
            if next.text == stops[i]:
                return True
        return False

    def _block(mut self, stops: List[String]) raises -> Int:
        var node = self.tree.add(N_BLOCK, self._at())
        while True:
            if self._kind() == T_EOF:
                break
            if self._at_end_tag(stops):
                break
            var child = self._statement()
            if child != NO_NODE:
                self.tree.nodes[node].items.append(child)
        return node

    def _statement(mut self) raises -> Int:
        var kind = self._kind()
        if kind == T_TEXT:
            var node = self.tree.add(N_TEXT, self._at())
            self.tree.nodes[node].text = self.toks[self.pos].text
            self.pos += 1
            return node
        if kind == T_VAR_BEGIN:
            var at = self._at()
            self.pos += 1
            var expr = self._expression()
            if self._kind() != T_VAR_END:
                self._fail(self._at(), "expected `}}`")
            self.pos += 1
            var node = self.tree.add(N_OUTPUT, at)
            self.tree.nodes[node].a = expr
            return node
        if kind != T_BLOCK_BEGIN:
            self._fail(
                self._at(),
                "unexpected `" + _describe(self.toks[self.pos]) + "`",
            )
        return self._tag()

    def _end_tag(mut self) raises:
        if self._kind() != T_BLOCK_END:
            self._fail(self._at(), "expected `%}`")
        self.pos += 1

    def _tag(mut self) raises -> Int:
        var at = self._at()
        self.pos += 1
        if self._kind() != T_NAME:
            self._fail(self._at(), "expected a statement name after `{%`")
        var name = self.toks[self.pos].text
        var name_at = self._at()

        if _is_banned(name):
            self._fail(name_at, _ban_reason(name))

        self.pos += 1
        if name == "if":
            return self._if(at)
        if name == "for":
            return self._for(at)
        if name == "set":
            return self._set(at)
        if name == "macro":
            return self._macro(at)
        if name == "call":
            return self._call_block(at)
        if name == "filter":
            return self._filter_block(at)
        if name == "with":
            return self._with(at)
        if name == "block":
            return self._named_block(at)
        if name == "break" or name == "continue":
            self._end_tag()
            var node = self.tree.add(N_LOOP_CONTROL, at)
            self.tree.nodes[node].a = (
                BREAK_LOOP if name == "break" else CONTINUE_LOOP
            )
            return node
        self._fail(name_at, "unknown statement `" + name + "`")
        return NO_NODE

    def _if(mut self, at: Int) raises -> Int:
        var node = self.tree.add(N_IF, at)
        var got_a = self._expression()
        self.tree.nodes[node].a = got_a
        self._end_tag()
        var stops = List[String]()
        stops.append(String("elif"))
        stops.append(String("else"))
        stops.append(String("endif"))
        var got_b = self._block(stops)
        self.tree.nodes[node].b = got_b

        if self._kind() == T_EOF:
            self._fail(at, "`if` was never closed with `endif`")
        var next = self.toks[self.pos + 1].text
        var next_at = self.toks[self.pos + 1].at
        self.pos += 2
        if next == "elif":
            var got_c = self._if(next_at)
            self.tree.nodes[node].c = got_c
            return node
        if next == "else":
            self._end_tag()
            var only_end = List[String]()
            only_end.append(String("endif"))
            var got_c = self._block(only_end)
            self.tree.nodes[node].c = got_c
            if self._kind() == T_EOF:
                self._fail(at, "`if` was never closed with `endif`")
            self.pos += 2
        self._end_tag()
        return node

    def _targets(mut self) raises -> List[String]:
        """One name, or several separated by commas.

        Jinja allows nested tuple targets. Nothing in the corpus writes one and
        supporting them would mean the evaluator has to unpack a shape the
        parser guessed at, so a `(` here is a named error rather than a guess.
        """
        var names = List[String]()
        if self._is_op("("):
            self._fail(
                self._at(),
                (
                    "a nested tuple target is not supported, name the parts"
                    " directly"
                ),
            )
        names.append(self._want_identifier())
        while self._skip_op(","):
            if self._is_name("in"):
                break
            names.append(self._want_identifier())
        return names^

    def _for(mut self, at: Int) raises -> Int:
        var node = self.tree.add(N_FOR, at)
        var got_names = self._targets()
        self.tree.nodes[node].names = got_names^
        self._want_name("in")
        # `parse_tuple` with no condition, because `for x in a if b` reads the
        # `if` as the loop filter rather than as an inline conditional. That is
        # Jinja's rule and it is the one templates are written against.
        var got_a = self._expression(no_cond=True)
        self.tree.nodes[node].a = got_a
        if self._skip_name("if"):
            var got_d = self._expression(no_cond=True)
            self.tree.nodes[node].d = got_d
        if self._is_name("recursive"):
            self._fail(
                self._at(),
                (
                    "`recursive` loops are not supported, a chat template has"
                    " no tree to walk"
                ),
            )
        self._end_tag()

        var stops = List[String]()
        stops.append(String("else"))
        stops.append(String("endfor"))
        var got_b = self._block(stops)
        self.tree.nodes[node].b = got_b
        if self._kind() == T_EOF:
            self._fail(at, "`for` was never closed with `endfor`")
        var next = self.toks[self.pos + 1].text
        self.pos += 2
        if next == "else":
            self._end_tag()
            var only_end = List[String]()
            only_end.append(String("endfor"))
            var got_c = self._block(only_end)
            self.tree.nodes[node].c = got_c
            if self._kind() == T_EOF:
                self._fail(at, "`for` was never closed with `endfor`")
            self.pos += 2
        self._end_tag()
        return node

    def _set(mut self, at: Int) raises -> Int:
        var first = self._want_identifier()
        # `{% set ns.field = value %}` is how a template writes to something the
        # loop body can see from outside the loop, and it is the only assignment
        # to an attribute Jinja allows.
        if self._skip_op("."):
            var field = self._want_identifier()
            self._want_op("=")
            var node = self.tree.add(N_SET, at)
            node_names(self.tree, node, first)
            self.tree.nodes[node].text = field
            self.tree.nodes[node].flag = True
            var got_a = self._expression()
            self.tree.nodes[node].a = got_a
            self._end_tag()
            return node

        var names = List[String]()
        names.append(first)
        while self._skip_op(","):
            names.append(self._want_identifier())

        if self._skip_op("="):
            var node = self.tree.add(N_SET, at)
            self.tree.nodes[node].names = names^
            var got_a = self._expression()
            self.tree.nodes[node].a = got_a
            self._end_tag()
            return node

        # The block form. `{% set x %}...{% endset %}` captures what the body
        # renders, optionally through a filter chain.
        if len(names) != 1:
            self._fail(at, "a block `set` assigns one name")
        var node = self.tree.add(N_SET_BLOCK, at)
        self.tree.nodes[node].names = names^
        if self._skip_op("|"):
            var got_a = self._filter_chain(NO_NODE)
            self.tree.nodes[node].a = got_a
        self._end_tag()
        var stops = List[String]()
        stops.append(String("endset"))
        var got_b = self._block(stops)
        self.tree.nodes[node].b = got_b
        if self._kind() == T_EOF:
            self._fail(at, "`set` was never closed with `endset`")
        self.pos += 2
        self._end_tag()
        return node

    def _signature(mut self, mut node_index: Int) raises:
        """`(a, b='x')`, names in `names` and defaults in `items`.

        A parameter with no default gets `NO_NODE`, so the two lists are the
        same length and the evaluator does not have to count backwards.
        """
        self._want_op("(")
        while not self._is_op(")"):
            if len(self.tree.nodes[node_index].names) > 0:
                self._want_op(",")
                if self._is_op(")"):
                    break
            if self._is_op("*"):
                self._fail(
                    self._at(),
                    "a variable argument list is not supported in a macro",
                )
            var name = self._want_identifier()
            var default = NO_NODE
            if self._skip_op("="):
                default = self._expression()
            self.tree.nodes[node_index].names.append(name)
            self.tree.nodes[node_index].items.append(default)
        self._want_op(")")

    def _macro(mut self, at: Int) raises -> Int:
        var node = self.tree.add(N_MACRO, at)
        var got_text = self._want_identifier()
        self.tree.nodes[node].text = got_text^
        self._signature(node)
        self._end_tag()
        var stops = List[String]()
        stops.append(String("endmacro"))
        var got_b = self._block(stops)
        self.tree.nodes[node].b = got_b
        if self._kind() == T_EOF:
            self._fail(at, "`macro` was never closed with `endmacro`")
        self.pos += 2
        self._end_tag()
        return node

    def _call_block(mut self, at: Int) raises -> Int:
        var node = self.tree.add(N_CALL_BLOCK, at)
        if self._is_op("("):
            # `{% call(x) macro() %}` gives the block its own parameters, which
            # is what the macro passes to `caller()`.
            self._signature(node)
        var callee = self._primary()
        callee = self._postfix(callee)
        if self.tree.nodes[callee].kind != N_CALL:
            self._fail(at, "`call` takes a macro call, like `call thing()`")
        self.tree.nodes[node].a = callee
        self._end_tag()
        var stops = List[String]()
        stops.append(String("endcall"))
        var got_b = self._block(stops)
        self.tree.nodes[node].b = got_b
        if self._kind() == T_EOF:
            self._fail(at, "`call` was never closed with `endcall`")
        self.pos += 2
        self._end_tag()
        return node

    def _filter_block(mut self, at: Int) raises -> Int:
        var node = self.tree.add(N_FILTER_BLOCK, at)
        var got_a = self._filter_chain(NO_NODE)
        self.tree.nodes[node].a = got_a
        self._end_tag()
        var stops = List[String]()
        stops.append(String("endfilter"))
        var got_b = self._block(stops)
        self.tree.nodes[node].b = got_b
        if self._kind() == T_EOF:
            self._fail(at, "`filter` was never closed with `endfilter`")
        self.pos += 2
        self._end_tag()
        return node

    def _with(mut self, at: Int) raises -> Int:
        """`{% with a = 1, b = 2 %}`, which is a scope and some assignments.

        Written as a block whose first statements are the assignments, because
        that is exactly what it means and it saves the evaluator a node kind.
        The scope comes from the block itself.
        """
        var node = self.tree.add(N_BLOCK, at)
        self.tree.nodes[node].flag = True
        while self._kind() != T_BLOCK_END:
            if len(self.tree.nodes[node].items) > 0:
                _ = self._skip_op(",")
            var set_at = self._at()
            var name = self._want_identifier()
            self._want_op("=")
            var assign = self.tree.add(N_SET, set_at)
            self.tree.nodes[assign].names.append(name)
            var got_a = self._expression()
            self.tree.nodes[assign].a = got_a
            self.tree.nodes[node].items.append(assign)
        self._end_tag()
        var stops = List[String]()
        stops.append(String("endwith"))
        var body = self._block(stops)
        for i in range(len(self.tree.nodes[body].items)):
            var child = self.tree.nodes[body].items[i]
            self.tree.nodes[node].items.append(child)
        if self._kind() == T_EOF:
            self._fail(at, "`with` was never closed with `endwith`")
        self.pos += 2
        self._end_tag()
        return node

    def _named_block(mut self, at: Int) raises -> Int:
        """`{% block name %}`, which without inheritance is just its body.

        There is no loader and therefore no `extends`, so nothing can ever
        override one. Rendering the body is what Jinja does for a block in a
        template nobody extended, and refusing it would reject templates that
        are perfectly well defined.
        """
        _ = self._want_identifier()
        _ = self._skip_name("scoped")
        _ = self._skip_name("required")
        self._end_tag()
        var stops = List[String]()
        stops.append(String("endblock"))
        var body = self._block(stops)
        if self._kind() == T_EOF:
            self._fail(at, "`block` was never closed with `endblock`")
        self.pos += 2
        if self._kind() == T_NAME:
            self.pos += 1
        self._end_tag()
        return body

    # Expressions.

    def _expression(mut self, no_cond: Bool = False) raises -> Int:
        var node = self._or()
        if no_cond or not self._is_name("if"):
            return node
        var at = self._at()
        self.pos += 1
        var cond = self.tree.add(N_COND, at)
        self.tree.nodes[cond].a = node
        var got_b = self._or()
        self.tree.nodes[cond].b = got_b
        if self._skip_name("else"):
            var got_c = self._expression()
            self.tree.nodes[cond].c = got_c
        return cond

    def _or(mut self) raises -> Int:
        var left = self._and()
        while self._is_name("or"):
            var at = self._at()
            self.pos += 1
            var node = self.tree.add(N_OR, at)
            self.tree.nodes[node].a = left
            var got_b = self._and()
            self.tree.nodes[node].b = got_b
            left = node
        return left

    def _and(mut self) raises -> Int:
        var left = self._not()
        while self._is_name("and"):
            var at = self._at()
            self.pos += 1
            var node = self.tree.add(N_AND, at)
            self.tree.nodes[node].a = left
            var got_b = self._not()
            self.tree.nodes[node].b = got_b
            left = node
        return left

    def _not(mut self) raises -> Int:
        if self._is_name("not"):
            var at = self._at()
            self.pos += 1
            var node = self.tree.add(N_NOT, at)
            var got_a = self._not()
            self.tree.nodes[node].a = got_a
            return node
        return self._compare()

    def _compare(mut self) raises -> Int:
        var left = self._math1()
        while True:
            var at = self._at()
            var op = -1
            if self._kind() == T_OP:
                var text = self.toks[self.pos].text
                if text == "==":
                    op = OP_EQ
                elif text == "!=":
                    op = OP_NE
                elif text == "<":
                    op = OP_LT
                elif text == "<=":
                    op = OP_LE
                elif text == ">":
                    op = OP_GT
                elif text == ">=":
                    op = OP_GE
                if op >= 0:
                    self.pos += 1
            elif self._is_name("in"):
                op = OP_IN
                self.pos += 1
            elif (
                self._is_name("not")
                and self.toks[self.pos + 1].kind == T_NAME
                and self.toks[self.pos + 1].text == "in"
            ):
                op = OP_NOT_IN
                self.pos += 2
            if op < 0:
                return left
            var node = self.tree.add(N_BINOP, at)
            self.tree.nodes[node].i = op
            self.tree.nodes[node].a = left
            var got_b = self._math1()
            self.tree.nodes[node].b = got_b
            left = node

    def _math1(mut self) raises -> Int:
        var left = self._concat()
        while self._is_op("+") or self._is_op("-"):
            var at = self._at()
            var op = OP_ADD if self.toks[self.pos].text == "+" else OP_SUB
            self.pos += 1
            var node = self.tree.add(N_BINOP, at)
            self.tree.nodes[node].i = op
            self.tree.nodes[node].a = left
            var got_b = self._concat()
            self.tree.nodes[node].b = got_b
            left = node
        return left

    def _concat(mut self) raises -> Int:
        var left = self._math2()
        while self._is_op("~"):
            var at = self._at()
            self.pos += 1
            var node = self.tree.add(N_BINOP, at)
            self.tree.nodes[node].i = OP_CONCAT
            self.tree.nodes[node].a = left
            var got_b = self._math2()
            self.tree.nodes[node].b = got_b
            left = node
        return left

    def _math2(mut self) raises -> Int:
        var left = self._power()
        while True:
            var at = self._at()
            var op = -1
            if self._is_op("*"):
                op = OP_MUL
            elif self._is_op("//"):
                op = OP_FLOORDIV
            elif self._is_op("/"):
                op = OP_DIV
            elif self._is_op("%"):
                op = OP_MOD
            if op < 0:
                return left
            self.pos += 1
            var node = self.tree.add(N_BINOP, at)
            self.tree.nodes[node].i = op
            self.tree.nodes[node].a = left
            var got_b = self._power()
            self.tree.nodes[node].b = got_b
            left = node

    def _power(mut self) raises -> Int:
        # Right associative, so `2 ** 3 ** 2` is 512 and not 64. Recursing on
        # the right rather than looping is what makes it so, and this is the
        # one operator in the language that goes that way.
        var left = self._unary(True)
        if not self._is_op("**"):
            return left
        var at = self._at()
        self.pos += 1
        var node = self.tree.add(N_BINOP, at)
        self.tree.nodes[node].i = OP_POW
        self.tree.nodes[node].a = left
        var got_b = self._power()
        self.tree.nodes[node].b = got_b
        return node

    def _unary(mut self, with_filter: Bool) raises -> Int:
        var node: Int
        if self._is_op("-") or self._is_op("+"):
            var at = self._at()
            var op = OP_NEG if self.toks[self.pos].text == "-" else OP_POS
            self.pos += 1
            node = self.tree.add(N_UNARY, at)
            self.tree.nodes[node].i = op
            var got_a = self._unary(False)
            self.tree.nodes[node].a = got_a
        else:
            node = self._primary()
        node = self._postfix(node)
        if with_filter:
            node = self._trailer(node)
        return node

    def _trailer(mut self, value: Int) raises -> Int:
        """Filters, tests and calls, which all bind tighter than arithmetic."""
        var node = value
        while True:
            if self._is_op("|"):
                self.pos += 1
                node = self._filter_chain(node)
            elif self._is_name("is"):
                node = self._test(node)
            elif self._is_op("("):
                node = self._postfix(self._call(node))
            else:
                return node

    def _filter_chain(mut self, value: Int) raises -> Int:
        """One filter, and any that follow it separated by pipes."""
        var node = value
        while True:
            var at = self._at()
            var name = self._filter_name()
            var filter = self.tree.add(N_FILTER, at)
            self.tree.nodes[filter].text = name
            self.tree.nodes[filter].a = node
            if self._is_op("("):
                self._arguments(filter)
            node = filter
            if not self._skip_op("|"):
                return node

    def _filter_name(mut self) raises -> String:
        """A filter name, which may carry a dot in it.

        Nothing in the standard set does, but Jinja allows `a.b` as a filter
        name and reading it is one line, where getting it wrong turns into a
        confusing error about an unexpected dot.
        """
        var name = self._want_identifier()
        while self._is_op(".") and self.toks[self.pos + 1].kind == T_NAME:
            self.pos += 1
            name += "."
            name += self.toks[self.pos].text
            self.pos += 1
        return name

    def _test(mut self, value: Int) raises -> Int:
        var at = self._at()
        self.pos += 1
        var node = self.tree.add(N_TEST, at)
        self.tree.nodes[node].a = value
        if self._skip_name("not"):
            self.tree.nodes[node].flag = True
        # `is in(...)` names a test with a word that is otherwise an operator,
        # so the reserved word check does not apply in this one position.
        var got_text: String
        if self._kind() == T_NAME and self.toks[self.pos].text == "in":
            got_text = self.toks[self.pos].text
            self.pos += 1
        else:
            got_text = self._want_identifier()
        self.tree.nodes[node].text = got_text^
        # `is divisibleby 3` is allowed to leave the parentheses off, and
        # `is equalto x` likewise, so a bare argument is taken when what follows
        # could not be the start of something else.
        if self._is_op("("):
            self._arguments(node)
        elif self._starts_value():
            var bare = self._unary(False)
            self.tree.nodes[node].items.append(bare)
        return node

    def _starts_value(self) -> Bool:
        var kind = self._kind()
        if kind == T_STRING or kind == T_INT or kind == T_FLOAT:
            return True
        if kind == T_NAME:
            var text = self.toks[self.pos].text
            return not (
                text == "if"
                or text == "else"
                or text == "and"
                or text == "or"
                or text == "not"
                or text == "in"
                or text == "is"
                or text == "recursive"
            )
        return False

    def _arguments(mut self, into: Int) raises:
        """`(a, b, key=value)`. Positional in `items`, keywords in `names`.

        A keyword's name goes in `names` at the same index its value goes in
        `items`, and a positional argument gets the empty string, so one pair of
        parallel lists holds both and the evaluator reads them together.
        """
        self._want_op("(")
        var first = True
        while not self._is_op(")"):
            if not first:
                self._want_op(",")
                if self._is_op(")"):
                    break
            first = False
            if self._is_op("*"):
                self._fail(
                    self._at(),
                    "spreading arguments with `*` is not supported",
                )
            if (
                self._kind() == T_NAME
                and self.toks[self.pos + 1].kind == T_OP
                and self.toks[self.pos + 1].text == "="
            ):
                var key = self._want_identifier()
                self.pos += 1
                var value = self._expression()
                self.tree.nodes[into].names.append(key)
                self.tree.nodes[into].items.append(value)
            else:
                var value = self._expression()
                self.tree.nodes[into].names.append(String(""))
                self.tree.nodes[into].items.append(value)
        self._want_op(")")

    def _call(mut self, callee: Int) raises -> Int:
        var at = self._at()
        var node = self.tree.add(N_CALL, at)
        self.tree.nodes[node].a = callee
        self._arguments(node)
        return node

    def _postfix(mut self, value: Int) raises -> Int:
        var node = value
        while True:
            if self._is_op("."):
                var at = self._at()
                self.pos += 1
                if self._kind() == T_INT:
                    # `x.0` is item access on a list, which templates that
                    # unpack a pair sometimes write.
                    var item = self.tree.add(N_GETITEM, at)
                    var index = self.tree.add(N_INT, at)
                    self.tree.nodes[index].i = self.toks[self.pos].i
                    self.pos += 1
                    self.tree.nodes[item].a = node
                    self.tree.nodes[item].b = index
                    node = item
                    continue
                var attr = self.tree.add(N_GETATTR, at)
                self.tree.nodes[attr].a = node
                var got_text = self._want_identifier()
                self.tree.nodes[attr].text = got_text^
                node = attr
            elif self._is_op("["):
                node = self._subscript(node)
            elif self._is_op("("):
                # A call is postfix like the other two, so that
                # `content.split('</think>')[0].strip()` keeps going rather than
                # stopping at the first pair of brackets after a call.
                node = self._call(node)
            else:
                return node

    def _subscript(mut self, value: Int) raises -> Int:
        var at = self._at()
        self.pos += 1
        var lower = NO_NODE
        if not self._is_op(":"):
            lower = self._expression()
        if not self._is_op(":"):
            self._want_op("]")
            var node = self.tree.add(N_GETITEM, at)
            self.tree.nodes[node].a = value
            self.tree.nodes[node].b = lower
            return node
        self.pos += 1
        var node = self.tree.add(N_SLICE, at)
        self.tree.nodes[node].a = value
        self.tree.nodes[node].b = lower
        if not self._is_op("]") and not self._is_op(":"):
            var got_c = self._expression()
            self.tree.nodes[node].c = got_c
        if self._skip_op(":"):
            if not self._is_op("]"):
                var got_d = self._expression()
                self.tree.nodes[node].d = got_d
        self._want_op("]")
        return node

    def _primary(mut self) raises -> Int:
        var at = self._at()
        var kind = self._kind()
        if kind == T_STRING:
            var node = self.tree.add(N_STR, at)
            var text = self.toks[self.pos].text
            self.pos += 1
            # Adjacent string literals join, the way they do in Python, which
            # is how a long template writes a long constant without one very
            # long line.
            while self._kind() == T_STRING:
                text += self.toks[self.pos].text
                self.pos += 1
            self.tree.nodes[node].text = text
            return node
        if kind == T_INT:
            var node = self.tree.add(N_INT, at)
            self.tree.nodes[node].i = self.toks[self.pos].i
            self.pos += 1
            return node
        if kind == T_FLOAT:
            var node = self.tree.add(N_FLOAT, at)
            self.tree.nodes[node].f = self.toks[self.pos].f
            self.pos += 1
            return node
        if kind == T_NAME:
            var text = self.toks[self.pos].text
            if text == "true" or text == "True":
                self.pos += 1
                var node = self.tree.add(N_BOOL, at)
                self.tree.nodes[node].i = 1
                return node
            if text == "false" or text == "False":
                self.pos += 1
                return self.tree.add(N_BOOL, at)
            if text == "none" or text == "None":
                self.pos += 1
                return self.tree.add(N_NONE, at)
            self.pos += 1
            var node = self.tree.add(N_NAME, at)
            self.tree.nodes[node].text = text
            return node
        if self._is_op("("):
            self.pos += 1
            if self._skip_op(")"):
                return self.tree.add(N_TUPLE, at)
            var first = self._expression()
            if not self._is_op(","):
                self._want_op(")")
                return first
            var node = self.tree.add(N_TUPLE, at)
            self.tree.nodes[node].items.append(first)
            while self._skip_op(","):
                if self._is_op(")"):
                    break
                var element = self._expression()
                self.tree.nodes[node].items.append(element)
            self._want_op(")")
            return node
        if self._is_op("["):
            self.pos += 1
            var node = self.tree.add(N_LIST, at)
            while not self._is_op("]"):
                if len(self.tree.nodes[node].items) > 0:
                    self._want_op(",")
                    if self._is_op("]"):
                        break
                var element = self._expression()
                self.tree.nodes[node].items.append(element)
            self._want_op("]")
            return node
        if self._is_op("{"):
            self.pos += 1
            var node = self.tree.add(N_DICT, at)
            while not self._is_op("}"):
                if len(self.tree.nodes[node].items) > 0:
                    self._want_op(",")
                    if self._is_op("}"):
                        break
                var key = self._expression()
                self.tree.nodes[node].items.append(key)
                self._want_op(":")
                var value = self._expression()
                self.tree.nodes[node].items.append(value)
            self._want_op("}")
            return node
        self._fail(
            at,
            "expected a value, found `" + _describe(self.toks[self.pos]) + "`",
        )
        return NO_NODE


def node_names(mut tree: Tree, node: Int, first: String):
    tree.nodes[node].names.append(first)


def _describe(tok: Token) -> String:
    if tok.kind == T_EOF:
        return String("the end of the template")
    if tok.kind == T_TEXT:
        return String("template text")
    if tok.kind == T_VAR_BEGIN:
        return String("{{")
    if tok.kind == T_VAR_END:
        return String("}}")
    if tok.kind == T_BLOCK_BEGIN:
        return String("{%")
    if tok.kind == T_BLOCK_END:
        return String("%}")
    if tok.kind == T_INT:
        return String(tok.i)
    if tok.kind == T_FLOAT:
        return String(tok.f)
    if tok.kind == T_STRING:
        return "'" + tok.text + "'"
    return tok.text


def _is_reserved(name: String) -> Bool:
    return (
        name == "if"
        or name == "else"
        or name == "and"
        or name == "or"
        or name == "not"
        or name == "in"
        or name == "is"
    )


def _is_banned(name: String) -> Bool:
    return (
        name == "include"
        or name == "extends"
        or name == "import"
        or name == "from"
        or name == "do"
        or name == "autoescape"
        or name == "endautoescape"
        or name == "trans"
        or name == "pluralize"
        or name == "debug"
    )


def _ban_reason(name: String) raises -> String:
    if (
        name == "include"
        or name == "extends"
        or name == "import"
        or name == "from"
    ):
        return (
            "`"
            + name
            + "` needs a template loader, and a chat template is one self"
            " contained string with nothing to load"
        )
    if name == "do":
        return (
            "`do` is a Jinja extension that transformers does not enable, so"
            " this template does not render there either"
        )
    if name == "autoescape" or name == "endautoescape":
        return (
            "`autoescape` is not supported, a chat template renders text and"
            " not markup"
        )
    if name == "debug":
        return "`debug` is not supported"
    return (
        "`"
        + name
        + "` is part of the i18n extension, which transformers does not enable"
    )


def parse_template(source: String) raises -> Tree:
    """Source to a tree, which is the only entry point anything else needs."""
    var parser = Parser(source)
    parser.parse()
    return parser^.into_tree()
