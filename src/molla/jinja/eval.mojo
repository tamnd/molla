"""Walking the tree and writing the prompt.

The tree is read only and the state is all in `Env`, so rendering the same
template twice concurrently would be two `Env`s over one `Tree` and nothing
would need locking. That is why the parser hands back a plain value rather than
something that carries its own scratch space.

Statements return a control signal rather than raising to unwind a loop.
`{% break %}` inside three nested ifs has to leave all three and then leave the
loop, and a signal threaded back through the block walker does that without an
exception on a path that is not exceptional.

Everything is charged against the step budget in one place, at the top of
`evaluate` and `execute`. That is one branch per node, and it is what makes the
guarantee that a template cannot run forever a property of the evaluator rather
than something each statement has to remember.
"""

from molla.jinja.access import (
    B_DICT,
    B_LOOP_CHANGED,
    B_LOOP_CYCLE,
    B_NAMESPACE,
    B_RAISE_EXCEPTION,
    B_RANGE,
    B_STRFTIME_NOW,
    bind_method,
    call_builtin,
    get_attr,
    get_item,
    get_slice,
    global_builtin,
    is_namespace,
    iterate,
    value_in,
)
from molla.jinja.ast import (
    NO_NODE,
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
    OP_POW,
    OP_SUB,
    Tree,
)
from molla.jinja.env import Env, NO_FRAME
from molla.jinja.filters import apply_filter, apply_test, filter_id, test_id
from molla.jinja.parser import BREAK_LOOP, N_LOOP_CONTROL
from molla.jinja.strop import repeat
from molla.jinja.value import (
    Cell,
    TUPLE_FLAG,
    FALSE,
    NONE,
    TRUE,
    UNDEFINED,
    V_DICT,
    V_FLOAT,
    V_LIST,
    V_MACRO,
    V_STRING,
    as_float,
    as_int,
    compare,
    dict_set,
    equal,
    is_number,
    to_string,
    truthy,
)

comptime GO_ON = 0
comptime GO_BREAK = 1
comptime GO_CONTINUE = 2


def builtin_frame(mut env: Env) -> Int:
    """The outermost scope, holding the five names every template gets free.

    The caller binds the template's own variables into a child of this, so a
    template given a variable called `range` sees the variable, the way it would
    under the reference.
    """
    var root = env.push(NO_FRAME)
    _install_globals(env, root)
    return root


def render(mut env: Env, tree: Tree, frame: Int) raises:
    """The whole template, into the environment's buffer.

    `frame` is the scope the caller bound the template's variables into. The
    body renders in a child of it rather than in it, so a `{% set %}` at the top
    level shadows a variable rather than overwriting it.
    """
    var body = env.push(frame)
    _ = execute(env, tree, tree.root, body)


def render(mut env: Env, tree: Tree) raises:
    """The whole template with nothing bound but the builtins."""
    render(env, tree, builtin_frame(env))


def _install_globals(mut env: Env, frame: Int):
    """The five names a chat template gets for free."""
    env.bind(
        frame,
        String("range"),
        global_builtin(env.heap, B_RANGE, String("range")),
    )
    env.bind(
        frame, String("dict"), global_builtin(env.heap, B_DICT, String("dict"))
    )
    env.bind(
        frame,
        String("namespace"),
        global_builtin(env.heap, B_NAMESPACE, String("namespace")),
    )
    env.bind(
        frame,
        String("strftime_now"),
        global_builtin(env.heap, B_STRFTIME_NOW, String("strftime_now")),
    )
    env.bind(
        frame,
        String("raise_exception"),
        global_builtin(env.heap, B_RAISE_EXCEPTION, String("raise_exception")),
    )


def execute(mut env: Env, tree: Tree, node: Int, frame: Int) raises -> Int:
    """One statement, and the control signal it wants the caller to act on."""
    if node == NO_NODE:
        return GO_ON
    env.step()
    var kind = tree.nodes[node].kind

    if kind == N_BLOCK:
        # A `{% with %}` is a block with its own scope, which the parser marks
        # rather than giving it a node kind of its own.
        var here = env.push(frame) if tree.nodes[node].flag else frame
        for i in range(len(tree.nodes[node].items)):
            var signal = execute(env, tree, tree.nodes[node].items[i], here)
            if signal != GO_ON:
                return signal
        return GO_ON

    if kind == N_TEXT:
        env.emit(tree.nodes[node].text)
        return GO_ON

    if kind == N_OUTPUT:
        var value = evaluate(env, tree, tree.nodes[node].a, frame)
        env.emit(to_string(env.heap, value))
        return GO_ON

    if kind == N_IF:
        var here = node
        while here != NO_NODE and tree.nodes[here].kind == N_IF:
            var test = evaluate(env, tree, tree.nodes[here].a, frame)
            if truthy(env.heap, test):
                return execute(env, tree, tree.nodes[here].b, frame)
            here = tree.nodes[here].c
        return execute(env, tree, here, frame)

    if kind == N_FOR:
        return _for(env, tree, node, frame)

    if kind == N_SET:
        return _set(env, tree, node, frame)

    if kind == N_SET_BLOCK:
        var mark = env.mark()
        _ = execute(env, tree, tree.nodes[node].b, frame)
        var text = env.take(mark)
        var value = env.heap.str(text)
        if tree.nodes[node].a != NO_NODE:
            value = _chain(env, tree, tree.nodes[node].a, frame, value)
        env.bind(frame, tree.nodes[node].names[0], value)
        return GO_ON

    if kind == N_FILTER_BLOCK:
        var mark = env.mark()
        _ = execute(env, tree, tree.nodes[node].b, frame)
        var text = env.take(mark)
        var base = env.heap.str(text)
        var value = _chain(env, tree, tree.nodes[node].a, frame, base)
        env.emit(to_string(env.heap, value))
        return GO_ON

    if kind == N_MACRO:
        var cell = Cell(V_MACRO)
        cell.s = tree.nodes[node].text
        cell.node = node
        cell.scope = frame
        var made = env.heap.add(cell^)
        # The frame the macro closed over cannot be recycled by the next
        # iteration of whatever loop it was defined in, because the macro will
        # still be looking at it.
        env.frames[frame].captured = True
        env.bind(frame, tree.nodes[node].text, made)
        return GO_ON

    if kind == N_CALL_BLOCK:
        return _call_block(env, tree, node, frame)

    if kind == N_LOOP_CONTROL:
        return GO_BREAK if tree.nodes[node].a == BREAK_LOOP else GO_CONTINUE

    env.fail(tree.nodes[node].at, "this statement cannot be rendered")
    return GO_ON


def _set(mut env: Env, tree: Tree, node: Int, frame: Int) raises -> Int:
    var value = evaluate(env, tree, tree.nodes[node].a, frame)

    if tree.nodes[node].flag:
        # `{% set ns.field = value %}`, the one assignment that reaches through
        # a name instead of rebinding it.
        var target = env.lookup(frame, tree.nodes[node].names[0])
        if target < 0 or not is_namespace(env.heap, target):
            env.fail(
                tree.nodes[node].at,
                "'"
                + tree.nodes[node].names[0]
                + "' is not a namespace, and only a namespace can be assigned"
                " through",
            )
        dict_set(env.heap, target, tree.nodes[node].text, value)
        return GO_ON

    if len(tree.nodes[node].names) == 1:
        env.bind(frame, tree.nodes[node].names[0], value)
        return GO_ON

    var parts = _unpack(
        env, value, len(tree.nodes[node].names), tree.nodes[node].at
    )
    for i in range(len(parts)):
        env.bind(frame, tree.nodes[node].names[i], parts[i])
    return GO_ON


def _unpack(mut env: Env, value: Int, want: Int, at: Int) raises -> List[Int]:
    """Several names from one value, which has to be a sequence of that size."""
    if env.heap.kind(value) != V_LIST:
        env.fail(
            at,
            "several names were assigned from something that is not a sequence",
        )
    if len(env.heap.cells[value].items) != want:
        env.fail(
            at,
            "there are "
            + String(want)
            + " names here and "
            + String(len(env.heap.cells[value].items))
            + " values to give them",
        )
    var out = List[Int]()
    for i in range(want):
        out.append(env.heap.cells[value].items[i])
    return out^


comptime _LOOP_KEYS = String(
    "index0|index|revindex0|revindex|first|last|length|previtem|nextitem|"
    "depth0|depth|"
)
"""The loop object's plain fields, in the order they are written into it.

`cycle` and `changed` are not here because they are callables that hold state
and are made once per loop rather than once per iteration.
"""


def _for(mut env: Env, tree: Tree, node: Int, frame: Int) raises -> Int:
    var source = evaluate(env, tree, tree.nodes[node].a, frame)
    var items = iterate(env, source, tree.nodes[node].at)
    ref names = tree.nodes[node].names
    var one_name = len(names) == 1

    var body_frame = env.push(frame)

    # The loop filter runs in a scope where the target names are bound, because
    # `for m in messages if m.role == 'user'` is testing each message.
    if tree.nodes[node].d != NO_NODE:
        var kept = List[Int]()
        for i in range(len(items)):
            env.step()
            body_frame = env.reuse(body_frame, frame)
            _bind_targets(
                env, body_frame, names, items[i], one_name, tree.nodes[node].at
            )
            if truthy(
                env.heap, evaluate(env, tree, tree.nodes[node].d, body_frame)
            ):
                kept.append(items[i])
        items = kept^

    if len(items) == 0:
        return execute(env, tree, tree.nodes[node].c, frame)

    var cycle = bind_method(env.heap, B_LOOP_CYCLE, String("cycle"), -1)
    var loop = _make_loop(env, cycle)

    for i in range(len(items)):
        env.step()
        body_frame = env.reuse(body_frame, frame)
        _bind_targets(
            env, body_frame, names, items[i], one_name, tree.nodes[node].at
        )
        _fill_loop(env, loop, items, i)
        env.heap.cells[cycle].scope = i
        env.bind(body_frame, String("loop"), loop)
        var signal = execute(env, tree, tree.nodes[node].b, body_frame)
        if signal == GO_BREAK:
            break
    return GO_ON


def _bind_targets(
    mut env: Env,
    frame: Int,
    names: List[String],
    value: Int,
    one_name: Bool,
    at: Int,
) raises:
    if one_name:
        env.bind(frame, names[0], value)
        return
    var parts = _unpack(env, value, len(names), at)
    for i in range(len(names)):
        env.bind(frame, names[i], parts[i])


comptime _LOOP_FIXED = 2
"""Slots before the plain fields: `cycle` and `changed`."""


def _make_loop(mut env: Env, cycle: Int) raises -> Int:
    """The loop object, with every key it will ever have already in it.

    Built once per loop rather than once per iteration, so an iteration writes
    eleven values into slots it already knows and does not build eleven key
    strings and scan for them. On a twenty message conversation that is the
    difference between a few hundred allocations and none.
    """
    var loop = env.heap.dict()
    dict_set(env.heap, loop, String("cycle"), cycle)
    var changed = bind_method(env.heap, B_LOOP_CHANGED, String("changed"), -1)
    dict_set(env.heap, loop, String("changed"), changed)
    var start = 0
    var table = _LOOP_KEYS.as_bytes()
    for at in range(len(table)):
        if table[at] != 0x7C:
            continue
        var raw = List[UInt8]()
        for j in range(start, at):
            raw.append(table[j])
        dict_set(
            env.heap,
            loop,
            String(StringSpan(unsafe_from_utf8=Span(raw))),
            UNDEFINED,
        )
        start = at + 1
    return loop


def _fill_loop(mut env: Env, loop: Int, items: List[Int], i: Int) raises:
    var n = len(items)
    var values = List[Int]()
    values.append(env.heap.int(i))
    values.append(env.heap.int(i + 1))
    values.append(env.heap.int(n - i - 1))
    values.append(env.heap.int(n - i))
    values.append(TRUE if i == 0 else FALSE)
    values.append(TRUE if i == n - 1 else FALSE)
    values.append(env.heap.int(n))
    values.append(items[i - 1] if i > 0 else UNDEFINED)
    values.append(items[i + 1] if i + 1 < n else UNDEFINED)
    values.append(env.heap.int(0))
    values.append(env.heap.int(1))
    for slot in range(len(values)):
        env.heap.cells[loop].items[_LOOP_FIXED + slot] = values[slot]


def _call_block(mut env: Env, tree: Tree, node: Int, frame: Int) raises -> Int:
    """`{% call %}`, which is a macro call with a body attached as `caller`."""
    var caller = Cell(V_MACRO)
    caller.s = String("caller")
    caller.node = node
    caller.scope = frame
    var made = env.heap.add(caller^)
    env.frames[frame].captured = True

    var call = tree.nodes[node].a
    var args = List[Int]()
    var names = List[String]()
    _eval_args(env, tree, call, frame, args, names)
    var callee = evaluate(env, tree, tree.nodes[call].a, frame)
    if env.heap.kind(callee) != V_MACRO:
        env.fail(tree.nodes[node].at, "`call` needs a macro to call")
    var text = _invoke_macro(
        env, tree, callee, args, names, made, tree.nodes[node].at
    )
    env.emit(text)
    return GO_ON


def _invoke_macro(
    mut env: Env,
    tree: Tree,
    macro: Int,
    args: List[Int],
    names: List[String],
    caller: Int,
    at: Int,
) raises -> String:
    """Bind the parameters, render the body, hand back what it wrote.

    A macro's value in Jinja is the text of its body, which is why this returns
    a string rather than writing straight through. `{{ greet('x') }}` and
    `{% set g = greet('x') %}` then differ only in what the caller does with it.

    Defaults are evaluated at call time in the scope the macro was defined in.
    Jinja evaluates them at definition time, which is only visible when a
    default reads a name that changes between the two, and nothing writes that.
    """
    env.depth += 1
    if env.depth > env.limits.depth:
        raise Error(
            "macros nested more than "
            + String(env.limits.depth)
            + " deep, which means one of them calls itself"
        )

    var node = env.heap.cells[macro].node
    var scope = env.heap.cells[macro].scope
    var inner = env.push(scope)

    ref wanted = tree.nodes[node].names
    var given = 0
    for i in range(len(wanted)):
        var value = -1
        for j in range(len(names)):
            if names[j] == wanted[i]:
                value = args[j]
        if value < 0:
            # Positional arguments fill the parameters in order, and a keyword
            # already matched above takes its parameter out of that order.
            var seen = 0
            for j in range(len(args)):
                if names[j] != "":
                    continue
                if seen == given:
                    value = args[j]
                    break
                seen += 1
            if value >= 0:
                given += 1
        if value < 0:
            var default = tree.nodes[node].items[i]
            value = UNDEFINED if default == NO_NODE else evaluate(
                env, tree, default, scope
            )
        env.bind(inner, wanted[i], value)

    if caller >= 0:
        env.bind(inner, String("caller"), caller)

    var mark = env.mark()
    _ = execute(env, tree, tree.nodes[node].b, inner)
    var text = env.take(mark)
    env.depth -= 1
    return text


def evaluate(mut env: Env, tree: Tree, node: Int, frame: Int) raises -> Int:
    """One expression, to a value in the heap."""
    if node == NO_NODE:
        return UNDEFINED
    env.step()
    var kind = tree.nodes[node].kind

    if kind == N_STR:
        return env.heap.str(tree.nodes[node].text)
    if kind == N_INT:
        return env.heap.int(tree.nodes[node].i)
    if kind == N_FLOAT:
        return env.heap.float(tree.nodes[node].f)
    if kind == N_BOOL:
        return TRUE if tree.nodes[node].i != 0 else FALSE
    if kind == N_NONE:
        return NONE

    if kind == N_NAME:
        var got = env.lookup(frame, tree.nodes[node].text)
        return got if got >= 0 else UNDEFINED

    if kind == N_LIST or kind == N_TUPLE:
        var made = env.heap.list()
        if kind == N_TUPLE:
            env.heap.cells[made].i = TUPLE_FLAG
        for i in range(len(tree.nodes[node].items)):
            var element = evaluate(env, tree, tree.nodes[node].items[i], frame)
            env.heap.push(made, element)
        return made

    if kind == N_DICT:
        var made = env.heap.dict()
        var i = 0
        while i + 1 < len(tree.nodes[node].items):
            var key = evaluate(env, tree, tree.nodes[node].items[i], frame)
            var value = evaluate(
                env, tree, tree.nodes[node].items[i + 1], frame
            )
            dict_set(env.heap, made, to_string(env.heap, key), value)
            i += 2
        return made

    if kind == N_GETATTR:
        var target = evaluate(env, tree, tree.nodes[node].a, frame)
        return get_attr(env, target, tree.nodes[node].text, tree.nodes[node].at)

    if kind == N_GETITEM:
        var target = evaluate(env, tree, tree.nodes[node].a, frame)
        var key = evaluate(env, tree, tree.nodes[node].b, frame)
        return get_item(env, target, key, tree.nodes[node].at)

    if kind == N_SLICE:
        return _slice(env, tree, node, frame)

    if kind == N_NOT:
        var value = evaluate(env, tree, tree.nodes[node].a, frame)
        return FALSE if truthy(env.heap, value) else TRUE

    if kind == N_AND:
        # Python's `and` gives back one of its operands rather than a bool, and
        # a template writing `{{ x and x.name }}` depends on the short circuit.
        var left = evaluate(env, tree, tree.nodes[node].a, frame)
        if not truthy(env.heap, left):
            return left
        return evaluate(env, tree, tree.nodes[node].b, frame)

    if kind == N_OR:
        var left = evaluate(env, tree, tree.nodes[node].a, frame)
        if truthy(env.heap, left):
            return left
        return evaluate(env, tree, tree.nodes[node].b, frame)

    if kind == N_COND:
        var test = evaluate(env, tree, tree.nodes[node].b, frame)
        if truthy(env.heap, test):
            return evaluate(env, tree, tree.nodes[node].a, frame)
        return evaluate(env, tree, tree.nodes[node].c, frame)

    if kind == N_UNARY:
        var value = evaluate(env, tree, tree.nodes[node].a, frame)
        if tree.nodes[node].i != OP_NEG:
            return value
        if env.heap.kind(value) == V_FLOAT:
            return env.heap.float(-env.heap.cells[value].f)
        if not is_number(env.heap, value):
            env.fail(tree.nodes[node].at, "only a number can be negated")
        return env.heap.int(-as_int(env.heap, value))

    if kind == N_BINOP:
        return _binop(env, tree, node, frame)

    if kind == N_FILTER:
        var value = evaluate(env, tree, tree.nodes[node].a, frame)
        return _apply(env, tree, node, frame, value)

    if kind == N_TEST:
        return _test(env, tree, node, frame)

    if kind == N_CALL:
        return _call(env, tree, node, frame)

    env.fail(tree.nodes[node].at, "this expression cannot be evaluated")
    return UNDEFINED


def _slice(mut env: Env, tree: Tree, node: Int, frame: Int) raises -> Int:
    var target = evaluate(env, tree, tree.nodes[node].a, frame)
    var has_start = tree.nodes[node].b != NO_NODE
    var has_stop = tree.nodes[node].c != NO_NODE
    var has_step = tree.nodes[node].d != NO_NODE
    var start = 0
    var stop = 0
    var step = 1
    if has_start:
        start = as_int(env.heap, evaluate(env, tree, tree.nodes[node].b, frame))
    if has_stop:
        stop = as_int(env.heap, evaluate(env, tree, tree.nodes[node].c, frame))
    if has_step:
        step = as_int(env.heap, evaluate(env, tree, tree.nodes[node].d, frame))
    return get_slice(
        env,
        target,
        start,
        stop,
        step,
        has_start,
        has_stop,
        has_step,
        tree.nodes[node].at,
    )


def _eval_args(
    mut env: Env,
    tree: Tree,
    node: Int,
    frame: Int,
    mut args: List[Int],
    mut names: List[String],
) raises:
    for i in range(len(tree.nodes[node].items)):
        args.append(evaluate(env, tree, tree.nodes[node].items[i], frame))
        names.append(tree.nodes[node].names[i])


def _apply(
    mut env: Env, tree: Tree, node: Int, frame: Int, value: Int
) raises -> Int:
    var id = filter_id(tree.nodes[node].text)
    if id == 0:
        env.fail(
            tree.nodes[node].at,
            "there is no filter called '" + tree.nodes[node].text + "'",
        )
    var args = List[Int]()
    var names = List[String]()
    _eval_args(env, tree, node, frame, args, names)
    return apply_filter(env, id, value, args, names, tree.nodes[node].at)


def _chain(
    mut env: Env, tree: Tree, node: Int, frame: Int, base: Int
) raises -> Int:
    """A filter chain whose innermost input is a captured block, not a node."""
    var input = tree.nodes[node].a
    var value = base if input == NO_NODE else _chain(
        env, tree, input, frame, base
    )
    return _apply(env, tree, node, frame, value)


def _test(mut env: Env, tree: Tree, node: Int, frame: Int) raises -> Int:
    var id = test_id(tree.nodes[node].text)
    if id == 0:
        env.fail(
            tree.nodes[node].at,
            "there is no test called '" + tree.nodes[node].text + "'",
        )
    var value = evaluate(env, tree, tree.nodes[node].a, frame)
    var args = List[Int]()
    var names = List[String]()
    _eval_args(env, tree, node, frame, args, names)
    var got = apply_test(env, id, value, args, names, tree.nodes[node].at)
    if tree.nodes[node].flag:
        got = not got
    return TRUE if got else FALSE


def _call(mut env: Env, tree: Tree, node: Int, frame: Int) raises -> Int:
    var callee = evaluate(env, tree, tree.nodes[node].a, frame)
    var args = List[Int]()
    var names = List[String]()
    _eval_args(env, tree, node, frame, args, names)

    var kind = env.heap.kind(callee)
    if kind == V_MACRO:
        var text = _invoke_macro(
            env, tree, callee, args, names, -1, tree.nodes[node].at
        )
        return env.heap.str(text)
    if kind == V_STRING or kind == V_DICT or kind == V_LIST:
        env.fail(
            tree.nodes[node].at,
            "this value is not something that can be called",
        )
    return call_builtin(env, callee, args, names, tree.nodes[node].at)


def _binop(mut env: Env, tree: Tree, node: Int, frame: Int) raises -> Int:
    var op = tree.nodes[node].i
    var at = tree.nodes[node].at
    var a = evaluate(env, tree, tree.nodes[node].a, frame)
    var b = evaluate(env, tree, tree.nodes[node].b, frame)

    if op == OP_CONCAT:
        return env.heap.str(to_string(env.heap, a) + to_string(env.heap, b))
    if op == OP_EQ:
        return TRUE if equal(env.heap, a, b) else FALSE
    if op == OP_NE:
        return FALSE if equal(env.heap, a, b) else TRUE
    if op == OP_IN or op == OP_NOT_IN:
        var inside = value_in(env, a, b, at)
        if op == OP_NOT_IN:
            inside = not inside
        return TRUE if inside else FALSE
    if op == OP_LT or op == OP_LE or op == OP_GT or op == OP_GE:
        var c = compare(env.heap, a, b)
        var got = c < 0
        if op == OP_LE:
            got = c <= 0
        elif op == OP_GT:
            got = c > 0
        elif op == OP_GE:
            got = c >= 0
        return TRUE if got else FALSE

    var ka = env.heap.kind(a)
    var kb = env.heap.kind(b)

    if op == OP_ADD:
        if ka == V_STRING and kb == V_STRING:
            var left = env.heap.cells[a].s
            return env.heap.str(left + env.heap.cells[b].s)
        if ka == V_LIST and kb == V_LIST:
            var made = env.heap.list()
            for i in range(len(env.heap.cells[a].items)):
                var element = env.heap.cells[a].items[i]
                env.heap.push(made, element)
            for i in range(len(env.heap.cells[b].items)):
                var element = env.heap.cells[b].items[i]
                env.heap.push(made, element)
            return made

    if op == OP_MUL:
        if ka == V_STRING and is_number(env.heap, b):
            var text = env.heap.cells[a].s
            return env.heap.str(repeat(text, as_int(env.heap, b)))
        if kb == V_STRING and is_number(env.heap, a):
            var text = env.heap.cells[b].s
            return env.heap.str(repeat(text, as_int(env.heap, a)))
        if ka == V_LIST and is_number(env.heap, b):
            var made = env.heap.list()
            for _ in range(as_int(env.heap, b)):
                for i in range(len(env.heap.cells[a].items)):
                    var element = env.heap.cells[a].items[i]
                    env.heap.push(made, element)
            return made

    if not is_number(env.heap, a) or not is_number(env.heap, b):
        env.fail(at, "these two values cannot be combined with this operator")

    # `/` is always float, the way Python 3 does it, and `//` and `%` on two
    # integers stay integers with Python's sign rule rather than C's.
    if op == OP_DIV:
        var divisor = as_float(env.heap, b)
        if divisor == 0.0:
            env.fail(at, "division by zero")
        return env.heap.float(as_float(env.heap, a) / divisor)

    var whole = ka != V_FLOAT and kb != V_FLOAT
    if whole:
        var x = as_int(env.heap, a)
        var y = as_int(env.heap, b)
        if op == OP_ADD:
            return env.heap.int(x + y)
        if op == OP_SUB:
            return env.heap.int(x - y)
        if op == OP_MUL:
            return env.heap.int(x * y)
        if op == OP_POW:
            if y < 0:
                return env.heap.float(_power(Float64(x), y))
            var raised = 1
            for _ in range(y):
                raised = raised * x
            return env.heap.int(raised)
        if y == 0:
            env.fail(at, "division by zero")
        # Python floors towards negative infinity and takes the sign of the
        # divisor, where the machine truncates towards zero.
        var q = x // y
        var r = x - q * y
        if r != 0 and (r < 0) != (y < 0):
            q -= 1
            r += y
        return env.heap.int(q if op == OP_FLOORDIV else r)

    var x = as_float(env.heap, a)
    var y = as_float(env.heap, b)
    if op == OP_ADD:
        return env.heap.float(x + y)
    if op == OP_SUB:
        return env.heap.float(x - y)
    if op == OP_MUL:
        return env.heap.float(x * y)
    if op == OP_POW:
        return env.heap.float(_power(x, as_int(env.heap, b)))
    if y == 0.0:
        env.fail(at, "division by zero")
    var q = Float64(Int(x / y))
    if x / y < 0.0 and q != x / y:
        q -= 1.0
    return env.heap.float(q if op == OP_FLOORDIV else x - q * y)


def _power(base: Float64, exponent: Int) -> Float64:
    """Repeated multiplication, because the exponents templates write are small.
    """
    var n = exponent if exponent >= 0 else -exponent
    var raised = 1.0
    for _ in range(n):
        raised = raised * base
    return raised if exponent >= 0 else 1.0 / raised
