"""Tests for `molla.jinja`.

This engine has an answer key, the same way the text layer does: whatever
`jinja2` under `transformers.apply_chat_template` prints is right by
definition, and a difference is a bug here no matter how defensible it looks.
The full differential run lives outside the suite because it downloads
templates and needs a Python interpreter, and what it covers is written down in
`docs/validation/jinja.md`.

What is here is the part that run would find too late. Every check below was
either a bug that a real template caught or a behaviour that separates a
correct implementation from a plausible one. A lexer that strips indentation in
front of a tag that is not at the start of a line passes every single line
template. A parser that treats a call as an ending rather than as a postfix
renders every template that never indexes a call result. An evaluator that
loses the right operand of a binary node once the tree has grown past a
reallocation prints the left one, which looks like an answer.

The refusals matter as much as the renders. An unsupported construct has to
fail when the template is compiled, which is when the model loads, and not when
it is rendered, which is on somebody's request. Both directions are checked:
the named exclusions raise at compile time, and the four limits raise at render
time.
"""

from harness import Suite

from molla.jinja.env import Limits
from molla.jinja.template import (
    Binding,
    Cache,
    Template,
    flag_binding,
    json_binding,
    text_binding,
)

comptime NO_VARS = String("{}")


def _render(source: String, vars: String) raises -> String:
    var made = Template(source)
    return made.render_object(vars)


def _ok(
    mut suite: Suite, source: String, vars: String, want: String, name: String
):
    """Render and compare, reporting the two strings when they differ.

    A whitespace difference is the common failure here and is invisible in a
    diff of the two, so the report prints them quoted.
    """
    try:
        var got = _render(source, vars)
        if got == want:
            suite.check(True, name)
        else:
            suite.fail(name, "want '" + want + "' got '" + got + "'")
    except e:
        suite.fail(name, String(e))


def _ok(mut suite: Suite, source: String, want: String, name: String):
    _ok(suite, source, NO_VARS, want, name)


def _refused(mut suite: Suite, source: String, needle: String, name: String):
    """The template has to fail to compile, and say why in the message."""
    try:
        var made = Template(source)
        _ = made.digest
        suite.fail(name, "compiled instead of being refused")
    except e:
        var message = String(e)
        if needle in message:
            suite.check(True, name)
        else:
            suite.fail(name, "wrong reason: " + message)


def _render_fails(
    mut suite: Suite, source: String, needle: String, name: String
):
    try:
        var got = _render(source, NO_VARS)
        suite.fail(name, "rendered '" + got + "'")
    except e:
        var message = String(e)
        if needle in message:
            suite.check(True, name)
        else:
            suite.fail(name, "wrong reason: " + message)


def _lexer(mut suite: Suite):
    suite.group("jinja/lexer")

    _ok(suite, "hello", "hello", "text passes through")
    _ok(suite, "a{# note #}b", "ab", "a comment produces nothing")
    _ok(suite, "{% raw %}{{ x }}{% endraw %}", "{{ x }}", "raw is not lexed")

    # `trim_blocks` and `lstrip_blocks`, both of which `apply_chat_template`
    # turns on and neither of which is the default anywhere else.
    _ok(
        suite,
        "a\n  {% if true %}\nb\n  {% endif %}\nc",
        "a\nb\nc",
        "a tag on its own line contributes nothing",
    )
    # The bug this pins: the space here is not indentation, because the line
    # started before the first tag, so it has to survive.
    _ok(
        suite,
        "{{ 1 }} {% if true %}x{% endif %}",
        "1 x",
        "lstrip_blocks only strips at the start of a line",
    )
    _ok(
        suite,
        "a\n  {%+ if true %}b{% endif %}",
        "a\n  b",
        "a plus turns lstrip_blocks off for one tag",
    )
    _ok(
        suite,
        "a  {%- if true %}b{% endif %}",
        "ab",
        "a minus strips across the newline",
    )

    # `keep_trailing_newline` is off in the reference, so one newline at the
    # end of the source is not part of the template.
    _ok(suite, "a\n", "a", "one trailing newline is dropped")
    _ok(suite, "a\n\n", "a\n", "only one of them is dropped")

    _ok(suite, "a\r\nb", "a\nb", "crlf becomes one newline")

    # Python string escapes, which a template uses for a zero width space or an
    # accented word it would rather not put in the source as bytes.
    _ok(suite, "{{ 'caf\\u00e9' }}", "café", "a unicode escape decodes")
    _ok(suite, "{{ '\\x41\\101' }}", "AA", "hex and octal escapes decode")
    _ok(suite, "{{ 'a\\qb' }}", "a\\qb", "an unknown escape keeps its slash")

    _refused(
        suite, "{{ '\\N{BULLET}' }}", "Unicode name table", "no name escapes"
    )
    _refused(suite, "{{ 'x }}", "unterminated string", "an open quote fails")

    # A closing brace inside a dict literal is not the end of the block, which
    # is why the lexer counts brackets rather than scanning for the delimiter.
    _ok(suite, "{{ {'a': 1}['a'] }}", "1", "braces nest inside a block")


def _expressions(mut suite: Suite):
    suite.group("jinja/expressions")

    _ok(suite, "{{ 1 + 2 * 3 }}", "7", "precedence")
    _ok(suite, "{{ (1 + 2) * 3 }}", "9", "brackets group")
    _ok(suite, "{{ 2 ** 3 ** 2 }}", "512", "power is right associative")
    _ok(suite, "{{ 7 / 2 }}", "3.5", "a slash is always a float")
    _ok(suite, "{{ 7 // 2 }}", "3", "a double slash floors")
    _ok(suite, "{{ -7 // 2 }}", "-4", "flooring goes down, not toward zero")
    _ok(suite, "{{ -7 % 3 }}", "2", "modulo takes the sign of the divisor")
    _ok(suite, "{{ 'a' ~ 1 }}", "a1", "tilde stringifies both sides")
    _ok(suite, "{{ [1] + [2] }}", "[1, 2]", "lists concatenate")
    _ok(suite, "{{ 'ab' * 2 }}", "abab", "a string repeats")

    _ok(suite, "{{ true and false }}", "False", "booleans print capitalised")
    _ok(suite, "{{ none }}", "None", "so does none")
    _ok(suite, "{{ 1 < 2 < 3 }}", "True", "comparisons chain")
    _ok(suite, "{{ 'a' if false else 'b' }}", "b", "an inline if")
    _ok(suite, "{{ 'x' in 'text' }}", "True", "in on a string")
    _ok(suite, "{{ 1 not in [2, 3] }}", "True", "not in on a list")

    # Granite asks `if 'citations' in controls` where `controls` was never
    # bound. The reference answers False rather than raising.
    _ok(suite, "{{ 'a' in missing }}", "False", "in on an undefined name")

    _ok(suite, "{{ [1, 2, 3][1:] }}", "[2, 3]", "slicing from")
    _ok(suite, "{{ [1, 2, 3][:-1] }}", "[1, 2]", "slicing to a negative")
    _ok(suite, "{{ 'hello'[1:3] }}", "el", "slicing a string")
    _ok(suite, "{{ {'a': {'b': 2}}.a.b }}", "2", "attribute access nests")
    _ok(suite, "{{ (1, 2) }}", "(1, 2)", "a tuple prints with brackets")
    _ok(suite, "{{ (1,) }}", "(1,)", "and a one element tuple with a comma")

    # This is what made Qwen3 and DeepSeek R1 refuse to compile: a call is a
    # postfix, so indexing what it returned has to keep parsing.
    _ok(
        suite,
        "{{ 'a</think>b'.split('</think>')[0] }}",
        "a",
        "indexing the result of a call",
    )
    _ok(
        suite,
        "{{ ' a '.strip().upper() }}",
        "A",
        "calls chain after calls",
    )

    # The evaluator lost the right operand once the tree grew past a
    # reallocation, so a binop this far into a template is the check that
    # catches it and one at the top is not.
    _ok(
        suite,
        "{{ 1 }}{{ 2 }}{{ 3 }}{{ 4 }}{{ 5 }}{{ 6 }}{{ 7 }}{{ 8 }}{{ 1 + 2 }}",
        "123456783",
        "a binop late in a long template still has two sides",
    )


def _filters(mut suite: Suite):
    suite.group("jinja/filters")

    _ok(suite, "{{ ' a ' | trim }}", "a", "trim")
    _ok(suite, "{{ [1, 2] | length }}", "2", "length")
    _ok(suite, "{{ ['a', 'b'] | join('-') }}", "a-b", "join")
    _ok(suite, "{{ 'ab' | upper }}", "AB", "upper")
    _ok(suite, "{{ 'AB' | lower }}", "ab", "lower")
    _ok(suite, "{{ 'ab' | capitalize }}", "Ab", "capitalize")
    _ok(suite, "{{ 'a-b' | replace('-', '+') }}", "a+b", "replace")
    _ok(suite, "{{ missing | default('d') }}", "d", "default on undefined")
    _ok(suite, "{{ '' | default('d', true) }}", "d", "default with boolean")
    _ok(suite, "{{ [3, 1] | sort | first }}", "1", "sort then first")
    _ok(suite, "{{ [1, 2, 3] | last }}", "3", "last")
    _ok(suite, "{{ [1, 2, 3] | sum }}", "6", "sum")
    _ok(suite, "{{ [1, 2] | max }}", "2", "max")
    _ok(suite, "{{ 2.6 | round }}", "3.0", "round returns a float")
    _ok(suite, "{{ '3' | int + 1 }}", "4", "int")
    _ok(suite, "{{ [[1], [2]] | list | length }}", "2", "list")

    _ok(
        suite,
        "{{ [1, 2, 3] | select('odd') | list }}",
        "[1, 3]",
        "select with a test",
    )
    _ok(
        suite,
        "{{ [{'a': 1}, {'a': 2}] | selectattr('a', 'eq', 2) | length }}",
        "1",
        "selectattr with an argument",
    )
    _ok(
        suite,
        "{{ [{'a': 1}, {'a': 2}] | rejectattr('a', 'eq', 2) | length }}",
        "1",
        "rejectattr",
    )
    _ok(
        suite,
        "{{ [{'a': 1}, {'a': 2}] | map(attribute='a') | join(',') }}",
        "1,2",
        "map over an attribute",
    )

    # `tojson` is the one filter `transformers` replaces, and what it replaces
    # it with keeps key order and does not escape above ASCII.
    _ok(
        suite,
        "{{ {'b': 1, 'a': 2} | tojson }}",
        '{"b": 1, "a": 2}',
        "tojson keeps the order the keys went in",
    )
    _ok(
        suite,
        "{{ {'k': 'caf\\u00e9'} | tojson }}",
        '{"k": "café"}',
        "tojson does not escape above ascii",
    )
    _ok(
        suite,
        "{{ {'a': 1} | tojson(indent=2) }}",
        '{\n  "a": 1\n}',
        "tojson with an indent",
    )
    _ok(suite, '{{ "a\\"b" | tojson }}', '"a\\"b"', "tojson escapes a quote")

    _ok(suite, "{{ 1 is odd }}", "True", "the odd test")
    _ok(suite, "{{ missing is defined }}", "False", "the defined test")
    _ok(suite, "{{ 'a' is string }}", "True", "the string test")
    _ok(suite, "{{ [] is iterable }}", "True", "the iterable test")
    # `is in(...)` names a test with a word that is otherwise an operator.
    _ok(suite, "{{ 1 is in([1, 2]) }}", "True", "the in test by name")

    # `items` and `dictsort` yield tuples, and a tuple prints with brackets.
    _ok(
        suite,
        "{% for pair in {'a': 1}.items() %}{{ pair }}{% endfor %}",
        "('a', 1)",
        "items yields tuples",
    )


def _statements(mut suite: Suite):
    suite.group("jinja/statements")

    _ok(suite, "{% if 1 %}a{% endif %}", "a", "if")
    _ok(suite, "{% if 0 %}a{% else %}b{% endif %}", "b", "else")
    _ok(
        suite,
        "{% if 0 %}a{% elif 1 %}b{% else %}c{% endif %}",
        "b",
        "elif",
    )
    _ok(
        suite,
        "{% for i in [1, 2] %}{{ i }}{% endfor %}",
        "12",
        "for over a list",
    )
    _ok(
        suite,
        "{% for i in [] %}{{ i }}{% else %}none{% endfor %}",
        "none",
        "the else on an empty loop",
    )
    _ok(
        suite,
        "{% for a, b in [[1, 2], [3, 4]] %}{{ a }}{{ b }}{% endfor %}",
        "1234",
        "unpacking in a for",
    )
    _ok(
        suite,
        (
            "{% for i in range(5) %}{% if i == 2 %}{% break %}{% endif %}"
            "{{ i }}{% endfor %}"
        ),
        "01",
        "break, which needs the loopcontrols extension",
    )
    _ok(
        suite,
        (
            "{% for i in range(4) %}{% if i % 2 %}{% continue %}{% endif %}"
            "{{ i }}{% endfor %}"
        ),
        "02",
        "continue",
    )

    _ok(
        suite,
        "{% for i in [1, 2, 3] %}{{ loop.index }}{{ loop.last }}{% endfor %}",
        "1False2False3True",
        "loop.index and loop.last",
    )
    _ok(
        suite,
        "{% for i in [1, 2] %}{{ loop.revindex }}{{ loop.length }}{% endfor %}",
        "2212",
        "loop.revindex and loop.length",
    )
    _ok(
        suite,
        "{% for i in [1, 2] %}{{ loop.previtem is defined }}{% endfor %}",
        "FalseTrue",
        "loop.previtem is undefined on the first pass",
    )

    _ok(suite, "{% set x = 2 %}{{ x }}", "2", "set")
    _ok(
        suite,
        "{% set x %}body{% endset %}{{ x }}",
        "body",
        "the block form of set",
    )
    # A frame is rebuilt every iteration, so a counter has to be a namespace.
    _ok(
        suite,
        (
            "{% set ns = namespace(n=0) %}{% for i in [1, 2] %}"
            "{% set ns.n = ns.n + i %}{% endfor %}{{ ns.n }}"
        ),
        "3",
        "a namespace survives the loop body",
    )
    _ok(
        suite,
        "{% macro greet(who) %}hi {{ who }}{% endmacro %}{{ greet('you') }}",
        "hi you",
        "a macro",
    )
    _ok(
        suite,
        (
            "{% macro wrap() %}[{{ caller() }}]{% endmacro %}"
            "{% call wrap() %}in{% endcall %}"
        ),
        "[in]",
        "call and caller",
    )
    _ok(
        suite,
        "{% filter upper %}ab{% endfilter %}",
        "AB",
        "a filter block",
    )

    _ok(suite, "{{ range(3) | list }}", "[0, 1, 2]", "range")
    _ok(suite, "{{ dict(a=1) }}", "{'a': 1}", "dict")
    _ok(
        suite,
        "{{ strftime_now('%Y') | length }}",
        "4",
        "strftime_now returns a formatted date",
    )


def _refusals(mut suite: Suite) raises:
    suite.group("jinja/refusals")

    # Each exclusion is a named error at compile time, which is model load
    # time. There is no loader, because a chat template is one string.
    _refused(suite, "{% include 'x' %}", "include", "include is refused")
    _refused(suite, "{% extends 'x' %}", "extends", "extends is refused")
    _refused(suite, "{% import 'x' as y %}", "import", "import is refused")
    _refused(suite, "{% from 'x' import y %}", "from", "from is refused")
    _refused(suite, "{% do x %}", "do", "do is refused")
    _refused(
        suite,
        "{% autoescape true %}{% endautoescape %}",
        "autoescape",
        "autoescape is refused",
    )

    # A refusal has to say where, because a template is a few thousand bytes
    # and the construct is one of them.
    try:
        var made = Template(String("a\nb\n{% include 'x' %}"))
        _ = made.digest
        suite.fail("a refusal carries a position", "compiled")
    except e:
        var message = String(e)
        suite.check(
            "line 3" in message and "^" in message,
            "a refusal carries a line and a caret",
        )

    _refused(suite, "{% if 1 %}", "endif", "an unclosed block is refused")
    _refused(suite, "{{ 1 + }}", "column", "a broken expression says where")

    # The sandbox hands out no method that would mutate a list, which is what
    # the reference does and the reason a template cannot build one in place.
    _render_fails(
        suite,
        "{% set r = [] %}{% set _ = r.append(1) %}",
        "refuses",
        "list mutation is refused",
    )

    _render_fails(
        suite,
        "{{ raise_exception('nope') }}",
        "nope",
        "raise_exception carries its message",
    )


def _limits(mut suite: Suite) raises:
    suite.group("jinja/limits")

    # A template is untrusted input from a model repository, so all four
    # budgets are checked here rather than trusted to the two that are easy.
    var steps = Limits()
    steps.steps = 1000
    try:
        var made = Template(
            String("{% for i in range(1000000) %}x{% endfor %}")
        )
        var got = made.render_object(NO_VARS, steps)
        suite.fail("the step budget stops a long loop", "rendered " + got)
    except e:
        suite.check("step" in String(e), "the step budget stops a long loop")

    var output = Limits()
    output.output = 64
    try:
        var made = Template(
            String("{% for i in range(1000) %}xxxx{% endfor %}")
        )
        var got = made.render_object(NO_VARS, output)
        suite.fail("the output cap stops a big render", "rendered " + got)
    except e:
        suite.check("bytes" in String(e), "the output cap stops a big render")

    var depth = Limits()
    depth.depth = 8
    try:
        var made = Template(
            String(
                "{% macro down(n) %}{{ down(n + 1) }}{% endmacro %}"
                "{{ down(0) }}"
            )
        )
        var got = made.render_object(NO_VARS, depth)
        suite.fail("the depth limit stops recursion", "rendered " + got)
    except e:
        suite.check(
            "deep" in String(e) or "depth" in String(e),
            "the depth limit stops recursion",
        )

    var deadline = Limits()
    deadline.deadline_ms = 1
    try:
        var made = Template(
            String("{% for i in range(10000000) %}x{% endfor %}")
        )
        var got = made.render_object(NO_VARS, deadline)
        suite.fail("the deadline stops a slow render", "rendered " + got)
    except e:
        var message = String(e)
        suite.check(
            "ran for longer" in message, "the deadline stops a slow render"
        )


def _api(mut suite: Suite) raises:
    suite.group("jinja/api")

    var chat = String(
        "{% for m in messages %}"
        "<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    var vars = String(
        '{"messages": [{"role": "user", "content": "hi"}],'
        ' "add_generation_prompt": true}'
    )
    _ok(
        suite,
        chat,
        vars,
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
        "a chat template with messages from json",
    )

    var made = Template(chat)
    var bindings = List[Binding]()
    bindings.append(
        json_binding(
            String("messages"),
            String('[{"role": "user", "content": "hi"}]'),
        )
    )
    bindings.append(flag_binding(String("add_generation_prompt"), False))
    bindings.append(text_binding(String("unused"), String("x")))
    try:
        var got = made.render(bindings)
        suite.check(
            got == "<|im_start|>user\nhi<|im_end|>\n",
            "bindings render the same as a json object",
        )
    except e:
        suite.fail("bindings render the same as a json object", String(e))

    # The same source twice is one compiled tree, because most Qwen forks ship
    # the same bytes and a digest is cheap next to a parse.
    var cache = Cache()
    try:
        var first = cache.compile(chat)
        var second = cache.compile(chat)
        var other = cache.compile(String("x"))
        suite.check(
            first == second and other != first and len(cache.templates) == 2,
            "the cache finds a template by the digest of its source",
        )
    except e:
        suite.fail("the cache finds a template by its digest", String(e))

    var one = Template(chat)
    var two = Template(chat)
    suite.check(
        one.digest == two.digest and one.digest.byte_length() == 64,
        "the digest is the sha256 of the source",
    )

    # Rendering twice has to give the same answer, which is not free: the
    # evaluator writes into a heap and a leftover binding would show up here.
    try:
        var a = one.render_object(vars)
        var b = one.render_object(vars)
        suite.check(a == b, "two renders of one template agree")
    except e:
        suite.fail("two renders of one template agree", String(e))


def run(mut suite: Suite) raises:
    _lexer(suite)
    _expressions(suite)
    _filters(suite)
    _statements(suite)
    _refusals(suite)
    _limits(suite)
    _api(suite)
