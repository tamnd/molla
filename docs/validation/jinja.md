# Chat templates, byte identical to Python, in 60 microseconds

Issue #23 asks for a bounded Jinja2 subset rather than a general implementation, with every exclusion raising at model load time instead of at render time, execution limits because a template is untrusted input from the internet, and a gate: templates compile once and cache by content digest, and a 20 turn conversation renders in under 200 microseconds. This records what was built, what it was checked against, the four defects the corpus found that the unit tests did not, and the numbers.

## Why a template engine at all

D4 says model semantics come from the model's own artifacts. A chat template is the clearest case of that: the model author wrote the exact string their model was trained on, shipped it in `tokenizer_config.json`, and it is the one artifact that says where the system prompt goes, how a tool call is spelled, and whether a generation prompt ends with a newline. The alternative is a renderer per model family, maintained by us, wrong for every fork.

The cost is that we have to run their code. A chat template is Turing complete enough to loop forever and it arrives from a repository anybody can publish.

## What was built

Ten modules under `src/molla/jinja`, layered so nothing imports backwards. `diag` turns a byte offset into a line, a column and a snippet with a caret. `lexer` produces the whole token list up front. `ast` is a flat node list. `parser` builds it. Then `value` and `env` hold the render state, `strop` and `datefmt` and `access` and `filters` are the library, `eval` walks the tree, `template` is the public API, and `bench` is the command.

The engine is `{{ }}`, `{% %}`, `{# #}` and `{% raw %}`, with `trim_blocks`, `lstrip_blocks` and the explicit `-` and `+` markers. Statements are `if` and `elif` and `else`, `for` with the loop object and the `else` clause and `break` and `continue`, `set` in both the assignment and the block form, `macro`, `call`, and `filter` blocks. Expressions are the Python operator set with the Python precedences, inline `if`, slicing, attribute and item access, tests after `is`, and filters after `|`. There are 70 filters, 30 tests, and five globals: `range`, `dict`, `namespace`, `strftime_now` and `raise_exception`.

Seven constructs are excluded on purpose and each is a named error: `include`, `extends`, `import`, `from`, `do`, `autoescape` and the async forms. There is no template loader, because a chat template is one self contained string, so the first four have nothing to load from anyway.

## The reference

The answer key is what `transformers` builds, which is

```python
ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
```

with `tojson` replaced by `json.dumps(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)`, and `raise_exception` and `strftime_now` added as globals. Anything that environment prints is right by definition. A difference is a bug here even when our answer is the more defensible one, because the model was trained on their bytes.

Four of its behaviours are worth writing down because none of them is the default and all four changed the output of a real template. `keep_trailing_newline` is off, so one newline at the end of the source is not part of the template. `lstrip_blocks` strips only at a real line start, so the space in `{{ x }} {% if %}` survives. `x in undefined` is False rather than an error. And the sandbox refuses to hand out any method that would mutate a list, so `r.append(1)` raises rather than appending.

## The corpus

38 chat templates downloaded from the hub, nine conversation shapes each: plain, with a system message, multi turn, a long conversation, without a generation prompt, with tools, with a tool call and its result, with thinking on and with thinking off. Every render is compared for exact string equality against the reference environment, and a template that raises has to raise on both sides with the same message.

| | |
| --- | --- |
| Templates | 38 |
| Renders | 342 |
| Identical | 342 |
| Agreed refusals | 23 |
| Templates refused at compile time | 0 |

The 23 agreed refusals are the template's own `raise_exception` or a shared type error, mostly a template that will not accept tools or will not accept a conversation that starts with an assistant turn. Both sides refuse, so they count as agreement rather than as coverage.

The families are Qwen 2, 2.5, 3, QwQ and Coder, Llama 3.1 and 3.2, Mistral 7B and Nemo and Small, Mixtral, Gemma 2 and 3, Phi 3 and 3.5 and 4 and 4-mini, SmolLM2, OpenHermes, DeepSeek R1-Distill and V3, TinyLlama, Yi, InternLM, Granite, OLMo2, Zephyr, OpenChat, Dolphin, SOLAR, Starling, StarCoder2, GLM-4, Falcon and StableLM.

Alongside it there is a snippet suite of 149 cases, each one expression or statement, run through both implementations the same way. 148 are identical. The one that is not is `r.append(1)`, where both refuse and only the wording differs, since ours names the line and column and theirs names the attribute.

Both harnesses are the work of issue #24, which turns this into 300 repositories and puts it in CI. Until that lands these are run by hand and the numbers above are from the run on the merge commit.

## The four defects the corpus found

Every one of these was in code that passed its unit tests, and three of the four produced plausible looking output rather than an error.

A call was not a postfix. `content.split('</think>')[0]` stopped parsing at the closing bracket of the call, so Qwen3 and DeepSeek R1 refused to compile with a syntax error pointing at a `[` that is perfectly legal. This one at least failed loudly.

The evaluator lost the right operand of a binary node. `self.tree.nodes[node].b = self._math1()` takes an interior reference to the node list, then the call on the right appends to that same list and reallocates it, so the store lands in freed memory and `b` keeps whatever it had. Any binary operator far enough into a template to be past a reallocation silently evaluated to its left side, so `{{ 1 + 2 }}` at the top of a file printed 3 and the same expression at the bottom printed 1. All 34 sites in the parser now hoist the call into a local first.

`lstrip_blocks` stripped whitespace that was not indentation. The lexer only looked at the text since the last tag, so in `{{ x }} {% if %}` it saw a run consisting of one space, decided that was the indentation of a line, and ate it. It consults the source now to find out whether the line really started there.

`'citations' in controls` raised where the reference answers False. Granite asks that about a name that is only bound when the caller passes it, which is most of the time not.

## The limits

Four budgets, all on by default, all checked in one place. A step budget of 4000000, counted on every statement and every expression node, which is what stops `{% for i in range(10000000) %}`. An output cap of 16 MB. A recursion depth of 64, which is what stops a macro that calls itself. And a wall clock deadline of 2 seconds as the backstop, because the other three are counts and a count says nothing about how long a count takes. The clock is read every 1024 steps rather than every step, since reading it every step came out ahead of the actual work in the profile.

There is no filesystem access and no network access, because nothing in the engine can open either. The excluded constructs are the ones that could, and they are excluded at parse time.

`tests/test_jinja.mojo` asserts all four fire, and asserts each of the seven exclusions raises with its own name in the message and a line and a caret in the text.

## The gate

`molla template <template> <vars> [rounds]` compiles once and times the renders, which is what a server does. The conversation is 21 messages with a system prompt, tool definitions, a tool call and its result, and enough text in each turn to be realistic.

| M4, 20 turn conversation, 2000 rounds | Template | Prompt | Compile | Per render |
| --- | --- | --- | --- | --- |
| Llama 3.1 | 4614 bytes | 2649 bytes | 319 us | 59.9 us |
| Qwen3 | 4168 bytes | 2059 bytes | 253 us | 79.8 us |
| Mistral Small | 3959 bytes | 1617 bytes | 251 us | 85.4 us |
| Granite | 3543 bytes | 2563 bytes | 200 us | 43.4 us |

The gate is 200 microseconds and the slowest of the four is 85. The measurement is conservative in one way that matters: it parses the variables JSON on every round, because that is what a server does with a request body, so the JSON layer's time is inside the render number rather than beside it.

Compiling is 250 to 320 microseconds, which is why it happens once. `Cache` keys compiled trees by the SHA-256 of the source rather than by the model, since most Qwen forks ship the same bytes and a digest is cheaper than a parse.

There is no allocation number here, and that is deliberate. The engine allocates per render, into a heap that is thrown away at the end, and `AllocCounter` cannot see it because those allocations go through counter 0. Publishing a zero would read as a claim that is not true.

## Where it was run

| Machine | Suite | Corpus | Llama 3.1 per render |
| --- | --- | --- | --- |
| M4, macOS, arm64 | 1587 passed | 342/342 | 59.9 us |
| server1, EPYC, 4 cores, x86_64 | to fill in | to fill in | to fill in |
| server2, EPYC, 6 cores, x86_64 | to fill in | to fill in | to fill in |
| gpc, i9-13900K on WSL2, x86_64 | to fill in | to fill in | to fill in |

## What is still not covered

The corpus is 38 templates and issue #24 raises it to 300, which is where the refusal list starts being a backlog rather than an empty column. Nothing in these 38 uses a construct we refuse, which is a fact about the sample as much as about the engine.

`\N{...}` in a string literal is refused rather than decoded, since it needs the Unicode name table and no template in the corpus writes one. A non ASCII identifier is a syntax error, which Jinja allows and no chat template uses.

The engine is not incremental. A template renders to one string and there is no way to get the first half of a prompt while the second half is still rendering, which nobody needs at 60 microseconds.
