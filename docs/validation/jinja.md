# Chat templates, byte identical to Python, in 60 microseconds

Issue #23 asks for a bounded Jinja2 subset rather than a general implementation, with every exclusion raising at model load time instead of at render time, execution limits because a template is untrusted input from the internet, and a gate: templates compile once and cache by content digest, and a 20 turn conversation renders in under 200 microseconds. Issue #24 asks for the corpus that proves it, over hundreds of real repositories, gating every commit. This records what was built, what it was checked against, the five defects the corpus found that the unit tests did not, and the numbers.

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

Issue #24 asks for chat templates from at least 300 real repositories, a conversation matrix covering system prompts, tools, tool results, multi turn, images, thinking and the generation prompt both ways, exact string equality against Python, a record of every template we refuse and what construct it was, and a gate on every commit. This is that.

494 chat templates, one from each of 494 model repositories pinned by commit hash, rendered against 20 conversation shapes. The shapes are plain, without a generation prompt, with a system message, both of those without one, multi turn, a long conversation, an awkward one that starts with an assistant turn, with tools, with tools and no system message, with a tool call, with a tool call whose arguments are a JSON string rather than an object, with a tool result and no tool definitions, with an image, with several images across turns, with thinking on, with thinking off, with a thinking turn already in the history, with documents, with a date the template stamps into the prompt, and one that has all of it at once.

| | |
| --- | --- |
| Repositories | 494 |
| Distinct templates | 154 |
| Renders compared | 9500 |
| Identical | 9500 |
| Of which both sides refused | 750 |
| Templates refused at load time | 0 |
| Templates that differ on purpose | 19 |

The reference is `transformers.utils.chat_template_utils.render_jinja_template`, which is the function `apply_chat_template` calls to turn a conversation into a string. It is not a reimplementation of what `transformers` does, it is the thing itself, one call below the tokenizer so that no weights have to be downloaded to run it.

The 750 agreed refusals are the template's own `raise_exception` or a shared type error, mostly a template that will not accept tools, or will not accept an image, or will not accept a conversation that starts with an assistant turn. Both sides refuse the same case of the same template, so they count as agreement rather than as coverage. That is also why the numbers for the image cases are the highest: most models are text only and say so.

154 of the 494 templates are distinct by content digest, which is what the repository count hides. Most forks ship their parent's bytes, and a fork that changed one line is a different template with the same family name, which is exactly the case worth having. One repository per distinct template is the quick tier, for iterating locally; the full tier is all 494 and takes half a second, so that is what runs in CI.

Two things had to be pinned or the same input gives two answers. The clock, because a Llama template writes today's date into the system prompt, so both halves are handed 2025-01-01 00:00:00 and both run under `TZ=UTC`. And the shape of the variables, because `render_jinja_template` always binds `messages`, `tools`, `documents` and `add_generation_prompt` whether the caller passed them or not, so every case spells all four out with a null where it does not use one. A template asking `tools is defined` gets the same answer from both sides that way.

## The two we do not match

Nothing in 494 templates uses a construct the engine excludes, so the refusal list the issue asks for is empty and the `status` column says `ok` for everything except the two below. An empty refusal list is a fact about what model authors write rather than about the engine, and the column stays because the interesting direction is a construct appearing later.

There are two templates the reference and molla both render and do not agree on, and in both the reference is doing something that is Python rather than Jinja. They are marked `disputed` in the manifest, they are still compiled and still rendered so that a crash in one is still a failure, and the run prints them every time.

`ai-sage/GigaChat3.1-Audio-10B-A1.8B` calls `dict.from_keys(pairs)`, which is a typo for `dict.fromkeys`. Jinja's attribute lookup falls back to item lookup when the attribute is missing, so it evaluates `dict['from_keys']`, and in Python 3.9 and later subscripting a type gives a generic alias, which is callable and calls the type. The typo works by accident, and it builds the dict the author wanted. We answer that `dict` has no such member.

`zecanard/gemma-4-...` and its 17 quantisation siblings call `strftime_now('%Y-%m-%d %G:%i:%s')`. `%i` is not a `strftime` directive, Python hands the whole format to the platform, and what the platform writes for `%i` is a lone `i` on macOS and a literal `%i` on glibc. There is no single right answer to check against, so this one is excluded rather than picked. We raise on a directive we do not implement, which is the documented behaviour and the one that does not silently put a percent sign in a prompt.

## In CI

`Template conformance` runs on every commit, on the full tier, and is one of the jobs the required `CI OK` check waits for, so a mismatch blocks the merge. It fetches the 494 templates, checks every digest against the manifest, and runs `pixi run conformance-template-full`. The corpus directory is cached on the manifest hash, so an ordinary commit restores four megabytes rather than downloading it.

There is no Python in that job. The manifest carries the sha256 of the reference answer for every repository, which is what the Mojo side compares against, so the everyday check is one binary reading a directory of text files. `scripts/check-template.py` is the half that needs `transformers`, and it is run when the reference version moves, with `--refresh` to write the digests back.

## The five defects the corpus found

Every one of these was in code that passed its unit tests, and four of the five produced plausible looking output rather than an error.

A call was not a postfix. `content.split('</think>')[0]` stopped parsing at the closing bracket of the call, so Qwen3 and DeepSeek R1 refused to compile with a syntax error pointing at a `[` that is perfectly legal. This one at least failed loudly.

The evaluator lost the right operand of a binary node. `self.tree.nodes[node].b = self._math1()` takes an interior reference to the node list, then the call on the right appends to that same list and reallocates it, so the store lands in freed memory and `b` keeps whatever it had. Any binary operator far enough into a template to be past a reallocation silently evaluated to its left side, so `{{ 1 + 2 }}` at the top of a file printed 3 and the same expression at the bottom printed 1. All 34 sites in the parser now hoist the call into a local first.

`lstrip_blocks` stripped whitespace that was not indentation. The lexer only looked at the text since the last tag, so in `{{ x }} {% if %}` it saw a run consisting of one space, decided that was the indentation of a line, and ate it. It consults the source now to find out whether the line really started there.

`'citations' in controls` raised where the reference answers False. Granite asks that about a name that is only bound when the caller passes it, which is most of the time not.

`tojson` ignored every argument except `indent`. Kimi K2 and three others write `tojson(separators=(',', ':'))` to get the compact spelling, and got the spaced one, so their tool definitions went into the prompt with a space after every comma. The filter takes the signature transformers gives it now, which is `ensure_ascii`, `indent`, `separators` and `sort_keys` in that order, and the first positional argument is `ensure_ascii` rather than `indent`, which is worth knowing because it is not the order anybody guesses.

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

| Machine | Suite | Corpus | Llama 3.1 | Qwen3 | Mistral Small | Granite |
| --- | --- | --- | --- | --- | --- | --- |
| M4, macOS, arm64 | 1587 passed | 342/342 | 59.9 us | 79.8 us | 85.4 us | 43.4 us |
| gpc, i9-13900K on WSL2, x86_64 | 1588 passed, 1 failed | 342/342 | 41.5 us | 61.5 us | 63.4 us | 27.6 us |
| server1, EPYC, 4 cores, x86_64 | 1589 passed | 342/342 | 130.1 us | 188.0 us | 199.9 us | 101.0 us |
| server2, EPYC, 6 cores, x86_64 | 1589 passed | 342/342 | 193.9 us | 511.6 us | 689.3 us | 177.7 us |

The one failure on gpc is issue #87, the reactor backpressure test under WSL2, which fails on main in the same way and has nothing to do with this work. The Linux machines count two checks more than the M4 for reasons that predate this work.

The corpus column on the three Linux machines is a replay rather than a fresh differential run. The reference output was recorded on the M4, where it is byte identical to Python, and each machine renders the same 342 cases and compares against that recording. It is the same assertion with the Python half cached, and what it is really checking is that nothing in the engine depends on the platform.

Both EPYC machines were carrying a load average above 9 on 6 cores and above 38 on 4 cores throughout, from unrelated work belonging to whoever else is on them, so their render numbers are a ceiling on the time and not a measurement. Qwen3 and Mistral Small on server2 are past the gate on those readings, at 512 and 689 microseconds, and I am not going to claim they would pass on an idle machine without having seen an idle machine. What the two rows do establish is that the answers are the same everywhere, which is what they were run for.

## What is still not covered

The corpus is 494 templates that were on the hub in September 2026, weighted towards what people actually download, and it is a sample rather than the population. A construct nobody writes today is a construct the corpus cannot tell us about.

The conversation matrix is 20 shapes and it is hand written. It covers what the issue asks for and what the templates in front of us branch on, but a template that branches on something none of the 20 shapes carry gets its other branch rendered by nobody.

`\N{...}` in a string literal is refused rather than decoded, since it needs the Unicode name table and no template in the corpus writes one. A non ASCII identifier is a syntax error, which Jinja allows and no chat template uses.

The engine is not incremental. A template renders to one string and there is no way to get the first half of a prompt while the second half is still rendering, which nobody needs at 60 microseconds.
