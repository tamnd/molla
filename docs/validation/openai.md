# The OpenAI routes

Issue #29 asks for `/v1/chat/completions`, `/v1/completions` and `/v1/models`, streaming and not, with the OpenAI Python SDK as the thing that has to work. This records what was built, what was deliberately refused, the three bugs that were designed out rather than found, and what it actually does on a laptop.

## What was built

`molla.api.openai` is the wire format and nothing else. It parses a request body out of a `Document`, it writes a response into a `Writer`, and it has no idea what a model is. Everything under `molla.api` is somebody else's API written down in Mojo, which is decision D2, and the practical effect is that this is the only file that changes when a provider adds a field.

`molla.engine.runner` is one loaded model plus the state of the one request using it. It owns the GGUF mapping, the bound weights, the tokenizer, the compiled chat template, the session, the sampler and the incremental detokenizer. The protocol reaches it by address, which is the same arrangement the logger and the metrics view have been in since M1 and works for the same reason: the runner is a local of `run_serve`, the server is started and stopped inside that function, and nothing holding the address outlives it.

`molla.http.protocol` gained four route ids and about six hundred lines of handler. `molla.engine.serve` is `molla serve`, which is thirty lines of wiring and a lot of printing.

```console
molla serve ~/models/SmolLM2-135M-Instruct-Q8_0.gguf ~/models/tok/smollm2.json --port=8000
```

`--host`, `--port` and `--ctx` are the flags, and it binds 127.0.0.1:8000 when nothing says otherwise. The host has to be a dotted quad. There is no resolver on this path and there is not going to be one, because a listen address is a local interface and a name for a local interface is a way to bind the wrong one.

## One sequence at a time

The server runs one worker. There is one model, one session and one sampler behind these routes, so a second worker would be a second thread contending for a lock that does not exist. A second request arriving while one is running gets a 503 that says so, rather than being queued or interleaved.

That is a worse server than the one M3 will have, and it is an honest one. Queueing without a scheduler means a client waiting on a socket with no idea it is second. Interleaving without paging means two sequences writing over each other's cache, which does not crash: it produces fluent output that has forgotten the instruction, and nobody reads that as an error. The scheduler is issue #31 and it is the thing that turns the runner into a list.

The cost is visible in one place. A request that is not streaming holds the worker for its whole generation, so a health check behind a sixty token completion waits seven seconds for it. A streaming request goes back to the event loop between tokens, so `/molla/health` answers through one in about the time one token takes, which is checked by hand below.

## What is refused, and why refusing is the feature

The issue says not to stub fields in a way that reports success, and that turned out to be the design constraint that shaped the parser. A field that is accepted and ignored is worse than a 400, because the client gets a plausible answer and no way to tell that half its request was thrown away. `tools` accepted and ignored looks exactly like the model deciding not to call one.

So these are 400s, each naming the field:

| Field | Route | Why |
| --- | --- | --- |
| `tools`, `functions`, `tool_choice` | chat | M4 |
| `response_format` | chat | M4, the same grammar work |
| `logprobs`, `top_logprobs` | both | nothing on this path computes them |
| `n` above 1 | both | one sequence at a time |
| `best_of` | completions | the same |
| `suffix` | completions | no fill in the middle |

`logprobs: false` is not a refusal, because false is not asking for anything and a client library that sends it by default would otherwise be unable to talk to this server at all. `n: 1` is likewise fine, since it is what happens anyway.

Sampling settings are checked against OpenAI's documented ranges before they reach `SamplerConfig.check()`, so a `temperature` of 5 is a 400 rather than an argument about whose range is right. The llama.cpp extensions molla's sampler already has, which are `top_k`, `min_p`, `typical_p`, `repeat_penalty` and `repeat_last_n`, are read when they are present and absent from every response, since they are not part of the format.

A request naming a model this server did not load is a 404. The whole reference matches, which is what a client that read `/v1/models` sends back, and the last path segment matches too, because typing a home directory into a curl command to be told the server has no model is nobody's idea of a good time. Nothing else matches. A server with one model that answers to any name is a server that lies about which model produced the answer.

## Three bugs that were designed out

None of these was found by the compiler and one of them would not have been found by a test either.

**A stop string can straddle two tokens.** The generated text is kept whole in the runner, and what may go out to the client is the part of it that cannot still turn out to be the beginning of a stop string. So a token is sometimes produced and nothing is emitted for it, which is correct, and it is why the streaming loop asks for a delta rather than assuming one token is one chunk. Searching the finished text afterwards would work for a whole response and would already have sent the first half of the stop string in a streaming one.

**An event that does not fit must not be lost.** The stand in streaming routes advance their event counter before they know the event fits, which is harmless when the payload is a demo number and loses a token when it is not. The API path holds the built chunk in `api_delta` and only clears it once `stream.event` returns OK, and a stage transition happens only after a successful write.

**A stream that never yields makes the busy path unreachable.** `_pump_stream` produces until the ring is full, and on a loopback socket the ring is never full, so an entire completion would run inside one turn of the service loop and nothing else on that worker would be looked at until it finished. That is not a throughput problem, it is a correctness one: the 503 that a second request is supposed to get cannot be sent by a worker that is not being serviced. An API stream now produces one token and hands back, and it is applied only to API streams so the demo, soak, drain and allocation behaviour is exactly what it was.

Handing back once per token is not by itself enough, and that is the part the fleet found rather than the laptop. The reactor gives a connection eight rounds of read, produce and write per pass, which is the right budget when a round is a memcpy and the wrong one when a round is a forward pass through a language model, so a stream that returned after one token was simply asked for another seven before anything else on the worker was looked at. `Connection.yield_now` is the fix: a protocol that has just done something expensive sets it, the reactor ends the round loop rather than the round, and the slot goes on the same `again` list that a spent round budget uses, so the next pass comes straight back to it without blocking. What a request arriving mid stream now waits for is one token and one poll.

| Health check behind a 200 token stream | Steady state |
| --- | --- |
| Four tokens per hand back, eight rounds per pass | 1.8 to 2.1 s |
| One token per hand back, eight rounds per pass | 0.3 to 0.9 s |
| One token per hand back, yield ends the pass | 0.13 to 0.22 s |

The last row is one token time, which is the floor: a token is not interruptible and a request that arrives during one waits for it. The same is true of the prefill, and the prefill is the one visible remaining case, since a health check issued while a thirty six token prompt is being read waits the two or three seconds that takes. Total stream time is unchanged across all three rows, so none of this cost throughput.

## Streaming

Through the `StreamWriter` that has been there since M1. `event(name, data, id, now)` with an empty name and an empty id emits `data: {...}\n\n`, which is OpenAI's framing exactly, so there was nothing to add. The terminator is a literal `[DONE]` event and the chunked trailer is `end(now)`.

Chat streams three shapes, which are the three the specification has: a first chunk carrying the role and an empty content, middle chunks carrying text and a null `finish_reason`, and a last chunk carrying an empty delta, the reason, and the usage. Completions streams `text_completion` objects and not `text_completion.chunk`, which reads like an oversight in the specification and is what the clients check for.

`finish_reason` is written as null on every chunk that is not the last, rather than left out. An SDK reads the key, and a missing key and a null are not the same thing to a strongly typed client.

## What it does

macOS arm64, M4, SmolLM2 135M at q8_0, one client.

| | Prompt | Generated | Wall | Per token |
| --- | --- | --- | --- | --- |
| `/v1/completions` | 5 | 64 | 7.10s | 104ms |
| `/v1/chat/completions` | 36 | 64 | 10.92s | 120ms |

Both numbers include the prefill, which is a token at a time at the same cost as a decode, so the chat row is thirty one extra prompt tokens rather than a slower decode. That is the same 90 to 120 ms a token `molla generate` reports and it is what issue #120 exists to change. Nothing in the request path is measurable next to it: the parse, the template render and the response build together are under a millisecond, and the server does not allocate per request outside the model, which `molla allocs` still reports as zero.

## The done criterion, by hand

The OpenAI Python SDK 2.45.0, against `molla serve`, no molla specific configuration beyond `base_url`:

```console
$ python3 sdk_check.py
models: ['/Users/apple/models/SmolLM2-135M-Instruct-Q8_0.gguf']
chat: 'One colour is indigo.' stop 39
stream: 'Another colour is blue.' stop 59
text: ' 4 5'
text stream: ' d e f g'
bad model: NotFoundError "no model named 'nope' is loaded, this server has ..."
tools: BadRequestError 'tools are not imple...'
```

The second line of the conversation is streamed and carries the first line's answer back in `messages`, so the chat template is rendering an assistant turn as well as a user one. The last two lines are the error envelope doing its job: the SDK raises `NotFoundError` and `BadRequestError` rather than a generic `APIError`, which is what says the status code and the `{"error": {...}}` shape are both right.

The busy path and the event loop, by hand, with a two hundred token stream running:

```console
$ curl -s -w " %{http_code}\n" $U/molla/health
ok
 200
$ curl -s -w " %{http_code}\n" $U/v1/completions -d '{"prompt":"x","max_tokens":2}'
{"error":{"message":"this build decodes one sequence at a time and one is already running, ...","type":"server_error","param":null,"code":null}} 503
```

## What is tested and what is not

`tests/test_api.mojo` covers the parsing, the refusals, the response and chunk builders, and the route dispatch. None of it needs a model file: the parsers and writers are pure functions over bytes, and the routes are driven on a reactor against a protocol with no engine configured, which answers 503 and thereby proves the target resolved, the method check ran, and the handler reached is the API handler. A 404 there would have meant the route table was wrong and a 503 means it is right, which is everything about the dispatch that loading a model would not tell us more about.

The response tests read the bytes back and look for keys rather than comparing against a fixed string, because key order is not something the specification says anything about and a test of key order is a test that fails on the next refactor for no reason.

What is not tested in the suite is a real generation through the routes, because that needs a model file and the suite runs on machines that have none. That is what the by hand section above is for, and it is run on every platform before a release.
