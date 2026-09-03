# Choosing a backend

Issue #144. [device-decode.md](device-decode.md) is the forward pass that runs entirely on a card. This is the part that decides whether that pass is the one that runs, says so out loud, and refuses rather than quietly doing something slower.

## The claim

Every command that computes takes `--device`, all four of them agree on what the values mean, and the answer is printed before any tokens are. Asking for a backend that is not there is an error at startup rather than a fallback, and landing on the host when a card was hoped for is a sentence that says why.

## Why a fallback is the wrong default

A benchmark run against the wrong backend is worse than one that refused to start. The output of a host run and a device run is the same text, so there is no symptom other than the token rate, and a token rate is only wrong if you already know the right value. That is exactly the situation in #146, where the numbers get compared against llama.cpp, and a silent fallback would produce a table of honest looking numbers that are measuring the wrong thing.

So the two moods are separate. `--device=cuda` names a backend and is refused if that backend is absent, does not exist at that index, or the model does not fit on it. `--device=auto` asks for the best available and always answers, but when the answer is the host it carries the reason with it and prints it under the model line.

The first two of those refusals happen before the model file is opened, because they are answerable without it. That ordering is the difference between `--device=cuda` on a machine with no CUDA and a typo in the path reporting the typo, which is the wrong half of the problem to be told about first.

## Fitting is asked, not estimated

Whether a model fits is `plan_load` against that device with the real budget, and the answer is `left_behind == 0`. That is the same function that will place the tensors a minute later, so the decision and the load cannot disagree. An estimate here would have been a second copy of the reserve arithmetic and the two would have agreed until the day they did not.

The device forward pass needs every weight on the card, so half fitting does not count as fitting. That is stricter than what `molla load` allows on its own, and it is the rule the kernels impose: a device kernel handed a host address reads zeros without faulting.

Planning costs a header parse and two directory reads, so `choose_backend` opens the file and the repack cache a second time purely to ask. That second open is a few hundred microseconds against a load that is measured in seconds, and it buys one decision in one place instead of a half loaded state threaded through two entry points that differ in every other respect.

## A file the kernels cannot read is settled first

The device matvecs read the planar form of a quantized weight and nothing else, so an f16 or bf16 model has nothing on the card for them to read however much room there is. `device_refusal` asks that off the tensor directory, before fitting, because it is the more basic answer and a bigger card does not change it. A matrix counts and a norm does not, since a norm is one dimensional and is uploaded as floats whatever type it has.

It then splits the way every other question here splits. `auto` stays on the host and the reason is printed under the backend line, `--device=metal` is an error. Without it the failure arrives from inside the repack, as a cache that was written and still cannot be used, which is true and says nothing about the model that caused it.

`molla load` opts out of this one question, and it is the only caller that does. Copying an f16 tensor to a card is a perfectly good thing to time and that command launches no kernels, so what the matvecs support is not its business.

## What each command does with it

| Command | Default | What the flag does |
| --- | --- | --- |
| `molla generate` | `auto` | Picks the host or the device decode path |
| `molla serve` | `auto` | Same, and `/molla/version` reports which one is live |
| `molla load` | `auto` | Chooses the device the placement is planned against |
| `molla devices` | | Lists what the others can be asked for |

`molla generate` with no flag now runs on an accelerator when there is one the model fits on, which is a change: before this it was a host run unless `--device` was passed. `--device=cpu` is the way back to the old behaviour, and it is honoured even on a machine with a card.

## Two sessions, one of them present

`Runner` holds an `Optional[Decode]` and an `Optional[DeviceSession]` and exactly one of them is filled. A host server therefore allocates no device cache and a device server allocates no host cache, rather than both existing with one idle. Every call site goes through four small dispatch helpers, so the two paths cannot drift into having different reset or prefill semantics without the compiler noticing.

The context is made once, inside a function guarded on `has_accelerator`, for the reason in [device-decode.md](device-decode.md): a CUDA process gets one `DeviceContext` and the second one hangs on its first allocation.

## What is checked

`tests/test_backend.mojo`, in `pixi run test`, on every machine including the ones with no GPU. The flag parsing is machine independent, and the resolution is tested against device lists built in the test rather than against whatever is attached, so two cards of different sizes, a card the model does not fit on, and an api this fleet does not own are all covered on a laptop that has one of none of them.

What the fleet adds is the refusals under real enumeration. On the Linux box with no accelerator, `--device=cuda` fails at startup before the model file is opened, which is the case the tests can describe but not prove.

## What was run

On the M4, against SmolLM2 135M Q8_0.

| Asked | Answer |
| --- | --- |
| no flag | `metal Apple M4`, 28 ms/token |
| `--device=cpu` | `cpu host`, 89 ms/token |
| `--device=cuda` | `this machine has no cuda device. It has: cpu, metal` |
| `--device=metal:3` | `this machine has no metal device with index 3. The ones it has go up to 0` |
| `--device=vulkan` | `'vulkan' is not a backend molla knows` |

Each of the three errors is printed before the model file is opened. The server was run on both backends and answered a chat completion on each, with `/molla/version` naming the one it was on.

## What this is not

Not a way to split a model across two cards. `pick` returns one device, and a model that does not fit on the largest single accelerator is a host run with a note saying so.

Not a scheduler. The choice is made once at startup and is fixed for the life of the process, because the alternative is moving weights while a sequence is in flight and nothing here needs that.

Not a way to compile device code in. Whether this build can talk to a card at all is decided by the machine that ran the compiler, so `--device=metal` on a binary built without a GPU is an error that says exactly that rather than one about the hardware in front of you.
