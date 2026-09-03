# A whole forward pass on the device

Issue #143. [kernels.md](kernels.md) is the individual device operations and what each one was measured against. This is the thing that runs them in order without leaving, and the reason it had to exist as its own piece of work rather than as a caller of the other one.

## The claim

A token goes in, a row of logits comes out, and nothing in between crosses back to the host. The residual stream is a device vector from the embedding lookup to the final norm, the keys and values are written where they are going to be read and never copied, and the only transfer a token pays for is the logits, which somebody has to look at to pick the next one.

Before this there were twenty one device operations, each one checked against a host reference on its own inputs, and no way to run two of them in a row without the numbers going back and forth. At decode shapes that trip costs more than the arithmetic on either side of it, so the kernels were correct and the machine was not faster.

## Why a kernel level test could not have caught this

A kernel that is right about arithmetic and wrong about which buffer it was handed passes every test that only looks at one kernel. So does a layer that norms in the wrong order, a key that is rotated at the wrong offset in the cache, a residual add that lands on a stale copy, and a trace that records the stream one operation late. Each of those produces fluent output rather than a crash.

So the test is the whole pass against the whole host pass, on the same bytes. `tests/test_gpu_block.mojo` builds a two layer model twice out of one blob of planar q8_0, once with host addresses and once copied into a real `DevicePool`, runs five tokens through both, and asks three questions. Do the logits agree, does greedy sampling pick the same token, and does the residual stream agree layer by layer rather than only at the end. The last one is what lets a future disagreement be named by layer instead of noticed at the logits, which is what the corpus in [logits.md](logits.md) will want from the device path.

The shapes are chosen so that the obvious wrong answers are visible. Four query heads over two key heads, because a group size of one cannot tell a correct key head index from a wrong one. Two layers, because one layer cannot tell a residual add that lands in the right place from one that does not. A vocabulary wider than the residual stream, so a transposed output head is an error rather than a shuffle.

## Refusals, not fallbacks

Every weight a device kernel reads has to be in a device pool and in the planar layout. Both are refusals. A device kernel handed a host address does not fault, it reads zeros, so a model that half fits would answer fluently from a stack where some of the layers saw nothing at all, and there is nothing in the output that says so.

The residency is checked against the plan before any bytes are read, so a model that does not fit on the card says so in about a second rather than after a minute of copying. The planar requirement is met rather than refused when it can be: a model with no repack cache beside it gets one written first, which costs a full read of the file and prints that it is doing it.

## Offsets are pointers, not arguments

Every kernel writes from element zero of the pointer it is handed. A key projection lands in the layer cache at `slot * kv_width`, a bias covers one run of a query, and a per head norm covers one head of a key where it already lies in the cache. All three are the same call, `DeviceVec.ptr_at`, and no kernel signature changed to allow any of them. The alternative was an offset parameter on every kernel plus a bound it would have to be told, which is two more numbers per launch that can be wrong and no more expressive.

## One context per process

A CUDA process gets one `DeviceContext`. Constructing a second one succeeds and then hangs on the first buffer allocated against it, with the card idle and every thread asleep in a futex wait, which reads like a kernel that will not finish and is not one. Metal does not mind, so this is a rule the fleet found and the laptop never would have.

So `DeviceSession` owns the one context and hands it down. `load` takes it as an argument rather than making its own, the pool allocates against it, the model's norm gains are uploaded through it, and every kernel is queued on it.

## What was run

`molla generate <model.gguf> <tokenizer.json> "<prompt>" [n] [ctx] [--device]`, greedy, the same prompt on every row, on an M4 and on a 4090. The flag took no value at the time, and the same runs today are `--device=auto` against `--device=cpu`.

| Model | Machine | Host | Device |
| --- | --- | --- | --- |
| SmolLM2 135M Q8_0 | M4 | 116 ms/token | 29 ms/token |
| SmolLM2 135M Q8_0 | 4090 | 108 ms/token | 3 ms/token |
| Qwen 2.5 0.5B Q4_K_M | M4 | 381 ms/token | 92 ms/token |
| Qwen 2.5 0.5B Q4_K_M | 4090 | | 4 ms/token |
| Llama 3.1 8B Q4_K_M | 4090 | 5681 ms/token | 12 ms/token |

Every pair produced the same text. SmolLM2 answered "Paris. Paris is the largest city in France and the capital of the French department" on all four runs, Qwen answered "Paris. It is the largest city in Europe and the second largest in the world" on all three, and the 8B answered "a city of grandeur and beauty, with a rich history and culture that is" on both, which is the acceptance criterion the issue asked for.

The host column is a scalar decode with no threading in it, so it is a floor rather than a fair opponent. What it is there for is that the two columns are the same text. The opponent that matters is llama.cpp and it is #146.

## What is checked

- `tests/test_gpu_block.mojo`, in `pixi run test`, on every machine that has a device. Six checks, and they run on the M4 and the 4090 with the same tolerance.
- The refusals run everywhere, including the machines with no accelerator, because the mistake they catch is a binding mistake and a binding mistake is not made by the GPU.
- `pixi run conformance-block` still covers the individual operations. This does not replace it: a whole pass agreeing is weaker per operation than an operation compared on its own inputs, and an operation agreeing on its own inputs says nothing about the order.

## What this is not

Not batched. A prefill is still one position at a time, because attention takes a single query and a run of keys by construction, so a batched prefill is a different kernel rather than a different loop.

Backend selection came after this, in #144, and is written up in [backend.md](backend.md). While this issue was open `--device` was a bare flag on `molla generate` and nothing else read it.

Not the fastest arrangement of these kernels. A token is one synchronize and several hundred launches, and the launches are small. Fusing them, keeping the residual stream in registers across a sublayer, and giving the matvec more than one row per block are all real work and all measurable, and none of it is worth doing before there is a number from a rival to aim at.
