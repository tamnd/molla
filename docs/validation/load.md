# Weight loading and device placement

Issue #25 asks for tensors to get from a mapped file into a form the kernels want, without a copy where a copy is avoidable, and it is done when an 8B q4_K_M model loads on the M4 and on the 4090 with progress reported per stage. Both halves are below. This records what was built, what the two machines actually do, and the four things that were wrong before it worked.

## What a load is

A load is two things happening: bytes coming off disk into the page cache, and bytes going from the page cache to wherever the kernels will read them. Doing them one after the other costs the sum of the two. Doing them at the same time costs the larger of the two, and on a discrete card those two numbers are close enough that the overlap is most of the win.

So `molla.model.load` runs a pool of transfer threads over the mapping while the thread that owns the device drains a queue of finished tensors and enqueues their copies. A worker claims a tensor with one atomic add, calls `madvise(MADV_WILLNEED)` on its range, touches one byte per page to make sure the fault actually happened, and pushes the index onto an `MpscQueue`. The drain thread pops indices and enqueues a device copy for each one as it arrives, so the card starts moving the first tensor while the pool is still reading the second.

The queue is the whole design. Without it the two stages need a barrier between them, and with it neither stage ever waits for the other to finish, only for the next single tensor.

## Placement is a decision, not a special case

Every tensor gets one of three placements, and the choice is made once in `plan_load` before any byte moves.

| Placement | What it means | When |
| --- | --- | --- |
| `host` | The tensor is the mapping. No copy at all. | No accelerator, or the device ran out of budget. |
| `unified` | The tensor is the mapping, on a machine whose accelerator draws from the same physical memory. No copy at all. | Apple silicon, when the kernels are going to run on the host. |
| `device` | The tensor is copied into a slot in one device pool buffer. | Any accelerator with room for it, unified or not. |

`unified` is not `host` with a different name. They do the same amount of work, which is none, and the difference is what that none is worth. A host placement on a machine with an accelerator means the accelerator is not being used for those bytes. A unified placement means the bytes are already in the memory the accelerator draws from, so a pool the plan asks for costs nothing to fill from anywhere else. That is a fact about the budget and not about any address, which is the correction the last section of this document goes into.

What decides between `unified` and `device` on a machine that could do either is where the kernels are going to read from, and not what kind of memory the machine has. That is the one thing about this table that changed after the first device kernel ran. Sharing physical memory does not share an address space, so a weight a device kernel is going to read has to be in a pool on an M4 exactly as it does on a 4090.

## A weight knows which memory it is in

That question used to have no answer anywhere in the type system. A placement was a decision the loader made and then forgot, and every tensor that came out of `bind` was a host address whether or not a copy to a card had happened, so a host kernel handed a device weight would have dereferenced a driver handle and faulted on a thread with no idea a placement had ever taken place.

So a `Tensor` carries its place, and `Tensor.base`, which is the one call every host kernel makes to get at bytes, refuses a weight in a device pool and names its shape. The cost is an integer compare once per matvec rather than once per row, because a weight is in one memory for its whole life and the test is hoisted out of the row loop the same way the layout test already is. `Tensor.device_address` is the mirror of it and refuses everything that is not in a device pool, unified included, for a reason the last section of this document goes into.

What connects the two halves is `Residency`, which is what a load hands the binder: for each position in the file's tensor directory, which memory that tensor ended up in and where on the device if it moved. It is indexed by directory position rather than by name because that is what the binder already has after it looks a weight up and what a placement already carries, so the two meet without either growing a table of strings. An empty `Residency` answers host for everything, which is what every caller that loads with a device budget of zero gets, and that is still every caller that generates a token until [#143](https://github.com/tamnd/molla/issues/143) lands.

## The plan reads the copy that will be read

A load with a warm repack cache used to fault in the model file while `bind` pointed the kernels at the cache beside it. Every byte of that read was wasted and the bytes that were going to be read stayed cold.

`plan_load` can now be told about the cache, and then it plans against the copy of each weight that `bind` is going to use. A tensor the cache has is sourced from the cache in the planar layout, a tensor with no planar form is sourced from the file, and the same rule decides what the read stage warms and what the copy stage sends to the card. Which means a device pool holds the layout the device kernels want on a machine that has a cache, and the ggml layout on one that does not, and the tensor says which without anybody going back to the file to ask.

A placement therefore carries two sizes. `bytes` is what the tensor occupies in the model file and stays on record because the repack reads from there. `length` is the source that is actually read, which is larger for a quantized weight, because the planar layout trades size for a kernel that does not have to walk a block header per group. On the 8B that is 4685 MiB in the file and 9573 MiB read.

The load counts the bytes it copied to the device and refuses to finish if that disagrees with what the plan placed. They are computed from the same placements a few dozen lines apart, so the only way they differ is a drain loop that skipped a tensor or took one twice, and a card that is missing one weight out of 292 generates text that is almost right, which is the worst failure in the file.

The device side is one pool allocation rather than one allocation per tensor. 292 allocations of a few megabytes each is 292 driver round trips and a fragmented heap for no benefit, since the lifetime of every tensor is the lifetime of the model. Each tensor gets a byte offset into the pool, aligned to 256, and the test suite checks that slots are aligned, in order, and never overlap. Getting that wrong writes one tensor over another and the model still loads and still produces words, just wrong ones, so it is worth an assertion rather than a code review.

When the model does not fit, the planner runs in two passes. Anything read once per token goes to the card first, and the token embedding goes last, because the embedding is the largest single tensor in a q4_K_M 8B and it is read once per token for one row. Trading the whole embedding for four more attention blocks on the card is the right trade and the planner makes it without being asked.

## What the two machines do

Both runs are the same file, `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`, 4920739232 bytes, 292 tensors, 4685 MiB of tensor data.

| | macbook | gpc |
| --- | --- | --- |
| Built for | `metal:4` | `nvidia:sm_89` |
| Device | Apple M4, 16384 MiB, unified | RTX 4090, 21760 MiB, its own memory |
| Placement | 4685 MiB unified | 4685 MiB to the device pool, 0 left on the host |
| Budget | n/a | 19584 MiB |
| Workers | 5 | 8 |
| Read | 3746 ms, 1250 MiB/s | 486 ms, 9639 MiB/s |
| Copy | nothing to copy | 473 ms, 9904 MiB/s |
| Total | 3746 ms | 499 ms |

The M4 run, which is a host placement and is what `molla load --host` prints today. The wording of the second line has changed since this was taken, because the claim it made about a device reading the mapping turned out to be false, and the numbers have not.

```console
plan   292 tensors, 4685 MiB, metal Apple M4
plan   4685 MiB unified, which the device reads from the mapping
read   10%  397 MiB  2349 MiB/s
read   50%  2232 MiB  1214 MiB/s
read   100%  4685 MiB  1250 MiB/s
read   4685 MiB on 5 threads in 3746 ms, 1250 MiB/s
copy   nothing to copy, the weights are read where they lie
done   292 tensors in 3746 ms
```

The gpc run:

```console
plan   292 tensors, 4685 MiB, cuda NVIDIA GeForce RTX 4090
plan   4685 MiB to the device pool, 0 MiB left on the host, budget 19584 MiB
read   10%  4685 MiB  36601 MiB/s
read   50%  4685 MiB  16732 MiB/s
read   100%  4685 MiB  9639 MiB/s
read   4685 MiB on 8 threads in 486 ms, 9639 MiB/s
copy   4685 MiB to NVIDIA GeForce RTX 4090 in 473 ms, 9904 MiB/s
done   292 tensors in 499 ms
```

The two numbers are not comparable and it would be dishonest to present them as a comparison. The M4 read is off an SSD with a cold page cache and it is disk bound at about 1250 MiB/s. The gpc read is out of a warm page cache on a 31 GB box, so 9639 MiB/s is a memory bandwidth number, not a storage one. What the gpc run does show is the thing worth showing, which is the overlap.

**The gpc progress lines are the overlap, printed.** The percentage counts tensors drained and the MiB column counts bytes faulted in, and on gpc the byte column is already at 4685 MiB when the tensor column says 10 per cent. That is eight workers finishing the whole read in roughly a tenth of a second while the drain thread is still handing tensors to the driver one at a time. The read stage is not waiting on the copy stage and the copy stage is not waiting on the read stage, which is what the queue is for. The declining MiB/s figure on those lines is the same finished byte count divided by a growing elapsed time, so it decays toward the honest average rather than reporting anything new.

The 486 ms read and the 473 ms copy overlap almost completely, which is why the total is 499 ms and not 959 ms.

## What the pool holds once the cache is warm

Both runs above are a first run against a model, when there is no repack cache and the plan has nothing to read but the file. What residency changed is the second run, so both machines were run again on the same 8B after the cache had been written. The progress lines are cut from both.

The M4:

```console
repack hit, 226 tensors, 9572 MiB already repacked
plan   292 tensors, 4685 MiB, metal Apple M4
plan   9573 MiB unified, which the device reads from the mapping
copy   nothing to copy, the weights are read where they lie
repack nothing was repacked on this load
done   292 tensors in 3700 ms
```

gpc:

```console
repack hit, 226 tensors, 9572 MiB already repacked
plan   292 tensors, 4685 MiB, cuda NVIDIA GeForce RTX 4090
plan   9573 MiB to the device pool, 0 MiB left on the host, budget 19584 MiB
read   9573 MiB on 8 threads in 926 ms, 10338 MiB/s
copy   9573 MiB to NVIDIA GeForce RTX 4090 in 919 ms, 10416 MiB/s
copy   292 tensors resident in a 9573 MiB pool
repack nothing was repacked on this load
done   292 tensors in 954 ms
```

The two plan lines are the point. The first counts the file, which has not changed, and the second counts what is going to be read. The difference between 4685 and 9573 is the planar layout the repack wrote. Before this the planner knew only the first number, so the read stage faulted in the model file while the binder pointed the kernels at the cache beside it, and all 4685 MiB of that was work nobody used.

On the M4 the second number costs nothing to satisfy: 9573 MiB of weights ended up in the memory the GPU draws from with no allocation and no copy anywhere in the run. That is the saving unified names and it is the part that held up. What did not hold up is the sentence that used to follow it here, about a Metal kernel reading those bytes where they lie, and the last section is what replaced it and what a device load on the M4 costs instead. On the 4090 it costs a 9573 MiB pool, and the plan line, the copy line and the resident line are three separate counts of the same quantity, arrived at from the placements at three different points in the load, which the loader compares before it will report success.

The cold half of the gpc run is worth recording too, because the table above is a warm page cache and says so. The same 8B read for the first time after landing on the disk took 3551 ms at 1319 MiB/s, and the copy behind it took 695 ms. That 1319 MiB/s is the storage number for that box and the 9639 MiB/s in the table is not.

## Four things that were wrong first

**The job struct freed itself while eight threads were still writing to it.** This is the same Mojo lifetime hazard documented in `docs/validation/threading.md`, met for the second time. The drain loop ends when the last tensor is popped, which happens before the worker that pushed it has come back around its own loop. The drain loop was the last visible use of the job struct, so Mojo destroyed the job, freed the atomics and the queue inside it, and left live workers incrementing memory the allocator had handed to something else. It surfaced as `index 245249457586176 is out of bounds, valid range is 0 to 5`, which is a claim counter read back as a stale heap pointer, reported a long way from the cause. `molla.sys.mem.keep(job)` after the joins is the fix and there is a comment on it saying why.

**max-core refuses to compile on a machine with no GPU.** Three of the five boxes have no accelerator, and the first version of the device pool took `pixi run build` down on all three with `constraint failed: Unknown GPU architecture detected`. The constraint fires when a function is instantiated and not when a struct is declared, so a struct with `DeviceContext` and `DeviceBuffer` fields is fine on a GPU free box as long as nothing calls its `__init__`. Every call into the device path is behind `comptime if has_accelerator()`. This is worth stating plainly: accelerator support is decided when you compile, not when you run, and a molla binary built on a CPU only box has no device code in it.

**Device free memory cannot be used as a budget.** Under WSL2 the 4090 reports free and total as the same number, 21760 MiB, while `nvidia-smi` on the Windows side says 24564 MiB total with 50 MiB in use. Neither figure is one to size a pool against. So `device_budget` works off total minus a reserve, where the reserve is a tenth of the card or 512 MiB, whichever is larger, and a card smaller than its own reserve gets a budget of zero rather than a negative one. That gives 19584 MiB on the 4090, which is conservative and, more to the point, is the same number on every run.

**Page size is not 4096.** It is 16384 on Apple silicon. A loop that touches one byte per page with the wrong constant does four times too much work or reads a quarter of the pages, and `madvise` rejects an address that is not page aligned outright. `molla.sys.mmap.page_size` asks `sysconf`, which needs `_SC_PAGESIZE`, which is 29 on macOS and 30 on Linux. That is two different numbers for the same constant and it is the kind of thing that works on the machine you wrote it on.

## What the tests can and cannot check

`tests/test_load.mojo` is 58 checks over a six tensor fixture. Placement is arithmetic and it is tested against made up devices, because the interesting cases are a card that is too small and one that is unified, and no machine in the fleet is both. A `Device` is a plain struct with public fields, so a 2 GB discrete card and a 24 GB one are two lines of setup rather than two machines.

The transfer pool runs against a real file on real threads and what it checks is conservation. Every tensor comes back exactly once, the bytes the workers faulted in add up to the bytes the plan said were there, and one worker loads the same thing as four. A load that drops a tensor or claims one twice fails there rather than in a kernel that gets the wrong weights and still produces plausible words.

The device copy is checked on the two machines that have somewhere to copy to. `tests/test_gpu.mojo` allocates a real `DevicePool`, puts two tensors in it at the slots the plan would have chosen, and reads both back with a kernel, so the pool arithmetic and the asynchronous copy meet the same way they do in a load. Two tensors and not one, because a pool holding a single tensor cannot tell a correct base address from a correct offset. On the three machines with no accelerator the group reports a skip, since a build made there has no device code in it at all.

## The repack rides along on this pool

The fifth thing #25 asked for was that the ggml block layout is rewritten once at load into the layout a kernel wants, cached on disk. That was split into #120 and held until #26 had defined a destination, since writing a cache format against a layout nobody had chosen yet would have meant a cache that loads fine and answers badly the first time the layout moved.

It landed on this pool rather than beside it. A worker that has just faulted a tensor's pages in is holding the warmest copy of those bytes there will ever be, so that is the moment to transform them, and the alternative of a second pass after the load reads four gigabytes twice. What a worker does with the result is one `pwrite` per few megabytes at an offset decided before any thread started, so the repack adds no coordination between workers and none with the drain thread. A repack that fails records an errno in one atomic slot, the load finishes normally, and the half written cache is thrown away.

That is why there is a fourth stage in `stage_name` and a `repack` line in the report. See [repack.md](repack.md) for the layout, the cache key and the eight ways a cache is refused.

## Reproducing

```console
pixi run build
./build/molla devices
./build/molla load ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
./build/molla load ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --host
```

`molla load` takes an optional worker count as an argument. The default is half the cores capped at eight, which is a floor to keep a four thread box from spawning two threads and a ceiling because past eight the transfer pool is waiting on the storage device rather than on itself.

`--host` leaves every weight in the mapping and asks for no pool, which is what a load for host kernels costs. Without it the accelerator gets whatever fits, which is what a load for device kernels costs. On a machine with no accelerator the two are the same run.

## What a unified weight cannot do

The paragraph that used to be at the top of this document said that a unified weight passes both `base` and `device_address`, and that a device kernel reads the mapping where it lies with no copy between. The first device kernel, in [#141](https://github.com/tamnd/molla/issues/141), found that the second half of that is false, so this section replaces it and the tracking issue is [#152](https://github.com/tamnd/molla/issues/152).

It is at the end rather than the top because it is a correction and the record of what it cost to correct, and burying the measurements above it under it would be putting the apology before the work.

Probed on the M4 through max-core 26.5.0.

A Metal kernel handed a plain host allocation does not fault. It reads zeros. Every row of the output comes back exactly zero and nothing anywhere reports an error. Wrapping the host memory in a `DeviceBuffer` with `owning=False` does not change it: the pointer value is identical and it still reads zeros.

The other half of the probe explains why. A device buffer's `unsafe_ptr()` is a device address, 1099512152064 in the run that was recorded, and reading it from the host segfaults. `map_to_host()` on the same buffer returns 4470079744, a real host address for the same bytes, with no copy anywhere between the two. So the memory is shared and the address space is not. One physical pool, two different addresses into it, and a pointer from one side means nothing to the other.

Nothing in the measured part of this document changes. A unified plan still allocates nothing and copies nothing, the M4 still loads 9573 MiB with no transfer, and every number in the tables above was taken before this was known and is unaffected by it, because none of them ever dereferenced anything.

What changes is what the placement promises. `unified` means that the weights are in the memory the accelerator draws from, which is a statement about what a pool would cost rather than about what a kernel can be handed, and `Tensor.device_address` refuses a unified weight rather than returning a host address a kernel would read as zeros. Refusing is the right direction to be wrong in. A host kernel following a device address crashes on the spot; a device kernel following a host address produces a model that runs at full speed and answers with noise, and there is no test short of comparing logits that would catch it.

### So a unified machine gets a pool too

The second half of #152 was going to be a way of making `unified` mean something to a kernel again, by reading the weights into the host mapping of a device buffer so that one set of bytes had both addresses. What was measured instead says not to bother. Four things were probed on the M4, and the fourth is the one that decided it.

A pool of 1024 MiB allocates in under a millisecond, because the allocation is lazy and the pages arrive on first touch. Its host mapping is a real address that eight threads write into concurrently at 17964 MiB/s. Those writes are still there after the mapping is released, all 262144 pages of them, and a kernel reads them back correctly. So the two address scheme would have worked.

And the ordinary asynchronous `enqueue_copy` that the discrete path has used all along works on a unified pool as well, giving a kernel result identical to the mapped one, to the bit. There was no reason to write a second copy path. What a unified machine was missing was not a mechanism, it was permission: `device_budget` returned zero for it and `plan_load` skipped it when handing out slots, both on the assumption that a pool there was a copy bought for nothing. Deleting those two conditions is the whole change.

The reserve is the one number that had to differ. A tenth of a card is holding back memory nothing else wants, and a tenth of a unified machine is holding back memory the operating system, the page cache and every other process are drawing from. A unified device keeps a quarter of the machine back instead, which on a 16 GiB M4 leaves a budget of 12288 MiB and lands close to what Metal itself recommends as a working set.

### What the device address costs on the M4

Both runs are the same warm cache 8B, back to back on the same machine, and the only difference is the flag.

| | `molla load --host` | `molla load` |
| --- | --- | --- |
| Placement | 9573 MiB unified | 9573 MiB to the device pool |
| Budget | none asked for | 12288 MiB |
| Workers | 5 | 5 |
| Read | 4156 ms, 2303 MiB/s | 5287 ms, 1810 MiB/s |
| Copy | nothing to copy | 3851 ms, 2485 MiB/s |
| Total | 4157 ms | 5289 ms |

A device address on this machine costs 1132 ms on a 4157 ms load, which is 27 per cent. That is not the copy. The copy is 3851 ms and it is hidden almost entirely, exactly as it is on the 4090: the total is 5289 ms against a read of 5287 ms, so the drain thread never once waited for the transfer to catch up. What the copy costs is the read it slows down, 2303 MiB/s falling to 1810, because on a unified machine the read and the copy are competing for one memory bus rather than for a bus each. That is the honest shape of the number and it is the reason the two runs are printed side by side rather than the device one alone.

Neither run is the fallback of the other. `--host` is what the engine does today and is the right load when the kernels are going to run on the host, where a pool would be 9573 MiB of memory and 1132 ms bought for nothing. The default is the right load when they are going to run on the device, where the alternative is not a cheaper load but a model that answers with noise. Choosing between them from the backend rather than from a flag is [#144](https://github.com/tamnd/molla/issues/144).

These timings are from a laptop that is shared and usually busy, so they are worth reading as a ratio and not as a record. The pair was taken back to back for that reason.
