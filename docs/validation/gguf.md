# GGUF metadata over a memory map

The M0 spike for the model plane. Issue #4 asks whether a real model file can be mapped and its metadata walked without reading it into a heap buffer, and it is done when three real GGUF files from different families print a tensor directory matching what `gguf-dump` reports. Four were used rather than three. This records what was built, how it was checked, and the three things that were wrong before it was.

## What was built

`molla.sys.mmap` is a read only whole file mapping. It is about a hundred lines and most of them are there because of libc rather than because of mapping. `mmap` wants a NULL hint address and Mojo 1.0 has no null pointer, so the hint goes over as an integer and the result comes back as one, which is what the ABI does anyway on both architectures and has the useful side effect that the result can be compared against MAP_FAILED before it becomes a pointer. Size comes from `lseek` to the end rather than `fstat`, because `struct stat` has a different layout on macOS and on Linux and a different one again per architecture, so getting it right means writing out three layouts and getting all three right, while `lseek` returns the same number with no struct at all.

`molla.model.gguf` is the reader. Nothing in it copies the file. The header, the key value block and the tensor directory are walked once in place, and what is kept is a list of offsets: a key is a start and a length into the mapping rather than a `String`, and a value is a type tag plus the offset its payload begins at. Asking for a value seeks back to that offset and decodes it there. Arrays are located and measured but their elements are skipped, which is the whole reason this is cheap, because a tokenizer vocabulary is a 30522 entry array of strings and no question about the architecture needs it.

Every read is bounds checked against the length of the mapping. That is not defensive habit. A GGUF file becomes attacker controlled input the moment molla can be pointed at a downloaded model, and the failure mode for an unchecked length prefix on a memory map is a segfault rather than an error message.

`molla gguf <path>` prints the header, every metadata key with its type and value, and the tensor directory.

## What zero copy actually buys

Both tools below print the same information about the same 468 MB file. `gguf-dump` is Python and it materialises the arrays it walks past, so this is not a like for like comparison of two equivalent programs, and it is not offered as one. It is offered because it is the difference between mapping a file and reading it, which is the question issue #4 asks.

| qwen2.5-0.5b-instruct-q4_k_m, 468 MB, M4 | Best of five | Peak RSS |
| --- | --- | --- |
| `molla gguf` | 8.6ms | 14.4 MB |
| `gguf-dump` | 2420.8ms | 319 MB |

Across the four fixtures the resident set tracks the size of the output rather than the size of the file, which is the point.

| Fixture | File size | Peak RSS |
| --- | --- | --- |
| bge-small-en-v1.5-f16 | 64 MB | 9.3 MB |
| SmolLM2-135M-Instruct-Q8_0 | 138 MB | 10.5 MB |
| gemma-3-270m-it-Q8_0 | 278 MB | 13.2 MB |
| qwen2.5-0.5b-instruct-q4_k_m | 468 MB | 14.0 MB |

The file grows by 7x across that table and the resident set grows by 1.5x, all of which is the tensor directory being formatted for printing. The mapping itself costs the pages the header touches.

## How it was checked

Four models covering four architectures, chosen because their metadata differs in ways that matter rather than because they were small.

| Fixture | Architecture | Tensors | Keys | Quantization |
| --- | --- | --- | --- | --- |
| bge-small-en-v1.5-f16 | bert | 197 | 23 | F16 |
| SmolLM2-135M-Instruct-Q8_0 | llama | 272 | 37 | Q8_0 |
| gemma-3-270m-it-Q8_0 | gemma3 | 236 | 45 | Q8_0 |
| qwen2.5-0.5b-instruct-q4_k_m | qwen2 | 291 | 26 | Q4_K_M |

Every one of those 996 tensors was compared against `gguf-dump` on name, shape, ggml type, position in the directory, and data offset relative to the start of the tensor data. Every one of those 131 keys was compared on name, order, and value type, on the element type and count for arrays, and on the exact value for everything else including the strings. All of it matches.

Comparing the values and not only the tensors matters more than it looks. If a value were skipped by the wrong number of bytes the tensor names would come out as garbage and the mismatch would be obvious, so the tensor directory already proves the skipping arithmetic. What it does not prove is that the values themselves decode correctly, and the sign handling and the float decoding are only exercised by reading them.

The hand built files in `tests/test_gguf.mojo` cover what real models cannot. A model that llama.cpp produced is never truncated, never has bad magic and never carries a string length of 2^63, so those get a synthetic file with exactly one thing wrong with it. The test builder writes bytes directly rather than going through the reader's own helpers, so a byte order mistake in the reader cannot cancel itself out.

## Three things that were wrong

None of them were the parsing.

The first run against the bert model printed `architecture: bert` with a trailing space, a context length of 0 where `gguf-dump` said 512, and every tensor name with a doubled space after it. One cause. `text()` copied a region out of the mapping, appended a zero terminator and then handed the whole buffer including that zero to a `StringSpan`. A span carries its own length, so the terminator did not terminate anything, it just became the last character of the string. That is invisible when the string is printed on its own, which is why it took a wrong number to notice. It broke `context_length` because that key is architecture prefixed, so the lookup was for `bert\0.context_length` and nothing matched.

The second was the quantization figure, which said F32 for a file everyone calls F16. That was not a bug in the code, it was the wrong definition. It reported the tensor type that the most tensors used, and I had written in the docstring that this agrees with `general.file_type` on every file tested, which was true of the one file tested at the time. It is not true in general and not true of half these fixtures. bge-small-en-v1.5-f16 has 123 F32 norm and bias tensors against 74 F16 weight tensors, so counting says F32. qwen2.5-0.5b-instruct-q4_k_m has 133 Q5_0, 121 F32, 13 Q8_0, 12 Q6_K and 12 Q4_K, so counting says Q5_0, which is not even one of the names in the file name. The file states its own type in `general.file_type` and that is now what gets reported. Counting survives only as a fallback for files that do not carry the key, and it says so in the output when it is used.

The third only showed up because the checking was mechanical. Values are escaped so a dump stays one line per key, since `tokenizer.chat_template` is a Jinja template with real newlines in it and printing those raw turns one key into forty lines. The first version escaped the newline, the carriage return and the tab, and left the backslash alone. That is not reversible: those same templates contain the two characters backslash and n as literal text, so a reader could not tell a real newline from a literal one, and the comparison script could not either. Escaping the backslash as well makes the output busier on exactly the keys that were already busy, and makes it mean one thing.

There was also one thing that was not wrong. I first read the kv count as off by three on all four files. `gguf-dump` prepends three synthetic entries to its metadata dictionary for the version, the tensor count and the kv count, and its own `GGUF.kv_count` agreed with mine. The comparison was wrong, not the parser.

## The openat detour

The file is opened with `openat` and AT_FDCWD rather than with `open`, which looks like an odd choice in a file that does nothing else unusual. Declaring `open` through `external_call` collides with the declaration the Mojo standard library's own file API makes, the two signatures disagree, and the module is rejected with "existing function with conflicting signature". It only appeared once the tests linked both in one binary, because the tests write their fixtures with the standard library and read them back with this. `openat` has no such declaration anywhere and AT_FDCWD makes it behave exactly like `open`. The constant is one of the few where the two kernels chose different numbers, -2 on macOS and -100 on Linux, so it is selected at compile time.

## The fleet

Per D8 this was built and the suite was run on every Linux machine, which is what exercises the Linux AT_FDCWD value and the Linux mmap path rather than the macOS ones. The full suite is 167 checks and it passes everywhere.

| Machine | Threads | Result |
| --- | --- | --- |
| M4, aarch64-apple-darwin | 10 | 167 passed |
| gpc, i9-13900K on WSL2, x86_64-linux-gnu | 32 | 167 passed |
| server1, EPYC, x86_64-linux-gnu | 4 | 167 passed |
| server2, EPYC, x86_64-linux-gnu | 6 | 167 passed |
| server3, EPYC, x86_64-linux-gnu | 8 | 167 passed |

The bert fixture was also copied to server2 and dumped there. The output is byte for byte identical to the macOS output across all 232 lines, which is the check that matters for a reader that assembles its own integers: if the byte order handling were wrong on one platform the two dumps would disagree. Peak RSS on Linux was 10112 kB against 9552 kB on the M4 for the same file.

No timing is quoted for the fleet. These machines run other people's work, as recorded in `docs/validation/http.md`, and this operation is measured in milliseconds, so any number from them would be noise.

CI covers aarch64-linux-gnu on every push, which is the sixth combination and the one no machine here has.

## What is not covered

Nothing reads a tensor. The directory records where each one is and what type it is, and that is where issue #4 stops. Dequantisation is #5 and after that M1.

Arrays are measured and skipped rather than decoded, so there is no tokenizer vocabulary and no way to ask for element 400 of one. That is deliberate for now and it is the single thing most likely to be wanted next, since the tokenizer needs exactly that.

Only version 2 and version 3 are accepted. Version 1 counted string and array lengths in 32 bits and is rejected rather than half read. Big endian GGUF is rejected by the magic check, which compares bytes rather than assembling an integer so that a big endian file fails cleanly instead of being silently misread.

The reader trusts the tensor offsets it is given. They are checked to be inside the file but nothing checks that a tensor's data actually fits within the file from that offset, because that needs the block size table for every quantisation type and nothing reads tensor data yet. That check belongs with the first code that does.

`find` is a linear scan over the key list. A file has tens of keys so this has never mattered and probably never will, but it is the same shape as the `_index_of` scan noted in the HTTP writeup and it is worth naming rather than forgetting.

## What this says about D1

Not much, which is itself useful. The model plane does not care where the network edge lives. Mapping a file and walking a header is the same work in any language and it took no unsafe reasoning beyond the pointer origin workaround that the whole codebase already has, where a struct field cannot expose `AnyOrigin` so the address is held as an integer and handed out by a method.

The one thing worth carrying into #7 is that this went quickly and the HTTP work did not. If D1 ends up moving the network edge out of Mojo, none of this moves with it.
