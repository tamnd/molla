# JSON, in both directions, at 2 GB/s

Issue #13 asks for two modes over one scanner, exact numbers with no `strtod`, a serializer that preserves key order, and a gate: a 100 KB chat body parsed in one pass with no allocations outside the arena, and at least 1 GB/s on the M4. This records what was built, the two places the design deliberately departs from what the issue describes, and the numbers.

## What was built

`molla.json.scan` is the SIMD layer. It classifies bytes a vector at a time in the style of simdjson stage one, and the useful thing it does is find the quote, the backslash and any raw control byte with one mask. Validation is then free rather than a second pass: a string that has none of the three is copied nowhere and checked nowhere, because the scan already proved it is clean.

`molla.json.decimal` is the exact decimal expansion, a port of Go's `strconv/decimal.go` shifting. Two deliberate differences. Scratch is filled from the right rather than using a leftcheat table, so the digit count comes out of where the write pointer stopped, and digits are held as values rather than as ASCII, which saves a subtract in every inner loop.

`molla.json.number` is both directions of the conversion. Three paths, tried in order: an integer with no fraction and no exponent that fits in 64 bits stays an integer, a double whose digits fit in 53 bits with an exponent between -22 and 22 goes through Clinger's fast path, and everything else goes through the exact decimal. Printing goes back through the same struct, expanding the two midpoints to the neighbouring doubles and walking the three digit strings until they separate, which gives the shortest form that reads back as the same double.

`molla.json.reader` is streaming mode, a pull loop over events. `molla.json.dom` is DOM mode, a flat node list with linked list children. `molla.json.serialize` is the writer.

## Two departures from the issue, and why

The issue asks for streaming mode to emit "events into a handler". It is a pull loop instead. A push handler has to carry a duplicate of the parser's state machine, because it arrives with an event and no idea where in the document it is, so every caller ends up writing a small parser of its own to interpret the callbacks. A pull loop lets the caller write ordinary straight line code with the position implied by where it is in the loop. Building push on top of pull is a loop and a virtual call; building pull on top of push needs a coroutine or a thread.

The issue also says "without the two stage tape", and there is no tape. A tape is worth building when a document is going to be walked more than once or walked in an order the parser did not choose. A request body is parsed once into a typed struct, so the tape would be a materialised intermediate that is read exactly once and thrown away, which is a memory write per structural element bought for nothing.

The reader is also not resumable across network reads, which is a third departure worth stating because it is the one most likely to be questioned. Holding a partial token across the gap means copying it somewhere, and that copy is the allocation this layer exists to avoid. `molla.http.body` already buffers or spills a body before handing it over, so the reader is always given a complete document.

## Numbers, which is where converters go wrong

The number code is where a JSON library is either right or nearly right, and nearly right is the kind of thing that shows up months later as one customer whose floats come back different. So the tests are the inputs that break converters rather than the ones that look like numbers.

`2.2250738585072011e-308` is the largest subnormal and is the value that used to hang PHP's `strtod` in an infinite loop. `7.2057594037927933e16` is a halfway case that separates a correct rounder from one that is close. `5e-324` is the smallest subnormal and has to come out as one bit rather than as zero, with `2.4703282292062327e-324` rounding to zero and `2.4703282292062328e-324` rounding up to that one bit. `1.7976931348623157e308` is the largest finite double and anything past it is an infinity rather than a wrap.

Integers are the other half. `9223372036854775807` and `-9223372036854775808` are the two ends, and `12345678901234567890` is the one that catches a parser that counts digits instead of checking the value. That last one found a real bug during development: the guard was on the digit count, so a twenty digit number came back as its first nineteen digits, silently. The fix is on the value and on the exponent together.

Printing is checked against the two JavaScript cutoffs, because a number written here and read by a browser should print the same on both sides. `1e20` prints in full as `100000000000000000000` and `1e21` prints as `1e+21`. `1e-6` prints as `0.000001` and `1e-7` prints as `1e-7`. Negative zero keeps its sign, since it is a different double. A NaN or an infinity has no JSON spelling, so it goes out as `null` and the writer counts how many times that happened.

There is also a round trip over four thousand doubles built from random bit patterns, which is there for the bugs that are asymmetric. It found one. The slow path was being handed the number text starting at the minus sign, so any negative number large enough to miss the fast path came back as negative zero. `-1.5e176` parsed as `-0.0` and no fixed test case would have caught it, because every hand written negative in the test file was small enough to take the fast path.

## The gate

`molla jsonbench 100 2000` builds a chat shaped body with the package's own writer, warms up four times so scratch growth is not counted, then times two thousand parses with the allocation counter read before and after.

The body is generated rather than checked in. A fixture that is one long message is a benchmark of `memcpy` and a fixture that is a thousand tiny numbers is a benchmark of the number parser. This one has the shape of the traffic: a handful of settings, a list of messages with quotes and newlines in them so the escape path is on the measured route, a nested tool schema, and a few floats.

| M4, 100 kB body, 2000 rounds | Streaming | DOM |
| --- | --- | --- |
| Throughput | 2283 MB/s | 1920 MB/s |
| Elapsed | 90 ms | 107 ms |
| Allocations | 0 | 5 |

2474 events per parse, 1642 strings of which 402 needed decoding into the scratch. The gate is 1 GB/s and zero allocations, and both are cleared.

The DOM number is measured separately because it is a different promise. DOM mode allocates by design, since a tree that outlives the parse has to live somewhere, and the number worth knowing is how much: five allocations for a 1234 node document, which is the node list growing and then staying grown.

The zero is also asserted in the test suite rather than only in the benchmark, so it fails a run rather than a reading. `tests/test_json.mojo` parses the same shape two thousand times through a reader built once, and checks the allocation counter did not move.

## Where it was run

| Machine | Result |
| --- | --- |
| M4, macOS, arm64 | 809 passed, 0 failed |
| server1, EPYC, x86_64 | 809 passed, 0 failed |
| server2, EPYC, x86_64 | 809 passed, 0 failed |
| gpc, i9-13900K on WSL2, x86_64 | 808 passed, 1 failed |

The one failure on gpc is issue #87, the reactor backpressure test under WSL2, and is unrelated to this work. It fails on main in the same way.

## What is still not covered

No comments, no trailing commas, no NaN and no Infinity, all deliberate. RFC 8259 has none of them and a parser that accepts them disagrees with whatever is on the other end.

No streaming across reads, for the reason above. No number preserved as text, so a JSON document whose integers exceed 64 bits loses precision the same way every other parser does. No duplicate key rejection: both members are kept in order and a lookup returns the first, which is what preserves what the model actually emitted.
