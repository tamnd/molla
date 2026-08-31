# Config, logs, metrics, and the admin routes

Issue #16. None of this makes molla faster and all of it is what makes molla debuggable by somebody who is not holding the source open, which is the only state it will ever be in once it leaves this repository.

The acceptance criterion was two things: the admin routes respond correctly, and logging at a disabled level costs no allocations. Both are checked in the suite, and the first is also checked by `molla drain`, which is the only place outside the tests where all of this runs at once under load.

## What is covered

| Area | What it does |
| --- | --- |
| `ops.config` | settings, their defaults, and the file, environment and flag that override them |
| `ops.log` | a byte ring per worker, written by that worker and flushed by somebody else |
| `ops.metrics` | one set of counters per worker, added up when somebody scrapes them |
| `sys.clock` | `monotonic_ns` for durations, `realtime_ns` for something a person reads |
| `http.protocol` | the request path that counts and logs, and `/molla/version`, `/molla/health`, `/molla/metrics` |
| `molla config get` | the effective value of a setting and where it came from |

## Precedence is the answer, not the value

Anybody can print a number. The question somebody has at three in the morning is why the number is that and not what the file says, and the answer is nearly always that a flag or an environment variable is quietly winning. So every setting carries where its value came from, and `molla config get` prints both.

```
$ molla config get
workers = 0 (default)
idle_timeout_ms = 60000 (default)
drain_deadline_ms = 10000 (default)
log_level = info (default)
log_queue_bytes = 65536 (default)
admin_routes = true (default)
metrics = true (default)

$ MOLLA_WORKERS=6 molla config get workers --workers=9
workers = 9 (flag, default 0)
```

The order is flag over environment over file over default, read in ascending order of strength so the code reads the way the precedence does. The default is only shown on the line when something overrode it, because `workers = 0 (default, default 0)` is noise and `workers = 9 (flag, default 0)` is the answer to the question that was asked.

A value that is not the right shape is reported rather than guessed at. `--workers=banana` prints a complaint that names the value and the source, and reading the setting falls back to the default. A log level that is not a level is caught the same way, which is why the level constants live in `config.mojo` rather than in `log.mojo`: checking one should not drag in the whole ring.

## Logging that costs nothing when nobody is listening

One byte ring per worker, written only by that worker and read only by the housekeeping thread. That is single producer single consumer, so the whole of the synchronisation is two positions that only ever go up, and nothing on the request path waits for anything.

Records are written straight into the ring rather than into a scratch buffer and copied. The worker is the only writer and the reader only ever looks below the published write position, so the bytes of a half built record are invisible until the length header goes in and the position is published. The header goes in last, which is what makes that true.

A worker that finds its ring full drops the record and counts the drop. A server that stalls a request to write a line about the request has its priorities backwards, and a drop counter says so out loud.

The criterion was no allocations at a disabled level, and the cost is lower than that: one atomic load and a comparison. `begin` returns an entry with `ok` false and every method on it returns immediately, so a call site writes the fields it wants without asking first.

```
ops log costs nothing when off
  ok       a thousand calls at a disabled level allocate nothing at all
  ok       and write nothing
  ok       and turning the level on writes again
```

The last line is there because a check that only measures zero cannot tell the difference between a level check that works and a loop the compiler deleted.

The format is logfmt, not JSON. Both are structured and only one of them can be read by a person with grep and no tools.

```
ts=1788217700657000000 level=error worker=0 msg="handler raised" slot=0 detail=the handler for /boom raised, which is its whole job
```

Control bytes in a value become dots and a double quote becomes a single one. A newline in a value means the next line is a forgery, and a request header is a value.

## Counters that do not fight over a cache line

A counter every worker increments is a cache line every core has to take a turn owning, and molla has one of those per request. So each worker gets its own copy of every series on its own cache line, and the sum only happens when `/molla/metrics` is scraped, which is once every fifteen seconds by a machine that does not care how long it takes.

The series are a fixed list rather than a registry with names in a map. A map lookup per increment costs more than the increment.

Statuses are bucketed by class rather than counted per code. A per code counter is a label with an unbounded range in it, which is the standard way to make a metrics backend fall over, and the question anybody actually asks is what fraction of responses were our fault.

Durations go out as integer nanoseconds in a `_ns_total` counter beside a `_count` counter. That pair is what a histogram degrades to when you do not have one, molla does not have one yet, and saying that in the HELP text is better than exporting a number that looks like a quantile and is not.

## Which clock

A duration is always measured on the monotonic clock, because a wall clock that jumps when ntp corrects it turns a two millisecond request into a two second one or into a negative one. The wall clock is only for something a person or a client reads: a Date header, a log line timestamp. Nothing subtracts two realtime readings.

Every duration molla exports is an integer count of nanoseconds rather than a float of seconds, so no reader has to guess the unit and no precision is lost on the way out.

## The admin routes share the main port

`/molla/version`, `/molla/health` and `/molla/metrics` are served on the same listener as everything else. A second listener on a second port is the safer answer and it is also a second thing to configure, a second thing to expose in a container, and a second thing to forget. Everything these routes return is information molla already prints on startup. The day one of them can change something is the day the port question gets asked again.

They are off unless a caller turns them on, so a server nobody configured does not answer questions about itself. With them off the paths are a 404 like any other unknown path, not a 403, because a 403 tells you the route exists.

The exposition goes out with the content type the format asks for, down to the version parameter, or Prometheus falls back to a guess. The body is built in the connection's scratch buffer, which is the same buffer a stream event uses and is reused across requests, so a scrape costs a memcpy rather than an allocation.

## What was run

`molla drain` turns all of it on, so a drain run is now also an operations run: the three routes are asked for before the load starts, and the counters are printed after the drain so they include everything it flushed.

```
$ molla drain 8 3000
drain 8 connections, 3000 ms deadline
  workers        10
  port           64282
  handler raise  500
  still serving  yes
  admin routes   3 of 3
  in flight      8 of 8 connections
  before signal  208 of 8192 answers
  shutdown       SIGTERM, drained cleanly in 5ms, 11 connections served, 0 dropped
  answered       8 of 8 connections
  truncated      0
  accepted       11
  requests       8197
  responses      8196 2xx, 1 5xx
  handler raises 1
  logs dropped   0
  result         pass
```

The gap between 8197 requests and the 208 answers the clients had read before the signal is the drain's whole subject, and it is now a number rather than a claim.

| Machine | Kernel | Cores | Suite | drain 64, 20 runs |
| --- | --- | --- | --- | --- |
| macbook, M4 | Darwin 24.6, macOS 15.8 | 10 | 1018 passed | clean |
| server1, EPYC | Linux 6.8, Ubuntu 24.04 | 4 | 1019 passed | 20 of 20 |
| server2, EPYC | Linux 6.8, Ubuntu 24.04 | 6 | 1019 passed | 20 of 20 |
| gpc, i9-13900K | Linux 6.18 on WSL2, Ubuntu 26.04 | 32 | 1018 passed, 1 failed | 20 of 20 |

The suite is one check longer on Linux than on macOS, because sharded accept only exists there.

The one failure on gpc is issue #87, the reactor backpressure test on WSL2, which fails on the commit before this one too.

One flake was found and fixed on the way. Two of these tests read a log ring after the sink that owns it had been dropped, which Mojo is entitled to do at the sink's last mention, and which is not a use it can see when the thing holding the address is a copied `Logger`. It passed every time on macOS and failed about two runs in five on server1.

## What is not covered

The log level cannot be changed from outside a running process yet. The mechanism is there, it is one atomic store and every worker sees it on its next call, and there is nothing wired to it because the only thing that could do the wiring is an admin route that changes something, which is the decision the port question above is waiting on.

There is no histogram, so there are no quantiles. The mean is what the two duration series give you.

Nothing here is sampled or rate limited. A server logging every request at debug under real load will spend real time in the flush, and the drop counter is what will tell you.
