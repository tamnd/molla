"""The molla binary.

Nothing serves yet. At M0 the only job is to prove that one pinned toolchain
produces a working binary on every machine in the fleet, and to print what that
binary thinks it is running on. The real command surface arrives with M2.
"""

from std.sys import argv, exit

from molla.build_info import MOJO_PIN, VERSION
from molla.engine.backend import Backend, Request, choose_backend, parse_backend
from molla.engine.generate import run_generate
from molla.engine.generate_device import run_generate_device
from molla.engine.sample import SamplerConfig
from molla.engine.serve import run_serve
from molla.host import detect
from molla.http.server import run_http
from molla.jinja.bench import run_template
from molla.json.bench import run_json_bench
from molla.model.gguf import run_gguf
from molla.model.load import run_load
from molla.model.repo import run_model_spec
from molla.model.safetensors import run_safetensors
from molla.net.allocs import run_allocs
from molla.net.drain import run_drain
from molla.net.echo import run_echo
from molla.net.soak import run_soak
from molla.net.soak_http import run_http_soak
from molla.net.soak_net import run_net_soak
from molla.ops.config import describe_setting, load_config
from molla.registry.pull import run_pull
from molla.sys.device import run_devices
from molla.sys.mem import AllocCounter
from molla.sys.poll import USES_KQUEUE
from molla.tls.client import probe, run_tls
from molla.tokenizer.tokenizer import run_tokenize


def print_version():
    var host = detect()
    print("molla", VERSION)
    print("  mojo      ", MOJO_PIN)
    print("  target    ", host.os, host.arch)
    print(
        "  cores     ",
        host.physical_cores,
        "physical,",
        host.logical_cores,
        "logical",
    )
    print("  simd width", host.simd_f32, "x f32,", host.simd_f16, "x f16")
    print("  poller    ", "kqueue" if USES_KQUEUE else "epoll")

    # Printed here rather than left to the first pull that fails. TLS is loaded
    # at runtime, so a machine without it runs everything except HTTPS, and the
    # only honest way to say that is in the command everybody runs first.
    var tls = probe()
    if tls.available:
        print("  tls       ", tls.backend, "up to", tls.max_protocol)
    else:
        print("  tls        unavailable, HTTPS is off:", tls.detail)


def print_usage():
    print("usage: molla <command>")
    print()
    print("commands:")
    print("  version         print the version, toolchain, and host details")
    print("  echo [port]     run the M0 TCP echo spike on 127.0.0.1")
    print("  soak [n] [sec]  hold n echo connections for sec seconds")
    print("  netsoak [n] [s] hold n connections against the threaded reactor")
    print(
        "  httpsoak [n] [s] hold n connections of mixed HTTP traffic against"
        " the server"
    )
    print(
        "  drain [n] [ms]  load n connections, shut down on a signal, check the"
        " drain"
    )
    print(
        "  allocs [n] [r]  run a mixed load twice and check the second pass"
        " allocated nothing"
    )
    print("  jsonbench [kb] [n] parse an n round chat body and print the rate")
    print(
        "  template <t> <v> [n] render a chat template with a json variables"
        " file, and time n renders"
    )
    print("  http [port]     run the M0 HTTP/1.1 spike on 127.0.0.1")
    print(
        "  gguf <path>     print the metadata and tensor directory of a model"
    )
    print(
        "  safetensors <p> print the header and tensor directory of a"
        " safetensors model"
    )
    print(
        "  spec <path>     print what a model is and what this build can do"
        " with it"
    )
    print(
        "  load <path> [n] [--host] [--device=..]  load a model's weights and"
        " report each stage"
    )
    print(
        "  generate <model> <tokenizer.json> <prompt> [n] [ctx]  generate text"
    )
    print(
        "                  sampling: --temp --top-k --top-p --min-p --typical"
    )
    print(
        "                  --repeat-penalty --frequency-penalty"
        " --presence-penalty"
    )
    print("                  --repeat-last-n --seed, and no flags means greedy")
    print(
        "                  --device=auto|cpu|metal|cuda picks the backend, and"
        " auto is"
    )
    print(
        "                  the default. An api may be followed by a colon and"
        " an index."
    )
    print(
        "  serve <model> <tokenizer.json>  answer OpenAI requests against a"
        " model"
    )
    print(
        "                  --host --port --ctx --device, and 127.0.0.1:8000"
        " when nothing says"
    )
    print(
        "  tokenize <tokenizer.json> <prompt> [--ids]  print how many tokens a"
        " prompt is"
    )
    print("  devices         list what this machine can put a tensor on")
    print("  config get [key] print a setting and where its value came from")
    print("  tls <host>      connect over TLS and print what was negotiated")
    print("  pull <ref>      pull a blob from ghcr.io and check its digest")
    print("  help            print this message")
    print()
    print("flags:")
    print(
        "  --config=PATH   read settings from this file rather than molla.conf"
    )
    print(
        "  --<key>=VALUE   set any configuration key, and beat the environment"
        " and the file"
    )
    print(
        "  --insecure      do not check the certificate of the host named on"
        " this"
    )
    print(
        "                  command line. Accepted by tls and pull, and it"
        " covers"
    )
    print("                  that one host, not a redirect it sends you to.")
    print()
    print("See the roadmap for what lands when:")
    print("  https://github.com/tamnd/molla/blob/main/docs/roadmap.md")


def run_config(args: List[String]):
    """`molla config get [key]`.

    Prints the effective value and the reason it is the effective value, which
    is the half that matters. Anybody can print a number. The question somebody
    has at three in the morning is why the number is that and not what the file
    says, and the answer is nearly always that a flag or an environment
    variable is quietly winning.
    """
    if len(args) < 3 or args[2] != "get":
        print("usage: molla config get [key]")
        print()
        print("Prints every setting when no key is named. Settings are read")
        print("from the file, then the environment, then the flags, and the")
        print("last one to speak wins.")
        exit(2)

    var config = load_config(args)
    var problems = config.problems()
    for i in range(len(problems)):
        print("molla config: " + problems[i])

    if len(args) < 4:
        print(config.describe(), end="")
        if len(problems) > 0:
            exit(1)
        return

    var key = args[3]
    var at = config.index_of(key)
    if at < 0:
        print("molla config: there is no setting called '", key, "'", sep="")
        print("Run 'molla config get' for the ones there are.")
        exit(2)
    print(describe_setting(config.get(key)))
    if len(problems) > 0:
        exit(1)


def _flag_float(key: String, text: String) raises -> Float32:
    try:
        return Float32(Float64(text))
    except:
        raise Error("--" + key + " wants a number and got '" + text + "'")


def _flag_int(key: String, text: String) raises -> Int:
    try:
        return atol(text)
    except:
        raise Error("--" + key + " wants a whole number and got '" + text + "'")


def _flag_value(arg: String) -> String:
    """Everything after the first equals sign of a `--name=value` flag.

    Empty when there is nothing after it, which every caller treats the same as
    the flag's own default rather than as an error, because `--device=` with
    nothing on the end reads as somebody who changed their mind mid command.
    """
    var eq = arg.find("=")
    if eq < 0:
        return String("")
    return String(arg[byte = eq + 1 : arg.byte_length()].strip())


def sampling_flag(mut config: SamplerConfig, arg: String) raises -> Bool:
    """One `--name=value` sampling flag, or False when it is not one.

    The same long form with an equals sign the configuration flags use, so
    there is one shape of flag in the binary rather than two. A space separated
    form needs a table of which flags take a value, and that table is what goes
    stale.
    """
    if not arg.startswith("--"):
        return False
    var body = arg[byte = 2 : arg.byte_length()]
    var eq = body.find("=")
    if eq < 0:
        return False
    var key = String(body[byte=0:eq].strip())
    var val = String(body[byte = eq + 1 : body.byte_length()].strip())
    if key == "temp":
        config.temperature = _flag_float(key, val)
    elif key == "top-k":
        config.top_k = _flag_int(key, val)
    elif key == "top-p":
        config.top_p = _flag_float(key, val)
    elif key == "min-p":
        config.min_p = _flag_float(key, val)
    elif key == "typical":
        config.typical_p = _flag_float(key, val)
    elif key == "repeat-penalty":
        config.repeat_penalty = _flag_float(key, val)
    elif key == "frequency-penalty":
        config.frequency_penalty = _flag_float(key, val)
    elif key == "presence-penalty":
        config.presence_penalty = _flag_float(key, val)
    elif key == "repeat-last-n":
        config.repeat_last_n = _flag_int(key, val)
    elif key == "seed":
        config.seed = UInt64(_flag_int(key, val))
    else:
        return False
    return True


def main():
    # --insecure is pulled out here rather than parsed per command, so it can
    # be written before or after the argument it applies to. Everything else is
    # positional, which is all the M0 commands need.
    var raw = argv()
    var args = List[String]()
    var insecure = False
    for i in range(len(raw)):
        var arg = String(raw[i])
        if arg == "--insecure":
            insecure = True
        else:
            args.append(arg^)

    if len(args) < 2:
        print_usage()
        return

    var command = args[1]
    if insecure and command != "tls" and command != "pull":
        print("molla: --insecure means nothing to '", command, "'", sep="")
        exit(2)
    if command == "version" or command == "--version" or command == "-V":
        print_version()
    elif command == "help" or command == "--help" or command == "-h":
        print_usage()
    elif command == "echo":
        # Port 0 means let the kernel choose, and the server prints what it got.
        var port: UInt16 = 0
        if len(args) > 2:
            try:
                port = UInt16(Int(args[2]))
            except:
                print(
                    "molla: '",
                    args[2],
                    "' is not a port number",
                    sep="",
                )
                exit(2)
        try:
            run_echo(port)
        except e:
            print("molla echo:", e)
            exit(1)
    elif command == "soak":
        # Defaults are the numbers issue #2 asks for.
        var connections = 1000
        var seconds = 60
        try:
            if len(args) > 2:
                connections = Int(args[2])
            if len(args) > 3:
                seconds = Int(args[3])
        except:
            print(
                "molla: soak takes a connection count and a duration in seconds"
            )
            exit(2)
        try:
            exit(run_soak(connections, seconds))
        except e:
            print("molla soak:", e)
            exit(1)
    elif command == "netsoak":
        # Defaults are the numbers issue #10 asks for. An hour is a long time to
        # wait for a test, and it is the number in the acceptance criterion.
        var net_connections = 1000
        var net_seconds = 3600
        try:
            if len(args) > 2:
                net_connections = Int(args[2])
            if len(args) > 3:
                net_seconds = Int(args[3])
        except:
            print(
                "molla: netsoak takes a connection count and a duration in"
                " seconds"
            )
            exit(2)
        try:
            exit(run_net_soak(net_connections, net_seconds))
        except e:
            print("molla netsoak:", e)
            exit(1)
    elif command == "httpsoak":
        # Defaults are the numbers issue #18 asks for: a thousand connections
        # of mixed traffic for an hour. The nightly workflow runs exactly this
        # with no arguments.
        var http_connections = 1000
        var http_seconds = 3600
        try:
            if len(args) > 2:
                http_connections = Int(args[2])
            if len(args) > 3:
                http_seconds = Int(args[3])
        except:
            print(
                "molla: httpsoak takes a connection count and a duration in"
                " seconds"
            )
            exit(2)
        try:
            exit(run_http_soak(http_connections, http_seconds))
        except e:
            print("molla httpsoak:", e)
            exit(1)
    elif command == "drain":
        # Small by default. The point is to run it a hundred times, not to run
        # it once with a big number.
        var drain_connections = 64
        var drain_deadline = 5000
        try:
            if len(args) > 2:
                drain_connections = Int(args[2])
            if len(args) > 3:
                drain_deadline = Int(args[3])
        except:
            print(
                "molla: drain takes a connection count and a deadline in"
                " milliseconds"
            )
            exit(2)
        try:
            exit(run_drain(drain_connections, drain_deadline))
        except e:
            print("molla drain:", e)
            exit(1)
    elif command == "jsonbench":
        # The two numbers issue #13 asks for, on whatever machine is running it.
        var json_kb = 100
        var json_rounds = 2000
        try:
            if len(args) > 2:
                json_kb = Int(args[2])
            if len(args) > 3:
                json_rounds = Int(args[3])
        except:
            print("molla: jsonbench takes a body size in kB and a round count")
            exit(2)
        exit(run_json_bench(json_kb * 1024, json_rounds))
    elif command == "template":
        if len(args) < 4:
            print(
                "molla: template takes a template file and a json file of the"
                " variables"
            )
            exit(2)
        var template_rounds = 0
        try:
            if len(args) > 4:
                template_rounds = Int(args[4])
        except:
            print("molla: template takes a round count as its third argument")
            exit(2)
        exit(run_template(String(args[2]), String(args[3]), template_rounds))
    elif command == "http":
        var http_port: UInt16 = 0
        if len(args) > 2:
            try:
                http_port = UInt16(Int(args[2]))
            except:
                print(
                    "molla: '",
                    args[2],
                    "' is not a port number",
                    sep="",
                )
                exit(2)
        try:
            run_http(http_port)
        except e:
            print("molla http:", e)
            exit(1)
    elif command == "gguf":
        if len(args) < 3:
            print("molla gguf: expected a path to a .gguf file")
            exit(2)
        try:
            run_gguf(args[2])
        except e:
            print("molla gguf:", e)
            exit(1)
    elif command == "safetensors":
        if len(args) < 3:
            print("molla safetensors: expected a path to a file or a directory")
            exit(2)
        var st_counter = AllocCounter()
        try:
            run_safetensors(args[2], st_counter.raw())
        except e:
            print("molla safetensors:", e)
            st_counter.close()
            exit(1)
        st_counter.close()
    elif command == "spec":
        if len(args) < 3:
            print("molla spec: expected a model file or a model directory")
            exit(2)
        try:
            run_model_spec(args[2])
        except e:
            print("molla spec:", e)
            exit(1)
    elif command == "load":
        if len(args) < 3:
            print("molla load: expected a gguf file")
            exit(2)
        var workers = 0
        var host_only = False
        var load_want = Request()
        try:
            for i in range(3, len(args)):
                if args[i] == "--host":
                    host_only = True
                elif args[i].startswith("--device="):
                    load_want = parse_backend(_flag_value(args[i]))
                else:
                    workers = atol(args[i])
            # `--host` is the older spelling of `--device=cpu` and it still
            # wins, so a script written before there was a flag keeps timing
            # the load it was timing.
            var picked = choose_backend(args[2], load_want, False)
            if not picked.on_device:
                host_only = True
            run_load(args[2], workers, True, host_only, picked.device)
        except e:
            print("molla load:", e)
            exit(1)
    elif command == "generate":
        if len(args) < 5:
            print(
                "molla generate: expected a gguf file, a tokenizer.json, and a"
                " prompt"
            )
            exit(2)
        var limit = 0
        var context = 0
        var sampling = SamplerConfig()
        # `--device` on its own means auto, which is what it meant when it was
        # a bare flag in #143 on a machine with one card. Everything else comes
        # through `molla.engine.backend`, which is also what decides whether an
        # unspelled run ends up on a card at all.
        var want = Request()
        try:
            # The two numbers stay positional and the sampling settings are
            # named, in either order. Nobody is going to remember a ninth
            # position, and the two that were already there are in scripts.
            var positional = 0
            for i in range(5, len(args)):
                if sampling_flag(sampling, args[i]):
                    continue
                if args[i] == "--device":
                    want = Request()
                    continue
                if args[i].startswith("--device="):
                    want = parse_backend(_flag_value(args[i]))
                    continue
                if args[i].startswith("--"):
                    raise Error(
                        String("'") + args[i] + "' is not a flag this takes"
                    )
                if positional == 0:
                    limit = atol(args[i])
                elif positional == 1:
                    context = atol(args[i])
                else:
                    raise Error(
                        String("'")
                        + args[i]
                        + "' is one argument more than this takes"
                    )
                positional += 1
            var picked = choose_backend(args[2], want)
            if picked.on_device:
                run_generate_device(
                    args[2], args[3], args[4], limit, context, sampling, picked
                )
            else:
                run_generate(
                    args[2], args[3], args[4], limit, context, sampling, picked
                )
        except e:
            print("molla generate:", e)
            exit(1)
    elif command == "serve":
        if len(args) < 4:
            print("molla serve: expected a gguf file and a tokenizer.json")
            exit(2)
        var serve_host = String("127.0.0.1")
        var serve_port: UInt16 = 8000
        var serve_context = 0
        var serve_want = Request()
        try:
            # Named flags rather than positions, because a host and a port and
            # a context length are three numbers nobody is going to remember
            # the order of, and every one of them has a default worth having.
            for i in range(4, len(args)):
                var arg = args[i]
                if not arg.startswith("--"):
                    raise Error(
                        String("'")
                        + arg
                        + "' is one argument more than this takes"
                    )
                var body = arg[byte = 2 : arg.byte_length()]
                var eq = body.find("=")
                if eq < 0:
                    raise Error(
                        String("'") + arg + "' wants a value, as --name=value"
                    )
                var key = String(body[byte=0:eq].strip())
                var val = String(
                    body[byte = eq + 1 : body.byte_length()].strip()
                )
                if key == "host":
                    serve_host = val
                elif key == "port":
                    serve_port = UInt16(_flag_int(key, val))
                elif key == "ctx":
                    serve_context = _flag_int(key, val)
                elif key == "device":
                    serve_want = parse_backend(val)
                else:
                    raise Error(
                        String("'") + arg + "' is not a flag this takes"
                    )
            exit(
                run_serve(
                    args[2],
                    args[3],
                    serve_host,
                    serve_port,
                    serve_context,
                    choose_backend(args[2], serve_want),
                )
            )
        except e:
            print("molla serve:", e)
            exit(1)
    elif command == "tokenize":
        # No model file, because encoding a prompt does not need one. The
        # benchmark harness asks this to find out how much work it is about to
        # hand three engines, and loading eight gigabytes to count 512 tokens
        # would have made the measurement cost more than the thing measured.
        if len(args) < 4:
            print("molla tokenize: expected a tokenizer.json and a prompt")
            exit(2)
        try:
            run_tokenize(args[2], args[3], len(args) > 4 and args[4] == "--ids")
        except e:
            print("molla tokenize:", e)
            exit(1)
    elif command == "devices":
        try:
            run_devices()
        except e:
            print("molla devices:", e)
            exit(1)
    elif command == "tls":
        if len(args) < 3:
            print("molla tls: expected a hostname")
            exit(2)
        var tls_port: UInt16 = 443
        if len(args) > 3:
            try:
                tls_port = UInt16(Int(args[3]))
            except:
                print(
                    "molla: '",
                    args[3],
                    "' is not a port number",
                    sep="",
                )
                exit(2)
        try:
            run_tls(args[2], tls_port, insecure)
        except e:
            print("molla tls:", e)
            exit(1)
    elif command == "pull":
        if len(args) < 3:
            print("molla pull: expected an image reference")
            print("  for example: molla pull linuxcontainers/alpine:latest")
            exit(2)
        try:
            run_pull(args[2], insecure)
        except e:
            print("molla pull:", e)
            exit(1)
    elif command == "allocs":
        # Small by default. The check is whether the number is zero, not how
        # long it took to get there.
        var alloc_connections = 4
        var alloc_rounds = 4
        try:
            if len(args) > 2:
                alloc_connections = Int(args[2])
            if len(args) > 3:
                alloc_rounds = Int(args[3])
        except:
            print("molla allocs: expected two numbers")
            exit(2)
        try:
            exit(run_allocs(alloc_connections, alloc_rounds))
        except e:
            print("molla allocs:", e)
            exit(1)
    elif command == "config":
        run_config(args)
    else:
        print("molla: unknown command '", command, "'", sep="")
        print("Run 'molla help' for the list of commands.")
        exit(2)
