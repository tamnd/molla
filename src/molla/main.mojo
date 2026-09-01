"""The molla binary.

Nothing serves yet. At M0 the only job is to prove that one pinned toolchain
produces a working binary on every machine in the fleet, and to print what that
binary thinks it is running on. The real command surface arrives with M2.
"""

from std.sys import argv, exit

from molla.build_info import MOJO_PIN, VERSION
from molla.host import detect
from molla.http.server import run_http
from molla.json.bench import run_json_bench
from molla.model.gguf import run_gguf
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
from molla.sys.mem import AllocCounter
from molla.sys.poll import USES_KQUEUE
from molla.tls.client import probe, run_tls


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
    print("Nothing serves yet. See the roadmap for what lands when:")
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
