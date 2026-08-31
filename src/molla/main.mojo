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
from molla.net.echo import run_echo
from molla.net.soak import run_soak
from molla.net.soak_net import run_net_soak
from molla.registry.pull import run_pull
from molla.sys.poll import USES_KQUEUE
from molla.tls.client import run_tls


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


def print_usage():
    print("usage: molla <command>")
    print()
    print("commands:")
    print("  version         print the version, toolchain, and host details")
    print("  echo [port]     run the M0 TCP echo spike on 127.0.0.1")
    print("  soak [n] [sec]  hold n echo connections for sec seconds")
    print("  netsoak [n] [s] hold n connections against the threaded reactor")
    print("  jsonbench [kb] [n] parse an n round chat body and print the rate")
    print("  http [port]     run the M0 HTTP/1.1 spike on 127.0.0.1")
    print(
        "  gguf <path>     print the metadata and tensor directory of a model"
    )
    print("  tls <host>      connect over TLS and print what was negotiated")
    print("  pull <ref>      pull a blob from ghcr.io and check its digest")
    print("  help            print this message")
    print()
    print("Nothing serves yet. See the roadmap for what lands when:")
    print("  https://github.com/tamnd/molla/blob/main/docs/roadmap.md")


def main():
    var args = argv()
    if len(args) < 2:
        print_usage()
        return

    var command = String(args[1])
    if command == "version" or command == "--version" or command == "-V":
        print_version()
    elif command == "help" or command == "--help" or command == "-h":
        print_usage()
    elif command == "echo":
        # Port 0 means let the kernel choose, and the server prints what it got.
        var port: UInt16 = 0
        if len(args) > 2:
            try:
                port = UInt16(Int(String(args[2])))
            except:
                print(
                    "molla: '",
                    String(args[2]),
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
                connections = Int(String(args[2]))
            if len(args) > 3:
                seconds = Int(String(args[3]))
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
                net_connections = Int(String(args[2]))
            if len(args) > 3:
                net_seconds = Int(String(args[3]))
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
    elif command == "jsonbench":
        # The two numbers issue #13 asks for, on whatever machine is running it.
        var json_kb = 100
        var json_rounds = 2000
        try:
            if len(args) > 2:
                json_kb = Int(String(args[2]))
            if len(args) > 3:
                json_rounds = Int(String(args[3]))
        except:
            print("molla: jsonbench takes a body size in kB and a round count")
            exit(2)
        exit(run_json_bench(json_kb * 1024, json_rounds))
    elif command == "http":
        var http_port: UInt16 = 0
        if len(args) > 2:
            try:
                http_port = UInt16(Int(String(args[2])))
            except:
                print(
                    "molla: '",
                    String(args[2]),
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
    elif command == "tls":
        if len(args) < 3:
            print("molla tls: expected a hostname")
            exit(2)
        var tls_port: UInt16 = 443
        if len(args) > 3:
            try:
                tls_port = UInt16(Int(String(args[3])))
            except:
                print(
                    "molla: '",
                    String(args[3]),
                    "' is not a port number",
                    sep="",
                )
                exit(2)
        try:
            run_tls(String(args[2]), tls_port)
        except e:
            print("molla tls:", e)
            exit(1)
    elif command == "pull":
        if len(args) < 3:
            print("molla pull: expected an image reference")
            print("  for example: molla pull linuxcontainers/alpine:latest")
            exit(2)
        try:
            run_pull(String(args[2]))
        except e:
            print("molla pull:", e)
            exit(1)
    else:
        print("molla: unknown command '", command, "'", sep="")
        print("Run 'molla help' for the list of commands.")
        exit(2)
