"""The molla binary.

Nothing serves yet. At M0 the only job is to prove that one pinned toolchain
produces a working binary on every machine in the fleet, and to print what that
binary thinks it is running on. The real command surface arrives with M2.
"""

from std.sys import argv, exit

from molla.build_info import MOJO_PIN, VERSION
from molla.host import detect


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


def print_usage():
    print("usage: molla <command>")
    print()
    print("commands:")
    print("  version   print the version, toolchain, and host details")
    print("  help      print this message")
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
    else:
        print("molla: unknown command '", command, "'", sep="")
        print("Run 'molla help' for the list of commands.")
        exit(2)
