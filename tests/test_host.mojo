"""Tests for host detection.

Deliberately loose. The point is not to assert the exact machine, it is to catch
a target that reports nonsense, which is what a broken toolchain or a bad FFI
binding looks like from the outside.
"""

from harness import Suite

from molla.build_info import MOJO_PIN, VERSION
from molla.host import arch_name, detect, os_name


def run(mut suite: Suite):
    suite.group("host")

    var os = os_name()
    suite.check(os == "macos" or os == "linux", "os is a supported target")
    suite.check(arch_name().byte_length() > 0, "arch is reported")

    var host = detect()
    suite.check(host.physical_cores >= 1, "physical core count is at least one")
    suite.check(
        host.logical_cores >= host.physical_cores,
        "logical cores are at least physical cores",
    )
    suite.check(host.simd_f32 >= 1, "f32 simd width is at least one")
    suite.check(
        host.simd_f32 & (host.simd_f32 - 1) == 0,
        "f32 simd width is a power of two",
    )
    suite.check(
        host.simd_f16 >= host.simd_f32, "f16 has at least as many lanes as f32"
    )
    suite.check(host.is_64, "target is 64 bit")

    suite.check(String(VERSION).byte_length() > 0, "version string is set")
    suite.check(
        MOJO_PIN == "1.0.0", "mojo pin matches the toolchain in pixi.toml"
    )
