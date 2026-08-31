"""Host facts resolved at compile time and at startup.

Everything here is cheap and allocation free. The point is that `molla version`
can answer "what machine am I on and what did this binary get compiled for"
without loading a model or touching a device driver.
"""

from std.sys.info import (
    CompilationTarget,
    is_64bit,
    num_logical_cores,
    num_physical_cores,
    simd_width_of,
)


def os_name() -> StaticString:
    """The operating system this binary was compiled for."""

    comptime if CompilationTarget.is_macos():
        return "macos"
    elif CompilationTarget.is_linux():
        return "linux"
    else:
        return "unknown"


def arch_name() -> StaticString:
    """The CPU architecture this binary was compiled for."""

    comptime if CompilationTarget.is_apple_silicon():
        return "arm64 (apple silicon)"
    elif CompilationTarget.has_neon():
        return "arm64"
    elif CompilationTarget.has_avx512f():
        return "x86_64 (avx512)"
    elif CompilationTarget.has_avx2():
        return "x86_64 (avx2)"
    else:
        return "x86_64"


@fieldwise_init
struct HostInfo(Copyable, Movable):
    """A snapshot of the host, taken once at startup."""

    var os: StaticString
    var arch: StaticString
    var physical_cores: Int
    var logical_cores: Int
    var simd_f32: Int
    var simd_f16: Int
    var is_64: Bool


def detect() -> HostInfo:
    """Read the host facts we care about.

    Core counts come from the runtime rather than the compile target, so a
    binary built on one machine reports honestly when it runs on another.
    """
    return HostInfo(
        os=os_name(),
        arch=arch_name(),
        physical_cores=num_physical_cores(),
        logical_cores=num_logical_cores(),
        simd_f32=simd_width_of[DType.float32](),
        simd_f16=simd_width_of[DType.float16](),
        is_64=is_64bit(),
    )
