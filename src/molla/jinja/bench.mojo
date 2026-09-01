"""`molla template`, which renders a chat template and times it.

Two jobs in one command because they want the same two files. Given a template
and a variables file it prints the prompt, which is what somebody debugging a
model's prompt wants. Given a round count as well it renders that many times and
prints microseconds per render, which is the number issue #23 is measured
against.

The template is compiled once and rendered in a loop, because that is what a
server does. Compiling per render would be measuring the parser, which is the
part that happens when a model loads and never again.
"""

from molla.jinja.env import Limits
from molla.jinja.template import Template
from molla.sys.clock import monotonic_ns
from molla.sys.file import FileInfo, close_fd, fstat, open_read, pread_all

comptime MAX_TEMPLATE_BYTES = 4194304


def read_text(path: StringSpan) -> String:
    """The whole file, or empty when it cannot be read."""
    var opened = open_read(path)
    if not opened.is_ok():
        return String("")
    var fd = opened.value
    var info = FileInfo()
    var sized = fstat(fd, info)
    if not sized.is_ok() or info.size <= 0 or info.size > MAX_TEMPLATE_BYTES:
        _ = close_fd(fd)
        return String("")
    var size = info.size
    var raw = List[UInt8](capacity=size + 1)
    for _ in range(size):
        raw.append(0)
    var read = pread_all(fd, raw.unsafe_ptr(), size, 0)
    _ = close_fd(fd)
    if not read.is_ok():
        return String("")
    return String(StringSlice(unsafe_from_utf8=Span(raw)))


def run_template(template_path: String, vars_path: String, rounds: Int) -> Int:
    var source = read_text(template_path)
    if source == "":
        print("molla template: could not read", template_path)
        return 1
    var vars = read_text(vars_path)
    if vars == "":
        print("molla template: could not read", vars_path)
        return 1

    var compile_start = monotonic_ns()
    var compiled: Template
    try:
        compiled = Template(source)
    except e:
        print("molla template:", e)
        return 1
    var compile_ns = monotonic_ns() - compile_start

    if rounds <= 0:
        try:
            print(compiled.render_object(vars), end="")
        except e:
            print("molla template:", e)
            return 1
        return 0

    var limits = Limits()
    var out_bytes = 0
    try:
        for _ in range(4):
            out_bytes = compiled.render_object(vars, limits).byte_length()
    except e:
        print("molla template:", e)
        return 1

    var start = monotonic_ns()
    try:
        for _ in range(rounds):
            _ = compiled.render_object(vars, limits)
    except e:
        print("molla template:", e)
        return 1
    var elapsed = monotonic_ns() - start

    print("template  ", source.byte_length(), "bytes")
    print("compile   ", compile_ns // 1000, "us")
    print("prompt    ", out_bytes, "bytes")
    print("rounds    ", rounds)
    print("per render", elapsed // rounds, "ns")
    return 0
