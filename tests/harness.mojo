"""A very small test runner.

Mojo 1.0 ships assertions in `std.testing` but no `mojo test` command, so molla
brings its own runner. When the toolchain grows a real one this file goes away.

The shape is assertion based rather than function based. A test module gets a
`Suite` and calls `check` on it. That is a deliberate choice: Mojo 1.0 will not
bind a plain top level function to a closure trait, so a registry of test
functions costs more metaprogramming than the thing is worth. Checks are also
finer grained than whole test functions, so a failure names the exact condition
rather than the enclosing test.

A module that raises rather than returning false is still handled. `section`
wraps the call and reports the exception against the module.
"""

from std.sys import exit


struct Suite(Movable):
    """Collects results across one run and reports at the end."""

    var passed: Int
    var failed: Int

    def __init__(out self):
        self.passed = 0
        self.failed = 0

    def group(mut self, name: String):
        """Start a named group. Purely to keep the output readable."""
        print()
        print(name)

    def check(mut self, condition: Bool, name: String):
        """Record one assertion."""
        if condition:
            self.passed += 1
            print("  ok      ", name)
        else:
            self.failed += 1
            print("  FAILED  ", name)

    def fail(mut self, name: String, detail: String):
        """Record a failure that came from an exception rather than a check."""
        self.failed += 1
        print("  FAILED  ", name)
        print("            ", detail)

    def report(self) -> Int:
        """Print the summary and return the process exit code."""
        print()
        if self.failed == 0:
            print(self.passed, "checks passed")
            return 0
        print(self.passed, "passed,", self.failed, "failed")
        return 1


def finish(suite: Suite):
    """Report and exit. Call this at the end of a test main."""
    var code = suite.report()
    if code != 0:
        exit(code)
