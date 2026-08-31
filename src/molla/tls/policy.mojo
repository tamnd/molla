"""Which hosts molla is willing to talk to without checking the certificate.

There is no global insecure flag here and there is not going to be one, which
is the whole reason this is a type rather than a boolean. A global flag turns
verification off for every connection the process makes, and a registry pull is
not one connection: ghcr.io answers a blob request with a redirect to a signed
URL on a host it names, so a global flag would also disable verification for a
host chosen by the response rather than by the operator. Naming the host makes
that impossible to express by accident.

The rule is exact host match, case folded. No wildcards, no suffix matching, no
`*.internal`. A pattern language here is a way to turn off more verification
than was intended, and the case that actually comes up is one registry on one
name with a certificate the platform does not trust.

Nothing in molla constructs a policy that skips verification unless a person
typed the host on the command line, and every connection that skips it says so
on stdout. An insecure connection nobody can see is the one that stays insecure
for a year.
"""


def _fold(host: StringSpan) -> String:
    """Lowercase ASCII, for comparing two host names.

    Host names are ASCII by the time they get here. A name with non ASCII in it
    has to be punycoded before it can be resolved at all, and punycode is ASCII.
    """
    var out = List[UInt8]()
    var p = host.unsafe_ptr()
    for i in range(host.byte_length()):
        var c = p.unsafe_load(i)
        if c >= 65 and c <= 90:
            c += 32
        out.append(c)
    return String(StringSpan(unsafe_from_utf8=out))


struct TlsPolicy(Copyable, Movable):
    """Per host TLS settings. Verifies everything unless told otherwise."""

    var insecure_hosts: List[String]
    """Hosts whose certificate is not checked, folded to lowercase. Empty in
    every code path that does not go through a command line flag."""

    def __init__(out self):
        self.insecure_hosts = List[String]()

    def allow_insecure(mut self, host: StringSpan):
        """Stop verifying certificates for one named host.

        Named, singular. The caller has to know the host it is turning this off
        for, so a redirect to somewhere else stays verified.
        """
        var folded = _fold(host)
        for i in range(len(self.insecure_hosts)):
            if self.insecure_hosts[i] == folded:
                return
        self.insecure_hosts.append(folded)

    def verifies(self, host: StringSpan) -> Bool:
        """Whether the certificate for `host` has to check out.

        True is the answer for every host that was not named, including the one
        a redirect just pointed at.
        """
        var folded = _fold(host)
        for i in range(len(self.insecure_hosts)):
            if self.insecure_hosts[i] == folded:
                return False
        return True

    def any_insecure(self) -> Bool:
        """Whether this policy skips verification for anything at all. For the
        one line a command prints when it does."""
        return len(self.insecure_hosts) > 0

    def insecure_list(self) -> String:
        """The named hosts, comma separated, for that same line."""
        var out = String("")
        for i in range(len(self.insecure_hosts)):
            if i > 0:
                out += ", "
            out += self.insecure_hosts[i]
        return out^
