"""Client TLS.

D1 allows foreign code in three modules and this is the second of them. molla
does not implement TLS and is not going to. What it does is bind whatever the
platform already trusts, through dlopen so that a machine without a TLS library
still runs molla and only loses HTTPS.

`client.mojo` is the interface everything above uses. `openssl.mojo` and
`darwin.mojo` are the two bindings and nothing outside this package should import
them directly.

Server side TLS is not here and is not planned. molla binds loopback by default
under D9, and terminating TLS for a local server is a job for whatever is already
in front of it.
"""
