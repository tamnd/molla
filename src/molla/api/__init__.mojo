"""The shapes other people specified.

Everything under here is somebody else's API, written down in Mojo. There is no
molla request format and no molla response format, which is decision D2, and the
practical consequence is that this package holds no opinions of its own: when
OpenAI says a streaming chunk carries `finish_reason` on the last one and null on
every other, that is what goes out, whether or not it is the design anybody here
would have picked.

The parsing and the writing live apart from `molla.http`, which frames bytes and
knows nothing about what is in them, and apart from `molla.engine`, which
generates tokens and knows nothing about who asked. That leaves this package as
the only place that has to change when a provider adds a field.
"""
