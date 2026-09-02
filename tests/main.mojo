"""Entry point for the test suite. Add new test modules here.

Modules that can raise should be wrapped in try and except, reporting through
`suite.fail` so one bad module does not hide the rest of the run. Mojo warns on
an unreachable except, so only wrap the ones that actually raise.
"""

from harness import Suite, finish

import test_allocs
import test_client
import test_concurrency
import test_dns
import test_gguf
import test_load
import test_quant
import test_arch
import test_attention
import test_block
import test_engine
import test_kernel
import test_nnmodel
import test_rope
import test_sample
import test_tensor
import test_host
import test_http
import test_io
import test_jinja
import test_json
import test_net
import test_ops
import test_reactor
import test_registry
import test_safetensors
import test_sha256
import test_soak_http
import test_spec
import test_stream
import test_sys
import test_text
import test_tls
import test_tokenizer


def main():
    var suite = Suite()

    test_host.run(suite)
    test_net.run(suite)
    test_http.run(suite)
    test_sha256.run(suite)
    try:
        test_gguf.run(suite)
    except e:
        suite.fail("test_gguf", String(e))
    try:
        test_spec.run(suite)
    except e:
        suite.fail("test_spec", String(e))
    try:
        test_safetensors.run(suite)
    except e:
        suite.fail("test_safetensors", String(e))
    try:
        test_load.run(suite)
    except e:
        suite.fail("test_load", String(e))
    try:
        test_quant.run(suite)
    except e:
        suite.fail("test_quant", String(e))
    try:
        test_tensor.run(suite)
    except e:
        suite.fail("test_tensor", String(e))
    try:
        test_kernel.run(suite)
    except e:
        suite.fail("test_kernel", String(e))
    try:
        test_rope.run(suite)
    except e:
        suite.fail("test_rope", String(e))
    try:
        test_attention.run(suite)
    except e:
        suite.fail("test_attention", String(e))
    try:
        test_arch.run(suite)
    except e:
        suite.fail("test_arch", String(e))
    try:
        test_block.run(suite)
    except e:
        suite.fail("test_block", String(e))
    try:
        test_nnmodel.run(suite)
    except e:
        suite.fail("test_nnmodel", String(e))
    try:
        test_engine.run(suite)
    except e:
        suite.fail("test_engine", String(e))
    try:
        test_sample.run(suite)
    except e:
        suite.fail("test_sample", String(e))
    try:
        test_dns.run(suite)
    except e:
        suite.fail("test_dns", String(e))
    try:
        test_client.run(suite)
    except e:
        suite.fail("test_client", String(e))
    try:
        test_registry.run(suite)
    except e:
        suite.fail("test_registry", String(e))
    try:
        test_sys.run(suite)
    except e:
        suite.fail("test_sys", String(e))
    try:
        test_io.run(suite)
    except e:
        suite.fail("test_io", String(e))
    try:
        test_reactor.run(suite)
    except e:
        suite.fail("test_reactor", String(e))
    try:
        test_concurrency.run(suite)
    except e:
        suite.fail("test_concurrency", String(e))
    try:
        test_ops.run(suite)
    except e:
        suite.fail("test_ops", String(e))
    try:
        test_allocs.run(suite)
    except e:
        suite.fail("test_allocs", String(e))
    try:
        test_soak_http.run(suite)
    except e:
        suite.fail("test_soak_http", String(e))
    try:
        test_text.run(suite)
    except e:
        suite.fail("test_text", String(e))
    try:
        test_tokenizer.run(suite)
    except e:
        suite.fail("test_tokenizer", String(e))
    try:
        test_jinja.run(suite)
    except e:
        suite.fail("test_jinja", String(e))
    test_stream.run(suite)
    test_json.run(suite)
    test_tls.run(suite)

    finish(suite)
