"""Ask max/kernels for its Apple GPU quantised matmul on this machine."""

from std.memory import alloc

from layout import Layout, LayoutTensor, lt_to_tt
from linalg.matmul.gpu.apple.int8_matmul import enqueue_apple_int8_matmul
from max.gpu.host import DeviceContext

comptime M = 64
comptime N = 64
comptime K = 64


def main() raises:
    var ctx = DeviceContext()
    print("api", ctx.api(), "name", ctx.name(), "cc", ctx.compute_capability())

    var d_c = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    var d_a = ctx.enqueue_create_buffer[DType.int8](M * K)
    var d_b = ctx.enqueue_create_buffer[DType.int8](N * K)
    var d_as = ctx.enqueue_create_buffer[DType.float32](M)
    var d_bs = ctx.enqueue_create_buffer[DType.float32](N)
    var d_bias = ctx.enqueue_create_buffer[DType.bfloat16](1)

    var c = LayoutTensor[DType.bfloat16, Layout.row_major(M, N)](d_c.unsafe_ptr())
    var a = LayoutTensor[DType.int8, Layout.row_major(M, K)](d_a.unsafe_ptr())
    var b = LayoutTensor[DType.int8, Layout.row_major(N, K)](d_b.unsafe_ptr())
    var a_s = LayoutTensor[DType.float32, Layout.row_major(M)](d_as.unsafe_ptr())
    var b_s = LayoutTensor[DType.float32, Layout.row_major(N)](d_bs.unsafe_ptr())
    var bias = LayoutTensor[DType.bfloat16, Layout.row_major(1)](d_bias.unsafe_ptr())

    try:
        enqueue_apple_int8_matmul(
            lt_to_tt(c),
            lt_to_tt(a),
            lt_to_tt(b),
            lt_to_tt(a_s),
            lt_to_tt(b_s),
            lt_to_tt(bias),
            ctx,
        )
        ctx.synchronize()
        print("launched")
    except e:
        print("refused:", e)
