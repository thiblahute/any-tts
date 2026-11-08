// CUDA kernels for ops Candle runs as several passes over a tensor.

#include <cuda_bf16.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

// SnakeBeta, x + sin(alpha x)^2 * inv_beta per channel, in one pass over a
// contiguous (batch, channels, len) tensor; alpha and inv_beta are f32
// (exp(alpha), 1 / (exp(beta) + 1e-9)). Two values per thread.
extern "C" __global__ void snake_beta(const bf16 *__restrict__ x, const float *__restrict__ alpha,
                                      const float *__restrict__ inv_beta, bf16 *__restrict__ y,
                                      int channels, int len, long long n) {
    const long long stride = (long long)gridDim.x * blockDim.x * 2;
    for (long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 2; i < n; i += stride) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const long long at = i + j;
            if (at >= n) break;
            const int c = (int)((at / len) % channels);
            const float v = __bfloat162float(x[at]);
            const float s = sinf(alpha[c] * v);
            y[at] = __float2bfloat16(v + s * s * inv_beta[c]);
        }
    }
}
