// Kernels generating Qwen3-TTS frames on CUDA, one token (or the code
// predictor's two-token prefill) at a time, on static buffers so that a
// whole frame (talker step, code predictor, sampling) can be captured into
// one CUDA graph. Weights are bf16 and row-major (out x in), head_dim 128.
//
// The rounding follows Candle's separate ops where it shows: a projection's
// output is rounded to bf16 before its bias or residual is added, RoPE and
// the SiLU gate round each step as Candle's bf16 kernels do.

#include <cuda_bf16.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

#define HEAD_DIM 128

__device__ __forceinline__ float f32(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16(x); }
__device__ __forceinline__ float rnd(float x) { return f32(to_bf16(x)); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024).
__device__ float block_sum(float v) {
    __shared__ float partial[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    v = lane < (int)(blockDim.x >> 5) ? partial[lane] : 0.f;
    return warp_sum(v);
}

__device__ __forceinline__ void unpack8(uint4 u, float *f) {
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

// Per-utterance state, in device memory so that a captured frame reads it.
struct State {
    int pos;            // talker position of the next token (KV length)
    int step;           // frames generated so far
    int trailing_len;   // rows of trailing text
    int window;         // repetition window, 0 for all the history
    float temperature;  // talker sampling
    float penalty;      // repetition penalty
    int top_k;
    int history;        // tokens recorded in the history so far
    unsigned long long seed;
    unsigned long long draws;
};

// y[m] = x[m] / rms(x[m]) * w, rounded once, as Candle's rmsnorm. One block
// per row.
extern "C" __global__ void rms_norm(const bf16 *__restrict__ x, const bf16 *__restrict__ w,
                                    bf16 *__restrict__ y, int n, float eps) {
    x += (size_t)blockIdx.x * n;
    y += (size_t)blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = f32(x[i]);
        s += v * v;
    }
    const float scale = rsqrtf(block_sum(s) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = to_bf16(scale * f32(x[i]) * f32(w[i]));
}

// y[m][n] = sum_k w[n][k] x[m][k] for m < M (1 or 2), K a multiple of 8.
// x is staged in shared memory, where `gated` makes it Candle's
// silu(gate) * up of a [gate | up] row of 2K. A warp makes an output row,
// its lanes streaming 16 bytes of the weights at a time (evict-first: they
// are read once a frame, and the 4 MB of L2 could not keep them anyway),
// with f32 accumulation. The output is rounded, then the bias or the
// residual added and rounded again; `resid` may be `y` itself.
extern "C" __global__ void gemv(const bf16 *__restrict__ w, const bf16 *__restrict__ x, bf16 *y,
                                int K, int N, int M, int x_stride, int y_stride,
                                const bf16 *__restrict__ bias, const bf16 *resid,
                                int resid_stride, int gated) {
    extern __shared__ uint4 shared_x[];
    bf16 *xs = (bf16 *)shared_x;
    for (int m = 0; m < M; m++) {
        const bf16 *row = x + (size_t)m * x_stride;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            if (gated) {
                const float g = f32(row[k]), u = f32(row[K + k]);
                const float e = rnd(__expf(-g));
                const float sig = rnd(1.f / rnd(1.f + e));
                xs[m * K + k] = to_bf16(rnd(g * sig) * u);
            } else {
                xs[m * K + k] = row[k];
            }
        }
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (n >= N) return;
    const bf16 *wrow = w + (size_t)n * K;
    float acc0 = 0.f, acc1 = 0.f;
    for (int k = lane * 8; k < K; k += 256) {
        float wf[8], x0[8];
        unpack8(__ldcs((const uint4 *)(wrow + k)), wf);
        unpack8(*(const uint4 *)(xs + k), x0);
#pragma unroll
        for (int i = 0; i < 8; i++) acc0 += wf[i] * x0[i];
        if (M > 1) {
            float x1[8];
            unpack8(*(const uint4 *)(xs + K + k), x1);
#pragma unroll
            for (int i = 0; i < 8; i++) acc1 += wf[i] * x1[i];
        }
    }
    for (int m = 0; m < M; m++) {
        const float sum = warp_sum(m == 0 ? acc0 : acc1);
        if (lane == 0) {
            float v = rnd(sum);
            if (bias) v += f32(bias[n]);
            if (resid) v += f32(resid[(size_t)m * resid_stride + n]);
            y[(size_t)m * y_stride + n] = to_bf16(v);
        }
    }
}

// After the qkv projection of token m at position p = *pos + base + m: each
// q and k head RMS-normalized and rotated (half rotation), q to q_out, k
// and v to the caches ([kv_heads][cap][128]) at p. One block of 64 threads
// per head, a thread holding a rotation pair (d, d + 64).
extern "C" __global__ void qk_norm_rope(const bf16 *__restrict__ qkv, int qkv_stride,
                                        const bf16 *__restrict__ q_norm,
                                        const bf16 *__restrict__ k_norm, float eps,
                                        const bf16 *__restrict__ cos, const bf16 *__restrict__ sin,
                                        const int *__restrict__ pos, int base,
                                        bf16 *__restrict__ q_out, bf16 *__restrict__ k_cache,
                                        bf16 *__restrict__ v_cache, int cap, int heads,
                                        int kv_heads) {
    const int head = blockIdx.x, m = blockIdx.y, t = threadIdx.x;
    const int p = (pos ? *pos : 0) + base + m;
    const bf16 *row = qkv + (size_t)m * qkv_stride;
    if (head >= heads + kv_heads) {
        const int h = head - heads - kv_heads;
        const bf16 *src = row + (size_t)(heads + kv_heads + h) * HEAD_DIM;
        bf16 *dst = v_cache + ((size_t)h * cap + p) * HEAD_DIM;
        dst[t] = src[t];
        dst[t + 64] = src[t + 64];
        return;
    }
    const bool is_q = head < heads;
    const int h = is_q ? head : head - heads;
    const bf16 *src = row + (size_t)head * HEAD_DIM;
    float a = f32(src[t]), b = f32(src[t + 64]);
    const bf16 *norm = is_q ? q_norm : k_norm;
    if (norm) {
        __shared__ float partial[2];
        const float s = warp_sum(a * a + b * b);
        if ((t & 31) == 0) partial[t >> 5] = s;
        __syncthreads();
        const float scale = rsqrtf((partial[0] + partial[1]) / HEAD_DIM + eps);
        a = rnd(scale * a * f32(norm[t]));
        b = rnd(scale * b * f32(norm[t + 64]));
    }
    const float c = f32(cos[(size_t)p * HEAD_DIM + t]), s = f32(sin[(size_t)p * HEAD_DIM + t]);
    bf16 *dst = is_q ? q_out + ((size_t)m * heads + h) * HEAD_DIM
                     : k_cache + ((size_t)h * cap + p) * HEAD_DIM;
    dst[t] = to_bf16(rnd(a * c) - rnd(b * s));
    dst[t + 64] = to_bf16(rnd(a * s) + rnd(b * c));
}

#define ATTN_WARPS 4
#define MAX_GROUP 4

// Attention of token m (causal: keys 0..p) for the query heads sharing kv
// head blockIdx.x, over the keys of split blockIdx.y: each warp walks its
// keys with an online softmax, lane l holding dims 4l..4l+3, then the
// warps are merged into a partial (max, sum, weighted values) per head.
extern "C" __global__ void attn_partial(const bf16 *__restrict__ q,
                                        const bf16 *__restrict__ k_cache,
                                        const bf16 *__restrict__ v_cache,
                                        const int *__restrict__ pos, int base, int cap, int heads,
                                        int kv_heads, float scale, float *__restrict__ part_ml,
                                        float *__restrict__ part_acc) {
    const int kvh = blockIdx.x, split = blockIdx.y, splits = gridDim.y, m = blockIdx.z;
    const int group = heads / kv_heads;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int len = (pos ? *pos : 0) + base + m + 1;
    const int chunk = (len + splits - 1) / splits;
    const int start = split * chunk, end = min(len, start + chunk);

    float qf[MAX_GROUP][4], mx[MAX_GROUP], l[MAX_GROUP], acc[MAX_GROUP][4];
#pragma unroll
    for (int g = 0; g < MAX_GROUP; g++) {
        mx[g] = -INFINITY;
        l[g] = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            acc[g][i] = 0.f;
            qf[g][i] = g < group
                ? f32(q[((size_t)m * heads + kvh * group + g) * HEAD_DIM + lane * 4 + i]) * scale
                : 0.f;
        }
    }
    const bf16 *kbase = k_cache + (size_t)kvh * cap * HEAD_DIM + lane * 4;
    const bf16 *vbase = v_cache + (size_t)kvh * cap * HEAD_DIM + lane * 4;
    for (int j = start + warp; j < end; j += ATTN_WARPS) {
        const uint2 kr = *(const uint2 *)(kbase + (size_t)j * HEAD_DIM);
        const uint2 vr = *(const uint2 *)(vbase + (size_t)j * HEAD_DIM);
        const float kf[4] = {__uint_as_float(kr.x << 16), __uint_as_float(kr.x & 0xffff0000u),
                             __uint_as_float(kr.y << 16), __uint_as_float(kr.y & 0xffff0000u)};
        const float vf[4] = {__uint_as_float(vr.x << 16), __uint_as_float(vr.x & 0xffff0000u),
                             __uint_as_float(vr.y << 16), __uint_as_float(vr.y & 0xffff0000u)};
#pragma unroll
        for (int g = 0; g < MAX_GROUP; g++) {
            if (g >= group) break;
            const float d = warp_sum(qf[g][0] * kf[0] + qf[g][1] * kf[1] + qf[g][2] * kf[2] +
                                     qf[g][3] * kf[3]);
            const float nm = fmaxf(mx[g], d);
            const float corr = __expf(mx[g] - nm), p = __expf(d - nm);
            l[g] = l[g] * corr + p;
#pragma unroll
            for (int i = 0; i < 4; i++) acc[g][i] = acc[g][i] * corr + p * vf[i];
            mx[g] = nm;
        }
    }
    __shared__ float s_ml[ATTN_WARPS][MAX_GROUP][2];
    __shared__ float s_acc[ATTN_WARPS][MAX_GROUP][HEAD_DIM];
    for (int g = 0; g < group; g++) {
        if (lane == 0) {
            s_ml[warp][g][0] = mx[g];
            s_ml[warp][g][1] = l[g];
        }
#pragma unroll
        for (int i = 0; i < 4; i++) s_acc[warp][g][lane * 4 + i] = acc[g][i];
    }
    __syncthreads();
    // threadIdx.x indexes a dim: blockDim.x == HEAD_DIM.
    const int d = threadIdx.x;
    for (int g = 0; g < group; g++) {
        float M = -INFINITY;
        for (int w = 0; w < ATTN_WARPS; w++) M = fmaxf(M, s_ml[w][g][0]);
        float L = 0.f, A = 0.f;
        if (M != -INFINITY) {
            for (int w = 0; w < ATTN_WARPS; w++) {
                const float e = __expf(s_ml[w][g][0] - M);
                L += s_ml[w][g][1] * e;
                A += s_acc[w][g][d] * e;
            }
        }
        const size_t at = ((size_t)m * heads + kvh * group + g) * splits + split;
        if (d == 0) {
            part_ml[at * 2] = M;
            part_ml[at * 2 + 1] = L;
        }
        part_acc[at * HEAD_DIM + d] = A;
    }
}

// Merges the splits of attn_partial: out[m][h][d], one block of 128 per
// (head, token).
extern "C" __global__ void attn_combine(const float *__restrict__ part_ml,
                                        const float *__restrict__ part_acc,
                                        bf16 *__restrict__ out, int heads, int splits) {
    const int h = blockIdx.x, m = blockIdx.y, d = threadIdx.x;
    const size_t at = ((size_t)m * heads + h) * splits;
    float M = -INFINITY;
    for (int s = 0; s < splits; s++) M = fmaxf(M, part_ml[(at + s) * 2]);
    float L = 0.f, A = 0.f;
    for (int s = 0; s < splits; s++) {
        const float ms = part_ml[(at + s) * 2];
        if (ms == -INFINITY) continue;
        const float e = __expf(ms - M);
        L += part_ml[(at + s) * 2 + 1] * e;
        A += part_acc[(at + s) * HEAD_DIM + d] * e;
    }
    out[((size_t)m * heads + h) * HEAD_DIM + d] = to_bf16(A / L);
}

#define SAMPLE_THREADS 1024
#define SAMPLE_PER_THREAD 4
#define FRAME_SLOTS 4
#define FRAME_CODES 16

__device__ __forceinline__ uint32_t order_key(float x) {
    const uint32_t u = __float_as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float uniform(unsigned long long seed, unsigned long long draw,
                                         uint32_t i) {
    unsigned long long z = seed + draw * 0x9E3779B97F4A7C15ull + (unsigned long long)i * 0xD1B54A32D192ED03ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return ((float)(z >> 40) + 0.5f) * (1.0f / 16777216.0f);
}

// Samples one token from softmax(logits / temperature) over the top k, in
// one block: the k-th largest logit by a radix select (8 bits a pass), then
// Gumbel-max among the logits at or above it, ties included as the host
// sampler does. For the talker (`talker`), the logits first get the
// repetition penalty, the special-token suppression and, before the third
// frame, the EOS suppression, and the token joins the history. The token
// goes to `token` and to slot `code` of the frame's record.
extern "C" __global__ void sample(const bf16 *__restrict__ logits, int vocab, State *state,
                                  int talker, float temperature, int top_k, int suppress_start,
                                  int eos, uint32_t *__restrict__ counts,
                                  uint32_t *__restrict__ ring, uint32_t *__restrict__ token,
                                  uint32_t *__restrict__ frames, int code) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_k;
    __shared__ float s_score[32];
    __shared__ uint32_t s_index[32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int step = state->step;
    const unsigned long long seed = state->seed, draw = state->draws;
    if (talker) {
        temperature = state->temperature;
        top_k = state->top_k;
    }
    const float penalty = state->penalty;
    float v[SAMPLE_PER_THREAD];
    uint32_t key[SAMPLE_PER_THREAD];
#pragma unroll
    for (int i = 0; i < SAMPLE_PER_THREAD; i++) {
        const int idx = tid + i * SAMPLE_THREADS;
        float x = -INFINITY;
        if (idx < vocab) {
            x = f32(logits[idx]);
            if (talker) {
                if (counts[idx] > 0) x = x > 0.f ? x / penalty : x * penalty;
                if ((idx >= suppress_start && idx != eos) || (idx == eos && step < 2)) x = -INFINITY;
            }
        }
        v[i] = x;
        key[i] = order_key(x);
    }

    // The k-th largest key, exactly: four passes of 8 bits, most significant
    // first, over the keys matching the prefix found so far.
    uint32_t threshold = 0;
    if (temperature > 0.f && top_k > 0 && top_k < vocab) {
        uint32_t prefix = 0, mask = 0;
        uint32_t k = top_k;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = tid; i < 256; i += SAMPLE_THREADS) hist[i] = 0;
            __syncthreads();
#pragma unroll
            for (int i = 0; i < SAMPLE_PER_THREAD; i++)
                if (tid + i * SAMPLE_THREADS < vocab && (key[i] & mask) == prefix)
                    atomicAdd(&hist[(key[i] >> shift) & 255u], 1u);
            __syncthreads();
            if (warp == 0) {
                // Lane l sums bins 255 - 8l down to 248 - 8l: counts from the top.
                uint32_t c[8], total = 0;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    c[i] = hist[255 - lane * 8 - i];
                    total += c[i];
                }
                uint32_t above = total;  // inclusive scan over lanes
#pragma unroll
                for (int o = 1; o < 32; o <<= 1) {
                    const uint32_t n = __shfl_up_sync(0xffffffffu, above, o);
                    if (lane >= o) above += n;
                }
                above -= total;  // exclusive: keys in higher lanes' bins
                if (above < k && above + total >= k) {
                    for (int i = 0; i < 8; i++) {
                        if (above + c[i] >= k) {
                            s_prefix = prefix | ((uint32_t)(255 - lane * 8 - i) << shift);
                            s_k = k - above;
                            break;
                        }
                        above += c[i];
                    }
                }
            }
            __syncthreads();
            prefix = s_prefix;
            k = s_k;
            mask |= 255u << shift;
            __syncthreads();
        }
        threshold = prefix;
    }

    // Gumbel-max over the kept logits (plain argmax at temperature 0).
    float best = -INFINITY;
    uint32_t best_index = 0xffffffffu;
#pragma unroll
    for (int i = 0; i < SAMPLE_PER_THREAD; i++) {
        const uint32_t idx = tid + i * SAMPLE_THREADS;
        if (idx >= (uint32_t)vocab || !isfinite(v[i]) || key[i] < threshold) continue;
        float score = v[i];
        if (temperature > 0.f)
            score = v[i] / temperature - __logf(-__logf(uniform(seed, draw, idx)));
        if (score > best || (score == best && idx < best_index)) {
            best = score;
            best_index = idx;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const float other = __shfl_down_sync(0xffffffffu, best, o);
        const uint32_t other_index = __shfl_down_sync(0xffffffffu, best_index, o);
        if (other > best || (other == best && other_index < best_index)) {
            best = other;
            best_index = other_index;
        }
    }
    if (lane == 0) {
        s_score[warp] = best;
        s_index[warp] = best_index;
    }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < SAMPLE_THREADS / 32; w++) {
            if (s_score[w] > best || (s_score[w] == best && s_index[w] < best_index)) {
                best = s_score[w];
                best_index = s_index[w];
            }
        }
        const uint32_t t = best_index == 0xffffffffu ? 0u : best_index;
        *token = t;
        frames[(step % FRAME_SLOTS) * FRAME_CODES + code] = t;
        state->draws = draw + 1;
        if (talker) {
            const int n = state->history, window = state->window;
            if (window > 0) {
                if (n >= window) counts[ring[n % window]] -= 1;
                ring[n % window] = t;
            }
            counts[t] += 1;
            state->history = n + 1;
        }
    }
}

// out = table[*token] (n values); then sum = out (mode 1), or
// sum = bf16(sum + out) (mode 2), as Candle's add.
extern "C" __global__ void embed(const bf16 *__restrict__ table, const uint32_t *__restrict__ token,
                                 int n, bf16 *__restrict__ out, bf16 *__restrict__ sum, int mode) {
    const bf16 *row = table + (size_t)(*token) * n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const bf16 e = row[i];
        if (out) out[i] = e;
        if (mode == 1) sum[i] = e;
        else if (mode == 2) sum[i] = to_bf16(f32(sum[i]) + f32(e));
    }
}

// The talker's next input: the frame's summed embeddings plus the trailing
// text row of this step (the last one once past them).
extern "C" __global__ void talker_input(const bf16 *__restrict__ sum,
                                        const bf16 *__restrict__ text, const State *state, int n,
                                        bf16 *__restrict__ x) {
    const int row = min(state->step, state->trailing_len - 1);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] = to_bf16(f32(sum[i]) + f32(text[(size_t)row * n + i]));
}

extern "C" __global__ void copy_rows(const bf16 *__restrict__ src, bf16 *__restrict__ dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}

// Copies a prefill's cache (heads of len rows, src_stride apart) into a
// cache of dst_stride per head.
extern "C" __global__ void copy_cache(const bf16 *__restrict__ src, long long src_stride,
                                      bf16 *__restrict__ dst, long long dst_stride, int n) {
    const int h = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        dst[h * dst_stride + i] = src[h * src_stride + i];
}

extern "C" __global__ void advance(State *state) {
    state->pos += 1;
    state->step += 1;
}
