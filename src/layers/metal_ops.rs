//! Metal kernels for autoregressive decoding, where Candle's generic paths
//! are slow.
//!
//! - [`Linear`]: decoding multiplies one row at a time, which Candle's
//!   Metal matmul runs through its tiled GEMM at a fraction of the memory
//!   bandwidth; a bf16 GEMV kernel (one SIMD group per four output rows,
//!   16-byte loads, f32 accumulation) takes those, `candle_nn::Linear` the
//!   rest. [`stacked`] merges projections of the same input into one.
//! - [`sample_top_k`]: top-k sampling at a temperature in one kernel,
//!   instead of a sort, softmax and draw of a dozen kernels each.

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

#[derive(Clone, Debug)]
pub struct Linear {
    inner: candle_nn::Linear,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self {
            inner: candle_nn::Linear::new(weight, bias),
        }
    }

    pub fn weight(&self) -> &Tensor {
        self.inner.weight()
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.inner.bias()
    }
}

pub fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    Ok(Linear {
        inner: candle_nn::linear(in_dim, out_dim, vb)?,
    })
}

/// Several projections of the same input as one: their weights (and biases)
/// stacked, so a decoding step reads them in one kernel. Split the output
/// with `narrow` on the last dimension, in the order given.
pub fn stacked(parts: &[Linear]) -> Result<Linear> {
    let weights: Vec<&Tensor> = parts.iter().map(|part| part.weight()).collect();
    let weight = Tensor::cat(&weights, 0)?;
    let bias = if parts.iter().all(|part| part.bias().is_some()) {
        let biases: Vec<&Tensor> = parts.iter().filter_map(|part| part.bias()).collect();
        Some(Tensor::cat(&biases, 0)?)
    } else if parts.iter().all(|part| part.bias().is_none()) {
        None
    } else {
        candle_core::bail!("stacked projections must all have a bias, or none")
    };
    Ok(Linear::new(weight, bias))
}

pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    Ok(Linear {
        inner: candle_nn::linear_no_bias(in_dim, out_dim, vb)?,
    })
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Some(y) = metal::gemv(x, self.inner.weight())? {
            return match self.inner.bias() {
                Some(bias) => y.broadcast_add(bias),
                None => Ok(y),
            };
        }
        self.inner.forward(x)
    }
}

#[cfg(feature = "metal")]
mod metal {
    use std::cell::RefCell;
    use std::collections::HashMap;

    use candle_core::backend::BackendStorage;
    use candle_core::{CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor};
    use candle_metal_kernels::metal::ComputePipeline;
    use objc2_metal::{MTLResourceUsage, MTLSize};

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// y[n] = dot(x, w[n, :]) for a row-major w of N rows of K: each SIMD group
// computes ROWS outputs, each lane reading 8 contiguous values per step.
constant uint ROWS = 4;
constant uint SIMDS = 8;

kernel void gemv_bf16(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device bfloat *y [[buffer(2)]],
    constant uint &K [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint row0 = (tg * SIMDS + sg) * ROWS;
    if (row0 >= N) return;
    float acc[ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint full = K & ~7u;
    for (uint k = lane * 8; k < full; k += 256) {
        const float4 xa = float4(*((device const bfloat4 *)(x + k)));
        const float4 xb = float4(*((device const bfloat4 *)(x + k + 4)));
        for (uint r = 0; r < ROWS; r++) {
            const uint row = row0 + r;
            if (row < N) {
                device const bfloat *wr = w + (ulong)row * K + k;
                const float4 wa = float4(*((device const bfloat4 *)wr));
                const float4 wb = float4(*((device const bfloat4 *)(wr + 4)));
                acc[r] += dot(xa, wa) + dot(xb, wb);
            }
        }
    }
    for (uint k = full + lane; k < K; k += 32) {
        for (uint r = 0; r < ROWS; r++) {
            const uint row = row0 + r;
            if (row < N) acc[r] += float(x[k]) * float(w[(ulong)row * K + k]);
        }
    }
    for (uint r = 0; r < ROWS; r++) {
        const float sum = simd_sum(acc[r]);
        const uint row = row0 + r;
        if (lane == 0 && row < N) y[row] = bfloat(sum);
    }
}
"#;

    const SAMPLE_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Samples one token from softmax(logits / temperature) restricted to the
// top_k logits, in one threadgroup: a bitonic sort in threadgroup memory
// gives the k-th largest logit, then Gumbel-max picks among those at or
// above it. Noise comes from a hash of (seed, index).
constant uint CAPACITY = 4096;

static inline float uniform01(uint2 seed, uint i) {
    uint h = seed.x ^ (i * 0x9E3779B9u);
    h ^= seed.y + 0x7F4A7C15u + (h << 6) + (h >> 2);
    h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
    return (float(h >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

kernel void sample_top_k(
    device const bfloat *logits [[buffer(0)]],
    device uint *out [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &padded [[buffer(3)]],
    constant uint &k [[buffer(4)]],
    constant float &inv_temperature [[buffer(5)]],
    constant uint2 &seed [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]])
{
    threadgroup float sorted[CAPACITY];
    threadgroup float best_score[32];
    threadgroup uint best_index[32];
    for (uint i = tid; i < padded; i += threads) {
        sorted[i] = i < n ? float(logits[i]) : -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Bitonic sort, descending.
    for (uint size = 2; size <= padded; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint i = tid; i < padded; i += threads) {
                uint j = i ^ stride;
                if (j > i) {
                    bool descending = (i & size) == 0;
                    float a = sorted[i], b = sorted[j];
                    if ((a < b) == descending) { sorted[i] = b; sorted[j] = a; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    const float threshold = sorted[min(k, n) - 1];
    float score = -INFINITY;
    uint index = 0;
    for (uint i = tid; i < n; i += threads) {
        float logit = float(logits[i]);
        if (logit >= threshold) {
            float g = -log(-log(uniform01(seed, i)));
            float s = logit * inv_temperature + g;
            if (s > score) { score = s; index = i; }
        }
    }
    // Argmax across the threadgroup.
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float other = simd_shuffle_down(score, offset);
        uint other_index = simd_shuffle_down(index, offset);
        if (other > score) { score = other; index = other_index; }
    }
    if (lane == 0) { best_score[simd] = score; best_index[simd] = index; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint groups = (threads + 31) / 32;
        float s = best_score[0]; uint b = best_index[0];
        for (uint g = 1; g < groups; g++) {
            if (best_score[g] > s) { s = best_score[g]; b = best_index[g]; }
        }
        out[0] = b;
    }
}
"#;

    thread_local! {
        static SAMPLE_PIPELINES: RefCell<HashMap<u64, ComputePipeline>> = RefCell::new(HashMap::new());
    }

    struct SampleTopK {
        top_k: usize,
        temperature: f32,
        seed: u64,
    }

    impl candle_core::CustomOp1 for SampleTopK {
        fn name(&self) -> &'static str {
            "sample-top-k"
        }

        fn cpu_fwd(
            &self,
            _: &candle_core::CpuStorage,
            _: &Layout,
        ) -> Result<(candle_core::CpuStorage, Shape)> {
            candle_core::bail!("sample-top-k runs on Metal only")
        }

        fn metal_fwd(&self, logits: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
            let device = logits.device();
            let n = layout.shape().elem_count();
            let padded = n.next_power_of_two().max(2);
            if padded > 4096 {
                candle_core::bail!("sample-top-k supports up to 4096 logits, got {n}")
            }
            let key = device.metal_device().registry_id();
            let pipeline = SAMPLE_PIPELINES.with(|cell| -> Result<ComputePipeline> {
                if let Some(found) = cell.borrow().get(&key) {
                    return Ok(found.clone());
                }
                let library = device
                    .metal_device()
                    .new_library_with_source(SAMPLE_SOURCE, None)
                    .map_err(candle_core::Error::wrap)?;
                let function = library
                    .get_function("sample_top_k", None)
                    .map_err(candle_core::Error::wrap)?;
                let pipeline = device
                    .metal_device()
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(candle_core::Error::wrap)?;
                cell.borrow_mut().insert(key, pipeline.clone());
                Ok(pipeline)
            })?;
            let output = device.new_buffer(1, DType::U32, "sample")?;
            let encoder = device.command_encoder()?;
            encoder.set_label("sample-top-k");
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(logits.buffer()), layout.start_offset() * 2);
            encoder.set_buffer(1, Some(&output), 0);
            encoder.set_bytes(2, &(n as u32));
            encoder.set_bytes(3, &(padded as u32));
            encoder.set_bytes(4, &(self.top_k as u32));
            encoder.set_bytes(5, &(1.0f32 / self.temperature));
            encoder.set_bytes(6, &[self.seed as u32, (self.seed >> 32) as u32]);
            encoder.use_resource(logits.buffer(), MTLResourceUsage::Read);
            encoder.use_resource(output.as_ref(), MTLResourceUsage::Write);
            encoder.dispatch_thread_groups(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 1024, height: 1, depth: 1 },
            );
            Ok((
                MetalStorage::new(output, device.clone(), 1, DType::U32),
                Shape::from(1),
            ))
        }
    }

    pub(super) fn sample_top_k(
        logits: &Tensor,
        top_k: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<Option<Tensor>> {
        if !logits.device().is_metal() || logits.dtype() != DType::BF16 || top_k == 0 {
            return Ok(None);
        }
        let logits = logits.flatten_all()?.contiguous()?;
        logits
            .apply_op1_no_bwd(&SampleTopK {
                top_k,
                temperature: temperature as f32,
                seed,
            })
            .map(Some)
    }

    thread_local! {
        static PIPELINES: RefCell<HashMap<u64, ComputePipeline>> = RefCell::new(HashMap::new());
    }

    fn pipeline(device: &candle_core::MetalDevice) -> Result<ComputePipeline> {
        let key = device.metal_device().registry_id();
        PIPELINES.with(|cell| {
            if let Some(found) = cell.borrow().get(&key) {
                return Ok(found.clone());
            }
            let library = device
                .metal_device()
                .new_library_with_source(SOURCE, None)
                .map_err(candle_core::Error::wrap)?;
            let function = library
                .get_function("gemv_bf16", None)
                .map_err(candle_core::Error::wrap)?;
            let pipeline = device
                .metal_device()
                .new_compute_pipeline_state_with_function(&function)
                .map_err(candle_core::Error::wrap)?;
            cell.borrow_mut().insert(key, pipeline.clone());
            Ok(pipeline)
        })
    }

    struct Gemv;

    impl CustomOp2 for Gemv {
        fn name(&self) -> &'static str {
            "gemv-bf16"
        }

        fn cpu_fwd(
            &self,
            _: &candle_core::CpuStorage,
            _: &Layout,
            _: &candle_core::CpuStorage,
            _: &Layout,
        ) -> Result<(candle_core::CpuStorage, Shape)> {
            candle_core::bail!("gemv-bf16 runs on Metal only")
        }

        fn metal_fwd(
            &self,
            x: &MetalStorage,
            lx: &Layout,
            w: &MetalStorage,
            lw: &Layout,
        ) -> Result<(MetalStorage, Shape)> {
            let device = x.device();
            let (n, k) = lw.shape().dims2()?;
            let pipeline = pipeline(device)?;
            let output = device.new_buffer(n, DType::BF16, "gemv")?;
            let encoder = device.command_encoder()?;
            encoder.set_label("gemv-bf16");
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(x.buffer()), lx.start_offset() * 2);
            encoder.set_buffer(1, Some(w.buffer()), lw.start_offset() * 2);
            encoder.set_buffer(2, Some(&output), 0);
            encoder.set_bytes(3, &(k as u32));
            encoder.set_bytes(4, &(n as u32));
            encoder.use_resource(x.buffer(), MTLResourceUsage::Read);
            encoder.use_resource(w.buffer(), MTLResourceUsage::Read);
            encoder.use_resource(output.as_ref(), MTLResourceUsage::Write);
            let rows_per_group = 4 * 8;
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: n.div_ceil(rows_per_group),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            let mut dims = lx.dims().to_vec();
            *dims.last_mut().unwrap() = n;
            Ok((
                MetalStorage::new(output, device.clone(), n, DType::BF16),
                Shape::from(dims),
            ))
        }
    }

    /// `x @ w.T` through the GEMV kernel when `x` is one bf16 row on Metal
    /// and both operands are contiguous and 16-byte aligned.
    pub(super) fn gemv(x: &Tensor, w: &Tensor) -> Result<Option<Tensor>> {
        if !x.device().is_metal() || x.dtype() != DType::BF16 || w.dtype() != DType::BF16 {
            return Ok(None);
        }
        let Ok((_, k)) = w.dims2() else {
            return Ok(None);
        };
        if x.elem_count() != k || !x.is_contiguous() || !w.is_contiguous() {
            return Ok(None);
        }
        let aligned = |t: &Tensor| {
            let (storage, layout) = t.storage_and_layout();
            drop(storage);
            layout.start_offset() % 8 == 0
        };
        if !aligned(x) || !aligned(w) {
            return Ok(None);
        }
        x.apply_op2_no_bwd(w, &Gemv).map(Some)
    }
}

/// One token sampled from softmax(logits / temperature) over the `top_k`
/// largest logits, in a single Metal kernel: a (1,) u32 tensor that stays
/// on the GPU. `None` when the logits are not bf16 on Metal.
#[cfg(feature = "metal")]
pub fn sample_top_k(logits: &Tensor, top_k: usize, temperature: f64, seed: u64) -> Result<Option<Tensor>> {
    metal::sample_top_k(logits, top_k, temperature, seed)
}
