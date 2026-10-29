//! 1-D convolutions on Metal as one implicit GEMM each, for the audio
//! decoders' long sequences: Candle's Metal conv1d builds the whole im2col
//! matrix (channels x taps x samples) in memory before its matmul, and its
//! transposed convolution runs one thread per output sample.
//!
//! The kernel tiles the output, 32 rows (output channels) x 128 samples or
//! 64 x 64 per threadgroup of four SIMD groups, each accumulating a 32 x 32
//! block in 8 x 8 simdgroup matrices of f32. For each block of 8 input
//! channels it loads the weights of every tap and the input window the taps
//! cover into threadgroup memory once, so the taps' shifted inputs are
//! offsets into the same tile. The weights are packed at load time in the
//! order the tiles read them, and the tiles hold f16, at least as precise as
//! bf16 over the range of values here. Small tiles and blocks leave room for
//! more threadgroups per core, which hides the loads better than larger
//! tiles reuse them.
//!
//! A causal transposed convolution whose kernel is a multiple of its stride
//! is the same GEMM: output sample `q * stride + r` is a plain convolution of
//! the input around `q` with the weights of phase `r`, so the phases are
//! stacked as output rows and written interleaved.
//!
//! The kernel can apply SnakeBeta to its input as it loads it, and add a
//! bias and a residual to its output, which saves a pass over the tensor
//! each.

use candle_core::{DType, Module, Result, Tensor};

/// Input channels per block of the GEMM's inner dimension.
const CHANNELS: usize = 8;
/// Up to this many outputs per phase, a single column of tiles where each
/// threadgroup runs through all the inputs alone, Candle's GEMM splits the
/// work better.
const MIN_STEPS: usize = 128;
/// Threadgroup memory available to a threadgroup.
const MAX_MEMORY: usize = 32 * 1024;

/// SnakeBeta's parameters, `x + sin(exp(alpha) x)^2 / (exp(beta) + 1e-9)`,
/// precomputed in f32 for the convolution kernel.
#[derive(Clone, Debug)]
pub struct Snake {
    alpha: Tensor,
    inv_beta: Tensor,
}

impl Snake {
    /// From the log-scale `alpha` and `beta` as the checkpoints store them.
    pub fn new(alpha: &Tensor, beta: &Tensor) -> Result<Self> {
        let alpha = alpha.to_dtype(DType::F32)?.exp()?.contiguous()?;
        let inv_beta = (beta.to_dtype(DType::F32)?.exp()? + 1e-9)?
            .recip()?
            .contiguous()?;
        Ok(Self { alpha, inv_beta })
    }

    /// SnakeBeta on a (batch, channels, len) tensor, in f32.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = (1, self.alpha.dim(0)?, 1);
        let h = x.to_dtype(DType::F32)?;
        let s = h.broadcast_mul(&self.alpha.reshape(shape)?)?.sin()?.sqr()?;
        (h + s.broadcast_mul(&self.inv_beta.reshape(shape)?)?)?.to_dtype(x.dtype())
    }
}

#[derive(Clone, Debug)]
enum Fallback {
    Conv(candle_nn::Conv1d),
    Transposed(candle_nn::ConvTranspose1d),
}

/// A causal convolution on Metal, its input already padded on the left
/// (with the previous chunk's last samples when streaming): through the GEMM
/// kernel, or through Candle's ops when the output is too short to fill
/// the GPU.
#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub struct MetalConv {
    /// (row tiles, channel blocks, taps, tile rows, CHANNELS) in f16.
    weight: Tensor,
    bias: Option<Tensor>,
    fallback: Fallback,
    in_channels: usize,
    out_channels: usize,
    /// 1, or the stride of a transposed convolution.
    phases: usize,
    taps: usize,
    dilation: usize,
    /// Output rows and samples per tile: 32 x 128 or 64 x 64.
    tile: (usize, usize),
}

impl MetalConv {
    /// A convolution without padding. `None` off Metal or for shapes the
    /// kernel does not cover.
    pub fn conv1d(conv: &candle_nn::Conv1d) -> Result<Option<Self>> {
        let weight = conv.weight();
        let (out_channels, in_channels, taps) = weight.dims3()?;
        let config = conv.config();
        if config.padding != 0 || config.stride != 1 || config.groups != 1 {
            return Ok(None);
        }
        let tile = Self::tile(out_channels);
        if !Self::fits(weight, in_channels, taps, config.dilation, tile) {
            return Ok(None);
        }
        // (taps, out, in)
        let rows = weight.permute((2, 0, 1))?;
        let fallback = Fallback::Conv(conv.clone());
        Self::pack(
            &rows,
            conv.bias(),
            fallback,
            out_channels,
            1,
            config.dilation,
            tile,
        )
        .map(Some)
    }

    /// A causal transposed convolution, its kernel a multiple of its stride,
    /// without the right overhang: (len - kernel / stride + 1) * stride
    /// samples out of `len`, whose first `kernel / stride - 1` are the
    /// previous inputs. `None` off Metal or for shapes the kernel does not
    /// cover.
    pub fn conv_transpose1d(conv: &candle_nn::ConvTranspose1d) -> Result<Option<Self>> {
        let weight = conv.weight();
        let (in_channels, out_channels, kernel) = weight.dims3()?;
        let config = conv.config();
        let stride = config.stride;
        if config.padding != 0
            || config.output_padding != 0
            || config.dilation != 1
            || config.groups != 1
        {
            return Ok(None);
        }
        if stride == 0 || kernel % stride != 0 {
            return Ok(None);
        }
        let taps = kernel / stride;
        let tile = Self::tile(stride * out_channels);
        if !Self::fits(weight, in_channels, taps, 1, tile) {
            return Ok(None);
        }
        // Tap t reads input q + t - (taps - 1), through kernel index
        // r + (taps - 1 - t) * stride for phase r. (Candle's Metal
        // index_select reads a strided input wrongly, hence the copy.)
        let reversed: Vec<u32> = (0..taps as u32).rev().collect();
        let reversed = Tensor::new(reversed.as_slice(), weight.device())?;
        let rows = weight
            .reshape((in_channels, out_channels, taps, stride))?
            .permute((2, 3, 1, 0))? // (taps, phase, out, in)
            .contiguous()?
            .index_select(&reversed, 0)?
            .reshape((taps, stride * out_channels, in_channels))?;
        let fallback = Fallback::Transposed(conv.clone());
        Self::pack(&rows, conv.bias(), fallback, out_channels, stride, 1, tile).map(Some)
    }

    /// 64 x 64 tiles reuse each input more, 32 x 128 waste fewer rows of
    /// few-channel outputs.
    fn tile(rows: usize) -> (usize, usize) {
        if rows >= 192 {
            (64, 64)
        } else {
            (32, 128)
        }
    }

    fn fits(
        weight: &Tensor,
        in_channels: usize,
        taps: usize,
        dilation: usize,
        tile: (usize, usize),
    ) -> bool {
        cfg!(feature = "metal")
            && weight.device().is_metal()
            && taps > 0
            && in_channels % CHANNELS == 0
            && Self::memory(taps, dilation, tile).0 <= MAX_MEMORY
    }

    /// The threadgroup memory a tile takes, in bytes, and its input
    /// window's row stride: the weights of all taps, then the window, in f16.
    fn memory(taps: usize, dilation: usize, (rows, columns): (usize, usize)) -> (usize, usize) {
        let stride = (columns + (taps - 1) * dilation).next_multiple_of(8);
        let bytes = (taps * rows * CHANNELS + CHANNELS * stride) * 2;
        (bytes.next_multiple_of(16), stride)
    }

    /// Packs (taps, rows, in) weights tile by tile.
    fn pack(
        rows: &Tensor,
        bias: Option<&Tensor>,
        fallback: Fallback,
        out_channels: usize,
        phases: usize,
        dilation: usize,
        tile: (usize, usize),
    ) -> Result<Self> {
        let (taps, count, in_channels) = rows.dims3()?;
        let tiles = count.div_ceil(tile.0);
        let blocks = in_channels / CHANNELS;
        let weight = rows
            .to_dtype(DType::F32)?
            .pad_with_zeros(1, 0, tiles * tile.0 - count)?
            .reshape((taps, tiles, tile.0, blocks, CHANNELS))?
            .permute((1, 3, 0, 2, 4))?
            .to_dtype(DType::F16)?
            .contiguous()?;
        let bias = bias
            .map(|b| b.to_dtype(DType::F32)?.contiguous())
            .transpose()?;
        Ok(Self {
            weight,
            bias,
            fallback,
            in_channels,
            out_channels,
            phases,
            taps,
            dilation,
            tile,
        })
    }

    /// Whether [`MetalConv::forward`] takes `x`: bf16 on Metal.
    pub fn supports(&self, x: &Tensor) -> bool {
        x.device().is_metal() && x.dtype() == DType::BF16
    }

    /// `conv(snake(x)) + bias + residual` for `x` of (batch, in, len) that
    /// [`MetalConv::supports`], `residual` shaped as the output.
    pub fn forward(
        &self,
        x: &Tensor,
        snake: Option<&Snake>,
        residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        let steps = self.steps(x)?;
        if steps > MIN_STEPS {
            return self.gemm(x, snake, residual);
        }
        let h = match snake {
            Some(snake) => snake.forward(x)?,
            None => x.clone(),
        };
        let y = match &self.fallback {
            Fallback::Conv(conv) => conv.forward(&h)?,
            Fallback::Transposed(conv) => {
                let skip = (self.taps - 1) * self.phases;
                conv.forward(&h)?.narrow(2, skip, steps * self.phases)?
            }
        };
        match residual {
            Some(residual) => y + residual,
            None => Ok(y),
        }
    }

    /// [`MetalConv::forward`] through the GEMM kernel whatever the size.
    pub fn gemm(
        &self,
        x: &Tensor,
        snake: Option<&Snake>,
        residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.steps(x)?;
        #[cfg(feature = "metal")]
        return metal::forward(self, x, snake, residual);
        #[cfg(not(feature = "metal"))]
        {
            let _ = (snake, residual);
            candle_core::bail!("metal-conv: built without Metal")
        }
    }

    /// Outputs per phase for `x`, or an error if `x` does not fit.
    fn steps(&self, x: &Tensor) -> Result<usize> {
        let (_, in_channels, len) = x.dims3()?;
        let reach = (self.taps - 1) * self.dilation;
        if !self.supports(x) || in_channels != self.in_channels || len <= reach {
            candle_core::bail!(
                "metal-conv: unsupported input {:?} {:?}",
                x.shape(),
                x.dtype()
            )
        }
        Ok(len - reach)
    }
}

/// A depthwise convolution without padding, `weight` of (channels, 1,
/// taps), in one kernel. `None` unless `x` and the weights are bf16 on Metal.
pub fn depthwise_conv1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    let (_, channels, len) = x.dims3()?;
    let (out, group, taps) = weight.dims3()?;
    let bf16 = |t: &Tensor| t.device().is_metal() && t.dtype() == DType::BF16;
    if !cfg!(feature = "metal")
        || !bf16(x)
        || !bf16(weight)
        || !bias.map_or(true, bf16)
        || out != channels
        || group != 1
        || len < taps
    {
        return Ok(None);
    }
    #[cfg(feature = "metal")]
    return metal::depthwise(x, weight, bias).map(Some);
    #[cfg(not(feature = "metal"))]
    Ok(None)
}

#[cfg(feature = "metal")]
mod metal {
    use std::cell::RefCell;
    use std::collections::HashMap;

    use candle_core::backend::BackendStorage;
    use candle_core::{CustomOp2, DType, Layout, MetalStorage, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::{Buffer, ComputePipeline};
    use objc2_metal::{MTLResourceUsage, MTLSize};

    use super::{MetalConv, Result, Snake};

    const SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct Params {
    uint in_channels;
    uint out_channels;
    uint phases;    // 1, or the stride of a transposed convolution
    uint steps;     // outputs per phase
    uint len;       // input length
    uint taps;
    uint dilation;
    uint stride;    // of the input window's rows in threadgroup memory
    uint flags;
};

constant uint SNAKE = 1, BIAS = 2, RESIDUAL = 4;
constant uint BK = 8, THREADS = 128;

// y[o, n * phases + r] = sum over taps t and inputs c of
// w(r, o; t, c) * snake(x[c, n + t * dilation]), plus bias and residual,
// for a BM x BN tile of rows (r, o) and outputs n.
template <uint BM, uint BN>
void conv_gemm(
    device const bfloat *x,
    device const half *w,
    device const float *bias,
    device const float *alpha,
    device const float *inv_beta,
    device const bfloat *res,
    device bfloat *y,
    constant Params &p,
    uint3 tg,
    uint sg,
    uint lane,
    uint tid,
    threadgroup half *ws)   // [taps][BM][BK], then the window [BK][stride]
{
    // Each SIMD group computes 32 x 32.
    constexpr uint WN = BN / 32;
    const uint sm = (sg / WN) * 32, sn = (sg % WN) * 32;
    threadgroup half *xs = ws + p.taps * BM * BK;

    const uint rows = p.phases * p.out_channels;
    const uint m0 = tg.y * BM, n0 = tg.x * BN;
    const uint span = min(BN + (p.taps - 1) * p.dilation, p.len - n0);
    const uint block = p.taps * BM * BK;
    x += (ulong)tg.z * p.in_channels * p.len;
    w += (ulong)tg.y * (p.in_channels / BK) * block;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint c0 = 0; c0 < p.in_channels; c0 += BK, w += block) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < block / 4; i += THREADS)
            ((threadgroup half4 *)ws)[i] = ((device const half4 *)w)[i];
        for (uint c = 0; c < BK; c++) {
            device const bfloat *row = x + (ulong)(c0 + c) * p.len + n0;
            const float a = (p.flags & SNAKE) ? alpha[c0 + c] : 0.0f;
            const float b = (p.flags & SNAKE) ? inv_beta[c0 + c] : 0.0f;
            for (uint s = tid; s < p.stride; s += THREADS) {
                float v = 0.0f;
                if (s < span) {
                    v = float(row[s]);
                    if (p.flags & SNAKE) {
                        const float t = sin(a * v);
                        v += t * t * b;
                    }
                }
                xs[c * p.stride + s] = half(v);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < p.taps; t++) {
            simdgroup_half8x8 a[4], b[4];
            for (uint i = 0; i < 4; i++)
                simdgroup_load(a[i], ws + (t * BM + sm + i * 8) * BK, BK);
            for (uint j = 0; j < 4; j++)
                simdgroup_load(b[j], xs + sn + j * 8 + t * p.dilation, p.stride);
            for (uint i = 0; i < 4; i++)
                for (uint j = 0; j < 4; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
    }

    // Each lane holds two adjacent values of every 8 x 8 result.
    const uint quad = lane / 4;
    const uint fm = (quad & 4) + (lane / 2) % 4;
    const uint fn = (quad & 2) * 2 + (lane % 2) * 2;
    const ulong out_len = (ulong)p.steps * p.phases;
    const ulong y_off = (ulong)tg.z * p.out_channels * out_len;
    for (uint i = 0; i < 4; i++) {
        const uint row = m0 + sm + i * 8 + fm;
        if (row >= rows) continue;
        const uint r = row / p.out_channels, o = row % p.out_channels;
        const float add = (p.flags & BIAS) ? bias[o] : 0.0f;
        for (uint j = 0; j < 4; j++) {
            const auto values = acc[i][j].thread_elements();
            for (uint q = 0; q < 2; q++) {
                const uint n = n0 + sn + j * 8 + fn + q;
                if (n >= p.steps) continue;
                const ulong index = y_off + (ulong)o * out_len + (ulong)n * p.phases + r;
                float v = values[q] + add;
                if (p.flags & RESIDUAL) v += float(res[index]);
                y[index] = bfloat(v);
            }
        }
    }
}

#define CONV_GEMM(BM, BN)                                                      \
kernel void conv_gemm_##BM##x##BN(                                             \
    device const bfloat *x [[buffer(0)]],                                      \
    device const half *w [[buffer(1)]],                                        \
    device const float *bias [[buffer(2)]],                                    \
    device const float *alpha [[buffer(3)]],                                   \
    device const float *inv_beta [[buffer(4)]],                                \
    device const bfloat *res [[buffer(5)]],                                    \
    device bfloat *y [[buffer(6)]],                                            \
    constant Params &p [[buffer(7)]],                                          \
    uint3 tg [[threadgroup_position_in_grid]],                                 \
    uint sg [[simdgroup_index_in_threadgroup]],                                \
    uint lane [[thread_index_in_simdgroup]],                                   \
    uint tid [[thread_index_in_threadgroup]],                                  \
    threadgroup half *ws [[threadgroup(0)]])                                   \
{                                                                              \
    conv_gemm<BM, BN>(x, w, bias, alpha, inv_beta, res, y, p, tg, sg, lane, tid, ws); \
}

CONV_GEMM(32, 128)
CONV_GEMM(64, 64)

// A depthwise convolution, one filter per channel, one output per thread.
kernel void depthwise_conv(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat *y [[buffer(3)]],
    constant uint4 &p [[buffer(4)]],    // len, steps, taps, has bias
    uint3 gid [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]])
{
    const uint len = p.x, steps = p.y, taps = p.z;
    if (gid.x >= steps) return;
    const uint channel = gid.y;
    const ulong row = (ulong)gid.z * grid.y + channel;
    device const bfloat *in = x + row * len + gid.x;
    float acc = p.w ? float(bias[channel]) : 0.0f;
    for (uint k = 0; k < taps; k++) acc += float(w[channel * taps + k]) * float(in[k]);
    y[row * steps + gid.x] = bfloat(acc);
}
"#;

    thread_local! {
        static PIPELINES: RefCell<HashMap<(u64, &'static str), ComputePipeline>> = RefCell::new(HashMap::new());
    }

    fn pipeline(device: &candle_core::MetalDevice, name: &'static str) -> Result<ComputePipeline> {
        let key = (device.metal_device().registry_id(), name);
        PIPELINES.with(|cell| {
            if let Some(found) = cell.borrow().get(&key) {
                return Ok(found.clone());
            }
            let library = device
                .metal_device()
                .new_library_with_source(SOURCE, None)
                .map_err(candle_core::Error::wrap)?;
            let function = library
                .get_function(name, None)
                .map_err(candle_core::Error::wrap)?;
            let pipeline = device
                .metal_device()
                .new_compute_pipeline_state_with_function(&function)
                .map_err(candle_core::Error::wrap)?;
            cell.borrow_mut().insert(key, pipeline.clone());
            Ok(pipeline)
        })
    }

    #[repr(C)]
    struct Params {
        in_channels: u32,
        out_channels: u32,
        phases: u32,
        steps: u32,
        len: u32,
        taps: u32,
        dilation: u32,
        stride: u32,
        flags: u32,
    }

    struct Conv<'a> {
        conv: &'a MetalConv,
        snake: Option<&'a Snake>,
        residual: Option<&'a Tensor>,
    }

    /// Runs `f` with a contiguous tensor's Metal buffer and byte offset.
    fn with_buffer<T>(tensor: &Tensor, f: impl FnOnce(&Buffer, usize) -> Result<T>) -> Result<T> {
        let (storage, layout) = tensor.storage_and_layout();
        let Storage::Metal(storage) = &*storage else {
            candle_core::bail!("conv-gemm: tensor not on Metal")
        };
        f(
            storage.buffer(),
            layout.start_offset() * tensor.dtype().size_in_bytes(),
        )
    }

    impl CustomOp2 for Conv<'_> {
        fn name(&self) -> &'static str {
            "conv-gemm"
        }

        fn cpu_fwd(
            &self,
            _: &candle_core::CpuStorage,
            _: &Layout,
            _: &candle_core::CpuStorage,
            _: &Layout,
        ) -> Result<(candle_core::CpuStorage, Shape)> {
            candle_core::bail!("conv-gemm runs on Metal only")
        }

        fn metal_fwd(
            &self,
            x: &MetalStorage,
            lx: &Layout,
            w: &MetalStorage,
            lw: &Layout,
        ) -> Result<(MetalStorage, Shape)> {
            let conv = self.conv;
            let device = x.device();
            let (batch, in_channels, len) = lx.shape().dims3()?;
            let steps = len - (conv.taps - 1) * conv.dilation;
            let out_len = steps * conv.phases;
            let count = batch * conv.out_channels * out_len;
            let output = device.new_buffer(count, DType::BF16, "conv-gemm")?;
            let (memory, stride) = MetalConv::memory(conv.taps, conv.dilation, conv.tile);
            let flags = self.snake.map_or(0, |_| 1)
                | conv.bias.as_ref().map_or(0, |_| 2)
                | self.residual.map_or(0, |_| 4);
            let params = Params {
                in_channels: in_channels as u32,
                out_channels: conv.out_channels as u32,
                phases: conv.phases as u32,
                steps: steps as u32,
                len: len as u32,
                taps: conv.taps as u32,
                dilation: conv.dilation as u32,
                stride: stride as u32,
                flags,
            };
            let name = match conv.tile {
                (64, 64) => "conv_gemm_64x64",
                _ => "conv_gemm_32x128",
            };
            let pipeline = pipeline(device, name)?;
            let encoder = device.command_encoder()?;
            encoder.set_label("conv-gemm");
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(x.buffer()), lx.start_offset() * 2);
            encoder.set_buffer(1, Some(w.buffer()), lw.start_offset() * 2);
            encoder.use_resource(x.buffer(), MTLResourceUsage::Read);
            encoder.use_resource(w.buffer(), MTLResourceUsage::Read);
            // Unused inputs are bound to x, which the flags keep unread.
            let bind = |index: usize, tensor: Option<&Tensor>| -> Result<()> {
                match tensor {
                    Some(tensor) => with_buffer(tensor, |buffer, offset| {
                        encoder.set_buffer(index, Some(buffer), offset);
                        encoder.use_resource(buffer, MTLResourceUsage::Read);
                        Ok(())
                    }),
                    None => {
                        encoder.set_buffer(index, Some(x.buffer()), 0);
                        Ok(())
                    }
                }
            };
            bind(2, conv.bias.as_ref())?;
            bind(3, self.snake.map(|snake| &snake.alpha))?;
            bind(4, self.snake.map(|snake| &snake.inv_beta))?;
            bind(5, self.residual)?;
            encoder.set_buffer(6, Some(&output), 0);
            encoder.use_resource(output.as_ref(), MTLResourceUsage::Write);
            encoder.set_bytes(7, &params);
            encoder.set_threadgroup_memory_length(0, memory);
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: steps.div_ceil(conv.tile.1),
                    height: (conv.phases * conv.out_channels).div_ceil(conv.tile.0),
                    depth: batch,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            Ok((
                MetalStorage::new(output, device.clone(), count, DType::BF16),
                Shape::from((batch, conv.out_channels, out_len)),
            ))
        }
    }

    pub(super) fn forward(
        conv: &MetalConv,
        x: &Tensor,
        snake: Option<&Snake>,
        residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        let x = x.contiguous()?;
        let residual = residual
            .map(|r| r.to_dtype(DType::BF16)?.contiguous())
            .transpose()?;
        let op = Conv {
            conv,
            snake,
            residual: residual.as_ref(),
        };
        x.apply_op2_no_bwd(&conv.weight, &op)
    }

    struct Depthwise<'a> {
        bias: Option<&'a Tensor>,
    }

    impl CustomOp2 for Depthwise<'_> {
        fn name(&self) -> &'static str {
            "depthwise-conv"
        }

        fn cpu_fwd(
            &self,
            _: &candle_core::CpuStorage,
            _: &Layout,
            _: &candle_core::CpuStorage,
            _: &Layout,
        ) -> Result<(candle_core::CpuStorage, Shape)> {
            candle_core::bail!("depthwise-conv runs on Metal only")
        }

        fn metal_fwd(
            &self,
            x: &MetalStorage,
            lx: &Layout,
            w: &MetalStorage,
            lw: &Layout,
        ) -> Result<(MetalStorage, Shape)> {
            let device = x.device();
            let (batch, channels, len) = lx.shape().dims3()?;
            let taps = lw.shape().dims3()?.2;
            let steps = len + 1 - taps;
            let count = batch * channels * steps;
            let output = device.new_buffer(count, DType::BF16, "depthwise-conv")?;
            let pipeline = pipeline(device, "depthwise_conv")?;
            let encoder = device.command_encoder()?;
            encoder.set_label("depthwise-conv");
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(x.buffer()), lx.start_offset() * 2);
            encoder.set_buffer(1, Some(w.buffer()), lw.start_offset() * 2);
            encoder.use_resource(x.buffer(), MTLResourceUsage::Read);
            encoder.use_resource(w.buffer(), MTLResourceUsage::Read);
            match self.bias {
                Some(bias) => with_buffer(bias, |buffer, offset| {
                    encoder.set_buffer(2, Some(buffer), offset);
                    encoder.use_resource(buffer, MTLResourceUsage::Read);
                    Ok(())
                })?,
                None => encoder.set_buffer(2, Some(x.buffer()), 0),
            }
            encoder.set_buffer(3, Some(&output), 0);
            encoder.use_resource(output.as_ref(), MTLResourceUsage::Write);
            encoder.set_bytes(
                4,
                &[
                    len as u32,
                    steps as u32,
                    taps as u32,
                    self.bias.is_some() as u32,
                ],
            );
            encoder.dispatch_thread_groups(
                MTLSize {
                    width: steps.div_ceil(64),
                    height: channels,
                    depth: batch,
                },
                MTLSize {
                    width: 64,
                    height: 1,
                    depth: 1,
                },
            );
            Ok((
                MetalStorage::new(output, device.clone(), count, DType::BF16),
                Shape::from((batch, channels, steps)),
            ))
        }
    }

    pub(super) fn depthwise(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let x = x.contiguous()?;
        let weight = weight.contiguous()?;
        let bias = bias.map(|b| b.contiguous()).transpose()?;
        x.apply_op2_no_bwd(
            &weight,
            &Depthwise {
                bias: bias.as_ref(),
            },
        )
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Largest difference from `reference`, relative to its largest value.
    fn error(y: &Tensor, reference: &Tensor) -> Result<f32> {
        let y = y.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let max = (&y - reference)?.abs()?.max_all()?.to_scalar::<f32>()?;
        Ok(max / reference.abs()?.max_all()?.to_scalar::<f32>()?)
    }

    fn randn(device: &Device, std: f32, shape: &[usize]) -> Result<Tensor> {
        Tensor::randn(0f32, std, shape, device)?.to_dtype(DType::BF16)
    }

    /// The same tensor in f32 on the CPU, for a reference computed there.
    fn cpu(t: &Tensor) -> Result<Tensor> {
        t.to_device(&Device::Cpu)?.to_dtype(DType::F32)
    }

    fn snake(x: &Tensor, alpha: &Tensor, beta: &Tensor) -> Result<Tensor> {
        let shape = (1, alpha.dim(0)?, 1);
        crate::tensor_utils::snake_beta(
            x,
            &cpu(alpha)?.reshape(shape)?,
            &cpu(beta)?.reshape(shape)?,
        )
    }

    #[test]
    fn conv1d_matches_candle() -> Result<()> {
        let Ok(device) = Device::new_metal(0) else {
            return Ok(());
        };
        // 96 rows take 32 x 128 tiles, 256 take 64 x 64; lengths leave
        // partial tiles.
        for (channels, out, dilation, len) in [(96, 96, 3, 300), (64, 256, 9, 200), (32, 1, 1, 150)]
        {
            let x = randn(&device, 1.0, &[2, channels, len])?;
            let weight = randn(&device, 0.1, &[out, channels, 7])?;
            let bias = randn(&device, 0.1, &[out])?;
            let alpha = randn(&device, 0.3, &[channels])?;
            let beta = randn(&device, 0.3, &[channels])?;
            let steps = len - 6 * dilation;
            let residual = randn(&device, 1.0, &[2, out, steps])?;
            let config = candle_nn::Conv1dConfig {
                dilation,
                ..Default::default()
            };
            let conv = candle_nn::Conv1d::new(weight.clone(), Some(bias.clone()), config);
            let metal = MetalConv::conv1d(&conv)?.expect("conv on Metal");
            let y = metal.gemm(&x, Some(&Snake::new(&alpha, &beta)?), Some(&residual))?;

            let reference = candle_nn::Conv1d::new(cpu(&weight)?, Some(cpu(&bias)?), config);
            let expected =
                (reference.forward(&snake(&cpu(&x)?, &alpha, &beta)?)? + cpu(&residual)?)?;
            assert_eq!(y.dims(), expected.dims());
            let error = error(&y, &expected)?;
            assert!(
                error < 0.01,
                "{channels}->{out} dilation {dilation}: error {error}"
            );
        }
        Ok(())
    }

    #[test]
    fn conv_transpose1d_matches_candle_and_streams() -> Result<()> {
        let Ok(device) = Device::new_metal(0) else {
            return Ok(());
        };
        for (channels, out, kernel, stride, len) in [
            (64, 32, 6, 3, 300),
            (32, 64, 10, 5, 260),
            (64, 64, 2, 2, 200),
        ] {
            let x = randn(&device, 1.0, &[1, channels, len])?;
            let weight = randn(&device, 0.1, &[channels, out, kernel])?;
            let bias = randn(&device, 0.1, &[out])?;
            let config = candle_nn::ConvTranspose1dConfig {
                stride,
                ..Default::default()
            };
            let conv = candle_nn::ConvTranspose1d::new(weight.clone(), Some(bias.clone()), config);
            let metal = MetalConv::conv_transpose1d(&conv)?.expect("transposed conv on Metal");
            let pad = kernel / stride - 1;
            let y = metal.gemm(&x.pad_with_zeros(2, pad, 0)?, None, None)?;

            // The causal reference: Candle's full output without its right overhang.
            let reference =
                candle_nn::ConvTranspose1d::new(cpu(&weight)?, Some(cpu(&bias)?), config);
            let expected = reference.forward(&cpu(&x)?)?.narrow(2, 0, len * stride)?;
            assert_eq!(y.dims(), expected.dims());
            let error = error(&y, &expected)?;
            assert!(
                error < 0.01,
                "kernel {kernel} stride {stride}: error {error}"
            );

            // Two chunks, the second led by the first's last inputs, give the same.
            let split = 140;
            let first = metal.gemm(
                &x.narrow(2, 0, split)?.pad_with_zeros(2, pad, 0)?,
                None,
                None,
            )?;
            let second = metal.gemm(&x.narrow(2, split - pad, len - split + pad)?, None, None)?;
            let streamed = Tensor::cat(&[&first, &second], 2)?;
            let difference = (streamed.to_dtype(DType::F32)? - y.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert_eq!(difference, 0.0, "kernel {kernel} stride {stride}");
        }
        Ok(())
    }

    #[test]
    fn depthwise_matches_candle() -> Result<()> {
        let Ok(device) = Device::new_metal(0) else {
            return Ok(());
        };
        let x = randn(&device, 1.0, &[2, 48, 70])?;
        let weight = randn(&device, 0.3, &[48, 1, 7])?;
        let bias = randn(&device, 0.1, &[48])?;
        let y = depthwise_conv1d(&x, &weight, Some(&bias))?.expect("depthwise on Metal");
        let config = candle_nn::Conv1dConfig {
            groups: 48,
            ..Default::default()
        };
        let reference = candle_nn::Conv1d::new(cpu(&weight)?, Some(cpu(&bias)?), config);
        let expected = reference.forward(&cpu(&x)?)?;
        assert_eq!(y.dims(), expected.dims());
        let error = error(&y, &expected)?;
        assert!(error < 0.01, "error {error}");
        Ok(())
    }
}
