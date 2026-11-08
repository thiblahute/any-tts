//! Qwen3-TTS frames on CUDA as one CUDA graph each.
//!
//! Through Candle, a frame (one talker step, fifteen code predictor steps
//! and sixteen samplings) is about 3,500 small kernel launches, each with
//! its allocation, so the GPU waits on the host most of the time. Here the
//! talker's prompt still runs through Candle, then every frame runs on
//! static buffers with a few kernels per layer (`cuda_frames.cu`): RMS norm,
//! a bf16 GEMV per projection (the residual add and the SiLU gate fused),
//! q/k norm with RoPE writing the caches, split attention, and the top-k
//! sampling of each code on the GPU. Nothing in a frame depends on the host,
//! so it is captured once into a CUDA graph and replayed, the host reading
//! the frame's sixteen codes back after each.
//!
//! `QWEN3_TTS_CUDA=candle` turns this off, `QWEN3_TTS_CUDA=eager` launches
//! the same kernels one by one instead of through the graph, and
//! `QWEN3_TTS_SEED` makes the sampling reproducible.

use candle_core::cuda::cudarc::cublas::CudaBlas;
use candle_core::cuda::cudarc::driver::{
    sys, CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use candle_core::cuda::{CudaDevice, CudaStorageSlice, WrapErr};
use candle_core::{DType, Device, Result, Storage, Tensor};
use std::sync::Arc;
use tracing::info;

use super::code_predictor::CodePredictor;
use super::config::TalkerConfig;
use super::talker::TalkerLm;
use crate::layers::transformer::BlockWeights;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/qwen3_tts_frames.ptx"));
const MODULE: &str = "qwen3_tts_frames";
const HEAD_DIM: usize = 128;
/// Frames recorded on the device before the host reads them (`FRAME_SLOTS`).
const FRAME_SLOTS: usize = 4;
const FRAME_CODES: usize = 16;
/// Code predictor positions: its two-token prefill, then one per group.
const CP_POSITIONS: usize = 16;
/// Key splits of the talker's attention: its keys are the whole utterance.
const TALKER_SPLITS: u32 = 16;
const GEMV_WARPS: u32 = 8;
/// How much of the next GEMV's weights a GEMV prefetches into L2.
const PREFETCH_BYTES: i32 = 1 << 20;
const SAMPLE_THREADS: u32 = 1024;
const MAX_VOCAB: usize = 4096;

/// Mirror of the kernels' `State`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct State {
    pos: i32,
    step: i32,
    trailing_len: i32,
    window: i32,
    temperature: f32,
    penalty: f32,
    top_k: i32,
    history: i32,
    seed: u64,
    draws: u64,
}

/// What a GEMV multiplies: its input as is, RMS-normalized (weights,
/// epsilon), or the SiLU-gated half of a [gate | up] row.
#[derive(Clone, Copy)]
enum Input {
    Plain,
    Normed(u64, f32),
    Gated,
}

/// How the talker samples group 0, from its generation config.
pub struct FrameSampling {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub repetition_window: Option<usize>,
}

/// Which way frames are made on CUDA, from `QWEN3_TTS_CUDA`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Eager,
    Graph,
}

struct Kernels {
    rms_norm: CudaFunction,
    gemv: CudaFunction,
    qk_norm_rope: CudaFunction,
    attn_partial: CudaFunction,
    attn_combine: CudaFunction,
    causal_softmax: CudaFunction,
    sample: CudaFunction,
    embed: CudaFunction,
    talker_input: CudaFunction,
    copy_rows: CudaFunction,
    silu_mul: CudaFunction,
    advance: CudaFunction,
}

struct Layer {
    input_norm: u64,
    qkv: u64,
    qkv_bias: u64,
    q_norm: u64,
    k_norm: u64,
    o: u64,
    post_norm: u64,
    gate_up: u64,
    down: u64,
}

/// A transformer's weights and its activation buffers (rows for `rows`
/// tokens at once), and its caches of `cap` positions per layer.
struct Stack {
    hidden: usize,
    inter: usize,
    heads: usize,
    kv_heads: usize,
    eps: f32,
    layers: Vec<Layer>,
    final_norm: u64,
    rope_cos: u64,
    rope_sin: u64,
    cap: usize,
    k_cache: CudaSlice<u16>,
    v_cache: CudaSlice<u16>,
    x: CudaSlice<u16>,
    qkv: CudaSlice<u16>,
    q: CudaSlice<u16>,
    attn: CudaSlice<u16>,
    gate_up: CudaSlice<u16>,
    splits: u32,
}

/// Holds the CUDA graph; the engine's mutex serializes its use.
struct Graph(CudaGraph);
// SAFETY: a graph is only launched under the engine's lock, one thread at a time.
unsafe impl Send for Graph {}

pub struct CudaFrames {
    stream: Arc<CudaStream>,
    blas: Arc<CudaBlas>,
    k: Kernels,
    mode: Mode,
    /// Weights the raw pointers below point into.
    _weights: Vec<Tensor>,
    talker: Stack,
    cp: Stack,
    talker_vocab: usize,
    cp_vocab: usize,
    suppress_start: i32,
    eos: u32,
    codec_head: u64,
    codec_embed: u64,
    cp_proj: Option<(u64, u64)>,
    cp_heads: Vec<u64>,
    cp_embeds: Vec<u64>,
    cp_temperature: f32,
    cp_top_k: i32,
    /// Row 0: the talker's last hidden state (normed); row 1: group 0's embedding.
    cp_in: CudaSlice<u16>,
    /// Embedding of the latest code predictor group, in the talker's space.
    emb: CudaSlice<u16>,
    summed: CudaSlice<u16>,
    trailing: CudaSlice<u16>,
    trailing_cap: usize,
    logits: CudaSlice<u16>,
    part_ml: CudaSlice<f32>,
    part_acc: CudaSlice<f32>,
    token: CudaSlice<u32>,
    frames: CudaSlice<u32>,
    counts: CudaSlice<u32>,
    ring: CudaSlice<u32>,
    state: CudaSlice<u8>,
    seed: Option<u64>,
    utterances: u64,
    graph: Option<Graph>,
}

fn kernel(device: &CudaDevice, name: &str) -> Result<CudaFunction> {
    Ok((*device.get_or_load_custom_func(name, MODULE, PTX)?).clone())
}

/// Device address of a bf16 tensor's first element.
fn address(tensor: &Tensor, stream: &CudaStream) -> Result<u64> {
    if tensor.dtype() != DType::BF16 {
        candle_core::bail!("CUDA frames need bf16 weights, not {:?}", tensor.dtype())
    }
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("CUDA frames need tensors on the GPU")
    };
    let CudaStorageSlice::BF16(slice) = &storage.slice else {
        candle_core::bail!("CUDA frames need bf16 storage")
    };
    let (base, _) = slice.device_ptr(stream);
    Ok(base + 2 * layout.start_offset() as u64)
}

fn contiguous_address(tensor: &Tensor, stream: &CudaStream) -> Result<u64> {
    if !tensor.is_contiguous() {
        candle_core::bail!("CUDA frames need contiguous weights")
    }
    address(tensor, stream)
}

fn slice_address<T>(slice: &CudaSlice<T>, stream: &CudaStream) -> u64 {
    slice.device_ptr(stream).0
}

impl Stack {
    #[allow(clippy::too_many_arguments)]
    fn new(
        blocks: &[BlockWeights],
        final_norm: &Tensor,
        rope: (&Tensor, &Tensor),
        rows: usize,
        cap: usize,
        splits: u32,
        stream: &Arc<CudaStream>,
        keep: &mut Vec<Tensor>,
    ) -> Result<Self> {
        let first = &blocks[0];
        if first.head_dim != HEAD_DIM
            || first.heads % first.kv_heads != 0
            || first.heads / first.kv_heads > 4
        {
            candle_core::bail!(
                "CUDA frames need heads of {HEAD_DIM} and at most 4 queries per kv head"
            )
        }
        let mut address_of = |t: &Tensor| -> Result<u64> {
            keep.push(t.clone());
            contiguous_address(t, stream)
        };
        let mut layers = Vec::with_capacity(blocks.len());
        for block in blocks {
            layers.push(Layer {
                input_norm: address_of(&block.input_norm)?,
                qkv: address_of(&block.qkv)?,
                qkv_bias: block
                    .qkv_bias
                    .as_ref()
                    .map(&mut address_of)
                    .transpose()?
                    .unwrap_or(0),
                q_norm: block
                    .q_norm
                    .as_ref()
                    .map(&mut address_of)
                    .transpose()?
                    .unwrap_or(0),
                k_norm: block
                    .k_norm
                    .as_ref()
                    .map(&mut address_of)
                    .transpose()?
                    .unwrap_or(0),
                o: address_of(&block.o)?,
                post_norm: address_of(&block.post_norm)?,
                gate_up: address_of(&block.gate_up)?,
                down: address_of(&block.down)?,
            });
        }
        let (hidden, inter) = (first.hidden, first.intermediate);
        let (heads, kv_heads) = (first.heads, first.kv_heads);
        let qkv_dim = (heads + 2 * kv_heads) * HEAD_DIM;
        let cache = (blocks.len() * kv_heads * cap * HEAD_DIM).max(1);
        Ok(Self {
            hidden,
            inter,
            heads,
            kv_heads,
            eps: first.eps as f32,
            final_norm: address_of(final_norm)?,
            rope_cos: address_of(rope.0)?,
            rope_sin: address_of(rope.1)?,
            layers,
            cap,
            k_cache: stream.alloc_zeros(cache).w()?,
            v_cache: stream.alloc_zeros(cache).w()?,
            x: stream.alloc_zeros(rows * hidden).w()?,
            qkv: stream.alloc_zeros(rows * qkv_dim).w()?,
            q: stream.alloc_zeros(rows * heads * HEAD_DIM).w()?,
            attn: stream.alloc_zeros(rows * heads * HEAD_DIM).w()?,
            gate_up: stream.alloc_zeros(rows * 2 * inter).w()?,
            splits,
        })
    }

    fn cache_layer(&self, cache: &CudaSlice<u16>, layer: usize, stream: &CudaStream) -> u64 {
        slice_address(cache, stream) + (2 * layer * self.kv_heads * self.cap * HEAD_DIM) as u64
    }
}

impl CudaFrames {
    /// The engine for a loaded model, or `None` when it does not apply:
    /// not on CUDA, not bf16, or turned off by `QWEN3_TTS_CUDA=candle`.
    pub fn new(
        talker: &TalkerLm,
        cp: &CodePredictor,
        config: &TalkerConfig,
        device: &Device,
    ) -> Result<Option<Self>> {
        let Device::Cuda(cuda) = device else {
            return Ok(None);
        };
        let mode = match std::env::var("QWEN3_TTS_CUDA").as_deref() {
            Ok("candle") => return Ok(None),
            Ok("eager") => Mode::Eager,
            _ => Mode::Graph,
        };
        if config.vocab_size > MAX_VOCAB || cp.vocab_size() > MAX_VOCAB {
            return Ok(None);
        }
        let stream = cuda.cuda_stream();
        let k = Kernels {
            rms_norm: kernel(cuda, "rms_norm")?,
            gemv: kernel(cuda, "gemv")?,
            qk_norm_rope: kernel(cuda, "qk_norm_rope")?,
            attn_partial: kernel(cuda, "attn_partial")?,
            attn_combine: kernel(cuda, "attn_combine")?,
            causal_softmax: kernel(cuda, "causal_softmax")?,
            sample: kernel(cuda, "sample")?,
            embed: kernel(cuda, "embed")?,
            talker_input: kernel(cuda, "talker_input")?,
            copy_rows: kernel(cuda, "copy_rows")?,
            silu_mul: kernel(cuda, "silu_mul")?,
            advance: kernel(cuda, "advance")?,
        };
        let mut keep = Vec::new();
        let talker_blocks: Vec<BlockWeights> = talker.blocks().map(|b| b.weights()).collect();
        let cp_blocks: Vec<BlockWeights> = cp.blocks().map(|b| b.weights()).collect();
        // The talker's caches are sized for each utterance (`prepare`).
        let talker_stack = Stack::new(
            &talker_blocks,
            talker.norm_weight(),
            talker.rope(),
            1,
            0,
            TALKER_SPLITS,
            &stream,
            &mut keep,
        )?;
        let cp_stack = Stack::new(
            &cp_blocks,
            cp.norm_weight(),
            cp.rope(),
            2,
            CP_POSITIONS,
            1,
            &stream,
            &mut keep,
        )?;
        let mut address_of = |t: &Tensor| -> Result<u64> {
            keep.push(t.clone());
            contiguous_address(t, &stream)
        };
        let codec_head = address_of(talker.codec_head_weight())?;
        let codec_embed = address_of(talker.codec_embedding())?;
        let cp_proj = match cp.input_proj() {
            Some(proj) => {
                let bias = proj.bias().map(&mut address_of).transpose()?.unwrap_or(0);
                Some((address_of(proj.weight())?, bias))
            }
            None => None,
        };
        let cp_heads = cp
            .head_weights()
            .map(&mut address_of)
            .collect::<Result<Vec<_>>>()?;
        let cp_embeds = cp
            .group_embeddings()
            .map(&mut address_of)
            .collect::<Result<Vec<_>>>()?;
        if cp_heads.len() + 1 != FRAME_CODES || cp_embeds.len() + 1 != FRAME_CODES {
            return Ok(None);
        }
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;
        let max_heads = talker_stack.heads.max(cp_stack.heads);
        let parts = 2 * max_heads * TALKER_SPLITS as usize;
        let seed = std::env::var("QWEN3_TTS_SEED")
            .ok()
            .and_then(|s| s.parse().ok());
        info!("Qwen3-TTS frames run as CUDA {mode:?}");
        Ok(Some(Self {
            k,
            mode,
            _weights: keep,
            talker_vocab: vocab,
            cp_vocab: cp.vocab_size(),
            suppress_start: if vocab > 1024 { vocab - 1024 } else { vocab } as i32,
            eos: config.codec_eos_token_id,
            codec_head,
            codec_embed,
            cp_proj,
            cp_heads,
            cp_embeds,
            cp_temperature: cp.temperature() as f32,
            cp_top_k: cp.top_k() as i32,
            cp_in: stream.alloc_zeros(2 * hidden).w()?,
            emb: stream.alloc_zeros(hidden).w()?,
            summed: stream.alloc_zeros(hidden).w()?,
            trailing: stream.alloc_zeros(hidden).w()?,
            trailing_cap: 1,
            logits: stream.alloc_zeros(MAX_VOCAB).w()?,
            part_ml: stream.alloc_zeros(parts * 2).w()?,
            part_acc: stream.alloc_zeros(parts * HEAD_DIM).w()?,
            token: stream.alloc_zeros(1).w()?,
            frames: stream.alloc_zeros(FRAME_SLOTS * FRAME_CODES).w()?,
            counts: stream.alloc_zeros(vocab).w()?,
            ring: stream.alloc_zeros(vocab.max(64)).w()?,
            state: stream.alloc_zeros(std::mem::size_of::<State>()).w()?,
            seed,
            utterances: 0,
            graph: None,
            talker: talker_stack,
            cp: cp_stack,
            blas: cuda.cublas_handle(),
            stream,
        }))
    }

    fn config(grid: (u32, u32, u32), block: u32, shared: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: grid,
            block_dim: (block, 1, 1),
            shared_mem_bytes: shared,
        }
    }

    fn rms_norm(&self, x: u64, w: u64, y: u64, n: usize, rows: usize, eps: f32) -> Result<()> {
        let n = n as i32;
        let mut b = self.stream.launch_builder(&self.k.rms_norm);
        b.arg(&x).arg(&w).arg(&y).arg(&n).arg(&eps);
        unsafe { b.launch(Self::config((rows as u32, 1, 1), 256, 0)) }.w()?;
        Ok(())
    }

    /// y = w x (+ bias | + resid) for `m` rows, x first RMS-normalized
    /// with `norm` (if not 0) or gated, prefetching the start of `next`
    /// (if not 0), the weights the next GEMV reads; see the kernel.
    #[allow(clippy::too_many_arguments)]
    fn gemv(
        &self,
        w: u64,
        x: u64,
        y: u64,
        k: usize,
        n: usize,
        m: usize,
        bias: u64,
        resid: u64,
        input: Input,
        next: u64,
    ) -> Result<()> {
        let (k32, n32, m32) = (k as i32, n as i32, m as i32);
        let (gated, norm, eps) = match input {
            Input::Plain => (0i32, 0u64, 0f32),
            Input::Gated => (1, 0, 0.),
            Input::Normed(norm, eps) => (0, norm, eps),
        };
        let x_stride = if gated == 1 { 2 * k32 } else { k32 };
        let blocks = (n as u32).div_ceil(GEMV_WARPS);
        let shared = (m * k * 2) as u32;
        let mut b = self.stream.launch_builder(&self.k.gemv);
        b.arg(&w).arg(&x).arg(&y).arg(&k32).arg(&n32).arg(&m32);
        b.arg(&x_stride).arg(&n32).arg(&bias).arg(&resid).arg(&n32);
        let next_bytes = if next == 0 { 0 } else { PREFETCH_BYTES };
        b.arg(&gated)
            .arg(&norm)
            .arg(&eps)
            .arg(&next)
            .arg(&next_bytes);
        unsafe { b.launch(Self::config((blocks, 1, 1), GEMV_WARPS * 32, shared)) }.w()?;
        Ok(())
    }

    /// Layer `layer`'s attention for `rows` tokens from their q/k/v
    /// projections: q and k normalized and rotated, k and v into the
    /// caches at `*pos + base` on, then attention over each token's causal
    /// range in `splits` key splits (partials in `ml`, `acc`), to `out`;
    /// no attention with no splits.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        s: &Stack,
        layer: usize,
        rows: usize,
        (pos, base): (u64, i32),
        qkv: u64,
        q: u64,
        out: u64,
        (ml, acc, splits): (u64, u64, u32),
    ) -> Result<()> {
        let st = &self.stream;
        let l = &s.layers[layer];
        let k_cache = s.cache_layer(&s.k_cache, layer, st);
        let v_cache = s.cache_layer(&s.v_cache, layer, st);
        let (heads, kv_heads, cap) = (s.heads as i32, s.kv_heads as i32, s.cap as i32);
        let qkv_stride = ((s.heads + 2 * s.kv_heads) * HEAD_DIM) as i32;
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let mut b = st.launch_builder(&self.k.qk_norm_rope);
        b.arg(&qkv).arg(&qkv_stride).arg(&l.q_norm).arg(&l.k_norm);
        b.arg(&s.eps)
            .arg(&s.rope_cos)
            .arg(&s.rope_sin)
            .arg(&pos)
            .arg(&base);
        b.arg(&q).arg(&k_cache).arg(&v_cache).arg(&cap);
        b.arg(&heads).arg(&kv_heads);
        let grid = ((s.heads + 2 * s.kv_heads) as u32, rows as u32, 1);
        unsafe { b.launch(Self::config(grid, (HEAD_DIM / 2) as u32, 0)) }.w()?;
        if splits == 0 {
            return Ok(());
        }
        let mut b = st.launch_builder(&self.k.attn_partial);
        b.arg(&q).arg(&k_cache).arg(&v_cache).arg(&pos).arg(&base);
        b.arg(&cap)
            .arg(&heads)
            .arg(&kv_heads)
            .arg(&scale)
            .arg(&ml)
            .arg(&acc);
        let grid = (s.kv_heads as u32, splits, rows as u32);
        unsafe { b.launch(Self::config(grid, HEAD_DIM as u32, 0)) }.w()?;
        let splits = splits as i32;
        let mut b = st.launch_builder(&self.k.attn_combine);
        b.arg(&ml).arg(&acc).arg(&out).arg(&heads).arg(&splits);
        let grid = (s.heads as u32, rows as u32, 1);
        unsafe { b.launch(Self::config(grid, HEAD_DIM as u32, 0)) }.w()?;
        Ok(())
    }

    /// y (m x n) = x (m x k) w^T, w being n x k, all row-major bf16, plus y
    /// itself when `accumulate`; through cuBLAS, for the prompt's rows.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        w: u64,
        x: u64,
        y: u64,
        m: usize,
        k: usize,
        n: usize,
        accumulate: bool,
    ) -> Result<()> {
        use candle_core::cuda::cudarc::cublas::{result::gemm_ex, sys as blas};
        let (alpha, beta) = (1f32, if accumulate { 1f32 } else { 0f32 });
        let bf16 = blas::cudaDataType::CUDA_R_16BF;
        let (m, k, n) = (m as i32, k as i32, n as i32);
        // Column-major: y^T (n x m) = w (n x k, w^T stored) x^T (k x m).
        unsafe {
            gemm_ex(
                *self.blas.handle(),
                blas::cublasOperation_t::CUBLAS_OP_T,
                blas::cublasOperation_t::CUBLAS_OP_N,
                n,
                m,
                k,
                &alpha as *const f32 as *const _,
                w as *const _,
                bf16,
                k,
                x as *const _,
                bf16,
                k,
                &beta as *const f32 as *const _,
                y as *mut _,
                bf16,
                n,
                blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(|error| candle_core::Error::Msg(format!("cuBLAS: {error:?}")))
    }

    /// The prompt's attention in talker layer `layer`, from its queries
    /// `q` ([rows][heads][128]) and the caches, to `out` (same layout):
    /// each head's scores and weighted values as batched GEMMs (tensor
    /// cores, which a prompt of a few hundred positions keeps busy), and
    /// the causal softmax between them, through `scores` (f32) and `probs`
    /// (bf16), heads x rows x rows each.
    fn prompt_attention(
        &self,
        layer: usize,
        rows: usize,
        q: u64,
        out: u64,
        (scores, probs): (u64, u64),
    ) -> Result<()> {
        use candle_core::cuda::cudarc::cublas::{result::gemm_strided_batched_ex, sys as blas};
        let t = &self.talker;
        let st = &self.stream;
        let group = t.heads / t.kv_heads;
        let (k_cache, v_cache) = (
            t.cache_layer(&t.k_cache, layer, st),
            t.cache_layer(&t.v_cache, layer, st),
        );
        let (n, d) = (rows as i32, HEAD_DIM as i32);
        let row = (t.heads * HEAD_DIM) as i32;
        let cache_stride = (t.cap * HEAD_DIM) as i64;
        let square = (rows * rows) as i64;
        let (scale, one, zero) = (1.0f32 / (HEAD_DIM as f32).sqrt(), 1f32, 0f32);
        let (bf16, f32_) = (
            blas::cudaDataType::CUDA_R_16BF,
            blas::cudaDataType::CUDA_R_32F,
        );
        let (op_n, op_t) = (
            blas::cublasOperation_t::CUBLAS_OP_N,
            blas::cublasOperation_t::CUBLAS_OP_T,
        );
        let (compute, algo) = (
            blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        );
        let error = |e| candle_core::Error::Msg(format!("cuBLAS: {e:?}"));
        // Query heads g, g + group, ... share kv heads 0, 1, ...: one batch each.
        let batch = t.kv_heads as i32;
        for g in 0..group {
            let q = q + 2 * (g * HEAD_DIM) as u64;
            let s = scores + 4 * (g as u64) * square as u64;
            // Column-major S^T (keys x queries) = K (keys x 128) Q^T.
            unsafe {
                gemm_strided_batched_ex(
                    *self.blas.handle(),
                    op_t,
                    op_n,
                    n,
                    n,
                    d,
                    &scale as *const f32 as *const _,
                    k_cache as *const _,
                    bf16,
                    d,
                    cache_stride,
                    q as *const _,
                    bf16,
                    row,
                    (group * HEAD_DIM) as i64,
                    &zero as *const f32 as *const _,
                    s as *mut _,
                    f32_,
                    n,
                    group as i64 * square,
                    batch,
                    compute,
                    algo,
                )
            }
            .map_err(error)?;
        }
        let mut b = st.launch_builder(&self.k.causal_softmax);
        b.arg(&scores).arg(&probs).arg(&n);
        unsafe { b.launch(Self::config((rows as u32, t.heads as u32, 1), 128, 0)) }.w()?;
        for g in 0..group {
            let p = probs + 2 * (g as u64) * square as u64;
            let o = out + 2 * (g * HEAD_DIM) as u64;
            // Column-major O^T (128 x queries) = V^T (128 x keys) P^T.
            unsafe {
                gemm_strided_batched_ex(
                    *self.blas.handle(),
                    op_n,
                    op_n,
                    d,
                    n,
                    n,
                    &one as *const f32 as *const _,
                    v_cache as *const _,
                    bf16,
                    d,
                    cache_stride,
                    p as *const _,
                    bf16,
                    n,
                    group as i64 * square,
                    &zero as *const f32 as *const _,
                    o as *mut _,
                    bf16,
                    row,
                    (group * HEAD_DIM) as i64,
                    batch,
                    compute,
                    algo,
                )
            }
            .map_err(error)?;
        }
        Ok(())
    }

    /// Runs a prompt (`rows` embeddings at `embeds`) through the talker
    /// from position 0, filling its caches, and leaves the last position's
    /// normed hidden state where a frame starts. The projections go
    /// through cuBLAS, a few hundred rows at once, the rest through the
    /// frames' kernels.
    fn prefill(&self, embeds: u64, rows: usize) -> Result<()> {
        let st = &self.stream;
        let t = &self.talker;
        if t.layers.iter().any(|layer| layer.qkv_bias != 0) {
            candle_core::bail!("CUDA frames' prefill has no q/k/v bias")
        }
        let (hidden, inter) = (t.hidden, t.inter);
        let qkv_dim = (t.heads + 2 * t.kv_heads) * HEAD_DIM;
        let q_dim = t.heads * HEAD_DIM;
        let x = st.alloc_zeros::<u16>(rows * hidden).w()?;
        let h = st.alloc_zeros::<u16>(rows * hidden).w()?;
        let qkv = st.alloc_zeros::<u16>(rows * qkv_dim).w()?;
        let q = st.alloc_zeros::<u16>(rows * q_dim).w()?;
        let attn = st.alloc_zeros::<u16>(rows * q_dim).w()?;
        let gate_up = st.alloc_zeros::<u16>(rows * 2 * inter).w()?;
        let act = st.alloc_zeros::<u16>(rows * inter).w()?;
        let scores = st.alloc_zeros::<f32>(t.heads * rows * rows).w()?;
        let probs = st.alloc_zeros::<u16>(t.heads * rows * rows).w()?;
        let [x, h, qkv, q, attn, gate_up, act] =
            [&x, &h, &qkv, &q, &attn, &gate_up, &act].map(|b| slice_address(b, st));
        let (scores, probs) = (slice_address(&scores, st), slice_address(&probs, st));
        self.copy_rows(embeds, x, rows * hidden)?;
        for (i, layer) in t.layers.iter().enumerate() {
            self.rms_norm(x, layer.input_norm, h, hidden, rows, t.eps)?;
            self.gemm(layer.qkv, h, qkv, rows, hidden, qkv_dim, false)?;
            self.attention(t, i, rows, (0, 0), qkv, q, 0, (0, 0, 0))?;
            self.prompt_attention(i, rows, q, attn, (scores, probs))?;
            self.gemm(layer.o, attn, x, rows, q_dim, hidden, true)?;
            self.rms_norm(x, layer.post_norm, h, hidden, rows, t.eps)?;
            self.gemm(layer.gate_up, h, gate_up, rows, hidden, 2 * inter, false)?;
            let (n, inter32) = ((rows * inter) as i32, inter as i32);
            let mut b = st.launch_builder(&self.k.silu_mul);
            b.arg(&gate_up).arg(&act).arg(&inter32).arg(&n);
            let blocks = (n as u32).div_ceil(256).min(1024);
            unsafe { b.launch(Self::config((blocks, 1, 1), 256, 0)) }.w()?;
            self.gemm(layer.down, act, x, rows, inter, hidden, true)?;
        }
        let last = x + 2 * ((rows - 1) * hidden) as u64;
        let cp_in = slice_address(&self.cp_in, st);
        self.rms_norm(last, t.final_norm, cp_in, hidden, 1, t.eps)
    }

    /// Runs `rows` tokens through a stack, from its `x` (in place), the
    /// first at position `*pos + base` (`pos` 0 for none); `after` is the
    /// next weights read after the stack, to prefetch.
    fn forward(&self, s: &Stack, rows: usize, pos: u64, base: i32, after: u64) -> Result<()> {
        let st = &self.stream;
        let x = slice_address(&s.x, st);
        let (qkv, q, attn) = (
            slice_address(&s.qkv, st),
            slice_address(&s.q, st),
            slice_address(&s.attn, st),
        );
        let gate_up = slice_address(&s.gate_up, st);
        let (ml, acc) = (
            slice_address(&self.part_ml, st),
            slice_address(&self.part_acc, st),
        );
        let qkv_dim = (s.heads + 2 * s.kv_heads) * HEAD_DIM;
        let q_dim = s.heads * HEAD_DIM;
        for (i, layer) in s.layers.iter().enumerate() {
            let normed = Input::Normed(layer.input_norm, s.eps);
            let (w, bias) = (layer.qkv, layer.qkv_bias);
            self.gemv(w, x, qkv, s.hidden, qkv_dim, rows, bias, 0, normed, layer.o)?;
            let parts = (ml, acc, s.splits);
            self.attention(s, i, rows, (pos, base), qkv, q, attn, parts)?;
            let (o, plain) = (layer.o, Input::Plain);
            self.gemv(
                o,
                attn,
                x,
                q_dim,
                s.hidden,
                rows,
                0,
                x,
                plain,
                layer.gate_up,
            )?;
            let normed = Input::Normed(layer.post_norm, s.eps);
            let (w, inter) = (layer.gate_up, 2 * s.inter);
            self.gemv(
                w, x, gate_up, s.hidden, inter, rows, 0, 0, normed, layer.down,
            )?;
            let (w, gated) = (layer.down, Input::Gated);
            let next = s.layers.get(i + 1).map_or(after, |next| next.qkv);
            self.gemv(w, gate_up, x, s.inter, s.hidden, rows, 0, x, gated, next)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn sample(&self, logits: u64, vocab: usize, talker: bool, code: i32) -> Result<()> {
        let st = &self.stream;
        let (state, counts, ring) = (
            slice_address(&self.state, st),
            slice_address(&self.counts, st),
            slice_address(&self.ring, st),
        );
        let (token, frames) = (
            slice_address(&self.token, st),
            slice_address(&self.frames, st),
        );
        let vocab = vocab as i32;
        let talker = talker as i32;
        let mut b = st.launch_builder(&self.k.sample);
        b.arg(&logits)
            .arg(&vocab)
            .arg(&state)
            .arg(&talker)
            .arg(&self.cp_temperature);
        b.arg(&self.cp_top_k)
            .arg(&self.suppress_start)
            .arg(&self.eos)
            .arg(&counts)
            .arg(&ring);
        b.arg(&token).arg(&frames).arg(&code);
        unsafe { b.launch(Self::config((1, 1, 1), SAMPLE_THREADS, 0)) }.w()?;
        Ok(())
    }

    fn embed(&self, table: u64, n: usize, out: u64, sum: u64, mode: i32) -> Result<()> {
        let token = slice_address(&self.token, &self.stream);
        let n = n as i32;
        let mut b = self.stream.launch_builder(&self.k.embed);
        b.arg(&table)
            .arg(&token)
            .arg(&n)
            .arg(&out)
            .arg(&sum)
            .arg(&mode);
        unsafe { b.launch(Self::config((4, 1, 1), 256, 0)) }.w()?;
        Ok(())
    }

    fn copy_rows(&self, src: u64, dst: u64, n: usize) -> Result<()> {
        let n = n as i32;
        let mut b = self.stream.launch_builder(&self.k.copy_rows);
        b.arg(&src).arg(&dst).arg(&n);
        unsafe { b.launch(Self::config((4, 1, 1), 256, 0)) }.w()?;
        Ok(())
    }

    /// Enqueues one frame: group 0 sampled from the talker's last hidden
    /// state, groups 1-15 by the code predictor, the talker step on their
    /// summed embeddings (its next hidden state), then the counters.
    fn record_frame(&self) -> Result<()> {
        let st = &self.stream;
        let (t, c) = (&self.talker, &self.cp);
        let hidden = t.hidden;
        let cp_in = slice_address(&self.cp_in, st);
        let emb = slice_address(&self.emb, st);
        let summed = slice_address(&self.summed, st);
        let logits = slice_address(&self.logits, st);
        let state = slice_address(&self.state, st);

        // Group 0.
        let (w, vocab) = (self.codec_head, self.talker_vocab);
        // The code predictor's first weights, read after each head.
        let cp_first = self.cp_proj.map_or(c.layers[0].qkv, |(w, _)| w);
        self.gemv(
            w,
            cp_in,
            logits,
            hidden,
            vocab,
            1,
            0,
            0,
            Input::Plain,
            cp_first,
        )?;
        self.sample(logits, self.talker_vocab, true, 0)?;
        self.embed(
            self.codec_embed,
            hidden,
            cp_in + 2 * hidden as u64,
            summed,
            1,
        )?;

        // Groups 1-15: the prefill on [hidden state, group 0], then a step a group.
        let cx = slice_address(&c.x, st);
        for group in 1..FRAME_CODES {
            let (input, rows) = if group == 1 { (cp_in, 2) } else { (emb, 1) };
            match self.cp_proj {
                Some((w, bias)) => {
                    let (plain, next) = (Input::Plain, c.layers[0].qkv);
                    self.gemv(w, input, cx, hidden, c.hidden, rows, bias, 0, plain, next)?
                }
                None => self.copy_rows(input, cx, rows * hidden)?,
            }
            let base = if group == 1 { 0 } else { group as i32 };
            self.forward(c, rows, 0, base, self.cp_heads[group - 1])?;
            let last = cx + 2 * ((rows - 1) * c.hidden) as u64;
            let (w, normed) = (self.cp_heads[group - 1], Input::Normed(c.final_norm, c.eps));
            let next = if group + 1 < FRAME_CODES {
                cp_first
            } else {
                t.layers[0].qkv
            };
            self.gemv(
                w,
                last,
                logits,
                c.hidden,
                self.cp_vocab,
                1,
                0,
                0,
                normed,
                next,
            )?;
            self.sample(logits, self.cp_vocab, false, group as i32)?;
            self.embed(self.cp_embeds[group - 1], hidden, emb, summed, 2)?;
        }

        // The talker step on the frame's embeddings, then its hidden state.
        let tx = slice_address(&t.x, st);
        let trailing = slice_address(&self.trailing, st);
        let n = hidden as i32;
        let mut b = st.launch_builder(&self.k.talker_input);
        b.arg(&summed).arg(&trailing).arg(&state).arg(&n).arg(&tx);
        unsafe { b.launch(Self::config((4, 1, 1), 256, 0)) }.w()?;
        // `pos` is the state's first field.
        self.forward(t, 1, state, 0, self.codec_head)?;
        self.rms_norm(tx, t.final_norm, cp_in, hidden, 1, t.eps)?;
        let mut b = st.launch_builder(&self.k.advance);
        b.arg(&state);
        unsafe { b.launch(Self::config((1, 1, 1), 1, 0)) }.w()?;
        Ok(())
    }

    /// Sizes the talker's caches and the trailing text for an utterance;
    /// a new size drops the graph, which points at them.
    fn reserve(&mut self, positions: usize, trailing_rows: usize) -> Result<()> {
        let t = &mut self.talker;
        if positions > t.cap {
            let cap = positions.next_multiple_of(256);
            let size = t.layers.len() * t.kv_heads * cap * HEAD_DIM;
            // Free the old caches before allocating the new ones.
            t.k_cache = self.stream.alloc_zeros(1).w()?;
            t.v_cache = self.stream.alloc_zeros(1).w()?;
            t.k_cache = self.stream.alloc_zeros(size).w()?;
            t.v_cache = self.stream.alloc_zeros(size).w()?;
            t.cap = cap;
            self.graph = None;
        }
        if trailing_rows > self.trailing_cap {
            self.trailing = self.stream.alloc_zeros(trailing_rows * t.hidden).w()?;
            self.trailing_cap = trailing_rows;
            self.graph = None;
        }
        Ok(())
    }

    /// Generates an utterance's frames: the prompt through Candle's talker,
    /// then frame after frame until EOS or `max_tokens`, handing each to
    /// `on_frame`. Returns the group-0 codes and the other groups' per frame.
    pub fn generate(
        &mut self,
        talker: &TalkerLm,
        text_embeds: &Tensor,
        trailing_text_hidden: &Tensor,
        sampling: FrameSampling,
        mut on_frame: Option<&mut dyn FnMut(u32, &[u32]) -> Result<()>>,
    ) -> Result<(Vec<u32>, Vec<Vec<u32>>)> {
        let prefix = text_embeds.dim(1)?;
        let trailing_rows = trailing_text_hidden.dim(1)?;
        let available = talker.positions().saturating_sub(prefix) + 1;
        let max_tokens = sampling.max_tokens.min(available);
        if max_tokens == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        self.reserve(prefix + max_tokens, trailing_rows)?;

        let st = self.stream.clone();
        let embeds = text_embeds.contiguous()?;
        self.prefill(address(&embeds, &st)?, prefix)?;
        let t = &self.talker;
        let trailing = trailing_text_hidden.contiguous()?;
        self.copy_rows(
            address(&trailing, &st)?,
            slice_address(&self.trailing, &st),
            trailing_rows * t.hidden,
        )?;
        st.memset_zeros(&mut self.counts).w()?;
        self.utterances += 1;
        let seed = match self.seed {
            Some(seed) => seed.wrapping_add(self.utterances),
            None => {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0)
                    ^ self.utterances.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            }
        };
        let state = State {
            pos: prefix as i32,
            step: 0,
            trailing_len: trailing_rows as i32,
            window: sampling.repetition_window.unwrap_or(0).min(self.ring.len()) as i32,
            temperature: sampling.temperature as f32,
            penalty: sampling.repetition_penalty,
            top_k: sampling.top_k as i32,
            history: 0,
            seed,
            draws: 0,
        };
        // SAFETY: State is plain old data.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &state as *const State as *const u8,
                std::mem::size_of::<State>(),
            )
        };
        st.memcpy_htod(bytes, &mut self.state).w()?;

        if self.mode == Mode::Graph && self.graph.is_none() {
            st.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .w()?;
            let recorded = self.record_frame();
            let graph = st
                .end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .w()?;
            recorded?;
            let Some(graph) = graph else {
                candle_core::bail!("capturing a CUDA frame gave no graph")
            };
            graph.upload().w()?;
            self.graph = Some(Graph(graph));
        }

        let mut g0 = Vec::new();
        let mut groups = Vec::new();
        let mut codes = [0u32; FRAME_CODES];
        for frame in 0..max_tokens {
            match &self.graph {
                Some(graph) => graph.0.launch().w()?,
                None => self.record_frame()?,
            }
            let slot = frame % FRAME_SLOTS * FRAME_CODES;
            st.memcpy_dtoh(&self.frames.slice(slot..slot + FRAME_CODES), &mut codes)
                .w()?;
            if codes[0] == self.eos {
                break;
            }
            g0.push(codes[0]);
            groups.push(codes[1..].to_vec());
            if let Some(on_frame) = on_frame.as_deref_mut() {
                on_frame(codes[0], &codes[1..])?;
            }
        }
        Ok((g0, groups))
    }
}
