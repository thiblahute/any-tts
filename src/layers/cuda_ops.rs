//! CUDA kernels for ops Candle runs as several passes over a tensor
//! (`cuda_ops.cu`, built to PTX by nvcc).

use candle_core::backend::BackendStorage;
use candle_core::cuda::cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};
use candle_core::cuda::{CudaStorage, CudaStorageSlice, WrapErr};
use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, Result, Shape, Storage, Tensor};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cuda_ops.ptx"));
const MODULE: &str = "any_tts_cuda_ops";

/// SnakeBeta, `x + sin(exp(alpha) x)^2 / (exp(beta) + 1e-9)` per channel,
/// in one kernel instead of Candle's six passes over `x`, its parameters'
/// exponentials taken once.
pub struct Snake {
    alpha: Tensor,
    inv_beta: Tensor,
}

impl Snake {
    /// From the log-space `alpha` and `beta`, one value per channel each,
    /// if they are on CUDA.
    pub fn new(alpha: &Tensor, beta: &Tensor) -> Result<Option<Self>> {
        if !matches!(alpha.device(), Device::Cuda(_)) {
            return Ok(None);
        }
        let alpha = alpha.flatten_all()?.to_dtype(DType::F32)?.exp()?;
        let inv_beta = (beta.flatten_all()?.to_dtype(DType::F32)?.exp()? + 1e-9)?.recip()?;
        Ok(Some(Self { alpha, inv_beta }))
    }

    /// SnakeBeta of a bf16 (batch, channels, len) `x`; `None` for others.
    pub fn forward(&self, x: &Tensor) -> Result<Option<Tensor>> {
        if x.dtype() != DType::BF16 || x.rank() != 3 || x.dim(1)? != self.alpha.elem_count() {
            return Ok(None);
        }
        let (_, channels, len) = x.dims3()?;
        let op = SnakeOp {
            alpha: self.alpha.clone(),
            inv_beta: self.inv_beta.clone(),
            channels,
            len,
        };
        x.contiguous()?.apply_op1_no_bwd(&op).map(Some)
    }
}

struct SnakeOp {
    alpha: Tensor,
    inv_beta: Tensor,
    channels: usize,
    len: usize,
}

fn f32_address(t: &Tensor, stream: &candle_core::cuda::cudarc::driver::CudaStream) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    match &*storage {
        Storage::Cuda(CudaStorage {
            slice: CudaStorageSlice::F32(slice),
            ..
        }) => Ok(slice.device_ptr(stream).0 + 4 * layout.start_offset() as u64),
        _ => candle_core::bail!("snake-beta needs f32 parameters on CUDA"),
    }
}

impl CustomOp1 for SnakeOp {
    fn name(&self) -> &'static str {
        "snake-beta"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("snake-beta runs on CUDA only")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let device = storage.device().clone();
        let stream = device.cuda_stream();
        let CudaStorageSlice::BF16(x) = &storage.slice else {
            candle_core::bail!("snake-beta needs bf16")
        };
        let n = layout.shape().elem_count();
        let x = x.device_ptr(&stream).0 + 2 * layout.start_offset() as u64;
        let y = unsafe { device.alloc::<half::bf16>(n)? };
        let y_address = y.device_ptr(&stream).0;
        let alpha = f32_address(&self.alpha, &stream)?;
        let inv_beta = f32_address(&self.inv_beta, &stream)?;
        let (channels, len, count) = (self.channels as i32, self.len as i32, n as i64);
        let func = device.get_or_load_custom_func("snake_beta", MODULE, PTX)?;
        let mut b = stream.launch_builder(&func);
        b.arg(&x).arg(&alpha).arg(&inv_beta).arg(&y_address);
        b.arg(&channels).arg(&len).arg(&count);
        let blocks = (n as u32).div_ceil(512).clamp(1, 4096);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(config) }.w()?;
        Ok((
            CudaStorage::wrap_cuda_slice(y, device),
            layout.shape().clone(),
        ))
    }
}
