//! Times the Qwen3-TTS decoder's convolution layers at the shapes of a chunk
//! of `frames` codes (25 by default), Candle's ops against the Metal kernels
//! of `any_tts::layers::metal_conv`, and prints how far apart their outputs
//! are. The GEMM kernel runs at every size here, also where the decoder
//! leaves short outputs to Candle.
//!
//!   cargo run --release --features metal --example bench_conv_layers -- [frames]

use any_tts::layers::metal_conv::{self, MetalConv, Snake};
use any_tts::tensor_utils::snake_beta;
use candle_core::{DType, Device, Module, Tensor};
use std::time::Instant;

fn time<F: FnMut() -> candle_core::Result<Tensor>>(
    dev: &Device,
    mut f: F,
) -> anyhow::Result<(f64, Tensor)> {
    for _ in 0..3 {
        f()?;
    }
    dev.synchronize()?;
    let started = Instant::now();
    let mut y = f()?;
    for _ in 0..19 {
        y = f()?;
    }
    dev.synchronize()?;
    Ok((started.elapsed().as_secs_f64() * 1000.0 / 20.0, y))
}

fn diff(a: &Tensor, b: &Tensor) -> anyhow::Result<(f32, f32)> {
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?;
    let max = (&a - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;
    let scale = a.abs()?.max_all()?.to_scalar::<f32>()?;
    Ok((max, scale))
}

fn randn(dev: &Device, std: f32, shape: &[usize]) -> candle_core::Result<Tensor> {
    Tensor::randn(0f32, std, shape, dev)?.to_dtype(DType::BF16)
}

struct Row {
    count: usize,
    candle: f64,
    gemm: f64,
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_metal(0)?;
    let f: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(25);
    let mut rows: Vec<Row> = Vec::new();
    let mut report = |name: String,
                      count: usize,
                      candle: f64,
                      gemm: f64,
                      (max, scale): (f32, f32)| {
        println!(
            "{name:<44} x{count}: candle {candle:7.3} ms, gemm {gemm:7.3} ms, max diff {max:.4} of {scale:.2}"
        );
        rows.push(Row {
            count,
            candle,
            gemm,
        });
    };

    // Residual units: SnakeBeta, dilated k7 conv, SnakeBeta, 1x1 conv, residual.
    for &(c, len) in &[
        (768usize, 32 * f),
        (384, 160 * f),
        (192, 640 * f),
        (96, 1920 * f),
    ] {
        for &dil in &[1usize, 3, 9] {
            let x = randn(&dev, 1.0, &[1, c, len])?;
            let w = randn(&dev, 0.05, &[c, c, 7])?;
            let b = randn(&dev, 0.1, &[c])?;
            let alpha = randn(&dev, 0.3, &[c])?;
            let beta = randn(&dev, 0.3, &[c])?;
            let snake = Snake::new(&alpha, &beta)?;
            let a3 = alpha.reshape((1, c, 1))?;
            let b3 = beta.reshape((1, c, 1))?;
            let conv = candle_nn::Conv1d::new(
                w.clone(),
                Some(b.clone()),
                candle_nn::Conv1dConfig {
                    dilation: dil,
                    ..Default::default()
                },
            );
            let (t0, y0) = time(&dev, || {
                conv.forward(&snake_beta(&x, &a3, &b3)?.pad_with_zeros(2, 6 * dil, 0)?)
            })?;
            let gemm = MetalConv::conv1d(&conv)?.unwrap();
            let (t1, y1) = time(&dev, || {
                gemm.gemm(&x.pad_with_zeros(2, 6 * dil, 0)?, Some(&snake), None)
            })?;
            report(
                format!("snake+conv1d c={c} len={len} dil={dil}"),
                1,
                t0,
                t1,
                diff(&y0, &y1)?,
            );
        }
        let x = randn(&dev, 1.0, &[1, c, len])?;
        let h = randn(&dev, 1.0, &[1, c, len])?;
        let w = randn(&dev, 0.1, &[c, c, 1])?;
        let b = randn(&dev, 0.1, &[c])?;
        let alpha = randn(&dev, 0.3, &[c])?;
        let beta = randn(&dev, 0.3, &[c])?;
        let snake = Snake::new(&alpha, &beta)?;
        let a3 = alpha.reshape((1, c, 1))?;
        let b3 = beta.reshape((1, c, 1))?;
        let conv = candle_nn::Conv1d::new(w.clone(), Some(b.clone()), Default::default());
        let (t0, y0) = time(&dev, || &x + conv.forward(&snake_beta(&h, &a3, &b3)?)?)?;
        let gemm = MetalConv::conv1d(&conv)?.unwrap();
        let (t1, y1) = time(&dev, || gemm.gemm(&h, Some(&snake), Some(&x)))?;
        report(
            format!("snake+conv1x1+res c={c} len={len}"),
            3,
            t0,
            t1,
            diff(&y0, &y1)?,
        );
    }

    // Decoder stages' upsampling: SnakeBeta, causal transposed conv (k = 2 stride).
    for &(cin, cout, k, len) in &[
        (1536usize, 768usize, 16usize, 4 * f),
        (768, 384, 10, 32 * f),
        (384, 192, 8, 160 * f),
        (192, 96, 6, 640 * f),
    ] {
        let s = k / 2;
        let x = randn(&dev, 1.0, &[1, cin, len])?;
        let w = randn(&dev, 0.05, &[cin, cout, k])?;
        let b = randn(&dev, 0.1, &[cout])?;
        let alpha = randn(&dev, 0.3, &[cin])?;
        let beta = randn(&dev, 0.3, &[cin])?;
        let snake = Snake::new(&alpha, &beta)?;
        let a3 = alpha.reshape((1, cin, 1))?;
        let b3 = beta.reshape((1, cin, 1))?;
        let conv = candle_nn::ConvTranspose1d::new(
            w.clone(),
            Some(b.clone()),
            candle_nn::ConvTranspose1dConfig {
                stride: s,
                ..Default::default()
            },
        );
        let (t0, y0) = time(&dev, || {
            let y = conv.forward(&snake_beta(&x, &a3, &b3)?)?;
            let n = y.dim(2)?;
            y.narrow(2, 0, n - s)
        })?;
        let gemm = MetalConv::conv_transpose1d(&conv)?.unwrap();
        let (t1, y1) = time(&dev, || {
            gemm.gemm(&x.pad_with_zeros(2, 1, 0)?, Some(&snake), None)
        })?;
        report(
            format!("snake+convT {cin}->{cout} k={k} len={len}"),
            1,
            t0,
            t1,
            diff(&y0, &y1)?,
        );
    }

    // Upsample stages: transposed conv k2 s2, no overlap.
    for &len in &[f, 2 * f] {
        let c = 1024;
        let x = randn(&dev, 1.0, &[1, c, len])?;
        let w = randn(&dev, 0.05, &[c, c, 2])?;
        let b = randn(&dev, 0.1, &[c])?;
        let conv = candle_nn::ConvTranspose1d::new(
            w.clone(),
            Some(b.clone()),
            candle_nn::ConvTranspose1dConfig {
                stride: 2,
                ..Default::default()
            },
        );
        let (t0, y0) = time(&dev, || conv.forward(&x))?;
        let gemm = MetalConv::conv_transpose1d(&conv)?.unwrap();
        let (t1, y1) = time(&dev, || gemm.gemm(&x, None, None))?;
        report(
            format!("convT 1024 k=2 s=2 len={len}"),
            1,
            t0,
            t1,
            diff(&y0, &y1)?,
        );
    }

    // Plain causal convs: pre-conv, decoder's first conv, final SnakeBeta + conv to one channel.
    for &(cin, cout, k, len, act) in &[
        (512usize, 1024usize, 3usize, f, false),
        (1024, 1536, 7, 4 * f, false),
        (96, 1, 7, 1920 * f, true),
    ] {
        let x = randn(&dev, 1.0, &[1, cin, len])?;
        let w = randn(&dev, 0.05, &[cout, cin, k])?;
        let b = randn(&dev, 0.1, &[cout])?;
        let alpha = randn(&dev, 0.3, &[cin])?;
        let beta = randn(&dev, 0.3, &[cin])?;
        let snake = Snake::new(&alpha, &beta)?;
        let a3 = alpha.reshape((1, cin, 1))?;
        let b3 = beta.reshape((1, cin, 1))?;
        let conv = candle_nn::Conv1d::new(w.clone(), Some(b.clone()), Default::default());
        let (t0, y0) = time(&dev, || {
            let h = if act {
                snake_beta(&x, &a3, &b3)?
            } else {
                x.clone()
            };
            conv.forward(&h.pad_with_zeros(2, k - 1, 0)?)
        })?;
        let gemm = MetalConv::conv1d(&conv)?.unwrap();
        let (t1, y1) = time(&dev, || {
            gemm.gemm(&x.pad_with_zeros(2, k - 1, 0)?, act.then_some(&snake), None)
        })?;
        report(
            format!("conv1d {cin}->{cout} k={k} len={len} snake={act}"),
            1,
            t0,
            t1,
            diff(&y0, &y1)?,
        );
    }

    // ConvNeXt's depthwise k7 conv as the decoder runs it: shifted multiply-adds.
    for &len in &[2 * f, 4 * f] {
        let c = 1024;
        let x = randn(&dev, 1.0, &[1, c, len + 6])?;
        let w = randn(&dev, 0.1, &[c, 1, 7])?.reshape((1, c, 7))?;
        let b = randn(&dev, 0.1, &[c])?.reshape((1, c, 1))?;
        let (t0, y0) = time(&dev, || {
            let mut out = x.narrow(2, 0, len)?.broadcast_mul(&w.narrow(2, 0, 1)?)?;
            for k in 1..7 {
                out = (out + x.narrow(2, k, len)?.broadcast_mul(&w.narrow(2, k, 1)?)?)?;
            }
            out.broadcast_add(&b)
        })?;
        let w3 = w.reshape((c, 1, 7))?;
        let b1 = b.flatten_all()?;
        let (t1, y1) = time(&dev, || {
            Ok(metal_conv::depthwise_conv1d(&x, &w3, Some(&b1))?.unwrap())
        })?;
        let (max, scale) = diff(&y0, &y1)?;
        println!("depthwise c={c} len={len}: shifted mul-adds {t0:.3} ms, kernel {t1:.3} ms, max diff {max:.4} of {scale:.2}");
    }

    let candle: f64 = rows.iter().map(|r| r.candle * r.count as f64).sum();
    let gemm: f64 = rows.iter().map(|r| r.gemm * r.count as f64).sum();
    println!("per {f}-frame chunk: candle {candle:.2} ms, gemm {gemm:.2} ms");
    Ok(())
}
