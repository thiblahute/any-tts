//! Compiles the CUDA kernels to PTX when the `cuda` feature is on (nvcc,
//! as Candle's own kernels need). `CUDA_COMPUTE_CAP` picks the PTX target,
//! compute_70 by default: the driver compiles it for the GPU at load.

use std::path::PathBuf;
use std::process::Command;

const KERNELS: &[(&str, &str)] = &[
    (
        "src/models/qwen3_tts/cuda_frames.cu",
        "qwen3_tts_frames.ptx",
    ),
    ("src/layers/cuda_ops.cu", "cuda_ops.ptx"),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=NVCC");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cap = std::env::var("CUDA_COMPUTE_CAP")
        .ok()
        .map(|cap| cap.replace('.', ""))
        .unwrap_or_else(|| "70".into());
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    for (source, ptx) in KERNELS {
        println!("cargo:rerun-if-changed={source}");
        let status = Command::new(&nvcc)
            .args([
                "--ptx",
                "-O3",
                "-std=c++17",
                &format!("-arch=compute_{cap}"),
                "-o",
            ])
            .arg(out.join(ptx))
            .arg(source)
            .status()
            .unwrap_or_else(|error| panic!("cannot run {nvcc} for {source}: {error}"));
        assert!(status.success(), "{nvcc} failed on {source}");
    }
}
