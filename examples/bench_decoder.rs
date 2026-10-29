//! Times the Qwen3-TTS speech tokenizer decoder on recorded codes, decoding
//! them as a stream of fixed-size chunks, and writes the audio as raw f32.
//!
//!   cargo run --release --features metal --example bench_decoder -- \
//!     <model-dir> <codes.u32> <out.f32> [chunk-frames] [cpu]
//!
//! `codes.u32` holds 16 rows of T little-endian u32 codes. With `cpu`, the
//! decoder runs in f32 on the CPU, a reference for the GPU's output.

use any_tts::models::qwen3_tts::speech_tokenizer::{
    DecoderStream, SpeechTokenizerConfig, SpeechTokenizerDecoder,
};
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model_dir, codes_path, out_path] = [&args[0], &args[1], &args[2]];
    let chunk: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(25);
    let cpu = args.get(4).is_some_and(|s| s == "cpu");
    let (device, dtype) = if cpu {
        (Device::Cpu, DType::F32)
    } else {
        (Device::new_metal(0)?, DType::BF16)
    };

    let weights = format!("{model_dir}/speech_tokenizer/model.safetensors");
    let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights], dtype, &device)? };
    let decoder = SpeechTokenizerDecoder::load(&SpeechTokenizerConfig::default(), vb, &device)?;

    let bytes = std::fs::read(codes_path)?;
    let codes: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let frames = codes.len() / 16;
    let codes = Tensor::from_vec(codes, (1, 16, frames), &device)?;

    let decode_all = || -> anyhow::Result<(Vec<f32>, Vec<f64>)> {
        let mut stream = DecoderStream::new();
        let mut audio = Vec::new();
        let mut times = Vec::new();
        let mut start = 0;
        while start < frames {
            let n = chunk.min(frames - start);
            let part = codes.narrow(2, start, n)?.contiguous()?;
            let started = Instant::now();
            let out = decoder.decode_chunk(&part, &mut stream)?;
            let out = out.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            if n == chunk {
                times.push(started.elapsed().as_secs_f64() * 1000.0);
            }
            audio.extend(out);
            start += n;
        }
        Ok((audio, times))
    };

    // The first pass compiles kernels and is not counted.
    decode_all()?;
    let mut all = Vec::new();
    let mut audio = Vec::new();
    for _ in 0..3 {
        let (out, times) = decode_all()?;
        all.extend(times);
        audio = out;
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    println!(
        "{frames} frames in chunks of {chunk}: per chunk min {:.2} ms, median {:.2} ms, mean {mean:.2} ms ({} chunks)",
        all[0],
        all[all.len() / 2],
        all.len()
    );
    let raw: Vec<u8> = audio.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(out_path, raw)?;
    Ok(())
}
