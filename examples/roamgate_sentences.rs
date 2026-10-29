//! Times Qwen3-TTS on sentences from a TSV (`en|fr<TAB>text`), loading the
//! model once, and writes one WAV per sentence.
//!
//!   cargo run --release --features metal --example roamgate_sentences -- \
//!     <model-dir> <sentences.tsv> <out-dir> [speaker | voice.json]
//!
//! A voice JSON (from scripts/qwen3_tts_voice.py) clones that voice with a
//! Qwen3-TTS Base model instead of using a named speaker.

use any_tts::{
    load_model, DeviceSelection, ModelType, SynthesisRequest, TtsConfig, VoiceEmbedding,
};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model_dir, sentences, out_dir] = [&args[0], &args[1], &args[2]];
    let speaker = args.get(3).cloned().unwrap_or_else(|| "ryan".into());
    let voice = if speaker.ends_with(".json") {
        Some(VoiceEmbedding::load(&speaker)?)
    } else {
        None
    };
    let with_voice = |request: SynthesisRequest| match &voice {
        Some(voice) => request.with_voice_embedding(voice.clone()),
        None => request.with_voice(&speaker),
    };
    std::fs::create_dir_all(out_dir)?;

    let started = Instant::now();
    let model = load_model(
        TtsConfig::new(ModelType::Qwen3Tts)
            .with_model_path(model_dir)
            .with_device(DeviceSelection::Metal(0)),
    )?;
    println!("load {:.0} ms, speakers {:?}", started.elapsed().as_secs_f64() * 1000.0, model.supported_voices());

    // A first run warms the GPU kernels up; it is not counted.
    // With a cloned voice, it also prepares the voice, once.
    let warm = with_voice(SynthesisRequest::new("Warming up.").with_language("English"));
    for _ in 0..2 {
        let started = Instant::now();
        let audio = model.synthesize(&warm)?;
        println!(
            "warm-up {:.0} ms for {:.1} s",
            started.elapsed().as_secs_f64() * 1000.0,
            audio.samples.len() as f64 / audio.sample_rate as f64
        );
    }

    for (index, line) in std::fs::read_to_string(sentences)?.lines().enumerate() {
        let Some((lang, text)) = line.split_once('\t') else { continue };
        let language = if lang == "fr" { "French" } else { "English" };
        let request = with_voice(SynthesisRequest::new(text).with_language(language));
        let started = Instant::now();
        let mut first_audio_ms = None;
        let mut samples = Vec::new();
        let mut rate = 24_000;
        model.synthesize_stream(&request, &mut |chunk| {
            if !chunk.samples.is_empty() {
                first_audio_ms.get_or_insert(started.elapsed().as_secs_f64() * 1000.0);
                rate = chunk.sample_rate;
                samples.extend_from_slice(&chunk.samples);
            }
            Ok(())
        })?;
        let total = started.elapsed().as_secs_f64();
        let seconds = samples.len() as f64 / rate as f64;
        let audio = any_tts::AudioSamples::new(samples, rate);
        let path = format!("{out_dir}/{:02}-{lang}.wav", index + 1);
        audio.save_wav(&path)?;
        println!(
            "{:02} {lang} first audio {:.0} ms, total {:.0} ms, audio {:.1} s, RTF {:.2}: {}",
            index + 1,
            first_audio_ms.unwrap_or(0.0),
            total * 1000.0,
            seconds,
            if seconds > 0.0 { total / seconds } else { f64::NAN },
            text
        );
    }
    Ok(())
}
