//! Qwen3-TTS full inference model.
//!
//! Pipeline:
//!   Text + Speaker + Instruct → Talker LM (codec group 0)
//!                                   ↓
//!                        Code Predictor (codec groups 1..N)
//!                                   ↓
//!                    Speech Tokenizer Decoder (codes → 24kHz waveform)

use candle_core::{DType, Device, Tensor};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use tracing::info;

use crate::audio::AudioSamples;
use crate::config::{ModelFiles, TtsConfig};
use crate::error::TtsError;
use crate::tokenizer::TextTokenizer;
use crate::traits::{
    ModelInfo, ReferenceAudio, SynthesisRequest, TtsModel, VoiceCloning, VoiceEmbedding,
};

use super::code_predictor::CodePredictor;
use super::config::Qwen3TtsConfig;
use super::speech_tokenizer::{DecoderStream, SpeechTokenizerConfig, SpeechTokenizerDecoder};
use super::talker::{TalkerGenerationConfig, TalkerLm};

/// How cloned voices are exported, since the speaker and speech encoders
/// are not ported yet.
const VOICE_EXPORT_HINT: &str = "export the voice with scripts/qwen3_tts_voice.py \
     (mlx-audio) and pass it with SynthesisRequest::with_voice_embedding";

/// In-context cloning repeats codes easily; like the reference, it uses a
/// stronger repetition penalty over the latest tokens.
const ICL_REPETITION_PENALTY: f32 = 1.5;
const ICL_REPETITION_WINDOW: usize = 64;

/// A cloned voice, prepared once and reused for each sentence.
struct PreparedVoice {
    key: u64,
    /// The speaker x-vector, in the codec prefix: (1, 1, hidden).
    speaker: Tensor,
    /// The reference prompt, for in-context cloning.
    icl: Option<IclPrompt>,
}

/// What in-context cloning adds to the prompt, and the decoder state after
/// the reference audio.
struct IclPrompt {
    /// The reference transcript's tokens, read before the sentence's.
    text_ids: Vec<u32>,
    /// The reference frames' summed codebook embeddings over tts_pad, which
    /// follow codec_bos at the end of the prompt: (1, frames, hidden).
    codec_rows: Tensor,
    /// A speech-tokenizer stream that has decoded the reference frames, so
    /// that each sentence's audio continues from them without decoding
    /// them again.
    primed: DecoderStream,
}

/// The speaker slot of the codec prefix.
#[derive(Clone, Copy)]
enum Speaker<'a> {
    Id(u32),
    XVector(&'a Tensor),
}

/// Qwen3-TTS-12Hz TTS model.
///
/// 1.7B parameter multi-codebook language model for 10 languages.
/// Supports named speakers and instruction-based style control.
pub struct Qwen3TtsModel {
    config: Qwen3TtsConfig,
    device: Device,
    dtype: DType,
    files: ModelFiles,
    /// Loaded once: loading it takes longer than a sentence's prompt.
    tokenizer: TextTokenizer,
    /// Wrapped in Mutex for interior mutability (KV-cache updates during generation).
    talker: Mutex<TalkerLm>,
    /// Wrapped in Mutex for interior mutability (KV-cache updates during prediction).
    code_predictor: Option<Mutex<CodePredictor>>,
    speech_tokenizer: Option<SpeechTokenizerDecoder>,
    /// The last cloned voice used, prepared.
    voice: Mutex<Option<Arc<PreparedVoice>>>,
}

impl TtsModel for Qwen3TtsModel {
    fn load(config: TtsConfig) -> Result<Self, TtsError> {
        let device = config.device.resolve()?;
        let mut dtype = config.dtype.to_candle();

        // CPU does not support BF16 matmul; fall back to F32 automatically.
        if matches!(device, Device::Cpu) && dtype == DType::BF16 {
            info!("BF16 is not supported on CPU; falling back to F32");
            dtype = DType::F32;
        }
        info!("Loading Qwen3-TTS model on {:?}", device);

        // ── Resolve files (explicit → directory → HF download) ────────
        let files = config.resolve_files()?;

        // ── Parse model config ────────────────────────────────────────
        let config_bytes = files
            .config
            .as_ref()
            .expect("validated by resolve_files")
            .read_bytes()?;
        let model_config = Qwen3TtsConfig::from_bytes(config_bytes.as_ref())?;

        info!(
            "Qwen3-TTS config loaded: type={}, talker layers={}, hidden_size={}, code_groups={}",
            model_config.tts_model_type,
            model_config.talker_config.num_hidden_layers,
            model_config.talker_config.hidden_size,
            model_config.talker_config.num_code_groups,
        );

        let tokenizer = TextTokenizer::from_asset(
            files
                .tokenizer
                .as_ref()
                .expect("validated by resolve_files"),
        )?;

        // ── Load main model weights ───────────────────────────────────
        let vb = ModelFiles::load_safetensors_vb(&files.weights, dtype, &device)?;

        // ── Build Talker LM ──────────────────────────────────────────
        let talker = TalkerLm::load(&model_config.talker_config, vb.pp("talker"), &device, dtype)
            .map_err(|e| {
            TtsError::WeightLoadError(format!("Failed to build Talker LM: {}", e))
        })?;
        info!(
            "Talker LM loaded ({} layers)",
            model_config.talker_config.num_hidden_layers
        );

        // ── Build Code Predictor ──────────────────────────────────────
        let code_predictor =
            if let Some(ref cp_config) = model_config.talker_config.code_predictor_config {
                let cp = CodePredictor::load(
                    cp_config,
                    model_config.talker_config.hidden_size,
                    vb.pp("talker").pp("code_predictor"),
                    &device,
                    dtype,
                )
                .map_err(|e| {
                    TtsError::WeightLoadError(format!("Failed to build Code Predictor: {}", e))
                })?;
                info!(
                    "Code Predictor loaded ({} layers)",
                    cp_config.num_hidden_layers
                );
                Some(cp)
            } else {
                info!("No code_predictor_config found; single-group mode");
                None
            };

        // ── Load speech tokenizer weights ─────────────────────────────
        let speech_tokenizer = if !files.speech_tokenizer_weights.is_empty() {
            let st_vb =
                ModelFiles::load_safetensors_vb(&files.speech_tokenizer_weights, dtype, &device)?;
            let st_config = SpeechTokenizerConfig {
                num_groups: model_config.talker_config.num_code_groups,
                ..SpeechTokenizerConfig::default()
            };
            let st = SpeechTokenizerDecoder::load(&st_config, st_vb, &device).map_err(|e| {
                TtsError::WeightLoadError(format!(
                    "Failed to build Speech Tokenizer Decoder: {}",
                    e
                ))
            })?;
            info!(
                "Speech Tokenizer Decoder loaded (upsample {}x)",
                st.upsample_factor()
            );
            Some(st)
        } else {
            info!("Speech tokenizer weights not available; decode step will fail");
            None
        };

        Ok(Self {
            config: model_config,
            device,
            dtype,
            files,
            tokenizer,
            talker: Mutex::new(talker),
            code_predictor: code_predictor.map(Mutex::new),
            speech_tokenizer,
            voice: Mutex::new(None),
        })
    }

    fn synthesize(&self, request: &SynthesisRequest) -> Result<AudioSamples, TtsError> {
        info!("Qwen3-TTS synthesize: \"{}\"", request.text);

        self.validate_request(request)?;
        let voice = self.prepare_voice(request)?;
        let (full_input, trailing_text_hidden) = self.prepare_inputs(request, voice.as_deref())?;

        info!("Input sequence shape: {:?}", full_input.shape());

        let icl = voice.as_ref().and_then(|voice| voice.icl.as_ref());
        let (codec_tokens_g0, hidden_states) = self.generate_codec_tokens(
            &full_input,
            &trailing_text_hidden,
            request,
            icl.is_some(),
            None,
        )?;
        let all_codes = self.assemble_all_codes(&codec_tokens_g0, &hidden_states)?;
        let samples = self.decode_to_audio(&all_codes, icl.map(|icl| &icl.primed))?;

        info!(
            "Synthesized {} samples ({:.2}s at 24000Hz)",
            samples.len(),
            samples.len() as f32 / 24000.0
        );

        Ok(AudioSamples::new(samples, 24000))
    }

    fn synthesize_stream(
        &self,
        request: &SynthesisRequest,
        on_chunk: &mut dyn FnMut(AudioSamples) -> Result<(), TtsError>,
    ) -> Result<(), TtsError> {
        // The first chunk is short so that audio starts early; later ones
        // grow, each decode having a fixed cost, once playback is covered.
        const CHUNKS: [usize; 3] = [4, 8, 25];

        self.validate_request(request)?;
        let speech_tokenizer = self.speech_tokenizer.as_ref().ok_or_else(|| {
            TtsError::ModelError("Qwen3-TTS: speech tokenizer not loaded".to_string())
        })?;
        let voice = self.prepare_voice(request)?;
        let (full_input, trailing_text_hidden) = self.prepare_inputs(request, voice.as_deref())?;
        let icl = voice.as_ref().and_then(|voice| voice.icl.as_ref());

        // A cloned voice's audio continues from its reference's.
        let mut stream = icl.map_or_else(DecoderStream::new, |icl| icl.primed.clone());
        let mut pending: Vec<(u32, Vec<u32>)> = Vec::new();
        let mut chunks = 0usize;
        let mut failure: Option<TtsError> = None;
        let mut decode = |frames: &[(u32, Vec<u32>)],
                          stream: &mut DecoderStream|
         -> Result<(), TtsError> {
            let g0: Vec<u32> = frames.iter().map(|(g0, _)| *g0).collect();
            let groups: Vec<Vec<u32>> = frames.iter().map(|(_, groups)| groups.clone()).collect();
            let codes = self.assemble_all_codes(&g0, &groups)?;
            let audio = speech_tokenizer
                .decode_chunk(&codes, stream)
                .and_then(|audio| audio.to_dtype(candle_core::DType::F32))
                .and_then(|audio| audio.to_vec1::<f32>())
                .map_err(TtsError::ComputeError)?;
            on_chunk(AudioSamples::new(audio, 24000))
        };
        let mut on_frame = |g0: u32, groups: &[u32]| -> candle_core::Result<()> {
            pending.push((g0, groups.to_vec()));
            let due = CHUNKS[chunks.min(CHUNKS.len() - 1)];
            if pending.len() >= due {
                if let Err(error) = decode(&pending, &mut stream) {
                    let message = error.to_string();
                    failure = Some(error);
                    return Err(candle_core::Error::Msg(message));
                }
                chunks += 1;
                pending.clear();
            }
            Ok(())
        };
        let generated = self.generate_codec_tokens(
            &full_input,
            &trailing_text_hidden,
            request,
            icl.is_some(),
            Some(&mut on_frame),
        );
        if let Some(error) = failure {
            return Err(error);
        }
        generated?;
        if !pending.is_empty() {
            decode(&pending, &mut stream)?;
        }
        Ok(())
    }

    fn supports_streaming(&self) -> bool {
        self.speech_tokenizer.is_some()
    }

    fn sample_rate(&self) -> u32 {
        24000
    }

    fn supported_languages(&self) -> Vec<String> {
        let mut langs = self.config.languages();
        langs.sort();
        if !langs.contains(&"auto".to_string()) {
            langs.insert(0, "auto".to_string());
        }
        langs
    }

    fn supported_voices(&self) -> Vec<String> {
        self.config.speakers()
    }

    fn model_info(&self) -> ModelInfo {
        let name = if self.config.is_voice_design() {
            "Qwen3-TTS-12Hz-1.7B-VoiceDesign".to_string()
        } else if self.is_base() {
            "Qwen3-TTS-12Hz-1.7B-Base".to_string()
        } else {
            "Qwen3-TTS-12Hz-1.7B-CustomVoice".to_string()
        };
        ModelInfo {
            name,
            variant: format!(
                "{} ({})",
                self.config.tts_model_type, self.config.tokenizer_type
            ),
            parameters: 1_700_000_000,
            sample_rate: 24000,
            languages: self.supported_languages(),
            voices: self.supported_voices(),
        }
    }
}

impl Qwen3TtsModel {
    /// Get a reference to the compute device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the model config.
    pub fn config(&self) -> &Qwen3TtsConfig {
        &self.config
    }

    /// Get the resolved file paths.
    pub fn files(&self) -> &ModelFiles {
        &self.files
    }

    fn is_base(&self) -> bool {
        self.config.tts_model_type == "base"
    }

    /// Tokenize the request, resolve its speaker and language, and build the
    /// prompt embeddings, with the cloned voice when there is one.
    fn prepare_inputs(
        &self,
        request: &SynthesisRequest,
        voice: Option<&PreparedVoice>,
    ) -> Result<(Tensor, Tensor), TtsError> {
        let token_ids = self.tokenize_text(&request.text)?;
        let instruct_token_ids = self.tokenize_instruct(request.instruct.as_deref())?;
        let (speaker_id, language_id) = self.resolve_speaker_language(request);
        let speaker = match voice {
            Some(voice) => Some(Speaker::XVector(&voice.speaker)),
            None => speaker_id.map(Speaker::Id),
        };
        self.build_input_embeddings(
            &token_ids,
            instruct_token_ids.as_deref(),
            speaker,
            language_id,
            voice.and_then(|voice| voice.icl.as_ref()),
        )
    }

    /// The request's cloned voice, prepared: from the cache when it is the
    /// voice of the previous request, else computed once here.
    fn prepare_voice(
        &self,
        request: &SynthesisRequest,
    ) -> Result<Option<Arc<PreparedVoice>>, TtsError> {
        let Some(embedding) = request.voice_embedding.as_ref() else {
            return Ok(None);
        };
        let key = voice_key(embedding);
        if let Some(voice) = self.voice.lock().unwrap().as_ref() {
            if voice.key == key {
                return Ok(Some(voice.clone()));
            }
        }

        let tc = &self.config.talker_config;
        let hidden = tc.hidden_size;
        if embedding.data().len() != hidden {
            return Err(TtsError::ModelError(format!(
                "Qwen3-TTS: speaker embedding has {} values, the model needs {hidden}",
                embedding.data().len()
            )));
        }
        let speaker = Tensor::new(embedding.data(), &self.device)
            .and_then(|t| t.to_dtype(self.dtype))
            .and_then(|t| t.reshape((1, 1, hidden)))
            .map_err(TtsError::ComputeError)?;

        let icl = match embedding.prompt() {
            Some(prompt) => Some(self.prepare_icl_prompt(&prompt.codes, &prompt.text)?),
            None => None,
        };
        let voice = Arc::new(PreparedVoice { key, speaker, icl });
        *self.voice.lock().unwrap() = Some(voice.clone());
        Ok(Some(voice))
    }

    /// Embed a voice's reference frames for the prompt, tokenize its
    /// transcript, and prime a decoder stream with the frames.
    fn prepare_icl_prompt(&self, codes: &[Vec<u32>], text: &str) -> Result<IclPrompt, TtsError> {
        let groups = self.config.talker_config.num_code_groups;
        let codebook = 2048;
        if codes.is_empty()
            || codes
                .iter()
                .any(|frame| frame.len() != groups || frame.iter().any(|&c| c >= codebook))
        {
            return Err(TtsError::ModelError(format!(
                "Qwen3-TTS: voice prompt codes must be frames of {groups} codes below {codebook}"
            )));
        }
        if text.trim().is_empty() {
            return Err(TtsError::ModelError(
                "Qwen3-TTS: voice prompt has codes but no transcript".into(),
            ));
        }
        let speech_tokenizer = self.speech_tokenizer.as_ref().ok_or_else(|| {
            TtsError::ModelError("Qwen3-TTS: speech tokenizer not loaded".to_string())
        })?;

        // The transcript alone: <|im_start|>assistant\n{text}<|im_end|>\n
        // without its 3 role tokens and 2 closing ones, as the reference does.
        let ids = self.tokenizer.encode(&format!("<|im_start|>assistant\n{text}<|im_end|>\n"))?;
        if ids.len() < 6 {
            return Err(TtsError::ModelError(
                "Qwen3-TTS: voice prompt transcript too short".into(),
            ));
        }
        let text_ids = ids[3..ids.len() - 2].to_vec();

        let frames = codes.len();
        let flat: Vec<u32> = codes.iter().flatten().copied().collect();
        // (1, frames, groups) for the embeddings, (1, groups, frames) to decode.
        let codes = Tensor::from_vec(flat, (1, frames, groups), &self.device)
            .map_err(TtsError::ComputeError)?;
        let codec_rows = {
            let talker = self.talker.lock().unwrap();
            let cp = self
                .code_predictor
                .as_ref()
                .ok_or_else(|| {
                    TtsError::ModelError("Qwen3-TTS: voice cloning needs the code predictor".into())
                })?
                .lock()
                .unwrap();
            let group = |g: usize| codes.narrow(2, g, 1).and_then(|c| c.squeeze(2));
            let mut sum = talker
                .embed_codec(&group(0).map_err(TtsError::ComputeError)?)
                .map_err(TtsError::ComputeError)?;
            for g in 1..groups {
                let ids = group(g).map_err(TtsError::ComputeError)?;
                let embed = cp.embed_group(g, &ids).map_err(TtsError::ComputeError)?;
                sum = (sum + embed).map_err(TtsError::ComputeError)?;
            }
            let pad = Tensor::new(&[[self.config.tts_pad_token_id]], &self.device)
                .and_then(|ids| talker.embed_text(&ids))
                .map_err(TtsError::ComputeError)?;
            sum.broadcast_add(&pad).map_err(TtsError::ComputeError)?
        };

        // Decoding the reference here, once, leaves the stream where each
        // sentence's audio starts; the reference's own samples are dropped.
        let mut primed = DecoderStream::new();
        let tokens = codes
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .map_err(TtsError::ComputeError)?;
        let dropped = speech_tokenizer
            .decode_chunk(&tokens, &mut primed)
            .and_then(|audio| audio.dim(0))
            .map_err(TtsError::ComputeError)?;
        debug_assert_eq!(dropped, frames * speech_tokenizer.upsample_factor());
        info!(
            "Voice prompt: {} transcript tokens, {frames} frames ({dropped} reference samples)",
            text_ids.len()
        );
        Ok(IclPrompt {
            text_ids,
            codec_rows,
            primed,
        })
    }

    /// Normalize a user-supplied language tag to the key used in `codec_language_id`.
    ///
    /// Accepts ISO 639-1 codes ("en"), full names ("english"), or "auto".
    /// Returns the canonical key that exists in the model's `codec_language_id` map,
    /// or the input unchanged if no mapping is found (validation will catch it later).
    fn normalize_language(&self, lang: &str) -> String {
        // "auto" is always valid
        if lang == "auto" {
            return "auto".to_string();
        }

        // If the key already exists (case-insensitive match), return it.
        let lower = lang.to_lowercase();
        for key in self.config.talker_config.codec_language_id.keys() {
            if key.to_lowercase() == lower {
                return key.clone();
            }
        }

        // Map ISO 639-1 codes to full names.
        let full_name = match lower.as_str() {
            "en" => "english",
            "zh" | "cmn" => "chinese",
            "de" => "german",
            "fr" => "french",
            "es" => "spanish",
            "ja" | "jp" => "japanese",
            "ko" | "kr" => "korean",
            "pt" => "portuguese",
            "ru" => "russian",
            "it" => "italian",
            _ => return lang.to_string(),
        };

        // Find the canonical-cased key that matches.
        for key in self.config.talker_config.codec_language_id.keys() {
            if key.to_lowercase() == full_name {
                return key.clone();
            }
        }

        full_name.to_string()
    }

    /// Validate the synthesis request (voice cloning, voice name, language).
    fn validate_request(&self, request: &SynthesisRequest) -> Result<(), TtsError> {
        if request.speed.is_some() {
            return Err(TtsError::ModelError(
                "Qwen3-TTS does not support explicit speaking speed".into(),
            ));
        }
        if request.max_tokens == Some(0) || request.temperature.is_some_and(|v| !v.is_finite()) {
            return Err(TtsError::ConfigError(
                "max_tokens must be positive and temperature finite".into(),
            ));
        }
        if request.reference_audio.is_some() {
            return Err(TtsError::ModelError(format!(
                "Qwen3-TTS: cloning from raw reference audio is not implemented; {VOICE_EXPORT_HINT}"
            )));
        }

        if let Some(ref embedding) = request.voice_embedding {
            // A cloned voice replaces any named speaker.
            if !self.is_base() {
                return Err(TtsError::ModelError(format!(
                    "Qwen3-TTS: voice cloning needs the Base model, not {}",
                    self.config.tts_model_type
                )));
            }
            if embedding.model_type() != "qwen3-tts" {
                return Err(TtsError::ModelError(format!(
                    "Qwen3-TTS: voice embedding is for {}, not qwen3-tts",
                    embedding.model_type()
                )));
            }
        } else if self.config.is_voice_design() {
            if request.voice.is_some() {
                return Err(TtsError::ModelError(
                    "Qwen3-TTS VoiceDesign uses `instruct` descriptions instead of named speakers."
                        .to_string(),
                ));
            }
        } else {
            if let Some(ref voice) = request.voice {
                if !self.config.talker_config.spk_id.contains_key(voice) {
                    return Err(TtsError::UnknownVoice(voice.clone()));
                }
            }
        }

        if let Some(ref lang) = request.language {
            let normalized = self.normalize_language(lang);
            if normalized != "auto"
                && !self
                    .config
                    .talker_config
                    .codec_language_id
                    .contains_key(&normalized)
            {
                return Err(TtsError::UnsupportedLanguage(lang.clone()));
            }
        }

        Ok(())
    }

    /// Tokenize the input text using the chat template format.
    ///
    /// Returns the token IDs for the full assistant template:
    /// `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
    /// The caller should use the first 3 tokens as the role prefix and
    /// strip the last 5 tokens (closing template).
    fn tokenize_text(&self, text: &str) -> Result<Vec<u32>, TtsError> {

        // Build the chat template: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
        let template_text = format!(
            "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
            text
        );
        let token_ids = self.tokenizer.encode(&template_text)?;

        if token_ids.len() < 8 {
            return Err(TtsError::ModelError(
                "Qwen3-TTS: text too short after tokenization".to_string(),
            ));
        }

        info!("Tokenized {} tokens (chat template)", token_ids.len());
        Ok(token_ids)
    }

    /// Tokenize an optional voice/style instruction using the upstream user-role template.
    fn tokenize_instruct(&self, instruct: Option<&str>) -> Result<Option<Vec<u32>>, TtsError> {
        let Some(instruct) = instruct.map(str::trim).filter(|s| !s.is_empty()) else {
            return Ok(None);
        };


        let template_text = format!("<|im_start|>user\n{}<|im_end|>\n", instruct);
        let token_ids = self.tokenizer.encode(&template_text)?;
        info!("Tokenized {} instruction tokens", token_ids.len());
        Ok(Some(token_ids))
    }

    /// Resolve speaker and language IDs from the request.
    fn resolve_speaker_language(&self, request: &SynthesisRequest) -> (Option<u32>, Option<u32>) {
        let speaker_id = request
            .voice
            .as_deref()
            .filter(|_| request.voice_embedding.is_none())
            .and_then(|v| self.config.talker_config.spk_id.get(v).copied());

        let lang_str = self.normalize_language(request.language.as_deref().unwrap_or("auto"));
        let language_id = self
            .config
            .talker_config
            .codec_language_id
            .get(&lang_str)
            .copied();

        info!(
            "Speaker ID: {:?}, Language ID: {:?}",
            speaker_id, language_id
        );
        (speaker_id, language_id)
    }

    /// Build the input embedding tensor from text tokens, speaker, and language.
    ///
    /// With a voice prompt (in-context cloning), the reference transcript is
    /// read before the text, and the reference frames follow codec_bos, so
    /// that generation continues the reference audio with the text.
    ///
    /// Returns `(input_embeds, trailing_text_hidden)`:
    /// - `input_embeds`: The combined text+codec embedding for the prefix.
    /// - `trailing_text_hidden`: Text embeddings to add at each generation step.
    ///   During generation step i, add `trailing_text_hidden[i]` to the codec embedding.
    ///   If i >= len, add tts_pad_embed instead.
    fn build_input_embeddings(
        &self,
        token_ids: &[u32],
        instruct_token_ids: Option<&[u32]>,
        speaker: Option<Speaker>,
        language_id: Option<u32>,
        icl: Option<&IclPrompt>,
    ) -> Result<(Tensor, Tensor), TtsError> {
        let talker = self.talker.lock().unwrap();
        let tc = &self.config.talker_config;

        // Extract role prefix (first 3 tokens) and text content (skip first 3, last 5)
        let role_tokens = &token_ids[..3]; // <|im_start|> assistant \n
        let text_content = if token_ids.len() > 8 {
            &token_ids[3..token_ids.len() - 5]
        } else {
            &token_ids[3..token_ids.len().saturating_sub(5).max(3)]
        };

        // --- Build text channel (all positions) ---
        let mut text_ids: Vec<u32> = Vec::new();
        let mut codec_ids: Vec<u32> = Vec::new();
        let mut codec_mask: Vec<f32> = Vec::new(); // 0.0 = no codec, 1.0 = add codec

        // Positions 0-2: role prefix (text only, no codec)
        for &tok in role_tokens {
            text_ids.push(tok);
            codec_ids.push(0);
            codec_mask.push(0.0);
        }

        // Positions 3+: think/language/speaker tokens
        // Build codec prefix based on language presence
        let codec_pad = tc.codec_pad_id.unwrap_or(2148);
        let codec_bos = tc.codec_bos_id.unwrap_or(2149);
        let codec_think = tc.codec_think_id.unwrap_or(2154);
        let codec_nothink = tc.codec_nothink_id.unwrap_or(2155);
        let codec_think_bos = tc.codec_think_bos_id.unwrap_or(2156);
        let codec_think_eos = tc.codec_think_eos_id.unwrap_or(2157);

        let tts_pad = self.config.tts_pad_token_id;
        let tts_bos = self.config.tts_bos_token_id;
        let tts_eos = self.config.tts_eos_token_id;

        if let Some(lang_id) = language_id {
            // With language: think, think_bos, language, think_eos
            let think_codec = [codec_think, codec_think_bos, lang_id, codec_think_eos];
            for &c in &think_codec {
                text_ids.push(tts_pad);
                codec_ids.push(c);
                codec_mask.push(1.0);
            }
        } else {
            // Without language: nothink, think_bos, think_eos
            let think_codec = [codec_nothink, codec_think_bos, codec_think_eos];
            for &c in &think_codec {
                text_ids.push(tts_pad);
                codec_ids.push(c);
                codec_mask.push(1.0);
            }
        }

        // Speaker position: a speaker's codec token, or an x-vector added
        // to the embeddings below.
        let xvector_position = text_ids.len();
        match speaker {
            Some(Speaker::Id(spk_id)) => {
                text_ids.push(tts_pad);
                codec_ids.push(spk_id);
                codec_mask.push(1.0);
            }
            Some(Speaker::XVector(_)) => {
                text_ids.push(tts_pad);
                codec_ids.push(0);
                codec_mask.push(0.0);
            }
            None => {}
        }

        // Codec pad position — reference uses tts_bos here (not tts_pad).
        // The reference builds: [tts_pad]*N + tts_bos, combined with codec[:, :-1].
        // So the last position before text has tts_bos on the text side with codec_pad on the codec side.
        text_ids.push(tts_bos);
        codec_ids.push(codec_pad);
        codec_mask.push(1.0);

        // Text content positions (each text token + codec_pad), after the
        // voice prompt's transcript
        let prompt_text = icl.map_or(&[][..], |icl| &icl.text_ids);
        for &tok in prompt_text.iter().chain(text_content) {
            text_ids.push(tok);
            codec_ids.push(codec_pad);
            codec_mask.push(1.0);
        }

        // tts_eos position
        text_ids.push(tts_eos);
        codec_ids.push(codec_pad);
        codec_mask.push(1.0);

        // Generation start: tts_pad + codec_bos
        text_ids.push(tts_pad);
        codec_ids.push(codec_bos);
        codec_mask.push(1.0);

        let seq_len = text_ids.len();
        info!(
            "Input sequence: {} positions (role:3, think:{}, text:{}, special:3)",
            seq_len,
            if language_id.is_some() { 4 } else { 3 } + if speaker.is_some() { 1 } else { 0 },
            prompt_text.len() + text_content.len(),
        );

        // --- Compute embeddings ---
        let text_ids_tensor = Tensor::new(text_ids.as_slice(), &self.device)
            .map_err(TtsError::ComputeError)?
            .unsqueeze(0)
            .map_err(TtsError::ComputeError)?;

        let codec_ids_tensor = Tensor::new(codec_ids.as_slice(), &self.device)
            .map_err(TtsError::ComputeError)?
            .unsqueeze(0)
            .map_err(TtsError::ComputeError)?;

        // text_projection(text_embedding(text_ids))
        let text_embeds = talker
            .embed_text(&text_ids_tensor)
            .map_err(TtsError::ComputeError)?;

        // codec_embedding(codec_ids)
        let codec_embeds = talker
            .embed_codec(&codec_ids_tensor)
            .map_err(TtsError::ComputeError)?;

        // Apply codec mask: text_embeds + codec_embeds * mask
        let mask_tensor = Tensor::new(codec_mask.as_slice(), &self.device)
            .map_err(TtsError::ComputeError)?
            .to_dtype(text_embeds.dtype())
            .map_err(TtsError::ComputeError)?
            .reshape((1, seq_len, 1))
            .map_err(TtsError::ComputeError)?;

        let masked_codec = codec_embeds
            .broadcast_mul(&mask_tensor)
            .map_err(TtsError::ComputeError)?;

        let mut combined = text_embeds
            .add(&masked_codec)
            .map_err(TtsError::ComputeError)?;

        if let Some(Speaker::XVector(xvector)) = speaker {
            let at = xvector_position;
            combined = Tensor::cat(
                &[
                    &combined.narrow(1, 0, at)?,
                    &combined.narrow(1, at, 1)?.add(xvector)?,
                    &combined.narrow(1, at + 1, seq_len - at - 1)?,
                ],
                1,
            )
            .map_err(TtsError::ComputeError)?;
        }
        if let Some(icl) = icl {
            combined = Tensor::cat(&[&combined, &icl.codec_rows], 1).map_err(TtsError::ComputeError)?;
        }

        let combined = if let Some(instruct_token_ids) =
            instruct_token_ids.filter(|tokens| !tokens.is_empty())
        {
            let instruct_ids_tensor = Tensor::new(instruct_token_ids, &self.device)
                .map_err(TtsError::ComputeError)?
                .unsqueeze(0)
                .map_err(TtsError::ComputeError)?;
            let instruct_embeds = talker
                .embed_text(&instruct_ids_tensor)
                .map_err(TtsError::ComputeError)?;
            Tensor::cat(&[&instruct_embeds, &combined], 1).map_err(TtsError::ComputeError)?
        } else {
            combined
        };

        // Compute trailing_text_hidden: text embedding added at each generation step.
        //
        // In non-streaming mode (default for CustomVoice), ALL text content is
        // already embedded in the prefill. The trailing text for generation is
        // just tts_pad_embed at every step (matching the reference implementation).
        let trailing_ids: Vec<u32> = vec![tts_pad];

        let trailing_ids_tensor = Tensor::new(trailing_ids.as_slice(), &self.device)
            .map_err(TtsError::ComputeError)?
            .unsqueeze(0)
            .map_err(TtsError::ComputeError)?;

        let trailing_text_hidden = talker
            .embed_text(&trailing_ids_tensor)
            .map_err(TtsError::ComputeError)?;

        info!(
            "trailing_text_hidden len={} (non-streaming: tts_pad only)",
            trailing_ids.len()
        );

        Ok((combined, trailing_text_hidden))
    }

    /// Run the Talker LM to produce group-0 codec tokens and per-step group tokens.
    ///
    /// The generation loop interleaves with the Code Predictor: at each step,
    /// after the talker samples a group-0 token, the code predictor predicts
    /// groups 1-15, and the summed embedding of all 16 groups is fed back as
    /// the next step's input (matching the reference implementation).
    ///
    /// Returns `(g0_tokens, per_step_group_tokens)` where `per_step_group_tokens[i]`
    /// contains the predicted tokens for groups 1..N-1 at generation step i.
    fn generate_codec_tokens(
        &self,
        full_input: &Tensor,
        trailing_text_hidden: &Tensor,
        request: &SynthesisRequest,
        in_context: bool,
        mut on_frame: Option<&mut dyn FnMut(u32, &[u32]) -> candle_core::Result<()>>,
    ) -> Result<(Vec<u32>, Vec<Vec<u32>>), TtsError> {
        let max_tokens = request.max_tokens.unwrap_or(2048);
        // Default to 0.9 temperature with top-k 50 (matching reference)
        let temperature = request.temperature.unwrap_or(0.9);
        let top_k = 50;
        let generation_config = || {
            let config = TalkerGenerationConfig::new(max_tokens, temperature, top_k);
            if in_context {
                config.with_repetition_penalty(ICL_REPETITION_PENALTY, Some(ICL_REPETITION_WINDOW))
            } else {
                config
            }
        };

        // Build the code predictor callback if available.
        // The callback is called at each generation step with:
        //   (past_hidden, g0_token, g0_embed, device) -> (summed_embed, group_tokens)
        let has_cp = self.code_predictor.is_some();

        let (codec_tokens_g0, group_tokens) = if has_cp {
            let cp_mutex = self.code_predictor.as_ref().unwrap();
            let mut predict_fn = |past_hidden: &Tensor,
                                  g0_token: u32,
                                  g0_embed: &Tensor,
                                  dev: &Device|
             -> candle_core::Result<(Tensor, Vec<u32>)> {
                let mut cp = cp_mutex.lock().unwrap();
                let out = cp.predict_step_and_sum(past_hidden, g0_token, g0_embed, dev)?;
                drop(cp);
                if let Some(on_frame) = on_frame.as_deref_mut() {
                    on_frame(g0_token, &out.1)?;
                }
                Ok(out)
            };
            let generation_config = generation_config().with_predict_and_sum_fn(&mut predict_fn);
            self.talker
                .lock()
                .unwrap()
                .generate(full_input, trailing_text_hidden, generation_config)
                .map_err(TtsError::ComputeError)?
        } else {
            self.talker
                .lock()
                .unwrap()
                .generate(full_input, trailing_text_hidden, generation_config())
                .map_err(TtsError::ComputeError)?
        };

        if codec_tokens_g0.is_empty() {
            return Err(TtsError::ModelError(
                "Qwen3-TTS: Talker generated no codec tokens".to_string(),
            ));
        }

        info!(
            "Talker generated {} codec tokens (group 0)",
            codec_tokens_g0.len()
        );
        Ok((codec_tokens_g0, group_tokens))
    }

    /// Assemble all codec groups into a single tensor from the generation results.
    ///
    /// Group 0 comes from the Talker's generated tokens.
    /// Groups 1..N-1 come from the Code Predictor's per-step predictions
    /// (collected during the generation loop).
    ///
    /// Returns tensor of shape (1, num_groups, seq_len).
    fn assemble_all_codes(
        &self,
        codec_tokens_g0: &[u32],
        per_step_group_tokens: &[Vec<u32>],
    ) -> Result<Tensor, TtsError> {
        let num_groups = self.config.talker_config.num_code_groups;

        if per_step_group_tokens.is_empty() || per_step_group_tokens[0].is_empty() {
            // No code predictor — repeat group 0 for all groups
            info!("No Code Predictor; using single group mode");
            return Tensor::new(codec_tokens_g0, &self.device)
                .map_err(TtsError::ComputeError)?
                .unsqueeze(0)
                .map_err(TtsError::ComputeError)?
                .unsqueeze(1)
                .map_err(TtsError::ComputeError)?
                .repeat(&[1, num_groups, 1])
                .map_err(TtsError::ComputeError);
        }

        let num_extra_groups = num_groups - 1;

        // Group 0: (1, 1, seq_len)
        let g0 = Tensor::new(codec_tokens_g0, &self.device)
            .map_err(TtsError::ComputeError)?
            .unsqueeze(0)
            .map_err(TtsError::ComputeError)?
            .unsqueeze(1)
            .map_err(TtsError::ComputeError)?;

        let mut group_tensors: Vec<Tensor> = vec![g0];

        // Groups 1..N-1: extract from per_step_group_tokens
        for g in 0..num_extra_groups {
            let tokens: Vec<u32> = per_step_group_tokens
                .iter()
                .map(|step_tokens| {
                    if g < step_tokens.len() {
                        step_tokens[g]
                    } else {
                        0 // fallback padding
                    }
                })
                .collect();

            let gt = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(TtsError::ComputeError)?
                .unsqueeze(0)
                .map_err(TtsError::ComputeError)?
                .unsqueeze(1)
                .map_err(TtsError::ComputeError)?;

            group_tensors.push(gt);
        }

        let refs: Vec<&Tensor> = group_tensors.iter().collect();
        let codes = Tensor::cat(&refs, 1).map_err(TtsError::ComputeError)?;

        info!("Assembled all codes: {:?}", codes.shape());

        Ok(codes)
    }

    /// Decode codec tokens to audio samples via the speech tokenizer,
    /// continuing a primed stream (a cloned voice's reference) if given.
    fn decode_to_audio(
        &self,
        all_codes: &Tensor,
        primed: Option<&DecoderStream>,
    ) -> Result<Vec<f32>, TtsError> {
        let speech_tokenizer = self.speech_tokenizer.as_ref().ok_or_else(|| {
            TtsError::ModelError(
                "Qwen3-TTS: speech tokenizer not loaded; cannot decode to audio. \
                 Ensure speech_tokenizer_weights are provided or downloadable."
                    .to_string(),
            )
        })?;

        let waveform = match primed {
            Some(primed) => speech_tokenizer.decode_chunk(all_codes, &mut primed.clone()),
            None => speech_tokenizer.decode(all_codes),
        }
        .map_err(TtsError::ComputeError)?;

        // The decoder may produce BF16 on Metal/GPU — always cast to F32 for output.
        let waveform = waveform
            .to_dtype(candle_core::DType::F32)
            .map_err(TtsError::ComputeError)?;
        let samples: Vec<f32> = waveform.to_vec1().map_err(TtsError::ComputeError)?;

        Ok(samples)
    }
}

/// Identifies a voice embedding, to reuse its prepared form.
fn voice_key(embedding: &VoiceEmbedding) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for value in embedding.data() {
        value.to_bits().hash(&mut hasher);
    }
    embedding.shape().hash(&mut hasher);
    if let Some(prompt) = embedding.prompt() {
        prompt.codes.hash(&mut hasher);
        prompt.text.hash(&mut hasher);
    }
    hasher.finish()
}

impl VoiceCloning for Qwen3TtsModel {
    /// The Base model clones voices exported by `scripts/qwen3_tts_voice.py`.
    fn supports_voice_cloning(&self) -> bool {
        self.is_base() && self.speech_tokenizer.is_some()
    }

    fn extract_voice(&self, _audio: &ReferenceAudio) -> Result<VoiceEmbedding, TtsError> {
        Err(TtsError::ModelError(format!(
            "Qwen3-TTS: the speaker and speech encoders are not ported; {VOICE_EXPORT_HINT}"
        )))
    }

    fn synthesize_with_voice(
        &self,
        request: &SynthesisRequest,
        voice: &VoiceEmbedding,
    ) -> Result<crate::audio::AudioSamples, TtsError> {
        self.synthesize(&request.clone().with_voice_embedding(voice.clone()))
    }
}

impl std::fmt::Debug for Qwen3TtsModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3TtsModel")
            .field("config_type", &self.config.tts_model_type)
            .field("device", &self.device)
            .field(
                "talker_layers",
                &self.config.talker_config.num_hidden_layers,
            )
            .field("has_code_predictor", &self.code_predictor.is_some())
            .field("has_speech_tokenizer", &self.speech_tokenizer.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sentence decoded from a voice's primed stream equals the reference
    /// and the sentence decoded whole, cut after the reference's samples.
    /// Needs QWEN3_TTS_BASE (a Base model directory) and QWEN3_TTS_VOICE (a
    /// voice JSON with a prompt); QWEN3_TTS_CPU runs it on the CPU in f32.
    #[test]
    #[ignore = "needs the Qwen3-TTS Base weights"]
    fn primed_decode_matches_whole_decode() -> Result<(), TtsError> {
        let (Some(dir), Some(voice)) = (
            std::env::var_os("QWEN3_TTS_BASE"),
            std::env::var_os("QWEN3_TTS_VOICE"),
        ) else {
            return Ok(());
        };
        let mut config =
            TtsConfig::new(crate::ModelType::Qwen3Tts).with_model_path(dir.to_string_lossy().into_owned());
        if std::env::var_os("QWEN3_TTS_CPU").is_some() {
            config = config.with_device(crate::DeviceSelection::Cpu);
        }
        let model = Qwen3TtsModel::load(config)?;
        let embedding = VoiceEmbedding::load(voice)?;
        let reference = embedding.prompt().expect("voice has a prompt").codes.clone();
        let request = SynthesisRequest::new("Hello.").with_voice_embedding(embedding);
        let voice = model.prepare_voice(&request)?.expect("voice");
        let primed = &voice.icl.as_ref().expect("in-context voice").primed;
        assert_eq!(primed.frames(), reference.len());

        // Stand-in generated frames: the reference's first ones, reversed.
        let generated: Vec<Vec<u32>> = reference.iter().take(40).rev().cloned().collect();
        let tokens = |frames: &[Vec<u32>]| -> candle_core::Result<Tensor> {
            let flat: Vec<u32> = frames.iter().flatten().copied().collect();
            Tensor::from_vec(flat, (1, frames.len(), 16), &model.device)?
                .transpose(1, 2)?
                .contiguous()
        };
        let speech_tokenizer = model.speech_tokenizer.as_ref().expect("speech tokenizer");
        let whole: Vec<f32> = speech_tokenizer
            .decode(&tokens(&[reference.clone(), generated.clone()].concat())?)?
            .to_dtype(DType::F32)?
            .to_vec1()?;
        let cut = reference.len() * speech_tokenizer.upsample_factor();
        let continued = model.decode_to_audio(&tokens(&generated)?, Some(primed))?;
        assert_eq!(continued.len(), whole.len() - cut);

        let (mut error, mut energy, mut max) = (0f64, 0f64, 0f32);
        for (a, b) in continued.iter().zip(&whole[cut..]) {
            error += ((a - b) as f64).powi(2);
            energy += (*b as f64).powi(2);
            max = max.max((a - b).abs());
        }
        let relative = (error / energy).sqrt();
        println!("primed vs whole decode: max |diff| {max:.5}, relative RMS {relative:.5}");
        assert!(relative < 0.02, "relative RMS difference {relative}");
        Ok(())
    }
}
