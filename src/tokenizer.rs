//! Text tokenizer wrapper.
//!
//! Wraps the HuggingFace `tokenizers` crate for BPE tokenization used by
//! both model backends.

use crate::config::ModelAsset;
use crate::error::TtsError;
use std::path::Path;

/// How Qwen2's tokenizer splits text before byte-level BPE.
const QWEN2_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// A BPE text tokenizer loaded from vocab/merges files or a tokenizer.json.
pub struct TextTokenizer {
    inner: tokenizers::Tokenizer,
}

impl TextTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file, or from a `vocab.json`
    /// (with `merges.txt` and `tokenizer_config.json` beside it).
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TtsError> {
        if path.as_ref().file_name().is_some_and(|name| name == "vocab.json") {
            return Self::from_vocab_merges(path.as_ref().parent().unwrap_or(Path::new(".")));
        }
        let inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|e| TtsError::TokenizerError(format!("Failed to load tokenizer: {}", e)))?;
        Ok(Self { inner })
    }

    /// Load a tokenizer from in-memory `tokenizer.json` bytes.
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self, TtsError> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| TtsError::TokenizerError(format!("Failed to load tokenizer: {}", e)))?;
        Ok(Self { inner })
    }

    /// Load a tokenizer from a resolved model asset.
    pub fn from_asset(asset: &ModelAsset) -> Result<Self, TtsError> {
        if let Some(path) = asset.as_path() {
            return Self::from_file(path);
        }
        let bytes = asset.read_bytes()?;
        Self::from_bytes(bytes.as_ref())
    }

    /// Load a tokenizer from a pretrained model directory.
    /// Looks for `tokenizer.json` first, then falls back to `vocab.json` + `merges.txt`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self, TtsError> {
        let dir = dir.as_ref();

        // Try tokenizer.json first
        let tokenizer_json = dir.join("tokenizer.json");
        if tokenizer_json.exists() {
            return Self::from_file(tokenizer_json);
        }

        // Fall back to vocab.json + merges.txt
        let vocab_path = dir.join("vocab.json");
        let merges_path = dir.join("merges.txt");

        if vocab_path.exists() && merges_path.exists() {
            Self::from_vocab_merges(dir)
        } else {
            Err(TtsError::TokenizerError(format!(
                "No tokenizer files found in {}",
                dir.display()
            )))
        }
    }

    /// Builds a Qwen2 byte-level BPE tokenizer (`Qwen2Tokenizer`, as the
    /// Qwen3-TTS checkpoints ship it) from `vocab.json`, `merges.txt` and the
    /// added tokens of `tokenizer_config.json`, the way transformers does.
    pub fn from_vocab_merges(dir: impl AsRef<Path>) -> Result<Self, TtsError> {
        let dir = dir.as_ref();
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name)).map_err(|e| {
                TtsError::TokenizerError(format!("Failed to read {}: {}", dir.join(name).display(), e))
            })
        };
        let vocab: serde_json::Value = serde_json::from_str(&read("vocab.json")?)?;
        let merges_text = read("merges.txt")?;
        let merges: Vec<&str> = merges_text
            .lines()
            .filter(|line| !line.is_empty() && !line.starts_with("#version"))
            .collect();
        let mut added_tokens = Vec::new();
        if let Ok(config) = read("tokenizer_config.json") {
            let config: serde_json::Value = serde_json::from_str(&config)?;
            if let Some(added) = config["added_tokens_decoder"].as_object() {
                for (id, token) in added {
                    let id: u64 = id.parse().map_err(|_| {
                        TtsError::TokenizerError(format!("Bad added token id {id}"))
                    })?;
                    added_tokens.push(serde_json::json!({
                        "id": id,
                        "content": token["content"],
                        "single_word": token["single_word"].as_bool().unwrap_or(false),
                        "lstrip": token["lstrip"].as_bool().unwrap_or(false),
                        "rstrip": token["rstrip"].as_bool().unwrap_or(false),
                        "normalized": token["normalized"].as_bool().unwrap_or(false),
                        "special": token["special"].as_bool().unwrap_or(true),
                    }));
                }
            }
        }
        let byte_level = serde_json::json!({
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false
        });
        let definition = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added_tokens,
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": QWEN2_SPLIT},
                    "behavior": "Isolated",
                    "invert": false
                },
                byte_level
            ]},
            "post_processor": byte_level,
            "decoder": byte_level,
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocab,
                "merges": merges
            }
        });
        Self::from_bytes(serde_json::to_vec(&definition)?)
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TtsError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| TtsError::TokenizerError(format!("Encoding failed: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back into text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TtsError> {
        self.inner
            .decode(ids, true)
            .map_err(|e| TtsError::TokenizerError(format!("Decoding failed: {}", e)))
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get the token ID for a specific token string, if it exists.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3-TTS's vocab.json and merges.txt give transformers' token ids.
    /// Set QWEN3_TTS_MODEL to a Qwen3-TTS checkpoint directory to run it.
    #[test]
    #[ignore]
    fn vocab_merges_match_transformers() {
        let dir = std::env::var("QWEN3_TTS_MODEL").expect("QWEN3_TTS_MODEL");
        let tokenizer = TextTokenizer::from_vocab_merges(dir).unwrap();
        // From transformers' AutoTokenizer on the same directory.
        let cases: Vec<(&str, Vec<u32>)> = vec![
            ("<|im_start|>assistant\nHello, world! It's 22 h 39.<|im_end|>\n<|im_start|>assistant\n", vec![151644, 77091, 198, 9707, 11, 1879, 0, 1084, 594, 220, 17, 17, 305, 220, 18, 24, 13, 151645, 198, 151644, 77091, 198]),
            ("Le serveur a redémarré à 22 h 39, et la réponse est arrivée.", vec![2304, 8683, 324, 264, 2518, 16647, 1118, 963, 3784, 220, 17, 17, 305, 220, 18, 24, 11, 1842, 1187, 89409, 1788, 67903, 7888, 13]),
            ("I removed \"clean_model\": \"haiku\" from ~/.config/roamgate/voice.json.", vec![40, 6963, 330, 18377, 5047, 788, 330, 4223, 38242, 1, 504, 39958, 1676, 14, 299, 309, 24601, 14, 9862, 4323, 13]),
            ("日本語のテキスト  and   spaces\n\nnew lines\ttabs 3,8 Go 1234567", vec![101059, 102819, 15767, 56833, 61803, 70534, 220, 323, 256, 12621, 271, 931, 5128, 3244, 3435, 220, 18, 11, 23, 5994, 220, 16, 17, 18, 19, 20, 21, 22]),
            ("<tts_pad><tts_text_bos> ¿Qué tal? Ça va très bien, merci !", vec![151671, 151672, 28286, 65706, 8210, 30, 131098, 11164, 24901, 14370, 11, 16481, 72, 753]),
        ];
        for (text, ids) in cases {
            assert_eq!(tokenizer.encode(text).unwrap(), ids, "{text:?}");
        }
    }
}
