//! Code Predictor — generates codec groups 1..N from group-0 hidden states.
//!
//! After the Talker LM produces group-0 tokens and hidden states, the Code
//! Predictor fills in the remaining codebook groups for each time step.
//!
//! Architecture:
//! ```text
//! talker_hidden ─► projection ─► Transformer ─┬─► head[1] ─► group-1 token
//!                                              ├─► head[2] ─► group-2 token
//!           prev_group_embed ─────────────────┘   …
//!                                              └─► head[N-1] ─► group-(N-1) token
//! ```

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use crate::layers::metal_ops::Linear;
use candle_nn::{Embedding, Module, VarBuilder};

use crate::layers::attention::GqaConfig;
use crate::layers::transformer::TransformerBlock;
use crate::tensor_utils::{precompute_rope_freqs, RmsNorm};

use super::config::CodePredictorConfig;

/// Code predictor sampling, from the generation_config.json of the model
/// (subtalker_temperature; subtalker_top_k is 50 and subtalker_top_p 1.0).
const SUBTALKER_TEMPERATURE: f64 = 0.9;

/// The Code Predictor sub-model.
pub struct CodePredictor {
    /// Project from Talker hidden size to Code Predictor hidden size.
    input_proj: Linear,
    /// Per-group codec embeddings: group_embeds[g] for groups 1..N-1.
    group_embeds: Vec<Embedding>,
    /// Transformer layers (smaller than the Talker).
    layers: Vec<TransformerBlock>,
    /// Final norm.
    norm: RmsNorm,
    /// Per-group prediction heads.
    group_heads: Vec<Linear>,
    /// Precomputed RoPE cosine.
    rope_cos: Tensor,
    /// Precomputed RoPE sine.
    rope_sin: Tensor,
    /// Number of codebook groups (including group 0 from Talker).
    num_groups: usize,
    /// Config reference.
    config: CodePredictorConfig,
}

impl CodePredictor {
    /// Load the Code Predictor from a VarBuilder.
    pub fn load(
        config: &CodePredictorConfig,
        talker_hidden_size: usize,
        vb: VarBuilder,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let rope_theta = 1_000_000.0;
        let max_position_embeddings = 128; // CP only needs ~18 positions max

        let gqa_config = GqaConfig::with_head_dim(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            max_position_embeddings,
            rope_theta,
            1e-6,
        );

        // Projection from Talker hidden → Code Predictor hidden
        // Actual name: talker.code_predictor.small_to_mtp_projection.{weight,bias}
        let input_proj = crate::layers::metal_ops::linear(
            talker_hidden_size,
            config.hidden_size,
            vb.pp("small_to_mtp_projection"),
        )?;

        // Per-group embeddings (for groups 1..N-1)
        // Actual name: talker.code_predictor.model.codec_embedding.N.weight
        // Embed dim = talker_hidden_size (2048), NOT code predictor hidden (1024).
        // The flow is: codec_embed + talker_hidden → projection → transformer.
        let num_extra_groups = config.num_code_groups - 1;
        let model_vb = vb.pp("model");
        let mut group_embeds = Vec::with_capacity(num_extra_groups);
        for g in 0..num_extra_groups {
            let emb = candle_nn::embedding(
                config.vocab_size,
                talker_hidden_size,
                model_vb.pp(format!("codec_embedding.{}", g)),
            )?;
            group_embeds.push(emb);
        }

        // Transformer layers under talker.code_predictor.model.layers.N
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let block = TransformerBlock::load(
                &gqa_config,
                config.intermediate_size,
                model_vb.pp(format!("layers.{}", i)),
            )?;
            layers.push(block);
        }

        // Final norm: talker.code_predictor.model.norm
        let norm = RmsNorm::load(config.hidden_size, 1e-6, model_vb.pp("norm"))?;

        // Per-group heads: talker.code_predictor.lm_head.N.weight
        let mut group_heads = Vec::with_capacity(num_extra_groups);
        for g in 0..num_extra_groups {
            let head = crate::layers::metal_ops::linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp(format!("lm_head.{}", g)),
            )?;
            group_heads.push(head);
        }

        // Precompute RoPE for the code predictor transformer
        let (rope_cos, rope_sin) = precompute_rope_freqs(
            config.head_dim,
            max_position_embeddings,
            rope_theta,
            device,
            dtype,
        )?;

        Ok(Self {
            input_proj,
            group_embeds,
            layers,
            norm,
            group_heads,
            rope_cos,
            rope_sin,
            num_groups: config.num_code_groups,
            config: config.clone(),
        })
    }

    /// Embed codes of codebook `group` (1..N-1) in the talker's hidden space.
    pub fn embed_group(&self, group: usize, token_ids: &Tensor) -> Result<Tensor> {
        self.group_embeds[group - 1].forward(token_ids)
    }

    /// Predict groups 1..N-1 for a single time step and return the summed
    /// embedding of all groups (used in the talker's feedback loop), plus
    /// the predicted token IDs for groups 1..N-1.
    ///
    /// Reference flow per generation step:
    /// 1. `last_id_hidden = talker.codec_embedding(group0_token)` — shape (1, 1, talker_hidden)
    /// 2. `code_predictor.generate(cat(past_hidden, last_id_hidden), max_new_tokens=N-1)`
    /// 3. `codec_hiddens = [last_id_hidden] + [cp_embed[i](predicted[i]) for i in 0..N-2]`
    /// 4. `return codec_hiddens.sum(1, keepdim=True)` — shape (1, 1, talker_hidden)
    ///
    /// * `past_hidden` — (1, talker_hidden_size) from talker's last hidden state
    /// * `g0_token` — group-0 token ID
    /// * `g0_embed` — (1, 1, talker_hidden_size) from talker.codec_embedding(g0_token)
    ///
    /// Returns `(summed_embed, predicted_tokens)`:
    /// - `summed_embed`: (1, 1, talker_hidden_size) — the summed embedding for all 16 groups
    /// - `predicted_tokens`: Vec<u32> of length num_groups-1 — predicted tokens for groups 1..N-1
    pub fn predict_step_and_sum(
        &mut self,
        past_hidden: &Tensor,
        _g0_token: u32,
        g0_embed: &Tensor,
        device: &Device,
    ) -> Result<(Tensor, Vec<u32>)> {
        let num_extra_groups = self.num_groups - 1;
        let mut predicted_tokens: Vec<u32> = Vec::with_capacity(num_extra_groups);

        // Start with group-0 embedding in the sum
        let mut summed = g0_embed.clone(); // (1, 1, hidden)

        // Clear KV-caches
        for layer in &mut self.layers {
            layer.clear_cache();
        }

        // Prefill: [past_hidden(1,1,H), g0_embed(1,1,H)] in talker-hidden space
        let past_hidden_3d = past_hidden.unsqueeze(1)?; // (1, 1, H)
        let prefill = Tensor::cat(&[&past_hidden_3d, g0_embed], 1)?; // (1, 2, H)

        // Project to CP hidden size
        let prefill = self.input_proj.forward(&prefill)?; // (1, 2, cp_hidden)

        let mut hidden = prefill;

        // Build causal mask for 2-token prefill: position 0 can only see itself,
        // position 1 can see both (matching HuggingFace's causal mask behavior).
        let prefill_mask = {
            let mask_data = vec![0.0f32, f32::NEG_INFINITY, 0.0f32, 0.0f32];
            let mask = Tensor::from_vec(mask_data, (2, 2), device)?.to_dtype(hidden.dtype())?;
            mask.unsqueeze(0)?.unsqueeze(0)?
        };

        // Run transformer on prefill with real RoPE and causal mask
        for layer in &mut self.layers {
            hidden = layer.forward(
                &hidden,
                &self.rope_cos,
                &self.rope_sin,
                0,
                Some(&prefill_mask),
            )?;
        }

        let hidden = self.norm.forward(&hidden)?;

        // Predict group 1 from last position
        let last_h = hidden.i((.., hidden.dims()[1] - 1.., ..))?; // (1, 1, cp_hidden)
        let logits = self.group_heads[0].forward(&last_h)?;
        // Sampled on the GPU: the code stays there as the next step's input,
        // and the frame's codes reach the CPU once, at its end.
        let mut prev_predicted_token = sample_top_k_top_p_gpu(&logits, SUBTALKER_TEMPERATURE, 50, 1.0)?; // (1,) u32
        let mut predicted: Vec<Tensor> = Vec::with_capacity(num_extra_groups);
        predicted.push(prev_predicted_token.clone());

        // Embed predicted group-1 token and add to sum
        let tok_tensor = prev_predicted_token.unsqueeze(0)?;
        let tok_embed = self.group_embeds[0].forward(&tok_tensor)?; // (1, 1, talker_hidden)
        summed = summed.add(&tok_embed)?;

        // For remaining groups, use incremental decoding with KV-cache
        for gen_step in 1..num_extra_groups {
            let pos = gen_step + 1; // Two prefill positions, then one per group.
                                    // Embed the previous step's predicted token using its group embedding
            let prev_tok_tensor = prev_predicted_token.unsqueeze(0)?;
            let embed_idx = (gen_step - 1).min(self.group_embeds.len() - 1);
            let prev_embed = self.group_embeds[embed_idx].forward(&prev_tok_tensor)?;

            // Project and run one step with real RoPE
            let step_input = self.input_proj.forward(&prev_embed)?;
            let mut h = step_input;
            for layer in &mut self.layers {
                h = layer.forward(&h, &self.rope_cos, &self.rope_sin, pos, None)?;
            }
            let h = self.norm.forward(&h)?;

            let logits = self.group_heads[gen_step].forward(&h)?;
            let predicted_token = sample_top_k_top_p_gpu(&logits, SUBTALKER_TEMPERATURE, 50, 1.0)?;
            predicted.push(predicted_token.clone());

            // Embed and add to sum
            let pred_tensor = predicted_token.unsqueeze(0)?;
            let pred_embed_idx = gen_step.min(self.group_embeds.len() - 1);
            let pred_embed = self.group_embeds[pred_embed_idx].forward(&pred_tensor)?;
            summed = summed.add(&pred_embed)?;

            prev_predicted_token = predicted_token;
        }

        // Clear caches for next call
        for layer in &mut self.layers {
            layer.clear_cache();
        }

        predicted_tokens.extend(Tensor::cat(&predicted, 0)?.to_vec1::<u32>()?);
        Ok((summed, predicted_tokens))
    }
}

impl std::fmt::Debug for CodePredictor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodePredictor")
            .field("num_groups", &self.num_groups)
            .field("num_layers", &self.layers.len())
            .field("hidden_size", &self.config.hidden_size)
            .finish()
    }
}

/// Top-k then top-p sampling done entirely on the logits' device: returns
/// the chosen token id as a (1,) u32 tensor, with no copy to the CPU. With
/// top_p 1.0 (the reference setting) on Metal it is one fused kernel;
/// otherwise the draw is Gumbel-max over the kept probabilities.
///
/// Candle's Metal sort is wrong for rows over 1024 values, so the top k is
/// taken in two stages: per block of 1024, then among those candidates.
fn sample_top_k_top_p_gpu(
    logits: &Tensor,
    temperature: f64,
    top_k: usize,
    top_p: f32,
) -> Result<Tensor> {
    // One fused kernel for the reference settings (top_p 1.0).
    #[cfg(feature = "metal")]
    if top_p >= 1.0 {
        use std::sync::atomic::{AtomicU64, Ordering};
        static DRAWS: AtomicU64 = AtomicU64::new(0);
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
            .wrapping_add(DRAWS.fetch_add(1, Ordering::Relaxed).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        if let Some(token) = crate::layers::metal_ops::sample_top_k(logits, top_k, temperature, seed)? {
            return Ok(token);
        }
    }
    const BLOCK: usize = 1024;
    let device = logits.device();
    let logits = (logits.to_dtype(DType::F32)?.flatten_all()? / temperature)?;
    let vocab = logits.dims1()?;
    let blocks = vocab.div_ceil(BLOCK);
    let logits = if vocab % BLOCK == 0 {
        logits
    } else {
        let pad = Tensor::full(f32::NEG_INFINITY, blocks * BLOCK - vocab, device)?;
        Tensor::cat(&[&logits, &pad], 0)?
    };
    let k = top_k.min(BLOCK).min(vocab);
    let (values, order) = logits.reshape((blocks, BLOCK))?.sort_last_dim(false)?;
    let values = values.narrow(1, 0, k)?.flatten_all()?;
    let offsets: Vec<u32> = (0..blocks).map(|b| (b * BLOCK) as u32).collect();
    let offsets = Tensor::new(offsets.as_slice(), device)?.unsqueeze(1)?;
    let ids = order.narrow(1, 0, k)?.broadcast_add(&offsets)?.flatten_all()?;
    let (sorted, pick) = values.sort_last_dim(false)?;
    let sorted = sorted.narrow(0, 0, k)?;
    let ids = ids.gather(&pick.narrow(0, 0, k)?.contiguous()?, 0)?;
    let probs = candle_nn::ops::softmax_last_dim(&sorted)?;
    // Keep the smallest prefix whose mass reaches top_p (the first always).
    let before = (probs.cumsum(0)? - &probs)?;
    let keep = before.lt(top_p as f64)?.to_dtype(DType::F32)?;
    let uniform = Tensor::rand(1e-6f32, 1.0f32, k, device)?;
    let gumbel = uniform.log()?.neg()?.log()?.neg()?;
    // Excluded codes are pushed far down rather than given log(0).
    let excluded = ((keep.neg()? + 1.0)? * 1e9)?;
    let scores = ((probs.log()? + gumbel)? - excluded)?;
    let chosen = scores.argmax(0)?.unsqueeze(0)?; // (1,) u32
    ids.gather(&chosen, 0)
}
