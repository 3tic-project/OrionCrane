//! Model backend abstraction for the inference engine.
//!
//! The [`ModelBackend`] trait decouples the engine from specific model implementations,
//! allowing any compatible LLM to be served through the OpenAI-compatible API.
//!
//! # Capability Levels
//!
//! | Capability        | Required | Effect when absent                            |
//! |-------------------|----------|-----------------------------------------------|
//! | `forward_step`    | Yes      | —                                             |
//! | KV cache swap     | No       | `max_concurrent` capped to 1                  |
//! | Batch decode      | No       | Sequences decoded sequentially per step        |

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::paged_kv::PagedKvLayout;

#[derive(Debug, Clone, Copy, Default)]
pub struct BatchDecodeSetupTimings {
    pub kv_len_scan_us: u64,
    pub pad_stack_us: u64,
    pub contiguous_us: u64,
    pub extra_room_alloc_us: u64,
    pub cache_assign_us: u64,
    pub batched_equal_length_layers: u64,
    pub batched_ragged_layers: u64,
    pub batched_ragged_rows: u64,
    pub total_us: u64,
    pub layers: usize,
    pub sequences: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BatchDecodeExtractTimings {
    pub narrow_us: u64,
    pub contiguous_us: u64,
    pub cache_clear_us: u64,
    pub total_us: u64,
    pub layers: usize,
    pub sequences: usize,
}

impl From<crane_core::models::qwen3::modeling::BatchDecodeSetupStats> for BatchDecodeSetupTimings {
    fn from(value: crane_core::models::qwen3::modeling::BatchDecodeSetupStats) -> Self {
        Self {
            kv_len_scan_us: value.kv_len_scan_us,
            pad_stack_us: value.pad_stack_us,
            contiguous_us: value.contiguous_us,
            extra_room_alloc_us: value.extra_room_alloc_us,
            cache_assign_us: value.cache_assign_us,
            batched_equal_length_layers: value.batched_equal_length_layers,
            batched_ragged_layers: value.batched_ragged_layers,
            batched_ragged_rows: value.batched_ragged_rows,
            total_us: value.total_us,
            layers: value.layers,
            sequences: value.sequences,
        }
    }
}

impl From<crane_core::models::qwen3::modeling::BatchDecodeExtractStats>
    for BatchDecodeExtractTimings
{
    fn from(value: crane_core::models::qwen3::modeling::BatchDecodeExtractStats) -> Self {
        Self {
            narrow_us: value.narrow_us,
            contiguous_us: value.contiguous_us,
            cache_clear_us: value.cache_clear_us,
            total_us: value.total_us,
            layers: value.layers,
            sequences: value.sequences,
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Trait
// ─────────────────────────────────────────────────────────────

/// Core abstraction over different model backends.
///
/// All models must support single-sequence forward passes and KV cache clearing.
/// Optionally, models can support KV cache extraction/restoration (for concurrent
/// sequence serving) and batched decoding (for GPU-efficient parallel generation).
pub trait ModelBackend: Send + 'static {
    /// Run a forward pass for a single sequence.
    ///
    /// * `input_ids` — token IDs to process
    /// * `start_pos` — KV cache position (0 for a fresh sequence)
    ///
    /// Returns logits tensor, typically `[1, seq_len, vocab_size]`.
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor>;

    /// Clear all KV caches.
    fn clear_kv_cache(&mut self);

    /// Number of transformer layers (for KV cache vector sizing).
    fn num_layers(&self) -> usize;

    /// Device the model is running on.
    fn device(&self) -> &Device;

    /// Data type of model weights.
    fn dtype(&self) -> DType;

    /// Reference to the underlying tokenizer.
    fn tokenizer(&self) -> &tokenizers::Tokenizer;

    /// The model's end-of-sequence token ID(s).
    fn eos_token_id(&self) -> Vec<u32>;

    /// Warm up the model with a small forward pass.
    fn warmup(&mut self);

    // ── KV cache swap (for concurrent sequence serving) ───────

    /// Whether this backend supports extracting and restoring KV caches.
    fn supports_kv_swap(&self) -> bool {
        false
    }

    /// Extract per-layer KV caches from the model.
    fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        vec![]
    }

    /// Read raw active KV buffers from the model.
    fn get_kv_cache_buffers(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.get_kv_caches()
    }

    /// Restore per-layer KV caches into the model.
    fn set_kv_caches(&mut self, _caches: Vec<Option<(Tensor, Tensor)>>) {}

    /// Compute bytes held by the model's active KV caches without copying.
    /// Used for memory tracking without the overhead of `get_kv_caches()`.
    fn active_kv_cache_bytes(&self) -> u64 {
        0
    }

    fn kv_cache_layout(&self) -> Option<PagedKvLayout> {
        None
    }

    // ── Batch decode (GPU-efficient concurrent serving) ───────

    /// Whether this backend supports batched decoding.
    fn supports_batch_decode(&self) -> bool {
        false
    }

    /// Pad and load per-sequence KV caches for batched decoding.
    fn setup_batch_decode(
        &mut self,
        _seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        _extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        candle_core::bail!("Batch decode not supported by this backend")
    }

    /// Round 9: adopt a per-layer batched gather output as the KV workspace,
    /// avoiding the per-row pad-stack loop. Default implementation falls back
    /// to per-row `setup_batch_decode` after rebuilding per-row tensors.
    fn setup_batch_decode_batched(
        &mut self,
        _per_layer: &[Option<(Tensor, Tensor)>],
        _kv_lens: &[usize],
        _max_total_len: usize,
        _extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        candle_core::bail!("Batched decode setup not supported by this backend")
    }

    fn last_batch_decode_setup_timings(&self) -> BatchDecodeSetupTimings {
        BatchDecodeSetupTimings::default()
    }

    #[allow(dead_code)]
    fn batch_decode_workspace_generation(&self) -> u64 {
        0
    }

    fn release_batch_decode_workspaces(&mut self) -> usize {
        0
    }

    /// Run one batched decode step.
    fn step_batch_decode(
        &mut self,
        _input_ids: &Tensor,
        _positions: &[usize],
        _attention_mask: Option<&Tensor>,
        _batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        candle_core::bail!("Batch decode not supported by this backend")
    }

    fn step_batch_decode_fixed_width(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        _fixed_cache_width: usize,
    ) -> candle_core::Result<Tensor> {
        self.step_batch_decode(input_ids, positions, attention_mask, None)
    }

    fn step_batch_decode_fixed_width_with_position_ids(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        _position_ids: &Tensor,
        _append_offset: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        fixed_cache_width: usize,
    ) -> candle_core::Result<Tensor> {
        self.step_batch_decode_fixed_width(input_ids, positions, attention_mask, fixed_cache_width)
    }

    fn step_batch_decode_paged_attention(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
        _paged_attention: &crane_core::fused_ops::PagedAttentionDecodeContext,
    ) -> candle_core::Result<Tensor> {
        self.step_batch_decode(input_ids, positions, attention_mask, batch_kv_info)
    }

    fn last_paged_attention_layer_hits(&self) -> usize {
        0
    }

    fn last_paged_attention_layer_fallbacks(&self) -> usize {
        0
    }

    /// Extract per-sequence KV caches from batched state.
    fn extract_batch_kv(
        &mut self,
        _kv_lens: &[usize],
        _original_max_kv: usize,
        _rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        candle_core::bail!("Batch decode not supported by this backend")
    }

    fn extract_batch_kv_selective(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
        keep: &[bool],
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        let extracted = self.extract_batch_kv(kv_lens, original_max_kv, rounds_done)?;
        Ok(extracted
            .into_iter()
            .enumerate()
            .map(|(idx, caches)| {
                if keep.get(idx).copied().unwrap_or(true) {
                    caches
                } else {
                    vec![None; caches.len()]
                }
            })
            .collect())
    }

    fn last_batch_decode_extract_timings(&self) -> BatchDecodeExtractTimings {
        BatchDecodeExtractTimings::default()
    }

    /// Build attention mask for batched decoding.
    fn build_batch_decode_mask(
        &self,
        _kv_lens: &[usize],
        _original_max_kv: usize,
        _max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        candle_core::bail!("Batch decode not supported by this backend")
    }

    fn build_batch_decode_fixed_width_mask(
        &self,
        _kv_lens: &[usize],
        _original_max_kv: usize,
        _total_width: usize,
        _active_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        candle_core::bail!("Batch decode not supported by this backend")
    }
}

// ─────────────────────────────────────────────────────────────
//  Qwen 3 Backend
// ─────────────────────────────────────────────────────────────

pub struct Qwen3Backend {
    pub model: crane_core::models::qwen3::Model,
}

impl Qwen3Backend {
    pub fn new_with_format(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: crane_core::models::qwen3::ModelFormat,
    ) -> Result<Self> {
        let model =
            crane_core::models::qwen3::Model::new_with_format(model_path, device, dtype, format)?;
        Ok(Self { model })
    }
}

impl ModelBackend for Qwen3Backend {
    fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        self.model
            .forward_step(input_ids, start_pos)
            .map_err(Into::into)
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }

    fn num_layers(&self) -> usize {
        self.model.num_layers()
    }

    fn device(&self) -> &Device {
        &self.model.device
    }

    fn dtype(&self) -> DType {
        self.model.dtype
    }

    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.model.tokenizer.tokenizer
    }

    fn eos_token_id(&self) -> Vec<u32> {
        // Qwen3 chat models stop at <|im_end|> (151645).
        // Also include <|endoftext|> (151643) as a fallback.
        let tok = &self.model.tokenizer.tokenizer;
        let mut ids = Vec::new();
        if let Some(id) = tok.token_to_id("<|im_end|>") {
            ids.push(id);
        }
        if let Some(id) = tok.token_to_id("<|endoftext|>") {
            ids.push(id);
        }
        if ids.is_empty() {
            ids.push(151645);
            ids.push(151643);
        }
        ids
    }

    fn warmup(&mut self) {
        self.model.warmup();
    }

    // ── KV swap ──

    fn supports_kv_swap(&self) -> bool {
        true
    }

    fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.model.get_kv_caches()
    }

    fn get_kv_cache_buffers(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.model.get_kv_cache_buffers()
    }

    fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        self.model.set_kv_caches(caches);
    }

    fn active_kv_cache_bytes(&self) -> u64 {
        self.model.active_kv_cache_bytes()
    }

    fn kv_cache_layout(&self) -> Option<PagedKvLayout> {
        let config = self.model.config();
        Some(PagedKvLayout {
            num_layers: config.num_hidden_layers,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim(),
            dtype_size_bytes: self.model.dtype.size_in_bytes(),
        })
    }

    // ── Batch decode ──

    fn supports_batch_decode(&self) -> bool {
        true
    }

    fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.model.setup_batch_decode(seq_kv_caches, extra_room)
    }

    fn setup_batch_decode_batched(
        &mut self,
        per_layer: &[Option<(Tensor, Tensor)>],
        kv_lens: &[usize],
        max_total_len: usize,
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.model
            .setup_batch_decode_batched(per_layer, kv_lens, max_total_len, extra_room)
    }

    fn last_batch_decode_setup_timings(&self) -> BatchDecodeSetupTimings {
        self.model.last_batch_decode_setup_stats().into()
    }

    fn batch_decode_workspace_generation(&self) -> u64 {
        self.model.batch_decode_workspace_generation()
    }

    fn release_batch_decode_workspaces(&mut self) -> usize {
        self.model.release_batch_decode_workspaces()
    }

    fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        self.model.step_batch_decode_with_input_ids(
            input_ids,
            positions,
            attention_mask,
            batch_kv_info,
        )
    }

    fn step_batch_decode_fixed_width(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        fixed_cache_width: usize,
    ) -> candle_core::Result<Tensor> {
        self.model.step_batch_decode_with_input_ids_fixed_width(
            input_ids,
            positions,
            attention_mask,
            fixed_cache_width,
        )
    }

    fn step_batch_decode_fixed_width_with_position_ids(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        position_ids: &Tensor,
        append_offset: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        fixed_cache_width: usize,
    ) -> candle_core::Result<Tensor> {
        self.model
            .step_batch_decode_with_input_ids_fixed_width_and_position_ids(
                input_ids,
                positions,
                position_ids,
                append_offset,
                attention_mask,
                fixed_cache_width,
            )
    }

    fn step_batch_decode_paged_attention(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        _batch_kv_info: Option<(&[usize], usize)>,
        paged_attention: &crane_core::fused_ops::PagedAttentionDecodeContext,
    ) -> candle_core::Result<Tensor> {
        self.model.step_batch_decode_with_input_ids_paged_attention(
            input_ids,
            positions,
            attention_mask,
            paged_attention,
        )
    }

    fn last_paged_attention_layer_hits(&self) -> usize {
        self.model.last_paged_attention_layer_hits()
    }

    fn last_paged_attention_layer_fallbacks(&self) -> usize {
        self.model.last_paged_attention_layer_fallbacks()
    }

    fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        self.model
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)
    }

    fn extract_batch_kv_selective(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
        keep: &[bool],
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        self.model
            .extract_batch_kv_selective(kv_lens, original_max_kv, rounds_done, keep)
    }

    fn last_batch_decode_extract_timings(&self) -> BatchDecodeExtractTimings {
        self.model.last_batch_decode_extract_stats().into()
    }

    fn build_batch_decode_mask(
        &self,
        kv_lens: &[usize],
        original_max_kv: usize,
        max_total_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        crane_core::models::qwen3::modeling::build_batch_decode_mask(
            kv_lens,
            original_max_kv,
            max_total_width,
            self.device(),
            self.dtype(),
        )
    }

    fn build_batch_decode_fixed_width_mask(
        &self,
        kv_lens: &[usize],
        original_max_kv: usize,
        total_width: usize,
        active_width: usize,
    ) -> candle_core::Result<Option<Tensor>> {
        crane_core::models::qwen3::modeling::build_batch_decode_fixed_width_mask(
            kv_lens,
            original_max_kv,
            total_width,
            active_width,
            self.device(),
            self.dtype(),
        )
    }
}
