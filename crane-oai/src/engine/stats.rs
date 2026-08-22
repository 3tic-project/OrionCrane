//! Engine statistics — lock-free counters shared with API handlers.

use std::sync::atomic::{AtomicU64, Ordering};

/// Lock-free engine statistics counters.
pub struct EngineStats {
    pub total_requests: AtomicU64,
    pub completed_requests: AtomicU64,
    pub cancelled_requests: AtomicU64,
    pub failed_requests: AtomicU64,
    pub total_prompt_tokens: AtomicU64,
    pub total_completion_tokens: AtomicU64,
    pub total_prefill_time_us: AtomicU64,
    pub total_decode_steps: AtomicU64,
    pub total_decode_time_us: AtomicU64,
    pub total_kv_swap_count: AtomicU64,
    pub active_sequences: AtomicU64,
    pub waiting_sequences: AtomicU64,
    pub total_queue_wait_time_us: AtomicU64,
    pub total_time_to_first_token_us: AtomicU64,
    pub total_prefill_steps: AtomicU64,
    pub total_prefill_forward_time_us: AtomicU64,
    pub total_prefill_sampling_time_us: AtomicU64,
    pub total_prefill_swap_time_us: AtomicU64,
    pub total_prefix_cache_lookups: AtomicU64,
    pub total_prefix_cache_hits: AtomicU64,
    pub total_prefix_cache_hit_tokens: AtomicU64,
    pub total_prefix_cache_inserts: AtomicU64,
    pub total_prefix_cache_insert_tokens: AtomicU64,
    pub total_prefix_cache_insert_time_us: AtomicU64,
    pub prefix_cache_entries: AtomicU64,
    pub prefix_cache_bytes: AtomicU64,
    pub total_batch_decode_calls: AtomicU64,
    pub total_batch_decode_tokens: AtomicU64,
    pub total_batch_decode_time_us: AtomicU64,
    pub total_batch_decode_setup_time_us: AtomicU64,
    pub total_batch_decode_setup_kv_len_scan_time_us: AtomicU64,
    pub total_batch_decode_setup_pad_stack_time_us: AtomicU64,
    pub total_batch_decode_setup_contiguous_time_us: AtomicU64,
    pub total_batch_decode_setup_extra_room_time_us: AtomicU64,
    pub total_batch_decode_setup_cache_assign_time_us: AtomicU64,
    pub total_batch_decode_mask_time_us: AtomicU64,
    pub total_batch_decode_forward_time_us: AtomicU64,
    pub total_batch_decode_sampling_time_us: AtomicU64,
    pub total_batch_decode_extract_time_us: AtomicU64,
    pub total_batch_decode_extract_narrow_time_us: AtomicU64,
    pub total_batch_decode_extract_contiguous_time_us: AtomicU64,
    pub total_batch_decode_extract_cache_clear_time_us: AtomicU64,
    pub total_batch_decode_extract_state_replace_time_us: AtomicU64,
    pub total_batch_decode_device_token_input_hits: AtomicU64,
    pub total_batch_decode_device_token_input_tokens: AtomicU64,
    pub total_sequential_decode_calls: AtomicU64,
    pub total_sequential_decode_tokens: AtomicU64,
    pub total_sequential_decode_time_us: AtomicU64,
    pub total_sequential_decode_forward_time_us: AtomicU64,
    pub total_sequential_decode_sampling_time_us: AtomicU64,
    pub total_sampling_batch_greedy_calls: AtomicU64,
    pub total_sampling_batch_greedy_tokens: AtomicU64,
    pub total_sampling_batch_greedy_fallbacks: AtomicU64,
    pub total_sampling_batch_greedy_cuda_plain_calls: AtomicU64,
    pub total_sampling_batch_greedy_cuda_plain_tokens: AtomicU64,
    pub total_sampling_batch_greedy_cuda_penalty_calls: AtomicU64,
    pub total_sampling_batch_greedy_cuda_penalty_tokens: AtomicU64,
    pub total_sampling_batch_greedy_tensor_fallback_calls: AtomicU64,
    pub total_sampling_batch_greedy_tensor_fallback_tokens: AtomicU64,
    pub total_sampling_batch_non_greedy_calls: AtomicU64,
    pub total_sampling_batch_non_greedy_tokens: AtomicU64,
    pub total_sampling_batch_non_greedy_cuda_bf16_calls: AtomicU64,
    pub total_sampling_batch_non_greedy_cuda_bf16_tokens: AtomicU64,
    pub total_sampling_batch_non_greedy_fallback_calls: AtomicU64,
    pub total_sampling_batch_non_greedy_fallback_tokens: AtomicU64,
    pub total_sampling_row_greedy_tokens: AtomicU64,
    pub total_sampling_non_greedy_tokens: AtomicU64,
    pub total_sampling_failures: AtomicU64,
    pub total_paged_kv_metadata_syncs: AtomicU64,
    pub total_paged_kv_new_pages: AtomicU64,
    pub total_paged_kv_reused_pages: AtomicU64,
    pub total_paged_kv_released_pages: AtomicU64,
    pub total_paged_kv_compactions: AtomicU64,
    pub total_paged_kv_compacted_pages: AtomicU64,
    pub total_paged_kv_idle_resets: AtomicU64,
    pub total_paged_kv_idle_reset_pages: AtomicU64,
    pub total_paged_kv_pressure_skips: AtomicU64,
    pub total_paged_kv_pressure_released_pages: AtomicU64,
    pub total_paged_kv_gather_extracts: AtomicU64,
    pub total_paged_kv_gather_extract_layers: AtomicU64,
    /// Wall time spent inside `gather_layer_right_aligned` GPU calls
    /// summed across all layers (one K + one V gather per layer per step).
    /// Excludes the per-row narrow+contiguous loop measured separately.
    pub total_paged_kv_gather_kernel_time_us: AtomicU64,
    /// Wall time spent in the per-row `narrow + contiguous` loop after
    /// each layer gather. Each row's contiguous() launches one
    /// copy_strided_src kernel, so this scales as 2 * batch * num_layers.
    /// Always 0 after Round 9 (per-row loop eliminated).
    pub total_paged_kv_gather_per_row_time_us: AtomicU64,
    /// Round 9: number of times `gather_batched_kv_for_batch` re-gathered
    /// because the next batch composition differed from the published cache.
    pub total_paged_kv_gather_regathers: AtomicU64,
    /// Round 9: number of times the engine consumed a previously published
    /// `BatchedKvExtract` directly (steady-state same-batch fast path).
    pub total_paged_kv_batched_setup_hits: AtomicU64,
    /// Round 9: number of times batched setup happened via fresh re-gather
    /// (batch composition changed since the last extract).
    pub total_paged_kv_batched_setup_regather: AtomicU64,
    /// Round 9: wall time inside the model's batched setup.
    pub total_paged_kv_batched_setup_us: AtomicU64,
    pub total_paged_kv_batched_setup_equal_length_layers: AtomicU64,
    pub total_paged_kv_batched_setup_ragged_layers: AtomicU64,
    pub total_paged_kv_batched_setup_ragged_rows: AtomicU64,
    pub total_paged_kv_batched_setup_pending_batch_mismatch: AtomicU64,
    pub total_paged_kv_batched_setup_pending_token_mismatch: AtomicU64,
    pub total_paged_kv_batched_setup_fallback_per_seq_cache: AtomicU64,
    pub total_paged_kv_batched_setup_fallback_regather_unavailable: AtomicU64,
    pub total_paged_kv_batched_setup_fallback_regather_error: AtomicU64,
    pub total_paged_kv_attention_contexts: AtomicU64,
    pub total_paged_kv_attention_decode_calls: AtomicU64,
    pub total_paged_kv_attention_decode_tokens: AtomicU64,
    pub total_paged_kv_attention_layer_hits: AtomicU64,
    pub total_paged_kv_attention_layer_fallbacks: AtomicU64,
    pub total_paged_kv_attention_fallbacks: AtomicU64,
    pub total_cuda_graph_decode_rounds: AtomicU64,
    pub total_cuda_graph_decode_eligible_rounds: AtomicU64,
    pub total_cuda_graph_decode_capture_attempts: AtomicU64,
    pub total_cuda_graph_decode_capture_successes: AtomicU64,
    pub total_cuda_graph_decode_capture_failures: AtomicU64,
    pub total_cuda_graph_decode_replay_calls: AtomicU64,
    pub total_cuda_graph_decode_replay_tokens: AtomicU64,
    pub total_cuda_graph_decode_fallbacks: AtomicU64,
    pub total_cuda_graph_decode_fallback_tokens: AtomicU64,
    pub total_cuda_graph_decode_miss_no_bucket: AtomicU64,
    pub total_cuda_graph_decode_miss_mask: AtomicU64,
    pub total_cuda_graph_decode_miss_paged_attention: AtomicU64,
    pub total_cuda_graph_decode_miss_dynamic_kv: AtomicU64,
    pub total_cuda_graph_decode_miss_device: AtomicU64,
    pub paged_kv_block_size: AtomicU64,
    pub paged_kv_live_pages: AtomicU64,
    pub paged_kv_free_pages: AtomicU64,
    pub paged_kv_live_tokens: AtomicU64,
    pub paged_kv_reserved_tokens: AtomicU64,
    pub paged_kv_fragment_tokens: AtomicU64,
    pub paged_kv_reserved_bytes: AtomicU64,
    pub paged_kv_gpu_capacity_pages: AtomicU64,
    pub paged_kv_gpu_capacity_bytes: AtomicU64,
    pub paged_kv_total_alloc_pages: AtomicU64,
    pub paged_kv_total_reused_pages: AtomicU64,
    pub paged_kv_total_freed_pages: AtomicU64,
    pub tracked_kv_cache_bytes: AtomicU64,
    pub gpu_memory_used_bytes: AtomicU64,
    pub gpu_memory_total_bytes: AtomicU64,
    /// How many times the engine entered the wait-batching path
    /// (drain_with_wait_batch) waiting for additional requests.
    pub total_sched_wait_calls: AtomicU64,
    /// New requests captured during wait-batching windows.
    pub total_sched_wait_arrivals: AtomicU64,
    /// Cumulative time spent in the wait-batching spin loop.
    pub total_sched_wait_time_us: AtomicU64,
    /// Wait-batching calls that exited early because the batch target was met.
    pub total_sched_wait_target_hits: AtomicU64,
}

impl EngineStats {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            completed_requests: AtomicU64::new(0),
            cancelled_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_completion_tokens: AtomicU64::new(0),
            total_prefill_time_us: AtomicU64::new(0),
            total_decode_steps: AtomicU64::new(0),
            total_decode_time_us: AtomicU64::new(0),
            total_kv_swap_count: AtomicU64::new(0),
            active_sequences: AtomicU64::new(0),
            waiting_sequences: AtomicU64::new(0),
            total_queue_wait_time_us: AtomicU64::new(0),
            total_time_to_first_token_us: AtomicU64::new(0),
            total_prefill_steps: AtomicU64::new(0),
            total_prefill_forward_time_us: AtomicU64::new(0),
            total_prefill_sampling_time_us: AtomicU64::new(0),
            total_prefill_swap_time_us: AtomicU64::new(0),
            total_prefix_cache_lookups: AtomicU64::new(0),
            total_prefix_cache_hits: AtomicU64::new(0),
            total_prefix_cache_hit_tokens: AtomicU64::new(0),
            total_prefix_cache_inserts: AtomicU64::new(0),
            total_prefix_cache_insert_tokens: AtomicU64::new(0),
            total_prefix_cache_insert_time_us: AtomicU64::new(0),
            prefix_cache_entries: AtomicU64::new(0),
            prefix_cache_bytes: AtomicU64::new(0),
            total_batch_decode_calls: AtomicU64::new(0),
            total_batch_decode_tokens: AtomicU64::new(0),
            total_batch_decode_time_us: AtomicU64::new(0),
            total_batch_decode_setup_time_us: AtomicU64::new(0),
            total_batch_decode_setup_kv_len_scan_time_us: AtomicU64::new(0),
            total_batch_decode_setup_pad_stack_time_us: AtomicU64::new(0),
            total_batch_decode_setup_contiguous_time_us: AtomicU64::new(0),
            total_batch_decode_setup_extra_room_time_us: AtomicU64::new(0),
            total_batch_decode_setup_cache_assign_time_us: AtomicU64::new(0),
            total_batch_decode_mask_time_us: AtomicU64::new(0),
            total_batch_decode_forward_time_us: AtomicU64::new(0),
            total_batch_decode_sampling_time_us: AtomicU64::new(0),
            total_batch_decode_extract_time_us: AtomicU64::new(0),
            total_batch_decode_extract_narrow_time_us: AtomicU64::new(0),
            total_batch_decode_extract_contiguous_time_us: AtomicU64::new(0),
            total_batch_decode_extract_cache_clear_time_us: AtomicU64::new(0),
            total_batch_decode_extract_state_replace_time_us: AtomicU64::new(0),
            total_batch_decode_device_token_input_hits: AtomicU64::new(0),
            total_batch_decode_device_token_input_tokens: AtomicU64::new(0),
            total_sequential_decode_calls: AtomicU64::new(0),
            total_sequential_decode_tokens: AtomicU64::new(0),
            total_sequential_decode_time_us: AtomicU64::new(0),
            total_sequential_decode_forward_time_us: AtomicU64::new(0),
            total_sequential_decode_sampling_time_us: AtomicU64::new(0),
            total_sampling_batch_greedy_calls: AtomicU64::new(0),
            total_sampling_batch_greedy_tokens: AtomicU64::new(0),
            total_sampling_batch_greedy_fallbacks: AtomicU64::new(0),
            total_sampling_batch_greedy_cuda_plain_calls: AtomicU64::new(0),
            total_sampling_batch_greedy_cuda_plain_tokens: AtomicU64::new(0),
            total_sampling_batch_greedy_cuda_penalty_calls: AtomicU64::new(0),
            total_sampling_batch_greedy_cuda_penalty_tokens: AtomicU64::new(0),
            total_sampling_batch_greedy_tensor_fallback_calls: AtomicU64::new(0),
            total_sampling_batch_greedy_tensor_fallback_tokens: AtomicU64::new(0),
            total_sampling_batch_non_greedy_calls: AtomicU64::new(0),
            total_sampling_batch_non_greedy_tokens: AtomicU64::new(0),
            total_sampling_batch_non_greedy_cuda_bf16_calls: AtomicU64::new(0),
            total_sampling_batch_non_greedy_cuda_bf16_tokens: AtomicU64::new(0),
            total_sampling_batch_non_greedy_fallback_calls: AtomicU64::new(0),
            total_sampling_batch_non_greedy_fallback_tokens: AtomicU64::new(0),
            total_sampling_row_greedy_tokens: AtomicU64::new(0),
            total_sampling_non_greedy_tokens: AtomicU64::new(0),
            total_sampling_failures: AtomicU64::new(0),
            total_paged_kv_metadata_syncs: AtomicU64::new(0),
            total_paged_kv_new_pages: AtomicU64::new(0),
            total_paged_kv_reused_pages: AtomicU64::new(0),
            total_paged_kv_released_pages: AtomicU64::new(0),
            total_paged_kv_compactions: AtomicU64::new(0),
            total_paged_kv_compacted_pages: AtomicU64::new(0),
            total_paged_kv_idle_resets: AtomicU64::new(0),
            total_paged_kv_idle_reset_pages: AtomicU64::new(0),
            total_paged_kv_pressure_skips: AtomicU64::new(0),
            total_paged_kv_pressure_released_pages: AtomicU64::new(0),
            total_paged_kv_gather_extracts: AtomicU64::new(0),
            total_paged_kv_gather_extract_layers: AtomicU64::new(0),
            total_paged_kv_gather_kernel_time_us: AtomicU64::new(0),
            total_paged_kv_gather_per_row_time_us: AtomicU64::new(0),
            total_paged_kv_gather_regathers: AtomicU64::new(0),
            total_paged_kv_batched_setup_hits: AtomicU64::new(0),
            total_paged_kv_batched_setup_regather: AtomicU64::new(0),
            total_paged_kv_batched_setup_us: AtomicU64::new(0),
            total_paged_kv_batched_setup_equal_length_layers: AtomicU64::new(0),
            total_paged_kv_batched_setup_ragged_layers: AtomicU64::new(0),
            total_paged_kv_batched_setup_ragged_rows: AtomicU64::new(0),
            total_paged_kv_batched_setup_pending_batch_mismatch: AtomicU64::new(0),
            total_paged_kv_batched_setup_pending_token_mismatch: AtomicU64::new(0),
            total_paged_kv_batched_setup_fallback_per_seq_cache: AtomicU64::new(0),
            total_paged_kv_batched_setup_fallback_regather_unavailable: AtomicU64::new(0),
            total_paged_kv_batched_setup_fallback_regather_error: AtomicU64::new(0),
            total_paged_kv_attention_contexts: AtomicU64::new(0),
            total_paged_kv_attention_decode_calls: AtomicU64::new(0),
            total_paged_kv_attention_decode_tokens: AtomicU64::new(0),
            total_paged_kv_attention_layer_hits: AtomicU64::new(0),
            total_paged_kv_attention_layer_fallbacks: AtomicU64::new(0),
            total_paged_kv_attention_fallbacks: AtomicU64::new(0),
            total_cuda_graph_decode_rounds: AtomicU64::new(0),
            total_cuda_graph_decode_eligible_rounds: AtomicU64::new(0),
            total_cuda_graph_decode_capture_attempts: AtomicU64::new(0),
            total_cuda_graph_decode_capture_successes: AtomicU64::new(0),
            total_cuda_graph_decode_capture_failures: AtomicU64::new(0),
            total_cuda_graph_decode_replay_calls: AtomicU64::new(0),
            total_cuda_graph_decode_replay_tokens: AtomicU64::new(0),
            total_cuda_graph_decode_fallbacks: AtomicU64::new(0),
            total_cuda_graph_decode_fallback_tokens: AtomicU64::new(0),
            total_cuda_graph_decode_miss_no_bucket: AtomicU64::new(0),
            total_cuda_graph_decode_miss_mask: AtomicU64::new(0),
            total_cuda_graph_decode_miss_paged_attention: AtomicU64::new(0),
            total_cuda_graph_decode_miss_dynamic_kv: AtomicU64::new(0),
            total_cuda_graph_decode_miss_device: AtomicU64::new(0),
            paged_kv_block_size: AtomicU64::new(0),
            paged_kv_live_pages: AtomicU64::new(0),
            paged_kv_free_pages: AtomicU64::new(0),
            paged_kv_live_tokens: AtomicU64::new(0),
            paged_kv_reserved_tokens: AtomicU64::new(0),
            paged_kv_fragment_tokens: AtomicU64::new(0),
            paged_kv_reserved_bytes: AtomicU64::new(0),
            paged_kv_gpu_capacity_pages: AtomicU64::new(0),
            paged_kv_gpu_capacity_bytes: AtomicU64::new(0),
            paged_kv_total_alloc_pages: AtomicU64::new(0),
            paged_kv_total_reused_pages: AtomicU64::new(0),
            paged_kv_total_freed_pages: AtomicU64::new(0),
            tracked_kv_cache_bytes: AtomicU64::new(0),
            total_sched_wait_calls: AtomicU64::new(0),
            total_sched_wait_arrivals: AtomicU64::new(0),
            total_sched_wait_time_us: AtomicU64::new(0),
            total_sched_wait_target_hits: AtomicU64::new(0),
            gpu_memory_used_bytes: AtomicU64::new(0),
            gpu_memory_total_bytes: AtomicU64::new(0),
        }
    }

    /// Snapshot for JSON serialization.
    pub fn snapshot(&self) -> StatsSnapshot {
        fn avg_ms(total_us: u64, count: u64) -> f64 {
            if count > 0 {
                total_us as f64 / count as f64 / 1000.0
            } else {
                0.0
            }
        }
        fn avg_us(total_us: u64, count: u64) -> f64 {
            if count > 0 {
                total_us as f64 / count as f64
            } else {
                0.0
            }
        }

        let total_decode = self.total_decode_steps.load(Ordering::Relaxed);
        let total_decode_us = self.total_decode_time_us.load(Ordering::Relaxed);
        let avg_decode_tok_s = if total_decode_us > 0 {
            (total_decode as f64) / (total_decode_us as f64 / 1_000_000.0)
        } else {
            0.0
        };
        let total_prefill_us = self.total_prefill_time_us.load(Ordering::Relaxed);
        let total_prompt = self.total_prompt_tokens.load(Ordering::Relaxed);
        let avg_prefill_tok_s = if total_prefill_us > 0 {
            (total_prompt as f64) / (total_prefill_us as f64 / 1_000_000.0)
        } else {
            0.0
        };
        let total_prefill_steps = self.total_prefill_steps.load(Ordering::Relaxed);
        let total_queue_wait_us = self.total_queue_wait_time_us.load(Ordering::Relaxed);
        let total_ttft_us = self.total_time_to_first_token_us.load(Ordering::Relaxed);
        let total_prefill_forward_us = self.total_prefill_forward_time_us.load(Ordering::Relaxed);
        let total_prefill_sampling_us = self.total_prefill_sampling_time_us.load(Ordering::Relaxed);
        let total_prefill_swap_us = self.total_prefill_swap_time_us.load(Ordering::Relaxed);
        let total_batch_decode_calls = self.total_batch_decode_calls.load(Ordering::Relaxed);
        let total_batch_decode_tokens = self.total_batch_decode_tokens.load(Ordering::Relaxed);
        let total_batch_decode_us = self.total_batch_decode_time_us.load(Ordering::Relaxed);
        let total_batch_decode_setup_us = self
            .total_batch_decode_setup_time_us
            .load(Ordering::Relaxed);
        let total_batch_decode_setup_kv_len_scan_us = self
            .total_batch_decode_setup_kv_len_scan_time_us
            .load(Ordering::Relaxed);
        let total_batch_decode_setup_pad_stack_us = self
            .total_batch_decode_setup_pad_stack_time_us
            .load(Ordering::Relaxed);
        let total_batch_decode_setup_contiguous_us = self
            .total_batch_decode_setup_contiguous_time_us
            .load(Ordering::Relaxed);
        let total_batch_decode_setup_extra_room_us = self
            .total_batch_decode_setup_extra_room_time_us
            .load(Ordering::Relaxed);
        let total_batch_decode_setup_cache_assign_us = self
            .total_batch_decode_setup_cache_assign_time_us
            .load(Ordering::Relaxed);
        let total_batch_decode_mask_us =
            self.total_batch_decode_mask_time_us.load(Ordering::Relaxed);
        let total_batch_decode_forward_us = self
            .total_batch_decode_forward_time_us
            .load(Ordering::Relaxed);
        let total_batch_decode_sampling_us = self
            .total_batch_decode_sampling_time_us
            .load(Ordering::Relaxed);
        let total_batch_decode_extract_us = self
            .total_batch_decode_extract_time_us
            .load(Ordering::Relaxed);
        let total_batch_decode_extract_narrow_us = self
            .total_batch_decode_extract_narrow_time_us
            .load(Ordering::Relaxed);
        let total_batch_decode_extract_contiguous_us = self
            .total_batch_decode_extract_contiguous_time_us
            .load(Ordering::Relaxed);
        let total_batch_decode_extract_cache_clear_us = self
            .total_batch_decode_extract_cache_clear_time_us
            .load(Ordering::Relaxed);
        let total_batch_decode_extract_state_replace_us = self
            .total_batch_decode_extract_state_replace_time_us
            .load(Ordering::Relaxed);
        let total_batch_decode_device_token_input_hits = self
            .total_batch_decode_device_token_input_hits
            .load(Ordering::Relaxed);
        let total_batch_decode_device_token_input_tokens = self
            .total_batch_decode_device_token_input_tokens
            .load(Ordering::Relaxed);
        let total_sequential_decode_calls =
            self.total_sequential_decode_calls.load(Ordering::Relaxed);
        let total_sequential_decode_tokens =
            self.total_sequential_decode_tokens.load(Ordering::Relaxed);
        let total_sequential_decode_us =
            self.total_sequential_decode_time_us.load(Ordering::Relaxed);
        let total_sequential_decode_forward_us = self
            .total_sequential_decode_forward_time_us
            .load(Ordering::Relaxed);
        let total_sequential_decode_sampling_us = self
            .total_sequential_decode_sampling_time_us
            .load(Ordering::Relaxed);

        StatsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            completed_requests: self.completed_requests.load(Ordering::Relaxed),
            cancelled_requests: self.cancelled_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            total_prompt_tokens: total_prompt,
            total_completion_tokens: self.total_completion_tokens.load(Ordering::Relaxed),
            active_sequences: self.active_sequences.load(Ordering::Relaxed),
            waiting_sequences: self.waiting_sequences.load(Ordering::Relaxed),
            total_kv_swaps: self.total_kv_swap_count.load(Ordering::Relaxed),
            avg_decode_tokens_per_sec: avg_decode_tok_s,
            avg_prefill_tokens_per_sec: avg_prefill_tok_s,
            total_queue_wait_time_us: total_queue_wait_us,
            total_time_to_first_token_us: total_ttft_us,
            total_prefill_steps,
            total_prefill_forward_time_us: total_prefill_forward_us,
            total_prefill_sampling_time_us: total_prefill_sampling_us,
            total_prefill_swap_time_us: total_prefill_swap_us,
            total_prefix_cache_lookups: self.total_prefix_cache_lookups.load(Ordering::Relaxed),
            total_prefix_cache_hits: self.total_prefix_cache_hits.load(Ordering::Relaxed),
            total_prefix_cache_hit_tokens: self
                .total_prefix_cache_hit_tokens
                .load(Ordering::Relaxed),
            total_prefix_cache_inserts: self.total_prefix_cache_inserts.load(Ordering::Relaxed),
            total_prefix_cache_insert_tokens: self
                .total_prefix_cache_insert_tokens
                .load(Ordering::Relaxed),
            total_prefix_cache_insert_time_us: self
                .total_prefix_cache_insert_time_us
                .load(Ordering::Relaxed),
            prefix_cache_entries: self.prefix_cache_entries.load(Ordering::Relaxed),
            prefix_cache_bytes: self.prefix_cache_bytes.load(Ordering::Relaxed),
            avg_queue_wait_ms: avg_ms(total_queue_wait_us, total_prefill_steps),
            avg_time_to_first_token_ms: avg_ms(total_ttft_us, total_prefill_steps),
            avg_prefill_step_ms: avg_ms(total_prefill_us, total_prefill_steps),
            avg_prefill_forward_ms: avg_ms(total_prefill_forward_us, total_prefill_steps),
            avg_prefill_sampling_ms: avg_ms(total_prefill_sampling_us, total_prefill_steps),
            avg_prefill_swap_ms: avg_ms(total_prefill_swap_us, total_prefill_steps),
            total_batch_decode_calls,
            total_batch_decode_tokens,
            total_batch_decode_time_us: total_batch_decode_us,
            total_batch_decode_setup_time_us: total_batch_decode_setup_us,
            total_batch_decode_setup_kv_len_scan_time_us: total_batch_decode_setup_kv_len_scan_us,
            total_batch_decode_setup_pad_stack_time_us: total_batch_decode_setup_pad_stack_us,
            total_batch_decode_setup_contiguous_time_us: total_batch_decode_setup_contiguous_us,
            total_batch_decode_setup_extra_room_time_us: total_batch_decode_setup_extra_room_us,
            total_batch_decode_setup_cache_assign_time_us: total_batch_decode_setup_cache_assign_us,
            total_batch_decode_mask_time_us: total_batch_decode_mask_us,
            total_batch_decode_forward_time_us: total_batch_decode_forward_us,
            total_batch_decode_sampling_time_us: total_batch_decode_sampling_us,
            total_batch_decode_extract_time_us: total_batch_decode_extract_us,
            total_batch_decode_extract_narrow_time_us: total_batch_decode_extract_narrow_us,
            total_batch_decode_extract_contiguous_time_us: total_batch_decode_extract_contiguous_us,
            total_batch_decode_extract_cache_clear_time_us:
                total_batch_decode_extract_cache_clear_us,
            total_batch_decode_extract_state_replace_time_us:
                total_batch_decode_extract_state_replace_us,
            total_batch_decode_device_token_input_hits,
            total_batch_decode_device_token_input_tokens,
            avg_batch_decode_step_ms: avg_ms(total_batch_decode_us, total_batch_decode_calls),
            avg_batch_decode_setup_ms: avg_ms(
                total_batch_decode_setup_us,
                total_batch_decode_calls,
            ),
            avg_batch_decode_setup_kv_len_scan_ms: avg_ms(
                total_batch_decode_setup_kv_len_scan_us,
                total_batch_decode_calls,
            ),
            avg_batch_decode_setup_pad_stack_ms: avg_ms(
                total_batch_decode_setup_pad_stack_us,
                total_batch_decode_calls,
            ),
            avg_batch_decode_setup_contiguous_ms: avg_ms(
                total_batch_decode_setup_contiguous_us,
                total_batch_decode_calls,
            ),
            avg_batch_decode_setup_extra_room_ms: avg_ms(
                total_batch_decode_setup_extra_room_us,
                total_batch_decode_calls,
            ),
            avg_batch_decode_setup_cache_assign_ms: avg_ms(
                total_batch_decode_setup_cache_assign_us,
                total_batch_decode_calls,
            ),
            avg_batch_decode_mask_ms: avg_ms(total_batch_decode_mask_us, total_batch_decode_calls),
            avg_batch_decode_forward_ms: avg_ms(
                total_batch_decode_forward_us,
                total_batch_decode_calls,
            ),
            avg_batch_decode_extract_ms: avg_ms(
                total_batch_decode_extract_us,
                total_batch_decode_calls,
            ),
            avg_batch_decode_extract_narrow_ms: avg_ms(
                total_batch_decode_extract_narrow_us,
                total_batch_decode_calls,
            ),
            avg_batch_decode_extract_contiguous_ms: avg_ms(
                total_batch_decode_extract_contiguous_us,
                total_batch_decode_calls,
            ),
            avg_batch_decode_extract_cache_clear_ms: avg_ms(
                total_batch_decode_extract_cache_clear_us,
                total_batch_decode_calls,
            ),
            avg_batch_decode_extract_state_replace_ms: avg_ms(
                total_batch_decode_extract_state_replace_us,
                total_batch_decode_calls,
            ),
            avg_batch_decode_sampling_us_per_token: avg_us(
                total_batch_decode_sampling_us,
                total_batch_decode_tokens,
            ),
            total_sequential_decode_calls,
            total_sequential_decode_tokens,
            total_sequential_decode_time_us: total_sequential_decode_us,
            total_sequential_decode_forward_time_us: total_sequential_decode_forward_us,
            total_sequential_decode_sampling_time_us: total_sequential_decode_sampling_us,
            total_sampling_batch_greedy_calls: self
                .total_sampling_batch_greedy_calls
                .load(Ordering::Relaxed),
            total_sampling_batch_greedy_tokens: self
                .total_sampling_batch_greedy_tokens
                .load(Ordering::Relaxed),
            total_sampling_batch_greedy_fallbacks: self
                .total_sampling_batch_greedy_fallbacks
                .load(Ordering::Relaxed),
            total_sampling_batch_greedy_cuda_plain_calls: self
                .total_sampling_batch_greedy_cuda_plain_calls
                .load(Ordering::Relaxed),
            total_sampling_batch_greedy_cuda_plain_tokens: self
                .total_sampling_batch_greedy_cuda_plain_tokens
                .load(Ordering::Relaxed),
            total_sampling_batch_greedy_cuda_penalty_calls: self
                .total_sampling_batch_greedy_cuda_penalty_calls
                .load(Ordering::Relaxed),
            total_sampling_batch_greedy_cuda_penalty_tokens: self
                .total_sampling_batch_greedy_cuda_penalty_tokens
                .load(Ordering::Relaxed),
            total_sampling_batch_greedy_tensor_fallback_calls: self
                .total_sampling_batch_greedy_tensor_fallback_calls
                .load(Ordering::Relaxed),
            total_sampling_batch_greedy_tensor_fallback_tokens: self
                .total_sampling_batch_greedy_tensor_fallback_tokens
                .load(Ordering::Relaxed),
            total_sampling_batch_non_greedy_calls: self
                .total_sampling_batch_non_greedy_calls
                .load(Ordering::Relaxed),
            total_sampling_batch_non_greedy_tokens: self
                .total_sampling_batch_non_greedy_tokens
                .load(Ordering::Relaxed),
            total_sampling_batch_non_greedy_cuda_bf16_calls: self
                .total_sampling_batch_non_greedy_cuda_bf16_calls
                .load(Ordering::Relaxed),
            total_sampling_batch_non_greedy_cuda_bf16_tokens: self
                .total_sampling_batch_non_greedy_cuda_bf16_tokens
                .load(Ordering::Relaxed),
            total_sampling_batch_non_greedy_fallback_calls: self
                .total_sampling_batch_non_greedy_fallback_calls
                .load(Ordering::Relaxed),
            total_sampling_batch_non_greedy_fallback_tokens: self
                .total_sampling_batch_non_greedy_fallback_tokens
                .load(Ordering::Relaxed),
            total_sampling_row_greedy_tokens: self
                .total_sampling_row_greedy_tokens
                .load(Ordering::Relaxed),
            total_sampling_non_greedy_tokens: self
                .total_sampling_non_greedy_tokens
                .load(Ordering::Relaxed),
            total_sampling_failures: self.total_sampling_failures.load(Ordering::Relaxed),
            total_paged_kv_metadata_syncs: self
                .total_paged_kv_metadata_syncs
                .load(Ordering::Relaxed),
            total_paged_kv_new_pages: self.total_paged_kv_new_pages.load(Ordering::Relaxed),
            total_paged_kv_reused_pages: self.total_paged_kv_reused_pages.load(Ordering::Relaxed),
            total_paged_kv_released_pages: self
                .total_paged_kv_released_pages
                .load(Ordering::Relaxed),
            total_paged_kv_compactions: self.total_paged_kv_compactions.load(Ordering::Relaxed),
            total_paged_kv_compacted_pages: self
                .total_paged_kv_compacted_pages
                .load(Ordering::Relaxed),
            total_paged_kv_idle_resets: self.total_paged_kv_idle_resets.load(Ordering::Relaxed),
            total_paged_kv_idle_reset_pages: self
                .total_paged_kv_idle_reset_pages
                .load(Ordering::Relaxed),
            total_paged_kv_pressure_skips: self
                .total_paged_kv_pressure_skips
                .load(Ordering::Relaxed),
            total_paged_kv_pressure_released_pages: self
                .total_paged_kv_pressure_released_pages
                .load(Ordering::Relaxed),
            total_paged_kv_gather_extracts: self
                .total_paged_kv_gather_extracts
                .load(Ordering::Relaxed),
            total_paged_kv_gather_extract_layers: self
                .total_paged_kv_gather_extract_layers
                .load(Ordering::Relaxed),
            total_paged_kv_gather_kernel_time_us: self
                .total_paged_kv_gather_kernel_time_us
                .load(Ordering::Relaxed),
            total_paged_kv_gather_per_row_time_us: self
                .total_paged_kv_gather_per_row_time_us
                .load(Ordering::Relaxed),
            total_paged_kv_gather_regathers: self
                .total_paged_kv_gather_regathers
                .load(Ordering::Relaxed),
            total_paged_kv_batched_setup_hits: self
                .total_paged_kv_batched_setup_hits
                .load(Ordering::Relaxed),
            total_paged_kv_batched_setup_regather: self
                .total_paged_kv_batched_setup_regather
                .load(Ordering::Relaxed),
            total_paged_kv_batched_setup_us: self
                .total_paged_kv_batched_setup_us
                .load(Ordering::Relaxed),
            total_paged_kv_batched_setup_equal_length_layers: self
                .total_paged_kv_batched_setup_equal_length_layers
                .load(Ordering::Relaxed),
            total_paged_kv_batched_setup_ragged_layers: self
                .total_paged_kv_batched_setup_ragged_layers
                .load(Ordering::Relaxed),
            total_paged_kv_batched_setup_ragged_rows: self
                .total_paged_kv_batched_setup_ragged_rows
                .load(Ordering::Relaxed),
            total_paged_kv_batched_setup_pending_batch_mismatch: self
                .total_paged_kv_batched_setup_pending_batch_mismatch
                .load(Ordering::Relaxed),
            total_paged_kv_batched_setup_pending_token_mismatch: self
                .total_paged_kv_batched_setup_pending_token_mismatch
                .load(Ordering::Relaxed),
            total_paged_kv_batched_setup_fallback_per_seq_cache: self
                .total_paged_kv_batched_setup_fallback_per_seq_cache
                .load(Ordering::Relaxed),
            total_paged_kv_batched_setup_fallback_regather_unavailable: self
                .total_paged_kv_batched_setup_fallback_regather_unavailable
                .load(Ordering::Relaxed),
            total_paged_kv_batched_setup_fallback_regather_error: self
                .total_paged_kv_batched_setup_fallback_regather_error
                .load(Ordering::Relaxed),
            total_paged_kv_attention_contexts: self
                .total_paged_kv_attention_contexts
                .load(Ordering::Relaxed),
            total_paged_kv_attention_decode_calls: self
                .total_paged_kv_attention_decode_calls
                .load(Ordering::Relaxed),
            total_paged_kv_attention_decode_tokens: self
                .total_paged_kv_attention_decode_tokens
                .load(Ordering::Relaxed),
            total_paged_kv_attention_layer_hits: self
                .total_paged_kv_attention_layer_hits
                .load(Ordering::Relaxed),
            total_paged_kv_attention_layer_fallbacks: self
                .total_paged_kv_attention_layer_fallbacks
                .load(Ordering::Relaxed),
            total_paged_kv_attention_fallbacks: self
                .total_paged_kv_attention_fallbacks
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_rounds: self
                .total_cuda_graph_decode_rounds
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_eligible_rounds: self
                .total_cuda_graph_decode_eligible_rounds
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_capture_attempts: self
                .total_cuda_graph_decode_capture_attempts
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_capture_successes: self
                .total_cuda_graph_decode_capture_successes
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_capture_failures: self
                .total_cuda_graph_decode_capture_failures
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_replay_calls: self
                .total_cuda_graph_decode_replay_calls
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_replay_tokens: self
                .total_cuda_graph_decode_replay_tokens
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_fallbacks: self
                .total_cuda_graph_decode_fallbacks
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_fallback_tokens: self
                .total_cuda_graph_decode_fallback_tokens
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_miss_no_bucket: self
                .total_cuda_graph_decode_miss_no_bucket
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_miss_mask: self
                .total_cuda_graph_decode_miss_mask
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_miss_paged_attention: self
                .total_cuda_graph_decode_miss_paged_attention
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_miss_dynamic_kv: self
                .total_cuda_graph_decode_miss_dynamic_kv
                .load(Ordering::Relaxed),
            total_cuda_graph_decode_miss_device: self
                .total_cuda_graph_decode_miss_device
                .load(Ordering::Relaxed),
            paged_kv_block_size: self.paged_kv_block_size.load(Ordering::Relaxed),
            paged_kv_live_pages: self.paged_kv_live_pages.load(Ordering::Relaxed),
            paged_kv_free_pages: self.paged_kv_free_pages.load(Ordering::Relaxed),
            paged_kv_live_tokens: self.paged_kv_live_tokens.load(Ordering::Relaxed),
            paged_kv_reserved_tokens: self.paged_kv_reserved_tokens.load(Ordering::Relaxed),
            paged_kv_fragment_tokens: self.paged_kv_fragment_tokens.load(Ordering::Relaxed),
            paged_kv_reserved_bytes: self.paged_kv_reserved_bytes.load(Ordering::Relaxed),
            paged_kv_gpu_capacity_pages: self.paged_kv_gpu_capacity_pages.load(Ordering::Relaxed),
            paged_kv_gpu_capacity_bytes: self.paged_kv_gpu_capacity_bytes.load(Ordering::Relaxed),
            paged_kv_total_alloc_pages: self.paged_kv_total_alloc_pages.load(Ordering::Relaxed),
            paged_kv_total_reused_pages: self.paged_kv_total_reused_pages.load(Ordering::Relaxed),
            paged_kv_total_freed_pages: self.paged_kv_total_freed_pages.load(Ordering::Relaxed),
            avg_sequential_decode_step_ms: avg_ms(
                total_sequential_decode_us,
                total_sequential_decode_calls,
            ),
            avg_sequential_decode_forward_ms: avg_ms(
                total_sequential_decode_forward_us,
                total_sequential_decode_calls,
            ),
            avg_sequential_decode_sampling_us_per_token: avg_us(
                total_sequential_decode_sampling_us,
                total_sequential_decode_tokens,
            ),
            tracked_kv_cache_bytes: self.tracked_kv_cache_bytes.load(Ordering::Relaxed),
            gpu_memory_used_bytes: self.gpu_memory_used_bytes.load(Ordering::Relaxed),
            gpu_memory_total_bytes: self.gpu_memory_total_bytes.load(Ordering::Relaxed),
            total_sched_wait_calls: self.total_sched_wait_calls.load(Ordering::Relaxed),
            total_sched_wait_arrivals: self.total_sched_wait_arrivals.load(Ordering::Relaxed),
            total_sched_wait_time_us: self.total_sched_wait_time_us.load(Ordering::Relaxed),
            total_sched_wait_target_hits: self.total_sched_wait_target_hits.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct StatsSnapshot {
    pub total_requests: u64,
    pub completed_requests: u64,
    pub cancelled_requests: u64,
    pub failed_requests: u64,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub active_sequences: u64,
    pub waiting_sequences: u64,
    pub total_kv_swaps: u64,
    pub avg_decode_tokens_per_sec: f64,
    pub avg_prefill_tokens_per_sec: f64,
    pub total_queue_wait_time_us: u64,
    pub total_time_to_first_token_us: u64,
    pub total_prefill_steps: u64,
    pub total_prefill_forward_time_us: u64,
    pub total_prefill_sampling_time_us: u64,
    pub total_prefill_swap_time_us: u64,
    pub total_prefix_cache_lookups: u64,
    pub total_prefix_cache_hits: u64,
    pub total_prefix_cache_hit_tokens: u64,
    pub total_prefix_cache_inserts: u64,
    pub total_prefix_cache_insert_tokens: u64,
    pub total_prefix_cache_insert_time_us: u64,
    pub prefix_cache_entries: u64,
    pub prefix_cache_bytes: u64,
    pub avg_queue_wait_ms: f64,
    pub avg_time_to_first_token_ms: f64,
    pub avg_prefill_step_ms: f64,
    pub avg_prefill_forward_ms: f64,
    pub avg_prefill_sampling_ms: f64,
    pub avg_prefill_swap_ms: f64,
    pub total_batch_decode_calls: u64,
    pub total_batch_decode_tokens: u64,
    pub total_batch_decode_time_us: u64,
    pub total_batch_decode_setup_time_us: u64,
    pub total_batch_decode_setup_kv_len_scan_time_us: u64,
    pub total_batch_decode_setup_pad_stack_time_us: u64,
    pub total_batch_decode_setup_contiguous_time_us: u64,
    pub total_batch_decode_setup_extra_room_time_us: u64,
    pub total_batch_decode_setup_cache_assign_time_us: u64,
    pub total_batch_decode_mask_time_us: u64,
    pub total_batch_decode_forward_time_us: u64,
    pub total_batch_decode_sampling_time_us: u64,
    pub total_batch_decode_extract_time_us: u64,
    pub total_batch_decode_extract_narrow_time_us: u64,
    pub total_batch_decode_extract_contiguous_time_us: u64,
    pub total_batch_decode_extract_cache_clear_time_us: u64,
    pub total_batch_decode_extract_state_replace_time_us: u64,
    pub total_batch_decode_device_token_input_hits: u64,
    pub total_batch_decode_device_token_input_tokens: u64,
    pub avg_batch_decode_step_ms: f64,
    pub avg_batch_decode_setup_ms: f64,
    pub avg_batch_decode_setup_kv_len_scan_ms: f64,
    pub avg_batch_decode_setup_pad_stack_ms: f64,
    pub avg_batch_decode_setup_contiguous_ms: f64,
    pub avg_batch_decode_setup_extra_room_ms: f64,
    pub avg_batch_decode_setup_cache_assign_ms: f64,
    pub avg_batch_decode_mask_ms: f64,
    pub avg_batch_decode_forward_ms: f64,
    pub avg_batch_decode_extract_ms: f64,
    pub avg_batch_decode_extract_narrow_ms: f64,
    pub avg_batch_decode_extract_contiguous_ms: f64,
    pub avg_batch_decode_extract_cache_clear_ms: f64,
    pub avg_batch_decode_extract_state_replace_ms: f64,
    pub avg_batch_decode_sampling_us_per_token: f64,
    pub total_sequential_decode_calls: u64,
    pub total_sequential_decode_tokens: u64,
    pub total_sequential_decode_time_us: u64,
    pub total_sequential_decode_forward_time_us: u64,
    pub total_sequential_decode_sampling_time_us: u64,
    pub total_sampling_batch_greedy_calls: u64,
    pub total_sampling_batch_greedy_tokens: u64,
    pub total_sampling_batch_greedy_fallbacks: u64,
    pub total_sampling_batch_greedy_cuda_plain_calls: u64,
    pub total_sampling_batch_greedy_cuda_plain_tokens: u64,
    pub total_sampling_batch_greedy_cuda_penalty_calls: u64,
    pub total_sampling_batch_greedy_cuda_penalty_tokens: u64,
    pub total_sampling_batch_greedy_tensor_fallback_calls: u64,
    pub total_sampling_batch_greedy_tensor_fallback_tokens: u64,
    pub total_sampling_batch_non_greedy_calls: u64,
    pub total_sampling_batch_non_greedy_tokens: u64,
    pub total_sampling_batch_non_greedy_cuda_bf16_calls: u64,
    pub total_sampling_batch_non_greedy_cuda_bf16_tokens: u64,
    pub total_sampling_batch_non_greedy_fallback_calls: u64,
    pub total_sampling_batch_non_greedy_fallback_tokens: u64,
    pub total_sampling_row_greedy_tokens: u64,
    pub total_sampling_non_greedy_tokens: u64,
    pub total_sampling_failures: u64,
    pub total_paged_kv_metadata_syncs: u64,
    pub total_paged_kv_new_pages: u64,
    pub total_paged_kv_reused_pages: u64,
    pub total_paged_kv_released_pages: u64,
    pub total_paged_kv_compactions: u64,
    pub total_paged_kv_compacted_pages: u64,
    pub total_paged_kv_idle_resets: u64,
    pub total_paged_kv_idle_reset_pages: u64,
    pub total_paged_kv_pressure_skips: u64,
    pub total_paged_kv_pressure_released_pages: u64,
    pub total_paged_kv_gather_extracts: u64,
    pub total_paged_kv_gather_extract_layers: u64,
    pub total_paged_kv_gather_kernel_time_us: u64,
    pub total_paged_kv_gather_per_row_time_us: u64,
    pub total_paged_kv_gather_regathers: u64,
    pub total_paged_kv_batched_setup_hits: u64,
    pub total_paged_kv_batched_setup_regather: u64,
    pub total_paged_kv_batched_setup_us: u64,
    pub total_paged_kv_batched_setup_equal_length_layers: u64,
    pub total_paged_kv_batched_setup_ragged_layers: u64,
    pub total_paged_kv_batched_setup_ragged_rows: u64,
    pub total_paged_kv_batched_setup_pending_batch_mismatch: u64,
    pub total_paged_kv_batched_setup_pending_token_mismatch: u64,
    pub total_paged_kv_batched_setup_fallback_per_seq_cache: u64,
    pub total_paged_kv_batched_setup_fallback_regather_unavailable: u64,
    pub total_paged_kv_batched_setup_fallback_regather_error: u64,
    pub total_paged_kv_attention_contexts: u64,
    pub total_paged_kv_attention_decode_calls: u64,
    pub total_paged_kv_attention_decode_tokens: u64,
    pub total_paged_kv_attention_layer_hits: u64,
    pub total_paged_kv_attention_layer_fallbacks: u64,
    pub total_paged_kv_attention_fallbacks: u64,
    pub total_cuda_graph_decode_rounds: u64,
    pub total_cuda_graph_decode_eligible_rounds: u64,
    pub total_cuda_graph_decode_capture_attempts: u64,
    pub total_cuda_graph_decode_capture_successes: u64,
    pub total_cuda_graph_decode_capture_failures: u64,
    pub total_cuda_graph_decode_replay_calls: u64,
    pub total_cuda_graph_decode_replay_tokens: u64,
    pub total_cuda_graph_decode_fallbacks: u64,
    pub total_cuda_graph_decode_fallback_tokens: u64,
    pub total_cuda_graph_decode_miss_no_bucket: u64,
    pub total_cuda_graph_decode_miss_mask: u64,
    pub total_cuda_graph_decode_miss_paged_attention: u64,
    pub total_cuda_graph_decode_miss_dynamic_kv: u64,
    pub total_cuda_graph_decode_miss_device: u64,
    pub paged_kv_block_size: u64,
    pub paged_kv_live_pages: u64,
    pub paged_kv_free_pages: u64,
    pub paged_kv_live_tokens: u64,
    pub paged_kv_reserved_tokens: u64,
    pub paged_kv_fragment_tokens: u64,
    pub paged_kv_reserved_bytes: u64,
    pub paged_kv_gpu_capacity_pages: u64,
    pub paged_kv_gpu_capacity_bytes: u64,
    pub paged_kv_total_alloc_pages: u64,
    pub paged_kv_total_reused_pages: u64,
    pub paged_kv_total_freed_pages: u64,
    pub avg_sequential_decode_step_ms: f64,
    pub avg_sequential_decode_forward_ms: f64,
    pub avg_sequential_decode_sampling_us_per_token: f64,
    pub tracked_kv_cache_bytes: u64,
    pub gpu_memory_used_bytes: u64,
    pub gpu_memory_total_bytes: u64,
    pub total_sched_wait_calls: u64,
    pub total_sched_wait_arrivals: u64,
    pub total_sched_wait_time_us: u64,
    pub total_sched_wait_target_hits: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn new_stats_are_zero() {
        let s = EngineStats::new();
        assert_eq!(s.total_requests.load(Ordering::Relaxed), 0);
        assert_eq!(s.completed_requests.load(Ordering::Relaxed), 0);
        assert_eq!(s.cancelled_requests.load(Ordering::Relaxed), 0);
        assert_eq!(s.failed_requests.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_prompt_tokens.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_completion_tokens.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_prefill_time_us.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_decode_steps.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_decode_time_us.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_kv_swap_count.load(Ordering::Relaxed), 0);
        assert_eq!(s.active_sequences.load(Ordering::Relaxed), 0);
        assert_eq!(s.waiting_sequences.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_prefill_steps.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_prefix_cache_hits.load(Ordering::Relaxed), 0);
        assert_eq!(s.prefix_cache_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_batch_decode_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            s.total_batch_decode_setup_pad_stack_time_us
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            s.total_batch_decode_extract_contiguous_time_us
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            s.total_batch_decode_device_token_input_hits
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(s.total_sequential_decode_calls.load(Ordering::Relaxed), 0);
        assert_eq!(
            s.total_sampling_batch_greedy_tokens.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            s.total_sampling_batch_greedy_cuda_penalty_tokens
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            s.total_sampling_non_greedy_tokens.load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            s.total_sampling_batch_non_greedy_tokens
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            s.total_sampling_batch_non_greedy_cuda_bf16_tokens
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            s.total_sampling_batch_non_greedy_fallback_tokens
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(s.total_paged_kv_metadata_syncs.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_paged_kv_compactions.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_paged_kv_idle_resets.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_paged_kv_pressure_skips.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_paged_kv_gather_extracts.load(Ordering::Relaxed), 0);
        assert_eq!(
            s.total_paged_kv_batched_setup_ragged_layers
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            s.total_paged_kv_attention_decode_calls
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(s.total_cuda_graph_decode_rounds.load(Ordering::Relaxed), 0);
        assert_eq!(
            s.total_cuda_graph_decode_miss_dynamic_kv
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(s.paged_kv_live_pages.load(Ordering::Relaxed), 0);
        assert_eq!(s.paged_kv_reserved_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(s.paged_kv_gpu_capacity_bytes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn snapshot_copies_all_counters() {
        let s = EngineStats::new();
        s.total_requests.store(10, Ordering::Relaxed);
        s.completed_requests.store(7, Ordering::Relaxed);
        s.cancelled_requests.store(2, Ordering::Relaxed);
        s.failed_requests.store(1, Ordering::Relaxed);
        s.total_prompt_tokens.store(500, Ordering::Relaxed);
        s.total_completion_tokens.store(1000, Ordering::Relaxed);
        s.total_kv_swap_count.store(3, Ordering::Relaxed);
        s.active_sequences.store(4, Ordering::Relaxed);
        s.waiting_sequences.store(2, Ordering::Relaxed);
        s.total_prefill_steps.store(5, Ordering::Relaxed);
        s.total_prefix_cache_lookups.store(9, Ordering::Relaxed);
        s.total_prefix_cache_hits.store(7, Ordering::Relaxed);
        s.total_prefix_cache_hit_tokens
            .store(1792, Ordering::Relaxed);
        s.prefix_cache_entries.store(2, Ordering::Relaxed);
        s.prefix_cache_bytes.store(4096, Ordering::Relaxed);
        s.total_queue_wait_time_us.store(50_000, Ordering::Relaxed);
        s.total_time_to_first_token_us
            .store(150_000, Ordering::Relaxed);
        s.total_batch_decode_calls.store(4, Ordering::Relaxed);
        s.total_batch_decode_tokens.store(32, Ordering::Relaxed);
        s.total_batch_decode_setup_pad_stack_time_us
            .store(40_000, Ordering::Relaxed);
        s.total_batch_decode_setup_contiguous_time_us
            .store(20_000, Ordering::Relaxed);
        s.total_batch_decode_extract_narrow_time_us
            .store(8_000, Ordering::Relaxed);
        s.total_batch_decode_extract_contiguous_time_us
            .store(12_000, Ordering::Relaxed);
        s.total_batch_decode_extract_state_replace_time_us
            .store(4_000, Ordering::Relaxed);
        s.total_batch_decode_device_token_input_hits
            .store(11, Ordering::Relaxed);
        s.total_batch_decode_device_token_input_tokens
            .store(88, Ordering::Relaxed);
        s.total_sequential_decode_calls.store(2, Ordering::Relaxed);
        s.total_sequential_decode_tokens.store(8, Ordering::Relaxed);
        s.total_sampling_batch_greedy_calls
            .store(6, Ordering::Relaxed);
        s.total_sampling_batch_greedy_tokens
            .store(48, Ordering::Relaxed);
        s.total_sampling_batch_greedy_cuda_plain_calls
            .store(2, Ordering::Relaxed);
        s.total_sampling_batch_greedy_cuda_plain_tokens
            .store(8, Ordering::Relaxed);
        s.total_sampling_batch_greedy_cuda_penalty_calls
            .store(3, Ordering::Relaxed);
        s.total_sampling_batch_greedy_cuda_penalty_tokens
            .store(24, Ordering::Relaxed);
        s.total_sampling_batch_greedy_tensor_fallback_calls
            .store(1, Ordering::Relaxed);
        s.total_sampling_batch_greedy_tensor_fallback_tokens
            .store(16, Ordering::Relaxed);
        s.total_sampling_batch_non_greedy_calls
            .store(7, Ordering::Relaxed);
        s.total_sampling_batch_non_greedy_tokens
            .store(56, Ordering::Relaxed);
        s.total_sampling_row_greedy_tokens
            .store(3, Ordering::Relaxed);
        s.total_sampling_non_greedy_tokens
            .store(5, Ordering::Relaxed);
        s.total_paged_kv_metadata_syncs.store(11, Ordering::Relaxed);
        s.total_paged_kv_new_pages.store(12, Ordering::Relaxed);
        s.total_paged_kv_reused_pages.store(13, Ordering::Relaxed);
        s.total_paged_kv_released_pages.store(14, Ordering::Relaxed);
        s.total_paged_kv_compactions.store(15, Ordering::Relaxed);
        s.total_paged_kv_compacted_pages
            .store(16, Ordering::Relaxed);
        s.total_paged_kv_idle_resets.store(17, Ordering::Relaxed);
        s.total_paged_kv_idle_reset_pages
            .store(18, Ordering::Relaxed);
        s.total_paged_kv_pressure_skips.store(19, Ordering::Relaxed);
        s.total_paged_kv_pressure_released_pages
            .store(20, Ordering::Relaxed);
        s.total_paged_kv_gather_extracts
            .store(21, Ordering::Relaxed);
        s.total_paged_kv_gather_extract_layers
            .store(22, Ordering::Relaxed);
        s.total_paged_kv_batched_setup_hits
            .store(23, Ordering::Relaxed);
        s.total_paged_kv_batched_setup_regather
            .store(24, Ordering::Relaxed);
        s.total_paged_kv_batched_setup_us
            .store(25, Ordering::Relaxed);
        s.total_paged_kv_batched_setup_equal_length_layers
            .store(26, Ordering::Relaxed);
        s.total_paged_kv_batched_setup_ragged_layers
            .store(27, Ordering::Relaxed);
        s.total_paged_kv_batched_setup_ragged_rows
            .store(28, Ordering::Relaxed);
        s.total_paged_kv_batched_setup_pending_batch_mismatch
            .store(29, Ordering::Relaxed);
        s.total_paged_kv_batched_setup_pending_token_mismatch
            .store(30, Ordering::Relaxed);
        s.total_paged_kv_batched_setup_fallback_per_seq_cache
            .store(31, Ordering::Relaxed);
        s.total_paged_kv_batched_setup_fallback_regather_unavailable
            .store(32, Ordering::Relaxed);
        s.total_paged_kv_batched_setup_fallback_regather_error
            .store(33, Ordering::Relaxed);
        s.total_paged_kv_attention_contexts
            .store(34, Ordering::Relaxed);
        s.total_paged_kv_attention_decode_calls
            .store(35, Ordering::Relaxed);
        s.total_paged_kv_attention_decode_tokens
            .store(36, Ordering::Relaxed);
        s.total_paged_kv_attention_layer_hits
            .store(37, Ordering::Relaxed);
        s.total_paged_kv_attention_layer_fallbacks
            .store(38, Ordering::Relaxed);
        s.total_paged_kv_attention_fallbacks
            .store(39, Ordering::Relaxed);
        s.total_cuda_graph_decode_rounds
            .store(34, Ordering::Relaxed);
        s.total_cuda_graph_decode_eligible_rounds
            .store(35, Ordering::Relaxed);
        s.total_cuda_graph_decode_capture_attempts
            .store(36, Ordering::Relaxed);
        s.total_cuda_graph_decode_capture_successes
            .store(37, Ordering::Relaxed);
        s.total_cuda_graph_decode_capture_failures
            .store(38, Ordering::Relaxed);
        s.total_cuda_graph_decode_replay_calls
            .store(39, Ordering::Relaxed);
        s.total_cuda_graph_decode_replay_tokens
            .store(40, Ordering::Relaxed);
        s.total_cuda_graph_decode_fallbacks
            .store(41, Ordering::Relaxed);
        s.total_cuda_graph_decode_fallback_tokens
            .store(42, Ordering::Relaxed);
        s.total_cuda_graph_decode_miss_no_bucket
            .store(43, Ordering::Relaxed);
        s.total_cuda_graph_decode_miss_mask
            .store(44, Ordering::Relaxed);
        s.total_cuda_graph_decode_miss_paged_attention
            .store(45, Ordering::Relaxed);
        s.total_cuda_graph_decode_miss_dynamic_kv
            .store(46, Ordering::Relaxed);
        s.total_cuda_graph_decode_miss_device
            .store(47, Ordering::Relaxed);
        s.paged_kv_block_size.store(16, Ordering::Relaxed);
        s.paged_kv_live_pages.store(17, Ordering::Relaxed);
        s.paged_kv_free_pages.store(18, Ordering::Relaxed);
        s.paged_kv_live_tokens.store(19, Ordering::Relaxed);
        s.paged_kv_reserved_tokens.store(20, Ordering::Relaxed);
        s.paged_kv_fragment_tokens.store(21, Ordering::Relaxed);
        s.paged_kv_reserved_bytes.store(22, Ordering::Relaxed);
        s.paged_kv_gpu_capacity_pages.store(26, Ordering::Relaxed);
        s.paged_kv_gpu_capacity_bytes.store(27, Ordering::Relaxed);
        s.paged_kv_total_alloc_pages.store(23, Ordering::Relaxed);
        s.paged_kv_total_reused_pages.store(24, Ordering::Relaxed);
        s.paged_kv_total_freed_pages.store(25, Ordering::Relaxed);
        s.tracked_kv_cache_bytes.store(1024, Ordering::Relaxed);

        let snap = s.snapshot();
        assert_eq!(snap.total_requests, 10);
        assert_eq!(snap.completed_requests, 7);
        assert_eq!(snap.cancelled_requests, 2);
        assert_eq!(snap.failed_requests, 1);
        assert_eq!(snap.total_prompt_tokens, 500);
        assert_eq!(snap.total_completion_tokens, 1000);
        assert_eq!(snap.total_kv_swaps, 3);
        assert_eq!(snap.active_sequences, 4);
        assert_eq!(snap.waiting_sequences, 2);
        assert_eq!(snap.total_prefill_steps, 5);
        assert_eq!(snap.total_batch_decode_calls, 4);
        assert_eq!(snap.total_batch_decode_tokens, 32);
        assert_eq!(snap.total_batch_decode_setup_pad_stack_time_us, 40_000);
        assert_eq!(snap.total_batch_decode_setup_contiguous_time_us, 20_000);
        assert_eq!(snap.total_batch_decode_extract_narrow_time_us, 8_000);
        assert_eq!(snap.total_batch_decode_extract_contiguous_time_us, 12_000);
        assert_eq!(snap.total_batch_decode_extract_state_replace_time_us, 4_000);
        assert_eq!(snap.total_batch_decode_device_token_input_hits, 11);
        assert_eq!(snap.total_batch_decode_device_token_input_tokens, 88);
        assert_eq!(snap.total_sequential_decode_calls, 2);
        assert_eq!(snap.total_sequential_decode_tokens, 8);
        assert_eq!(snap.total_sampling_batch_greedy_calls, 6);
        assert_eq!(snap.total_sampling_batch_greedy_tokens, 48);
        assert_eq!(snap.total_sampling_batch_greedy_cuda_plain_calls, 2);
        assert_eq!(snap.total_sampling_batch_greedy_cuda_plain_tokens, 8);
        assert_eq!(snap.total_sampling_batch_greedy_cuda_penalty_calls, 3);
        assert_eq!(snap.total_sampling_batch_greedy_cuda_penalty_tokens, 24);
        assert_eq!(snap.total_sampling_batch_greedy_tensor_fallback_calls, 1);
        assert_eq!(snap.total_sampling_batch_greedy_tensor_fallback_tokens, 16);
        assert_eq!(snap.total_sampling_batch_non_greedy_calls, 7);
        assert_eq!(snap.total_sampling_batch_non_greedy_tokens, 56);
        assert_eq!(snap.total_sampling_row_greedy_tokens, 3);
        assert_eq!(snap.total_sampling_non_greedy_tokens, 5);
        assert_eq!(snap.total_paged_kv_metadata_syncs, 11);
        assert_eq!(snap.total_paged_kv_new_pages, 12);
        assert_eq!(snap.total_paged_kv_reused_pages, 13);
        assert_eq!(snap.total_paged_kv_released_pages, 14);
        assert_eq!(snap.total_paged_kv_compactions, 15);
        assert_eq!(snap.total_paged_kv_compacted_pages, 16);
        assert_eq!(snap.total_paged_kv_idle_resets, 17);
        assert_eq!(snap.total_paged_kv_idle_reset_pages, 18);
        assert_eq!(snap.total_paged_kv_pressure_skips, 19);
        assert_eq!(snap.total_paged_kv_pressure_released_pages, 20);
        assert_eq!(snap.total_paged_kv_gather_extracts, 21);
        assert_eq!(snap.total_paged_kv_gather_extract_layers, 22);
        assert_eq!(snap.total_paged_kv_batched_setup_hits, 23);
        assert_eq!(snap.total_paged_kv_batched_setup_regather, 24);
        assert_eq!(snap.total_paged_kv_batched_setup_us, 25);
        assert_eq!(snap.total_paged_kv_batched_setup_equal_length_layers, 26);
        assert_eq!(snap.total_paged_kv_batched_setup_ragged_layers, 27);
        assert_eq!(snap.total_paged_kv_batched_setup_ragged_rows, 28);
        assert_eq!(snap.total_paged_kv_batched_setup_pending_batch_mismatch, 29);
        assert_eq!(snap.total_paged_kv_batched_setup_pending_token_mismatch, 30);
        assert_eq!(snap.total_paged_kv_batched_setup_fallback_per_seq_cache, 31);
        assert_eq!(
            snap.total_paged_kv_batched_setup_fallback_regather_unavailable,
            32
        );
        assert_eq!(
            snap.total_paged_kv_batched_setup_fallback_regather_error,
            33
        );
        assert_eq!(snap.total_paged_kv_attention_contexts, 34);
        assert_eq!(snap.total_paged_kv_attention_decode_calls, 35);
        assert_eq!(snap.total_paged_kv_attention_decode_tokens, 36);
        assert_eq!(snap.total_paged_kv_attention_layer_hits, 37);
        assert_eq!(snap.total_paged_kv_attention_layer_fallbacks, 38);
        assert_eq!(snap.total_paged_kv_attention_fallbacks, 39);
        assert_eq!(snap.total_cuda_graph_decode_rounds, 34);
        assert_eq!(snap.total_cuda_graph_decode_eligible_rounds, 35);
        assert_eq!(snap.total_cuda_graph_decode_capture_attempts, 36);
        assert_eq!(snap.total_cuda_graph_decode_capture_successes, 37);
        assert_eq!(snap.total_cuda_graph_decode_capture_failures, 38);
        assert_eq!(snap.total_cuda_graph_decode_replay_calls, 39);
        assert_eq!(snap.total_cuda_graph_decode_replay_tokens, 40);
        assert_eq!(snap.total_cuda_graph_decode_fallbacks, 41);
        assert_eq!(snap.total_cuda_graph_decode_fallback_tokens, 42);
        assert_eq!(snap.total_cuda_graph_decode_miss_no_bucket, 43);
        assert_eq!(snap.total_cuda_graph_decode_miss_mask, 44);
        assert_eq!(snap.total_cuda_graph_decode_miss_paged_attention, 45);
        assert_eq!(snap.total_cuda_graph_decode_miss_dynamic_kv, 46);
        assert_eq!(snap.total_cuda_graph_decode_miss_device, 47);
        assert_eq!(snap.paged_kv_block_size, 16);
        assert_eq!(snap.paged_kv_live_pages, 17);
        assert_eq!(snap.paged_kv_free_pages, 18);
        assert_eq!(snap.paged_kv_live_tokens, 19);
        assert_eq!(snap.paged_kv_reserved_tokens, 20);
        assert_eq!(snap.paged_kv_fragment_tokens, 21);
        assert_eq!(snap.paged_kv_reserved_bytes, 22);
        assert_eq!(snap.paged_kv_gpu_capacity_pages, 26);
        assert_eq!(snap.paged_kv_gpu_capacity_bytes, 27);
        assert_eq!(snap.paged_kv_total_alloc_pages, 23);
        assert_eq!(snap.paged_kv_total_reused_pages, 24);
        assert_eq!(snap.paged_kv_total_freed_pages, 25);
        assert_eq!(snap.tracked_kv_cache_bytes, 1024);
        assert_eq!(snap.total_prefix_cache_lookups, 9);
        assert_eq!(snap.total_prefix_cache_hits, 7);
        assert_eq!(snap.total_prefix_cache_hit_tokens, 1792);
        assert_eq!(snap.prefix_cache_entries, 2);
        assert_eq!(snap.prefix_cache_bytes, 4096);
        assert!((snap.avg_queue_wait_ms - 10.0).abs() < 0.01);
        assert!((snap.avg_time_to_first_token_ms - 30.0).abs() < 0.01);
        assert!((snap.avg_batch_decode_setup_pad_stack_ms - 10.0).abs() < 0.01);
        assert!((snap.avg_batch_decode_extract_contiguous_ms - 3.0).abs() < 0.01);
        assert!((snap.avg_batch_decode_extract_state_replace_ms - 1.0).abs() < 0.01);
    }

    #[test]
    fn snapshot_decode_rate_calculation() {
        let s = EngineStats::new();
        // 100 decode steps in 1 second (1_000_000 μs)
        s.total_decode_steps.store(100, Ordering::Relaxed);
        s.total_decode_time_us.store(1_000_000, Ordering::Relaxed);

        let snap = s.snapshot();
        assert!((snap.avg_decode_tokens_per_sec - 100.0).abs() < 0.01);
    }

    #[test]
    fn snapshot_prefill_rate_calculation() {
        let s = EngineStats::new();
        // 500 prompt tokens prefilled in 0.5 seconds (500_000 μs)
        s.total_prompt_tokens.store(500, Ordering::Relaxed);
        s.total_prefill_time_us.store(500_000, Ordering::Relaxed);

        let snap = s.snapshot();
        assert!((snap.avg_prefill_tokens_per_sec - 1000.0).abs() < 0.01);
    }

    #[test]
    fn snapshot_zero_time_gives_zero_rate() {
        let s = EngineStats::new();
        s.total_decode_steps.store(50, Ordering::Relaxed);
        // time stays 0
        let snap = s.snapshot();
        assert_eq!(snap.avg_decode_tokens_per_sec, 0.0);
        assert_eq!(snap.avg_prefill_tokens_per_sec, 0.0);
    }

    #[test]
    fn snapshot_serializes_to_json() {
        let s = EngineStats::new();
        s.total_requests.store(5, Ordering::Relaxed);
        let snap = s.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"total_requests\":5"));
        assert!(json.contains("avg_decode_tokens_per_sec"));
        assert!(json.contains("avg_prefill_tokens_per_sec"));
        assert!(json.contains("avg_time_to_first_token_ms"));
        assert!(json.contains("total_batch_decode_tokens"));
        assert!(json.contains("avg_batch_decode_setup_pad_stack_ms"));
        assert!(json.contains("avg_batch_decode_extract_state_replace_ms"));
        assert!(json.contains("total_sampling_batch_greedy_tokens"));
        assert!(json.contains("total_sampling_batch_greedy_cuda_penalty_tokens"));
        assert!(json.contains("total_paged_kv_metadata_syncs"));
        assert!(json.contains("total_paged_kv_compactions"));
        assert!(json.contains("total_paged_kv_idle_resets"));
        assert!(json.contains("total_paged_kv_pressure_skips"));
        assert!(json.contains("total_paged_kv_gather_extracts"));
        assert!(json.contains("total_cuda_graph_decode_rounds"));
        assert!(json.contains("total_cuda_graph_decode_miss_dynamic_kv"));
        assert!(json.contains("paged_kv_reserved_bytes"));
        assert!(json.contains("paged_kv_gpu_capacity_bytes"));
        assert!(json.contains("tracked_kv_cache_bytes"));
    }

    #[test]
    fn atomic_fetch_add_works() {
        let s = EngineStats::new();
        s.total_requests.fetch_add(1, Ordering::Relaxed);
        s.total_requests.fetch_add(1, Ordering::Relaxed);
        s.total_requests.fetch_add(1, Ordering::Relaxed);
        assert_eq!(s.snapshot().total_requests, 3);
    }
}
