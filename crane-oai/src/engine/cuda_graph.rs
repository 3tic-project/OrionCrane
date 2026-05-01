//! CUDA Graph decode bucket planning and safe instrumentation.
//!
//! The Qwen3 fixed-width decode path can now opt into full CUDA Graph
//! capture/replay with stable input metadata, device-side append offsets, and a
//! persistent batch-decode KV workspace. Capture remains gated by
//! `CRANE_CUDA_GRAPH_DECODE_CAPTURE=1`; this module owns bucket parsing,
//! per-round dispatch decisions, replay cache keys, and miss counters.

use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

#[cfg(feature = "cuda")]
use candle_core::Tensor;
use candle_core::{DType, Device};

use super::{env_flag, env_flag_default, is_cuda_device, stats::EngineStats, InferenceEngine};

const DEFAULT_BUCKETS: &[usize] = &[1, 2, 4, 8, 16];

#[derive(Debug, Clone)]
pub(super) struct CudaGraphDecodePlanner {
    enabled: bool,
    fixed_width_decode: bool,
    capture_runtime: bool,
    /// Capture the greedy sampling argmax kernel inside the decode graph
    /// (P4-A). Eliminates one out-of-graph `cuLaunchKernel` per decode step
    /// and fuses the argmax into the same `cuGraphLaunch`. Read back via a
    /// single DtoH after replay. Defaults ON when capture_runtime is on.
    capture_sampling: bool,
    /// Maximum replays per captured graph entry before forced re-capture.
    /// 0 = unlimited (legacy behaviour). Setting this to a small value (e.g. 4)
    /// is a workaround for the stale-pointer drift bug; see qwen3_round5.
    max_replays: u32,
    buckets: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CudaGraphDecodeDecision {
    Disabled,
    Miss(CudaGraphDecodeMiss),
    BlockedDynamicKv { bucket: usize },
    Ready { bucket: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CudaGraphDecodeMiss {
    UnsupportedDevice,
    NoBucket,
    AttentionMask,
    PagedAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg(feature = "cuda")]
pub(super) struct CudaGraphDecodeKey {
    pub bucket: usize,
    pub fixed_cache_width: usize,
    pub has_mask: bool,
}

#[cfg(feature = "cuda")]
pub(super) struct CudaGraphDecodeEntry {
    pub graph: crane_core::fused_ops::CudaGraphExec,
    pub logits: Tensor,
    /// How many tokens the captured argmax wrote into
    /// `engine.cuda_graph_sampling_buffers.output_tokens`.
    /// `None` means sampling was NOT captured (greedy capture path disabled,
    /// non-greedy round, or kernel launch failed inside capture); the caller
    /// must run the eager sampling path on `logits` after replay.
    pub captured_sample_batch: Option<usize>,
    /// How many times this captured graph has been replayed since capture.
    /// Used by `CRANE_CUDA_GRAPH_DECODE_MAX_REPLAYS` to cap reuse — a workaround
    /// for the stale-pointer drift bug documented in qwen3_round5.
    pub replays_used: u32,
}

impl CudaGraphDecodePlanner {
    pub(super) fn from_env() -> Self {
        // `CRANE_CUDA_GRAPH_DECODE` defaults OFF.
        //
        // Round 5 (2026-05-08) bisection — see
        // docs/qwen3/benchmarks/qwen3_round5_cuda_graph_2026_05_08.md:
        //
        // * Per-round capture (`CRANE_CUDA_GRAPH_DECODE_NO_REUSE=1`):
        //   tokens generated == eager (3815 vs 3802), but capture overhead
        //   dwarfs the launch-overhead win → +48 % wall time.
        // * Capture + reuse (`CRANE_CUDA_GRAPH_DECODE_CAPTURE=1`): forward
        //   3.5× faster (296 → 84 µs/token) but tokens generated +77 %
        //   (3802 → 6734) because the captured graph reads stale device
        //   memory on replay → strict-JSON outputs corrupt → +30 % wall time.
        //
        // Root cause is not in our code: candle/cudarc's stream allocator
        // frees the per-call intermediate tensors after capture, then hands
        // those slots to subsequent ops. The captured graph nodes have those
        // device pointers baked in, so replay reads whoever wrote there last.
        // The fix needs `cudaGraphAddMemAllocNode` (not exposed by cudarc) or
        // a graph-aware allocator, both of which require modifying the
        // candle/cudarc dependency chain.
        //
        // Until then, capture stays opt-in with a hard startup warning. The
        // fixed-width decode path itself is correct and can be enabled
        // independently as a shape stabilizer (planner-only mode), but it
        // does not measurably beat eager batch_decode at the concurrencies
        // we run, so it also defaults OFF.
        let enabled = env_flag("CRANE_CUDA_GRAPH_DECODE");
        let buckets = parse_buckets(
            std::env::var("CRANE_CUDA_GRAPH_DECODE_BUCKETS")
                .ok()
                .as_deref(),
        );
        let fixed_width_decode =
            enabled && env_flag_default("CRANE_CUDA_GRAPH_FIXED_WIDTH_DECODE", true);
        let capture_runtime = enabled && env_flag_default("CRANE_CUDA_GRAPH_DECODE_CAPTURE", false);
        let capture_sampling =
            capture_runtime && env_flag_default("CRANE_CUDA_GRAPH_DECODE_CAPTURE_SAMPLING", false);
        let max_replays = std::env::var("CRANE_CUDA_GRAPH_DECODE_MAX_REPLAYS")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(0);
        Self {
            enabled,
            fixed_width_decode,
            capture_runtime,
            capture_sampling,
            max_replays,
            buckets,
        }
    }

    pub(super) fn max_replays(&self) -> u32 {
        self.max_replays
    }

    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(super) fn bucket_csv(&self) -> String {
        self.buckets
            .iter()
            .map(|bucket| bucket.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    pub(super) fn fixed_width_decode(&self) -> bool {
        self.fixed_width_decode
    }

    pub(super) fn capture_runtime(&self) -> bool {
        self.capture_runtime
    }

    pub(super) fn capture_sampling(&self) -> bool {
        self.capture_sampling
    }

    pub(super) fn classify_round(
        &self,
        batch_size: usize,
        device: &Device,
        dtype: DType,
        has_attention_mask: bool,
        fixed_width_mask: bool,
        has_paged_attention: bool,
        graph_safe_kv_append: bool,
    ) -> CudaGraphDecodeDecision {
        if !self.enabled {
            return CudaGraphDecodeDecision::Disabled;
        }
        if !is_cuda_device(device) || dtype != DType::BF16 {
            return CudaGraphDecodeDecision::Miss(CudaGraphDecodeMiss::UnsupportedDevice);
        }
        let Some(bucket) = self.bucket_for(batch_size) else {
            return CudaGraphDecodeDecision::Miss(CudaGraphDecodeMiss::NoBucket);
        };
        if has_paged_attention {
            return CudaGraphDecodeDecision::Miss(CudaGraphDecodeMiss::PagedAttention);
        }
        if mask_blocks_capture(has_attention_mask, fixed_width_mask) {
            return CudaGraphDecodeDecision::Miss(CudaGraphDecodeMiss::AttentionMask);
        }

        if graph_safe_kv_append {
            return CudaGraphDecodeDecision::Ready { bucket };
        }

        CudaGraphDecodeDecision::BlockedDynamicKv { bucket }
    }

    fn bucket_for(&self, batch_size: usize) -> Option<usize> {
        self.buckets
            .iter()
            .copied()
            .find(|&bucket| bucket == batch_size)
    }
}

impl InferenceEngine {
    pub(super) fn record_cuda_graph_decode_decision(
        &self,
        decision: CudaGraphDecodeDecision,
        active_rows: u64,
    ) {
        record_decision(&self.stats, decision, active_rows);
    }

    #[cfg(feature = "cuda")]
    pub(super) fn record_cuda_graph_decode_capture_attempt(&self) {
        self.stats
            .total_cuda_graph_decode_capture_attempts
            .fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "cuda")]
    pub(super) fn record_cuda_graph_decode_capture_success(&self) {
        self.stats
            .total_cuda_graph_decode_capture_successes
            .fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "cuda")]
    pub(super) fn record_cuda_graph_decode_capture_failure(&self) {
        self.stats
            .total_cuda_graph_decode_capture_failures
            .fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "cuda")]
    pub(super) fn record_cuda_graph_decode_replay(&self, active_rows: u64) {
        self.stats
            .total_cuda_graph_decode_replay_calls
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_cuda_graph_decode_replay_tokens
            .fetch_add(active_rows, Ordering::Relaxed);
    }

    pub(super) fn record_cuda_graph_decode_fallback(&self, active_rows: u64) {
        self.stats
            .total_cuda_graph_decode_fallbacks
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_cuda_graph_decode_fallback_tokens
            .fetch_add(active_rows, Ordering::Relaxed);
    }
}

fn record_decision(stats: &EngineStats, decision: CudaGraphDecodeDecision, active_rows: u64) {
    match decision {
        CudaGraphDecodeDecision::Disabled => {}
        CudaGraphDecodeDecision::BlockedDynamicKv { .. } => {
            stats
                .total_cuda_graph_decode_rounds
                .fetch_add(1, Ordering::Relaxed);
            stats
                .total_cuda_graph_decode_eligible_rounds
                .fetch_add(1, Ordering::Relaxed);
            stats
                .total_cuda_graph_decode_fallbacks
                .fetch_add(1, Ordering::Relaxed);
            stats
                .total_cuda_graph_decode_fallback_tokens
                .fetch_add(active_rows, Ordering::Relaxed);
            stats
                .total_cuda_graph_decode_miss_dynamic_kv
                .fetch_add(1, Ordering::Relaxed);
        }
        CudaGraphDecodeDecision::Ready { .. } => {
            stats
                .total_cuda_graph_decode_rounds
                .fetch_add(1, Ordering::Relaxed);
            stats
                .total_cuda_graph_decode_eligible_rounds
                .fetch_add(1, Ordering::Relaxed);
        }
        CudaGraphDecodeDecision::Miss(reason) => {
            stats
                .total_cuda_graph_decode_rounds
                .fetch_add(1, Ordering::Relaxed);
            stats
                .total_cuda_graph_decode_fallbacks
                .fetch_add(1, Ordering::Relaxed);
            stats
                .total_cuda_graph_decode_fallback_tokens
                .fetch_add(active_rows, Ordering::Relaxed);
            match reason {
                CudaGraphDecodeMiss::UnsupportedDevice => stats
                    .total_cuda_graph_decode_miss_device
                    .fetch_add(1, Ordering::Relaxed),
                CudaGraphDecodeMiss::NoBucket => stats
                    .total_cuda_graph_decode_miss_no_bucket
                    .fetch_add(1, Ordering::Relaxed),
                CudaGraphDecodeMiss::AttentionMask => stats
                    .total_cuda_graph_decode_miss_mask
                    .fetch_add(1, Ordering::Relaxed),
                CudaGraphDecodeMiss::PagedAttention => stats
                    .total_cuda_graph_decode_miss_paged_attention
                    .fetch_add(1, Ordering::Relaxed),
            };
        }
    }
}

fn parse_buckets(raw: Option<&str>) -> Vec<usize> {
    let mut buckets = BTreeSet::new();
    if let Some(raw) = raw {
        for part in raw.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(value) = trimmed.parse::<usize>() {
                if value > 0 {
                    buckets.insert(value);
                }
            }
        }
    }
    if buckets.is_empty() {
        buckets.extend(DEFAULT_BUCKETS.iter().copied());
    }
    buckets.into_iter().collect()
}

fn mask_blocks_capture(has_attention_mask: bool, fixed_width_mask: bool) -> bool {
    has_attention_mask && !fixed_width_mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_buckets_filters_sorts_and_deduplicates() {
        assert_eq!(parse_buckets(Some("8, 1, bad, 4, 4, 0")), vec![1, 4, 8]);
    }

    #[test]
    fn parse_buckets_uses_defaults_when_empty() {
        assert_eq!(parse_buckets(Some("bad,0")), vec![1, 2, 4, 8, 16]);
        assert_eq!(parse_buckets(None), vec![1, 2, 4, 8, 16]);
    }

    #[test]
    fn disabled_planner_does_not_classify() {
        let planner = CudaGraphDecodePlanner {
            enabled: false,
            fixed_width_decode: false,
            capture_runtime: false,
            capture_sampling: false,
            max_replays: 0,
            buckets: vec![1],
        };
        let decision =
            planner.classify_round(1, &Device::Cpu, DType::BF16, false, false, false, false);
        assert_eq!(decision, CudaGraphDecodeDecision::Disabled);
    }

    #[test]
    fn enabled_cpu_round_is_device_miss() {
        let planner = CudaGraphDecodePlanner {
            enabled: true,
            fixed_width_decode: false,
            capture_runtime: false,
            capture_sampling: false,
            max_replays: 0,
            buckets: vec![1],
        };
        let decision =
            planner.classify_round(1, &Device::Cpu, DType::BF16, false, false, false, false);
        assert_eq!(
            decision,
            CudaGraphDecodeDecision::Miss(CudaGraphDecodeMiss::UnsupportedDevice)
        );
    }

    #[test]
    fn only_dynamic_masks_block_capture() {
        assert!(mask_blocks_capture(true, false));
        assert!(!mask_blocks_capture(true, true));
        assert!(!mask_blocks_capture(false, false));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fixed_width_mask_reaches_dynamic_kv_blocker_without_append_metadata_on_cuda_bf16() {
        let device = Device::new_cuda(0).unwrap();
        let planner = CudaGraphDecodePlanner {
            enabled: true,
            fixed_width_decode: true,
            capture_runtime: false,
            capture_sampling: false,
            max_replays: 0,
            buckets: vec![1],
        };
        let decision = planner.classify_round(1, &device, DType::BF16, true, true, false, false);

        assert_eq!(
            decision,
            CudaGraphDecodeDecision::BlockedDynamicKv { bucket: 1 }
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fixed_width_append_metadata_marks_round_ready_on_cuda_bf16() {
        let device = Device::new_cuda(0).unwrap();
        let planner = CudaGraphDecodePlanner {
            enabled: true,
            fixed_width_decode: true,
            capture_runtime: false,
            capture_sampling: false,
            max_replays: 0,
            buckets: vec![1],
        };
        let decision = planner.classify_round(1, &device, DType::BF16, true, true, false, true);

        assert_eq!(decision, CudaGraphDecodeDecision::Ready { bucket: 1 });
    }

    #[test]
    fn record_blocked_round_updates_dynamic_kv_counters() {
        let stats = EngineStats::new();
        record_decision(
            &stats,
            CudaGraphDecodeDecision::BlockedDynamicKv { bucket: 4 },
            3,
        );
        assert_eq!(
            stats
                .total_cuda_graph_decode_eligible_rounds
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            stats
                .total_cuda_graph_decode_miss_dynamic_kv
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            stats
                .total_cuda_graph_decode_fallback_tokens
                .load(Ordering::Relaxed),
            3
        );
    }
}
