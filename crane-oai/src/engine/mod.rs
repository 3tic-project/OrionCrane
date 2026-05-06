//! Continuous-batching inference engine.
//!
//! # Architecture
//!
//! ```text
//! API handlers ──(request channel)──► Engine thread
//!       ◄──(per-request response channel)──┘
//!
//! Engine loop (each iteration = one "step"):
//!   1. Drain new requests from channel
//!   2. Detect & cancel disconnected clients
//!   3. Scheduler picks next batch (prefill > decode)
//!   4. Prefill step: run full prompt for ONE new sequence
//!   5. Decode step: batched or sequential forward for running sequences
//!   6. If idle → wait for new request while running idle maintenance
//! ```
//!
//! # Module layout
//!
//! | Module          | Responsibility                                   |
//! |-----------------|--------------------------------------------------|
//! | `types`         | Public request/response types + `EngineHandle`   |
//! | `stats`         | Lock-free counters shared with API layer          |
//! | `lifecycle`     | Response, finish, error, and cleanup handling     |
//! | `paged_kv`      | Qwen3 paged KV metadata and allocator             |
//! | `sampling`      | Token sampling (top-k, top-p, Gumbel-max, etc.) |
//! | `scheduler`     | FIFO scheduler with prefill priority              |
//! | `sequence`      | Per-request lifecycle state                       |
//! | `backend`       | `ModelBackend` trait + concrete implementations   |
//! | `model_factory` | Auto-detection and factory creation               |

pub mod backend;
mod cuda_graph;
mod cuda_memory;
mod lifecycle;
pub mod model_factory;
pub mod paged_kv;
mod paged_kv_runtime;
pub mod sampling;
pub mod scheduler;
pub mod sequence;
pub mod stats;
pub mod types;

// Re-export commonly used items for convenience.
pub use stats::{EngineStats, StatsSnapshot};
pub use types::{EngineHandle, EngineRequest, EngineResponse};

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use backend::{BatchDecodeExtractTimings, ModelBackend};
use crane_core::utils::token_output_stream::TokenOutputStream;
use paged_kv::{PagedKvAllocator, PagedKvGpuPageStore, DEFAULT_BLOCK_SIZE};
use sampling::SamplingBuffers;
use scheduler::{Scheduler, SchedulerOutput};
use sequence::{Sequence, SequenceStatus};

// ─────────────────────────────────────────────────────────────
//  Memory configuration
// ─────────────────────────────────────────────────────────────

/// Configuration for GPU memory limits.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum tokens per sequence (prompt + completion). 0 = unlimited.
    pub max_seq_len: usize,
    /// GPU memory limit in bytes. 0 = unlimited.
    /// This is an **absolute** limit on total GPU memory usage.
    pub gpu_memory_limit_bytes: u64,
    /// Baseline GPU memory recorded after model load + warmup.
    /// The memory gate compares `(current_used - baseline)` against
    /// `(gpu_memory_limit_bytes - baseline)` so that the limit represents
    /// the *total* allowed usage, not just KV-cache growth.
    pub baseline_gpu_bytes: u64,
}

impl MemoryConfig {
    /// Parse memory configuration from CLI arguments.
    ///
    /// `gpu_memory_limit` accepts:
    ///   - Absolute sizes: "5G", "8G", "5120M", "5368709120" (bytes)
    ///   - Utilization fraction: "0.7" (70% of total GPU memory)
    pub fn parse(max_seq_len: usize, gpu_memory_limit: Option<&str>, device: &Device) -> Self {
        let gpu_memory_limit_bytes = match gpu_memory_limit {
            Some(s) => Self::parse_memory_limit(s, device),
            None => 0,
        };
        Self {
            max_seq_len,
            gpu_memory_limit_bytes,
            baseline_gpu_bytes: 0,
        }
    }

    fn parse_memory_limit(s: &str, device: &Device) -> u64 {
        let s = s.trim();
        if s.is_empty() || s == "0" {
            return 0;
        }

        // Try absolute sizes: "5G", "8G", "5120M", "1024K"
        let upper = s.to_uppercase();
        if upper.ends_with('G') {
            if let Ok(n) = upper[..upper.len() - 1].trim().parse::<f64>() {
                return (n * (1u64 << 30) as f64) as u64;
            }
        }
        if upper.ends_with('M') {
            if let Ok(n) = upper[..upper.len() - 1].trim().parse::<f64>() {
                return (n * (1u64 << 20) as f64) as u64;
            }
        }

        // Try as a fraction (0.0 - 1.0)
        if let Ok(frac) = s.parse::<f64>() {
            if (0.0..=1.0).contains(&frac) {
                let total = Self::query_total_gpu_memory(device);
                if total > 0 {
                    return (frac * total as f64) as u64;
                }
            }
            // If > 1.0, treat as bytes
            if frac > 1.0 {
                return frac as u64;
            }
        }

        tracing::warn!("Could not parse gpu_memory_limit '{}', ignoring", s);
        0
    }

    /// Record baseline GPU memory (call after model load + warmup).
    pub fn record_baseline(&mut self, device: &Device) {
        let (used, _total) = query_gpu_memory_usage(device);
        self.baseline_gpu_bytes = used;
    }

    /// Query total GPU memory (bytes). Returns 0 if unavailable.
    fn query_total_gpu_memory(_device: &Device) -> u64 {
        #[cfg(feature = "cuda")]
        {
            if let Device::Cuda(_) = _device {
                if let Ok((_free, total)) =
                    candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
                {
                    return total as u64;
                }
            }
        }
        0
    }
}

/// Query current GPU memory usage. Returns (used_bytes, total_bytes).
/// Returns (0, 0) if not on CUDA.
fn query_gpu_memory_usage(_device: &Device) -> (u64, u64) {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = _device {
            if let Ok((free, total)) =
                candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
            {
                return ((total - free) as u64, total as u64);
            }
        }
    }
    (0, 0)
}

/// Format a byte count as a human-readable string (used in engine log messages).
fn format_bytes_engine(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1}G", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0}M", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{}B", bytes)
    }
}

fn env_flag(name: &str) -> bool {
    env_flag_default(name, false)
}

fn env_flag_default(name: &str, default: bool) -> bool {
    match std::env::var(name).ok().as_deref() {
        Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES") | Some("on")
        | Some("ON") => true,
        Some("0") | Some("false") | Some("FALSE") | Some("no") | Some("NO") | Some("off")
        | Some("OFF") => false,
        Some(_) => default,
        None => default,
    }
}

fn env_flag_is_explicit(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1")
            | Some("true")
            | Some("TRUE")
            | Some("yes")
            | Some("YES")
            | Some("on")
            | Some("ON")
            | Some("0")
            | Some("false")
            | Some("FALSE")
            | Some("no")
            | Some("NO")
            | Some("off")
            | Some("OFF")
    )
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

fn env_duration_secs(name: &str, default_secs: u64) -> Option<Duration> {
    let parse_default = || (default_secs > 0).then(|| Duration::from_secs(default_secs));
    let raw = match std::env::var(name) {
        Ok(value) => value,
        Err(_) => return parse_default(),
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return parse_default();
    }
    match trimmed.parse::<u64>() {
        Ok(0) => None,
        Ok(secs) => Some(Duration::from_secs(secs)),
        Err(_) => {
            warn!(
                value = %trimmed,
                default_secs,
                "Could not parse {name}; using default"
            );
            parse_default()
        }
    }
}

fn is_cuda_device(_device: &Device) -> bool {
    #[cfg(feature = "cuda")]
    {
        matches!(_device, Device::Cuda(_))
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

// ─────────────────────────────────────────────────────────────
//  InferenceEngine
// ─────────────────────────────────────────────────────────────

/// KV-to-GPU overhead factor.
///
/// `tracked_kv_bytes` only captures live per-sequence KV cache tensors, which
/// is roughly 15-20% of the *real* GPU memory consumed.  Batch-decode setup
/// creates padded copies, the CUDA caching allocator retains freed blocks,
/// and forward-pass intermediates add extra pressure.  Empirically the ratio
/// between actual GPU growth over baseline and tracked KV bytes is 5-8×.
///
/// We use 6× so that `kv_budget = (limit - baseline) / 6`.  This gives the
/// engine a realistic estimate of how much KV it can afford before the GPU
/// runs out of memory.
const KV_GPU_OVERHEAD_FACTOR: u64 = 6;

/// Continuous-batching inference engine.
///
/// Runs on a dedicated OS thread (model forward passes are synchronous).
/// Communicates with async API handlers via channels.
pub struct InferenceEngine {
    model: Box<dyn ModelBackend>,
    sequences: HashMap<String, Sequence>,
    token_streams: HashMap<String, TokenOutputStream>,
    scheduler: Scheduler,
    request_rx: mpsc::UnboundedReceiver<EngineRequest>,
    active_seq_id: Option<String>,
    num_layers: usize,
    stats: Arc<EngineStats>,
    /// How many tokens to decode for one sequence before switching.
    decode_tokens_per_seq: usize,
    /// Engine start time for uptime calculation.
    start_time: Instant,
    /// Step counter for periodic stats logging.
    step_counter: u64,
    sampling_buffers: SamplingBuffers,
    paged_kv_allocator: PagedKvAllocator,
    /// Memory configuration for VRAM limits.
    memory_config: MemoryConfig,
    /// When fully idle for this long, trim CUDA's async memory pool back to the driver.
    idle_cuda_mem_trim_after: Option<Duration>,
    /// Timestamp of last memory-limit warning (to throttle log spam).
    last_mem_warn: Instant,
    /// Tracked total KV cache bytes across all sequences (not relying on
    /// `cuMemGetInfo` which includes CUDA allocator pool bloat).
    tracked_kv_bytes: u64,
    /// Steps remaining before cuMemGetInfo checks are re-enabled after eviction.
    /// The CUDA caching allocator doesn't instantly reflect freed memory, so we
    /// grant a short cooldown after preemption to avoid a deadlock where
    /// cuMemGetInfo always reports over-limit.
    eviction_cooldown: u32,
    /// Emit per-stage profiling logs when CRANE_PROFILE=1.
    profile_enabled: bool,
    /// Debug-only shadow check for the paged-KV migration path. When enabled,
    /// batch decode setup validates page-store gather against direct packing.
    paged_kv_shadow_validate: bool,
    /// Limit shadow validation to the first N layers to keep debug runs bounded.
    paged_kv_shadow_max_layers: usize,
    /// Guarded first GPU-backed page store used by native append validation.
    paged_kv_gpu_store: Option<PagedKvGpuPageStore>,
    /// When enabled, append generated batch-decode K/V directly into GPU pages.
    paged_kv_native_append: bool,
    /// Reserved headroom near the GPU memory limit where validation-only page copies are skipped.
    paged_kv_pressure_reserve_bytes: u64,
    /// Use GPU page gather to rebuild per-sequence K/V caches instead of extracting from batch buffers.
    paged_kv_gather_extract: bool,
    /// Use GPU page storage directly for decode attention when all live rows are resident.
    paged_kv_attention: bool,
    /// Minimum active rows before the current paged-attention kernel is worth trying.
    paged_kv_attention_min_active_rows: usize,
    /// Minimum past sequence length before page-backed decode attention is worth trying.
    paged_kv_attention_min_seq_len: usize,
    /// CUDA Graph decode bucket planner and instrumentation gate.
    cuda_graph_decode: cuda_graph::CudaGraphDecodePlanner,
    /// Reusable device buffer for graph-candidate single-token batch input ids.
    cuda_graph_input_ids: crane_core::fused_ops::ReusableU32TensorBuffer,
    /// Reusable device buffer for graph-candidate RoPE position ids.
    cuda_graph_position_ids: crane_core::fused_ops::ReusableU32TensorBuffer,
    /// Reusable device buffer holding the current fixed-width K/V append slot.
    cuda_graph_append_offset: crane_core::fused_ops::ReusableU32TensorBuffer,
    /// Reusable device buffer for fixed-width attention masks captured by CUDA Graphs.
    cuda_graph_mask: crane_core::fused_ops::ReusableTensorBuffer,
    /// Persistent argmax token output buffer captured into the decode graph.
    /// One shared buffer is reused across buckets — `gpu_argmax_batch_kernel_only`
    /// only reallocates if the requested batch grows.
    #[cfg(feature = "cuda")]
    cuda_graph_sampling_buffers: crane_core::fused_ops::BatchGreedyCudaBuffers,
    /// Round 9: per-layer batched gather output published by the most recent
    /// `maybe_extract_paged_kv_gather`, paired with the batch order it was
    /// extracted for. Consumed by the next `setup_batch_decode` when the batch
    /// composition matches; otherwise dropped and `gather_batched_kv_for_batch`
    /// regathers fresh data for the new batch.
    pending_batched_kv_extract: Option<(Vec<String>, paged_kv_runtime::BatchedKvExtract)>,
    /// Generation of model-owned batch decode workspaces currently backing captured graphs.
    #[cfg(feature = "cuda")]
    cuda_graph_workspace_generation: u64,
    /// Captured graphs are valid only for the current batch-decode KV buffers.
    #[cfg(feature = "cuda")]
    cuda_graph_decode_entries:
        HashMap<cuda_graph::CudaGraphDecodeKey, cuda_graph::CudaGraphDecodeEntry>,
    /// Keys for which capture or replay has already failed once; we stop retrying
    /// to avoid burning capture overhead on every step.
    #[cfg(feature = "cuda")]
    cuda_graph_decode_poisoned: std::collections::HashSet<cuda_graph::CudaGraphDecodeKey>,
    /// Wait-batching: max time (µs) to wait for additional in-flight requests
    /// after `drain_requests`, when the current batch is below the configured
    /// target. 0 disables the feature (default).
    sched_wait_batch_us: u64,
    /// Wait-batching: target value of `running.len() + waiting.len()`.
    /// The spin loop exits early once this target is met.
    sched_wait_batch_target: usize,
    /// Wait-batching: poll interval (µs) between `try_recv` attempts.
    sched_wait_batch_poll_us: u64,
}

impl InferenceEngine {
    /// Create the engine and return a handle for submitting requests.
    pub fn new(
        model: Box<dyn ModelBackend>,
        max_concurrent: usize,
        decode_tokens_per_seq: usize,
        memory_config: MemoryConfig,
    ) -> (Self, EngineHandle) {
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let num_layers = model.num_layers();

        // Cap max_concurrent to 1 for models without KV cache swapping.
        let effective_max = if model.supports_kv_swap() {
            max_concurrent
        } else {
            1.min(max_concurrent)
        };
        if effective_max != max_concurrent {
            info!(
                "Model does not support KV swap — limiting max_concurrent to {}",
                effective_max
            );
        }

        let stats = Arc::new(EngineStats::new());
        let paged_kv_default_enabled =
            is_cuda_device(model.device()) && model.dtype() == DType::BF16;
        let paged_kv_block_size = env_usize("CRANE_PAGED_KV_BLOCK_SIZE", DEFAULT_BLOCK_SIZE);
        let paged_kv_layout = model.kv_cache_layout().unwrap_or_else(|| {
            warn!("Model did not expose KV layout; paged KV byte counters will be zero");
            paged_kv::PagedKvLayout {
                num_layers,
                num_kv_heads: 0,
                head_dim: 0,
                dtype_size_bytes: 0,
            }
        });
        let paged_kv_allocator = PagedKvAllocator::new(paged_kv_block_size, paged_kv_layout);
        let paged_kv_shadow_validate = env_flag("CRANE_PAGED_KV_SHADOW_VALIDATE");
        let paged_kv_shadow_max_layers = env_usize("CRANE_PAGED_KV_SHADOW_MAX_LAYERS", num_layers)
            .min(num_layers)
            .max(1);
        let paged_kv_native_append =
            env_flag_default("CRANE_PAGED_KV_NATIVE_APPEND", paged_kv_default_enabled);
        let paged_kv_pressure_reserve_bytes =
            (env_usize("CRANE_PAGED_KV_PRESSURE_RESERVE_MB", 512) as u64) << 20;
        let paged_kv_gather_extract = paged_kv_native_append
            && env_flag_default("CRANE_PAGED_KV_GATHER_EXTRACT", paged_kv_default_enabled);
        // Default OFF: enabling the current paged attention kernel regresses end-to-end throughput
        // on Qwen3 short/medium translation contexts compared to the contiguous GQA path. Keep the
        // env var as an explicit opt-in until the kernel is competitive across the heuristic range.
        let paged_kv_attention =
            paged_kv_native_append && env_flag_default("CRANE_PAGED_KV_ATTENTION", false);
        let paged_kv_attention_min_active_rows =
            env_usize("CRANE_PAGED_KV_ATTENTION_MIN_ACTIVE_ROWS", 1);
        let paged_kv_attention_min_seq_len =
            env_usize("CRANE_PAGED_KV_ATTENTION_MIN_SEQ_LEN", 1024);
        let cuda_graph_decode = cuda_graph::CudaGraphDecodePlanner::from_env();
        let paged_kv_gpu_store = if paged_kv_native_append {
            if is_cuda_device(model.device()) && model.dtype() == DType::BF16 {
                Some(PagedKvGpuPageStore::new(
                    paged_kv_block_size,
                    paged_kv_layout,
                    model.dtype(),
                    model.device(),
                ))
            } else if env_flag_is_explicit("CRANE_PAGED_KV_NATIVE_APPEND") {
                warn!(
                    dtype = ?model.dtype(),
                    "CRANE_PAGED_KV_NATIVE_APPEND requested but requires CUDA BF16; continuing with contiguous fallback"
                );
                None
            } else {
                None
            }
        } else {
            None
        };
        #[cfg(feature = "cuda")]
        let cuda_graph_workspace_generation = model.batch_decode_workspace_generation();
        let engine = Self {
            model,
            sequences: HashMap::new(),
            token_streams: HashMap::new(),
            scheduler: Scheduler::new(effective_max),
            request_rx,
            active_seq_id: None,
            num_layers,
            stats: stats.clone(),
            decode_tokens_per_seq: decode_tokens_per_seq.max(1),
            start_time: Instant::now(),
            step_counter: 0,
            sampling_buffers: SamplingBuffers::new(),
            paged_kv_allocator,
            memory_config,
            idle_cuda_mem_trim_after: env_duration_secs("CRANE_IDLE_CUDA_MEM_TRIM_SECS", 120),
            last_mem_warn: Instant::now() - std::time::Duration::from_secs(60),
            tracked_kv_bytes: 0,
            eviction_cooldown: 0,
            profile_enabled: env_flag("CRANE_PROFILE"),
            paged_kv_shadow_validate,
            paged_kv_shadow_max_layers,
            paged_kv_gpu_store,
            paged_kv_native_append,
            paged_kv_pressure_reserve_bytes,
            paged_kv_gather_extract,
            paged_kv_attention,
            paged_kv_attention_min_active_rows,
            paged_kv_attention_min_seq_len,
            cuda_graph_decode,
            cuda_graph_input_ids: crane_core::fused_ops::ReusableU32TensorBuffer::new(),
            cuda_graph_position_ids: crane_core::fused_ops::ReusableU32TensorBuffer::new(),
            cuda_graph_append_offset: crane_core::fused_ops::ReusableU32TensorBuffer::new(),
            cuda_graph_mask: crane_core::fused_ops::ReusableTensorBuffer::new(),
            #[cfg(feature = "cuda")]
            cuda_graph_sampling_buffers: crane_core::fused_ops::BatchGreedyCudaBuffers::new(),
            pending_batched_kv_extract: None,
            #[cfg(feature = "cuda")]
            cuda_graph_workspace_generation,
            #[cfg(feature = "cuda")]
            cuda_graph_decode_entries: HashMap::new(),
            #[cfg(feature = "cuda")]
            cuda_graph_decode_poisoned: std::collections::HashSet::new(),
            sched_wait_batch_us: env_usize("CRANE_SCHED_WAIT_BATCH_US", 0) as u64,
            sched_wait_batch_target: env_usize("CRANE_SCHED_WAIT_BATCH_TARGET", effective_max),
            sched_wait_batch_poll_us: env_usize("CRANE_SCHED_WAIT_BATCH_POLL_US", 50) as u64,
        };
        stats
            .paged_kv_block_size
            .store(paged_kv_block_size as u64, Ordering::Relaxed);
        if paged_kv_shadow_validate {
            info!(
                layers = paged_kv_shadow_max_layers,
                "CRANE_PAGED_KV_SHADOW_VALIDATE enabled; batch setup will compare paged gather with direct packing"
            );
        }
        if paged_kv_native_append {
            info!(
                enabled = engine.paged_kv_gpu_store.is_some(),
                pressure_reserve_bytes = engine.paged_kv_pressure_reserve_bytes,
                gather_extract = engine.paged_kv_gather_extract,
                paged_attention = engine.paged_kv_attention,
                "CRANE_PAGED_KV_NATIVE_APPEND enabled; batch past K/V and generated K/V will be copied into GPU pages before fallback extraction"
            );
        }
        if paged_kv_attention {
            info!(
                enabled = engine.paged_kv_gpu_store.is_some() && engine.paged_kv_native_append,
                min_active_rows = engine.paged_kv_attention_min_active_rows,
                min_seq_len = engine.paged_kv_attention_min_seq_len,
                "CRANE_PAGED_KV_ATTENTION enabled; resident decode rows will use GPU page-backed attention when the batch passes the runtime heuristic"
            );
        }
        if engine.cuda_graph_decode.enabled() {
            info!(
                buckets = %engine.cuda_graph_decode.bucket_csv(),
                fixed_width_decode = engine.cuda_graph_decode.fixed_width_decode(),
                capture_runtime = engine.cuda_graph_decode.capture_runtime(),
                "CRANE_CUDA_GRAPH_DECODE enabled; decode rounds are bucketed, fixed-width candidates use the eager graph-shape baseline unless CRANE_CUDA_GRAPH_DECODE_CAPTURE=1 is set, and miss reasons are counted"
            );
            if engine.cuda_graph_decode.capture_runtime() {
                info!(
                    "CRANE_CUDA_GRAPH_DECODE_CAPTURE=1: graph capture+reuse is enabled. \
                     The Round 5 stale-RoPE-max_position drift was fixed (2026-04-30); \
                     replays now produce baseline-equivalent token distributions. \
                     Performance is on par with eager batched decode at MC<=64 in current \
                     measurements — opt-in for determinism, not for speedup. \
                     See docs/qwen3/benchmarks/qwen3_round5_cuda_graph_2026_05_08.md."
                );
            }
        }
        if is_cuda_device(engine.model.device()) {
            if let Some(after) = engine.idle_cuda_mem_trim_after {
                info!(
                    idle_secs = after.as_secs(),
                    "CRANE_IDLE_CUDA_MEM_TRIM_SECS enabled; idle CUDA memory pool trim will run after the timeout"
                );
            } else {
                info!("CRANE_IDLE_CUDA_MEM_TRIM_SECS=0; idle CUDA memory pool trim disabled");
            }
        }
        let handle = EngineHandle { request_tx, stats };
        (engine, handle)
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn capture_cuda_graph_decode_round(
        &mut self,
        key: cuda_graph::CudaGraphDecodeKey,
        input_ids: &Tensor,
        positions: &[usize],
        position_ids: &Tensor,
        append_offset: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        fixed_cache_width: usize,
    ) -> candle_core::Result<(Tensor, Option<usize>)> {
        let device = self.model.device().clone();
        let capture = crane_core::fused_ops::CudaGraphCapture::begin(&device).map_err(|e| {
            candle_core::Error::Msg(format!("cuda graph decode capture begin failed: {e}"))
        })?;
        let logits = match self.model.step_batch_decode_fixed_width_with_position_ids(
            input_ids,
            positions,
            position_ids,
            append_offset,
            attention_mask,
            fixed_cache_width,
        ) {
            Ok(logits) => logits,
            Err(e) => {
                let _ = capture.end();
                return Err(e);
            }
        };
        // P4-A: also capture the greedy argmax kernel inside the same graph,
        // writing tokens into the engine-owned persistent buffer. The matching
        // DtoH happens after `cuGraphLaunch` in the replay path. Failures here
        // are non-fatal — we just fall back to out-of-graph sampling.
        let captured_sample_batch = if self.cuda_graph_decode.capture_sampling() {
            match crane_core::fused_ops::gpu_argmax_batch_kernel_only(
                &logits,
                &mut self.cuda_graph_sampling_buffers,
            ) {
                Ok(0) => None,
                Ok(n) => Some(n),
                Err(e) => {
                    warn!(?key, error = %e, "in-graph argmax capture failed; sampling will run out-of-graph");
                    None
                }
            }
        } else {
            None
        };
        let graph = capture
            .end()
            .map_err(|e| {
                candle_core::Error::Msg(format!("cuda graph decode capture end failed: {e}"))
            })?
            .ok_or_else(|| candle_core::Error::Msg("empty CUDA Graph decode capture".into()))?;
        graph.upload().map_err(|e| {
            candle_core::Error::Msg(format!("cuda graph decode upload failed: {e}"))
        })?;
        graph.launch().map_err(|e| {
            candle_core::Error::Msg(format!("cuda graph decode first launch failed: {e}"))
        })?;
        self.cuda_graph_decode_entries.insert(
            key,
            cuda_graph::CudaGraphDecodeEntry {
                graph,
                logits: logits.clone(),
                captured_sample_batch,
                replays_used: 0,
            },
        );
        Ok((logits, captured_sample_batch))
    }

    // ─────────────────────────────────────────────────────────
    //  Main loop
    // ─────────────────────────────────────────────────────────

    /// Run the engine loop (blocking — call from a dedicated thread).
    pub fn run(mut self) {
        // Log effective memory budget.
        let baseline = self.memory_config.baseline_gpu_bytes;
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit > 0 {
            let kv_budget = self.kv_budget_bytes();
            if kv_budget == 0 || limit <= baseline {
                warn!(
                    "gpu_memory_limit ({}) <= model baseline ({}). \
                     KV-cache budget is 0 — all sequences will be immediately preempted. \
                     Set CRANE_DISABLE_GPU_MEM_HARD_CHECK=1 to bypass on shared GPUs \
                     (the baseline is recorded via cuMemGetInfo which sees the WHOLE device, \
                     including memory used by other processes).",
                    format_bytes_engine(limit),
                    format_bytes_engine(baseline),
                );
            } else {
                info!(
                    "Memory budget: total_limit={}, model_baseline={}, kv_budget={} (overhead={}x, also checked by cuMemGetInfo)",
                    format_bytes_engine(limit),
                    format_bytes_engine(baseline),
                    format_bytes_engine(kv_budget),
                    KV_GPU_OVERHEAD_FACTOR,
                );
            }
        }
        info!(
            "Engine started (max_concurrent={}, decode_tokens_per_seq={}, max_seq_len={})",
            self.scheduler.max_running,
            self.decode_tokens_per_seq,
            if self.memory_config.max_seq_len == 0 {
                "unlimited".to_string()
            } else {
                self.memory_config.max_seq_len.to_string()
            },
        );
        if self.profile_enabled {
            info!(
                target: "crane_profile",
                max_concurrent = self.scheduler.max_running,
                decode_tokens_per_seq = self.decode_tokens_per_seq,
                "CRANE_PROFILE enabled; emitting per-stage inference timings",
            );
        }
        if self.sched_wait_batch_us > 0 {
            info!(
                wait_us = self.sched_wait_batch_us,
                target = self.sched_wait_batch_target,
                poll_us = self.sched_wait_batch_poll_us,
                "CRANE_SCHED_WAIT_BATCH enabled: scheduler will wait briefly for in-flight requests when batch < target"
            );
        }
        if env_flag("CRANE_DISABLE_GPU_MEM_HARD_CHECK") {
            warn!(
                "CRANE_DISABLE_GPU_MEM_HARD_CHECK=1: KV-budget eviction disabled. \
                 The engine will rely on tracked_kv_bytes only and will NOT preempt under memory \
                 pressure. Use this only on shared GPUs (where cuMemGetInfo is polluted by other \
                 processes) or when you're confident your workload fits in VRAM."
            );
        }

        loop {
            self.drain_with_wait_batch();
            self.check_cancelled();

            // Decrement eviction cooldown (cuMemGetInfo grace period).
            self.eviction_cooldown = self.eviction_cooldown.saturating_sub(1);

            self.stats
                .active_sequences
                .store(self.scheduler.running.len() as u64, Ordering::Relaxed);
            self.stats
                .waiting_sequences
                .store(self.scheduler.waiting.len() as u64, Ordering::Relaxed);
            self.stats
                .tracked_kv_cache_bytes
                .store(self.tracked_kv_bytes, Ordering::Relaxed);

            let output = self.scheduler.schedule();

            // TEMP DIAGNOSTIC: log every schedule decision when CRANE_SCHED_TRACE=1.
            if env_flag("CRANE_SCHED_TRACE") {
                if let Some(o) = &output {
                    eprintln!(
                        "[sched] running={} waiting={} max={} eff_max={:?} -> {} batch={}",
                        self.scheduler.running.len(),
                        self.scheduler.waiting.len(),
                        self.scheduler.max_running,
                        self.scheduler.effective_max_running,
                        if o.is_prefill { "PREFILL" } else { "DECODE" },
                        o.batch.len()
                    );
                }
            }

            match output {
                Some(output) => {
                    // KV cache budget gate: if a prefill is scheduled but we're
                    // over the KV budget, first try to evict (preempt) the
                    // largest running sequence to make room. If still over,
                    // defer the prefill and drain existing sequences.
                    if output.is_prefill && self.is_over_kv_budget() {
                        // Attempt eviction before deferring.
                        self.evict_if_needed();

                        if self.is_over_kv_budget() && !self.scheduler.running.is_empty() {
                            // Still over budget and have running sequences to drain.
                            for seq_id in &output.batch {
                                self.scheduler.waiting.push_front(seq_id.clone());
                            }
                            let decode_batch: Vec<String> =
                                self.scheduler.running.iter().cloned().collect();
                            let decode_output = SchedulerOutput {
                                batch: decode_batch,
                                is_prefill: false,
                            };
                            self.execute_step(decode_output);
                        } else {
                            // Budget OK after eviction (or nothing running) — proceed.
                            self.execute_step(output);
                        }
                    } else {
                        self.execute_step(output);
                    }
                    self.step_counter += 1;

                    if self.step_counter % 50 == 0 {
                        self.log_stats();
                    }
                }
                None => match self.wait_for_request_while_idle() {
                    Some(req) => self.accept_request(req),
                    None => {
                        info!("Engine channel closed, shutting down");
                        self.log_stats();
                        return;
                    }
                },
            }
        }
    }

    fn wait_for_request_while_idle(&mut self) -> Option<EngineRequest> {
        let idle_started = Instant::now();
        let mut idle_trim_done = false;
        let idle_poll = Duration::from_millis(20);

        loop {
            match self.request_rx.try_recv() {
                Ok(req) => return Some(req),
                Err(mpsc::error::TryRecvError::Disconnected) => return None,
                Err(mpsc::error::TryRecvError::Empty) => {}
            }

            if !idle_trim_done {
                if let Some(trim_after) = self.idle_cuda_mem_trim_after {
                    let idle_for = idle_started.elapsed();
                    if idle_for >= trim_after {
                        self.run_idle_cuda_memory_trim(idle_for);
                        idle_trim_done = true;
                    }
                }
            }

            let sleep_for = if idle_trim_done {
                idle_poll
            } else if let Some(trim_after) = self.idle_cuda_mem_trim_after {
                trim_after
                    .saturating_sub(idle_started.elapsed())
                    .min(idle_poll)
            } else {
                idle_poll
            };
            std::thread::sleep(sleep_for.max(Duration::from_millis(1)));
        }
    }

    fn run_idle_cuda_memory_trim(&mut self, idle_for: Duration) {
        self.clear_idle_request_cache_state();
        match cuda_memory::trim_idle_cuda_memory_pool(self.model.device()) {
            Ok(Some(report)) => {
                self.stats
                    .gpu_memory_used_bytes
                    .store(report.gpu_used_after_bytes, Ordering::Relaxed);
                self.stats
                    .gpu_memory_total_bytes
                    .store(report.gpu_total_bytes, Ordering::Relaxed);
                info!(
                    idle_secs = idle_for.as_secs(),
                    gpu_before = %format_bytes_engine(report.gpu_used_before_bytes),
                    gpu_after = %format_bytes_engine(report.gpu_used_after_bytes),
                    gpu_reclaimed = %format_bytes_engine(report.gpu_reclaimed_bytes()),
                    pool_reserved_before = %format_optional_bytes_engine(report.pool_reserved_before_bytes),
                    pool_reserved_after = %format_optional_bytes_engine(report.pool_reserved_after_bytes),
                    pool_reserved_reclaimed = %format_optional_bytes_engine(report.pool_reserved_reclaimed_bytes()),
                    pool_used_before = %format_optional_bytes_engine(report.pool_used_before_bytes),
                    pool_used_after = %format_optional_bytes_engine(report.pool_used_after_bytes),
                    "trimmed idle CUDA memory pool"
                );
            }
            Ok(None) => {
                debug!(
                    idle_secs = idle_for.as_secs(),
                    "idle CUDA memory pool trim skipped"
                );
            }
            Err(err) => {
                warn!(
                    idle_secs = idle_for.as_secs(),
                    error = %err,
                    "idle CUDA memory pool trim failed"
                );
            }
        }
    }

    fn log_stats(&self) {
        let uptime = self.start_time.elapsed().as_secs();
        let (gpu_used, gpu_total) = query_gpu_memory_usage(self.model.device());
        self.stats
            .tracked_kv_cache_bytes
            .store(self.tracked_kv_bytes, Ordering::Relaxed);
        self.stats
            .gpu_memory_used_bytes
            .store(gpu_used, Ordering::Relaxed);
        self.stats
            .gpu_memory_total_bytes
            .store(gpu_total, Ordering::Relaxed);
        let snap = self.stats.snapshot();
        let budget = self.kv_budget_bytes();
        let budget_info = if budget < u64::MAX {
            format!(" kv_budget: {}", format_bytes_engine(budget))
        } else {
            String::new()
        };
        let gpu_info = if gpu_total > 0 {
            format!(
                " | gpu_mem: {:.1}G/{:.1}G ({:.0}%) | kv_cache: {}{}",
                gpu_used as f64 / (1u64 << 30) as f64,
                gpu_total as f64 / (1u64 << 30) as f64,
                gpu_used as f64 / gpu_total as f64 * 100.0,
                format_bytes_engine(self.tracked_kv_bytes),
                budget_info,
            )
        } else {
            format!(
                " | kv_cache: {}{}",
                format_bytes_engine(self.tracked_kv_bytes),
                budget_info
            )
        };
        info!(
            "Engine stats | uptime={}s | requests: total={} completed={} cancelled={} failed={} | \
             sequences: active={} waiting={} | \
             tokens: prompt={} completion={} | \
             kv_swaps={} | \
             speed: prefill={:.1} tok/s decode={:.1} tok/s{}",
            uptime,
            snap.total_requests,
            snap.completed_requests,
            snap.cancelled_requests,
            snap.failed_requests,
            snap.active_sequences,
            snap.waiting_sequences,
            snap.total_prompt_tokens,
            snap.total_completion_tokens,
            snap.total_kv_swaps,
            snap.avg_prefill_tokens_per_sec,
            snap.avg_decode_tokens_per_sec,
            gpu_info,
        );
    }

    // ─────────────────────────────────────────────────────────
    //  Memory management
    // ─────────────────────────────────────────────────────────

    /// Recount `tracked_kv_bytes` from all sequences.
    /// For the active sequence, bytes are in the model (uses `active_kv_cache_bytes`).
    /// For other sequences, bytes are stored in `seq.kv_caches`.
    fn recount_kv_bytes(&mut self) {
        let mut total: u64 = 0;
        for (id, seq) in &self.sequences {
            if self.active_seq_id.as_deref() == Some(id.as_str()) {
                total += self.model.active_kv_cache_bytes();
            } else {
                total += sequence::kv_cache_bytes(&seq.kv_caches);
            }
        }
        self.tracked_kv_bytes = total;
        self.stats
            .tracked_kv_cache_bytes
            .store(self.tracked_kv_bytes, Ordering::Relaxed);
    }

    /// KV cache budget **in KV-cache bytes** (not raw GPU bytes).
    ///
    /// Each byte of live KV cache costs roughly `KV_GPU_OVERHEAD_FACTOR` bytes
    /// of real GPU memory (due to padded batch copies, CUDA pool bloat, and
    /// forward-pass intermediates).  The budget is therefore:
    ///
    /// ```text
    /// kv_budget = (gpu_limit - baseline) / KV_GPU_OVERHEAD_FACTOR
    /// ```
    ///
    /// Returns `u64::MAX` when no limit is configured.
    fn kv_budget_bytes(&self) -> u64 {
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit == 0 {
            return u64::MAX;
        }
        let raw = limit.saturating_sub(self.memory_config.baseline_gpu_bytes);
        raw / KV_GPU_OVERHEAD_FACTOR
    }

    /// Check whether the engine should block new prefills due to memory
    /// pressure.  Two complementary checks:
    ///
    /// 1. **KV budget** — `tracked_kv_bytes > kv_budget_bytes()`.  This is the
    ///    primary admission control, using an overhead factor to estimate real
    ///    GPU cost from the tracked KV cache bytes.
    ///
    /// 2. **cuMemGetInfo hard safety** — if actual GPU memory (as reported by
    ///    the driver) exceeds the configured limit, block prefills.  This
    ///    catches cases where the overhead factor underestimates.  The check
    ///    is skipped during `eviction_cooldown` to avoid a deadlock (the CUDA
    ///    caching allocator doesn't instantly reflect freed memory).
    fn is_over_kv_budget(&mut self) -> bool {
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit == 0 {
            return false;
        }

        // On a SHARED GPU, both `baseline_gpu_bytes` (recorded at startup via
        // cuMemGetInfo, which is whole-device) and the runtime cuMemGetInfo
        // check below are polluted by other processes' allocations. Operators
        // running on a shared device should set this env var to disable both
        // checks and rely on continuous batching's own memory accounting
        // (model weights are loaded; KV growth is bounded by max_concurrent
        // and max_seq_len). Without this knob, every prefill triggers a
        // false-positive eviction, capping `effective_max_running` at zero
        // and starving batched decode entirely.
        if env_flag("CRANE_DISABLE_GPU_MEM_HARD_CHECK") {
            return false;
        }

        let budget = self.kv_budget_bytes();
        if budget == 0 {
            return true; // limit <= baseline
        }

        // Check 1: tracked KV bytes vs overhead-adjusted budget.
        if self.tracked_kv_bytes > budget {
            let now = Instant::now();
            if now.duration_since(self.last_mem_warn).as_secs() >= 5 {
                self.last_mem_warn = now;
                warn!(
                    "KV budget exceeded: kv_used={} > kv_budget={} (limit={} baseline={} overhead={}x)",
                    format_bytes_engine(self.tracked_kv_bytes),
                    format_bytes_engine(budget),
                    format_bytes_engine(limit),
                    format_bytes_engine(self.memory_config.baseline_gpu_bytes),
                    KV_GPU_OVERHEAD_FACTOR,
                );
            }
            return true;
        }

        // Check 2: cuMemGetInfo hard safety (skip during cooldown).
        //
        // NOTE: On a SHARED GPU, `cuMemGetInfo` reports whole-device usage —
        // including memory consumed by other processes. That can trigger a
        // false-positive eviction loop on every prefill (capping
        // `effective_max_running` to 0 → engine permanently stuck at batch=1
        // → batched decode never fires). Set `CRANE_DISABLE_GPU_MEM_HARD_CHECK=1`
        // when running on a shared device to disable this check and rely on
        // `tracked_kv_bytes` budgeting only.
        if self.eviction_cooldown == 0 && !env_flag("CRANE_DISABLE_GPU_MEM_HARD_CHECK") {
            let (gpu_used, _) = query_gpu_memory_usage(self.model.device());
            if gpu_used > 0 && gpu_used > limit {
                let now = Instant::now();
                if now.duration_since(self.last_mem_warn).as_secs() >= 5 {
                    self.last_mem_warn = now;
                    warn!(
                        "GPU memory hard limit exceeded: gpu_used={} > limit={} (kv_tracked={})",
                        format_bytes_engine(gpu_used),
                        format_bytes_engine(limit),
                        format_bytes_engine(self.tracked_kv_bytes),
                    );
                }
                return true;
            }
        }

        false
    }

    /// Preempt (evict) running sequences until KV usage is within budget.
    ///
    /// Eviction policy: **longest-output-first** — the sequence that has
    /// generated the most tokens (and therefore holds the largest KV cache)
    /// is evicted first. Its KV cache is dropped and it is moved back to
    /// the waiting queue for later re-prefill.
    ///
    /// This mirrors sglang's retraction strategy.
    fn evict_if_needed(&mut self) {
        let budget = self.kv_budget_bytes();
        if budget == u64::MAX {
            return;
        }

        while self.tracked_kv_bytes > budget && !self.scheduler.running.is_empty() {
            // Find the running sequence with the most generated tokens (largest KV).
            let victim_id = self
                .scheduler
                .running
                .iter()
                .filter_map(|id| {
                    self.sequences
                        .get(id)
                        .map(|seq| (id.clone(), seq.tokens.len()))
                })
                .max_by_key(|(_, len)| *len)
                .map(|(id, _)| id);

            let victim_id = match victim_id {
                Some(id) => id,
                None => break,
            };

            // Compute bytes being freed.
            let freed = self
                .sequences
                .get(&victim_id)
                .map(|seq| sequence::kv_cache_bytes(&seq.kv_caches))
                .unwrap_or(0);

            info!(
                id = %victim_id,
                freed_bytes = %format_bytes_engine(freed),
                kv_used = %format_bytes_engine(self.tracked_kv_bytes),
                kv_budget = %format_bytes_engine(budget),
                "Preempting sequence (KV cache eviction) — will re-prefill later",
            );

            // If this sequence's KV is currently loaded in the model, clear it.
            if self.active_seq_id.as_deref() == Some(&victim_id) {
                self.model.clear_kv_cache();
                self.active_seq_id = None;
            }

            // Drop KV caches and reset sequence state to Waiting.
            if let Some(seq) = self.sequences.get_mut(&victim_id) {
                seq.kv_caches = vec![None; self.num_layers];
                seq.status = SequenceStatus::Waiting;
                // Reset tokens to just the prompt to allow re-prefill.
                seq.tokens.truncate(seq.prompt_len);
            }
            self.release_paged_kv_for_sequence(&victim_id);

            self.tracked_kv_bytes = self.tracked_kv_bytes.saturating_sub(freed);
            self.stats
                .tracked_kv_cache_bytes
                .store(self.tracked_kv_bytes, Ordering::Relaxed);

            // Move from running back to waiting (back, not front — avoid
            // immediate re-prefill which would cause thrashing).
            self.scheduler.running.retain(|id| id != &victim_id);
            self.scheduler.waiting.push_back(victim_id);
        }

        // Cap effective max_running to the post-eviction running count.
        // This prevents the scheduler from admitting new sequences that
        // would immediately exceed the budget again (eviction thrashing).
        // The cap is lifted when a sequence finishes naturally.
        let post_eviction_running = self.scheduler.running.len();
        self.scheduler.effective_max_running = Some(post_eviction_running);
        info!(
            "Eviction complete: capping concurrent sequences at {} (was {})",
            post_eviction_running, self.scheduler.max_running,
        );

        // Grant a cooldown period so the cuMemGetInfo hard-safety check
        // doesn't immediately re-trigger (CUDA pool retains freed blocks).
        self.eviction_cooldown = 5;
    }

    /// Effective max_tokens for a request, taking server-level max_seq_len into account.
    fn effective_max_tokens(&self, prompt_len: usize, requested_max_tokens: usize) -> usize {
        if self.memory_config.max_seq_len == 0 {
            return requested_max_tokens;
        }
        let remaining = self.memory_config.max_seq_len.saturating_sub(prompt_len);
        requested_max_tokens.min(remaining)
    }

    // ─────────────────────────────────────────────────────────
    //  Request handling
    // ─────────────────────────────────────────────────────────

    fn drain_requests(&mut self) {
        while let Ok(req) = self.request_rx.try_recv() {
            self.accept_request(req);
        }
    }

    /// Wait-batching: after the immediate non-blocking drain, optionally spin
    /// briefly to capture additional in-flight requests so they can be packed
    /// into the same prefill burst (and subsequent decode batch).
    ///
    /// This trades a small (µs-scale) latency cost for a larger steady-state
    /// decode batch, directly addressing the batch-size variance observed in
    /// the Round 3 profile (batch=3 ↔ batch=13 → 396 ↔ 643 tok/s).
    ///
    /// Conditions for waiting:
    ///   - `sched_wait_batch_us > 0` (feature opted in)
    ///   - `running` is non-empty (idle path is handled by the bottom-of-loop
    ///     `blocking_recv`; we never block when there's truly no work)
    ///   - `running.len() + waiting.len() < sched_wait_batch_target`
    ///
    /// Exits early when the target is met. Worst-case wait is `sched_wait_batch_us`
    /// per loop iteration.
    fn drain_with_wait_batch(&mut self) {
        self.drain_requests();

        let wait_us = self.sched_wait_batch_us;
        if wait_us == 0 {
            return;
        }
        let running = self.scheduler.running.len();
        if running == 0 {
            // Idle: let the bottom-of-loop blocking_recv handle it.
            return;
        }
        let target = self.sched_wait_batch_target.max(1);
        let current = running + self.scheduler.waiting.len();
        if current >= target {
            return;
        }

        let start = Instant::now();
        let deadline = start + std::time::Duration::from_micros(wait_us);
        let poll = std::time::Duration::from_micros(self.sched_wait_batch_poll_us.max(1));
        let mut arrivals: u64 = 0;
        let mut hit_target = false;

        loop {
            // Drain whatever has arrived since the last poll.
            while let Ok(req) = self.request_rx.try_recv() {
                self.accept_request(req);
                arrivals += 1;
            }
            let cur = self.scheduler.running.len() + self.scheduler.waiting.len();
            if cur >= target {
                hit_target = true;
                break;
            }
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            // Sleep for the smaller of `poll` and remaining time.
            let remaining = deadline - now;
            std::thread::sleep(if remaining < poll { remaining } else { poll });
        }

        let elapsed_us = start.elapsed().as_micros() as u64;
        self.stats
            .total_sched_wait_calls
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_sched_wait_arrivals
            .fetch_add(arrivals, Ordering::Relaxed);
        self.stats
            .total_sched_wait_time_us
            .fetch_add(elapsed_us, Ordering::Relaxed);
        if hit_target {
            self.stats
                .total_sched_wait_target_hits
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn accept_request(&mut self, req: EngineRequest) {
        let prompt_len = req.tokens.len();
        let tokenizer = self.model.tokenizer().clone();

        // Reject prompts that already exceed max_seq_len.
        if self.memory_config.max_seq_len > 0 && prompt_len > self.memory_config.max_seq_len {
            warn!(
                id = %req.id,
                prompt_len,
                max_seq_len = self.memory_config.max_seq_len,
                "Prompt exceeds max_seq_len, rejecting request",
            );
            let _ = req.response_tx.send(EngineResponse::Error(format!(
                "Prompt length ({}) exceeds server max_seq_len ({})",
                prompt_len, self.memory_config.max_seq_len,
            )));
            self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Cap max_tokens to respect max_seq_len.
        let effective_max_tokens = self.effective_max_tokens(prompt_len, req.max_tokens);

        info!(
            id = %req.id,
            prompt_len,
            max_tokens = effective_max_tokens,
            temp = ?req.temperature,
            top_p = ?req.top_p,
            top_k = ?req.top_k,
            rep_penalty = req.repetition_penalty,
            "New request accepted (queue: waiting={} running={})",
            self.scheduler.waiting.len() + 1,
            self.scheduler.running.len(),
        );

        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_prompt_tokens
            .fetch_add(prompt_len as u64, Ordering::Relaxed);

        let sampling_seed = sampling::rand_seed();
        let seq = Sequence {
            id: req.id.clone(),
            status: SequenceStatus::Waiting,
            created_at: Instant::now(),
            tokens: req.tokens,
            prompt_len,
            kv_caches: vec![None; self.num_layers],
            paged_kv: paged_kv::PagedKvSequence::new(self.paged_kv_allocator.block_size()),
            logits_processor: candle_transformers::generation::LogitsProcessor::new(
                sampling_seed,
                req.temperature,
                req.top_p,
            ),
            sampling_seed,
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            max_tokens: effective_max_tokens,
            eos_token_id: req.eos_token_id,
            repetition_penalty: req.repetition_penalty,
            repeat_last_n: 64,
            response_tx: req.response_tx,
        };

        let stream = TokenOutputStream::new(tokenizer);
        self.sequences.insert(req.id.clone(), seq);
        self.token_streams.insert(req.id.clone(), stream);
        self.scheduler.add(req.id);
    }

    // ─────────────────────────────────────────────────────────
    //  Cancellation detection
    // ─────────────────────────────────────────────────────────

    fn check_cancelled(&mut self) {
        let cancelled: Vec<String> = self
            .sequences
            .iter()
            .filter(|(_, seq)| seq.response_tx.is_closed())
            .map(|(id, _)| id.clone())
            .collect();

        for id in cancelled {
            warn!(id = %id, "Client disconnected, cancelling sequence");
            self.stats
                .cancelled_requests
                .fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(&id);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Step execution dispatch
    // ─────────────────────────────────────────────────────────

    fn execute_step(&mut self, output: SchedulerOutput) {
        if output.is_prefill {
            debug_assert_eq!(output.batch.len(), 1);
            let seq_id = &output.batch[0];
            self.step_prefill(seq_id.clone());
        } else if self.model.supports_batch_decode() && output.batch.len() > 1 {
            // True batched decode only when there are multiple sequences.
            // For a single sequence the sequential path is far cheaper: it
            // keeps the KV cache resident in the model and avoids the
            // extract→pad→stack→extract GPU-copy cycle that batch decode
            // performs every scheduling round.
            self.step_decode_batch(output.batch);
        } else {
            self.step_decode_sequential(output.batch);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Prefill
    // ─────────────────────────────────────────────────────────

    fn step_prefill(&mut self, seq_id: String) {
        let t0 = Instant::now();
        let queue_wait_us = self
            .sequences
            .get(&seq_id)
            .map(|seq| seq.created_at.elapsed().as_micros() as u64)
            .unwrap_or(0);

        let t_swap = Instant::now();
        if !self.swap_in(&seq_id) {
            self.send_error(&seq_id, "KV swap-in failed");
            return;
        }
        let mut swap_us = t_swap.elapsed().as_micros() as u64;

        let (input_ids, start_pos) = {
            let seq = self.sequences.get(&seq_id).unwrap();
            (seq.next_input_ids().to_vec(), seq.start_pos())
        };

        let prompt_len = input_ids.len();

        let t_forward = Instant::now();
        let logits = match self.model.forward_step(&input_ids, start_pos) {
            Ok(l) => l,
            Err(e) => {
                self.send_error(&seq_id, &format!("Prefill forward failed: {e}"));
                return;
            }
        };
        let forward_us = t_forward.elapsed().as_micros() as u64;

        let t_sampling = Instant::now();
        let row_greedy = self
            .sequences
            .get(&seq_id)
            .map_or(false, sampling::is_greedy);
        let sampled_token = {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            sampling::sample(&seq_id, seq, &logits, &mut self.sampling_buffers)
        };
        let sampling_us = t_sampling.elapsed().as_micros() as u64;
        let next_token = match sampled_token {
            Ok(t) => {
                if row_greedy {
                    self.stats
                        .total_sampling_row_greedy_tokens
                        .fetch_add(1, Ordering::Relaxed);
                } else {
                    self.stats
                        .total_sampling_non_greedy_tokens
                        .fetch_add(1, Ordering::Relaxed);
                }
                t
            }
            Err(e) => {
                self.stats
                    .total_sampling_failures
                    .fetch_add(1, Ordering::Relaxed);
                self.send_error(&seq_id, &format!("Sampling failed: {e}"));
                return;
            }
        };

        self.maybe_import_paged_kv_batch_past(
            &std::slice::from_ref(&seq_id),
            &[prompt_len],
            prompt_len,
        );

        let t_swap = Instant::now();
        self.swap_out(&seq_id);
        swap_us += t_swap.elapsed().as_micros() as u64;

        let prefill_us = t0.elapsed().as_micros() as u64;
        let ttft_us = self
            .sequences
            .get(&seq_id)
            .map(|seq| seq.created_at.elapsed().as_micros() as u64)
            .unwrap_or(prefill_us);
        self.stats
            .total_prefill_time_us
            .fetch_add(prefill_us, Ordering::Relaxed);
        self.stats
            .total_prefill_steps
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_queue_wait_time_us
            .fetch_add(queue_wait_us, Ordering::Relaxed);
        self.stats
            .total_time_to_first_token_us
            .fetch_add(ttft_us, Ordering::Relaxed);
        self.stats
            .total_prefill_forward_time_us
            .fetch_add(forward_us, Ordering::Relaxed);
        self.stats
            .total_prefill_sampling_time_us
            .fetch_add(sampling_us, Ordering::Relaxed);
        self.stats
            .total_prefill_swap_time_us
            .fetch_add(swap_us, Ordering::Relaxed);

        let prefill_tok_s = if prefill_us > 0 {
            (prompt_len as f64) / (prefill_us as f64 / 1_000_000.0)
        } else {
            0.0
        };

        {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            seq.tokens.push(next_token);
            seq.status = SequenceStatus::Running;
        }
        self.sync_paged_kv_for_sequence(&seq_id, prompt_len);

        if self.profile_enabled {
            info!(
                target: "crane_profile",
                stage = "prefill",
                id = %seq_id,
                prompt_len,
                queue_wait_us,
                ttft_us,
                total_us = prefill_us,
                forward_us,
                sampling_us,
                swap_us,
                first_token = next_token,
                kv_cache_bytes = self.tracked_kv_bytes,
                "profile prefill",
            );
        }

        info!(
            id = %seq_id,
            prompt_len,
            prefill_ms = prefill_us / 1000,
            prefill_tok_s = format!("{:.1}", prefill_tok_s),
            "Prefill complete, first token generated",
        );

        self.send_token(&seq_id, next_token);

        if self.sequences.get(&seq_id).unwrap().should_stop() {
            self.finish_sequence(&seq_id);
        } else {
            self.scheduler.promote_to_running(seq_id);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Batched decode
    // ─────────────────────────────────────────────────────────

    /// Decode step for all running sequences — TRUE BATCHED forward.
    ///
    /// Uses **lazy eviction**: when a sequence completes or is cancelled
    /// mid-loop, it stays in the batch tensor (wasting trivial compute)
    /// rather than triggering an expensive extract→re-setup cycle.
    fn step_decode_batch(&mut self, batch: Vec<String>) {
        let t0 = Instant::now();
        let mut forward_us = 0u64;
        let mut sampling_us = 0u64;
        let mut extract_us = 0u64;

        // Filter cancelled sequences.
        let cancelled: Vec<String> = batch
            .iter()
            .filter(|id| {
                self.sequences
                    .get(id.as_str())
                    .map_or(true, |s| s.response_tx.is_closed())
            })
            .cloned()
            .collect();
        for id in &cancelled {
            warn!(id = %id, "Client disconnected before decode batch");
            self.stats
                .cancelled_requests
                .fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(id);
        }
        let batch: Vec<String> = batch
            .into_iter()
            .filter(|id| !cancelled.contains(id))
            .collect();
        if batch.is_empty() {
            return;
        }

        let batch_size = batch.len();

        // Flush model's internal KV cache state.
        if let Some(ref prev_id) = self.active_seq_id.take() {
            if self.sequences.contains_key(prev_id) {
                let caches = self.model.get_kv_caches();
                if let Some(seq) = self.sequences.get_mut(prev_id) {
                    seq.kv_caches = caches;
                }
            }
            self.model.clear_kv_cache();
        }
        self.recount_kv_bytes();

        // Round 9: prefer the batched fast path. If the most recent extract
        // published a per-layer batched form for this exact batch, adopt it
        // directly (one slice_set per layer per K/V plane). If the batch
        // composition changed, attempt to re-gather batched data from the
        // page store. Only fall back to the per-row pad-stack path if neither
        // batched source is available.
        let pending_match =
            if let Some((prev_batch, extract)) = self.pending_batched_kv_extract.as_ref() {
                if prev_batch != &batch {
                    self.stats
                        .total_paged_kv_batched_setup_pending_batch_mismatch
                        .fetch_add(1, Ordering::Relaxed);
                    false
                } else {
                    // Validate per-row totals against current sequence token counts.
                    // If the engine ran a single-seq decode between the prev extract
                    // and this batched setup, the stored extract is stale.
                    let token_match = prev_batch.iter().enumerate().all(|(row, seq_id)| {
                        let expected_total = self
                            .sequences
                            .get(seq_id)
                            .map(|seq| seq.paged_kv.token_len());
                        match (expected_total, extract.per_row_totals.get(row).copied()) {
                            (Some(actual), Some(Some(stored))) => actual == stored,
                            (Some(0), Some(None)) => true,
                            _ => false,
                        }
                    });
                    if !token_match {
                        self.stats
                            .total_paged_kv_batched_setup_pending_token_mismatch
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    token_match
                }
            } else {
                false
            };
        let mut batched_setup: Option<paged_kv_runtime::BatchedKvExtract> = None;
        let mut batched_setup_via_regather = false;
        if pending_match {
            batched_setup = self
                .pending_batched_kv_extract
                .take()
                .map(|(_, extract)| extract);
        } else {
            // Drop stale pending; if we have non-empty per-seq kv_caches, prefer
            // a fresh page-store gather first. Prefill now imports prompt KV into
            // the page store before swap-out, so mixed old+new batches can still
            // use the batched setup path.
            self.pending_batched_kv_extract = None;
            match self.gather_batched_kv_for_batch(&batch) {
                Ok(Some(extract)) => {
                    batched_setup = Some(extract);
                    batched_setup_via_regather = true;
                }
                Ok(None) => {
                    let has_per_seq_caches = batch.iter().any(|id| {
                        self.sequences
                            .get(id)
                            .map(|s| s.kv_caches.iter().any(|c| c.is_some()))
                            .unwrap_or(false)
                    });
                    if has_per_seq_caches {
                        self.stats
                            .total_paged_kv_batched_setup_fallback_per_seq_cache
                            .fetch_add(1, Ordering::Relaxed);
                        let missing_paged_rows: Vec<usize> = batch
                            .iter()
                            .enumerate()
                            .filter_map(|(row, id)| {
                                let seq = self.sequences.get(id)?;
                                let has_cache = seq.kv_caches.iter().any(|cache| cache.is_some());
                                (!has_cache && seq.paged_kv.token_len() > 0).then_some(row)
                            })
                            .collect();
                        if !missing_paged_rows.is_empty() {
                            match self
                                .materialize_paged_kv_rows_for_batch(&batch, &missing_paged_rows)
                            {
                                Ok(true) => {}
                                Ok(false) => {
                                    error!(
                                        rows = ?missing_paged_rows,
                                        "mixed batch has rows without per-sequence KV caches and paged KV could not be materialized"
                                    );
                                    for row in missing_paged_rows {
                                        if let Some(seq_id) = batch.get(row) {
                                            self.send_error(
                                                seq_id,
                                                "Paged KV mixed-batch setup failed",
                                            );
                                        }
                                    }
                                    return;
                                }
                                Err(err) => {
                                    error!(
                                        rows = ?missing_paged_rows,
                                        error = %err,
                                        "mixed batch paged-KV materialization failed"
                                    );
                                    for row in missing_paged_rows {
                                        if let Some(seq_id) = batch.get(row) {
                                            self.send_error(
                                                seq_id,
                                                "Paged KV mixed-batch setup failed",
                                            );
                                        }
                                    }
                                    return;
                                }
                            }
                        }
                    } else {
                        self.stats
                            .total_paged_kv_batched_setup_fallback_regather_unavailable
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(err) => {
                    self.stats
                        .total_paged_kv_batched_setup_fallback_regather_error
                        .fetch_add(1, Ordering::Relaxed);
                    warn!(error = %err, "paged KV batched re-gather failed; falling back to per-row setup");
                }
            }
        }

        // Collect KV caches and setup batched decode.
        let kv_caches: Vec<Vec<Option<(Tensor, Tensor)>>> = batch
            .iter()
            .map(|id| self.sequences.get(id).unwrap().kv_caches.clone())
            .collect();
        self.maybe_validate_paged_kv_shadow_gather(&batch, &kv_caches);

        let (kv_lens, original_max_kv) = if let Some(extract) = batched_setup.as_ref() {
            let t_batched = Instant::now();
            let kv_lens_for_setup: Vec<usize> = extract
                .per_row_totals
                .iter()
                .map(|t| t.unwrap_or(0))
                .collect();
            let setup_result = self.model.setup_batch_decode_batched(
                &extract.per_layer,
                &kv_lens_for_setup,
                extract.max_total_len,
                self.decode_tokens_per_seq,
            );
            self.stats
                .total_paged_kv_batched_setup_us
                .fetch_add(t_batched.elapsed().as_micros() as u64, Ordering::Relaxed);
            if batched_setup_via_regather {
                self.stats
                    .total_paged_kv_batched_setup_regather
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                self.stats
                    .total_paged_kv_batched_setup_hits
                    .fetch_add(1, Ordering::Relaxed);
            }
            match setup_result {
                Ok(r) => r,
                Err(e) => {
                    error!("Batch decode batched setup failed: {e}");
                    for seq_id in &batch {
                        self.send_error(seq_id, &format!("Batch decode batched setup failed: {e}"));
                    }
                    return;
                }
            }
        } else {
            match self
                .model
                .setup_batch_decode(&kv_caches, self.decode_tokens_per_seq)
            {
                Ok(r) => r,
                Err(e) => {
                    error!("Batch decode setup failed: {e}");
                    for seq_id in &batch {
                        self.send_error(seq_id, &format!("Batch decode setup failed: {e}"));
                    }
                    return;
                }
            }
        };
        let setup_timings = self.model.last_batch_decode_setup_timings();
        if batched_setup.is_some() {
            self.stats
                .total_paged_kv_batched_setup_equal_length_layers
                .fetch_add(setup_timings.batched_equal_length_layers, Ordering::Relaxed);
            self.stats
                .total_paged_kv_batched_setup_ragged_layers
                .fetch_add(setup_timings.batched_ragged_layers, Ordering::Relaxed);
            self.stats
                .total_paged_kv_batched_setup_ragged_rows
                .fetch_add(setup_timings.batched_ragged_rows, Ordering::Relaxed);
        }
        #[cfg(feature = "cuda")]
        {
            let workspace_generation = self.model.batch_decode_workspace_generation();
            if workspace_generation != self.cuda_graph_workspace_generation {
                self.cuda_graph_decode_entries.clear();
                self.cuda_graph_decode_poisoned.clear();
                self.cuda_graph_workspace_generation = workspace_generation;
            }
        }
        self.maybe_import_paged_kv_batch_past(&batch, &kv_lens, original_max_kv);
        drop(kv_caches);

        // Now that setup_batch_decode has consumed the KV views (building its
        // own padded buffer), drop the per-sequence cache references.  With
        // zero-copy narrow views from get_kv_caches(), these still pin the
        // old pre-allocated buffers — clearing them here lets CUDA free that
        // VRAM before the decode loop allocates intermediates.
        for seq_id in &batch {
            if let Some(seq) = self.sequences.get_mut(seq_id) {
                seq.kv_caches = vec![None; self.num_layers];
            }
        }

        let t_setup = t0.elapsed();

        // Pre-build attention mask.
        let max_total_width = original_max_kv + self.decode_tokens_per_seq;
        // CUDA Graph cache reuse: optionally bucket the fixed-width K/V view
        // to the next power of two (matches `ensure_batch_decode_kv_workspace`'s
        // capacity rounding). Successive batch_decode calls within one
        // workspace generation that fall under the same power-of-two width
        // then share the same captured graph instead of recapturing per call.
        // Slots in `[mask_width, bucketed_cache_width)` are masked out so
        // attention math is identical to the eager `max_total_width` path.
        //
        // **Default ON**: bucketing collapses graph cache keys ~16× and lifts
        // greedy throughput by ~6-10% on the fixed-width path. Verified safe
        // (no failures) when capture is OFF in
        // docs/qwen3/benchmarks/qwen3_cuda_graph_ab_throughput_2026_04_30.md.
        // Set `CRANE_CUDA_GRAPH_DECODE_WIDTH_BUCKET=0` to disable.
        let bucketed_cache_width = if env_flag_default("CRANE_CUDA_GRAPH_DECODE_WIDTH_BUCKET", true)
        {
            max_total_width.max(1).next_power_of_two()
        } else {
            max_total_width
        };

        // CUDA Graph correctness gate: any captured decode graph holds device
        // pointers into the persistent reusable buffers (input_ids,
        // position_ids, append_offset, mask). When `ReusableTensorBuffer`
        // grows it allocates a fresh device tensor, but cudarc keeps the old
        // backing alive for the captured graph — replaying that graph would
        // then read stale memory that `copy_from` no longer updates. We
        // detect would-be reallocations BEFORE the round loop and invalidate
        // every cached graph (and the poison set) so the next round
        // recaptures against the new addresses.
        #[cfg(feature = "cuda")]
        if self.cuda_graph_decode.fixed_width_decode() {
            let needed_input_ids_len = batch_size;
            let needed_position_ids_len = batch_size;
            let needed_append_offset_len = 1usize;
            let needed_mask_len = batch_size * bucketed_cache_width;
            let realloc = self.cuda_graph_input_ids.capacity() < needed_input_ids_len
                || self.cuda_graph_position_ids.capacity() < needed_position_ids_len
                || self.cuda_graph_append_offset.capacity() < needed_append_offset_len
                || self.cuda_graph_mask.capacity() < needed_mask_len;
            if realloc && !self.cuda_graph_decode_entries.is_empty() {
                self.cuda_graph_decode_entries.clear();
                self.cuda_graph_decode_poisoned.clear();
            }
        }

        let t_mask = Instant::now();
        let full_mask =
            match self
                .model
                .build_batch_decode_mask(&kv_lens, original_max_kv, max_total_width)
            {
                Ok(m) => m,
                Err(e) => {
                    error!("Mask build failed: {e}");
                    self.model.clear_kv_cache();
                    return;
                }
            };
        let mut mask_us = t_mask.elapsed().as_micros() as u64;

        // Multi-round decode loop with lazy eviction.
        let mut total_tokens_this_step = 0u64;
        let mut rounds_done = 0usize;
        let mut alive = vec![true; batch.len()];
        let mut pending_finish: Vec<String> = Vec::new();
        let mut pending_cancel: Vec<String> = Vec::new();

        let mut positions: Vec<usize> = batch
            .iter()
            .map(|id| self.sequences.get(id).unwrap().start_pos())
            .collect();
        let mut position_ids: Vec<u32> =
            positions.iter().map(|&position| position as u32).collect();

        let mut last_tokens: Vec<u32> = batch
            .iter()
            .map(|id| *self.sequences.get(id).unwrap().tokens.last().unwrap())
            .collect();
        let mut next_input_ids_device: Option<Tensor> = None;
        let decode_device_token_input_enabled =
            env_flag_default("CRANE_DECODE_DEVICE_TOKEN_INPUT", true);

        for round in 0..self.decode_tokens_per_seq {
            if alive.iter().all(|a| !a) {
                break;
            }

            let mask_width = original_max_kv + round + 1;
            let paged_attention_context =
                self.maybe_build_paged_attention_context(&batch, &kv_lens, round, &alive);
            let fixed_width_round =
                self.cuda_graph_decode.fixed_width_decode() && paged_attention_context.is_none();
            let use_device_input_ids = decode_device_token_input_enabled
                && !fixed_width_round
                && alive.iter().all(|&is_alive| is_alive);
            let input_ids = if use_device_input_ids {
                if let Some(t) = next_input_ids_device.take() {
                    self.stats
                        .total_batch_decode_device_token_input_hits
                        .fetch_add(1, Ordering::Relaxed);
                    self.stats
                        .total_batch_decode_device_token_input_tokens
                        .fetch_add(batch_size as u64, Ordering::Relaxed);
                    t
                } else {
                    match crane_core::fused_ops::copy_from_slice_u32(
                        &last_tokens,
                        self.model.device(),
                    )
                    .and_then(|t| t.reshape((batch_size, 1)))
                    {
                        Ok(t) => t,
                        Err(e) => {
                            error!("Decode input_ids upload failed: {e}");
                            self.model.clear_kv_cache();
                            return;
                        }
                    }
                }
            } else if fixed_width_round {
                match self
                    .cuda_graph_input_ids
                    .upload_1d(&last_tokens, self.model.device())
                    .and_then(|t| t.reshape((batch_size, 1)))
                {
                    Ok(t) => t,
                    Err(e) => {
                        error!("Fixed-width decode input_ids upload failed: {e}");
                        self.model.clear_kv_cache();
                        return;
                    }
                }
            } else {
                match crane_core::fused_ops::copy_from_slice_u32(&last_tokens, self.model.device())
                    .and_then(|t| t.reshape((batch_size, 1)))
                {
                    Ok(t) => t,
                    Err(e) => {
                        error!("Decode input_ids upload failed: {e}");
                        self.model.clear_kv_cache();
                        return;
                    }
                }
            };
            let fixed_position_ids = if fixed_width_round {
                match self
                    .cuda_graph_position_ids
                    .upload_1d(&position_ids, self.model.device())
                {
                    Ok(t) => Some(t),
                    Err(e) => {
                        error!("Fixed-width decode position_ids upload failed: {e}");
                        self.model.clear_kv_cache();
                        return;
                    }
                }
            } else {
                None
            };
            let fixed_append_offset = if fixed_width_round {
                let append_offset = [original_max_kv as u32 + round as u32];
                match self
                    .cuda_graph_append_offset
                    .upload_1d(&append_offset, self.model.device())
                {
                    Ok(t) => Some(t),
                    Err(e) => {
                        error!("Fixed-width decode append-offset upload failed: {e}");
                        self.model.clear_kv_cache();
                        return;
                    }
                }
            } else {
                None
            };
            let mask_for_round = if fixed_width_round {
                let t_fixed_mask = Instant::now();
                let mask = match self.model.build_batch_decode_fixed_width_mask(
                    &kv_lens,
                    original_max_kv,
                    bucketed_cache_width,
                    mask_width,
                ) {
                    Ok(mask) => mask,
                    Err(e) => {
                        error!("Fixed-width decode mask build failed: {e}");
                        self.model.clear_kv_cache();
                        return;
                    }
                };
                let mask = match mask {
                    Some(mask) => match self.cuda_graph_mask.copy_from(&mask) {
                        Ok(mask) => Some(mask),
                        Err(e) => {
                            error!("Fixed-width decode mask stable-copy failed: {e}");
                            self.model.clear_kv_cache();
                            return;
                        }
                    },
                    None => None,
                };
                mask_us += t_fixed_mask.elapsed().as_micros() as u64;
                mask
            } else {
                match &full_mask {
                    Some(full) => full.narrow(3, 0, mask_width).ok(),
                    None => None,
                }
            };
            let active_rows = alive.iter().filter(|&&is_alive| is_alive).count() as u64;
            let graph_decision = self.cuda_graph_decode.classify_round(
                batch_size,
                self.model.device(),
                self.model.dtype(),
                mask_for_round.is_some(),
                fixed_width_round,
                paged_attention_context.is_some(),
                fixed_append_offset.is_some(),
            );
            self.record_cuda_graph_decode_decision(graph_decision, active_rows);
            #[cfg(feature = "cuda")]
            let graph_key = match graph_decision {
                cuda_graph::CudaGraphDecodeDecision::Ready { bucket }
                    if fixed_width_round && self.cuda_graph_decode.capture_runtime() =>
                {
                    let candidate = cuda_graph::CudaGraphDecodeKey {
                        bucket,
                        fixed_cache_width: bucketed_cache_width,
                        has_mask: mask_for_round.is_some(),
                    };
                    if self.cuda_graph_decode_poisoned.contains(&candidate) {
                        None
                    } else {
                        Some(candidate)
                    }
                }
                _ => None,
            };

            let t_forward = Instant::now();
            #[cfg(feature = "cuda")]
            let graph_no_reuse = env_flag("CRANE_CUDA_GRAPH_DECODE_NO_REUSE");
            // P4-A: when greedy sampling was captured into the decode graph,
            // its argmax kernel runs as part of `cuGraphLaunch`. The matching
            // DtoH happens after `logits_result` resolves, in the sampling
            // branch below.
            #[cfg(feature = "cuda")]
            let mut captured_sample_batch: Option<usize> = None;
            let logits_result = if fixed_width_round {
                #[cfg(feature = "cuda")]
                if let Some(key) = graph_key {
                    if graph_no_reuse {
                        // Diagnostic: drop any cached graph so each round re-captures.
                        // Used to bisect "capture bug" vs "reuse bug".
                        self.cuda_graph_decode_entries.remove(&key);
                    }
                    let replay = self.cuda_graph_decode_entries.get_mut(&key).map(|entry| {
                        entry.replays_used = entry.replays_used.saturating_add(1);
                        let sample_batch = entry.captured_sample_batch;
                        entry
                            .graph
                            .launch()
                            .map(|()| (entry.logits.clone(), sample_batch))
                    });
                    match replay {
                        Some(Ok((logits, sample_batch))) => {
                            self.record_cuda_graph_decode_replay(active_rows);
                            captured_sample_batch = sample_batch;
                            // If a replay cap is set and we've reached it, evict the entry
                            // so the next round re-captures. Workaround for stale-pointer
                            // drift on long replay chains; see qwen3_round5.
                            let cap = self.cuda_graph_decode.max_replays();
                            if cap > 0 {
                                if let Some(entry) = self.cuda_graph_decode_entries.get(&key) {
                                    if entry.replays_used >= cap {
                                        self.cuda_graph_decode_entries.remove(&key);
                                    }
                                }
                            }
                            Ok(logits)
                        }
                        Some(Err(e)) => {
                            warn!(?key, error = %e, "CUDA Graph decode replay failed; falling back to eager fixed-width decode");
                            self.cuda_graph_decode_entries.remove(&key);
                            self.cuda_graph_decode_poisoned.insert(key);
                            self.record_cuda_graph_decode_fallback(active_rows);
                            self.model.step_batch_decode_fixed_width_with_position_ids(
                                &input_ids,
                                &positions,
                                fixed_position_ids
                                    .as_ref()
                                    .expect("fixed-width round must have position ids"),
                                fixed_append_offset.as_ref(),
                                mask_for_round.as_ref(),
                                bucketed_cache_width,
                            )
                        }
                        None => {
                            self.record_cuda_graph_decode_capture_attempt();
                            match self.capture_cuda_graph_decode_round(
                                key,
                                &input_ids,
                                &positions,
                                fixed_position_ids
                                    .as_ref()
                                    .expect("fixed-width round must have position ids"),
                                fixed_append_offset.as_ref(),
                                mask_for_round.as_ref(),
                                bucketed_cache_width,
                            ) {
                                Ok((logits, sample_batch)) => {
                                    self.record_cuda_graph_decode_capture_success();
                                    captured_sample_batch = sample_batch;
                                    Ok(logits)
                                }
                                Err(e) => {
                                    warn!(?key, error = %e, "CUDA Graph decode capture failed; falling back to eager fixed-width decode");
                                    self.record_cuda_graph_decode_capture_failure();
                                    self.cuda_graph_decode_poisoned.insert(key);
                                    self.record_cuda_graph_decode_fallback(active_rows);
                                    self.model.step_batch_decode_fixed_width_with_position_ids(
                                        &input_ids,
                                        &positions,
                                        fixed_position_ids
                                            .as_ref()
                                            .expect("fixed-width round must have position ids"),
                                        fixed_append_offset.as_ref(),
                                        mask_for_round.as_ref(),
                                        bucketed_cache_width,
                                    )
                                }
                            }
                        }
                    }
                } else {
                    self.record_cuda_graph_decode_fallback(active_rows);
                    self.model.step_batch_decode_fixed_width_with_position_ids(
                        &input_ids,
                        &positions,
                        fixed_position_ids
                            .as_ref()
                            .expect("fixed-width round must have position ids"),
                        fixed_append_offset.as_ref(),
                        mask_for_round.as_ref(),
                        bucketed_cache_width,
                    )
                }

                #[cfg(not(feature = "cuda"))]
                {
                    self.record_cuda_graph_decode_fallback(active_rows);
                    self.model.step_batch_decode_fixed_width_with_position_ids(
                        &input_ids,
                        &positions,
                        fixed_position_ids
                            .as_ref()
                            .expect("fixed-width round must have position ids"),
                        fixed_append_offset.as_ref(),
                        mask_for_round.as_ref(),
                        bucketed_cache_width,
                    )
                }
            } else if let Some(context) = paged_attention_context.as_ref() {
                self.model.step_batch_decode_paged_attention(
                    &input_ids,
                    &positions,
                    mask_for_round.as_ref(),
                    Some((&kv_lens, original_max_kv)),
                    context,
                )
            } else {
                self.model.step_batch_decode(
                    &input_ids,
                    &positions,
                    mask_for_round.as_ref(),
                    Some((&kv_lens, original_max_kv)),
                )
            };
            let logits = match logits_result {
                Ok(l) => l,
                Err(e) => {
                    error!("Batched decode forward failed (round {round}): {e}");
                    for (i, seq_id) in batch.iter().enumerate() {
                        if alive[i] {
                            self.send_error(seq_id, &format!("Batched decode failed: {e}"));
                        }
                    }
                    self.model.clear_kv_cache();
                    return;
                }
            };
            if paged_attention_context.is_some() {
                let layer_hits = self.model.last_paged_attention_layer_hits() as u64;
                let layer_fallbacks = self.model.last_paged_attention_layer_fallbacks() as u64;
                self.stats
                    .total_paged_kv_attention_decode_calls
                    .fetch_add(1, Ordering::Relaxed);
                self.stats
                    .total_paged_kv_attention_decode_tokens
                    .fetch_add(active_rows, Ordering::Relaxed);
                self.stats
                    .total_paged_kv_attention_layer_hits
                    .fetch_add(layer_hits, Ordering::Relaxed);
                self.stats
                    .total_paged_kv_attention_layer_fallbacks
                    .fetch_add(layer_fallbacks, Ordering::Relaxed);
            }
            forward_us += t_forward.elapsed().as_micros() as u64;

            rounds_done += 1;

            let batch_greedy = batch.iter().enumerate().all(|(i, seq_id)| {
                !alive[i]
                    || self
                        .sequences
                        .get(seq_id)
                        .map_or(false, sampling::is_greedy)
            });
            let mut batch_greedy_device_tokens: Option<Tensor> = None;
            // P4-A: when sampling was captured into the decode graph and no row
            // needs a repetition penalty, we can read the tokens from the
            // engine-owned device buffer with a single DtoH that subsumes the
            // post-graph wait. Skips the out-of-graph argmax kernel launch.
            #[cfg(feature = "cuda")]
            let captured_tokens: Option<Vec<u32>> = if batch_greedy {
                if let Some(sample_batch) = captured_sample_batch {
                    let no_penalty = batch.iter().enumerate().all(|(i, seq_id)| {
                        !alive[i]
                            || self
                                .sequences
                                .get(seq_id)
                                .map_or(false, sampling::is_greedy_no_penalty)
                    });
                    if no_penalty && sample_batch == batch.len() {
                        let t_sampling = Instant::now();
                        let device = self.model.device().clone();
                        let dev = match &device {
                            candle_core::Device::Cuda(d) => d.clone(),
                            _ => unreachable!("captured_sample_batch implies CUDA device"),
                        };
                        match crane_core::fused_ops::gpu_argmax_batch_readback(
                            &dev,
                            &self.cuda_graph_sampling_buffers,
                            sample_batch,
                        ) {
                            Ok(tokens) => {
                                sampling_us += t_sampling.elapsed().as_micros() as u64;
                                let active_rows =
                                    alive.iter().filter(|&&is_alive| is_alive).count() as u64;
                                self.stats
                                    .total_sampling_batch_greedy_calls
                                    .fetch_add(1, Ordering::Relaxed);
                                self.stats
                                    .total_sampling_batch_greedy_tokens
                                    .fetch_add(active_rows, Ordering::Relaxed);
                                self.stats
                                    .total_sampling_batch_greedy_cuda_plain_calls
                                    .fetch_add(1, Ordering::Relaxed);
                                self.stats
                                    .total_sampling_batch_greedy_cuda_plain_tokens
                                    .fetch_add(active_rows, Ordering::Relaxed);
                                Some(tokens)
                            }
                            Err(e) => {
                                sampling_us += t_sampling.elapsed().as_micros() as u64;
                                warn!(error = %e, "in-graph argmax readback failed; falling back to out-of-graph sampling");
                                None
                            }
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            #[cfg(not(feature = "cuda"))]
            let captured_tokens: Option<Vec<u32>> = None;
            let batch_greedy_tokens = if let Some(tokens) = captured_tokens {
                Some(tokens)
            } else if batch_greedy {
                let t_sampling = Instant::now();
                let seq_refs: Vec<&Sequence> = batch
                    .iter()
                    .map(|seq_id| self.sequences.get(seq_id).unwrap())
                    .collect();
                match sampling::sample_batch_greedy(
                    &logits,
                    &seq_refs,
                    &alive,
                    &mut self.sampling_buffers,
                ) {
                    Ok(sample) if sample.tokens.len() == batch.len() => {
                        sampling_us += t_sampling.elapsed().as_micros() as u64;
                        let active_rows = alive.iter().filter(|&&is_alive| is_alive).count() as u64;
                        let mode = sample.mode;
                        batch_greedy_device_tokens = sample.device_tokens;
                        self.stats
                            .total_sampling_batch_greedy_calls
                            .fetch_add(1, Ordering::Relaxed);
                        self.stats
                            .total_sampling_batch_greedy_tokens
                            .fetch_add(active_rows, Ordering::Relaxed);
                        match mode {
                            sampling::BatchGreedyMode::CudaBf16NoPenalty => {
                                self.stats
                                    .total_sampling_batch_greedy_cuda_plain_calls
                                    .fetch_add(1, Ordering::Relaxed);
                                self.stats
                                    .total_sampling_batch_greedy_cuda_plain_tokens
                                    .fetch_add(active_rows, Ordering::Relaxed);
                            }
                            sampling::BatchGreedyMode::CudaBf16Penalty => {
                                self.stats
                                    .total_sampling_batch_greedy_cuda_penalty_calls
                                    .fetch_add(1, Ordering::Relaxed);
                                self.stats
                                    .total_sampling_batch_greedy_cuda_penalty_tokens
                                    .fetch_add(active_rows, Ordering::Relaxed);
                            }
                            sampling::BatchGreedyMode::TensorFallback => {
                                self.stats
                                    .total_sampling_batch_greedy_tensor_fallback_calls
                                    .fetch_add(1, Ordering::Relaxed);
                                self.stats
                                    .total_sampling_batch_greedy_tensor_fallback_tokens
                                    .fetch_add(active_rows, Ordering::Relaxed);
                            }
                        }
                        Some(sample.tokens)
                    }
                    Ok(sample) => {
                        sampling_us += t_sampling.elapsed().as_micros() as u64;
                        self.stats
                            .total_sampling_batch_greedy_fallbacks
                            .fetch_add(1, Ordering::Relaxed);
                        debug!(
                            expected = batch.len(),
                            actual = sample.tokens.len(),
                            "Batch greedy argmax returned unexpected token count; falling back"
                        );
                        None
                    }
                    Err(e) => {
                        sampling_us += t_sampling.elapsed().as_micros() as u64;
                        self.stats
                            .total_sampling_batch_greedy_fallbacks
                            .fetch_add(1, Ordering::Relaxed);
                        debug!("Batch greedy sampling unavailable: {e}; falling back");
                        None
                    }
                }
            } else {
                None
            };

            let active_non_greedy = if batch_greedy_tokens.is_none() {
                batch
                    .iter()
                    .enumerate()
                    .filter(|(i, seq_id)| {
                        alive[*i]
                            && self
                                .sequences
                                .get(seq_id.as_str())
                                .map_or(false, |seq| !sampling::is_greedy(seq))
                    })
                    .count() as u64
            } else {
                0
            };

            #[cfg(feature = "cuda")]
            let batch_non_greedy_tokens = if batch_greedy_tokens.is_none() && active_non_greedy > 0
            {
                let t_sampling = Instant::now();
                let seq_refs: Vec<&Sequence> = batch
                    .iter()
                    .map(|seq_id| self.sequences.get(seq_id).unwrap())
                    .collect();
                match sampling::sample_batch_non_greedy_cuda(
                    &logits,
                    &seq_refs,
                    &alive,
                    &mut self.sampling_buffers,
                ) {
                    Ok(Some(sample)) if sample.tokens.len() == batch.len() => {
                        sampling_us += t_sampling.elapsed().as_micros() as u64;
                        self.stats
                            .total_sampling_batch_non_greedy_calls
                            .fetch_add(1, Ordering::Relaxed);
                        self.stats
                            .total_sampling_batch_non_greedy_tokens
                            .fetch_add(active_non_greedy, Ordering::Relaxed);
                        match sample.mode {
                            sampling::BatchNonGreedyMode::CudaBf16TopKTopP => {
                                self.stats
                                    .total_sampling_batch_non_greedy_cuda_bf16_calls
                                    .fetch_add(1, Ordering::Relaxed);
                                self.stats
                                    .total_sampling_batch_non_greedy_cuda_bf16_tokens
                                    .fetch_add(active_non_greedy, Ordering::Relaxed);
                            }
                        }
                        Some(sample.tokens)
                    }
                    Ok(Some(sample)) => {
                        sampling_us += t_sampling.elapsed().as_micros() as u64;
                        debug!(
                            expected = batch.len(),
                            actual = sample.tokens.len(),
                            active_rows = sample.active_rows,
                            "Batch non-greedy CUDA sampler returned unexpected token count; falling back"
                        );
                        None
                    }
                    Ok(None) => {
                        sampling_us += t_sampling.elapsed().as_micros() as u64;
                        None
                    }
                    Err(e) => {
                        sampling_us += t_sampling.elapsed().as_micros() as u64;
                        debug!("Batch non-greedy CUDA sampling unavailable: {e}; falling back");
                        None
                    }
                }
            } else {
                None
            };

            #[cfg(not(feature = "cuda"))]
            let batch_non_greedy_tokens: Option<Vec<u32>> = None;

            let prepared_sampling_logits =
                if batch_greedy_tokens.is_none() && batch_non_greedy_tokens.is_none() {
                    let t_sampling = Instant::now();
                    match sampling::prepare_batch_sampling_logits(&logits) {
                        Ok(prepared) => {
                            sampling_us += t_sampling.elapsed().as_micros() as u64;
                            if active_non_greedy > 0 {
                                self.stats
                                    .total_sampling_batch_non_greedy_calls
                                    .fetch_add(1, Ordering::Relaxed);
                                self.stats
                                    .total_sampling_batch_non_greedy_tokens
                                    .fetch_add(active_non_greedy, Ordering::Relaxed);
                                self.stats
                                    .total_sampling_batch_non_greedy_fallback_calls
                                    .fetch_add(1, Ordering::Relaxed);
                                self.stats
                                    .total_sampling_batch_non_greedy_fallback_tokens
                                    .fetch_add(active_non_greedy, Ordering::Relaxed);
                            }
                            Some(prepared)
                        }
                        Err(e) => {
                            error!("Batch sampling logits preparation failed: {e}");
                            for (i, seq_id) in batch.iter().enumerate() {
                                if alive[i] {
                                    self.send_error(seq_id, &format!("Sampling prep failed: {e}"));
                                }
                            }
                            self.model.clear_kv_cache();
                            return;
                        }
                    }
                } else {
                    None
                };

            for (i, seq_id) in batch.iter().enumerate() {
                if !alive[i] {
                    continue;
                }

                let next_token = if let Some(tokens) = batch_greedy_tokens.as_ref() {
                    tokens[i]
                } else if let Some(tokens) = batch_non_greedy_tokens.as_ref() {
                    tokens[i]
                } else {
                    let prepared = prepared_sampling_logits
                        .as_ref()
                        .expect("prepared sampling logits should exist for row fallback");
                    let seq_logits = match prepared.narrow(0, i, 1).and_then(|l| l.squeeze(0)) {
                        Ok(l) => l,
                        Err(e) => {
                            self.send_error(seq_id, &format!("Logits extraction failed: {e}"));
                            alive[i] = false;
                            continue;
                        }
                    };

                    let row_greedy = self
                        .sequences
                        .get(seq_id)
                        .map_or(false, sampling::is_greedy);
                    let t_sampling = Instant::now();
                    let sampled_token = {
                        let seq = self.sequences.get_mut(seq_id).unwrap();
                        let trace =
                            std::env::var("CRANE_SAMPLE_TRACE").ok().as_deref() == Some("1");
                        sampling::sample_from_f32_logits(
                            seq_id,
                            seq,
                            &seq_logits,
                            &mut self.sampling_buffers,
                            trace,
                            t_sampling,
                        )
                    };
                    sampling_us += t_sampling.elapsed().as_micros() as u64;
                    match sampled_token {
                        Ok(t) => {
                            if row_greedy {
                                self.stats
                                    .total_sampling_row_greedy_tokens
                                    .fetch_add(1, Ordering::Relaxed);
                            } else {
                                self.stats
                                    .total_sampling_non_greedy_tokens
                                    .fetch_add(1, Ordering::Relaxed);
                            }
                            t
                        }
                        Err(e) => {
                            self.stats
                                .total_sampling_failures
                                .fetch_add(1, Ordering::Relaxed);
                            self.send_error(seq_id, &format!("Sampling failed: {e}"));
                            alive[i] = false;
                            continue;
                        }
                    }
                };

                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    seq.tokens.push(next_token);
                }
                last_tokens[i] = next_token;

                total_tokens_this_step += 1;
                self.stats
                    .total_decode_steps
                    .fetch_add(1, Ordering::Relaxed);

                self.send_token(seq_id, next_token);

                if self.sequences.get(seq_id).map_or(true, |s| s.should_stop()) {
                    alive[i] = false;
                    pending_finish.push(seq_id.clone());
                } else if self
                    .sequences
                    .get(seq_id)
                    .map_or(true, |s| s.response_tx.is_closed())
                {
                    warn!(id = %seq_id, "Client disconnected mid-batch-decode");
                    alive[i] = false;
                    pending_cancel.push(seq_id.clone());
                }
            }

            next_input_ids_device =
                if batch_greedy_tokens.is_some() && alive.iter().all(|&is_alive| is_alive) {
                    batch_greedy_device_tokens
                } else {
                    None
                };

            for p in positions.iter_mut() {
                *p += 1;
            }
            for p in position_ids.iter_mut() {
                *p += 1;
            }
            if round + 1 < self.decode_tokens_per_seq
                && alive.iter().any(|&is_alive| is_alive)
                && self.should_attempt_paged_attention_for_round(
                    &batch,
                    &kv_lens,
                    round + 1,
                    &alive,
                )
            {
                self.maybe_append_paged_kv_native(
                    &batch,
                    &kv_lens,
                    original_max_kv,
                    rounds_done,
                    &alive,
                );
            }
        }

        let native_append_synced_pages = self.maybe_append_paged_kv_native(
            &batch,
            &kv_lens,
            original_max_kv,
            rounds_done,
            &alive,
        );

        // Extract per-sequence KV caches.
        let mut extract_timings = BatchDecodeExtractTimings::default();
        let mut extract_state_replace_us = 0u64;
        if rounds_done > 0 {
            let t_extract = Instant::now();
            // M2 batched setup: when enabled, publish the page-gathered batched
            // KV form for the next setup; otherwise materialize per-row caches.
            let batched_setup_enabled = env_flag_default("CRANE_PAGED_KV_BATCHED_SETUP", false);
            let paged_extract = match self.maybe_extract_paged_kv_gather(
                &batch,
                &kv_lens,
                rounds_done,
                &alive,
            ) {
                Ok(extracted) => extracted,
                Err(err) => {
                    warn!(error = %err, "paged KV gather extraction failed; falling back to batch-buffer extraction");
                    None
                }
            };

            if let Some(extracted) = paged_extract {
                if !batched_setup_enabled {
                    // Round 8 path: materialize per-row from the gather output
                    // into each seq.kv_caches; do NOT publish pending batched.
                    let t_clear = Instant::now();
                    self.model.clear_kv_cache();
                    extract_timings.cache_clear_us = t_clear.elapsed().as_micros() as u64;
                    let t_state_replace = Instant::now();
                    let per_row = match extracted.materialize_per_row(self.num_layers) {
                        Ok(m) => m,
                        Err(e) => {
                            error!("paged KV per-row materialize failed: {e}");
                            self.model.clear_kv_cache();
                            return;
                        }
                    };
                    for (i, seq_id) in batch.iter().enumerate() {
                        if alive[i] {
                            if let Some(seq) = self.sequences.get_mut(seq_id) {
                                if i < per_row.len() {
                                    seq.kv_caches = per_row[i].clone();
                                }
                            }
                            let kv_token_len = self
                                .sequences
                                .get(seq_id)
                                .map(|seq| seq.tokens.len().saturating_sub(1));
                            if !native_append_synced_pages {
                                if let Some(kv_token_len) = kv_token_len {
                                    self.sync_paged_kv_for_sequence(seq_id, kv_token_len);
                                }
                            } else if let Some(kv_token_len) = kv_token_len {
                                debug_assert_eq!(kv_token_len, kv_lens[i] + rounds_done);
                            }
                        }
                    }
                    extract_state_replace_us = t_state_replace.elapsed().as_micros() as u64;
                    self.recount_kv_bytes();
                } else {
                    let t_clear = Instant::now();
                    self.model.clear_kv_cache();
                    extract_timings.cache_clear_us = t_clear.elapsed().as_micros() as u64;
                    let t_state_replace = Instant::now();
                    // Round 9: per-row materialization is gone. Clear seq.kv_caches
                    // for alive seqs and publish the batched form on the engine for
                    // the next setup_batch_decode to consume directly.
                    for (i, seq_id) in batch.iter().enumerate() {
                        if alive[i] {
                            if let Some(seq) = self.sequences.get_mut(seq_id) {
                                seq.kv_caches = vec![None; self.num_layers];
                            }
                            let kv_token_len = self
                                .sequences
                                .get(seq_id)
                                .map(|seq| seq.tokens.len().saturating_sub(1));
                            if !native_append_synced_pages {
                                if let Some(kv_token_len) = kv_token_len {
                                    self.sync_paged_kv_for_sequence(seq_id, kv_token_len);
                                }
                            } else if let Some(kv_token_len) = kv_token_len {
                                debug_assert_eq!(kv_token_len, kv_lens[i] + rounds_done);
                            }
                        }
                    }
                    self.pending_batched_kv_extract = Some((batch.clone(), extracted));
                    extract_state_replace_us = t_state_replace.elapsed().as_micros() as u64;
                    self.recount_kv_bytes();
                }
            } else {
                match self.model.extract_batch_kv_selective(
                    &kv_lens,
                    original_max_kv,
                    rounds_done,
                    &alive,
                ) {
                    Ok(extracted) => {
                        extract_timings = self.model.last_batch_decode_extract_timings();
                        let t_state_replace = Instant::now();
                        for (i, seq_id) in batch.iter().enumerate() {
                            if alive[i] {
                                if let Some(seq) = self.sequences.get_mut(seq_id) {
                                    if i < extracted.len() {
                                        seq.kv_caches = extracted[i].clone();
                                    }
                                }
                                let kv_token_len = self
                                    .sequences
                                    .get(seq_id)
                                    .map(|seq| seq.tokens.len().saturating_sub(1));
                                if !native_append_synced_pages {
                                    if let Some(kv_token_len) = kv_token_len {
                                        self.sync_paged_kv_for_sequence(seq_id, kv_token_len);
                                    }
                                } else if let Some(kv_token_len) = kv_token_len {
                                    debug_assert_eq!(kv_token_len, kv_lens[i] + rounds_done);
                                }
                            }
                        }
                        extract_state_replace_us = t_state_replace.elapsed().as_micros() as u64;
                        // KV caches changed for multiple sequences — recount.
                        self.recount_kv_bytes();
                    }
                    Err(e) => {
                        error!("Final KV extraction failed: {e}");
                        self.model.clear_kv_cache();
                        self.recount_kv_bytes();
                    }
                }
            }
            extract_us = t_extract.elapsed().as_micros() as u64;
        }

        for id in &pending_finish {
            self.finish_sequence(id);
        }
        for id in &pending_cancel {
            self.stats
                .cancelled_requests
                .fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(id);
        }

        let decode_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_decode_time_us
            .fetch_add(decode_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_calls
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_batch_decode_tokens
            .fetch_add(total_tokens_this_step, Ordering::Relaxed);
        self.stats
            .total_batch_decode_time_us
            .fetch_add(decode_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_setup_time_us
            .fetch_add(t_setup.as_micros() as u64, Ordering::Relaxed);
        self.stats
            .total_batch_decode_setup_kv_len_scan_time_us
            .fetch_add(setup_timings.kv_len_scan_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_setup_pad_stack_time_us
            .fetch_add(setup_timings.pad_stack_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_setup_contiguous_time_us
            .fetch_add(setup_timings.contiguous_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_setup_extra_room_time_us
            .fetch_add(setup_timings.extra_room_alloc_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_setup_cache_assign_time_us
            .fetch_add(setup_timings.cache_assign_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_mask_time_us
            .fetch_add(mask_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_forward_time_us
            .fetch_add(forward_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_sampling_time_us
            .fetch_add(sampling_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_extract_time_us
            .fetch_add(extract_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_extract_narrow_time_us
            .fetch_add(extract_timings.narrow_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_extract_contiguous_time_us
            .fetch_add(extract_timings.contiguous_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_extract_cache_clear_time_us
            .fetch_add(extract_timings.cache_clear_us, Ordering::Relaxed);
        self.stats
            .total_batch_decode_extract_state_replace_time_us
            .fetch_add(extract_state_replace_us, Ordering::Relaxed);

        if total_tokens_this_step > 0 {
            let tok_s = if decode_us > 0 {
                (total_tokens_this_step as f64) / (decode_us as f64 / 1_000_000.0)
            } else {
                0.0
            };
            if self.profile_enabled {
                info!(
                    target: "crane_profile",
                    stage = "decode_batch",
                    batch_size,
                    tokens = total_tokens_this_step,
                    rounds = rounds_done,
                    finished = pending_finish.len(),
                    total_us = decode_us,
                    setup_us = t_setup.as_micros() as u64,
                    qwen3_setup_total_us = setup_timings.total_us,
                    qwen3_setup_kv_len_scan_us = setup_timings.kv_len_scan_us,
                    qwen3_setup_pad_stack_us = setup_timings.pad_stack_us,
                    qwen3_setup_contiguous_us = setup_timings.contiguous_us,
                    qwen3_setup_extra_room_us = setup_timings.extra_room_alloc_us,
                    qwen3_setup_cache_assign_us = setup_timings.cache_assign_us,
                    qwen3_setup_layers = setup_timings.layers,
                    qwen3_setup_sequences = setup_timings.sequences,
                    mask_us,
                    forward_us,
                    sampling_us,
                    extract_us,
                    qwen3_extract_total_us = extract_timings.total_us,
                    qwen3_extract_narrow_us = extract_timings.narrow_us,
                    qwen3_extract_contiguous_us = extract_timings.contiguous_us,
                    qwen3_extract_cache_clear_us = extract_timings.cache_clear_us,
                    qwen3_extract_state_replace_us = extract_state_replace_us,
                    qwen3_extract_layers = extract_timings.layers,
                    qwen3_extract_sequences = extract_timings.sequences,
                    tok_s = format!("{:.1}", tok_s),
                    kv_cache_bytes = self.tracked_kv_bytes,
                    "profile batch decode",
                );
            }
            debug!(
                batch_size,
                tokens = total_tokens_this_step,
                rounds = rounds_done,
                finished = pending_finish.len(),
                setup_ms = t_setup.as_millis() as u64,
                decode_ms = decode_us / 1000,
                tok_s = format!("{:.1}", tok_s),
                "Batched decode step complete",
            );
        }

        self.drain_requests();
        self.check_cancelled();
    }

    // ─────────────────────────────────────────────────────────
    //  Sequential decode
    // ─────────────────────────────────────────────────────────

    /// Sequential decode for backends without batch decode support.
    fn step_decode_sequential(&mut self, batch: Vec<String>) {
        let t0 = Instant::now();
        let mut total_tokens: u64 = 0;
        let mut forward_us = 0u64;
        let mut sampling_us = 0u64;

        for seq_id in &batch {
            if self
                .sequences
                .get(seq_id)
                .map_or(true, |s| s.response_tx.is_closed())
            {
                self.stats
                    .cancelled_requests
                    .fetch_add(1, Ordering::Relaxed);
                self.cleanup_sequence(seq_id);
                continue;
            }

            if !self.swap_in(seq_id) {
                self.send_error(seq_id, "KV swap-in failed");
                continue;
            }

            for _round in 0..self.decode_tokens_per_seq {
                let (input_ids, start_pos) = {
                    let seq = match self.sequences.get(seq_id) {
                        Some(s) => s,
                        None => break,
                    };
                    (seq.next_input_ids().to_vec(), seq.start_pos())
                };

                let t_forward = Instant::now();
                let logits = match self.model.forward_step(&input_ids, start_pos) {
                    Ok(l) => l,
                    Err(e) => {
                        self.send_error(seq_id, &format!("Decode forward failed: {e}"));
                        break;
                    }
                };
                forward_us += t_forward.elapsed().as_micros() as u64;

                let t_sampling = Instant::now();
                let row_greedy = self
                    .sequences
                    .get(seq_id)
                    .map_or(false, sampling::is_greedy);
                let sampled_token = {
                    let seq = self.sequences.get_mut(seq_id).unwrap();
                    sampling::sample(seq_id, seq, &logits, &mut self.sampling_buffers)
                };
                sampling_us += t_sampling.elapsed().as_micros() as u64;
                let next_token = match sampled_token {
                    Ok(t) => {
                        if row_greedy {
                            self.stats
                                .total_sampling_row_greedy_tokens
                                .fetch_add(1, Ordering::Relaxed);
                        } else {
                            self.stats
                                .total_sampling_non_greedy_tokens
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        t
                    }
                    Err(e) => {
                        self.stats
                            .total_sampling_failures
                            .fetch_add(1, Ordering::Relaxed);
                        self.send_error(seq_id, &format!("Sampling failed: {e}"));
                        break;
                    }
                };

                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    seq.tokens.push(next_token);
                }
                let kv_token_len = self
                    .sequences
                    .get(seq_id)
                    .map(|seq| seq.tokens.len().saturating_sub(1));
                if let Some(kv_token_len) = kv_token_len {
                    self.sync_paged_kv_for_sequence(seq_id, kv_token_len);
                }

                total_tokens += 1;
                self.stats
                    .total_decode_steps
                    .fetch_add(1, Ordering::Relaxed);

                self.send_token(seq_id, next_token);

                if self.sequences.get(seq_id).map_or(true, |s| s.should_stop()) {
                    self.finish_sequence(seq_id);
                    break;
                }

                if self
                    .sequences
                    .get(seq_id)
                    .map_or(true, |s| s.response_tx.is_closed())
                {
                    warn!(id = %seq_id, "Client disconnected mid-decode");
                    self.stats
                        .cancelled_requests
                        .fetch_add(1, Ordering::Relaxed);
                    self.cleanup_sequence(seq_id);
                    break;
                }
            }

            self.swap_out(seq_id);
        }

        let decode_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_decode_time_us
            .fetch_add(decode_us, Ordering::Relaxed);
        self.stats
            .total_sequential_decode_calls
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_sequential_decode_tokens
            .fetch_add(total_tokens, Ordering::Relaxed);
        self.stats
            .total_sequential_decode_time_us
            .fetch_add(decode_us, Ordering::Relaxed);
        self.stats
            .total_sequential_decode_forward_time_us
            .fetch_add(forward_us, Ordering::Relaxed);
        self.stats
            .total_sequential_decode_sampling_time_us
            .fetch_add(sampling_us, Ordering::Relaxed);

        if total_tokens > 0 {
            let tok_s = if decode_us > 0 {
                (total_tokens as f64) / (decode_us as f64 / 1_000_000.0)
            } else {
                0.0
            };
            if self.profile_enabled {
                info!(
                    target: "crane_profile",
                    stage = "decode_sequential",
                    batch_size = batch.len(),
                    tokens = total_tokens,
                    total_us = decode_us,
                    forward_us,
                    sampling_us,
                    tok_s = format!("{:.1}", tok_s),
                    kv_cache_bytes = self.tracked_kv_bytes,
                    "profile sequential decode",
                );
            }
            debug!(
                tokens = total_tokens,
                decode_ms = decode_us / 1000,
                tok_s = format!("{:.1}", tok_s),
                "Sequential decode step complete",
            );
        }

        self.drain_requests();
        self.check_cancelled();
    }

    // ─────────────────────────────────────────────────────────
    //  KV cache management
    // ─────────────────────────────────────────────────────────

    fn swap_in(&mut self, seq_id: &str) -> bool {
        if self.active_seq_id.as_deref() == Some(seq_id) {
            return true;
        }

        if !self.model.supports_kv_swap() {
            if self.active_seq_id.as_deref() != Some(seq_id) {
                self.model.clear_kv_cache();
                self.active_seq_id = Some(seq_id.to_string());
            }
            return true;
        }

        // Save previous active sequence's KV cache from the model.
        if let Some(ref prev_id) = self.active_seq_id.clone() {
            let caches = self.model.get_kv_caches();
            if let Some(prev_seq) = self.sequences.get_mut(prev_id) {
                prev_seq.kv_caches = caches;
            }
        }

        let needs_paged_materialize = self.sequences.get(seq_id).is_some_and(|seq| {
            !seq.kv_caches.iter().any(|cache| cache.is_some()) && seq.paged_kv.token_len() > 0
        });
        if needs_paged_materialize {
            let batch = vec![seq_id.to_string()];
            match self.materialize_paged_kv_rows_for_batch(&batch, &[0]) {
                Ok(true) => {}
                Ok(false) => {
                    error!(id = %seq_id, "paged KV swap-in materialization was unavailable");
                    return false;
                }
                Err(err) => {
                    error!(id = %seq_id, error = %err, "paged KV swap-in materialization failed");
                    return false;
                }
            }
        }

        // Load new sequence's KV cache into the model.
        let caches = self
            .sequences
            .get(seq_id)
            .map(|s| s.kv_caches.clone())
            .unwrap_or_else(|| vec![None; self.num_layers]);
        self.model.set_kv_caches(caches);
        self.active_seq_id = Some(seq_id.to_string());

        self.recount_kv_bytes();
        self.stats
            .total_kv_swap_count
            .fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Mark that the model finished processing `seq_id` for this scheduling
    /// round.  Instead of extracting full KV caches (expensive GPU copies),
    /// we only update byte tracking from the model's internal state.
    /// The actual KV tensors remain in the model and are saved lazily by
    /// `swap_in` when switching to a different sequence.
    fn swap_out(&mut self, seq_id: &str) {
        if !self.model.supports_kv_swap() {
            return;
        }
        if self.active_seq_id.as_deref() != Some(seq_id) {
            return;
        }
        // Drop stale seq cache references (from the last swap_in) to free
        // GPU memory.  swap_in will extract fresh caches from the model
        // when switching to a different sequence.
        if let Some(seq) = self.sequences.get_mut(seq_id) {
            if seq.kv_caches.iter().any(|c| c.is_some()) {
                seq.kv_caches = vec![None; seq.kv_caches.len()];
            }
        }
        self.recount_kv_bytes();
    }
}

fn format_optional_bytes_engine(bytes: Option<u64>) -> String {
    bytes
        .map(format_bytes_engine)
        .unwrap_or_else(|| "unknown".to_string())
}
