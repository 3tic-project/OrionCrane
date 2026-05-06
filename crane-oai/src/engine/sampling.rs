//! Token sampling utilities.
//!
//! Includes:
//! - Repetition penalty (in-place, GPU-friendly)
//! - Gumbel-max sampling (GPU-native, no CPU round-trip)
//! - Top-k / top-p filtering
//!
//! All routines are designed for zero-copy GPU operation where possible.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tracing::debug;

use super::sequence::Sequence;

/// Fast-path predicate for deterministic greedy sampling with no history penalty.
pub fn is_greedy(seq: &Sequence) -> bool {
    matches!(seq.temperature, Some(t) if t <= 0.0)
}

#[cfg(feature = "cuda")]
fn env_flag_default(name: &str, default: bool) -> bool {
    match std::env::var(name).ok().as_deref() {
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON") => true,
        Some("0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF") => false,
        Some(_) => default,
        None => default,
    }
}

#[cfg(feature = "cuda")]
pub fn is_greedy_no_penalty(seq: &Sequence) -> bool {
    is_greedy(seq) && seq.repetition_penalty == 1.0
}

#[cfg(feature = "cuda")]
const MAX_FUSED_REPEAT_LAST_N: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BatchGreedyMode {
    CudaBf16NoPenalty,
    CudaBf16Penalty,
    TensorFallback,
}

#[derive(Debug, Clone)]
pub struct BatchGreedySample {
    pub tokens: Vec<u32>,
    pub mode: BatchGreedyMode,
    pub device_tokens: Option<Tensor>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BatchNonGreedyMode {
    CudaBf16TopKTopP,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct BatchNonGreedySample {
    pub tokens: Vec<u32>,
    pub mode: BatchNonGreedyMode,
    pub active_rows: usize,
}

/// Persistent buffers for GPU-side top-k/top-p sampling.
///
/// Reuses GPU allocations across steps to avoid repeated mallocs.
pub struct SamplingBuffers {
    pub topk_cumsum_mats: HashMap<usize, Tensor>,
    pub topk_shift_bufs: HashMap<usize, Tensor>,
    pub topk_shift_idxs: HashMap<usize, Tensor>,
    pub topk_neg_vecs: HashMap<usize, Tensor>,
    #[cfg(feature = "cuda")]
    pub batch_recent_token_ids: Vec<u32>,
    #[cfg(feature = "cuda")]
    pub batch_recent_lengths: Vec<u32>,
    #[cfg(feature = "cuda")]
    pub batch_penalties: Vec<f32>,
    #[cfg(feature = "cuda")]
    pub batch_temperatures: Vec<f32>,
    #[cfg(feature = "cuda")]
    pub batch_top_ks: Vec<u32>,
    #[cfg(feature = "cuda")]
    pub batch_top_ps: Vec<f32>,
    #[cfg(feature = "cuda")]
    pub batch_sampling_seeds: Vec<u64>,
    #[cfg(feature = "cuda")]
    pub batch_greedy_cuda_buffers: crane_core::fused_ops::BatchGreedyCudaBuffers,
    #[cfg(feature = "cuda")]
    pub batch_non_greedy_cuda_buffers: crane_core::fused_ops::BatchNonGreedyCudaBuffers,
}

impl SamplingBuffers {
    pub fn new() -> Self {
        Self {
            topk_cumsum_mats: HashMap::new(),
            topk_shift_bufs: HashMap::new(),
            topk_shift_idxs: HashMap::new(),
            topk_neg_vecs: HashMap::new(),
            #[cfg(feature = "cuda")]
            batch_recent_token_ids: Vec::new(),
            #[cfg(feature = "cuda")]
            batch_recent_lengths: Vec::new(),
            #[cfg(feature = "cuda")]
            batch_penalties: Vec::new(),
            #[cfg(feature = "cuda")]
            batch_temperatures: Vec::new(),
            #[cfg(feature = "cuda")]
            batch_top_ks: Vec::new(),
            #[cfg(feature = "cuda")]
            batch_top_ps: Vec::new(),
            #[cfg(feature = "cuda")]
            batch_sampling_seeds: Vec::new(),
            #[cfg(feature = "cuda")]
            batch_greedy_cuda_buffers: crane_core::fused_ops::BatchGreedyCudaBuffers::new(),
            #[cfg(feature = "cuda")]
            batch_non_greedy_cuda_buffers: crane_core::fused_ops::BatchNonGreedyCudaBuffers::new(),
        }
    }

    pub fn get_topk_neg_vec(&mut self, k: usize, device: &Device) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_neg_vecs.get(&k) {
            if t.device().same_device(device) {
                return Ok(t.clone());
            }
        }
        let t = Tensor::full(-1e9f32, k, device)?;
        self.topk_neg_vecs.insert(k, t.clone());
        Ok(t)
    }

    pub fn get_topk_shift_idx(&mut self, k: usize, device: &Device) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_shift_idxs.get(&k) {
            if t.device().same_device(device) {
                return Ok(t.clone());
            }
        }
        if k <= 1 {
            candle_core::bail!("get_topk_shift_idx expects k > 1")
        }
        let t = Tensor::arange(1u32, k as u32, device)?;
        self.topk_shift_idxs.insert(k, t.clone());
        Ok(t)
    }

    pub fn get_topk_shift_buf(
        &mut self,
        k: usize,
        device: &Device,
        dtype: DType,
    ) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_shift_bufs.get(&k) {
            if t.device().same_device(device) && t.dtype() == dtype {
                return Ok(t.clone());
            }
        }
        let t = Tensor::zeros(k, dtype, device)?;
        self.topk_shift_bufs.insert(k, t.clone());
        Ok(t)
    }

    pub fn get_topk_cumsum_mat(
        &mut self,
        k: usize,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_cumsum_mats.get(&k) {
            if t.device().same_device(device) {
                return Ok(t.clone());
            }
        }
        let mut data = Vec::with_capacity(k * k);
        for row in 0..k {
            for col in 0..k {
                data.push(if row <= col { 1f32 } else { 0f32 });
            }
        }
        let t = Tensor::from_vec(data, (k, k), device)?;
        self.topk_cumsum_mats.insert(k, t.clone());
        Ok(t)
    }
}

#[cfg(feature = "cuda")]
fn derive_row_seed(seq: &Sequence, row: usize) -> u64 {
    let mut hash = seq.sampling_seed ^ 0x9e37_79b9_7f4a_7c15u64;
    for byte in seq.id.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash ^= (seq.tokens.len() as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash ^= (seq.num_generated() as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^= (row as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);
    hash
}

#[cfg(feature = "cuda")]
pub fn sample_batch_non_greedy_cuda(
    logits: &Tensor,
    seqs: &[&Sequence],
    alive: &[bool],
    buffers: &mut SamplingBuffers,
) -> Result<Option<BatchNonGreedySample>> {
    if seqs.len() != alive.len() {
        anyhow::bail!("sample_batch_non_greedy_cuda: seq/alive length mismatch")
    }
    if !logits.device().is_cuda() || logits.dtype() != DType::BF16 {
        return Ok(None);
    }
    if !env_flag_default("CRANE_BATCH_NON_GREEDY_SAMPLING", true) {
        return Ok(None);
    }
    let active_rows = alive.iter().filter(|&&is_alive| is_alive).count();
    if active_rows == 0 {
        return Ok(None);
    }

    let mut max_recent = 0usize;
    buffers.batch_temperatures.resize(seqs.len(), 0.0);
    buffers.batch_top_ks.resize(seqs.len(), 0);
    buffers.batch_top_ps.resize(seqs.len(), 1.0);
    buffers.batch_sampling_seeds.resize(seqs.len(), 0);
    buffers.batch_penalties.resize(seqs.len(), 1.0);
    buffers.batch_recent_lengths.resize(seqs.len(), 0);
    buffers.batch_penalties.fill(1.0);
    buffers.batch_recent_lengths.fill(0);

    for (row, (seq, &is_alive)) in seqs.iter().zip(alive.iter()).enumerate() {
        if !is_alive {
            continue;
        }

        let temperature = seq.temperature.unwrap_or(1.0);
        if temperature <= 0.0 {
            buffers.batch_temperatures[row] = 0.0;
            buffers.batch_top_ks[row] = 1;
            buffers.batch_top_ps[row] = 1.0;
        } else {
            let Some(top_k) = seq.top_k else {
                return Ok(None);
            };
            if top_k == 0 || top_k > 64 {
                return Ok(None);
            }
            let top_p = seq.top_p.unwrap_or(1.0);
            if !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
                return Ok(None);
            }
            buffers.batch_temperatures[row] = temperature as f32;
            buffers.batch_top_ks[row] = top_k as u32;
            buffers.batch_top_ps[row] = top_p as f32;
        }
        buffers.batch_sampling_seeds[row] = derive_row_seed(seq, row);
        buffers.batch_penalties[row] = seq.repetition_penalty;

        if seq.repetition_penalty != 1.0 {
            let recent_len = seq.tokens.len().min(seq.repeat_last_n);
            if recent_len > MAX_FUSED_REPEAT_LAST_N {
                return Ok(None);
            }
            max_recent = max_recent.max(recent_len);
        }
    }

    if max_recent > 0 {
        buffers
            .batch_recent_token_ids
            .resize(seqs.len() * max_recent, 0);
        buffers.batch_recent_token_ids.fill(0);
        for (row, (seq, &is_alive)) in seqs.iter().zip(alive.iter()).enumerate() {
            if !is_alive || seq.repetition_penalty == 1.0 {
                continue;
            }
            let recent_len = seq.tokens.len().min(seq.repeat_last_n).min(max_recent);
            let start = seq.tokens.len().saturating_sub(recent_len);
            buffers.batch_recent_lengths[row] = recent_len as u32;
            buffers.batch_recent_token_ids[row * max_recent..row * max_recent + recent_len]
                .copy_from_slice(&seq.tokens[start..]);
        }
    } else {
        buffers.batch_recent_token_ids.clear();
    }

    let tokens = crane_core::fused_ops::gpu_sample_topk_topp_batch_bf16_cached(
        logits,
        &buffers.batch_temperatures,
        &buffers.batch_top_ks,
        &buffers.batch_top_ps,
        &buffers.batch_sampling_seeds,
        &buffers.batch_recent_token_ids,
        &buffers.batch_recent_lengths,
        &buffers.batch_penalties,
        max_recent,
        &mut buffers.batch_non_greedy_cuda_buffers,
    )
    .map_err(anyhow::Error::from)?;

    Ok(Some(BatchNonGreedySample {
        tokens,
        mode: BatchNonGreedyMode::CudaBf16TopKTopP,
        active_rows,
    }))
}

/// Batch deterministic sampling for active greedy rows.
///
/// The no-penalty BF16 CUDA case uses Crane's custom batch argmax. If any
/// active row needs repetition penalty, this mirrors the existing row path by
/// converting logits to F32, applying per-row penalties, then doing one batched
/// argmax and one compact DtoH copy.
pub fn sample_batch_greedy(
    logits: &Tensor,
    seqs: &[&Sequence],
    alive: &[bool],
    buffers: &mut SamplingBuffers,
) -> Result<BatchGreedySample> {
    #[cfg(not(feature = "cuda"))]
    let _ = buffers;

    if seqs.len() != alive.len() {
        anyhow::bail!("sample_batch_greedy: seq/alive length mismatch")
    }
    if seqs
        .iter()
        .zip(alive.iter())
        .any(|(seq, &is_alive)| is_alive && !is_greedy(seq))
    {
        anyhow::bail!("sample_batch_greedy called with non-greedy active row")
    }

    #[cfg(feature = "cuda")]
    {
        let all_active_no_penalty = seqs
            .iter()
            .zip(alive.iter())
            .all(|(seq, &is_alive)| !is_alive || is_greedy_no_penalty(seq));
        if all_active_no_penalty && logits.device().is_cuda() && logits.dtype() == DType::BF16 {
            let tokens = crane_core::fused_ops::gpu_argmax_batch_cached(
                logits,
                &mut buffers.batch_greedy_cuda_buffers,
            )
            .map_err(anyhow::Error::from)?;
            let device_tokens = buffers
                .batch_greedy_cuda_buffers
                .output_tokens_tensor_from(logits, tokens.len())
                .map(Some)
                .unwrap_or_else(|err| {
                    debug!(error = %err, "greedy CUDA tokens cannot be exposed as input_ids tensor");
                    None
                });
            return Ok(BatchGreedySample {
                tokens,
                mode: BatchGreedyMode::CudaBf16NoPenalty,
                device_tokens,
            });
        }

        if logits.device().is_cuda() && logits.dtype() == DType::BF16 {
            let max_recent = seqs
                .iter()
                .zip(alive.iter())
                .filter_map(|(seq, &is_alive)| {
                    if is_alive && seq.repetition_penalty != 1.0 {
                        Some(seq.tokens.len().min(seq.repeat_last_n))
                    } else {
                        None
                    }
                })
                .max()
                .unwrap_or(0);

            if max_recent > 0 && max_recent <= MAX_FUSED_REPEAT_LAST_N {
                buffers
                    .batch_recent_token_ids
                    .resize(seqs.len() * max_recent, 0);
                buffers.batch_recent_token_ids.fill(0);
                buffers.batch_recent_lengths.resize(seqs.len(), 0);
                buffers.batch_recent_lengths.fill(0);
                buffers.batch_penalties.resize(seqs.len(), 1.0);
                buffers.batch_penalties.fill(1.0);

                for (row, (seq, &is_alive)) in seqs.iter().zip(alive.iter()).enumerate() {
                    if !is_alive || seq.repetition_penalty == 1.0 {
                        continue;
                    }
                    let recent_len = seq.tokens.len().min(seq.repeat_last_n).min(max_recent);
                    let start = seq.tokens.len().saturating_sub(recent_len);
                    buffers.batch_recent_lengths[row] = recent_len as u32;
                    buffers.batch_penalties[row] = seq.repetition_penalty;
                    buffers.batch_recent_token_ids[row * max_recent..row * max_recent + recent_len]
                        .copy_from_slice(&seq.tokens[start..]);
                }

                match crane_core::fused_ops::gpu_argmax_batch_with_repetition_penalty_cached(
                    logits,
                    &buffers.batch_recent_token_ids,
                    &buffers.batch_recent_lengths,
                    &buffers.batch_penalties,
                    max_recent,
                    &mut buffers.batch_greedy_cuda_buffers,
                ) {
                    Ok(tokens) => {
                        let device_tokens = buffers
                            .batch_greedy_cuda_buffers
                            .output_tokens_tensor_from(logits, tokens.len())
                            .map(Some)
                            .unwrap_or_else(|err| {
                                debug!(error = %err, "penalty greedy CUDA tokens cannot be exposed as input_ids tensor");
                                None
                            });
                        return Ok(BatchGreedySample {
                            tokens,
                            mode: BatchGreedyMode::CudaBf16Penalty,
                            device_tokens,
                        });
                    }
                    Err(err) => {
                        debug!("BF16 penalty batch argmax unavailable: {err}; falling back");
                    }
                }
            }
        }
    }

    let logits = if logits.rank() == 3 {
        logits.squeeze(1)?
    } else {
        logits.clone()
    }
    .to_dtype(DType::F32)?;

    for (row, (seq, &is_alive)) in seqs.iter().zip(alive.iter()).enumerate() {
        if !is_alive || seq.repetition_penalty == 1.0 {
            continue;
        }
        let start_at = seq.tokens.len().saturating_sub(seq.repeat_last_n);
        let row_logits = logits.narrow(0, row, 1)?.squeeze(0)?;
        apply_repeat_penalty_inplace(&row_logits, seq.repetition_penalty, &seq.tokens[start_at..])
            .map_err(anyhow::Error::from)?;
    }

    let tokens = logits.argmax(candle_core::D::Minus1)?;
    Ok(BatchGreedySample {
        tokens: tokens.to_vec1::<u32>()?,
        mode: BatchGreedyMode::TensorFallback,
        device_tokens: None,
    })
}

/// Sample a token from logits for a specific sequence.
///
/// Supports:
/// - Greedy decoding (temperature ≤ 0)
/// - Top-k filtering with GPU-native Gumbel-max sampling
/// - Top-p (nucleus) filtering with cumulative softmax masking
/// - CPU fallback via `LogitsProcessor` when needed
pub fn sample(
    seq_id: &str,
    seq: &mut Sequence,
    logits: &Tensor,
    buffers: &mut SamplingBuffers,
) -> Result<u32> {
    let trace = std::env::var("CRANE_SAMPLE_TRACE").ok().as_deref() == Some("1");
    let t0 = Instant::now();

    // ── Fast path: greedy + no repetition penalty ──────────────────────
    // Skip the bf16→f32 conversion and use GPU argmax directly on bf16
    // logits.  Saves one dtype-conversion kernel + less DtoH.
    #[cfg(feature = "cuda")]
    {
        if is_greedy_no_penalty(seq) && logits.device().is_cuda() {
            let flat = logits.squeeze(0)?.squeeze(0)?;
            let token = crane_core::fused_ops::gpu_argmax(&flat)?;
            if trace {
                let t_done = Instant::now();
                tracing::debug!(
                    id = %seq_id,
                    total_us = t_done.duration_since(t0).as_micros() as u64,
                    "sample(gpu_argmax_fast)"
                );
            }
            return Ok(token);
        }
    }

    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    sample_from_f32_logits(seq_id, seq, &logits, buffers, trace, t0)
}

/// Prepare batched logits once for row-wise sampling fallback.
pub fn prepare_batch_sampling_logits(logits: &Tensor) -> Result<Tensor> {
    let logits = if logits.rank() == 3 {
        logits.squeeze(1)?
    } else {
        logits.clone()
    };
    Ok(logits.to_dtype(DType::F32)?)
}

/// Sample from a single row of already-prepared F32 logits.
pub fn sample_from_f32_logits(
    seq_id: &str,
    seq: &mut Sequence,
    logits: &Tensor,
    buffers: &mut SamplingBuffers,
    trace: bool,
    t0: Instant,
) -> Result<u32> {
    if logits.rank() != 1 || logits.dtype() != DType::F32 {
        anyhow::bail!(
            "sample_from_f32_logits expects a 1D F32 tensor, got rank={} dtype={:?}",
            logits.rank(),
            logits.dtype()
        );
    }
    let t_after_prep = Instant::now();
    let greedy = is_greedy(seq);

    if seq.repetition_penalty != 1.0 {
        let start_at = seq.tokens.len().saturating_sub(seq.repeat_last_n);
        apply_repeat_penalty_inplace(&logits, seq.repetition_penalty, &seq.tokens[start_at..])
            .map_err(anyhow::Error::from)?;
    }
    let t_after_rep = Instant::now();

    if greedy {
        return Ok(logits.argmax(0)?.to_scalar::<u32>()?);
    }

    if logits.device().is_cuda() {
        let top_p = seq.top_p.unwrap_or(1.0);
        let top_p_active = top_p > 0.0 && top_p < 1.0;
        let vocab = logits.dim(0)?;
        let temperature = seq.temperature.unwrap_or(1.0);

        let mut top_k = seq.top_k.unwrap_or(0);
        if top_k == 0 && top_p_active {
            // For large vocabularies (>64 K tokens) where top_k was NOT
            // explicitly requested, avoid the expensive GPU topk kernel.
            // Fall back to CPU LogitsProcessor which handles temperature +
            // top-p natively and only needs a ~600 KB DtoH copy.
            // Set CRANE_FORCE_GPU_TOPK=1 to override this heuristic.
            if vocab > 65536 && std::env::var("CRANE_FORCE_GPU_TOPK").ok().as_deref() != Some("1") {
                let next_token = seq.logits_processor.sample(&logits)?;
                if trace {
                    let t_done = Instant::now();
                    debug!(
                        id = %seq_id,
                        vocab,
                        top_p = ?seq.top_p,
                        temp = ?seq.temperature,
                        total_us = t_done.duration_since(t0).as_micros() as u64,
                        "sample(cpu_logits_processor)"
                    );
                }
                return Ok(next_token);
            }
            top_k = std::env::var("CRANE_TOPP_FALLBACK_TOPK")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(64);
        }
        top_k = top_k.min(64).min(vocab);

        if top_k > 0 && top_k < vocab {
            let topk_idx =
                crane_core::fused_ops::topk_indices(&logits, top_k).map_err(anyhow::Error::from)?;
            let topk_logits = logits.gather(&topk_idx, candle_core::D::Minus1)?;
            let t_after_topk = Instant::now();

            if std::env::var("CRANE_TOPK_SAMPLE_ON_CPU").ok().as_deref() == Some("1") {
                let idx_cpu = topk_idx.to_vec1::<u32>()?;
                let logits_cpu = topk_logits.to_vec1::<f32>()?;
                let cpu_logits = Tensor::from_vec(logits_cpu, top_k, &Device::Cpu)?;

                let pos = seq.logits_processor.sample(&cpu_logits)?;
                let token = idx_cpu
                    .get(pos as usize)
                    .copied()
                    .unwrap_or_else(|| idx_cpu[0]);

                if trace {
                    let t_done = Instant::now();
                    debug!(
                        id = %seq_id,
                        top_k,
                        top_p = ?seq.top_p,
                        temp = ?seq.temperature,
                        prep_us = t_after_prep.duration_since(t0).as_micros() as u64,
                        rep_us = t_after_rep.duration_since(t_after_prep).as_micros() as u64,
                        topk_us = t_after_topk.duration_since(t_after_rep).as_micros() as u64,
                        total_us = t_done.duration_since(t0).as_micros() as u64,
                        "sample(topk->cpu)"
                    );
                }
                return Ok(token);
            }

            if top_p_active {
                let scaled = (&topk_logits / temperature)?;
                let probs = candle_nn::ops::softmax_last_dim(&scaled)?;
                let cumsum_mat = buffers.get_topk_cumsum_mat(top_k, logits.device())?;
                let cumsum = probs
                    .reshape((1, top_k))?
                    .matmul(&cumsum_mat)?
                    .reshape(top_k)?;
                let mask_le = cumsum.le(top_p)?;

                let shift = buffers.get_topk_shift_buf(top_k, logits.device(), mask_le.dtype())?;
                shift.zero_set()?;
                if top_k > 1 {
                    let idx = buffers.get_topk_shift_idx(top_k, logits.device())?;
                    let src = mask_le.narrow(candle_core::D::Minus1, 0, top_k - 1)?;
                    shift.scatter_set(&idx, &src, candle_core::D::Minus1)?;
                }
                let mask = (&mask_le + &shift)?.gt(0f64)?;

                let neg = buffers.get_topk_neg_vec(top_k, logits.device())?;
                let masked = mask.where_cond(&topk_logits, &neg)?;
                let mut pos = sample_gumbel_max_idx(&masked, temperature)?;
                if pos.rank() == 0 {
                    pos = pos.unsqueeze(0)?;
                }
                let token = topk_idx.gather(&pos, candle_core::D::Minus1)?;
                return Ok(token.squeeze(0)?.to_scalar::<u32>()?);
            }

            let mut pos = sample_gumbel_max_idx(&topk_logits, temperature)?;
            if pos.rank() == 0 {
                pos = pos.unsqueeze(0)?;
            }
            let token = topk_idx.gather(&pos, candle_core::D::Minus1)?;
            return Ok(token.squeeze(0)?.to_scalar::<u32>()?);
        }
    }

    let top_p = seq.top_p.unwrap_or(1.0);
    if top_p <= 0.0 || top_p >= 1.0 {
        let temperature = seq.temperature.unwrap_or(1.0);
        let idx = sample_gumbel_max_idx(&logits, temperature).map_err(anyhow::Error::from)?;
        return Ok(idx.to_scalar::<u32>()?);
    }

    let next_token = seq.logits_processor.sample(&logits)?;
    Ok(next_token)
}

/// Gumbel-max trick for GPU-native categorical sampling.
pub fn sample_gumbel_max_idx(logits: &Tensor, temperature: f64) -> candle_core::Result<Tensor> {
    if temperature <= 0.0 {
        return logits.argmax(candle_core::D::Minus1);
    }
    let minus_g = logits.rand_like(1e-7, 0.999)?.log()?.neg()?.log()?;
    if temperature == 1.0 {
        (logits - minus_g)?.argmax(candle_core::D::Minus1)
    } else {
        ((logits / temperature)? - minus_g)?.argmax(candle_core::D::Minus1)
    }
}

/// Apply repetition penalty in-place (GPU-friendly scatter/gather).
pub fn apply_repeat_penalty_inplace(
    logits: &Tensor,
    penalty: f32,
    context: &[u32],
) -> candle_core::Result<()> {
    if context.is_empty() {
        return Ok(());
    }

    let mut unique: HashSet<u32> = HashSet::with_capacity(context.len());
    for &t in context {
        unique.insert(t);
    }
    if unique.is_empty() {
        return Ok(());
    }
    let mut token_ids: Vec<u32> = unique.into_iter().collect();
    token_ids.sort_unstable();

    let idx = Tensor::new(token_ids.as_slice(), logits.device())?;
    let selected = logits.gather(&idx, candle_core::D::Minus1)?;
    let mask = selected.ge(0f64)?;
    let on_true = (&selected / penalty as f64)?;
    let on_false = (&selected * penalty as f64)?;
    let updated = mask.where_cond(&on_true, &on_false)?;
    logits.scatter_set(&idx, &updated, candle_core::D::Minus1)
}

/// Generate a random seed from system time.
pub fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::sequence::SequenceStatus;
    use candle_transformers::generation::LogitsProcessor;
    use std::time::Instant;
    use tokio::sync::mpsc;

    fn make_greedy_seq(tokens: Vec<u32>, repetition_penalty: f32) -> Sequence {
        let (tx, _rx) = mpsc::unbounded_channel();
        Sequence {
            id: "test-seq".into(),
            status: SequenceStatus::Running,
            created_at: Instant::now(),
            prompt_len: tokens.len(),
            tokens,
            kv_caches: vec![],
            paged_kv: crate::engine::paged_kv::PagedKvSequence::default(),
            logits_processor: LogitsProcessor::new(42, Some(0.0), Some(1.0)),
            sampling_seed: 42,
            temperature: Some(0.0),
            top_p: Some(1.0),
            top_k: None,
            max_tokens: 16,
            eos_token_id: vec![0],
            repetition_penalty,
            repeat_last_n: 64,
            response_tx: tx,
        }
    }

    #[test]
    fn rand_seed_is_nonzero() {
        let seed = rand_seed();
        assert_ne!(seed, 0);
    }

    #[test]
    fn rand_seed_varies_across_calls() {
        let s1 = rand_seed();
        // Spin a bit to ensure time advances.
        std::thread::sleep(std::time::Duration::from_millis(1));
        let s2 = rand_seed();
        assert_ne!(s1, s2);
    }

    // ── SamplingBuffers tests ──

    #[test]
    fn sampling_buffers_new_is_empty() {
        let b = SamplingBuffers::new();
        assert!(b.topk_cumsum_mats.is_empty());
        assert!(b.topk_shift_bufs.is_empty());
        assert!(b.topk_shift_idxs.is_empty());
        assert!(b.topk_neg_vecs.is_empty());
        #[cfg(feature = "cuda")]
        {
            assert!(b.batch_recent_token_ids.is_empty());
            assert!(b.batch_recent_lengths.is_empty());
            assert!(b.batch_penalties.is_empty());
            assert!(b.batch_temperatures.is_empty());
            assert!(b.batch_top_ks.is_empty());
            assert!(b.batch_top_ps.is_empty());
            assert!(b.batch_sampling_seeds.is_empty());
        }
    }

    #[test]
    fn get_topk_neg_vec_creates_and_caches() {
        let mut b = SamplingBuffers::new();
        let dev = Device::Cpu;

        let v1 = b.get_topk_neg_vec(5, &dev).unwrap();
        assert_eq!(v1.dims(), &[5]);
        // All values should be -1e9.
        let vals: Vec<f32> = v1.to_vec1().unwrap();
        for v in &vals {
            assert!((*v - (-1e9f32)).abs() < 1.0);
        }

        // Second call should return cached version.
        assert!(b.topk_neg_vecs.contains_key(&5));
        let v2 = b.get_topk_neg_vec(5, &dev).unwrap();
        assert_eq!(v2.dims(), &[5]);
    }

    #[test]
    fn get_topk_shift_idx_creates_range() {
        let mut b = SamplingBuffers::new();
        let dev = Device::Cpu;

        let idx = b.get_topk_shift_idx(5, &dev).unwrap();
        let vals: Vec<u32> = idx.to_vec1().unwrap();
        assert_eq!(vals, vec![1, 2, 3, 4]);
    }

    #[test]
    fn get_topk_shift_idx_k1_fails() {
        let mut b = SamplingBuffers::new();
        let dev = Device::Cpu;
        assert!(b.get_topk_shift_idx(1, &dev).is_err());
    }

    #[test]
    fn get_topk_shift_buf_zeros() {
        let mut b = SamplingBuffers::new();
        let dev = Device::Cpu;

        let buf = b.get_topk_shift_buf(4, &dev, DType::F32).unwrap();
        let vals: Vec<f32> = buf.to_vec1().unwrap();
        assert_eq!(vals, vec![0.0; 4]);
    }

    #[test]
    fn get_topk_cumsum_mat_upper_triangular() {
        let mut b = SamplingBuffers::new();
        let dev = Device::Cpu;

        let mat = b.get_topk_cumsum_mat(3, &dev).unwrap();
        assert_eq!(mat.dims(), &[3, 3]);
        let vals: Vec<Vec<f32>> = mat.to_vec2().unwrap();
        // Upper triangular with 1s and 0s below diagonal.
        // Row 0 (row <= col for all): [1, 1, 1]
        // Row 1 (row=1 <= col for col>=1): [0, 1, 1]
        // Row 2 (row=2 <= col for col>=2): [0, 0, 1]
        assert_eq!(vals[0], vec![1.0, 1.0, 1.0]);
        assert_eq!(vals[1], vec![0.0, 1.0, 1.0]);
        assert_eq!(vals[2], vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn cumsum_mat_cached_on_second_call() {
        let mut b = SamplingBuffers::new();
        let dev = Device::Cpu;

        let _ = b.get_topk_cumsum_mat(4, &dev).unwrap();
        assert!(b.topk_cumsum_mats.contains_key(&4));

        let mat2 = b.get_topk_cumsum_mat(4, &dev).unwrap();
        assert_eq!(mat2.dims(), &[4, 4]);
    }

    #[test]
    fn sample_batch_greedy_matches_row_argmax_without_penalty() {
        let logits =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 5.0, 6.0, 4.0], (2, 3), &Device::Cpu).unwrap();
        let seq0 = make_greedy_seq(vec![0], 1.0);
        let seq1 = make_greedy_seq(vec![1], 1.0);
        let mut buffers = SamplingBuffers::new();

        let sample =
            sample_batch_greedy(&logits, &[&seq0, &seq1], &[true, true], &mut buffers).unwrap();

        assert_eq!(sample.tokens, vec![2, 1]);
        assert_eq!(sample.mode, BatchGreedyMode::TensorFallback);
    }

    #[test]
    fn sample_batch_greedy_applies_repetition_penalty() {
        let logits =
            Tensor::from_vec(vec![5.0f32, 4.0, 1.0, 0.5, 1.0, 2.0], (2, 3), &Device::Cpu).unwrap();
        let seq0 = make_greedy_seq(vec![0], 2.0);
        let seq1 = make_greedy_seq(vec![1], 1.0);
        let mut buffers = SamplingBuffers::new();

        let sample =
            sample_batch_greedy(&logits, &[&seq0, &seq1], &[true, true], &mut buffers).unwrap();

        assert_eq!(sample.tokens, vec![1, 2]);
        assert_eq!(sample.mode, BatchGreedyMode::TensorFallback);
    }

    // ── apply_repeat_penalty_inplace tests ──

    #[test]
    fn repeat_penalty_empty_context_is_noop() {
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu).unwrap();
        apply_repeat_penalty_inplace(&logits, 2.0, &[]).unwrap();
        let vals: Vec<f32> = logits.to_vec1().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn repeat_penalty_reduces_positive_logits() {
        let logits = Tensor::new(&[4.0f32, 2.0, 6.0, 1.0], &Device::Cpu).unwrap();
        // Penalize tokens 0 and 2 with penalty=2.0.
        apply_repeat_penalty_inplace(&logits, 2.0, &[0, 2]).unwrap();
        let vals: Vec<f32> = logits.to_vec1().unwrap();
        // Positive logits are divided by penalty.
        assert!((vals[0] - 2.0).abs() < 0.01); // 4.0 / 2.0
        assert!((vals[1] - 2.0).abs() < 0.01); // untouched
        assert!((vals[2] - 3.0).abs() < 0.01); // 6.0 / 2.0
        assert!((vals[3] - 1.0).abs() < 0.01); // untouched
    }

    #[test]
    fn repeat_penalty_amplifies_negative_logits() {
        let logits = Tensor::new(&[-4.0f32, 2.0, -6.0], &Device::Cpu).unwrap();
        // Penalize tokens 0 and 2 with penalty=2.0.
        apply_repeat_penalty_inplace(&logits, 2.0, &[0, 2]).unwrap();
        let vals: Vec<f32> = logits.to_vec1().unwrap();
        // Negative logits are multiplied by penalty (making them more negative).
        assert!((vals[0] - (-8.0)).abs() < 0.01); // -4.0 * 2.0
        assert!((vals[1] - 2.0).abs() < 0.01); // untouched
        assert!((vals[2] - (-12.0)).abs() < 0.01); // -6.0 * 2.0
    }

    #[test]
    fn repeat_penalty_deduplicates_context() {
        let logits = Tensor::new(&[4.0f32, 2.0, 6.0], &Device::Cpu).unwrap();
        // Duplicate tokens in context should not double-penalize.
        apply_repeat_penalty_inplace(&logits, 2.0, &[0, 0, 0]).unwrap();
        let vals: Vec<f32> = logits.to_vec1().unwrap();
        assert!((vals[0] - 2.0).abs() < 0.01); // 4.0 / 2.0
    }

    #[test]
    fn repeat_penalty_no_effect_with_1() {
        let logits = Tensor::new(&[4.0f32, -2.0, 6.0], &Device::Cpu).unwrap();
        apply_repeat_penalty_inplace(&logits, 1.0, &[0, 1, 2]).unwrap();
        let vals: Vec<f32> = logits.to_vec1().unwrap();
        assert!((vals[0] - 4.0).abs() < 0.01);
        assert!((vals[1] - (-2.0)).abs() < 0.01);
        assert!((vals[2] - 6.0).abs() < 0.01);
    }
}
