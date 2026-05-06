//! Optimized Qwen3 transformer implementation.
//!
//! Focused Qwen3 text-generation model with these hot-path optimizations:
//!
//! 1. **Pre-allocated KV cache** with in-place `slice_set` writes
//!    — O(new_seq_len) per decode step instead of O(cache_len) `Tensor::cat`.
//! 2. **GQA-grouped SDPA for decode** (seq_len=1)
//!    — Reshapes Q into `[B*kv_heads, n_rep, D]` and uses 3-D batch matmul with
//!      implicit broadcasting, avoiding the expensive N× KV head expansion.
//! 3. **Fused RoPE kernel** via `candle_nn::rotary_emb::rope()`
//!    — One CUDA launch per Q/K instead of 5 manual tensor ops.
//!    — Precomputed `[max_pos, head_dim/2]` cos/sin tables (half-width, as
//!      required by the `rope()` API).
//! 4. **GGUF quantization** via the polymorphic `LinearLayer` enum
//!    — Same model code serves both safetensors (f16/f32/bf16) and GGUF weights.
//! 5. **Batched decode infrastructure**
//!    — `setup_batch_decode`, `step_batch_decode`, `extract_batch_kv` enable
//!      GPU-efficient concurrent sequence serving in the engine.
//! 6. **KV cache save/restore**
//!    — `get_kv_caches` / `set_kv_caches` for continuous-batching context swap.
//! 7. **Fused SiLU-mul MLP gate**
//!    — `fused_silu_mul` replaces the `narrow + silu + mul` op chain in each
//!      MLP block, reducing kernel launches and intermediate allocations.
//! 8. **Merged QKV / gate+up projections**
//!    — Q, K, V weights fused into one matmul; gate and up weights fused into
//!      one matmul — halves the number of linear-layer dispatches per layer.

use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::rotary_emb::rope;
use candle_nn::{linear_no_bias, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;
use std::io::{Read, Seek};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, Default)]
pub struct BatchDecodeSetupStats {
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
pub struct BatchDecodeExtractStats {
    pub narrow_us: u64,
    pub contiguous_us: u64,
    pub cache_clear_us: u64,
    pub total_us: u64,
    pub layers: usize,
    pub sequences: usize,
}

fn elapsed_us(start: Instant) -> u64 {
    start.elapsed().as_micros() as u64
}

fn env_flag_default(name: &str, default: bool) -> bool {
    match std::env::var(name).ok().as_deref() {
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON") => true,
        Some("0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF") => false,
        Some(_) => default,
        None => default,
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Qwen3RmsNorm {
    norm: RmsNorm,
    weight: Tensor,
    eps: f64,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl Qwen3RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self {
            norm: RmsNorm::new(weight.clone(), eps),
            weight,
            eps,
        }
    }

    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self::new(weight, eps))
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.norm.forward(xs)
    }

    fn weight(&self) -> &Tensor {
        &self.weight
    }

    fn eps(&self) -> f64 {
        self.eps
    }
}

pub struct Gguf<R: Read + Seek> {
    pub ct: gguf_file::Content,
    reader: R,
    device: Device,
    dtype: DType,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device, dtype: DType) -> Self {
        Self {
            ct,
            reader,
            device,
            dtype,
        }
    }

    pub fn linear(&mut self, name: &str) -> Result<LinearLayer> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        let qmm = candle_core::quantized::QMatMul::from_arc(Arc::new(ws))?;
        Ok(LinearLayer::Quantized(qmm))
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<Qwen3RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        let weight = ws.dequantize(&self.device)?.to_dtype(self.dtype)?;
        Ok(Qwen3RmsNorm::new(weight, eps))
    }

    pub fn embedding(&mut self, name: &str, hidden_size: usize) -> Result<candle_nn::Embedding> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        let weight = ws.dequantize(&self.device)?.to_dtype(self.dtype)?;
        Ok(candle_nn::Embedding::new(weight, hidden_size))
    }

    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }
}

pub enum LinearLayer {
    Standard(Linear),
    Quantized(candle_core::quantized::QMatMul),
}

impl Module for LinearLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(l) => l.forward(xs),
            Self::Quantized(q) => {
                let input_dtype = xs.dtype();
                let xs_f32 = if input_dtype != DType::F32 {
                    xs.to_dtype(DType::F32)?
                } else {
                    xs.clone()
                };
                let out = q.forward(&xs_f32)?;
                if input_dtype != DType::F32 {
                    out.to_dtype(input_dtype)
                } else {
                    Ok(out)
                }
            }
        }
    }
}

// ── Event-tracking RAII guard ────────────────────────────────────────────
//
// Candle defaults to tracking per-tensor CudaEvents for multi-stream safety.
// Crane uses a single CUDA stream — those events are pure overhead.
// This guard disables event tracking on first use and leaves it disabled
// (candle 0.9.x exposes only `disable_event_tracking`, not a re-enable).

#[cfg(feature = "cuda")]
struct EventTrackingGuard;

#[cfg(feature = "cuda")]
impl EventTrackingGuard {
    fn disable(device: &candle_core::Device) -> Self {
        if let candle_core::Device::Cuda(ref dev) = device {
            if dev.is_event_tracking() {
                // Safety: we ensure sequential use of a single CUDA stream.
                unsafe { dev.disable_event_tracking() };
            }
        }
        Self
    }
}

// ── Config ──────────────────────────────────────────────────────────────

fn default_true() -> bool {
    true
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub max_window_layers: usize,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

// ── Rotary Embedding (with cache) ───────────────────────────────────────

struct RotaryEmbedding {
    /// Pre-computed cos table: [max_pos, head_dim] — full table, zero-copy narrow per call.
    cos_table: Tensor,
    /// Pre-computed sin table: [max_pos, head_dim] — full table, zero-copy narrow per call.
    sin_table: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &Config, device: &Device) -> Result<Self> {
        let dim = config.head_dim();
        let base = config.rope_theta;
        let max_pos = config.max_position_embeddings;

        // inv_freq: [dim/2]
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv.as_slice(), device)?;

        // positions × inv_freq: [max_pos, dim/2]
        let positions: Vec<f32> = (0..max_pos).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?; // [max_pos, dim/2]

        // cos/sin tables: [max_pos, dim/2] — candle_nn::rotary_emb::rope() handles
        // the half-dim duplication internally, so we store the raw half-dim tables.
        let cos_table = freqs.cos()?.contiguous()?;
        let sin_table = freqs.sin()?.contiguous()?;

        Ok(Self {
            cos_table,
            sin_table,
        })
    }

    /// Return cos/sin slices for positions [0..seq_len].
    /// Both narrow() calls are zero-copy views — no CUDA kernel launched.
    fn forward(&self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos_table.narrow(0, 0, seq_len)?;
        let sin = self.sin_table.narrow(0, 0, seq_len)?;
        Ok((cos, sin))
    }

    /// Return the full cos/sin tables ([max_position_embeddings, head_dim/2]).
    ///
    /// Used by the indexed-RoPE path (decode kernels that take per-row absolute
    /// position indices). The kernel passes `max_position` as a host scalar baked
    /// into the launch args; using a `narrow()` slice here would let CUDA Graph
    /// capture freeze a *changing* `max_position` and clamp positions on replay
    /// to the round-N value, producing RoPE drift across replays. Always passing
    /// the full table guarantees `max_position` is constant for the model's life.
    fn full_tables(&self) -> (Tensor, Tensor) {
        (self.cos_table.clone(), self.sin_table.clone())
    }
}

// ── Attention ───────────────────────────────────────────────────────────

struct PagedAttentionDecodeRun<'a> {
    context: &'a crate::fused_ops::PagedAttentionDecodeContext,
    metadata: &'a crate::fused_ops::PagedAttentionMetadataCudaBuffers,
    layer_hits: &'a std::cell::Cell<usize>,
    layer_fallbacks: &'a std::cell::Cell<usize>,
}

struct Attention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    /// Merged QKV weight [q_dim + 2*kv_dim, hidden_size] — one gemv instead of 3.
    /// Only set for Standard (non-quantized) weights.
    qkv_proj: Option<Linear>,
    q_norm: Option<Qwen3RmsNorm>,
    k_norm: Option<Qwen3RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    /// Pre-allocated KV cache buffer (may be larger than `cache_seq_len`).
    kv_cache: Option<(Tensor, Tensor)>,
    /// Number of valid (filled) positions in the KV cache buffer.
    cache_seq_len: usize,
}

impl Attention {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let bias = config.attention_bias;

        let make_proj = |in_d: usize, out_d: usize, name: &str| -> Result<LinearLayer> {
            if bias {
                Ok(LinearLayer::Standard(candle_nn::linear(
                    in_d,
                    out_d,
                    vb.pp(name),
                )?))
            } else {
                Ok(LinearLayer::Standard(linear_no_bias(
                    in_d,
                    out_d,
                    vb.pp(name),
                )?))
            }
        };

        let q_proj = make_proj(config.hidden_size, num_heads * head_dim, "q_proj")?;
        let k_proj = make_proj(config.hidden_size, num_kv_heads * head_dim, "k_proj")?;
        let v_proj = make_proj(config.hidden_size, num_kv_heads * head_dim, "v_proj")?;
        let o_proj = make_proj(num_heads * head_dim, config.hidden_size, "o_proj")?;

        // Create merged QKV projection for Standard weights:
        // Concatenate [q_weight; k_weight; v_weight] along dim 0 so one gemv
        // replaces three.  `narrow` splits are zero-copy views.
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_proj = if let (
            LinearLayer::Standard(ref q),
            LinearLayer::Standard(ref k),
            LinearLayer::Standard(ref v),
        ) = (&q_proj, &k_proj, &v_proj)
        {
            let qkv_w = Tensor::cat(&[q.weight(), k.weight(), v.weight()], 0)?;
            let qkv_b = match (q.bias(), k.bias(), v.bias()) {
                (Some(qb), Some(kb), Some(vb)) => Some(Tensor::cat(&[qb, kb, vb], 0)?),
                _ => None,
            };
            Some(Linear::new(qkv_w, qkv_b))
        } else {
            None
        };

        let (q_norm, k_norm) = if config.use_qk_norm {
            (
                Some(Qwen3RmsNorm::load(
                    head_dim,
                    config.rms_norm_eps,
                    vb.pp("q_norm"),
                )?),
                Some(Qwen3RmsNorm::load(
                    head_dim,
                    config.rms_norm_eps,
                    vb.pp("k_norm"),
                )?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            qkv_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            q_dim,
            kv_dim,
            kv_cache: None,
            cache_seq_len: 0,
        })
    }

    /// Construct from GGUF quantized weights.
    fn new_from_gguf<R: Read + Seek>(
        config: &Config,
        gg: &mut Gguf<R>,
        layer_idx: usize,
    ) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let prefix = format!("blk.{layer_idx}");

        let q_proj = gg.linear(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.linear(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.linear(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.linear(&format!("{prefix}.attn_output.weight"))?;

        let (q_norm, k_norm) = if config.use_qk_norm {
            (
                Some(gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), config.rms_norm_eps)?),
                Some(gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), config.rms_norm_eps)?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            qkv_proj: None, // GGUF quantized — cannot merge
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            q_dim: num_heads * head_dim,
            kv_dim: num_kv_heads * head_dim,
            kv_cache: None,
            cache_seq_len: 0,
        })
    }

    /// Update the pre-allocated KV cache with new K,V tensors.
    ///
    /// Uses `slice_set` for O(1) in-place writes when the buffer has room.
    /// Falls back to cat + reallocate when the buffer is full.
    fn update_kv_cache(
        &mut self,
        k: Tensor,
        v: Tensor,
        fixed_cache_width: Option<usize>,
        append_offset: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let new_seq_len = k.dim(2)?;
        let cache_seq_len = self.cache_seq_len;

        match self.kv_cache.take() {
            Some((buf_k, buf_v)) => {
                let buf_len = buf_k.dim(2)?;
                let new_total = cache_seq_len + new_seq_len;

                if new_total <= buf_len {
                    let view_len = fixed_cache_width.unwrap_or(new_total);
                    if view_len < new_total || view_len > buf_len {
                        candle_core::bail!(
                            "fixed cache width {view_len} is incompatible with used KV length {new_total} and buffer length {buf_len}"
                        );
                    }
                    if let Some(append_offset) = append_offset.filter(|_| {
                        fixed_cache_width.is_some()
                            && new_seq_len == 1
                            && k.dtype() == DType::BF16
                            && v.dtype() == DType::BF16
                            && k.device().is_cuda()
                            && v.device().is_cuda()
                    }) {
                        crate::fused_ops::batch_kv_append_bf16_with_offset(
                            &buf_k,
                            &buf_v,
                            &k,
                            &v,
                            append_offset,
                            buf_len,
                            self.num_kv_heads,
                            self.head_dim,
                        )?;
                    } else {
                        // In-place write — O(new_seq_len).
                        buf_k.slice_set(&k, 2, cache_seq_len)?;
                        buf_v.slice_set(&v, 2, cache_seq_len)?;
                    }
                    let k_view = buf_k.narrow(2, 0, view_len)?;
                    let v_view = buf_v.narrow(2, 0, view_len)?;
                    self.kv_cache = Some((buf_k, buf_v));
                    self.cache_seq_len = new_total;
                    Ok((k_view, v_view))
                } else {
                    if fixed_cache_width.is_some() {
                        candle_core::bail!(
                            "fixed-width decode requires KV buffer length {buf_len}, but used length would become {new_total}"
                        );
                    }
                    // Buffer overflow — grow with extra room.
                    let cur_k = buf_k.narrow(2, 0, cache_seq_len)?;
                    let cur_v = buf_v.narrow(2, 0, cache_seq_len)?;
                    drop(buf_k);
                    drop(buf_v);
                    let full_k = Tensor::cat(&[&cur_k, &k], 2)?;
                    let full_v = Tensor::cat(&[&cur_v, &v], 2)?;
                    drop(cur_k);
                    drop(cur_v);
                    let total = full_k.dim(2)?;
                    let room = 256; // fixed small room — avoids 2x over-allocation
                    let (b, h, _, d) = full_k.dims4()?;
                    let new_buf_k = Tensor::zeros((b, h, total + room, d), k.dtype(), k.device())?;
                    let new_buf_v = Tensor::zeros((b, h, total + room, d), v.dtype(), v.device())?;
                    new_buf_k.slice_set(&full_k, 2, 0)?;
                    new_buf_v.slice_set(&full_v, 2, 0)?;
                    self.kv_cache = Some((new_buf_k, new_buf_v));
                    self.cache_seq_len = total;
                    Ok((full_k, full_v))
                }
            }
            None => {
                // First use — allocate buffer with extra room.
                let (b, h, s, d) = k.dims4()?;
                let room = fixed_cache_width
                    .map(|width| width.saturating_sub(s))
                    .unwrap_or(256); // fixed small room — avoids 2x over-allocation
                let buf_k = Tensor::zeros((b, h, s + room, d), k.dtype(), k.device())?;
                let buf_v = Tensor::zeros((b, h, s + room, d), v.dtype(), v.device())?;
                buf_k.slice_set(&k, 2, 0)?;
                buf_v.slice_set(&v, 2, 0)?;
                let k_view = if let Some(width) = fixed_cache_width {
                    if width < s || width > s + room {
                        candle_core::bail!(
                            "fixed cache width {width} is incompatible with initial KV length {s}"
                        );
                    }
                    buf_k.narrow(2, 0, width)?
                } else {
                    k
                };
                let v_view = if let Some(width) = fixed_cache_width {
                    buf_v.narrow(2, 0, width)?
                } else {
                    v
                };
                self.kv_cache = Some((buf_k, buf_v));
                self.cache_seq_len = s;
                Ok((k_view, v_view))
            }
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rope_positions: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        layer_idx: usize,
        paged_attention: Option<&PagedAttentionDecodeRun<'_>>,
        fixed_cache_width: Option<usize>,
        append_offset: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        // Use merged QKV if available — one gemv instead of three.
        let (q, k, v) = if let Some(ref qkv_proj) = self.qkv_proj {
            let qkv = qkv_proj.forward(hidden_states)?; // [B, S, q_dim+2*kv_dim]
            let q = qkv.narrow(D::Minus1, 0, self.q_dim)?;
            let k = qkv.narrow(D::Minus1, self.q_dim, self.kv_dim)?;
            let v = qkv.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?;
            (q, k, v)
        } else {
            let q = self.q_proj.forward(hidden_states)?;
            let k = self.k_proj.forward(hidden_states)?;
            let v = self.v_proj.forward(hidden_states)?;
            (q, k, v)
        };

        // [B, S, num_heads * head_dim] → [B, num_heads, S, head_dim]
        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = self.apply_qk_norm_and_rope(q, k, cos, sin, rope_positions)?;

        let use_paged_attention = seq_len == 1
            && paged_attention.is_some()
            && q.dtype() == DType::BF16
            && k.dtype() == DType::BF16
            && v.dtype() == DType::BF16
            && q.device().is_cuda();
        let current_k_for_paged = if use_paged_attention {
            Some(k.contiguous()?)
        } else {
            None
        };
        let current_v_for_paged = if use_paged_attention {
            Some(v.contiguous()?)
        } else {
            None
        };

        // Update KV cache (pre-allocated with slice_set)
        let (k, v) = self.update_kv_cache(
            current_k_for_paged.clone().unwrap_or(k),
            current_v_for_paged.clone().unwrap_or(v),
            fixed_cache_width,
            append_offset,
        )?;

        if let (Some(run), Some(current_k), Some(current_v)) = (
            paged_attention,
            current_k_for_paged.as_ref(),
            current_v_for_paged.as_ref(),
        ) {
            let scale = 1.0f32 / (self.head_dim as f32).sqrt();
            if let Ok(attn_output) = crate::fused_ops::paged_attention_decode_bf16_with_metadata(
                &run.context.pages,
                &q.contiguous()?,
                current_k,
                current_v,
                layer_idx,
                run.context.num_layers,
                run.context.block_size,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                scale,
                run.metadata,
            ) {
                run.layer_hits.set(run.layer_hits.get() + 1);
                let attn_output = attn_output.reshape((b_sz, 1, self.num_heads * self.head_dim))?;
                return self.o_proj.forward(&attn_output);
            } else {
                run.layer_fallbacks.set(run.layer_fallbacks.get() + 1);
            }
        } else if let Some(run) = paged_attention {
            run.layer_fallbacks.set(run.layer_fallbacks.get() + 1);
        }

        // ── SDPA ──
        let n_rep = self.num_heads / self.num_kv_heads;
        let _kv_s = k.dim(2)?;

        if n_rep > 1 && seq_len == 1 {
            // ── GQA-grouped SDPA for decode (seq_len=1) ──
            // Use 4D tensors throughout so candle's matmul only has to
            // flatten+contiguous the non-contiguous K narrow-view ONCE
            // instead of reshape(contiguous) + transpose + contiguous.
            let scale = 1.0 / (self.head_dim as f64).sqrt();

            // Q: [B, H, 1, D] → [B, kv_heads, n_rep, D], pre-scaled
            let q_g = (q.reshape((b_sz, self.num_kv_heads, n_rep, self.head_dim))? * scale)?;

            // K^T: [B, kv_heads, D, S] — just a view (0 copies here;
            //       matmul will flatten+contiguous in one pass).
            let k_t = k.transpose(2, 3)?;

            // scores: [B, kv_heads, n_rep, S]
            let attn_weights = q_g.matmul(&k_t)?;

            let attn_weights = match attention_mask {
                Some(mask) => {
                    // mask [B, 1, 1, S] broadcasts over kv_heads & n_rep
                    attn_weights.broadcast_add(mask)?
                }
                None => attn_weights,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

            // V: [B, kv_heads, S, D] — matmul handles non-contiguous
            let attn_output = attn_weights.matmul(&v)?; // [B, kv_heads, n_rep, D]

            // Reshape back: → [B, H, D] → [B, 1, H*D]
            let attn_output = attn_output
                .reshape((b_sz, self.num_heads, self.head_dim))?
                .reshape((b_sz, 1, self.num_heads * self.head_dim))?;
            return self.o_proj.forward(&attn_output);
        }

        // ── Standard SDPA for prefill or when n_rep == 1 ──
        let k = if n_rep > 1 {
            let (b, kv_heads, s, d) = k.dims4()?;
            k.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .reshape((b, kv_heads * n_rep, s, d))?
        } else {
            k
        };
        let v = if n_rep > 1 {
            let (b, kv_heads, s, d) = v.dims4()?;
            v.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .reshape((b, kv_heads * n_rep, s, d))?
        } else {
            v
        };

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // [B, H, S, D] → [B, S, H*D]
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz, seq_len, ()))?;

        self.o_proj.forward(&attn_output)
    }

    fn apply_rope(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rope_positions: Option<&Tensor>,
    ) -> Result<Tensor> {
        let x = x.contiguous()?;
        #[cfg(feature = "cuda")]
        if let Some(positions) = rope_positions {
            if x.device().is_cuda()
                && x.dtype() == DType::BF16
                && cos.dtype() == DType::F32
                && sin.dtype() == DType::F32
            {
                if let Ok(out) = crate::fused_ops::fused_rope_indexed_bf16(&x, cos, sin, positions)
                {
                    return Ok(out);
                }
            }
        }

        if let Some(positions) = rope_positions {
            let cos = cos
                .index_select(positions, 0)?
                .to_dtype(x.dtype())?
                .unsqueeze(1)?;
            let sin = sin
                .index_select(positions, 0)?
                .to_dtype(x.dtype())?
                .unsqueeze(1)?;
            return rope(&x, &cos, &sin);
        }

        rope(&x, cos, sin)
    }

    fn apply_qk_norm_and_rope(
        &self,
        q: Tensor,
        k: Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rope_positions: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if let (Some(q_norm), Some(k_norm), Some(positions)) =
            (self.q_norm.as_ref(), self.k_norm.as_ref(), rope_positions)
        {
            if env_flag_default("CRANE_FUSED_QK_NORM_ROPE", true)
                && q.device().is_cuda()
                && k.device().is_cuda()
                && q.dtype() == DType::BF16
                && k.dtype() == DType::BF16
                && q_norm.weight().dtype() == DType::BF16
                && k_norm.weight().dtype() == DType::BF16
                && cos.dtype() == DType::F32
                && sin.dtype() == DType::F32
            {
                if let Ok((q_out, k_out)) = crate::fused_ops::fused_qk_norm_rope_indexed_bf16(
                    &q,
                    &k,
                    q_norm.weight(),
                    k_norm.weight(),
                    cos,
                    sin,
                    positions,
                    q_norm.eps() as f32,
                ) {
                    return Ok((q_out, k_out));
                }
            }
        }

        // Per-head QK norm (Qwen3 applies before RoPE). RmsNorm normalises over
        // the last dim, so it operates directly on [B, H, S, D].
        let q = if let Some(ref norm) = self.q_norm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref norm) = self.k_norm {
            norm.forward(&k)?
        } else {
            k
        };

        let q = self.apply_rope(&q, cos, sin, rope_positions)?;
        let k = self.apply_rope(&k, cos, sin, rope_positions)?;
        Ok((q, k))
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
        self.cache_seq_len = 0;
    }
}

// ── MLP ─────────────────────────────────────────────────────────────────

/// Gate+up projection: either a merged [2*I, H] weight (Standard) or separate quantized projections.
enum MlpGateUp {
    /// Merged gate+up weight — one gemv instead of two. Standard (BF16/F16/F32) only.
    Merged {
        gate_up_proj: Linear,
        intermediate_size: usize,
    },
    /// Separate quantized gate and up projections (GGUF).
    Separate {
        gate_proj: LinearLayer,
        up_proj: LinearLayer,
    },
}

struct Mlp {
    gate_up: MlpGateUp,
    down_proj: LinearLayer,
}

impl Mlp {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let down_proj = LinearLayer::Standard(linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?);

        // Merge gate+up into a single weight, then drop the originals to save VRAM.
        let gate_up_w = Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?;
        // gate_proj and up_proj are dropped here — their VRAM is freed.
        let gate_up = MlpGateUp::Merged {
            gate_up_proj: Linear::new(gate_up_w, None),
            intermediate_size: config.intermediate_size,
        };

        Ok(Self { gate_up, down_proj })
    }

    fn new_from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, layer_idx: usize) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let gate_proj = gg.linear(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gg.linear(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.linear(&format!("{prefix}.ffn_down.weight"))?;
        Ok(Self {
            gate_up: MlpGateUp::Separate { gate_proj, up_proj },
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.gate_up {
            MlpGateUp::Merged {
                gate_up_proj,
                intermediate_size,
            } => {
                let gu = gate_up_proj.forward(x)?; // [B, S, 2*intermediate_size]

                // Use fused CUDA kernel when available: eliminates narrow + silu + mul
                // (3 kernel launches → 1).
                #[cfg(feature = "cuda")]
                {
                    if gu.device().is_cuda() {
                        let activated = crate::fused_ops::fused_silu_mul(
                            &gu.contiguous()?,
                            *intermediate_size,
                        )?;
                        return self.down_proj.forward(&activated);
                    }
                }

                // CPU / non-CUDA fallback
                let gate = gu.narrow(D::Minus1, 0, *intermediate_size)?;
                let up = gu.narrow(D::Minus1, *intermediate_size, *intermediate_size)?;
                let gate = candle_nn::Activation::Silu.forward(&gate)?;
                self.down_proj.forward(&(gate * up)?)
            }
            MlpGateUp::Separate { gate_proj, up_proj } => {
                let gate = gate_proj.forward(x)?;
                let gate = candle_nn::Activation::Silu.forward(&gate)?;
                let up = up_proj.forward(x)?;
                self.down_proj.forward(&(gate * up)?)
            }
        }
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: Qwen3RmsNorm,
    post_attention_layernorm: Qwen3RmsNorm,
}

impl DecoderLayer {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(config, vb.pp("self_attn"))?;
        let mlp = Mlp::new(config, vb.pp("mlp"))?;
        let input_layernorm = Qwen3RmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = Qwen3RmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn new_from_gguf<R: Read + Seek>(
        config: &Config,
        gg: &mut Gguf<R>,
        layer_idx: usize,
    ) -> Result<Self> {
        let self_attn = Attention::new_from_gguf(config, gg, layer_idx)?;
        let mlp = Mlp::new_from_gguf(gg, layer_idx)?;
        let prefix = format!("blk.{layer_idx}");
        let input_layernorm =
            gg.rms_norm(&format!("{prefix}.attn_norm.weight"), config.rms_norm_eps)?;
        let post_attention_layernorm =
            gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), config.rms_norm_eps)?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rope_positions: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        layer_idx: usize,
        paged_attention: Option<&PagedAttentionDecodeRun<'_>>,
        fixed_cache_width: Option<usize>,
        append_offset: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            cos,
            sin,
            rope_positions,
            attention_mask,
            layer_idx,
            paged_attention,
            fixed_cache_width,
            append_offset,
        )?;
        let (residual, hidden_states) =
            self.add_and_post_attention_norm(residual, &hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        &residual + &hidden_states
    }

    fn add_and_post_attention_norm(
        &self,
        residual: &Tensor,
        hidden: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        {
            let dims = residual.dims();
            let is_decode_step = dims.len() == 3 && dims[1] == 1 && hidden.dims() == dims;
            if is_decode_step
                && env_flag_default("CRANE_FUSED_ADD_RMSNORM", true)
                && residual.device().is_cuda()
                && hidden.device().is_cuda()
                && residual.dtype() == DType::BF16
                && hidden.dtype() == DType::BF16
                && self.post_attention_layernorm.weight().dtype() == DType::BF16
            {
                if let Ok((sum, norm)) = crate::fused_ops::fused_add_rmsnorm_bf16(
                    residual,
                    hidden,
                    self.post_attention_layernorm.weight(),
                    self.post_attention_layernorm.eps() as f32,
                ) {
                    return Ok((sum, norm));
                }
            }
        }

        let sum = (residual + hidden)?;
        let norm = self.post_attention_layernorm.forward(&sum)?;
        Ok((sum, norm))
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ── Full Model ──────────────────────────────────────────────────────────

pub struct Qwen3Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: Qwen3RmsNorm,
    lm_head: LinearLayer,
    rotary_emb: RotaryEmbedding,
    config: Config,
    dtype: DType,
    paged_attention_metadata: crate::fused_ops::PagedAttentionMetadataCudaBuffers,
    last_paged_attention_layer_hits: usize,
    last_paged_attention_layer_fallbacks: usize,
    last_batch_decode_setup_stats: BatchDecodeSetupStats,
    last_batch_decode_extract_stats: BatchDecodeExtractStats,
    batch_decode_kv_workspaces: Vec<Option<BatchDecodeKvWorkspace>>,
    batch_decode_kv_lens_buffer: crate::fused_ops::ReusableU32TensorBuffer,
    batch_decode_workspace_generation: u64,
}

struct BatchDecodeKvWorkspace {
    k: Tensor,
    v: Tensor,
    batch: usize,
    kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    dtype: DType,
}

impl BatchDecodeKvWorkspace {
    fn matches(
        &self,
        batch: usize,
        kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> bool {
        self.batch == batch
            && self.kv_heads == kv_heads
            && self.head_dim == head_dim
            && self.capacity >= capacity
            && self.dtype == dtype
            && self.k.device().same_device(device)
            && self.v.device().same_device(device)
    }
}

impl Qwen3Model {
    /// Construct from safetensors / HuggingFace checkpoint.
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let model_vb = vb.pp("model");
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            model_vb.pp("embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = model_vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(config, layers_vb.pp(i))?);
        }

        let norm =
            Qwen3RmsNorm::load(config.hidden_size, config.rms_norm_eps, model_vb.pp("norm"))?;

        let lm_head = if config.tie_word_embeddings {
            LinearLayer::Standard(Linear::new(embed_tokens.embeddings().clone(), None))
        } else {
            LinearLayer::Standard(linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp("lm_head"),
            )?)
        };

        let rotary_emb = RotaryEmbedding::new(config, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            config: config.clone(),
            dtype,
            paged_attention_metadata: crate::fused_ops::PagedAttentionMetadataCudaBuffers::new(),
            last_paged_attention_layer_hits: 0,
            last_paged_attention_layer_fallbacks: 0,
            last_batch_decode_setup_stats: BatchDecodeSetupStats::default(),
            last_batch_decode_extract_stats: BatchDecodeExtractStats::default(),
            batch_decode_kv_workspaces: std::iter::repeat_with(|| None)
                .take(config.num_hidden_layers)
                .collect(),
            batch_decode_kv_lens_buffer: crate::fused_ops::ReusableU32TensorBuffer::new(),
            batch_decode_workspace_generation: 0,
        })
    }

    /// Construct from a GGUF file.
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let mut gg = Gguf::new(ct, reader, device.clone(), dtype);
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle_core::bail!("cannot find {s} in GGUF metadata"),
            Some(v) => Ok(v.clone()),
        };

        let arch = gg
            .metadata()
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.clone())
            .unwrap_or_else(|| "qwen3".to_string());

        let num_attention_heads =
            md_get(&format!("{arch}.attention.head_count"))?.to_u32()? as usize;
        let num_kv_heads = md_get(&format!("{arch}.attention.head_count_kv"))?.to_u32()? as usize;
        let head_dim = gg
            .metadata()
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(128) as usize;
        let num_hidden_layers = md_get(&format!("{arch}.block_count"))?.to_u32()? as usize;
        let hidden_size = md_get(&format!("{arch}.embedding_length"))?.to_u32()? as usize;
        let intermediate_size = md_get(&format!("{arch}.feed_forward_length"))?.to_u32()? as usize;
        let max_position_embeddings = gg
            .metadata()
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(32768) as usize;
        let rms_norm_eps = gg
            .metadata()
            .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1e-6) as f64;
        let rope_theta = gg
            .metadata()
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1_000_000.0) as f64;

        let use_qk_norm = gg.ct.tensor_infos.contains_key("blk.0.attn_q_norm.weight");
        let tie_word_embeddings = !gg.ct.tensor_infos.contains_key("output.weight");

        let config = Config {
            vocab_size: 0, // updated below
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: num_kv_heads,
            head_dim: Some(head_dim),
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            attention_bias: false,
            use_qk_norm,
            tie_word_embeddings,
            sliding_window: None,
            max_window_layers: 0,
            use_sliding_window: false,
            eos_token_id: None,
        };

        let embed_tokens = gg.embedding("token_embd.weight", hidden_size)?;
        let actual_vocab_size = embed_tokens.embeddings().dim(0)?;
        let config = Config {
            vocab_size: actual_vocab_size,
            ..config
        };

        let mut layers = Vec::with_capacity(num_hidden_layers);
        for i in 0..num_hidden_layers {
            layers.push(DecoderLayer::new_from_gguf(&config, &mut gg, i)?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;

        let lm_head = if tie_word_embeddings {
            LinearLayer::Standard(Linear::new(embed_tokens.embeddings().clone(), None))
        } else {
            gg.linear("output.weight")?
        };

        let rotary_emb = RotaryEmbedding::new(&config, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            config,
            dtype,
            paged_attention_metadata: crate::fused_ops::PagedAttentionMetadataCudaBuffers::new(),
            last_paged_attention_layer_hits: 0,
            last_paged_attention_layer_fallbacks: 0,
            last_batch_decode_setup_stats: BatchDecodeSetupStats::default(),
            last_batch_decode_extract_stats: BatchDecodeExtractStats::default(),
            batch_decode_kv_workspaces: std::iter::repeat_with(|| None)
                .take(num_hidden_layers)
                .collect(),
            batch_decode_kv_lens_buffer: crate::fused_ops::ReusableU32TensorBuffer::new(),
            batch_decode_workspace_generation: 0,
        })
    }

    // ── Forward ─────────────────────────────────────────────────────────

    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;

        // Disable event tracking for the duration of the forward pass.
        // Crane uses a single CUDA stream; the per-tensor CudaEvents are
        // unnecessary and cost ~2×cuEventCreate+cuEventRecord per temp tensor.
        #[cfg(feature = "cuda")]
        let _event_guard = EventTrackingGuard::disable(input_ids.device());

        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;

        let total_len = start_pos + seq_len;
        let (full_cos, full_sin) = self.rotary_emb.forward(total_len)?;
        let cos = full_cos
            .narrow(0, start_pos, seq_len)?
            .to_dtype(self.dtype)?;
        let sin = full_sin
            .narrow(0, start_pos, seq_len)?
            .to_dtype(self.dtype)?;

        // Causal mask (only during prefill; skipped for single-token decode)
        let attention_mask = if seq_len > 1 {
            let mut mask_data = vec![0f32; seq_len * total_len];
            for i in 0..seq_len {
                for j in 0..total_len {
                    if j <= start_pos + i {
                        mask_data[i * total_len + j] = 1.0;
                    }
                }
            }
            let mask = Tensor::from_vec(mask_data, (seq_len, total_len), input_ids.device())?;
            let mask = mask
                .broadcast_lt(&Tensor::new(0.5f32, input_ids.device())?)?
                .to_dtype(self.dtype)?;
            let mask = (mask * (-1e9f64))?;
            Some(mask.unsqueeze(0)?.unsqueeze(0)?)
        } else {
            None
        };

        let mut hidden_states = hidden_states;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(
                &hidden_states,
                &cos,
                &sin,
                None,
                attention_mask.as_ref(),
                layer_idx,
                None,
                None,
                None,
            )?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self
            .lm_head
            .forward(&hidden_states.narrow(1, seq_len - 1, 1)?)?;
        Ok(logits)
    }

    // ── KV Cache Management ─────────────────────────────────────────────

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
        self.paged_attention_metadata.release();
    }

    pub fn release_batch_decode_workspaces(&mut self) -> usize {
        let mut released_layers = 0usize;
        for slot in self.batch_decode_kv_workspaces.iter_mut() {
            if slot.take().is_some() {
                released_layers += 1;
            }
        }
        self.batch_decode_kv_lens_buffer.clear();
        if released_layers > 0 {
            self.batch_decode_workspace_generation =
                self.batch_decode_workspace_generation.wrapping_add(1);
        }
        released_layers
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total bytes held by the model's KV caches (no GPU copies).
    pub fn active_kv_cache_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| {
                l.self_attn
                    .kv_cache
                    .as_ref()
                    .map(|(k, v)| {
                        let k_bytes = k.elem_count() as u64 * k.dtype().size_in_bytes() as u64;
                        let v_bytes = v.elem_count() as u64 * v.dtype().size_in_bytes() as u64;
                        k_bytes + v_bytes
                    })
                    .unwrap_or(0)
            })
            .sum()
    }

    /// Extract per-layer KV caches (valid portion only, zero-copy narrow views).
    ///
    /// The returned views still reference the pre-allocated buffer.  Callers
    /// that need to free the buffer (e.g. batch-decode extract) should use
    /// `Tensor::contiguous()` on their side, or clear `seq.kv_caches` after
    /// consuming the views.
    pub fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layers
            .iter()
            .map(|l| {
                l.self_attn.kv_cache.as_ref().map(|(k, v)| {
                    let len = l.self_attn.cache_seq_len;
                    if len > 0 && len < k.dim(2).unwrap_or(0) {
                        (
                            k.narrow(2, 0, len).unwrap_or_else(|_| k.clone()),
                            v.narrow(2, 0, len).unwrap_or_else(|_| v.clone()),
                        )
                    } else {
                        (k.clone(), v.clone())
                    }
                })
            })
            .collect()
    }

    /// Return the raw active KV cache buffers for each layer.
    ///
    /// Batch decode uses preallocated right-aligned buffers with extra room for
    /// generated tokens. Native paged-KV append reads from those contiguous
    /// buffers before extraction clears them, so this intentionally does not
    /// narrow to `cache_seq_len`.
    pub fn get_kv_cache_buffers(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layers
            .iter()
            .map(|l| {
                l.self_attn
                    .kv_cache
                    .as_ref()
                    .map(|(k, v)| (k.clone(), v.clone()))
            })
            .collect()
    }

    /// Restore per-layer KV caches.
    pub fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        for (layer, cache) in self.layers.iter_mut().zip(caches.into_iter()) {
            let seq_len = cache
                .as_ref()
                .map(|(k, _)| k.dim(2).unwrap_or(0))
                .unwrap_or(0);
            layer.self_attn.kv_cache = cache;
            layer.self_attn.cache_seq_len = seq_len;
        }
    }

    // ── Batched Decode ──────────────────────────────────────────────────

    /// Pad per-sequence KV caches to the same length and load into model layers.
    ///
    /// Returns `(kv_lens, max_kv_len)`.
    pub fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> Result<(Vec<usize>, usize)> {
        let total_start = Instant::now();
        let mut stats = BatchDecodeSetupStats {
            layers: self.layers.len(),
            sequences: seq_kv_caches.len(),
            ..BatchDecodeSetupStats::default()
        };
        let kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim();
        let device = self.embed_tokens.embeddings().device().clone();

        let t_kv_len = Instant::now();
        let kv_lens: Vec<usize> = seq_kv_caches
            .iter()
            .map(|caches| {
                caches
                    .first()
                    .and_then(|c| c.as_ref())
                    .map(|(k, _)| k.dim(2).unwrap_or(0))
                    .unwrap_or(0)
            })
            .collect();
        let max_kv_len = kv_lens.iter().copied().max().unwrap_or(0);
        stats.kv_len_scan_us += elapsed_us(t_kv_len);

        for layer_idx in 0..self.layers.len() {
            let layer_caches: Vec<&Option<(Tensor, Tensor)>> =
                seq_kv_caches.iter().map(|seq| &seq[layer_idx]).collect();

            if max_kv_len > 0 && layer_caches.iter().any(|cache| cache.is_some()) {
                let requested_width = max_kv_len + extra_room;
                let (buf_k, buf_v) = self.ensure_batch_decode_kv_workspace(
                    layer_idx,
                    seq_kv_caches.len(),
                    kv_heads,
                    head_dim,
                    requested_width,
                    &device,
                    self.dtype,
                    &mut stats,
                )?;

                let t_pad_stack = Instant::now();
                for (row, cache) in layer_caches.iter().enumerate() {
                    let Some((k, v)) = cache else {
                        continue;
                    };

                    let cur_len = k.dim(2)?;
                    if cur_len > max_kv_len {
                        candle_core::bail!(
                            "KV cache length {cur_len} exceeds batch max length {max_kv_len}"
                        );
                    }
                    let offset = max_kv_len - cur_len;
                    let row_k = buf_k.narrow(0, row, 1)?;
                    let row_v = buf_v.narrow(0, row, 1)?;
                    let t_contiguous = Instant::now();
                    let k = k.contiguous()?;
                    let v = v.contiguous()?;
                    stats.contiguous_us += elapsed_us(t_contiguous);
                    row_k.slice_set(&k, 2, offset)?;
                    row_v.slice_set(&v, 2, offset)?;
                }
                stats.pad_stack_us += elapsed_us(t_pad_stack);

                let t_assign = Instant::now();
                let layer = &mut self.layers[layer_idx];
                layer.self_attn.kv_cache = Some((buf_k, buf_v));
                layer.self_attn.cache_seq_len = max_kv_len;
                stats.cache_assign_us += elapsed_us(t_assign);
            } else {
                let t_assign = Instant::now();
                let layer = &mut self.layers[layer_idx];
                layer.self_attn.kv_cache = None;
                layer.self_attn.cache_seq_len = 0;
                stats.cache_assign_us += elapsed_us(t_assign);
            }
        }

        stats.total_us = elapsed_us(total_start);
        self.last_batch_decode_setup_stats = stats;
        Ok((kv_lens, max_kv_len))
    }

    /// Round 9 fast path: adopt a per-layer batched gather output as the KV
    /// workspace contents. Equal-length rows use one direct copy per layer;
    /// CUDA BF16 ragged rows use a fused right-aligned copy kernel and other
    /// devices keep the defensive rowwise path.
    ///
    /// `per_layer` is `[layer]([batch, kv_heads, max_total_len, head_dim] K, ...V)`,
    /// right-aligned per-row inside `max_total_len`. `kv_lens` is the per-row
    /// total token count.
    pub fn setup_batch_decode_batched(
        &mut self,
        per_layer: &[Option<(Tensor, Tensor)>],
        kv_lens: &[usize],
        max_total_len: usize,
        extra_room: usize,
    ) -> Result<(Vec<usize>, usize)> {
        let total_start = Instant::now();
        let mut stats = BatchDecodeSetupStats {
            layers: self.layers.len(),
            sequences: kv_lens.len(),
            ..BatchDecodeSetupStats::default()
        };
        let kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim();
        let device = self.embed_tokens.embeddings().device().clone();

        if per_layer.len() != self.layers.len() {
            candle_core::bail!(
                "setup_batch_decode_batched expects {} layers, got {}",
                self.layers.len(),
                per_layer.len()
            );
        }

        if max_total_len == 0 {
            for layer_idx in 0..self.layers.len() {
                let layer = &mut self.layers[layer_idx];
                layer.self_attn.kv_cache = None;
                layer.self_attn.cache_seq_len = 0;
            }
            stats.total_us = elapsed_us(total_start);
            self.last_batch_decode_setup_stats = stats;
            return Ok((kv_lens.to_vec(), 0));
        }

        let batch = kv_lens.len();
        let requested_width = max_total_len + extra_room;
        let ragged_kv_lens_tensor = if self.dtype == DType::BF16
            && device.is_cuda()
            && env_flag_default("CRANE_BATCH_KV_RAGGED_COPY", true)
            && kv_lens.iter().any(|&len| len != max_total_len)
        {
            let mut kv_lens_u32 = Vec::with_capacity(kv_lens.len());
            for &len in kv_lens {
                kv_lens_u32.push(u32::try_from(len).map_err(|_| {
                    candle_core::Error::Msg(format!(
                        "KV length {len} exceeds u32 range for CUDA ragged setup"
                    ))
                })?);
            }
            Some(
                self.batch_decode_kv_lens_buffer
                    .upload_1d(&kv_lens_u32, &device)?,
            )
        } else {
            None
        };
        for layer_idx in 0..self.layers.len() {
            let Some((layer_k, layer_v)) = per_layer[layer_idx].as_ref() else {
                let layer = &mut self.layers[layer_idx];
                layer.self_attn.kv_cache = None;
                layer.self_attn.cache_seq_len = 0;
                continue;
            };
            let (source_batch, source_heads, source_width, source_head_dim) = layer_k.dims4()?;
            if layer_v.dims() != layer_k.dims() {
                candle_core::bail!("batched setup layer {layer_idx} K/V shapes differ")
            }
            if source_batch != batch
                || source_heads != kv_heads
                || source_width != max_total_len
                || source_head_dim != head_dim
            {
                candle_core::bail!(
                    "batched setup layer {layer_idx} shape {:?} does not match [{batch}, {kv_heads}, {max_total_len}, {head_dim}]",
                    layer_k.dims()
                )
            }
            for (row, &kv_len) in kv_lens.iter().enumerate() {
                if kv_len > max_total_len {
                    candle_core::bail!(
                        "row {row} KV length {kv_len} exceeds batched width {max_total_len}"
                    )
                }
            }

            let (buf_k, buf_v) = self.ensure_batch_decode_kv_workspace(
                layer_idx,
                batch,
                kv_heads,
                head_dim,
                requested_width,
                &device,
                self.dtype,
                &mut stats,
            )?;

            let t_pad_stack = Instant::now();
            if kv_lens.iter().all(|&len| len == max_total_len) {
                stats.batched_equal_length_layers += 1;
                buf_k.slice_set(layer_k, 2, 0)?;
                buf_v.slice_set(layer_v, 2, 0)?;
            } else {
                stats.batched_ragged_layers += 1;
                if let Some(kv_lens_tensor) = ragged_kv_lens_tensor.as_ref() {
                    stats.batched_ragged_rows +=
                        kv_lens.iter().filter(|&&len| len > 0).count() as u64;
                    crate::fused_ops::batch_kv_copy_ragged_bf16(
                        &buf_k,
                        &buf_v,
                        layer_k,
                        layer_v,
                        kv_lens_tensor,
                        kv_heads,
                        head_dim,
                    )?;
                } else {
                    for (row, &kv_len) in kv_lens.iter().enumerate() {
                        if kv_len == 0 {
                            continue;
                        }
                        stats.batched_ragged_rows += 1;
                        let offset = max_total_len - kv_len;
                        let src_k = layer_k
                            .narrow(0, row, 1)?
                            .narrow(2, offset, kv_len)?
                            .contiguous()?;
                        let src_v = layer_v
                            .narrow(0, row, 1)?
                            .narrow(2, offset, kv_len)?
                            .contiguous()?;
                        let dst_k = buf_k.narrow(0, row, 1)?;
                        let dst_v = buf_v.narrow(0, row, 1)?;
                        dst_k.slice_set(&src_k, 2, offset)?;
                        dst_v.slice_set(&src_v, 2, offset)?;
                    }
                }
            }
            stats.pad_stack_us += elapsed_us(t_pad_stack);

            let t_assign = Instant::now();
            let layer = &mut self.layers[layer_idx];
            layer.self_attn.kv_cache = Some((buf_k, buf_v));
            layer.self_attn.cache_seq_len = max_total_len;
            stats.cache_assign_us += elapsed_us(t_assign);
        }

        stats.total_us = elapsed_us(total_start);
        self.last_batch_decode_setup_stats = stats;
        Ok((kv_lens.to_vec(), max_total_len))
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure_batch_decode_kv_workspace(
        &mut self,
        layer_idx: usize,
        batch: usize,
        kv_heads: usize,
        head_dim: usize,
        requested_width: usize,
        device: &Device,
        dtype: DType,
        stats: &mut BatchDecodeSetupStats,
    ) -> Result<(Tensor, Tensor)> {
        let capacity = requested_width.max(1).next_power_of_two();
        let slot = self
            .batch_decode_kv_workspaces
            .get_mut(layer_idx)
            .ok_or_else(|| {
                candle_core::Error::Msg("batch decode workspace layer out of range".into())
            })?;
        let needs_alloc = slot.as_ref().map_or(true, |workspace| {
            !workspace.matches(batch, kv_heads, head_dim, capacity, dtype, device)
        });
        if needs_alloc {
            let t_alloc = Instant::now();
            let k = Tensor::zeros((batch, kv_heads, capacity, head_dim), dtype, &device)?;
            let v = Tensor::zeros((batch, kv_heads, capacity, head_dim), dtype, &device)?;
            *slot = Some(BatchDecodeKvWorkspace {
                k,
                v,
                batch,
                kv_heads,
                head_dim,
                capacity,
                dtype,
            });
            self.batch_decode_workspace_generation =
                self.batch_decode_workspace_generation.wrapping_add(1);
            stats.extra_room_alloc_us += elapsed_us(t_alloc);
        }
        let workspace = slot
            .as_ref()
            .expect("batch decode workspace must exist after ensure");
        Ok((workspace.k.clone(), workspace.v.clone()))
    }

    pub fn last_batch_decode_setup_stats(&self) -> BatchDecodeSetupStats {
        self.last_batch_decode_setup_stats
    }

    pub fn batch_decode_workspace_generation(&self) -> u64 {
        self.batch_decode_workspace_generation
    }

    /// Run one batched decode step.
    pub fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        _batch_kv_info: Option<(&[usize], usize)>,
    ) -> Result<Tensor> {
        self.step_batch_decode_impl(input_ids, positions, None, None, attention_mask, None, None)
    }

    pub fn step_batch_decode_fixed_width(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        fixed_cache_width: usize,
    ) -> Result<Tensor> {
        self.step_batch_decode_impl(
            input_ids,
            positions,
            None,
            None,
            attention_mask,
            None,
            Some(fixed_cache_width),
        )
    }

    pub fn step_batch_decode_fixed_width_with_position_ids(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        position_ids: &Tensor,
        append_offset: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        fixed_cache_width: usize,
    ) -> Result<Tensor> {
        self.step_batch_decode_impl(
            input_ids,
            positions,
            Some(position_ids),
            append_offset,
            attention_mask,
            None,
            Some(fixed_cache_width),
        )
    }

    pub fn step_batch_decode_with_paged_attention(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        paged_attention: &crate::fused_ops::PagedAttentionDecodeContext,
    ) -> Result<Tensor> {
        if paged_attention.batch_size() != input_ids.dim(0)? {
            candle_core::bail!(
                "paged attention batch size {} does not match input batch {}",
                paged_attention.batch_size(),
                input_ids.dim(0)?
            )
        }
        self.paged_attention_metadata.upload(
            input_ids.device(),
            &paged_attention.indptr,
            &paged_attention.indices,
            &paged_attention.last_page_lens,
            &paged_attention.seq_lens,
        )?;
        self.step_batch_decode_impl(
            input_ids,
            positions,
            None,
            None,
            attention_mask,
            Some(paged_attention),
            None,
        )
    }

    pub fn last_paged_attention_layer_hits(&self) -> usize {
        self.last_paged_attention_layer_hits
    }

    pub fn last_paged_attention_layer_fallbacks(&self) -> usize {
        self.last_paged_attention_layer_fallbacks
    }

    fn step_batch_decode_impl(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        position_ids: Option<&Tensor>,
        append_offset: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        paged_attention: Option<&crate::fused_ops::PagedAttentionDecodeContext>,
        fixed_cache_width: Option<usize>,
    ) -> Result<Tensor> {
        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;

        let max_pos = positions.iter().copied().max().unwrap_or(0) + 1;
        let device = input_ids.device();
        let (_, decode_seq_len) = input_ids.dims2()?;
        if let Some(append_offset) = append_offset {
            if append_offset.dtype() != DType::U32 {
                candle_core::bail!(
                    "append offset tensor must be U32, got {:?}",
                    append_offset.dtype()
                );
            }
            if append_offset.dims() != &[1] {
                candle_core::bail!(
                    "append offset tensor shape {:?} does not match [1]",
                    append_offset.dims()
                );
            }
            if !append_offset.device().same_device(device) {
                candle_core::bail!("append offset tensor must be on the same device as input ids");
            }
        }
        let use_indexed_rope = decode_seq_len == 1
            && self.dtype == DType::BF16
            && device.is_cuda()
            && env_flag_default("CRANE_FUSED_ROPE_INDEXED", true);
        let pos_tensor = if let Some(position_ids) = position_ids {
            if position_ids.dtype() != DType::U32 {
                candle_core::bail!(
                    "position ids tensor must be U32, got {:?}",
                    position_ids.dtype()
                );
            }
            if !position_ids.device().same_device(device) {
                candle_core::bail!("position ids tensor must be on the same device as input ids");
            }
            if position_ids.dims() != &[positions.len()] {
                candle_core::bail!(
                    "position ids tensor shape {:?} does not match batch size {}",
                    position_ids.dims(),
                    positions.len()
                );
            }
            position_ids.clone()
        } else {
            let pos_ids: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
            Tensor::new(pos_ids.as_slice(), device)?
        };
        let (cos, sin, rope_positions) = if use_indexed_rope {
            // CUDA Graph correctness: the indexed-rope kernel reads
            // `max_position` as a host scalar baked into launch args. A
            // `narrow(0, 0, max_pos)` view would freeze a per-round value into
            // the captured graph, causing replays at later rounds to clamp
            // positions to the older bound and produce RoPE drift. Pass the
            // full pre-allocated tables so `max_position` is constant
            // (= max_position_embeddings) across all captures and replays.
            // See docs/qwen3/benchmarks/qwen3_round5_cuda_graph_2026_05_08.md.
            let (full_cos_table, full_sin_table) = self.rotary_emb.full_tables();
            (full_cos_table, full_sin_table, Some(pos_tensor))
        } else {
            let (full_cos, full_sin) = self.rotary_emb.forward(max_pos)?;
            let cos = full_cos
                .index_select(&pos_tensor, 0)?
                .to_dtype(self.dtype)?
                .unsqueeze(1)?;
            let sin = full_sin
                .index_select(&pos_tensor, 0)?
                .to_dtype(self.dtype)?
                .unsqueeze(1)?;
            (cos, sin, None)
        };

        let mut hidden_states = hidden_states;
        self.last_paged_attention_layer_hits = 0;
        self.last_paged_attention_layer_fallbacks = 0;
        let paged_attention_layer_hits = std::cell::Cell::new(0usize);
        let paged_attention_layer_fallbacks = std::cell::Cell::new(0usize);
        let paged_attention = paged_attention.map(|context| PagedAttentionDecodeRun {
            context,
            metadata: &self.paged_attention_metadata,
            layer_hits: &paged_attention_layer_hits,
            layer_fallbacks: &paged_attention_layer_fallbacks,
        });
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            hidden_states = layer.forward(
                &hidden_states,
                &cos,
                &sin,
                rope_positions.as_ref(),
                attention_mask,
                layer_idx,
                paged_attention.as_ref(),
                fixed_cache_width,
                append_offset,
            )?;
        }
        self.last_paged_attention_layer_hits = paged_attention_layer_hits.get();
        self.last_paged_attention_layer_fallbacks = paged_attention_layer_fallbacks.get();

        let hidden_states = self.norm.forward(&hidden_states)?;
        self.lm_head.forward(&hidden_states) // [N, 1, vocab]
    }

    /// Extract per-sequence KV caches from batched state.
    pub fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        let keep = vec![true; kv_lens.len()];
        self.extract_batch_kv_selective(kv_lens, original_max_kv, rounds_done, &keep)
    }

    pub fn extract_batch_kv_selective(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
        keep: &[bool],
    ) -> Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        let total_start = Instant::now();
        let n_seqs = kv_lens.len();
        let num_layers = self.layers.len();
        let mut stats = BatchDecodeExtractStats {
            layers: num_layers,
            sequences: n_seqs,
            ..BatchDecodeExtractStats::default()
        };
        let mut result: Vec<Vec<Option<(Tensor, Tensor)>>> = (0..n_seqs)
            .map(|_| Vec::with_capacity(num_layers))
            .collect();

        for layer in self.layers.iter_mut() {
            if let Some((ref full_k, ref full_v)) = layer.self_attn.kv_cache {
                let extracted = extract_batch_layer_kv(
                    full_k,
                    full_v,
                    kv_lens,
                    original_max_kv,
                    rounds_done,
                    keep,
                    &mut stats,
                )?;
                for (seq, cache) in result.iter_mut().zip(extracted.into_iter()) {
                    seq.push(cache);
                }
            } else {
                for i in 0..n_seqs {
                    result[i].push(None);
                }
            }
            let t_clear = Instant::now();
            layer.self_attn.kv_cache = None;
            layer.self_attn.cache_seq_len = 0;
            stats.cache_clear_us += elapsed_us(t_clear);
        }

        stats.total_us = elapsed_us(total_start);
        self.last_batch_decode_extract_stats = stats;
        Ok(result)
    }

    pub fn last_batch_decode_extract_stats(&self) -> BatchDecodeExtractStats {
        self.last_batch_decode_extract_stats
    }

    /// Access the model config.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Access the model dtype.
    pub fn model_dtype(&self) -> DType {
        self.dtype
    }
}

// ── Utilities ───────────────────────────────────────────────────────────

/// Build attention mask for batched decode with padding-aware masking.
pub fn build_batch_decode_mask(
    kv_lens: &[usize],
    original_max_kv: usize,
    total_width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Option<Tensor>> {
    if kv_lens.iter().all(|&l| l == original_max_kv) {
        return Ok(None);
    }
    let n = kv_lens.len();
    let mut mask_data = vec![0f32; n * total_width];
    for i in 0..n {
        let pad_end = (original_max_kv - kv_lens[i]).min(total_width);
        for j in 0..pad_end {
            mask_data[i * total_width + j] = -1e9;
        }
    }
    let mask = Tensor::from_vec(mask_data, (n, total_width), device)?.to_dtype(dtype)?;
    Ok(Some(mask.unsqueeze(1)?.unsqueeze(1)?))
}

/// Build a fixed-width decode mask for CUDA Graph candidate rounds.
///
/// `active_width` is the prefix of `total_width` that contains real history plus
/// generated tokens for the current round. Slots after it stay in the static K/V
/// view but are masked out so attention math matches the dynamic-width path.
pub fn build_batch_decode_fixed_width_mask(
    kv_lens: &[usize],
    original_max_kv: usize,
    total_width: usize,
    active_width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Option<Tensor>> {
    if active_width > total_width {
        candle_core::bail!(
            "active decode mask width {active_width} exceeds fixed width {total_width}"
        );
    }
    let needs_left_pad = kv_lens.iter().any(|&len| len != original_max_kv);
    if !needs_left_pad && active_width == total_width {
        return Ok(None);
    }

    let n = kv_lens.len();
    let mut mask_data = vec![0f32; n * total_width];
    for i in 0..n {
        let pad_end = (original_max_kv - kv_lens[i]).min(total_width);
        for j in 0..pad_end {
            mask_data[i * total_width + j] = -1e9;
        }
        for j in active_width..total_width {
            mask_data[i * total_width + j] = -1e9;
        }
    }
    let mask = Tensor::from_vec(mask_data, (n, total_width), device)?.to_dtype(dtype)?;
    Ok(Some(mask.unsqueeze(1)?.unsqueeze(1)?))
}

/// Pad per-sequence KV caches to `max_len` and stack (right-aligned).
#[allow(dead_code)]
fn pad_and_stack_kv_caches(
    caches: &[&Option<(Tensor, Tensor)>],
    max_len: usize,
    kv_heads: usize,
    head_dim: usize,
    device: &Device,
    dtype: DType,
    stats: &mut BatchDecodeSetupStats,
) -> Result<Option<(Tensor, Tensor)>> {
    if max_len == 0 {
        return Ok(None);
    }

    let t_pad_stack = Instant::now();
    let n = caches.len();
    let stacked_k = Tensor::zeros((n, kv_heads, max_len, head_dim), dtype, device)?;
    let stacked_v = Tensor::zeros((n, kv_heads, max_len, head_dim), dtype, device)?;

    for (row, cache) in caches.iter().enumerate() {
        let Some((k, v)) = cache else {
            continue;
        };

        let cur_len = k.dim(2)?;
        if cur_len > max_len {
            candle_core::bail!("KV cache length {cur_len} exceeds batch max length {max_len}");
        }
        let offset = max_len - cur_len;
        let row_k = stacked_k.narrow(0, row, 1)?;
        let row_v = stacked_v.narrow(0, row, 1)?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        row_k.slice_set(&k, 2, offset)?;
        row_v.slice_set(&v, 2, offset)?;
    }
    stats.pad_stack_us += elapsed_us(t_pad_stack);
    Ok(Some((stacked_k, stacked_v)))
}

fn extract_batch_layer_kv(
    full_k: &Tensor,
    full_v: &Tensor,
    kv_lens: &[usize],
    original_max_kv: usize,
    rounds_done: usize,
    keep: &[bool],
    stats: &mut BatchDecodeExtractStats,
) -> Result<Vec<Option<(Tensor, Tensor)>>> {
    let n_seqs = kv_lens.len();
    let mut result = Vec::with_capacity(n_seqs);
    for i in 0..n_seqs {
        if !keep.get(i).copied().unwrap_or(true) {
            result.push(None);
            continue;
        }

        let kv_len = kv_lens[i];
        if kv_len > original_max_kv {
            candle_core::bail!("KV length {kv_len} exceeds original batch max {original_max_kv}");
        }

        let t_narrow = Instant::now();
        let row_k = full_k.narrow(0, i, 1)?;
        let row_v = full_v.narrow(0, i, 1)?;
        let total = kv_len + rounds_done;
        let offset = original_max_kv - kv_len;
        let clean_k = row_k.narrow(2, offset, total)?;
        let clean_v = row_v.narrow(2, offset, total)?;
        stats.narrow_us += elapsed_us(t_narrow);

        let t_contiguous = Instant::now();
        result.push(Some((clean_k.contiguous()?, clean_v.contiguous()?)));
        stats.contiguous_us += elapsed_us(t_contiguous);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_and_stack_kv_caches_right_aligns_rows() {
        let device = Device::Cpu;
        let short_k = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 2, 2), &device).unwrap();
        let short_v =
            Tensor::from_vec(vec![11.0f32, 12.0, 13.0, 14.0], (1, 1, 2, 2), &device).unwrap();
        let full_k = Tensor::from_vec(
            vec![21.0f32, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0],
            (1, 1, 4, 2),
            &device,
        )
        .unwrap();
        let full_v = Tensor::from_vec(
            vec![31.0f32, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0],
            (1, 1, 4, 2),
            &device,
        )
        .unwrap();
        let caches = vec![Some((short_k, short_v)), Some((full_k, full_v)), None];
        let cache_refs: Vec<&Option<(Tensor, Tensor)>> = caches.iter().collect();
        let mut stats = BatchDecodeSetupStats::default();

        let (stacked_k, stacked_v) =
            pad_and_stack_kv_caches(&cache_refs, 4, 1, 2, &device, DType::F32, &mut stats)
                .unwrap()
                .unwrap();

        assert_eq!(stacked_k.dims(), &[3, 1, 4, 2]);
        assert_eq!(
            stacked_k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![
                0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
                28.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );
        assert_eq!(
            stacked_v.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![
                0.0, 0.0, 0.0, 0.0, 11.0, 12.0, 13.0, 14.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
                37.0, 38.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );
    }

    #[test]
    fn pad_and_stack_kv_caches_empty_max_len_returns_none() {
        let device = Device::Cpu;
        let caches = vec![None];
        let cache_refs: Vec<&Option<(Tensor, Tensor)>> = caches.iter().collect();
        let mut stats = BatchDecodeSetupStats::default();

        let packed =
            pad_and_stack_kv_caches(&cache_refs, 0, 1, 2, &device, DType::F32, &mut stats).unwrap();

        assert!(packed.is_none());
    }

    #[test]
    fn fixed_width_decode_mask_masks_left_pad_and_future_slots() {
        let device = Device::Cpu;
        let mask = build_batch_decode_fixed_width_mask(&[2, 4], 4, 7, 5, &device, DType::F32)
            .unwrap()
            .unwrap();

        assert_eq!(mask.dims(), &[2, 1, 1, 7]);
        assert_eq!(
            mask.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![-1e9, -1e9, 0.0, 0.0, 0.0, -1e9, -1e9, 0.0, 0.0, 0.0, 0.0, 0.0, -1e9, -1e9,]
        );
    }

    #[test]
    fn fixed_width_decode_mask_returns_none_when_no_slots_are_masked() {
        let device = Device::Cpu;
        let mask =
            build_batch_decode_fixed_width_mask(&[4, 4], 4, 5, 5, &device, DType::F32).unwrap();

        assert!(mask.is_none());
    }

    #[test]
    fn extract_batch_layer_kv_skips_non_live_rows() {
        let device = Device::Cpu;
        let full_k = Tensor::from_vec(
            vec![
                0.0f32, 0.0, 1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 0.0, 21.0,
                22.0, 23.0, 24.0, 25.0,
            ],
            (3, 1, 6, 1),
            &device,
        )
        .unwrap();
        let full_v = Tensor::from_vec(
            vec![
                0.0f32, 0.0, 31.0, 32.0, 33.0, 34.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 0.0, 51.0,
                52.0, 53.0, 54.0, 55.0,
            ],
            (3, 1, 6, 1),
            &device,
        )
        .unwrap();
        let mut stats = BatchDecodeExtractStats::default();

        let extracted = extract_batch_layer_kv(
            &full_k,
            &full_v,
            &[3, 5, 4],
            5,
            1,
            &[true, false, true],
            &mut stats,
        )
        .unwrap();

        assert_eq!(extracted.len(), 3);
        let (first_k, first_v) = extracted[0].as_ref().unwrap();
        assert_eq!(
            first_k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            first_v.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![31.0, 32.0, 33.0, 34.0]
        );
        assert!(extracted[1].is_none());
        let (third_k, third_v) = extracted[2].as_ref().unwrap();
        assert_eq!(
            third_k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![21.0, 22.0, 23.0, 24.0, 25.0]
        );
        assert_eq!(
            third_v.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![51.0, 52.0, 53.0, 54.0, 55.0]
        );
    }
}
