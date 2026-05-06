use super::*;

/// Reusable CUDA buffers for batched greedy sampling.
///
/// The hot decode path calls sampling once per batch-decode round. Allocating
/// output tokens and repetition-penalty metadata tensors in each round shows up
/// in the small-Qwen3 translation workload, so this cache keeps raw CudaSlices
/// alive and overwrites their contents with HtoD copies before launching kernels.
#[cfg(feature = "cuda")]
#[derive(Default)]
pub struct BatchGreedyCudaBuffers {
    output_tokens: Option<CudaSlice<u32>>,
    recent_token_ids: Option<CudaSlice<u32>>,
    recent_lengths: Option<CudaSlice<u32>>,
    penalties: Option<CudaSlice<f32>>,
    output_capacity: usize,
    recent_token_capacity: usize,
    recent_length_capacity: usize,
    penalty_capacity: usize,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub struct BatchNonGreedyCudaBuffers {
    output_tokens: Option<CudaSlice<i32>>,
    temperatures: Option<CudaSlice<f32>>,
    top_ks: Option<CudaSlice<u32>>,
    top_ps: Option<CudaSlice<f32>>,
    seeds: Option<CudaSlice<u64>>,
    recent_token_ids: Option<CudaSlice<u32>>,
    recent_lengths: Option<CudaSlice<u32>>,
    penalties: Option<CudaSlice<f32>>,
    output_capacity: usize,
    batch_capacity: usize,
    recent_token_capacity: usize,
    recent_length_capacity: usize,
    penalty_capacity: usize,
}

#[cfg(feature = "cuda")]
impl BatchNonGreedyCudaBuffers {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_output(&mut self, dev: &candle_core::CudaDevice, len: usize) -> Result<()> {
        if self.output_capacity < len {
            self.output_tokens = Some(unsafe { dev.alloc::<i32>(len)? });
            self.output_capacity = len;
        }
        Ok(())
    }

    fn ensure_batch_metadata(&mut self, dev: &candle_core::CudaDevice, len: usize) -> Result<()> {
        if self.batch_capacity < len {
            self.temperatures = Some(unsafe { dev.alloc::<f32>(len)? });
            self.top_ks = Some(unsafe { dev.alloc::<u32>(len)? });
            self.top_ps = Some(unsafe { dev.alloc::<f32>(len)? });
            self.seeds = Some(unsafe { dev.alloc::<u64>(len)? });
            self.batch_capacity = len;
        }
        Ok(())
    }

    fn upload_f32(
        slot: &mut Option<CudaSlice<f32>>,
        dev: &candle_core::CudaDevice,
        src: &[f32],
        missing: &str,
    ) -> Result<()> {
        let dst = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg(missing.into()))?;
        let mut dst = dst.slice_mut(0..src.len());
        dev.memcpy_htod(src, &mut dst)?;
        Ok(())
    }

    fn upload_u32(
        slot: &mut Option<CudaSlice<u32>>,
        dev: &candle_core::CudaDevice,
        src: &[u32],
        missing: &str,
    ) -> Result<()> {
        let dst = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg(missing.into()))?;
        let mut dst = dst.slice_mut(0..src.len());
        dev.memcpy_htod(src, &mut dst)?;
        Ok(())
    }

    fn upload_u64(
        slot: &mut Option<CudaSlice<u64>>,
        dev: &candle_core::CudaDevice,
        src: &[u64],
        missing: &str,
    ) -> Result<()> {
        let dst = slot
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg(missing.into()))?;
        let mut dst = dst.slice_mut(0..src.len());
        dev.memcpy_htod(src, &mut dst)?;
        Ok(())
    }

    fn upload_recent_tokens(&mut self, dev: &candle_core::CudaDevice, src: &[u32]) -> Result<()> {
        if self.recent_token_capacity < src.len() {
            self.recent_token_ids = Some(unsafe { dev.alloc::<u32>(src.len())? });
            self.recent_token_capacity = src.len();
        }
        if !src.is_empty() {
            Self::upload_u32(
                &mut self.recent_token_ids,
                dev,
                src,
                "missing non-greedy recent token buffer",
            )?;
        }
        Ok(())
    }

    fn upload_recent_lengths(&mut self, dev: &candle_core::CudaDevice, src: &[u32]) -> Result<()> {
        if self.recent_length_capacity < src.len() {
            self.recent_lengths = Some(unsafe { dev.alloc::<u32>(src.len())? });
            self.recent_length_capacity = src.len();
        }
        if !src.is_empty() {
            Self::upload_u32(
                &mut self.recent_lengths,
                dev,
                src,
                "missing non-greedy recent length buffer",
            )?;
        }
        Ok(())
    }

    fn upload_penalties(&mut self, dev: &candle_core::CudaDevice, src: &[f32]) -> Result<()> {
        if self.penalty_capacity < src.len() {
            self.penalties = Some(unsafe { dev.alloc::<f32>(src.len())? });
            self.penalty_capacity = src.len();
        }
        if !src.is_empty() {
            Self::upload_f32(
                &mut self.penalties,
                dev,
                src,
                "missing non-greedy penalty buffer",
            )?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl BatchGreedyCudaBuffers {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_output(&mut self, dev: &candle_core::CudaDevice, len: usize) -> Result<()> {
        if self.output_capacity < len {
            self.output_tokens = Some(unsafe { dev.alloc::<u32>(len)? });
            self.output_capacity = len;
        }
        Ok(())
    }

    pub fn output_tokens_tensor_from(&self, anchor: &Tensor, batch_size: usize) -> Result<Tensor> {
        if batch_size == 0 {
            candle_core::bail!("output_tokens_tensor_from requires non-empty batch")
        }
        let output_tokens = self
            .output_tokens
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("missing output token buffer".into()))?;
        if self.output_capacity < batch_size {
            candle_core::bail!(
                "output_tokens_tensor_from: buffer capacity {} < requested {}",
                self.output_capacity,
                batch_size
            )
        }
        anchor.apply_op1_no_bwd(&BatchGreedyOutputTokensTensor {
            output_tokens: output_tokens.clone(),
            batch_size,
        })
    }

    fn upload_recent_tokens(&mut self, dev: &candle_core::CudaDevice, src: &[u32]) -> Result<()> {
        if self.recent_token_capacity < src.len() {
            self.recent_token_ids = Some(unsafe { dev.alloc::<u32>(src.len())? });
            self.recent_token_capacity = src.len();
        }
        if !src.is_empty() {
            let dst = self
                .recent_token_ids
                .as_mut()
                .ok_or_else(|| candle_core::Error::Msg("missing recent token buffer".into()))?;
            let mut dst = dst.slice_mut(0..src.len());
            dev.memcpy_htod(src, &mut dst)?;
        }
        Ok(())
    }

    fn upload_recent_lengths(&mut self, dev: &candle_core::CudaDevice, src: &[u32]) -> Result<()> {
        if self.recent_length_capacity < src.len() {
            self.recent_lengths = Some(unsafe { dev.alloc::<u32>(src.len())? });
            self.recent_length_capacity = src.len();
        }
        if !src.is_empty() {
            let dst = self
                .recent_lengths
                .as_mut()
                .ok_or_else(|| candle_core::Error::Msg("missing recent length buffer".into()))?;
            let mut dst = dst.slice_mut(0..src.len());
            dev.memcpy_htod(src, &mut dst)?;
        }
        Ok(())
    }

    fn upload_penalties(&mut self, dev: &candle_core::CudaDevice, src: &[f32]) -> Result<()> {
        if self.penalty_capacity < src.len() {
            self.penalties = Some(unsafe { dev.alloc::<f32>(src.len())? });
            self.penalty_capacity = src.len();
        }
        if !src.is_empty() {
            let dst = self
                .penalties
                .as_mut()
                .ok_or_else(|| candle_core::Error::Msg("missing penalty buffer".into()))?;
            let mut dst = dst.slice_mut(0..src.len());
            dev.memcpy_htod(src, &mut dst)?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
struct BatchGreedyOutputTokensTensor {
    output_tokens: CudaSlice<u32>,
    batch_size: usize,
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for BatchGreedyOutputTokensTensor {
    fn name(&self) -> &'static str {
        "batch_greedy_output_tokens_tensor"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("batch_greedy_output_tokens_tensor requires CUDA storage")
    }

    fn cuda_fwd(&self, storage: &CudaStorage, _layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let dst = CudaStorage::wrap_cuda_slice(self.output_tokens.clone(), storage.device.clone());
        Ok((dst, Shape::from_dims(&[self.batch_size, 1])))
    }
}

/// Returns the token index as u32.
#[cfg(feature = "cuda")]
pub fn gpu_argmax(logits: &Tensor) -> Result<u32> {
    let device = logits.device();
    let dev = match device {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("gpu_argmax requires CUDA device"),
    };

    let logits = logits.contiguous()?.flatten_all()?;
    let vocab_size = logits.elem_count();

    // Get the underlying storage
    let (storage, layout) = logits.storage_and_layout();
    let cuda_storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("gpu_argmax: expected CUDA storage"),
    };

    let (o1, _o2) = match layout.contiguous_offsets() {
        Some(o) => o,
        None => candle_core::bail!("gpu_argmax: logits must be contiguous"),
    };

    // Phase 1: per-block reduction
    let num_blocks = 256u32.min((vocab_size as u32 + 1023) / 1024);
    let block_size = 256u32;

    let func1 = load_func!(dev, "gpu_argmax_bf16_phase1")?;
    let func2 = load_func!(dev, "gpu_argmax_phase2")?;

    // Allocate temporary buffers for block results
    let block_max_vals: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32> =
        unsafe { dev.alloc::<f32>(num_blocks as usize)? };
    let block_max_idxs: candle_core::cuda_backend::cudarc::driver::CudaSlice<i32> =
        unsafe { dev.alloc::<i32>(num_blocks as usize)? };
    let output_token: candle_core::cuda_backend::cudarc::driver::CudaSlice<i32> =
        unsafe { dev.alloc::<i32>(1)? };

    // Phase 1 launch
    let cfg1 = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    match &cuda_storage.slice {
        CudaStorageSlice::BF16(s) => {
            let s = s.slice(o1..);
            let mut builder = func1.builder();
            builder.arg(&s);
            builder.arg(&block_max_vals);
            builder.arg(&block_max_idxs);
            let vs = vocab_size as i32;
            builder.arg(&vs);
            unsafe { builder.launch(cfg1) }.w()?;
        }
        _ => candle_core::bail!("gpu_argmax currently only supports BF16"),
    }

    // Phase 2: reduce block results
    let cfg2 = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (num_blocks.min(256), 1, 1),
        shared_mem_bytes: 0,
    };

    {
        let mut builder = func2.builder();
        builder.arg(&block_max_vals);
        builder.arg(&block_max_idxs);
        builder.arg(&output_token);
        let nb = num_blocks as i32;
        builder.arg(&nb);
        unsafe { builder.launch(cfg2) }.w()?;
    }

    // DtoH: only 4 bytes!
    let result = dev.clone_dtoh(&output_token)?;
    Ok(result[0] as u32)
}

/// Batched BF16 greedy argmax for decode logits.
///
/// `logits` shape: `[batch, vocab]` or `[batch, 1, vocab]`.
/// Returns one token id per batch row.
#[cfg(feature = "cuda")]
pub fn gpu_argmax_batch(logits: &Tensor) -> Result<Vec<u32>> {
    let mut buffers = BatchGreedyCudaBuffers::new();
    gpu_argmax_batch_cached(logits, &mut buffers)
}

/// Batched BF16 greedy argmax using caller-owned CUDA output buffers.
#[cfg(feature = "cuda")]
pub fn gpu_argmax_batch_cached(
    logits: &Tensor,
    buffers: &mut BatchGreedyCudaBuffers,
) -> Result<Vec<u32>> {
    let dev = match logits.device() {
        Device::Cuda(dev) => dev.clone(),
        _ => candle_core::bail!("gpu_argmax_batch requires CUDA device"),
    };
    let batch_size = gpu_argmax_batch_kernel_only(logits, buffers)?;
    if batch_size == 0 {
        return Ok(Vec::new());
    }
    gpu_argmax_batch_readback(&dev, buffers, batch_size)
}

/// Launch the batched argmax kernel only — no DtoH. Output lives in
/// `buffers.output_tokens`. Returns the batch size (i.e. how many tokens
/// were written into the output slice). Use [`gpu_argmax_batch_readback`]
/// to materialise the tokens on the host once the GPU work has been
/// allowed to complete (or as part of a cuGraphLaunch).
///
/// This is the API used by the CUDA Graph capture path: launching the
/// kernel inside a captured region adds it as a graph node, eliminating
/// one out-of-graph `cuLaunchKernel` per decode step. The matching DtoH
/// happens after `cuGraphLaunch` returns, batched into a single sync.
#[cfg(feature = "cuda")]
pub fn gpu_argmax_batch_kernel_only(
    logits: &Tensor,
    buffers: &mut BatchGreedyCudaBuffers,
) -> Result<usize> {
    let device = logits.device();
    let dev = match device {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("gpu_argmax_batch requires CUDA device"),
    };

    let (batch_size, vocab_size) = match logits.dims() {
        [b, v] => (*b, *v),
        [b, 1, v] => (*b, *v),
        dims => candle_core::bail!(
            "gpu_argmax_batch expects [batch, vocab] or [batch, 1, vocab], got {dims:?}"
        ),
    };
    if batch_size == 0 || vocab_size == 0 {
        return Ok(0);
    }

    let logits = if logits.rank() == 3 {
        logits.squeeze(1)?.contiguous()?
    } else {
        logits.contiguous()?
    };

    let (storage, layout) = logits.storage_and_layout();
    let cuda_storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("gpu_argmax_batch: expected CUDA storage"),
    };
    let (o1, o2) = layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("gpu_argmax_batch: logits must be contiguous".into())
    })?;
    if o2 - o1 != batch_size * vocab_size {
        candle_core::bail!("gpu_argmax_batch: unexpected contiguous length")
    }

    let func = load_func!(dev, "gpu_argmax_batch_bf16")?;
    buffers.ensure_output(dev, batch_size)?;
    let output_tokens = buffers
        .output_tokens
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing output token buffer".into()))?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    match &cuda_storage.slice {
        CudaStorageSlice::BF16(s) => {
            let s = s.slice(o1..o2);
            let mut builder = func.builder();
            builder.arg(&s);
            builder.arg(output_tokens);
            let batch = batch_size as i32;
            let vocab = vocab_size as i32;
            builder.arg(&batch);
            builder.arg(&vocab);
            unsafe { builder.launch(cfg) }.w()?;
        }
        _ => candle_core::bail!("gpu_argmax_batch currently only supports BF16"),
    }

    Ok(batch_size)
}

/// Read back `batch_size` token ids from a previously-launched argmax
/// (see [`gpu_argmax_batch_kernel_only`]). This is the single DtoH that
/// waits for the captured graph (or eager kernels) to complete.
#[cfg(feature = "cuda")]
pub fn gpu_argmax_batch_readback(
    dev: &candle_core::CudaDevice,
    buffers: &BatchGreedyCudaBuffers,
    batch_size: usize,
) -> Result<Vec<u32>> {
    if batch_size == 0 {
        return Ok(Vec::new());
    }
    let output_tokens = buffers
        .output_tokens
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing output token buffer".into()))?;
    if buffers.output_capacity < batch_size {
        candle_core::bail!(
            "gpu_argmax_batch_readback: buffer capacity {} < requested {}",
            buffers.output_capacity,
            batch_size
        );
    }
    let slice = output_tokens.slice(0..batch_size);
    let result = dev.clone_dtoh(&slice)?;
    Ok(result)
}

/// Batched BF16 greedy argmax with repetition penalty applied on CUDA before argmax.
///
/// `recent_token_ids` is row-major `[batch, max_recent]`; `recent_lengths` and
/// `penalties` have length `batch`. Tokens outside `recent_lengths[row]` are ignored.
#[cfg(feature = "cuda")]
pub fn gpu_argmax_batch_with_repetition_penalty(
    logits: &Tensor,
    recent_token_ids: &[u32],
    recent_lengths: &[u32],
    penalties: &[f32],
    max_recent: usize,
) -> Result<Vec<u32>> {
    let mut buffers = BatchGreedyCudaBuffers::new();
    gpu_argmax_batch_with_repetition_penalty_cached(
        logits,
        recent_token_ids,
        recent_lengths,
        penalties,
        max_recent,
        &mut buffers,
    )
}

/// Batched BF16 greedy argmax with CUDA repetition penalty and caller-owned buffers.
#[cfg(feature = "cuda")]
pub fn gpu_argmax_batch_with_repetition_penalty_cached(
    logits: &Tensor,
    recent_token_ids: &[u32],
    recent_lengths: &[u32],
    penalties: &[f32],
    max_recent: usize,
    buffers: &mut BatchGreedyCudaBuffers,
) -> Result<Vec<u32>> {
    let device = logits.device();
    let dev = match device {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("gpu_argmax_batch_with_repetition_penalty requires CUDA device"),
    };

    let (batch_size, vocab_size) = match logits.dims() {
        [b, v] => (*b, *v),
        [b, 1, v] => (*b, *v),
        dims => candle_core::bail!(
            "gpu_argmax_batch_with_repetition_penalty expects [batch, vocab] or [batch, 1, vocab], got {dims:?}"
        ),
    };
    if batch_size == 0 || vocab_size == 0 {
        return Ok(Vec::new());
    }
    if max_recent == 0 {
        return gpu_argmax_batch_cached(logits, buffers);
    }
    if recent_lengths.len() != batch_size || penalties.len() != batch_size {
        candle_core::bail!("gpu_argmax_batch_with_repetition_penalty: metadata length mismatch")
    }
    if recent_token_ids.len() != batch_size * max_recent {
        candle_core::bail!("gpu_argmax_batch_with_repetition_penalty: recent token length mismatch")
    }

    let logits = if logits.rank() == 3 {
        logits.squeeze(1)?.contiguous()?
    } else {
        logits.contiguous()?
    };
    // Defensive copy of logits before the in-place penalty kernel. The hot
    // decode path passes the LM head output, which is technically a fresh
    // tensor that we own end-to-end, so the copy is logically unnecessary.
    // However, removing it has been observed to interact badly with the
    // batched-decode dispatch path under contended GPUs (2026-04-30 profile),
    // so we keep the copy on by default and gate the optimization behind
    // `CRANE_SAMPLING_SKIP_LOGITS_COPY=1` for further A/B work.
    let skip_copy = std::env::var("CRANE_SAMPLING_SKIP_LOGITS_COPY")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
        .unwrap_or(false);
    let logits = if max_recent > 0 && !skip_copy {
        logits.copy()?
    } else {
        logits
    };

    let (storage, layout) = logits.storage_and_layout();
    let cuda_storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("gpu_argmax_batch_with_repetition_penalty: expected CUDA storage"),
    };
    let (o1, o2) = layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(
            "gpu_argmax_batch_with_repetition_penalty: logits must be contiguous".into(),
        )
    })?;
    if o2 - o1 != batch_size * vocab_size {
        candle_core::bail!("gpu_argmax_batch_with_repetition_penalty: unexpected contiguous length")
    }

    buffers.upload_recent_tokens(dev, recent_token_ids)?;
    buffers.upload_recent_lengths(dev, recent_lengths)?;
    buffers.upload_penalties(dev, penalties)?;

    let recent = buffers
        .recent_token_ids
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing recent token buffer".into()))?
        .slice(0..recent_token_ids.len());
    let lengths = buffers
        .recent_lengths
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing recent length buffer".into()))?
        .slice(0..recent_lengths.len());
    let penalty = buffers
        .penalties
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing penalty buffer".into()))?
        .slice(0..penalties.len());

    let func = load_func!(dev, "gpu_apply_repetition_penalty_batch_bf16")?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    match &cuda_storage.slice {
        CudaStorageSlice::BF16(logits) => {
            let logits = logits.slice(o1..o2);
            let mut builder = func.builder();
            builder.arg(&logits);
            builder.arg(&recent);
            builder.arg(&lengths);
            builder.arg(&penalty);
            let batch = batch_size as i32;
            let vocab = vocab_size as i32;
            let max_recent = max_recent as i32;
            builder.arg(&batch);
            builder.arg(&vocab);
            builder.arg(&max_recent);
            unsafe { builder.launch(cfg) }.w()?;
        }
        _ => candle_core::bail!("gpu_argmax_batch_with_repetition_penalty expects BF16 logits"),
    }

    gpu_argmax_batch_cached(&logits, buffers)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_sample_topk_topp_batch_bf16_cached(
    logits: &Tensor,
    temperatures: &[f32],
    top_ks: &[u32],
    top_ps: &[f32],
    seeds: &[u64],
    recent_token_ids: &[u32],
    recent_lengths: &[u32],
    penalties: &[f32],
    max_recent: usize,
    buffers: &mut BatchNonGreedyCudaBuffers,
) -> Result<Vec<u32>> {
    let device = logits.device();
    let dev = match device {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("gpu_sample_topk_topp_batch_bf16_cached requires CUDA device"),
    };
    if logits.dtype() != DType::BF16 {
        candle_core::bail!("gpu_sample_topk_topp_batch_bf16_cached expects BF16 logits")
    }
    let (batch_size, vocab_size) = match logits.dims() {
        [b, v] => (*b, *v),
        [b, 1, v] => (*b, *v),
        dims => candle_core::bail!(
            "gpu_sample_topk_topp_batch_bf16_cached expects [batch, vocab] or [batch, 1, vocab], got {dims:?}"
        ),
    };
    if batch_size == 0 || vocab_size == 0 {
        return Ok(Vec::new());
    }
    if temperatures.len() != batch_size
        || top_ks.len() != batch_size
        || top_ps.len() != batch_size
        || seeds.len() != batch_size
        || recent_lengths.len() != batch_size
        || penalties.len() != batch_size
    {
        candle_core::bail!("gpu_sample_topk_topp_batch_bf16_cached metadata length mismatch")
    }
    if max_recent > 0 && recent_token_ids.len() != batch_size * max_recent {
        candle_core::bail!("gpu_sample_topk_topp_batch_bf16_cached recent token length mismatch")
    }
    let max_top_k = top_ks.iter().copied().max().unwrap_or(0).min(64) as usize;
    if max_top_k == 0 {
        candle_core::bail!(
            "gpu_sample_topk_topp_batch_bf16_cached requires at least one active row"
        )
    }

    let logits = if logits.rank() == 3 {
        logits.squeeze(1)?.contiguous()?
    } else {
        logits.contiguous()?
    };

    let (storage, layout) = logits.storage_and_layout();
    let cuda_storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("gpu_sample_topk_topp_batch_bf16_cached: expected CUDA storage"),
    };
    let (o1, o2) = layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(
            "gpu_sample_topk_topp_batch_bf16_cached: logits must be contiguous".into(),
        )
    })?;
    if o2 - o1 != batch_size * vocab_size {
        candle_core::bail!("gpu_sample_topk_topp_batch_bf16_cached: unexpected contiguous length")
    }

    if max_recent > 0 {
        buffers.upload_recent_tokens(dev, recent_token_ids)?;
        buffers.upload_recent_lengths(dev, recent_lengths)?;
        buffers.upload_penalties(dev, penalties)?;
        let recent = buffers
            .recent_token_ids
            .as_ref()
            .ok_or_else(|| {
                candle_core::Error::Msg("missing non-greedy recent token buffer".into())
            })?
            .slice(0..recent_token_ids.len());
        let lengths = buffers
            .recent_lengths
            .as_ref()
            .ok_or_else(|| {
                candle_core::Error::Msg("missing non-greedy recent length buffer".into())
            })?
            .slice(0..recent_lengths.len());
        let penalty = buffers
            .penalties
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("missing non-greedy penalty buffer".into()))?
            .slice(0..penalties.len());
        let func = load_func!(dev, "gpu_apply_repetition_penalty_batch_bf16")?;
        match &cuda_storage.slice {
            CudaStorageSlice::BF16(logits) => {
                let logits = logits.slice(o1..o2);
                let mut builder = func.builder();
                builder.arg(&logits);
                builder.arg(&recent);
                builder.arg(&lengths);
                builder.arg(&penalty);
                let batch = batch_size as i32;
                let vocab = vocab_size as i32;
                let max_recent = max_recent as i32;
                builder.arg(&batch);
                builder.arg(&vocab);
                builder.arg(&max_recent);
                unsafe {
                    builder.launch(LaunchConfig {
                        grid_dim: (batch_size as u32, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .w()?;
            }
            _ => candle_core::bail!("gpu_sample_topk_topp_batch_bf16_cached expects BF16 logits"),
        }
    }

    buffers.ensure_output(dev, batch_size)?;
    buffers.ensure_batch_metadata(dev, batch_size)?;
    BatchNonGreedyCudaBuffers::upload_f32(
        &mut buffers.temperatures,
        dev,
        temperatures,
        "missing temperature buffer",
    )?;
    BatchNonGreedyCudaBuffers::upload_u32(
        &mut buffers.top_ks,
        dev,
        top_ks,
        "missing top-k buffer",
    )?;
    BatchNonGreedyCudaBuffers::upload_f32(
        &mut buffers.top_ps,
        dev,
        top_ps,
        "missing top-p buffer",
    )?;
    BatchNonGreedyCudaBuffers::upload_u64(&mut buffers.seeds, dev, seeds, "missing seed buffer")?;

    let output_tokens = buffers
        .output_tokens
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing non-greedy output buffer".into()))?;
    let temperatures = buffers
        .temperatures
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing temperature buffer".into()))?
        .slice(0..batch_size);
    let top_ks = buffers
        .top_ks
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing top-k buffer".into()))?
        .slice(0..batch_size);
    let top_ps = buffers
        .top_ps
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing top-p buffer".into()))?
        .slice(0..batch_size);
    let seeds = buffers
        .seeds
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing seed buffer".into()))?
        .slice(0..batch_size);
    let func = load_func!(dev, "gpu_sample_topk_topp_batch_bf16")?;
    let block_dim = 64usize;
    let shared_mem_bytes =
        block_dim * max_top_k * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());
    match &cuda_storage.slice {
        CudaStorageSlice::BF16(logits) => {
            let logits = logits.slice(o1..o2);
            let batch = batch_size as i32;
            let vocab = vocab_size as i32;
            let max_top_k = max_top_k as i32;
            let mut builder = func.builder();
            builder.arg(&logits);
            builder.arg(output_tokens);
            builder.arg(&temperatures);
            builder.arg(&top_ks);
            builder.arg(&top_ps);
            builder.arg(&seeds);
            builder.arg(&batch);
            builder.arg(&vocab);
            builder.arg(&max_top_k);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (batch_size as u32, 1, 1),
                    block_dim: (block_dim as u32, 1, 1),
                    shared_mem_bytes: shared_mem_bytes as u32,
                })
            }
            .w()?;
        }
        _ => candle_core::bail!("gpu_sample_topk_topp_batch_bf16_cached expects BF16 logits"),
    }

    let output_tokens = output_tokens.slice(0..batch_size);
    let result = dev.clone_dtoh(&output_tokens)?;
    Ok(result
        .into_iter()
        .map(|token| token.max(0) as u32)
        .collect())
}

// =====================================================================
// 4. GPU TopK — returns indices of the top-k largest values
// =====================================================================

#[cfg(feature = "cuda")]
thread_local! {
    static TOPK_TMP: std::cell::RefCell<
        std::collections::HashMap<(candle_core::cuda_backend::DeviceId, usize), TopkTmpBufs>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

#[cfg(feature = "cuda")]
struct TopkTmpBufs {
    vals: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    idx: candle_core::cuda_backend::cudarc::driver::CudaSlice<u32>,
    cap_elems: usize,
}

/// GPU top-k indices for 1D f32 tensors (k ≤ 64).
///
/// Two-stage block reduction using custom CUDA kernels compiled from
/// `crane-core/kernels/fused_ops.cu`.
///
/// Returns a `[k]` U32 tensor of the indices of the k largest values,
/// sorted in descending order of value.
#[cfg(feature = "cuda")]
pub fn topk_indices(logits: &Tensor, k: usize) -> Result<Tensor> {
    if !logits.is_contiguous() {
        candle_core::bail!("topk_indices requires contiguous input");
    }
    if logits.rank() != 1 {
        candle_core::bail!("topk_indices expects a 1D tensor");
    }
    if k == 0 || k > 64 {
        candle_core::bail!("topk_indices expects 0 < k <= 64");
    }
    let n = logits.dims1()?;
    if k > n {
        candle_core::bail!("topk_indices expects k <= n");
    }
    logits.apply_op1_no_bwd(&TopKIndicesOp { k })
}

#[cfg(feature = "cuda")]
struct TopKIndicesOp {
    k: usize,
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for TopKIndicesOp {
    fn name(&self) -> &'static str {
        "topk_indices"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("topk_indices requires contiguous layout");
        }
        let k = self.k;
        let n = layout.shape().elem_count();
        let start = layout.start_offset();
        let end = start + n;

        let mut pairs: Vec<(f32, u32)> = match storage {
            candle_core::CpuStorage::F32(vs) => vs[start..end]
                .iter()
                .enumerate()
                .map(|(i, &v)| (v, i as u32))
                .collect(),
            _ => candle_core::bail!("topk_indices only supports f32"),
        };

        let kth = k.saturating_sub(1);
        pairs.select_nth_unstable_by(kth, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Greater)
        });
        pairs.truncate(k);
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Greater));

        let out: Vec<u32> = pairs.into_iter().map(|(_, i)| i).collect();
        Ok((candle_core::CpuStorage::U32(out), Shape::from_dims(&[k])))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &candle_core::Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("topk_indices requires contiguous layout");
        }
        let k = self.k;
        let k_u32 = k as u32;
        let n = layout.shape().elem_count();
        let n_u32 = n as u32;
        let dev = &storage.device;

        let x = storage.as_cuda_slice::<f32>()?;
        let (o1, o2) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("topk: need contiguous offsets".into()))?;
        let x = x.slice(o1..o2);

        let block_dim1 = 128u32;
        let block_dim2 = 128u32;
        let items_per_block = (block_dim1 as usize) * 8;
        let grid = ((n + items_per_block - 1) / items_per_block).clamp(1, 1024);
        let grid_dim = grid as u32;
        let shared1 =
            block_dim1 as usize * k * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());
        let shared2 =
            block_dim2 as usize * k * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());

        let cap_elems = grid * k;
        let dev_id = dev.id();
        let (tmp_vals, tmp_idx) = TOPK_TMP.with(|cell| -> Result<_> {
            let mut map = cell.borrow_mut();
            match map.get_mut(&(dev_id, k)) {
                Some(bufs) if bufs.cap_elems >= cap_elems => {
                    Ok((bufs.vals.clone(), bufs.idx.clone()))
                }
                _ => {
                    let vals = unsafe { dev.alloc::<f32>(cap_elems)? };
                    let idx = unsafe { dev.alloc::<u32>(cap_elems)? };
                    map.insert(
                        (dev_id, k),
                        TopkTmpBufs {
                            vals: vals.clone(),
                            idx: idx.clone(),
                            cap_elems,
                        },
                    );
                    Ok((vals, idx))
                }
            }
        })?;

        let out_idx = unsafe { dev.alloc::<u32>(k)? };

        // Stage 1
        let f1 = load_func!(dev, "topk_stage1_f32")?;
        let items_per_block_u32 = items_per_block as u32;
        {
            let mut builder = f1.builder();
            builder.arg(&x);
            builder.arg(&n_u32);
            builder.arg(&k_u32);
            builder.arg(&items_per_block_u32);
            builder.arg(&tmp_vals);
            builder.arg(&tmp_idx);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (block_dim1, 1, 1),
                    shared_mem_bytes: shared1 as u32,
                })
            }
            .w()?;
        }

        // Stage 2
        let m = grid_dim * k_u32;
        let f2 = load_func!(dev, "topk_stage2_f32")?;
        {
            let mut builder = f2.builder();
            builder.arg(&tmp_vals);
            builder.arg(&tmp_idx);
            builder.arg(&m);
            builder.arg(&k_u32);
            builder.arg(&out_idx);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_dim2, 1, 1),
                    shared_mem_bytes: shared2 as u32,
                })
            }
            .w()?;
        }

        let dst = CudaStorage::wrap_cuda_slice(out_idx, dev.clone());
        Ok((dst, Shape::from_dims(&[k])))
    }
}
