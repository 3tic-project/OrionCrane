use super::*;

/// Decode-time BF16 RoPE using per-row absolute positions.
///
/// This avoids materializing per-batch cos/sin slices during batch decode.
/// `x` must be `[batch, heads, 1, head_dim]` BF16 CUDA, `cos`/`sin` are the
/// full f32 tables `[max_position, head_dim / 2]`, and `positions` is `[batch]`.
#[cfg(feature = "cuda")]
pub fn fused_rope_indexed_bf16(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
) -> Result<Tensor> {
    if x.dtype() != DType::BF16 || cos.dtype() != DType::F32 || sin.dtype() != DType::F32 {
        candle_core::bail!("fused_rope_indexed_bf16 expects BF16 x and F32 cos/sin")
    }
    let (batch_size, num_heads, seq_len, head_dim) = x.dims4()?;
    if seq_len != 1 || head_dim == 0 || head_dim % 2 != 0 {
        candle_core::bail!("fused_rope_indexed_bf16 expects [batch, heads, 1, even_head_dim]")
    }
    let (max_position, cos_half_dim) = cos.dims2()?;
    if sin.dims2()? != (max_position, cos_half_dim) || cos_half_dim * 2 != head_dim {
        candle_core::bail!("fused_rope_indexed_bf16: cos/sin table shape mismatch")
    }
    if positions.dims() != [batch_size] || positions.dtype() != DType::U32 {
        candle_core::bail!("fused_rope_indexed_bf16 expects U32 positions shaped [batch]")
    }
    let dev = match x.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("fused_rope_indexed_bf16 requires CUDA input"),
    };
    for (name, tensor) in [("cos", cos), ("sin", sin), ("positions", positions)] {
        if !matches!(tensor.device(), Device::Cuda(_)) {
            candle_core::bail!("fused_rope_indexed_bf16: {name} must be on CUDA")
        }
    }

    let x = x.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.contiguous()?;
    // SAFETY: each logical thread writes both halves of one RoPE pair, so the
    // launch covers every output element.
    let output = unsafe { Tensor::empty(x.shape(), DType::BF16, x.device())? };

    let (x_storage, x_layout) = x.storage_and_layout();
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let (pos_storage, pos_layout) = positions.storage_and_layout();
    let (out_storage, out_layout) = output.storage_and_layout();
    let x_storage = match &*x_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_rope_indexed_bf16: expected CUDA x"),
    };
    let cos_storage = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_rope_indexed_bf16: expected CUDA cos"),
    };
    let sin_storage = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_rope_indexed_bf16: expected CUDA sin"),
    };
    let pos_storage = match &*pos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_rope_indexed_bf16: expected CUDA positions"),
    };
    let out_storage = match &*out_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_rope_indexed_bf16: expected CUDA output"),
    };
    let (x_o1, x_o2) = x_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("fused_rope_indexed_bf16: x must be contiguous".into())
    })?;
    let (cos_o1, cos_o2) = cos_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("fused_rope_indexed_bf16: cos must be contiguous".into())
    })?;
    let (sin_o1, sin_o2) = sin_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("fused_rope_indexed_bf16: sin must be contiguous".into())
    })?;
    let (pos_o1, pos_o2) = pos_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("fused_rope_indexed_bf16: positions must be contiguous".into())
    })?;
    let (out_o1, out_o2) = out_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("fused_rope_indexed_bf16: output must be contiguous".into())
    })?;

    let func = load_func!(dev, "fused_rope_indexed_bf16")?;
    let elems = batch_size * num_heads * (head_dim / 2);
    let cfg = LaunchConfig::for_num_elems(elems as u32);
    match (
        &x_storage.slice,
        &cos_storage.slice,
        &sin_storage.slice,
        &pos_storage.slice,
        &out_storage.slice,
    ) {
        (
            CudaStorageSlice::BF16(x),
            CudaStorageSlice::F32(cos),
            CudaStorageSlice::F32(sin),
            CudaStorageSlice::U32(positions),
            CudaStorageSlice::BF16(output),
        ) => {
            let x = x.slice(x_o1..x_o2);
            let cos = cos.slice(cos_o1..cos_o2);
            let sin = sin.slice(sin_o1..sin_o2);
            let positions = positions.slice(pos_o1..pos_o2);
            let output = output.slice(out_o1..out_o2);
            let batch_i = batch_size as i32;
            let heads_i = num_heads as i32;
            let head_dim_i = head_dim as i32;
            let max_pos_i = max_position as i32;
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&positions);
            builder.arg(&output);
            builder.arg(&batch_i);
            builder.arg(&heads_i);
            builder.arg(&head_dim_i);
            builder.arg(&max_pos_i);
            unsafe { builder.launch(cfg) }.w()?;
        }
        _ => candle_core::bail!("fused_rope_indexed_bf16: unexpected CUDA storage dtypes"),
    }

    Ok(output.clone())
}

/// Decode-time BF16 Q/K RMSNorm plus indexed RoPE.
///
/// This fuses Qwen3's per-head QK norm and RoPE for `seq_len == 1` batch
/// decode. Inputs are `[batch, heads, 1, head_dim]` BF16 CUDA tensors, weights
/// are `[head_dim]` BF16, and cos/sin are full f32 RoPE tables.
#[cfg(feature = "cuda")]
pub fn fused_qk_norm_rope_indexed_bf16(
    q: &Tensor,
    k: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    if q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || q_weight.dtype() != DType::BF16
        || k_weight.dtype() != DType::BF16
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
    {
        candle_core::bail!(
            "fused_qk_norm_rope_indexed_bf16 expects BF16 q/k/weights and F32 cos/sin"
        )
    }
    let (batch_size, q_heads, q_seq_len, head_dim) = q.dims4()?;
    let (k_batch_size, kv_heads, k_seq_len, k_head_dim) = k.dims4()?;
    if batch_size != k_batch_size || q_seq_len != 1 || k_seq_len != 1 || head_dim != k_head_dim {
        candle_core::bail!(
            "fused_qk_norm_rope_indexed_bf16 expects matching [batch, heads, 1, head_dim] q/k"
        )
    }
    if head_dim == 0 || head_dim % 2 != 0 {
        candle_core::bail!("fused_qk_norm_rope_indexed_bf16 expects an even non-zero head_dim")
    }
    if q_weight.dims() != [head_dim] || k_weight.dims() != [head_dim] {
        candle_core::bail!("fused_qk_norm_rope_indexed_bf16 expects q/k weights shaped [head_dim]")
    }
    let (max_position, cos_half_dim) = cos.dims2()?;
    if sin.dims2()? != (max_position, cos_half_dim) || cos_half_dim * 2 != head_dim {
        candle_core::bail!("fused_qk_norm_rope_indexed_bf16: cos/sin table shape mismatch")
    }
    if positions.dims() != [batch_size] || positions.dtype() != DType::U32 {
        candle_core::bail!("fused_qk_norm_rope_indexed_bf16 expects U32 positions shaped [batch]")
    }
    let dev = match q.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("fused_qk_norm_rope_indexed_bf16 requires CUDA q"),
    };
    for (name, tensor) in [
        ("k", k),
        ("q_weight", q_weight),
        ("k_weight", k_weight),
        ("cos", cos),
        ("sin", sin),
        ("positions", positions),
    ] {
        if !matches!(tensor.device(), Device::Cuda(_)) {
            candle_core::bail!("fused_qk_norm_rope_indexed_bf16: {name} must be on CUDA")
        }
    }

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let q_weight = q_weight.contiguous()?;
    let k_weight = k_weight.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.contiguous()?;
    // SAFETY: the grid has one block for every Q/K head and the kernel writes
    // both halves of every head dimension before these tensors are returned.
    let q_output = unsafe { Tensor::empty(q.shape(), DType::BF16, q.device())? };
    let k_output = unsafe { Tensor::empty(k.shape(), DType::BF16, k.device())? };

    let (q_storage, q_layout) = q.storage_and_layout();
    let (k_storage, k_layout) = k.storage_and_layout();
    let (q_weight_storage, q_weight_layout) = q_weight.storage_and_layout();
    let (k_weight_storage, k_weight_layout) = k_weight.storage_and_layout();
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let (pos_storage, pos_layout) = positions.storage_and_layout();
    let (q_out_storage, q_out_layout) = q_output.storage_and_layout();
    let (k_out_storage, k_out_layout) = k_output.storage_and_layout();

    let q_storage = match &*q_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qk_norm_rope_indexed_bf16: expected CUDA q"),
    };
    let k_storage = match &*k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qk_norm_rope_indexed_bf16: expected CUDA k"),
    };
    let q_weight_storage = match &*q_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qk_norm_rope_indexed_bf16: expected CUDA q_weight"),
    };
    let k_weight_storage = match &*k_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qk_norm_rope_indexed_bf16: expected CUDA k_weight"),
    };
    let cos_storage = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qk_norm_rope_indexed_bf16: expected CUDA cos"),
    };
    let sin_storage = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qk_norm_rope_indexed_bf16: expected CUDA sin"),
    };
    let pos_storage = match &*pos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qk_norm_rope_indexed_bf16: expected CUDA positions"),
    };
    let q_out_storage = match &*q_out_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qk_norm_rope_indexed_bf16: expected CUDA q output"),
    };
    let k_out_storage = match &*k_out_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_qk_norm_rope_indexed_bf16: expected CUDA k output"),
    };

    let (q_o1, q_o2) = q_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("fused_qk_norm_rope_indexed_bf16: q must be contiguous".into())
    })?;
    let (k_o1, k_o2) = k_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("fused_qk_norm_rope_indexed_bf16: k must be contiguous".into())
    })?;
    let (qw_o1, qw_o2) = q_weight_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(
            "fused_qk_norm_rope_indexed_bf16: q_weight must be contiguous".into(),
        )
    })?;
    let (kw_o1, kw_o2) = k_weight_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(
            "fused_qk_norm_rope_indexed_bf16: k_weight must be contiguous".into(),
        )
    })?;
    let (cos_o1, cos_o2) = cos_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("fused_qk_norm_rope_indexed_bf16: cos must be contiguous".into())
    })?;
    let (sin_o1, sin_o2) = sin_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("fused_qk_norm_rope_indexed_bf16: sin must be contiguous".into())
    })?;
    let (pos_o1, pos_o2) = pos_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(
            "fused_qk_norm_rope_indexed_bf16: positions must be contiguous".into(),
        )
    })?;
    let (q_out_o1, q_out_o2) = q_out_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(
            "fused_qk_norm_rope_indexed_bf16: q output must be contiguous".into(),
        )
    })?;
    let (k_out_o1, k_out_o2) = k_out_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(
            "fused_qk_norm_rope_indexed_bf16: k output must be contiguous".into(),
        )
    })?;

    let func = load_func!(dev, "fused_qk_norm_rope_indexed_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: ((batch_size * (q_heads + kv_heads)) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    match (
        &q_storage.slice,
        &k_storage.slice,
        &q_weight_storage.slice,
        &k_weight_storage.slice,
        &cos_storage.slice,
        &sin_storage.slice,
        &pos_storage.slice,
        &q_out_storage.slice,
        &k_out_storage.slice,
    ) {
        (
            CudaStorageSlice::BF16(q),
            CudaStorageSlice::BF16(k),
            CudaStorageSlice::BF16(q_weight),
            CudaStorageSlice::BF16(k_weight),
            CudaStorageSlice::F32(cos),
            CudaStorageSlice::F32(sin),
            CudaStorageSlice::U32(positions),
            CudaStorageSlice::BF16(q_output_slice),
            CudaStorageSlice::BF16(k_output_slice),
        ) => {
            let q = q.slice(q_o1..q_o2);
            let k = k.slice(k_o1..k_o2);
            let q_weight = q_weight.slice(qw_o1..qw_o2);
            let k_weight = k_weight.slice(kw_o1..kw_o2);
            let cos = cos.slice(cos_o1..cos_o2);
            let sin = sin.slice(sin_o1..sin_o2);
            let positions = positions.slice(pos_o1..pos_o2);
            let q_output_slice = q_output_slice.slice(q_out_o1..q_out_o2);
            let k_output_slice = k_output_slice.slice(k_out_o1..k_out_o2);
            let batch_i = batch_size as i32;
            let q_heads_i = q_heads as i32;
            let kv_heads_i = kv_heads as i32;
            let head_dim_i = head_dim as i32;
            let max_pos_i = max_position as i32;
            let mut builder = func.builder();
            builder.arg(&q);
            builder.arg(&k);
            builder.arg(&q_weight);
            builder.arg(&k_weight);
            builder.arg(&cos);
            builder.arg(&sin);
            builder.arg(&positions);
            builder.arg(&q_output_slice);
            builder.arg(&k_output_slice);
            builder.arg(&batch_i);
            builder.arg(&q_heads_i);
            builder.arg(&kv_heads_i);
            builder.arg(&head_dim_i);
            builder.arg(&max_pos_i);
            builder.arg(&eps);
            unsafe { builder.launch(cfg) }.w()?;
        }
        _ => candle_core::bail!("fused_qk_norm_rope_indexed_bf16: unexpected CUDA storage dtypes"),
    }

    Ok((q_output.clone(), k_output.clone()))
}
