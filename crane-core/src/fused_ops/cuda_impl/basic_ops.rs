use super::*;

// =====================================================================
// 1. Fused SiLU(gate) * up
// =====================================================================

/// Fused SiLU activation + element-wise multiply.
///
/// Takes a `gate_up` tensor of shape `[..., 2*intermediate_size]`
/// (gate and up projections concatenated along the last dim) and returns
/// `silu(gate) * up` of shape `[..., intermediate_size]`.
///
/// Replaces 3 candle ops: `narrow(gate)` + `silu(gate)` + `gate * up`.
pub struct FusedSiluMul {
    pub intermediate_size: usize,
}

impl candle_core::CustomOp1 for FusedSiluMul {
    fn name(&self) -> &'static str {
        "fused-silu-mul"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        // CPU fallback — just do it the slow way.
        use candle_core::CpuStorage as C;

        fn inner<T: WithDType>(
            src: &[T],
            layout: &Layout,
            intermediate_size: usize,
        ) -> Result<(candle_core::CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle_core::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let dims = layout.shape().dims();
            let last = *dims.last().unwrap();
            if last != 2 * intermediate_size {
                candle_core::bail!(
                    "last dim {last} != 2*intermediate_size {}",
                    2 * intermediate_size
                );
            }
            let n_rows = src.len() / last;
            let mut dst = vec![T::zero(); n_rows * intermediate_size];
            for row in 0..n_rows {
                let gate = &src[row * last..row * last + intermediate_size];
                let up = &src[row * last + intermediate_size..row * last + last];
                let out = &mut dst[row * intermediate_size..(row + 1) * intermediate_size];
                for i in 0..intermediate_size {
                    let g: f64 = gate[i].to_f64();
                    let u: f64 = up[i].to_f64();
                    let silu_g = g / (1.0 + (-g).exp());
                    out[i] = T::from_f64(silu_g * u);
                }
            }
            let mut out_dims = dims.to_vec();
            *out_dims.last_mut().unwrap() = intermediate_size;
            let storage = T::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(&out_dims)))
        }

        match storage {
            C::BF16(s) => inner(s, layout, self.intermediate_size),
            C::F16(s) => inner(s, layout, self.intermediate_size),
            C::F32(s) => inner(s, layout, self.intermediate_size),
            C::F64(s) => inner(s, layout, self.intermediate_size),
            _ => candle_core::bail!("unsupported dtype for fused_silu_mul"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        let dev = storage.device();
        let dims = layout.shape().dims();
        let last = *dims.last().unwrap();
        let intermediate_size = self.intermediate_size;

        if last != 2 * intermediate_size {
            candle_core::bail!(
                "fused_silu_mul: last dim {last} != 2*intermediate_size {}",
                2 * intermediate_size
            );
        }

        let (o1, o2) = match layout.contiguous_offsets() {
            None => candle_core::bail!("fused_silu_mul: input must be contiguous"),
            Some(offsets) => offsets,
        };

        let n_rows = (o2 - o1) / last;
        let out_el = n_rows * intermediate_size;

        // Choose kernel name and launch
        let fn_name = match storage.dtype() {
            DType::BF16 => "fused_silu_mul_bf16",
            DType::F16 => "fused_silu_mul_f16",
            DType::F32 => "fused_silu_mul_f32",
            dt => candle_core::bail!("fused_silu_mul: unsupported dtype {dt:?}"),
        };
        let func = load_func!(dev, fn_name)?;

        let block_size = 1024u32.min(intermediate_size as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let slice = match &storage.slice {
            CudaStorageSlice::BF16(s) => {
                let s = s.slice(o1..o2);
                let dst = unsafe { dev.alloc::<half::bf16>(out_el)? };
                let mut builder = func.builder();
                builder.arg(&s);
                builder.arg(&dst);
                let isize_i32 = intermediate_size as i32;
                builder.arg(&isize_i32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::BF16(dst)
            }
            CudaStorageSlice::F16(s) => {
                let s = s.slice(o1..o2);
                let dst = unsafe { dev.alloc::<half::f16>(out_el)? };
                let mut builder = func.builder();
                builder.arg(&s);
                builder.arg(&dst);
                let isize_i32 = intermediate_size as i32;
                builder.arg(&isize_i32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F16(dst)
            }
            CudaStorageSlice::F32(s) => {
                let s = s.slice(o1..o2);
                let dst = unsafe { dev.alloc::<f32>(out_el)? };
                let mut builder = func.builder();
                builder.arg(&s);
                builder.arg(&dst);
                let isize_i32 = intermediate_size as i32;
                builder.arg(&isize_i32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(dst)
            }
            _ => candle_core::bail!("fused_silu_mul: unsupported storage type"),
        };

        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().unwrap() = intermediate_size;
        let dst = CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, Shape::from_dims(&out_dims)))
    }
}

/// Convenience function: fused SiLU(gate) * up.
///
/// `gate_up` must have shape `[..., 2*intermediate_size]` and be contiguous.
pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    gate_up.apply_op1_no_bwd(&FusedSiluMul { intermediate_size })
}

// =====================================================================
// 2. Fused residual_add + RMSNorm
// =====================================================================

/// Fused residual addition + RMSNorm.
///
/// Computes: `residual += hidden; out = rmsnorm(residual, weight, eps)`
/// in one pass. The CUDA path returns a new residual tensor and a normalized
/// output tensor because Candle tensors do not expose a stable in-place write
/// API for this use site.
///
/// Returns the normalized output tensor.
pub struct FusedAddRmsNorm {
    pub eps: f32,
}

impl FusedAddRmsNorm {
    /// Execute add+rmsnorm and update `residual` with the summed tensor.
    pub fn fwd(&self, residual: &mut Tensor, hidden: &Tensor, weight: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if residual.device().is_cuda()
            && residual.dtype() == DType::BF16
            && hidden.dtype() == DType::BF16
            && weight.dtype() == DType::BF16
        {
            if let Ok((sum, norm)) = fused_add_rmsnorm_bf16(residual, hidden, weight, self.eps) {
                *residual = sum;
                return Ok(norm);
            }
        }

        let sum = (residual.contiguous()? + hidden)?;
        *residual = sum.clone();
        let norm = candle_nn::RmsNorm::new(weight.clone(), self.eps as f64);
        candle_core::Module::forward(&norm, &sum)
    }
}

/// CUDA BF16 residual add + RMSNorm.
///
/// Returns `(residual + hidden, rmsnorm(residual + hidden))`.
#[cfg(feature = "cuda")]
pub fn fused_add_rmsnorm_bf16(
    residual: &Tensor,
    hidden: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    if residual.dtype() != DType::BF16
        || hidden.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
    {
        candle_core::bail!("fused_add_rmsnorm_bf16 expects BF16 residual/hidden/weight")
    }
    if residual.dims() != hidden.dims() {
        candle_core::bail!("fused_add_rmsnorm_bf16: residual/hidden shape mismatch")
    }
    let dims = residual.dims();
    let ncols = *dims.last().ok_or_else(|| {
        candle_core::Error::Msg("fused_add_rmsnorm_bf16 expects rank >= 1".into())
    })?;
    if ncols == 0 || weight.dims() != [ncols] {
        candle_core::bail!("fused_add_rmsnorm_bf16: weight must be [hidden_size]")
    }
    let dev = match residual.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("fused_add_rmsnorm_bf16 requires CUDA residual"),
    };
    for (name, tensor) in [("hidden", hidden), ("weight", weight)] {
        if !matches!(tensor.device(), Device::Cuda(_)) {
            candle_core::bail!("fused_add_rmsnorm_bf16: {name} must be on CUDA")
        }
    }

    let residual = residual.contiguous()?;
    let hidden = hidden.contiguous()?;
    let weight = weight.contiguous()?;
    let residual_out = Tensor::zeros(residual.shape(), DType::BF16, residual.device())?;
    let norm_out = Tensor::zeros(residual.shape(), DType::BF16, residual.device())?;

    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let (hidden_storage, hidden_layout) = hidden.storage_and_layout();
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let (residual_out_storage, residual_out_layout) = residual_out.storage_and_layout();
    let (norm_out_storage, norm_out_layout) = norm_out.storage_and_layout();
    let residual_storage = match &*residual_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_add_rmsnorm_bf16: expected CUDA residual"),
    };
    let hidden_storage = match &*hidden_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_add_rmsnorm_bf16: expected CUDA hidden"),
    };
    let weight_storage = match &*weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_add_rmsnorm_bf16: expected CUDA weight"),
    };
    let residual_out_storage = match &*residual_out_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_add_rmsnorm_bf16: expected CUDA residual output"),
    };
    let norm_out_storage = match &*norm_out_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("fused_add_rmsnorm_bf16: expected CUDA norm output"),
    };

    let (residual_o1, residual_o2) = residual_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("fused_add_rmsnorm_bf16: residual must be contiguous".into())
    })?;
    let (hidden_o1, hidden_o2) = hidden_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("fused_add_rmsnorm_bf16: hidden must be contiguous".into())
    })?;
    let (weight_o1, weight_o2) = weight_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("fused_add_rmsnorm_bf16: weight must be contiguous".into())
    })?;
    let (residual_out_o1, residual_out_o2) =
        residual_out_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg(
                "fused_add_rmsnorm_bf16: residual output must be contiguous".into(),
            )
        })?;
    let (norm_out_o1, norm_out_o2) = norm_out_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("fused_add_rmsnorm_bf16: norm output must be contiguous".into())
    })?;

    let rows = residual.elem_count() / ncols;
    let func = load_func!(dev, "fused_add_rmsnorm_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    match (
        &residual_storage.slice,
        &hidden_storage.slice,
        &weight_storage.slice,
        &residual_out_storage.slice,
        &norm_out_storage.slice,
    ) {
        (
            CudaStorageSlice::BF16(residual),
            CudaStorageSlice::BF16(hidden),
            CudaStorageSlice::BF16(weight),
            CudaStorageSlice::BF16(residual_out_slice),
            CudaStorageSlice::BF16(norm_out_slice),
        ) => {
            let residual = residual.slice(residual_o1..residual_o2);
            let hidden = hidden.slice(hidden_o1..hidden_o2);
            let weight = weight.slice(weight_o1..weight_o2);
            let residual_out_slice = residual_out_slice.slice(residual_out_o1..residual_out_o2);
            let norm_out_slice = norm_out_slice.slice(norm_out_o1..norm_out_o2);
            let ncols_i = ncols as i32;
            let mut builder = func.builder();
            builder.arg(&residual);
            builder.arg(&hidden);
            builder.arg(&residual_out_slice);
            builder.arg(&norm_out_slice);
            builder.arg(&weight);
            builder.arg(&ncols_i);
            builder.arg(&eps);
            unsafe { builder.launch(cfg) }.w()?;
        }
        _ => candle_core::bail!("fused_add_rmsnorm_bf16: unexpected CUDA storage dtypes"),
    }

    Ok((residual_out.clone(), norm_out.clone()))
}
