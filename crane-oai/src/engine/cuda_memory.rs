//! CUDA memory-pool maintenance helpers.

use candle_core::Device;

#[derive(Debug, Clone, Copy)]
pub(super) struct CudaMemoryPoolTrimReport {
    pub gpu_used_before_bytes: u64,
    pub gpu_used_after_bytes: u64,
    pub gpu_total_bytes: u64,
    pub pool_reserved_before_bytes: Option<u64>,
    pub pool_reserved_after_bytes: Option<u64>,
    pub pool_used_before_bytes: Option<u64>,
    pub pool_used_after_bytes: Option<u64>,
}

impl CudaMemoryPoolTrimReport {
    pub fn gpu_reclaimed_bytes(&self) -> u64 {
        self.gpu_used_before_bytes
            .saturating_sub(self.gpu_used_after_bytes)
    }

    pub fn pool_reserved_reclaimed_bytes(&self) -> Option<u64> {
        Some(
            self.pool_reserved_before_bytes?
                .saturating_sub(self.pool_reserved_after_bytes?),
        )
    }
}

pub(super) fn trim_idle_cuda_memory_pool(
    device: &Device,
) -> Result<Option<CudaMemoryPoolTrimReport>, String> {
    #[cfg(feature = "cuda")]
    {
        trim_idle_cuda_memory_pool_cuda(device)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        Ok(None)
    }
}

#[cfg(feature = "cuda")]
fn trim_idle_cuda_memory_pool_cuda(
    device: &Device,
) -> Result<Option<CudaMemoryPoolTrimReport>, String> {
    use candle_core::cuda_backend::cudarc::driver::{result, sys};

    let cuda_device = match device {
        Device::Cuda(cuda_device) => cuda_device,
        _ => return Ok(None),
    };

    let stream = cuda_device.cuda_stream();
    let context = stream.context();
    if !context.has_async_alloc() {
        return Ok(None);
    }

    context
        .synchronize()
        .map_err(|err| format!("CUDA context synchronize before mempool trim failed: {err:?}"))?;
    let (free_before, total_before) = context
        .mem_get_info()
        .map_err(|err| format!("cuMemGetInfo before mempool trim failed: {err:?}"))?;

    let pool = match unsafe { result::device::get_mem_pool(context.cu_device()) } {
        Ok(pool) => pool,
        Err(first_err) => unsafe { result::device::get_default_mem_pool(context.cu_device()) }
            .map_err(|second_err| {
                format!(
                    "cuDeviceGetMemPool failed ({first_err:?}); cuDeviceGetDefaultMemPool failed ({second_err:?})"
                )
            })?,
    };

    let pool_reserved_before = pool_attr_u64(
        pool,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
    );
    let pool_used_before = pool_attr_u64(
        pool,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
    );

    unsafe { result::mem_pool::trim_to(pool, 0) }
        .map_err(|err| format!("cuMemPoolTrimTo(pool, 0) failed: {err:?}"))?;
    context
        .synchronize()
        .map_err(|err| format!("CUDA context synchronize after mempool trim failed: {err:?}"))?;

    let pool_reserved_after = pool_attr_u64(
        pool,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
    );
    let pool_used_after = pool_attr_u64(
        pool,
        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
    );
    let (free_after, total_after) = context
        .mem_get_info()
        .map_err(|err| format!("cuMemGetInfo after mempool trim failed: {err:?}"))?;

    Ok(Some(CudaMemoryPoolTrimReport {
        gpu_used_before_bytes: total_before.saturating_sub(free_before) as u64,
        gpu_used_after_bytes: total_after.saturating_sub(free_after) as u64,
        gpu_total_bytes: total_after as u64,
        pool_reserved_before_bytes: pool_reserved_before,
        pool_reserved_after_bytes: pool_reserved_after,
        pool_used_before_bytes: pool_used_before,
        pool_used_after_bytes: pool_used_after,
    }))
}

#[cfg(feature = "cuda")]
fn pool_attr_u64(
    pool: candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool,
    attr: candle_core::cuda_backend::cudarc::driver::sys::CUmemPool_attribute,
) -> Option<u64> {
    use candle_core::cuda_backend::cudarc::driver::result;

    let mut value = 0u64;
    unsafe {
        result::mem_pool::get_attribute(pool, attr, &mut value as *mut u64 as *mut std::ffi::c_void)
    }
    .ok()?;
    Some(value)
}
