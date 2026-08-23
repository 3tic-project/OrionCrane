//! CUDA implementations of fused ops using custom PTX kernels.
//!
//! The PTX is compiled from `kernels/fused_ops.cu` at build time via
//! bindgen_cuda and embedded as a const string. Launch wrappers are split by
//! concern under `cuda_impl/` while preserving the public fused-op API.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
use candle_core::{DType, Device, Layout, Result, Shape, Tensor, WithDType};

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

const MODULE_NAME: &str = "crane_fused_ops";

macro_rules! load_func {
    ($dev:expr, $fn_name:expr) => {
        $dev.get_or_load_custom_func($fn_name, MODULE_NAME, ptx::FUSED_OPS)
    };
}

mod basic_ops;
mod graph;
mod memory;
mod paged_kv;
mod rope;
mod sampling;
mod w8a16;

pub use basic_ops::*;
pub use graph::*;
pub use memory::*;
pub use paged_kv::*;
pub use rope::*;
pub use sampling::*;
pub use w8a16::*;
