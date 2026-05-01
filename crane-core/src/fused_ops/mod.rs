//! Fused CUDA kernels for Crane transformer inference.
//!
//! When the `cuda` feature is enabled, this module provides:
//! - `fused_silu_mul` — Fused SiLU(gate) * up in one pass
//! - `fused_add_rmsnorm` — Fused residual_add + RMSNorm
//! - `gpu_argmax` — GPU-side argmax for greedy sampling
//! - `topk_indices` — GPU top-k on 1D f32 tensors
//! - `copy_from_slice_u32` — HtoD: create a new CUDA U32 tensor from a host slice
//! - `copy_from_tensor_f32` — contiguous copy of a CUDA f32 tensor
//! - `paged_kv_append_bf16` — append generated K/V into GPU page storage
//! - `paged_attention_decode_bf16_with_metadata` — decode-only paged attention
//!
//! Each operation eliminates multiple kernel launches and intermediate
//! GMEM round-trips compared to the equivalent candle op chain.

#[derive(Clone)]
pub struct PagedAttentionDecodeContext {
    pub pages: candle_core::Tensor,
    pub indptr: Vec<u32>,
    pub indices: Vec<u32>,
    pub last_page_lens: Vec<u32>,
    pub seq_lens: Vec<u32>,
    pub block_size: usize,
    pub num_layers: usize,
}

impl PagedAttentionDecodeContext {
    pub fn batch_size(&self) -> usize {
        self.seq_lens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seq_lens.is_empty()
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl;

#[cfg(feature = "cuda")]
pub use cuda_impl::*;

// ── Non-CUDA fallbacks ──────────────────────────────────────────────

#[cfg(not(feature = "cuda"))]
mod fallback {
    use candle_core::{Result, Tensor};

    pub fn gpu_argmax(logits: &Tensor) -> Result<u32> {
        let logits = logits.flatten_all()?;
        logits.argmax(0)?.to_scalar::<u32>()
    }

    pub fn gpu_argmax_batch(logits: &Tensor) -> Result<Vec<u32>> {
        let dims = logits.dims();
        let (batch_size, vocab_size) = match dims {
            [b, v] => (*b, *v),
            [b, 1, v] => (*b, *v),
            _ => candle_core::bail!(
                "gpu_argmax_batch expects [batch, vocab] or [batch, 1, vocab], got {dims:?}"
            ),
        };
        let logits = if logits.rank() == 3 {
            logits.squeeze(1)?
        } else {
            logits.clone()
        };
        let mut out = Vec::with_capacity(batch_size);
        for row in 0..batch_size {
            let token = logits
                .narrow(0, row, 1)?
                .reshape(vocab_size)?
                .argmax(0)?
                .to_scalar::<u32>()?;
            out.push(token);
        }
        Ok(out)
    }

    pub fn topk_indices(logits: &Tensor, k: usize) -> Result<Tensor> {
        if logits.rank() != 1 {
            candle_core::bail!("topk_indices expects a 1D tensor");
        }
        let n = logits.dims1()?;
        if k == 0 || k > n {
            candle_core::bail!("topk_indices: invalid k");
        }
        let vals = logits.to_vec1::<f32>()?;
        let mut pairs: Vec<(f32, u32)> = vals
            .into_iter()
            .enumerate()
            .map(|(i, v)| (v, i as u32))
            .collect();
        let kth = k.saturating_sub(1);
        pairs.select_nth_unstable_by(kth, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Greater)
        });
        pairs.truncate(k);
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Greater));
        let out: Vec<u32> = pairs.into_iter().map(|(_, i)| i).collect();
        Tensor::new(out.as_slice(), logits.device())
    }

    pub fn copy_from_slice_u32(src: &[u32], device: &candle_core::Device) -> Result<Tensor> {
        Tensor::new(src, device)
    }

    pub fn copy_from_tensor_f32(src: &Tensor) -> Result<Tensor> {
        src.contiguous()
    }

    #[derive(Default)]
    pub struct ReusableU32TensorBuffer {
        tensor: Option<Tensor>,
        capacity: usize,
    }

    #[derive(Default)]
    pub struct ReusableTensorBuffer {
        tensor: Option<Tensor>,
        capacity: usize,
        dtype: Option<candle_core::DType>,
    }

    impl ReusableU32TensorBuffer {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn capacity(&self) -> usize {
            self.capacity
        }

        pub fn clear(&mut self) {
            self.tensor = None;
            self.capacity = 0;
        }

        pub fn upload_1d(&mut self, src: &[u32], device: &candle_core::Device) -> Result<Tensor> {
            if src.is_empty() {
                candle_core::bail!("ReusableU32TensorBuffer cannot upload an empty slice");
            }
            let needs_alloc = self.tensor.as_ref().map_or(true, |tensor| {
                self.capacity < src.len() || !tensor.device().same_device(device)
            });
            if needs_alloc {
                self.capacity = src.len().next_power_of_two();
                self.tensor = Some(Tensor::zeros(
                    (self.capacity,),
                    candle_core::DType::U32,
                    device,
                )?);
            }

            let tensor = self
                .tensor
                .as_ref()
                .expect("ReusableU32TensorBuffer must be allocated before upload");

            // Fast path on CUDA: single direct HtoD into the persistent
            // storage (eliminates a transient `Tensor::new` alloc + free and
            // the subsequent D2D `slice_set`). See P0 in
            // docs/qwen3/benchmarks/qwen3_profile_eager_vs_graph_2026_04_30.md.
            #[cfg(feature = "cuda")]
            if device.is_cuda() {
                tensor.copy_from_host_u32(src, 0)?;
                return tensor.narrow(0, 0, src.len());
            }

            let src_tensor = Tensor::new(src, device)?;
            tensor.slice_set(&src_tensor, 0, 0)?;
            tensor.narrow(0, 0, src.len())
        }
    }

    impl ReusableTensorBuffer {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn capacity(&self) -> usize {
            self.capacity
        }

        pub fn clear(&mut self) {
            self.tensor = None;
            self.capacity = 0;
            self.dtype = None;
        }

        pub fn copy_from(&mut self, src: &Tensor) -> Result<Tensor> {
            let len = src.elem_count();
            if len == 0 {
                candle_core::bail!("ReusableTensorBuffer cannot copy an empty tensor");
            }
            let dtype = src.dtype();
            let device = src.device();
            let needs_alloc = self.tensor.as_ref().map_or(true, |tensor| {
                self.capacity < len
                    || !tensor.device().same_device(device)
                    || self.dtype != Some(dtype)
            });
            if needs_alloc {
                self.capacity = len.next_power_of_two();
                self.dtype = Some(dtype);
                self.tensor = Some(Tensor::zeros((self.capacity,), dtype, device)?);
            }

            let src_flat = src.contiguous()?.flatten_all()?;
            let tensor = self
                .tensor
                .as_ref()
                .expect("ReusableTensorBuffer must be allocated before copy");
            tensor.slice_set(&src_flat, 0, 0)?;
            tensor.narrow(0, 0, len)?.reshape(src.dims().to_vec())
        }
    }

    #[derive(Default)]
    pub struct PagedKvCopyMetadataCudaBuffers {
        entries: usize,
    }

    #[derive(Default)]
    pub struct PagedAttentionMetadataCudaBuffers {
        batch_size: usize,
    }

    impl PagedKvCopyMetadataCudaBuffers {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn entries(&self) -> usize {
            self.entries
        }

        pub fn release(&mut self) {
            self.entries = 0;
        }

        pub fn upload(
            &mut self,
            _device: &candle_core::Device,
            page_ids: &[u32],
            token_offsets: &[u32],
            row_indices: &[u32],
            source_token_indices: &[u32],
        ) -> Result<()> {
            let entries = page_ids.len();
            if token_offsets.len() != entries
                || row_indices.len() != entries
                || source_token_indices.len() != entries
            {
                candle_core::bail!("paged KV copy metadata length mismatch")
            }
            self.entries = entries;
            Ok(())
        }
    }

    impl PagedAttentionMetadataCudaBuffers {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn batch_size(&self) -> usize {
            self.batch_size
        }

        pub fn release(&mut self) {
            self.batch_size = 0;
        }

        pub fn upload(
            &mut self,
            _device: &candle_core::Device,
            indptr: &[u32],
            _indices: &[u32],
            last_page_lens: &[u32],
            seq_lens: &[u32],
        ) -> Result<()> {
            let batch_size = seq_lens.len();
            if indptr.len() != batch_size + 1 || last_page_lens.len() != batch_size {
                candle_core::bail!("paged attention metadata length mismatch")
            }
            self.batch_size = batch_size;
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_kv_copy_bf16_with_metadata(
        _pages: &Tensor,
        _full_k: &Tensor,
        _full_v: &Tensor,
        _layer: usize,
        _num_layers: usize,
        _block_size: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _buffers: &PagedKvCopyMetadataCudaBuffers,
    ) -> Result<()> {
        candle_core::bail!("paged_kv_copy_bf16_with_metadata requires the cuda feature")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_kv_gather_bf16_with_metadata(
        _pages: &Tensor,
        _output: &Tensor,
        _layer: usize,
        _plane: usize,
        _max_len: usize,
        _num_layers: usize,
        _block_size: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _buffers: &PagedKvCopyMetadataCudaBuffers,
    ) -> Result<()> {
        candle_core::bail!("paged_kv_gather_bf16_with_metadata requires the cuda feature")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_attention_decode_bf16_with_metadata(
        _pages: &Tensor,
        _query: &Tensor,
        _current_k: &Tensor,
        _current_v: &Tensor,
        _layer: usize,
        _num_layers: usize,
        _block_size: usize,
        _num_heads: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
        _scale: f32,
        _buffers: &PagedAttentionMetadataCudaBuffers,
    ) -> Result<Tensor> {
        candle_core::bail!("paged_attention_decode_bf16_with_metadata requires the cuda feature")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn batch_kv_append_bf16_with_offset(
        _dst_k: &Tensor,
        _dst_v: &Tensor,
        _src_k: &Tensor,
        _src_v: &Tensor,
        _append_offset: &Tensor,
        _dst_width: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
    ) -> Result<()> {
        candle_core::bail!("batch_kv_append_bf16_with_offset requires the cuda feature")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn paged_kv_append_bf16(
        _pages: &Tensor,
        _full_k: &Tensor,
        _full_v: &Tensor,
        _page_ids: &[u32],
        _token_offsets: &[u32],
        _row_indices: &[u32],
        _round_indices: &[u32],
        _layer: usize,
        _original_max_kv: usize,
        _num_layers: usize,
        _block_size: usize,
        _num_kv_heads: usize,
        _head_dim: usize,
    ) -> Result<()> {
        candle_core::bail!("paged_kv_append_bf16 requires the cuda feature")
    }

    pub fn paged_kv_zero_pages_bf16(
        _pages: &Tensor,
        _page_ids: &[u32],
        _page_values: usize,
    ) -> Result<()> {
        candle_core::bail!("paged_kv_zero_pages_bf16 requires the cuda feature")
    }
}

#[cfg(not(feature = "cuda"))]
pub use fallback::*;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn reusable_u32_tensor_buffer_reuses_capacity_and_updates_contents() {
        let device = Device::Cpu;
        let mut buffer = ReusableU32TensorBuffer::new();

        let first = buffer.upload_1d(&[1, 2, 3], &device).unwrap();
        assert_eq!(first.to_vec1::<u32>().unwrap(), vec![1, 2, 3]);
        assert_eq!(buffer.capacity(), 4);

        let second = buffer.upload_1d(&[8, 9], &device).unwrap();
        assert_eq!(second.to_vec1::<u32>().unwrap(), vec![8, 9]);
        assert_eq!(buffer.capacity(), 4);

        let third = buffer.upload_1d(&[4, 5, 6, 7, 8], &device).unwrap();
        assert_eq!(third.to_vec1::<u32>().unwrap(), vec![4, 5, 6, 7, 8]);
        assert_eq!(buffer.capacity(), 8);
    }
}
