use super::*;

#[derive(Default)]
pub struct PagedKvCopyMetadataCudaBuffers {
    page_ids: Option<CudaSlice<u32>>,
    token_offsets: Option<CudaSlice<u32>>,
    row_indices: Option<CudaSlice<u32>>,
    source_token_indices: Option<CudaSlice<u32>>,
    capacity: usize,
    entries: usize,
}

#[cfg(feature = "cuda")]
#[derive(Default)]
pub struct PagedAttentionMetadataCudaBuffers {
    indptr: Option<CudaSlice<u32>>,
    indices: Option<CudaSlice<u32>>,
    last_page_lens: Option<CudaSlice<u32>>,
    seq_lens: Option<CudaSlice<u32>>,
    indptr_capacity: usize,
    indices_capacity: usize,
    indices_len: usize,
    batch_capacity: usize,
    batch_size: usize,
    cached_indptr: Vec<u32>,
    cached_indices: Vec<u32>,
    cached_last_page_lens: Vec<u32>,
    cached_seq_lens: Vec<u32>,
}

#[cfg(feature = "cuda")]
impl PagedAttentionMetadataCudaBuffers {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn release(&mut self) {
        self.indptr = None;
        self.indices = None;
        self.last_page_lens = None;
        self.seq_lens = None;
        self.indptr_capacity = 0;
        self.indices_capacity = 0;
        self.indices_len = 0;
        self.batch_capacity = 0;
        self.batch_size = 0;
        self.cached_indptr.clear();
        self.cached_indices.clear();
        self.cached_last_page_lens.clear();
        self.cached_seq_lens.clear();
    }

    pub fn upload(
        &mut self,
        device: &Device,
        indptr: &[u32],
        indices: &[u32],
        last_page_lens: &[u32],
        seq_lens: &[u32],
    ) -> Result<()> {
        let batch_size = seq_lens.len();
        if indptr.len() != batch_size + 1 || last_page_lens.len() != batch_size {
            candle_core::bail!("paged attention metadata length mismatch")
        }
        self.batch_size = batch_size;
        self.indices_len = indices.len();
        if batch_size == 0 {
            return Ok(());
        }

        let dev = match device {
            Device::Cuda(dev) => dev,
            _ => candle_core::bail!("paged attention metadata requires CUDA device"),
        };

        if self.indptr_capacity < indptr.len() {
            self.indptr = Some(unsafe { dev.alloc::<u32>(indptr.len())? });
            self.indptr_capacity = indptr.len();
        }
        let index_capacity = indices.len().max(1);
        if self.indices_capacity < index_capacity {
            self.indices = Some(unsafe { dev.alloc::<u32>(index_capacity)? });
            self.indices_capacity = index_capacity;
        }
        if self.batch_capacity < batch_size {
            self.last_page_lens = Some(unsafe { dev.alloc::<u32>(batch_size)? });
            self.seq_lens = Some(unsafe { dev.alloc::<u32>(batch_size)? });
            self.batch_capacity = batch_size;
        }

        if self.cached_indptr.as_slice() != indptr {
            let mut indptr_dst = self
                .indptr
                .as_mut()
                .ok_or_else(|| {
                    candle_core::Error::Msg("missing paged attention indptr buffer".into())
                })?
                .slice_mut(0..indptr.len());
            dev.memcpy_htod(indptr, &mut indptr_dst)?;
            self.cached_indptr.clear();
            self.cached_indptr.extend_from_slice(indptr);
        }

        if !indices.is_empty() && self.cached_indices.as_slice() != indices {
            let mut indices_dst = self
                .indices
                .as_mut()
                .ok_or_else(|| {
                    candle_core::Error::Msg("missing paged attention indices buffer".into())
                })?
                .slice_mut(0..indices.len());
            dev.memcpy_htod(indices, &mut indices_dst)?;
            self.cached_indices.clear();
            self.cached_indices.extend_from_slice(indices);
        }

        if self.cached_last_page_lens.as_slice() != last_page_lens {
            let mut last_page_lens_dst = self
                .last_page_lens
                .as_mut()
                .ok_or_else(|| {
                    candle_core::Error::Msg("missing paged attention last-page-lens buffer".into())
                })?
                .slice_mut(0..batch_size);
            dev.memcpy_htod(last_page_lens, &mut last_page_lens_dst)?;
            self.cached_last_page_lens.clear();
            self.cached_last_page_lens.extend_from_slice(last_page_lens);
        }
        if self.cached_seq_lens.as_slice() != seq_lens {
            let mut seq_lens_dst = self
                .seq_lens
                .as_mut()
                .ok_or_else(|| {
                    candle_core::Error::Msg("missing paged attention seq-lens buffer".into())
                })?
                .slice_mut(0..batch_size);
            dev.memcpy_htod(seq_lens, &mut seq_lens_dst)?;
            self.cached_seq_lens.clear();
            self.cached_seq_lens.extend_from_slice(seq_lens);
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl PagedKvCopyMetadataCudaBuffers {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn entries(&self) -> usize {
        self.entries
    }

    pub fn release(&mut self) {
        self.page_ids = None;
        self.token_offsets = None;
        self.row_indices = None;
        self.source_token_indices = None;
        self.capacity = 0;
        self.entries = 0;
    }

    pub fn upload(
        &mut self,
        device: &Device,
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
        if entries == 0 {
            return Ok(());
        }

        let dev = match device {
            Device::Cuda(dev) => dev,
            _ => candle_core::bail!("paged KV copy metadata requires CUDA device"),
        };
        if self.capacity < entries {
            self.page_ids = Some(unsafe { dev.alloc::<u32>(entries)? });
            self.token_offsets = Some(unsafe { dev.alloc::<u32>(entries)? });
            self.row_indices = Some(unsafe { dev.alloc::<u32>(entries)? });
            self.source_token_indices = Some(unsafe { dev.alloc::<u32>(entries)? });
            self.capacity = entries;
        }

        let mut page_ids_dst = self
            .page_ids
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing page id buffer".into()))?
            .slice_mut(0..entries);
        let mut token_offsets_dst = self
            .token_offsets
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing token offset buffer".into()))?
            .slice_mut(0..entries);
        let mut row_indices_dst = self
            .row_indices
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing row index buffer".into()))?
            .slice_mut(0..entries);
        let mut source_token_indices_dst = self
            .source_token_indices
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg("missing source token buffer".into()))?
            .slice_mut(0..entries);
        dev.memcpy_htod(page_ids, &mut page_ids_dst)?;
        dev.memcpy_htod(token_offsets, &mut token_offsets_dst)?;
        dev.memcpy_htod(row_indices, &mut row_indices_dst)?;
        dev.memcpy_htod(source_token_indices, &mut source_token_indices_dst)?;
        Ok(())
    }
}

/// Copy BF16 batch K/V tokens into GPU page storage using pre-uploaded metadata.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_kv_copy_bf16_with_metadata(
    pages: &Tensor,
    full_k: &Tensor,
    full_v: &Tensor,
    layer: usize,
    num_layers: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    buffers: &PagedKvCopyMetadataCudaBuffers,
) -> Result<()> {
    let entries = buffers.entries();
    if entries == 0 {
        return Ok(());
    }
    if pages.dtype() != DType::BF16
        || full_k.dtype() != DType::BF16
        || full_v.dtype() != DType::BF16
    {
        candle_core::bail!("paged_kv_copy_bf16_with_metadata expects BF16 tensors")
    }
    if pages.rank() != 6 {
        candle_core::bail!("paged_kv_copy_bf16_with_metadata: pages must be rank-6")
    }
    let src_dims = full_k.dims4()?;
    if full_v.dims4()? != src_dims {
        candle_core::bail!("paged_kv_copy_bf16_with_metadata: K/V source shapes differ")
    }
    let (_batch, src_heads, src_width, src_head_dim) = src_dims;
    if src_heads != num_kv_heads || src_head_dim != head_dim {
        candle_core::bail!("paged_kv_copy_bf16_with_metadata: source head layout mismatch")
    }

    let dev = match pages.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("paged_kv_copy_bf16_with_metadata requires CUDA page storage"),
    };
    let same_k_device = matches!(full_k.device(), Device::Cuda(k_dev) if k_dev.id() == dev.id());
    let same_v_device = matches!(full_v.device(), Device::Cuda(v_dev) if v_dev.id() == dev.id());
    if !same_k_device || !same_v_device {
        candle_core::bail!(
            "paged_kv_copy_bf16_with_metadata: tensors must be on the same CUDA device"
        )
    }

    let (page_storage, page_layout) = pages.storage_and_layout();
    let page_storage = match &*page_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("paged_kv_copy_bf16_with_metadata: expected CUDA page storage"),
    };
    let (page_o1, page_o2) = page_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("paged_kv_copy_bf16_with_metadata: pages must be contiguous".into())
    })?;

    let (k_storage, k_layout) = full_k.storage_and_layout();
    let k_storage = match &*k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("paged_kv_copy_bf16_with_metadata: expected CUDA K storage"),
    };
    let (k_o1, k_o2) = k_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(
            "paged_kv_copy_bf16_with_metadata: K source must be contiguous".into(),
        )
    })?;

    let (v_storage, v_layout) = full_v.storage_and_layout();
    let v_storage = match &*v_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("paged_kv_copy_bf16_with_metadata: expected CUDA V storage"),
    };
    let (v_o1, v_o2) = v_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(
            "paged_kv_copy_bf16_with_metadata: V source must be contiguous".into(),
        )
    })?;

    let page_ids = buffers
        .page_ids
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing page id metadata".into()))?
        .slice(0..entries);
    let token_offsets = buffers
        .token_offsets
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing token offset metadata".into()))?
        .slice(0..entries);
    let row_indices = buffers
        .row_indices
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing row index metadata".into()))?
        .slice(0..entries);
    let source_token_indices = buffers
        .source_token_indices
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing source token metadata".into()))?
        .slice(0..entries);

    let func = load_func!(dev, "paged_kv_append_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (entries as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    match (&page_storage.slice, &k_storage.slice, &v_storage.slice) {
        (CudaStorageSlice::BF16(pages), CudaStorageSlice::BF16(k), CudaStorageSlice::BF16(v)) => {
            let pages = pages.slice(page_o1..page_o2);
            let k = k.slice(k_o1..k_o2);
            let v = v.slice(v_o1..v_o2);
            let entries_i = entries as i32;
            let layer_i = layer as i32;
            let src_width_i = src_width as i32;
            let num_layers_i = num_layers as i32;
            let block_size_i = block_size as i32;
            let num_kv_heads_i = num_kv_heads as i32;
            let head_dim_i = head_dim as i32;
            let mut builder = func.builder();
            builder.arg(&pages);
            builder.arg(&k);
            builder.arg(&v);
            builder.arg(&page_ids);
            builder.arg(&token_offsets);
            builder.arg(&row_indices);
            builder.arg(&source_token_indices);
            builder.arg(&entries_i);
            builder.arg(&layer_i);
            builder.arg(&src_width_i);
            builder.arg(&num_layers_i);
            builder.arg(&block_size_i);
            builder.arg(&num_kv_heads_i);
            builder.arg(&head_dim_i);
            unsafe { builder.launch(cfg) }.w()?;
        }
        _ => candle_core::bail!("paged_kv_copy_bf16_with_metadata expects BF16 CUDA storage"),
    }

    Ok(())
}

/// Gather BF16 paged K/V into a right-aligned contiguous batch tensor using pre-uploaded metadata.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_kv_gather_bf16_with_metadata(
    pages: &Tensor,
    output: &Tensor,
    layer: usize,
    plane: usize,
    max_len: usize,
    num_layers: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    buffers: &PagedKvCopyMetadataCudaBuffers,
) -> Result<()> {
    let entries = buffers.entries();
    if entries == 0 {
        return Ok(());
    }
    if pages.dtype() != DType::BF16 || output.dtype() != DType::BF16 {
        candle_core::bail!("paged_kv_gather_bf16_with_metadata expects BF16 tensors")
    }
    if pages.rank() != 6 || output.rank() != 4 {
        candle_core::bail!(
            "paged_kv_gather_bf16_with_metadata: pages must be rank-6 and output rank-4"
        )
    }
    let (batch, out_heads, out_width, out_head_dim) = output.dims4()?;
    if out_heads != num_kv_heads || out_width != max_len || out_head_dim != head_dim {
        candle_core::bail!("paged_kv_gather_bf16_with_metadata: output layout mismatch")
    }
    if batch == 0 {
        return Ok(());
    }

    let dev = match pages.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("paged_kv_gather_bf16_with_metadata requires CUDA page storage"),
    };
    let same_output_device =
        matches!(output.device(), Device::Cuda(out_dev) if out_dev.id() == dev.id());
    if !same_output_device {
        candle_core::bail!(
            "paged_kv_gather_bf16_with_metadata: tensors must be on the same CUDA device"
        )
    }

    let (page_storage, page_layout) = pages.storage_and_layout();
    let page_storage = match &*page_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("paged_kv_gather_bf16_with_metadata: expected CUDA page storage"),
    };
    let (page_o1, page_o2) = page_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(
            "paged_kv_gather_bf16_with_metadata: pages must be contiguous".into(),
        )
    })?;

    let (output_storage, output_layout) = output.storage_and_layout();
    let output_storage = match &*output_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("paged_kv_gather_bf16_with_metadata: expected CUDA output storage"),
    };
    let (output_o1, output_o2) = output_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(
            "paged_kv_gather_bf16_with_metadata: output must be contiguous".into(),
        )
    })?;

    let page_ids = buffers
        .page_ids
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing page id metadata".into()))?
        .slice(0..entries);
    let token_offsets = buffers
        .token_offsets
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing token offset metadata".into()))?
        .slice(0..entries);
    let row_indices = buffers
        .row_indices
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing row index metadata".into()))?
        .slice(0..entries);
    let target_token_indices = buffers
        .source_token_indices
        .as_ref()
        .ok_or_else(|| candle_core::Error::Msg("missing target token metadata".into()))?
        .slice(0..entries);

    let func = load_func!(dev, "paged_kv_gather_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (entries as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    match (&page_storage.slice, &output_storage.slice) {
        (CudaStorageSlice::BF16(pages), CudaStorageSlice::BF16(output)) => {
            let pages = pages.slice(page_o1..page_o2);
            let output = output.slice(output_o1..output_o2);
            let entries_i = entries as i32;
            let layer_i = layer as i32;
            let plane_i = plane as i32;
            let max_len_i = max_len as i32;
            let num_layers_i = num_layers as i32;
            let block_size_i = block_size as i32;
            let num_kv_heads_i = num_kv_heads as i32;
            let head_dim_i = head_dim as i32;
            let mut builder = func.builder();
            builder.arg(&pages);
            builder.arg(&output);
            builder.arg(&page_ids);
            builder.arg(&token_offsets);
            builder.arg(&row_indices);
            builder.arg(&target_token_indices);
            builder.arg(&entries_i);
            builder.arg(&layer_i);
            builder.arg(&plane_i);
            builder.arg(&max_len_i);
            builder.arg(&num_layers_i);
            builder.arg(&block_size_i);
            builder.arg(&num_kv_heads_i);
            builder.arg(&head_dim_i);
            unsafe { builder.launch(cfg) }.w()?;
        }
        _ => candle_core::bail!("paged_kv_gather_bf16_with_metadata expects BF16 CUDA storage"),
    }

    Ok(())
}

/// Decode-only BF16 paged attention for Qwen3 GQA.
///
/// `pages` uses `[page, layer, K/V, block_token, kv_head, head_dim]`.
/// `query` is `[batch, q_heads, 1, head_dim]`, current K/V are
/// `[batch, kv_heads, 1, head_dim]`, and the returned tensor is
/// `[batch, q_heads, head_dim]`. Page-table sequence lengths describe past
/// tokens only; the current K/V token is always included by the kernel.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_decode_bf16_with_metadata(
    pages: &Tensor,
    query: &Tensor,
    current_k: &Tensor,
    current_v: &Tensor,
    layer: usize,
    num_layers: usize,
    block_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    buffers: &PagedAttentionMetadataCudaBuffers,
) -> Result<Tensor> {
    let batch_size = buffers.batch_size();
    if batch_size == 0 {
        return Tensor::zeros((0, num_heads, head_dim), DType::BF16, pages.device());
    }
    if pages.dtype() != DType::BF16
        || query.dtype() != DType::BF16
        || current_k.dtype() != DType::BF16
        || current_v.dtype() != DType::BF16
    {
        candle_core::bail!("paged_attention_decode_bf16_with_metadata expects BF16 tensors")
    }
    if pages.rank() != 6 {
        candle_core::bail!("paged_attention_decode_bf16_with_metadata: pages must be rank-6")
    }
    if head_dim == 0 || head_dim > 256 {
        candle_core::bail!("paged_attention_decode_bf16_with_metadata: head_dim must be in 1..=256")
    }
    if num_heads == 0 || num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        candle_core::bail!("paged_attention_decode_bf16_with_metadata: invalid GQA head layout")
    }

    match query.dims() {
        [b, h, 1, d] if *b == batch_size && *h == num_heads && *d == head_dim => {}
        dims => candle_core::bail!(
            "paged_attention_decode_bf16_with_metadata: query shape mismatch, got {dims:?}"
        ),
    }
    match current_k.dims() {
        [b, h, 1, d] if *b == batch_size && *h == num_kv_heads && *d == head_dim => {}
        dims => candle_core::bail!(
            "paged_attention_decode_bf16_with_metadata: current K shape mismatch, got {dims:?}"
        ),
    }
    if current_v.dims() != current_k.dims() {
        candle_core::bail!("paged_attention_decode_bf16_with_metadata: current K/V shapes differ")
    }

    let dev = match pages.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("paged_attention_decode_bf16_with_metadata requires CUDA pages"),
    };
    for (name, tensor) in [
        ("query", query),
        ("current K", current_k),
        ("current V", current_v),
    ] {
        let same_device = matches!(tensor.device(), Device::Cuda(other) if other.id() == dev.id());
        if !same_device {
            candle_core::bail!(
                "paged_attention_decode_bf16_with_metadata: {name} must be on the same CUDA device as pages"
            )
        }
    }

    let output = Tensor::zeros(
        (batch_size, num_heads, head_dim),
        DType::BF16,
        pages.device(),
    )?;

    {
        let (page_storage, page_layout) = pages.storage_and_layout();
        let page_storage = match &*page_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => {
                candle_core::bail!("paged_attention_decode_bf16_with_metadata: expected CUDA pages")
            }
        };
        let (page_o1, page_o2) = page_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg(
                "paged_attention_decode_bf16_with_metadata: pages must be contiguous".into(),
            )
        })?;

        let (query_storage, query_layout) = query.storage_and_layout();
        let query_storage = match &*query_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => {
                candle_core::bail!("paged_attention_decode_bf16_with_metadata: expected CUDA query")
            }
        };
        let (query_o1, query_o2) = query_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg(
                "paged_attention_decode_bf16_with_metadata: query must be contiguous".into(),
            )
        })?;

        let (k_storage, k_layout) = current_k.storage_and_layout();
        let k_storage = match &*k_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("paged_attention_decode_bf16_with_metadata: expected CUDA K"),
        };
        let (k_o1, k_o2) = k_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg(
                "paged_attention_decode_bf16_with_metadata: K must be contiguous".into(),
            )
        })?;

        let (v_storage, v_layout) = current_v.storage_and_layout();
        let v_storage = match &*v_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("paged_attention_decode_bf16_with_metadata: expected CUDA V"),
        };
        let (v_o1, v_o2) = v_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg(
                "paged_attention_decode_bf16_with_metadata: V must be contiguous".into(),
            )
        })?;

        let (output_storage, output_layout) = output.storage_and_layout();
        let output_storage = match &*output_storage {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!(
                "paged_attention_decode_bf16_with_metadata: expected CUDA output"
            ),
        };
        let (output_o1, output_o2) = output_layout.contiguous_offsets().ok_or_else(|| {
            candle_core::Error::Msg(
                "paged_attention_decode_bf16_with_metadata: output must be contiguous".into(),
            )
        })?;

        let indptr = buffers
            .indptr
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("missing paged attention indptr".into()))?
            .slice(0..batch_size + 1);
        let indices = buffers
            .indices
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("missing paged attention indices".into()))?
            .slice(0..buffers.indices_len.max(1));
        let last_page_lens = buffers
            .last_page_lens
            .as_ref()
            .ok_or_else(|| {
                candle_core::Error::Msg("missing paged attention last-page lens".into())
            })?
            .slice(0..batch_size);
        let seq_lens = buffers
            .seq_lens
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("missing paged attention sequence lens".into()))?
            .slice(0..batch_size);

        let func = load_func!(dev, "paged_attention_decode_bf16")?;
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, num_heads as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        match (
            &page_storage.slice,
            &query_storage.slice,
            &k_storage.slice,
            &v_storage.slice,
            &output_storage.slice,
        ) {
            (
                CudaStorageSlice::BF16(pages),
                CudaStorageSlice::BF16(query),
                CudaStorageSlice::BF16(current_k),
                CudaStorageSlice::BF16(current_v),
                CudaStorageSlice::BF16(output),
            ) => {
                let pages = pages.slice(page_o1..page_o2);
                let query = query.slice(query_o1..query_o2);
                let current_k = current_k.slice(k_o1..k_o2);
                let current_v = current_v.slice(v_o1..v_o2);
                let output = output.slice(output_o1..output_o2);
                let batch_size_i = batch_size as i32;
                let layer_i = layer as i32;
                let num_layers_i = num_layers as i32;
                let block_size_i = block_size as i32;
                let num_heads_i = num_heads as i32;
                let num_kv_heads_i = num_kv_heads as i32;
                let head_dim_i = head_dim as i32;
                let mut builder = func.builder();
                builder.arg(&pages);
                builder.arg(&query);
                builder.arg(&current_k);
                builder.arg(&current_v);
                builder.arg(&output);
                builder.arg(&indptr);
                builder.arg(&indices);
                builder.arg(&last_page_lens);
                builder.arg(&seq_lens);
                builder.arg(&batch_size_i);
                builder.arg(&layer_i);
                builder.arg(&num_layers_i);
                builder.arg(&block_size_i);
                builder.arg(&num_heads_i);
                builder.arg(&num_kv_heads_i);
                builder.arg(&head_dim_i);
                builder.arg(&scale);
                unsafe { builder.launch(cfg) }.w()?;
            }
            _ => candle_core::bail!(
                "paged_attention_decode_bf16_with_metadata expects BF16 CUDA storage"
            ),
        }
    }

    Ok(output)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn batch_kv_append_bf16_with_offset(
    dst_k: &Tensor,
    dst_v: &Tensor,
    src_k: &Tensor,
    src_v: &Tensor,
    append_offset: &Tensor,
    dst_width: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    if dst_k.dtype() != DType::BF16
        || dst_v.dtype() != DType::BF16
        || src_k.dtype() != DType::BF16
        || src_v.dtype() != DType::BF16
    {
        candle_core::bail!("batch_kv_append_bf16_with_offset expects BF16 K/V tensors")
    }
    if append_offset.dtype() != DType::U32 || append_offset.dims() != &[1] {
        candle_core::bail!("batch_kv_append_bf16_with_offset expects append_offset shape [1] U32")
    }

    let dst_dims = dst_k.dims4()?;
    if dst_v.dims4()? != dst_dims {
        candle_core::bail!("batch_kv_append_bf16_with_offset: destination K/V shapes differ")
    }
    let (batch_size, dst_heads, dst_seq, dst_head_dim) = dst_dims;
    if dst_seq != dst_width || dst_heads != num_kv_heads || dst_head_dim != head_dim {
        candle_core::bail!("batch_kv_append_bf16_with_offset: destination layout mismatch")
    }

    let src_dims = src_k.dims4()?;
    if src_v.dims4()? != src_dims || src_dims != (batch_size, num_kv_heads, 1, head_dim) {
        candle_core::bail!(
            "batch_kv_append_bf16_with_offset: source shape {:?} does not match [{batch_size}, {num_kv_heads}, 1, {head_dim}]",
            src_dims
        )
    }

    let dev = match dst_k.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("batch_kv_append_bf16_with_offset requires CUDA destination"),
    };
    for tensor in [dst_v, src_k, src_v, append_offset] {
        let same_device = matches!(tensor.device(), Device::Cuda(other) if other.id() == dev.id());
        if !same_device {
            candle_core::bail!(
                "batch_kv_append_bf16_with_offset tensors must be on the same CUDA device"
            )
        }
    }

    let (dst_k_storage, dst_k_layout) = dst_k.storage_and_layout();
    let dst_k_storage = match &*dst_k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("batch_kv_append_bf16_with_offset: expected CUDA dst K storage"),
    };
    let (dst_k_o1, dst_k_o2) = dst_k_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("batch_kv_append_bf16_with_offset: dst K must be contiguous".into())
    })?;

    let (dst_v_storage, dst_v_layout) = dst_v.storage_and_layout();
    let dst_v_storage = match &*dst_v_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("batch_kv_append_bf16_with_offset: expected CUDA dst V storage"),
    };
    let (dst_v_o1, dst_v_o2) = dst_v_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("batch_kv_append_bf16_with_offset: dst V must be contiguous".into())
    })?;

    let (src_k_storage, src_k_layout) = src_k.storage_and_layout();
    let src_k_storage = match &*src_k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("batch_kv_append_bf16_with_offset: expected CUDA src K storage"),
    };
    let (src_k_o1, src_k_o2) = src_k_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("batch_kv_append_bf16_with_offset: src K must be contiguous".into())
    })?;

    let (src_v_storage, src_v_layout) = src_v.storage_and_layout();
    let src_v_storage = match &*src_v_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("batch_kv_append_bf16_with_offset: expected CUDA src V storage"),
    };
    let (src_v_o1, src_v_o2) = src_v_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("batch_kv_append_bf16_with_offset: src V must be contiguous".into())
    })?;

    let (offset_storage, offset_layout) = append_offset.storage_and_layout();
    let offset_storage = match &*offset_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("batch_kv_append_bf16_with_offset: expected CUDA offset storage"),
    };
    let (offset_o1, offset_o2) = offset_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg(
            "batch_kv_append_bf16_with_offset: offset must be contiguous".into(),
        )
    })?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = load_func!(dev, "batch_kv_append_bf16_with_offset")?;

    match (
        &dst_k_storage.slice,
        &dst_v_storage.slice,
        &src_k_storage.slice,
        &src_v_storage.slice,
        &offset_storage.slice,
    ) {
        (
            CudaStorageSlice::BF16(dst_k),
            CudaStorageSlice::BF16(dst_v),
            CudaStorageSlice::BF16(src_k),
            CudaStorageSlice::BF16(src_v),
            CudaStorageSlice::U32(append_offset),
        ) => {
            let dst_k = dst_k.slice(dst_k_o1..dst_k_o2);
            let dst_v = dst_v.slice(dst_v_o1..dst_v_o2);
            let src_k = src_k.slice(src_k_o1..src_k_o2);
            let src_v = src_v.slice(src_v_o1..src_v_o2);
            let append_offset = append_offset.slice(offset_o1..offset_o2);
            let batch_size_i = batch_size as i32;
            let dst_width_i = dst_width as i32;
            let num_kv_heads_i = num_kv_heads as i32;
            let head_dim_i = head_dim as i32;
            let mut builder = func.builder();
            builder.arg(&dst_k);
            builder.arg(&dst_v);
            builder.arg(&src_k);
            builder.arg(&src_v);
            builder.arg(&append_offset);
            builder.arg(&batch_size_i);
            builder.arg(&dst_width_i);
            builder.arg(&num_kv_heads_i);
            builder.arg(&head_dim_i);
            unsafe { builder.launch(cfg) }.w()?;
        }
        _ => candle_core::bail!(
            "batch_kv_append_bf16_with_offset expects BF16 CUDA K/V and U32 offset storage"
        ),
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn batch_kv_copy_ragged_bf16(
    dst_k: &Tensor,
    dst_v: &Tensor,
    src_k: &Tensor,
    src_v: &Tensor,
    kv_lens: &Tensor,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    if dst_k.dtype() != DType::BF16
        || dst_v.dtype() != DType::BF16
        || src_k.dtype() != DType::BF16
        || src_v.dtype() != DType::BF16
    {
        candle_core::bail!("batch_kv_copy_ragged_bf16 expects BF16 K/V tensors")
    }
    if kv_lens.dtype() != DType::U32 {
        candle_core::bail!("batch_kv_copy_ragged_bf16 expects U32 kv_lens")
    }

    let src_dims = src_k.dims4()?;
    if src_v.dims4()? != src_dims {
        candle_core::bail!("batch_kv_copy_ragged_bf16: source K/V shapes differ")
    }
    let dst_dims = dst_k.dims4()?;
    if dst_v.dims4()? != dst_dims {
        candle_core::bail!("batch_kv_copy_ragged_bf16: destination K/V shapes differ")
    }
    let (batch_size, src_heads, src_width, src_head_dim) = src_dims;
    let (dst_batch, dst_heads, dst_width, dst_head_dim) = dst_dims;
    if batch_size == 0 || src_width == 0 {
        return Ok(());
    }
    if dst_batch != batch_size
        || src_heads != num_kv_heads
        || dst_heads != num_kv_heads
        || src_head_dim != head_dim
        || dst_head_dim != head_dim
        || dst_width < src_width
    {
        candle_core::bail!(
            "batch_kv_copy_ragged_bf16 layout mismatch: src={:?}, dst={:?}, expected heads={num_kv_heads}, head_dim={head_dim}, dst_width>=src_width",
            src_k.dims(),
            dst_k.dims()
        )
    }
    if kv_lens.dims() != &[batch_size] {
        candle_core::bail!(
            "batch_kv_copy_ragged_bf16 kv_lens shape {:?} does not match batch {batch_size}",
            kv_lens.dims()
        )
    }

    let dev = match dst_k.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("batch_kv_copy_ragged_bf16 requires CUDA destination"),
    };
    for tensor in [dst_v, src_k, src_v, kv_lens] {
        let same_device = matches!(tensor.device(), Device::Cuda(other) if other.id() == dev.id());
        if !same_device {
            candle_core::bail!("batch_kv_copy_ragged_bf16 tensors must share one CUDA device")
        }
    }

    let (dst_k_storage, dst_k_layout) = dst_k.storage_and_layout();
    let dst_k_storage = match &*dst_k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("batch_kv_copy_ragged_bf16: expected CUDA dst K storage"),
    };
    let (dst_k_o1, dst_k_o2) = dst_k_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("batch_kv_copy_ragged_bf16: dst K must be contiguous".into())
    })?;

    let (dst_v_storage, dst_v_layout) = dst_v.storage_and_layout();
    let dst_v_storage = match &*dst_v_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("batch_kv_copy_ragged_bf16: expected CUDA dst V storage"),
    };
    let (dst_v_o1, dst_v_o2) = dst_v_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("batch_kv_copy_ragged_bf16: dst V must be contiguous".into())
    })?;

    let (src_k_storage, src_k_layout) = src_k.storage_and_layout();
    let src_k_storage = match &*src_k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("batch_kv_copy_ragged_bf16: expected CUDA src K storage"),
    };
    let (src_k_o1, src_k_o2) = src_k_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("batch_kv_copy_ragged_bf16: src K must be contiguous".into())
    })?;

    let (src_v_storage, src_v_layout) = src_v.storage_and_layout();
    let src_v_storage = match &*src_v_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("batch_kv_copy_ragged_bf16: expected CUDA src V storage"),
    };
    let (src_v_o1, src_v_o2) = src_v_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("batch_kv_copy_ragged_bf16: src V must be contiguous".into())
    })?;

    let (kv_lens_storage, kv_lens_layout) = kv_lens.storage_and_layout();
    let kv_lens_storage = match &*kv_lens_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("batch_kv_copy_ragged_bf16: expected CUDA kv_lens storage"),
    };
    let (kv_lens_o1, kv_lens_o2) = kv_lens_layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("batch_kv_copy_ragged_bf16: kv_lens must be contiguous".into())
    })?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size as u32, src_width as u32, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let func = load_func!(dev, "batch_kv_copy_ragged_bf16")?;

    match (
        &dst_k_storage.slice,
        &dst_v_storage.slice,
        &src_k_storage.slice,
        &src_v_storage.slice,
        &kv_lens_storage.slice,
    ) {
        (
            CudaStorageSlice::BF16(dst_k),
            CudaStorageSlice::BF16(dst_v),
            CudaStorageSlice::BF16(src_k),
            CudaStorageSlice::BF16(src_v),
            CudaStorageSlice::U32(kv_lens),
        ) => {
            let dst_k = dst_k.slice(dst_k_o1..dst_k_o2);
            let dst_v = dst_v.slice(dst_v_o1..dst_v_o2);
            let src_k = src_k.slice(src_k_o1..src_k_o2);
            let src_v = src_v.slice(src_v_o1..src_v_o2);
            let kv_lens = kv_lens.slice(kv_lens_o1..kv_lens_o2);
            let batch_size_i = batch_size as i32;
            let src_width_i = src_width as i32;
            let dst_width_i = dst_width as i32;
            let num_kv_heads_i = num_kv_heads as i32;
            let head_dim_i = head_dim as i32;
            let mut builder = func.builder();
            builder.arg(&dst_k);
            builder.arg(&dst_v);
            builder.arg(&src_k);
            builder.arg(&src_v);
            builder.arg(&kv_lens);
            builder.arg(&batch_size_i);
            builder.arg(&src_width_i);
            builder.arg(&dst_width_i);
            builder.arg(&num_kv_heads_i);
            builder.arg(&head_dim_i);
            unsafe { builder.launch(cfg) }.w()?;
        }
        _ => candle_core::bail!(
            "batch_kv_copy_ragged_bf16 expects BF16 CUDA K/V and U32 kv_lens storage"
        ),
    }

    Ok(())
}

/// Append generated BF16 batch-decode K/V tokens into GPU page storage.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_kv_append_bf16(
    pages: &Tensor,
    full_k: &Tensor,
    full_v: &Tensor,
    page_ids: &[u32],
    token_offsets: &[u32],
    row_indices: &[u32],
    round_indices: &[u32],
    layer: usize,
    original_max_kv: usize,
    num_layers: usize,
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let source_token_indices: Vec<u32> = round_indices
        .iter()
        .map(|&round| original_max_kv as u32 + round)
        .collect();
    let mut buffers = PagedKvCopyMetadataCudaBuffers::new();
    buffers.upload(
        pages.device(),
        page_ids,
        token_offsets,
        row_indices,
        &source_token_indices,
    )?;
    paged_kv_copy_bf16_with_metadata(
        pages,
        full_k,
        full_v,
        layer,
        num_layers,
        block_size,
        num_kv_heads,
        head_dim,
        &buffers,
    )
}

/// Zero released BF16 page slots so completed requests do not retain readable KV content.
#[cfg(feature = "cuda")]
pub fn paged_kv_zero_pages_bf16(
    pages: &Tensor,
    page_ids: &[u32],
    page_values: usize,
) -> Result<()> {
    if page_ids.is_empty() {
        return Ok(());
    }
    if pages.dtype() != DType::BF16 {
        candle_core::bail!("paged_kv_zero_pages_bf16 expects BF16 page storage")
    }
    let dev = match pages.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("paged_kv_zero_pages_bf16 requires CUDA page storage"),
    };
    let (storage, layout) = pages.storage_and_layout();
    let storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("paged_kv_zero_pages_bf16: expected CUDA storage"),
    };
    let (o1, o2) = layout.contiguous_offsets().ok_or_else(|| {
        candle_core::Error::Msg("paged_kv_zero_pages_bf16: pages must be contiguous".into())
    })?;

    let mut page_ids_dev = unsafe { dev.alloc::<u32>(page_ids.len())? };
    dev.memcpy_htod(page_ids, &mut page_ids_dev)?;

    let func = load_func!(dev, "paged_kv_zero_pages_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (page_ids.len() as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    match &storage.slice {
        CudaStorageSlice::BF16(pages) => {
            let pages = pages.slice(o1..o2);
            let num_pages_i = page_ids.len() as i32;
            let page_values_i = page_values as i32;
            let mut builder = func.builder();
            builder.arg(&pages);
            builder.arg(&page_ids_dev);
            builder.arg(&num_pages_i);
            builder.arg(&page_values_i);
            unsafe { builder.launch(cfg) }.w()?;
        }
        _ => candle_core::bail!("paged_kv_zero_pages_bf16 expects BF16 CUDA storage"),
    }

    Ok(())
}
