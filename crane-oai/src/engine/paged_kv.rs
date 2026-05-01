//! Qwen3-only paged KV metadata.
//!
//! This module intentionally tracks page tables before it owns real KV tensors.
//! The first migration step is metadata/lifetime correctness; the next step can
//! use the same tables to gather pages into the existing contiguous fallback.

use std::{collections::HashMap, error::Error, fmt};

use candle_core::{DType, Device, Shape, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvLayout {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub dtype_size_bytes: usize,
}

impl PagedKvLayout {
    pub fn bytes_per_block(&self, block_size: usize) -> u64 {
        (self.num_layers as u64)
            * 2
            * (self.num_kv_heads as u64)
            * (block_size as u64)
            * (self.head_dim as u64)
            * (self.dtype_size_bytes as u64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvBlock {
    pub id: u64,
    pub start_token: usize,
    pub len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PagedKvPlane {
    Key,
    Value,
}

impl PagedKvPlane {
    fn index(self) -> usize {
        match self {
            Self::Key => 0,
            Self::Value => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum PagedKvStorageError {
    BlockSizeMismatch {
        storage_block_size: usize,
        sequence_block_size: usize,
    },
    InvalidLayer {
        layer: usize,
        num_layers: usize,
    },
    InvalidTokenIndex {
        token_index: usize,
        token_len: usize,
    },
    InvalidTokenOffset {
        token_offset: usize,
        block_size: usize,
    },
    MissingBlock {
        token_index: usize,
        block_index: usize,
    },
    MissingPage {
        page_id: u64,
    },
    SequenceTooLong {
        token_len: usize,
        max_len: usize,
    },
    TokenWidthMismatch {
        expected: usize,
        actual: usize,
    },
    LayerValueLenMismatch {
        expected: usize,
        actual: usize,
    },
    SequenceCountMismatch {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for PagedKvStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlockSizeMismatch {
                storage_block_size,
                sequence_block_size,
            } => write!(
                f,
                "paged KV block size mismatch: storage={storage_block_size}, sequence={sequence_block_size}"
            ),
            Self::InvalidLayer { layer, num_layers } => {
                write!(f, "paged KV layer {layer} is outside {num_layers} layers")
            }
            Self::InvalidTokenIndex {
                token_index,
                token_len,
            } => write!(
                f,
                "paged KV token index {token_index} is outside token length {token_len}"
            ),
            Self::InvalidTokenOffset {
                token_offset,
                block_size,
            } => write!(
                f,
                "paged KV token offset {token_offset} is outside block size {block_size}"
            ),
            Self::MissingBlock {
                token_index,
                block_index,
            } => write!(
                f,
                "paged KV token {token_index} maps to missing block index {block_index}"
            ),
            Self::MissingPage { page_id } => write!(f, "paged KV page {page_id} is not initialized"),
            Self::SequenceTooLong { token_len, max_len } => write!(
                f,
                "paged KV sequence length {token_len} exceeds gather width {max_len}"
            ),
            Self::TokenWidthMismatch { expected, actual } => write!(
                f,
                "paged KV token width mismatch: expected {expected}, got {actual}"
            ),
            Self::LayerValueLenMismatch { expected, actual } => write!(
                f,
                "paged KV layer value length mismatch: expected {expected}, got {actual}"
            ),
            Self::SequenceCountMismatch { expected, actual } => write!(
                f,
                "paged KV sequence count mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl Error for PagedKvStorageError {}

#[allow(dead_code)]
type PagedKvStorageResult<T> = std::result::Result<T, PagedKvStorageError>;

pub fn build_right_aligned_head_major_batch<T>(
    layout: PagedKvLayout,
    sequence_lens: &[usize],
    max_len: usize,
    values: &[Vec<T>],
) -> PagedKvStorageResult<Vec<T>>
where
    T: Clone + Default,
{
    if values.len() != sequence_lens.len() {
        return Err(PagedKvStorageError::SequenceCountMismatch {
            expected: sequence_lens.len(),
            actual: values.len(),
        });
    }

    let row_stride = layout.num_kv_heads * max_len * layout.head_dim;
    let mut output = vec![T::default(); sequence_lens.len() * row_stride];
    for (row, (&seq_len, row_values)) in sequence_lens.iter().zip(values.iter()).enumerate() {
        if seq_len > max_len {
            return Err(PagedKvStorageError::SequenceTooLong {
                token_len: seq_len,
                max_len,
            });
        }
        let expected = seq_len * layout.num_kv_heads * layout.head_dim;
        if row_values.len() != expected {
            return Err(PagedKvStorageError::LayerValueLenMismatch {
                expected,
                actual: row_values.len(),
            });
        }

        let target_start = max_len - seq_len;
        for token_index in 0..seq_len {
            for head in 0..layout.num_kv_heads {
                let src = head * seq_len * layout.head_dim + token_index * layout.head_dim;
                let dst = row * row_stride
                    + head * max_len * layout.head_dim
                    + (target_start + token_index) * layout.head_dim;
                output[dst..dst + layout.head_dim]
                    .clone_from_slice(&row_values[src..src + layout.head_dim]);
            }
        }
    }

    Ok(output)
}

pub fn gather_head_major_layer_via_pages<T>(
    block_size: usize,
    layout: PagedKvLayout,
    sequences: &[&PagedKvSequence],
    layer: usize,
    plane: PagedKvPlane,
    values: &[Vec<T>],
    max_len: usize,
) -> PagedKvStorageResult<Vec<T>>
where
    T: Clone + Default,
{
    if values.len() != sequences.len() {
        return Err(PagedKvStorageError::SequenceCountMismatch {
            expected: sequences.len(),
            actual: values.len(),
        });
    }

    let mut store = PagedKvPageStore::<T>::new(block_size, layout);
    for (sequence, row_values) in sequences.iter().zip(values.iter()) {
        store.write_sequence_layer_head_major(sequence, layer, plane, row_values)?;
    }
    store.gather_layer_right_aligned(sequences, layer, max_len, plane)
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PagedKvPageStore<T> {
    block_size: usize,
    layout: PagedKvLayout,
    pages: HashMap<u64, Vec<T>>,
}

#[allow(dead_code)]
impl<T> PagedKvPageStore<T>
where
    T: Clone + Default,
{
    pub fn new(block_size: usize, layout: PagedKvLayout) -> Self {
        Self {
            block_size: block_size.max(1),
            layout,
            pages: HashMap::new(),
        }
    }

    pub fn token_width(&self) -> usize {
        self.layout.num_kv_heads * self.layout.head_dim
    }

    pub fn reset_page(&mut self, page_id: u64) {
        self.pages
            .insert(page_id, vec![T::default(); self.values_per_page()]);
    }

    pub fn reset_sequence_pages(&mut self, sequence: &PagedKvSequence) {
        for block in &sequence.blocks {
            self.reset_page(block.id);
        }
    }

    pub fn write_token(
        &mut self,
        sequence: &PagedKvSequence,
        token_index: usize,
        layer: usize,
        plane: PagedKvPlane,
        values: &[T],
    ) -> PagedKvStorageResult<()> {
        self.validate_token_width(values.len())?;
        let (page_id, token_offset) = self.page_slot(sequence, token_index)?;
        let offset = self.page_offset(layer, plane, token_offset, 0)?;
        let page = self.page_mut(page_id);
        page[offset..offset + values.len()].clone_from_slice(values);
        Ok(())
    }

    pub fn write_sequence_layer_head_major(
        &mut self,
        sequence: &PagedKvSequence,
        layer: usize,
        plane: PagedKvPlane,
        values: &[T],
    ) -> PagedKvStorageResult<()> {
        self.validate_layer(layer)?;
        self.validate_sequence_block_size(sequence)?;
        let expected = sequence.token_len * self.token_width();
        if values.len() != expected {
            return Err(PagedKvStorageError::LayerValueLenMismatch {
                expected,
                actual: values.len(),
            });
        }

        let head_dim = self.layout.head_dim;
        for token_index in 0..sequence.token_len {
            let (page_id, token_offset) = self.page_slot(sequence, token_index)?;
            for head in 0..self.layout.num_kv_heads {
                let src = head * sequence.token_len * head_dim + token_index * head_dim;
                let dst = self.page_offset(layer, plane, token_offset, head)?;
                let page = self.page_mut(page_id);
                page[dst..dst + head_dim].clone_from_slice(&values[src..src + head_dim]);
            }
        }

        Ok(())
    }

    pub fn write_sequence_layer_kv_head_major(
        &mut self,
        sequence: &PagedKvSequence,
        layer: usize,
        key_values: &[T],
        value_values: &[T],
    ) -> PagedKvStorageResult<()> {
        self.write_sequence_layer_head_major(sequence, layer, PagedKvPlane::Key, key_values)?;
        self.write_sequence_layer_head_major(sequence, layer, PagedKvPlane::Value, value_values)
    }

    pub fn read_token(
        &self,
        sequence: &PagedKvSequence,
        token_index: usize,
        layer: usize,
        plane: PagedKvPlane,
    ) -> PagedKvStorageResult<Vec<T>> {
        let (page_id, token_offset) = self.page_slot(sequence, token_index)?;
        let offset = self.page_offset(layer, plane, token_offset, 0)?;
        let width = self.token_width();
        let page = self
            .pages
            .get(&page_id)
            .ok_or(PagedKvStorageError::MissingPage { page_id })?;
        Ok(page[offset..offset + width].to_vec())
    }

    pub fn gather_layer_right_aligned(
        &self,
        sequences: &[&PagedKvSequence],
        layer: usize,
        max_len: usize,
        plane: PagedKvPlane,
    ) -> PagedKvStorageResult<Vec<T>> {
        self.validate_layer(layer)?;
        let row_stride = self.layout.num_kv_heads * max_len * self.layout.head_dim;
        let mut output = vec![T::default(); sequences.len() * row_stride];

        for (row, sequence) in sequences.iter().enumerate() {
            if sequence.block_size != self.block_size {
                return Err(PagedKvStorageError::BlockSizeMismatch {
                    storage_block_size: self.block_size,
                    sequence_block_size: sequence.block_size,
                });
            }
            if sequence.token_len > max_len {
                return Err(PagedKvStorageError::SequenceTooLong {
                    token_len: sequence.token_len,
                    max_len,
                });
            }

            let target_start = max_len - sequence.token_len;
            for token_index in 0..sequence.token_len {
                let (page_id, token_offset) = self.page_slot(sequence, token_index)?;
                let page = self
                    .pages
                    .get(&page_id)
                    .ok_or(PagedKvStorageError::MissingPage { page_id })?;
                let target_token = target_start + token_index;
                for head in 0..self.layout.num_kv_heads {
                    let src = self.page_offset(layer, plane, token_offset, head)?;
                    let dst = row * row_stride
                        + head * max_len * self.layout.head_dim
                        + target_token * self.layout.head_dim;
                    output[dst..dst + self.layout.head_dim]
                        .clone_from_slice(&page[src..src + self.layout.head_dim]);
                }
            }
        }

        Ok(output)
    }

    fn values_per_page(&self) -> usize {
        self.layout.num_layers * 2 * self.block_size * self.token_width()
    }

    fn page_mut(&mut self, page_id: u64) -> &mut Vec<T> {
        let values_per_page = self.values_per_page();
        self.pages
            .entry(page_id)
            .or_insert_with(|| vec![T::default(); values_per_page])
    }

    fn page_slot(
        &self,
        sequence: &PagedKvSequence,
        token_index: usize,
    ) -> PagedKvStorageResult<(u64, usize)> {
        self.validate_sequence_block_size(sequence)?;
        if token_index >= sequence.token_len {
            return Err(PagedKvStorageError::InvalidTokenIndex {
                token_index,
                token_len: sequence.token_len,
            });
        }

        let block_index = token_index / self.block_size;
        let token_offset = token_index % self.block_size;
        let block = sequence
            .blocks
            .get(block_index)
            .ok_or(PagedKvStorageError::MissingBlock {
                token_index,
                block_index,
            })?;
        Ok((block.id, token_offset))
    }

    fn page_offset(
        &self,
        layer: usize,
        plane: PagedKvPlane,
        token_offset: usize,
        head: usize,
    ) -> PagedKvStorageResult<usize> {
        self.validate_layer(layer)?;
        if token_offset >= self.block_size {
            return Err(PagedKvStorageError::InvalidTokenOffset {
                token_offset,
                block_size: self.block_size,
            });
        }
        let layer_stride = 2 * self.block_size * self.token_width();
        let plane_stride = self.block_size * self.token_width();
        Ok(layer * layer_stride
            + plane.index() * plane_stride
            + token_offset * self.token_width()
            + head * self.layout.head_dim)
    }

    fn validate_layer(&self, layer: usize) -> PagedKvStorageResult<()> {
        if layer >= self.layout.num_layers {
            return Err(PagedKvStorageError::InvalidLayer {
                layer,
                num_layers: self.layout.num_layers,
            });
        }
        Ok(())
    }

    fn validate_sequence_block_size(&self, sequence: &PagedKvSequence) -> PagedKvStorageResult<()> {
        if sequence.block_size != self.block_size {
            return Err(PagedKvStorageError::BlockSizeMismatch {
                storage_block_size: self.block_size,
                sequence_block_size: sequence.block_size,
            });
        }
        Ok(())
    }

    fn validate_token_width(&self, actual: usize) -> PagedKvStorageResult<()> {
        let expected = self.token_width();
        if actual != expected {
            return Err(PagedKvStorageError::TokenWidthMismatch { expected, actual });
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PagedKvSequence {
    block_size: usize,
    token_len: usize,
    gpu_resident_token_len: usize,
    blocks: Vec<PagedKvBlock>,
}

impl PagedKvSequence {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(1),
            token_len: 0,
            gpu_resident_token_len: 0,
            blocks: Vec::new(),
        }
    }

    pub fn token_len(&self) -> usize {
        self.token_len
    }

    pub fn gpu_resident_token_len(&self) -> usize {
        self.gpu_resident_token_len.min(self.token_len)
    }

    pub fn mark_gpu_resident(&mut self, token_len: usize) {
        self.gpu_resident_token_len = token_len.min(self.token_len);
    }

    #[allow(dead_code)]
    pub fn blocks(&self) -> &[PagedKvBlock] {
        &self.blocks
    }

    pub fn page_slot(&self, token_index: usize) -> PagedKvStorageResult<(u64, usize)> {
        if token_index >= self.token_len {
            return Err(PagedKvStorageError::InvalidTokenIndex {
                token_index,
                token_len: self.token_len,
            });
        }
        let block_index = token_index / self.block_size;
        let token_offset = token_index % self.block_size;
        let block = self
            .blocks
            .get(block_index)
            .ok_or(PagedKvStorageError::MissingBlock {
                token_index,
                block_index,
            })?;
        Ok((block.id, token_offset))
    }

    pub fn reserved_tokens(&self) -> usize {
        self.blocks.len() * self.block_size
    }

    pub fn fragmentation_tokens(&self) -> usize {
        self.reserved_tokens().saturating_sub(self.token_len)
    }

    fn retarget(&mut self, block_size: usize) {
        if self.block_size == block_size {
            return;
        }
        self.block_size = block_size.max(1);
        self.token_len = 0;
        self.gpu_resident_token_len = 0;
        self.blocks.clear();
    }

    fn refresh_block_lengths(&mut self) {
        for (idx, block) in self.blocks.iter_mut().enumerate() {
            let start = idx * self.block_size;
            block.start_token = start;
            block.len = self.token_len.saturating_sub(start).min(self.block_size);
        }
    }
}

impl Default for PagedKvSequence {
    fn default() -> Self {
        Self::new(DEFAULT_BLOCK_SIZE)
    }
}

#[derive(Debug, Clone, Default)]
pub struct PagedKvNativeAppendPlan {
    pub page_ids: Vec<u32>,
    pub token_offsets: Vec<u32>,
    pub row_indices: Vec<u32>,
    pub source_token_indices: Vec<u32>,
}

impl PagedKvNativeAppendPlan {
    pub fn entries(&self) -> usize {
        self.page_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.page_ids.is_empty()
    }

    pub fn push(
        &mut self,
        page_id: u64,
        token_offset: usize,
        row_index: usize,
        source_token_index: usize,
    ) -> PagedKvStorageResult<()> {
        let page_id =
            u32::try_from(page_id).map_err(|_| PagedKvStorageError::MissingPage { page_id })?;
        let source_token_index = u32::try_from(source_token_index).map_err(|_| {
            PagedKvStorageError::InvalidTokenIndex {
                token_index: source_token_index,
                token_len: usize::MAX,
            }
        })?;
        self.page_ids.push(page_id);
        self.token_offsets.push(token_offset as u32);
        self.row_indices.push(row_index as u32);
        self.source_token_indices.push(source_token_index);
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PagedKvGatherPlan {
    pub page_ids: Vec<u32>,
    pub token_offsets: Vec<u32>,
    pub row_indices: Vec<u32>,
    pub target_token_indices: Vec<u32>,
}

impl PagedKvGatherPlan {
    pub fn entries(&self) -> usize {
        self.page_ids.len()
    }

    pub fn from_optional_sequences(
        block_size: usize,
        sequences: &[Option<&PagedKvSequence>],
        max_len: usize,
    ) -> PagedKvStorageResult<Self> {
        let mut plan = Self::default();
        for (row, sequence) in sequences.iter().enumerate() {
            let Some(sequence) = sequence else {
                continue;
            };
            if sequence.block_size != block_size {
                return Err(PagedKvStorageError::BlockSizeMismatch {
                    storage_block_size: block_size,
                    sequence_block_size: sequence.block_size,
                });
            }
            if sequence.token_len > max_len {
                return Err(PagedKvStorageError::SequenceTooLong {
                    token_len: sequence.token_len,
                    max_len,
                });
            }

            let target_start = max_len - sequence.token_len;
            for token_index in 0..sequence.token_len {
                let (page_id, token_offset) = sequence.page_slot(token_index)?;
                let page_id = u32::try_from(page_id)
                    .map_err(|_| PagedKvStorageError::MissingPage { page_id })?;
                let token_offset = u32::try_from(token_offset).map_err(|_| {
                    PagedKvStorageError::InvalidTokenOffset {
                        token_offset,
                        block_size,
                    }
                })?;
                let row_index =
                    u32::try_from(row).map_err(|_| PagedKvStorageError::InvalidTokenIndex {
                        token_index: row,
                        token_len: usize::MAX,
                    })?;
                let target_token_index =
                    u32::try_from(target_start + token_index).map_err(|_| {
                        PagedKvStorageError::InvalidTokenIndex {
                            token_index: target_start + token_index,
                            token_len: max_len,
                        }
                    })?;
                plan.page_ids.push(page_id);
                plan.token_offsets.push(token_offset);
                plan.row_indices.push(row_index);
                plan.target_token_indices.push(target_token_index);
            }
        }
        Ok(plan)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PagedKvBatchPageTable {
    pub indptr: Vec<u32>,
    pub indices: Vec<u32>,
    pub last_page_lens: Vec<u32>,
    pub seq_lens: Vec<u32>,
}

#[allow(dead_code)]
impl PagedKvBatchPageTable {
    pub fn from_sequences(
        block_size: usize,
        sequences: &[&PagedKvSequence],
    ) -> PagedKvStorageResult<Self> {
        let optional: Vec<Option<&PagedKvSequence>> = sequences.iter().copied().map(Some).collect();
        Self::from_optional_sequences(block_size, &optional)
    }

    pub fn from_optional_sequences(
        block_size: usize,
        sequences: &[Option<&PagedKvSequence>],
    ) -> PagedKvStorageResult<Self> {
        let mut table = Self {
            indptr: Vec::with_capacity(sequences.len() + 1),
            indices: Vec::new(),
            last_page_lens: Vec::with_capacity(sequences.len()),
            seq_lens: Vec::with_capacity(sequences.len()),
        };
        table.indptr.push(0);
        for sequence in sequences {
            let Some(sequence) = sequence else {
                table
                    .indptr
                    .push(u32::try_from(table.indices.len()).map_err(|_| {
                        PagedKvStorageError::InvalidTokenIndex {
                            token_index: table.indices.len(),
                            token_len: usize::MAX,
                        }
                    })?);
                table.seq_lens.push(0);
                table.last_page_lens.push(0);
                continue;
            };
            if sequence.block_size != block_size {
                return Err(PagedKvStorageError::BlockSizeMismatch {
                    storage_block_size: block_size,
                    sequence_block_size: sequence.block_size,
                });
            }
            for block in &sequence.blocks {
                table.indices.push(
                    u32::try_from(block.id)
                        .map_err(|_| PagedKvStorageError::MissingPage { page_id: block.id })?,
                );
            }
            table
                .indptr
                .push(u32::try_from(table.indices.len()).map_err(|_| {
                    PagedKvStorageError::InvalidTokenIndex {
                        token_index: table.indices.len(),
                        token_len: usize::MAX,
                    }
                })?);
            table
                .seq_lens
                .push(u32::try_from(sequence.token_len).map_err(|_| {
                    PagedKvStorageError::InvalidTokenIndex {
                        token_index: sequence.token_len,
                        token_len: usize::MAX,
                    }
                })?);
            let last_page_len = if sequence.token_len == 0 {
                0
            } else {
                ((sequence.token_len - 1) % block_size + 1) as u32
            };
            table.last_page_lens.push(last_page_len);
        }
        Ok(table)
    }
}

pub struct PagedKvGpuPageStore {
    block_size: usize,
    layout: PagedKvLayout,
    dtype: DType,
    device: Device,
    pages: Option<Tensor>,
    capacity_pages: usize,
    copy_metadata: crane_core::fused_ops::PagedKvCopyMetadataCudaBuffers,
}

impl PagedKvGpuPageStore {
    pub fn new(block_size: usize, layout: PagedKvLayout, dtype: DType, device: &Device) -> Self {
        Self {
            block_size: block_size.max(1),
            layout,
            dtype,
            device: device.clone(),
            pages: None,
            capacity_pages: 0,
            copy_metadata: crane_core::fused_ops::PagedKvCopyMetadataCudaBuffers::new(),
        }
    }

    pub fn capacity_pages(&self) -> usize {
        self.capacity_pages
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn layout(&self) -> PagedKvLayout {
        self.layout
    }

    pub fn pages(&self) -> Option<Tensor> {
        self.pages.clone()
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_pages as u64 * self.layout.bytes_per_block(self.block_size)
    }

    pub fn release_cached_storage(&mut self) -> usize {
        let released_pages = self.capacity_pages;
        self.pages = None;
        self.capacity_pages = 0;
        self.copy_metadata.release();
        released_pages
    }

    pub fn ensure_capacity(&mut self, max_page_id: u64) -> candle_core::Result<()> {
        let needed = usize::try_from(max_page_id).map_err(|_| {
            candle_core::Error::Msg(format!("paged KV page id {max_page_id} exceeds usize"))
        })?;
        if needed <= self.capacity_pages {
            return Ok(());
        }
        if self.layout.num_layers == 0 || self.layout.num_kv_heads == 0 || self.layout.head_dim == 0
        {
            return Ok(());
        }

        let new_capacity = needed.next_power_of_two();
        let new_pages = Tensor::zeros(
            Shape::from_dims(&[
                new_capacity,
                self.layout.num_layers,
                2,
                self.block_size,
                self.layout.num_kv_heads,
                self.layout.head_dim,
            ]),
            self.dtype,
            &self.device,
        )?;
        if let Some(old_pages) = self.pages.as_ref() {
            new_pages.slice_set(old_pages, 0, 0)?;
        }
        self.pages = Some(new_pages);
        self.capacity_pages = new_capacity;
        Ok(())
    }

    pub fn copy_layers_from_cache_buffers(
        &mut self,
        layer_buffers: &[Option<(Tensor, Tensor)>],
        plan: &PagedKvNativeAppendPlan,
    ) -> candle_core::Result<usize> {
        if plan.is_empty() {
            return Ok(0);
        }
        if self.dtype != DType::BF16 {
            candle_core::bail!("paged KV native append currently expects BF16 page storage")
        }
        if layer_buffers.len() != self.layout.num_layers {
            candle_core::bail!(
                "paged KV copy expected {} layers, got {}",
                self.layout.num_layers,
                layer_buffers.len()
            );
        }
        let max_page_id = plan.page_ids.iter().copied().max().unwrap_or(0) as u64;
        self.ensure_capacity(max_page_id)?;
        let pages = self.pages.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("paged KV page tensor is not allocated".into())
        })?;

        self.copy_metadata.upload(
            &self.device,
            &plan.page_ids,
            &plan.token_offsets,
            &plan.row_indices,
            &plan.source_token_indices,
        )?;

        let mut layers = 0usize;
        for (layer, cache) in layer_buffers.iter().enumerate() {
            let Some((full_k, full_v)) = cache else {
                candle_core::bail!("paged KV copy missing batch KV buffer for layer {layer}");
            };
            crane_core::fused_ops::paged_kv_copy_bf16_with_metadata(
                pages,
                full_k,
                full_v,
                layer,
                self.layout.num_layers,
                self.block_size,
                self.layout.num_kv_heads,
                self.layout.head_dim,
                &self.copy_metadata,
            )?;
            layers += 1;
        }

        Ok(layers)
    }

    pub fn gather_layer_right_aligned(
        &mut self,
        sequences: &[Option<&PagedKvSequence>],
        layer: usize,
        plane: PagedKvPlane,
        max_len: usize,
    ) -> candle_core::Result<Tensor> {
        if self.dtype != DType::BF16 {
            candle_core::bail!("paged KV GPU gather currently expects BF16 page storage")
        }
        if layer >= self.layout.num_layers {
            candle_core::bail!(
                "paged KV gather layer {layer} outside {} layers",
                self.layout.num_layers
            );
        }
        let output = Tensor::zeros(
            (
                sequences.len(),
                self.layout.num_kv_heads,
                max_len,
                self.layout.head_dim,
            ),
            self.dtype,
            &self.device,
        )?;
        let plan = PagedKvGatherPlan::from_optional_sequences(self.block_size, sequences, max_len)
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?;
        if plan.entries() == 0 {
            return Ok(output);
        }
        let pages = self.pages.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("paged KV page tensor is not allocated".into())
        })?;
        self.copy_metadata.upload(
            &self.device,
            &plan.page_ids,
            &plan.token_offsets,
            &plan.row_indices,
            &plan.target_token_indices,
        )?;
        crane_core::fused_ops::paged_kv_gather_bf16_with_metadata(
            pages,
            &output,
            layer,
            plane.index(),
            max_len,
            self.layout.num_layers,
            self.block_size,
            self.layout.num_kv_heads,
            self.layout.head_dim,
            &self.copy_metadata,
        )?;
        Ok(output)
    }

    pub fn zero_pages(&mut self, page_ids: &[u64]) -> candle_core::Result<()> {
        if page_ids.is_empty() || self.pages.is_none() {
            return Ok(());
        }
        if self.dtype != DType::BF16 {
            return Ok(());
        }
        let page_ids_u32: Vec<u32> = page_ids
            .iter()
            .copied()
            .filter(|&id| id > 0 && id as usize <= self.capacity_pages)
            .filter_map(|id| u32::try_from(id).ok())
            .collect();
        if page_ids_u32.is_empty() {
            return Ok(());
        }
        let pages = self.pages.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("paged KV page tensor is not allocated".into())
        })?;
        crane_core::fused_ops::paged_kv_zero_pages_bf16(pages, &page_ids_u32, self.page_values())
    }

    fn page_values(&self) -> usize {
        self.layout.num_layers
            * 2
            * self.block_size
            * self.layout.num_kv_heads
            * self.layout.head_dim
    }
}

pub const DEFAULT_BLOCK_SIZE: usize = 16;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PagedKvUpdate {
    pub allocated_pages: u64,
    pub reused_pages: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PagedKvCompaction {
    pub live_pages: u64,
    pub moved_pages: u64,
    pub dropped_free_pages: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedKvAllocatorSnapshot {
    pub block_size: usize,
    pub live_pages: u64,
    pub free_pages: u64,
    pub total_alloc_pages: u64,
    pub total_reused_pages: u64,
    pub total_freed_pages: u64,
    pub reserved_bytes: u64,
}

pub struct PagedKvAllocator {
    block_size: usize,
    layout: PagedKvLayout,
    next_page_id: u64,
    free_pages: Vec<u64>,
    live_pages: u64,
    total_alloc_pages: u64,
    total_reused_pages: u64,
    total_freed_pages: u64,
}

impl PagedKvAllocator {
    pub fn new(block_size: usize, layout: PagedKvLayout) -> Self {
        Self {
            block_size: block_size.max(1),
            layout,
            next_page_id: 1,
            free_pages: Vec::new(),
            live_pages: 0,
            total_alloc_pages: 0,
            total_reused_pages: 0,
            total_freed_pages: 0,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn layout(&self) -> PagedKvLayout {
        self.layout
    }

    pub fn bytes_per_block(&self) -> u64 {
        self.layout.bytes_per_block(self.block_size)
    }

    pub fn ensure_token_len(
        &mut self,
        sequence: &mut PagedKvSequence,
        token_len: usize,
    ) -> PagedKvUpdate {
        if sequence.block_size != self.block_size {
            sequence.retarget(self.block_size);
        }

        let required_pages = token_len.div_ceil(self.block_size);
        let mut update = PagedKvUpdate::default();
        while sequence.blocks.len() < required_pages {
            let (id, reused) = self.acquire_page();
            let idx = sequence.blocks.len();
            sequence.blocks.push(PagedKvBlock {
                id,
                start_token: idx * self.block_size,
                len: 0,
            });
            if reused {
                update.reused_pages += 1;
            } else {
                update.allocated_pages += 1;
            }
        }
        sequence.token_len = token_len;
        sequence.gpu_resident_token_len = sequence.gpu_resident_token_len.min(token_len);
        sequence.refresh_block_lengths();
        update
    }

    pub fn release_sequence(&mut self, sequence: &mut PagedKvSequence) -> u64 {
        let released = sequence.blocks.len() as u64;
        for block in sequence.blocks.drain(..) {
            self.free_pages.push(block.id);
        }
        sequence.token_len = 0;
        sequence.gpu_resident_token_len = 0;
        self.live_pages = self.live_pages.saturating_sub(released);
        self.total_freed_pages += released;
        released
    }

    pub fn reset_when_idle(&mut self) -> u64 {
        if self.live_pages != 0 {
            return 0;
        }
        let dropped = self.free_pages.len() as u64;
        self.free_pages.clear();
        self.next_page_id = 1;
        dropped
    }

    pub fn compact_sequences<'a, I>(&mut self, sequences: I) -> PagedKvCompaction
    where
        I: IntoIterator<Item = &'a mut PagedKvSequence>,
    {
        let dropped_free_pages = self.free_pages.len() as u64;
        self.free_pages.clear();

        let mut next_page_id = 1u64;
        let mut live_pages = 0u64;
        let mut moved_pages = 0u64;
        for sequence in sequences {
            for block in &mut sequence.blocks {
                if block.id != next_page_id {
                    moved_pages += 1;
                }
                block.id = next_page_id;
                next_page_id += 1;
                live_pages += 1;
            }
            sequence.gpu_resident_token_len = 0;
            sequence.refresh_block_lengths();
        }

        self.next_page_id = next_page_id;
        self.live_pages = live_pages;

        PagedKvCompaction {
            live_pages,
            moved_pages,
            dropped_free_pages,
        }
    }

    pub fn snapshot(&self) -> PagedKvAllocatorSnapshot {
        PagedKvAllocatorSnapshot {
            block_size: self.block_size,
            live_pages: self.live_pages,
            free_pages: self.free_pages.len() as u64,
            total_alloc_pages: self.total_alloc_pages,
            total_reused_pages: self.total_reused_pages,
            total_freed_pages: self.total_freed_pages,
            reserved_bytes: self.live_pages * self.bytes_per_block(),
        }
    }

    fn acquire_page(&mut self) -> (u64, bool) {
        if let Some(id) = self.free_pages.pop() {
            self.live_pages += 1;
            self.total_reused_pages += 1;
            return (id, true);
        }

        let id = self.next_page_id;
        self.next_page_id += 1;
        self.live_pages += 1;
        self.total_alloc_pages += 1;
        (id, false)
    }
}

#[cfg(test)]
mod tests;
