//! Runtime integration for the Qwen3 paged-KV migration path.
//!
//! This module keeps the migration-specific shadow validation, GPU page import,
//! native append, page release, and page stats logic out of the main engine loop.

use std::sync::atomic::Ordering;
use std::time::Instant;

use anyhow::{bail, Context, Result as AnyhowResult};
use candle_core::{DType, Tensor};
use tracing::{debug, info, warn};

use super::paged_kv::{
    build_right_aligned_head_major_batch, gather_head_major_layer_via_pages, PagedKvBatchPageTable,
    PagedKvNativeAppendPlan, PagedKvPlane,
};
use super::sequence::Sequence;
use super::{format_bytes_engine, query_gpu_memory_usage, InferenceEngine};

#[derive(Debug, Clone, Copy, Default)]
struct PagedKvShadowGatherReport {
    layers: usize,
    sequences: usize,
    values_compared: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct PagedKvNativeAppendReport {
    layers: usize,
    entries: usize,
    capacity_pages: usize,
}

/// Output of the gather-based paged-KV extract path.
///
/// Holds the per-layer `(key_batch, value_batch)` tensors of shape
/// `[batch, kv_heads, max_total_len, head_dim]`, right-aligned per-row inside
/// `max_total_len`.
#[derive(Debug, Clone)]
pub(super) struct BatchedKvExtract {
    /// One `(K, V)` per model layer. `None` means "no data for this layer"
    /// (currently unused — every layer is populated when there is any past).
    pub per_layer: Vec<Option<(Tensor, Tensor)>>,
    /// Width of dim 2 in every `per_layer` tensor; per-row data is right-aligned.
    pub max_total_len: usize,
    /// Per-row total token count (`kv_lens[r] + rounds_done`); `None` rows had no past.
    pub per_row_totals: Vec<Option<usize>>,
}

impl BatchedKvExtract {
    /// Materialize the batched extract back into per-sequence per-layer
    /// `(K, V)` tensors. Used by the legacy (Round 8) path when the Round 9
    /// batched-setup fast path is disabled. Returns `Vec<Vec<Option<(K,V)>>>`
    /// indexed by `[row][layer]`.
    pub(super) fn materialize_per_row(
        &self,
        num_layers: usize,
    ) -> AnyhowResult<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        let batch = self.per_row_totals.len();
        let mut out: Vec<Vec<Option<(Tensor, Tensor)>>> =
            (0..batch).map(|_| vec![None; num_layers]).collect();
        if self.max_total_len == 0 {
            return Ok(out);
        }
        for layer_idx in 0..num_layers {
            let Some((layer_k, layer_v)) = self.per_layer.get(layer_idx).and_then(|x| x.as_ref())
            else {
                continue;
            };
            for (row, total_opt) in self.per_row_totals.iter().enumerate() {
                let Some(total) = *total_opt else { continue };
                if total == 0 {
                    continue;
                }
                let offset = self.max_total_len - total;
                let row_k = layer_k
                    .narrow(0, row, 1)?
                    .narrow(2, offset, total)?
                    .contiguous()?;
                let row_v = layer_v
                    .narrow(0, row, 1)?
                    .narrow(2, offset, total)?
                    .contiguous()?;
                out[row][layer_idx] = Some((row_k, row_v));
            }
        }
        Ok(out)
    }
}

fn sequence_kv_cache_len(row: usize, caches: &[Option<(Tensor, Tensor)>]) -> AnyhowResult<usize> {
    let mut seen_len = None;
    for (layer, cache) in caches.iter().enumerate() {
        let Some((key, value)) = cache else {
            continue;
        };
        let (key_batch, _key_heads, key_len, _key_dim) = key
            .dims4()
            .with_context(|| format!("row {row} layer {layer} key cache is not 4D"))?;
        let (value_batch, _value_heads, value_len, _value_dim) = value
            .dims4()
            .with_context(|| format!("row {row} layer {layer} value cache is not 4D"))?;
        if key_batch != 1 || value_batch != 1 {
            bail!(
                "row {row} layer {layer} expected single-sequence KV cache, got key batch {key_batch}, value batch {value_batch}"
            );
        }
        if key_len != value_len {
            bail!(
                "row {row} layer {layer} key/value length mismatch: key={key_len}, value={value_len}"
            );
        }
        if let Some(prev_len) = seen_len {
            if prev_len != key_len {
                bail!(
                    "row {row} inconsistent KV lengths: previous={prev_len}, layer {layer}={key_len}"
                );
            }
        } else {
            seen_len = Some(key_len);
        }
    }
    Ok(seen_len.unwrap_or(0))
}

fn tensor_head_major_f32_values(
    tensor: &Tensor,
    layout: super::paged_kv::PagedKvLayout,
    seq_len: usize,
    row: usize,
    layer: usize,
    plane: PagedKvPlane,
) -> AnyhowResult<Vec<f32>> {
    let (batch, kv_heads, actual_len, head_dim) = tensor
        .dims4()
        .with_context(|| format!("row {row} layer {layer} {plane:?} cache is not 4D"))?;
    if batch != 1
        || kv_heads != layout.num_kv_heads
        || actual_len != seq_len
        || head_dim != layout.head_dim
    {
        bail!(
            "row {row} layer {layer} {plane:?} shape mismatch: got [{batch}, {kv_heads}, {actual_len}, {head_dim}], expected [1, {}, {seq_len}, {}]",
            layout.num_kv_heads,
            layout.head_dim
        );
    }

    let values = tensor
        .to_dtype(DType::F32)
        .with_context(|| format!("row {row} layer {layer} {plane:?} dtype conversion failed"))?
        .flatten_all()
        .with_context(|| format!("row {row} layer {layer} {plane:?} flatten failed"))?
        .to_vec1::<f32>()
        .with_context(|| format!("row {row} layer {layer} {plane:?} host copy failed"))?;
    let expected = seq_len * layout.num_kv_heads * layout.head_dim;
    if values.len() != expected {
        bail!(
            "row {row} layer {layer} {plane:?} flattened length mismatch: got {}, expected {expected}",
            values.len()
        );
    }
    Ok(values)
}

fn ensure_shadow_values_match(
    layer: usize,
    plane: PagedKvPlane,
    gathered: &[f32],
    direct: &[f32],
) -> AnyhowResult<usize> {
    if gathered.len() != direct.len() {
        bail!(
            "layer {layer} {plane:?} shadow length mismatch: gathered={}, direct={}",
            gathered.len(),
            direct.len()
        );
    }
    for (index, (&gathered_value, &direct_value)) in gathered.iter().zip(direct.iter()).enumerate()
    {
        if gathered_value.to_bits() != direct_value.to_bits() {
            bail!(
                "layer {layer} {plane:?} shadow mismatch at {index}: gathered={gathered_value}, direct={direct_value}"
            );
        }
    }
    Ok(gathered.len())
}

impl InferenceEngine {
    pub(super) fn refresh_paged_kv_stats(&self) {
        let allocator = self.paged_kv_allocator.snapshot();
        let live_tokens: u64 = self
            .sequences
            .values()
            .map(|seq| seq.paged_kv.token_len() as u64)
            .sum();
        let reserved_tokens: u64 = self
            .sequences
            .values()
            .map(|seq| seq.paged_kv.reserved_tokens() as u64)
            .sum();
        let fragment_tokens: u64 = self
            .sequences
            .values()
            .map(|seq| seq.paged_kv.fragmentation_tokens() as u64)
            .sum();

        self.stats
            .paged_kv_block_size
            .store(allocator.block_size as u64, Ordering::Relaxed);
        self.stats
            .paged_kv_live_pages
            .store(allocator.live_pages, Ordering::Relaxed);
        self.stats
            .paged_kv_free_pages
            .store(allocator.free_pages, Ordering::Relaxed);
        self.stats
            .paged_kv_total_alloc_pages
            .store(allocator.total_alloc_pages, Ordering::Relaxed);
        self.stats
            .paged_kv_total_reused_pages
            .store(allocator.total_reused_pages, Ordering::Relaxed);
        self.stats
            .paged_kv_total_freed_pages
            .store(allocator.total_freed_pages, Ordering::Relaxed);
        self.stats
            .paged_kv_live_tokens
            .store(live_tokens, Ordering::Relaxed);
        self.stats
            .paged_kv_reserved_tokens
            .store(reserved_tokens, Ordering::Relaxed);
        self.stats
            .paged_kv_fragment_tokens
            .store(fragment_tokens, Ordering::Relaxed);
        self.stats
            .paged_kv_reserved_bytes
            .store(allocator.reserved_bytes, Ordering::Relaxed);
        let (gpu_capacity_pages, gpu_capacity_bytes) = self
            .paged_kv_gpu_store
            .as_ref()
            .map(|store| (store.capacity_pages() as u64, store.capacity_bytes()))
            .unwrap_or((0, 0));
        self.stats
            .paged_kv_gpu_capacity_pages
            .store(gpu_capacity_pages, Ordering::Relaxed);
        self.stats
            .paged_kv_gpu_capacity_bytes
            .store(gpu_capacity_bytes, Ordering::Relaxed);
    }

    pub(super) fn sync_paged_kv_for_sequence(&mut self, seq_id: &str, token_len: usize) {
        let update = {
            let allocator = &mut self.paged_kv_allocator;
            self.sequences
                .get_mut(seq_id)
                .map(|seq| allocator.ensure_token_len(&mut seq.paged_kv, token_len))
        };

        if let Some(update) = update {
            self.stats
                .total_paged_kv_metadata_syncs
                .fetch_add(1, Ordering::Relaxed);
            self.stats
                .total_paged_kv_new_pages
                .fetch_add(update.allocated_pages, Ordering::Relaxed);
            self.stats
                .total_paged_kv_reused_pages
                .fetch_add(update.reused_pages, Ordering::Relaxed);
            self.refresh_paged_kv_stats();
        }
    }

    pub(super) fn release_paged_kv_for_sequence(&mut self, seq_id: &str) {
        let released = {
            let allocator = &mut self.paged_kv_allocator;
            self.sequences.get_mut(seq_id).map(|seq| {
                let page_ids = seq
                    .paged_kv
                    .blocks()
                    .iter()
                    .map(|block| block.id)
                    .collect::<Vec<_>>();
                let released = allocator.release_sequence(&mut seq.paged_kv);
                (released, page_ids)
            })
        };
        if let Some((released, page_ids)) = released {
            let no_live_pages = self.paged_kv_allocator.snapshot().live_pages == 0;
            if no_live_pages {
                if let Some(store) = self.paged_kv_gpu_store.as_mut() {
                    let released_capacity = store.release_cached_storage();
                    if released_capacity > 0 {
                        debug!(
                            released_capacity_pages = released_capacity,
                            "released idle GPU paged-KV storage"
                        );
                    }
                }
                let dropped_free_pages = self.paged_kv_allocator.reset_when_idle();
                if dropped_free_pages > 0 {
                    self.stats
                        .total_paged_kv_idle_resets
                        .fetch_add(1, Ordering::Relaxed);
                    self.stats
                        .total_paged_kv_idle_reset_pages
                        .fetch_add(dropped_free_pages, Ordering::Relaxed);
                    debug!(
                        dropped_free_pages,
                        "reset idle paged-KV allocator free list"
                    );
                }
            } else if let Some(store) = self.paged_kv_gpu_store.as_mut() {
                // Do not compact live paged-KV metadata here.  The GPU page
                // store is authoritative for rows whose per-sequence caches
                // have been dropped after batched extraction; compacting only
                // metadata and releasing storage would leave live sequences
                // marked GPU-resident while their page contents are gone.
                // Idle reset above is safe because there are no live rows.
                if let Err(err) = store.zero_pages(&page_ids) {
                    warn!(id = %seq_id, error = %err, "failed to zero released GPU paged-KV pages");
                }
            }
            self.stats
                .total_paged_kv_released_pages
                .fetch_add(released, Ordering::Relaxed);
            self.refresh_paged_kv_stats();
        }
    }

    fn skip_paged_kv_gpu_copy_for_pressure(&mut self, stage: &str) -> bool {
        if !self.paged_kv_native_append || self.paged_kv_gpu_store.is_none() {
            return false;
        }
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit == 0 || self.paged_kv_pressure_reserve_bytes == 0 {
            return false;
        }
        let (gpu_used, _gpu_total) = query_gpu_memory_usage(self.model.device());
        if gpu_used == 0 || gpu_used.saturating_add(self.paged_kv_pressure_reserve_bytes) <= limit {
            return false;
        }

        let released_capacity_pages = self
            .paged_kv_gpu_store
            .as_mut()
            .map(|store| store.release_cached_storage())
            .unwrap_or(0);
        for sequence in self.sequences.values_mut() {
            sequence.paged_kv.mark_gpu_resident(0);
        }
        self.stats
            .total_paged_kv_pressure_skips
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_paged_kv_pressure_released_pages
            .fetch_add(released_capacity_pages as u64, Ordering::Relaxed);
        self.refresh_paged_kv_stats();
        warn!(
            stage,
            gpu_used = %format_bytes_engine(gpu_used),
            limit = %format_bytes_engine(limit),
            reserve = %format_bytes_engine(self.paged_kv_pressure_reserve_bytes),
            released_capacity_pages,
            "skipping validation-only paged-KV GPU copy under memory pressure"
        );
        true
    }

    fn paged_kv_batch_past_is_resident(
        &self,
        batch: &[String],
        kv_lens: &[usize],
        keep: &[bool],
    ) -> bool {
        if batch.len() != kv_lens.len() || batch.len() != keep.len() {
            return false;
        }
        batch.iter().enumerate().all(|(row, seq_id)| {
            !keep[row]
                || self
                    .sequences
                    .get(seq_id)
                    .map(|seq| seq.paged_kv.gpu_resident_token_len() >= kv_lens[row])
                    .unwrap_or(false)
        })
    }

    fn paged_kv_batch_total_is_resident(
        &self,
        batch: &[String],
        kv_lens: &[usize],
        rounds_done: usize,
        keep: &[bool],
    ) -> bool {
        if batch.len() != kv_lens.len() || batch.len() != keep.len() {
            return false;
        }
        batch.iter().enumerate().all(|(row, seq_id)| {
            !keep[row]
                || self
                    .sequences
                    .get(seq_id)
                    .map(|seq| seq.paged_kv.gpu_resident_token_len() >= kv_lens[row] + rounds_done)
                    .unwrap_or(false)
        })
    }

    pub(super) fn should_attempt_paged_attention_for_round(
        &self,
        batch: &[String],
        kv_lens: &[usize],
        round: usize,
        keep: &[bool],
    ) -> bool {
        if !self.paged_kv_attention
            || !self.paged_kv_native_append
            || self.paged_kv_gpu_store.is_none()
        {
            return false;
        }
        if batch.len() != kv_lens.len() || batch.len() != keep.len() {
            return false;
        }

        let mut active_rows = 0usize;
        let mut max_past_len = 0usize;
        for (row, &is_live) in keep.iter().enumerate() {
            if !is_live {
                continue;
            }
            active_rows += 1;
            max_past_len = max_past_len.max(kv_lens[row].saturating_add(round));
        }

        active_rows >= self.paged_kv_attention_min_active_rows
            && max_past_len >= self.paged_kv_attention_min_seq_len
    }

    pub(super) fn maybe_build_paged_attention_context(
        &mut self,
        batch: &[String],
        kv_lens: &[usize],
        round: usize,
        keep: &[bool],
    ) -> Option<crane_core::fused_ops::PagedAttentionDecodeContext> {
        if !self.paged_kv_attention
            || !self.paged_kv_native_append
            || self.paged_kv_gpu_store.is_none()
        {
            return None;
        }
        if batch.len() != kv_lens.len() || batch.len() != keep.len() {
            self.stats
                .total_paged_kv_attention_fallbacks
                .fetch_add(1, Ordering::Relaxed);
            return None;
        }

        if !self.should_attempt_paged_attention_for_round(batch, kv_lens, round, keep) {
            return None;
        }

        let past_lens: Vec<usize> = kv_lens.iter().map(|&kv_len| kv_len + round).collect();
        if !self.paged_kv_batch_past_is_resident(batch, &past_lens, keep) {
            self.stats
                .total_paged_kv_attention_fallbacks
                .fetch_add(1, Ordering::Relaxed);
            return None;
        }

        let pages = self
            .paged_kv_gpu_store
            .as_ref()
            .and_then(|store| store.pages())?;
        let sequence_rows: Vec<Option<super::paged_kv::PagedKvSequence>> = batch
            .iter()
            .enumerate()
            .map(|(row, seq_id)| {
                if !keep.get(row).copied().unwrap_or(false) {
                    return None;
                }
                self.sequences.get(seq_id).and_then(|seq| {
                    (seq.paged_kv.token_len() == past_lens[row]).then(|| seq.paged_kv.clone())
                })
            })
            .collect();
        if sequence_rows
            .iter()
            .zip(keep.iter())
            .any(|(sequence, &is_live)| is_live && sequence.is_none())
        {
            self.stats
                .total_paged_kv_attention_fallbacks
                .fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let sequence_refs: Vec<Option<&super::paged_kv::PagedKvSequence>> = sequence_rows
            .iter()
            .map(|sequence| sequence.as_ref())
            .collect();
        let store = self.paged_kv_gpu_store.as_ref()?;
        let table = match PagedKvBatchPageTable::from_optional_sequences(
            store.block_size(),
            &sequence_refs,
        ) {
            Ok(table) => table,
            Err(err) => {
                warn!(error = %err, "failed to build paged attention page table");
                self.stats
                    .total_paged_kv_attention_fallbacks
                    .fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };
        self.stats
            .total_paged_kv_attention_contexts
            .fetch_add(1, Ordering::Relaxed);
        Some(crane_core::fused_ops::PagedAttentionDecodeContext {
            pages,
            indptr: table.indptr,
            indices: table.indices,
            last_page_lens: table.last_page_lens,
            seq_lens: table.seq_lens,
            block_size: store.block_size(),
            num_layers: store.layout().num_layers,
        })
    }

    pub(super) fn maybe_extract_paged_kv_gather(
        &mut self,
        batch: &[String],
        kv_lens: &[usize],
        rounds_done: usize,
        keep: &[bool],
    ) -> AnyhowResult<Option<BatchedKvExtract>> {
        if !self.paged_kv_gather_extract
            || !self.paged_kv_native_append
            || self.paged_kv_gpu_store.is_none()
            || rounds_done == 0
        {
            return Ok(None);
        }
        if !self.paged_kv_batch_total_is_resident(batch, kv_lens, rounds_done, keep) {
            return Ok(None);
        }

        let per_row_totals: Vec<Option<usize>> = kv_lens
            .iter()
            .zip(keep.iter())
            .map(|(&kv_len, &is_live)| {
                if is_live {
                    Some(kv_len + rounds_done)
                } else {
                    None
                }
            })
            .collect();
        let max_total_len = per_row_totals
            .iter()
            .filter_map(|t| t.as_ref().copied())
            .max()
            .unwrap_or(0);

        if max_total_len == 0 {
            self.stats
                .total_paged_kv_gather_extracts
                .fetch_add(1, Ordering::Relaxed);
            return Ok(Some(BatchedKvExtract {
                per_layer: vec![None; self.num_layers],
                max_total_len: 0,
                per_row_totals,
            }));
        }

        let sequence_rows: Vec<Option<super::paged_kv::PagedKvSequence>> = batch
            .iter()
            .enumerate()
            .map(|(row, seq_id)| {
                if keep.get(row).copied().unwrap_or(false) {
                    self.sequences.get(seq_id).map(|seq| seq.paged_kv.clone())
                } else {
                    None
                }
            })
            .collect();
        if sequence_rows
            .iter()
            .zip(keep.iter())
            .any(|(sequence, &is_live)| is_live && sequence.is_none())
        {
            return Ok(None);
        }
        let sequence_refs: Vec<Option<&super::paged_kv::PagedKvSequence>> = sequence_rows
            .iter()
            .map(|sequence| sequence.as_ref())
            .collect();

        let mut per_layer: Vec<Option<(Tensor, Tensor)>> = Vec::with_capacity(self.num_layers);
        let mut gathered_layers = 0u64;
        let mut gather_kernel_us: u64 = 0;
        let store = self
            .paged_kv_gpu_store
            .as_mut()
            .context("paged KV GPU store is not initialized")?;
        for layer in 0..self.num_layers {
            let t_gather = std::time::Instant::now();
            let key_batch = store
                .gather_layer_right_aligned(&sequence_refs, layer, PagedKvPlane::Key, max_total_len)
                .with_context(|| format!("gather paged KV layer {layer} keys"))?;
            let value_batch = store
                .gather_layer_right_aligned(
                    &sequence_refs,
                    layer,
                    PagedKvPlane::Value,
                    max_total_len,
                )
                .with_context(|| format!("gather paged KV layer {layer} values"))?;
            gather_kernel_us =
                gather_kernel_us.saturating_add(t_gather.elapsed().as_micros() as u64);
            gathered_layers += 1;
            per_layer.push(Some((key_batch, value_batch)));
        }

        self.stats
            .total_paged_kv_gather_extracts
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_paged_kv_gather_extract_layers
            .fetch_add(gathered_layers, Ordering::Relaxed);
        self.stats
            .total_paged_kv_gather_kernel_time_us
            .fetch_add(gather_kernel_us, Ordering::Relaxed);
        // per_row_us counter retained for compatibility but the new path skips the loop entirely.
        Ok(Some(BatchedKvExtract {
            per_layer,
            max_total_len,
            per_row_totals,
        }))
    }

    /// Re-gather a batched paged-KV extract for an arbitrary batch composition.
    ///
    /// Used by the next `setup_batch_decode` when the batch differs from the
    /// one published by the most recent `maybe_extract_paged_kv_gather`. Same
    /// per-layer batched form, no per-row materialization.
    pub(super) fn gather_batched_kv_for_batch(
        &mut self,
        batch: &[String],
    ) -> AnyhowResult<Option<BatchedKvExtract>> {
        if !self.paged_kv_gather_extract
            || !self.paged_kv_native_append
            || self.paged_kv_gpu_store.is_none()
        {
            return Ok(None);
        }

        let per_row_totals: Vec<Option<usize>> = batch
            .iter()
            .map(|seq_id| {
                self.sequences
                    .get(seq_id)
                    .map(|seq| seq.paged_kv.token_len())
            })
            .collect();
        // Require every requested seq to have its paged-KV resident.
        for (row, total) in per_row_totals.iter().enumerate() {
            let Some(total) = *total else { return Ok(None) };
            let Some(seq) = self.sequences.get(&batch[row]) else {
                return Ok(None);
            };
            if seq.paged_kv.gpu_resident_token_len() < total {
                return Ok(None);
            }
        }
        let max_total_len = per_row_totals
            .iter()
            .filter_map(|t| t.as_ref().copied())
            .max()
            .unwrap_or(0);
        if max_total_len == 0 {
            return Ok(Some(BatchedKvExtract {
                per_layer: vec![None; self.num_layers],
                max_total_len: 0,
                per_row_totals,
            }));
        }

        let sequence_rows: Vec<Option<super::paged_kv::PagedKvSequence>> = batch
            .iter()
            .map(|seq_id| self.sequences.get(seq_id).map(|seq| seq.paged_kv.clone()))
            .collect();
        let sequence_refs: Vec<Option<&super::paged_kv::PagedKvSequence>> =
            sequence_rows.iter().map(|s| s.as_ref()).collect();

        let mut per_layer = Vec::with_capacity(self.num_layers);
        let mut gathered_layers = 0u64;
        let mut gather_kernel_us: u64 = 0;
        let store = self
            .paged_kv_gpu_store
            .as_mut()
            .context("paged KV GPU store is not initialized")?;
        for layer in 0..self.num_layers {
            let t_gather = std::time::Instant::now();
            let key_batch = store
                .gather_layer_right_aligned(&sequence_refs, layer, PagedKvPlane::Key, max_total_len)
                .with_context(|| format!("regather paged KV layer {layer} keys"))?;
            let value_batch = store
                .gather_layer_right_aligned(
                    &sequence_refs,
                    layer,
                    PagedKvPlane::Value,
                    max_total_len,
                )
                .with_context(|| format!("regather paged KV layer {layer} values"))?;
            gather_kernel_us =
                gather_kernel_us.saturating_add(t_gather.elapsed().as_micros() as u64);
            gathered_layers += 1;
            per_layer.push(Some((key_batch, value_batch)));
        }

        self.stats
            .total_paged_kv_gather_regathers
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_paged_kv_gather_extract_layers
            .fetch_add(gathered_layers, Ordering::Relaxed);
        self.stats
            .total_paged_kv_gather_kernel_time_us
            .fetch_add(gather_kernel_us, Ordering::Relaxed);
        Ok(Some(BatchedKvExtract {
            per_layer,
            max_total_len,
            per_row_totals,
        }))
    }

    pub(super) fn materialize_paged_kv_rows_for_batch(
        &mut self,
        batch: &[String],
        rows: &[usize],
    ) -> AnyhowResult<bool> {
        if rows.is_empty() {
            return Ok(false);
        }

        let seq_ids: Vec<String> = rows
            .iter()
            .filter_map(|&row| batch.get(row).cloned())
            .collect();
        if seq_ids.is_empty() {
            return Ok(false);
        }

        let Some(extract) = self.gather_batched_kv_for_batch(&seq_ids)? else {
            return Ok(false);
        };
        let per_row = extract.materialize_per_row(self.num_layers)?;
        for (row, seq_id) in seq_ids.iter().enumerate() {
            if let Some(seq) = self.sequences.get_mut(seq_id) {
                if row < per_row.len() {
                    seq.kv_caches = per_row[row].clone();
                }
            }
        }
        Ok(true)
    }

    pub(super) fn maybe_append_paged_kv_native(
        &mut self,
        batch: &[String],
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
        keep: &[bool],
    ) -> bool {
        if !self.paged_kv_native_append || self.paged_kv_gpu_store.is_none() || rounds_done == 0 {
            return false;
        }
        if self.skip_paged_kv_gpu_copy_for_pressure("append") {
            return false;
        }
        if !self.paged_kv_batch_past_is_resident(batch, kv_lens, keep) {
            debug!("skipping paged KV native append because batch past K/V is not GPU-resident");
            return false;
        }

        let started = Instant::now();
        match self.append_paged_kv_native(batch, kv_lens, original_max_kv, rounds_done, keep) {
            Ok(report) => {
                if self.profile_enabled {
                    info!(
                        target: "crane_profile",
                        layers = report.layers,
                        entries = report.entries,
                        capacity_pages = report.capacity_pages,
                        elapsed_us = started.elapsed().as_micros() as u64,
                        "profile paged_kv_native_append",
                    );
                } else {
                    debug!(
                        layers = report.layers,
                        entries = report.entries,
                        capacity_pages = report.capacity_pages,
                        elapsed_us = started.elapsed().as_micros() as u64,
                        "paged KV native append completed"
                    );
                }
                true
            }
            Err(err) => {
                warn!(
                    error = %err,
                    elapsed_us = started.elapsed().as_micros() as u64,
                    "paged KV native append failed; keeping contiguous extraction fallback"
                );
                false
            }
        }
    }

    fn append_paged_kv_native(
        &mut self,
        batch: &[String],
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
        keep: &[bool],
    ) -> AnyhowResult<PagedKvNativeAppendReport> {
        if batch.len() != kv_lens.len() || batch.len() != keep.len() {
            bail!(
                "paged KV native append metadata mismatch: batch={} kv_lens={} keep={}",
                batch.len(),
                kv_lens.len(),
                keep.len()
            );
        }

        for (row, seq_id) in batch.iter().enumerate() {
            if keep[row] {
                self.sync_paged_kv_for_sequence(seq_id, kv_lens[row] + rounds_done);
            }
        }

        let plan = self.build_paged_kv_native_append_plan(
            batch,
            kv_lens,
            original_max_kv,
            rounds_done,
            keep,
        )?;
        if plan.is_empty() {
            return Ok(PagedKvNativeAppendReport::default());
        }

        let buffers = self.model.get_kv_cache_buffers();
        if buffers.len() != self.num_layers {
            bail!(
                "paged KV native append expected {} layers, model returned {}",
                self.num_layers,
                buffers.len()
            );
        }

        let store = self
            .paged_kv_gpu_store
            .as_mut()
            .context("paged KV GPU store is not initialized")?;
        let layers = store
            .copy_layers_from_cache_buffers(&buffers, &plan)
            .context("append generated K/V into paged KV GPU store")?;
        let capacity_pages = store.capacity_pages();

        for (row, seq_id) in batch.iter().enumerate() {
            if keep[row] {
                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    seq.paged_kv.mark_gpu_resident(kv_lens[row] + rounds_done);
                }
            }
        }
        self.refresh_paged_kv_stats();

        Ok(PagedKvNativeAppendReport {
            layers,
            entries: plan.entries(),
            capacity_pages,
        })
    }

    fn build_paged_kv_native_append_plan(
        &self,
        batch: &[String],
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
        keep: &[bool],
    ) -> AnyhowResult<PagedKvNativeAppendPlan> {
        let mut plan = PagedKvNativeAppendPlan::default();
        for (row, seq_id) in batch.iter().enumerate() {
            if !keep[row] {
                continue;
            }
            let seq = self
                .sequences
                .get(seq_id)
                .with_context(|| format!("missing sequence {seq_id} for paged KV append"))?;
            for round in 0..rounds_done {
                let token_index = kv_lens[row] + round;
                let (page_id, token_offset) = seq.paged_kv.page_slot(token_index)?;
                plan.push(page_id, token_offset, row, original_max_kv + round)?;
            }
        }
        Ok(plan)
    }

    pub(super) fn maybe_import_paged_kv_batch_past(
        &mut self,
        batch: &[String],
        kv_lens: &[usize],
        original_max_kv: usize,
    ) -> bool {
        if !self.paged_kv_native_append || self.paged_kv_gpu_store.is_none() {
            return false;
        }
        if self.skip_paged_kv_gpu_copy_for_pressure("import") {
            return false;
        }

        let started = Instant::now();
        match self.import_paged_kv_batch_past(batch, kv_lens, original_max_kv) {
            Ok(report) if report.entries > 0 => {
                if self.profile_enabled {
                    info!(
                        target: "crane_profile",
                        layers = report.layers,
                        entries = report.entries,
                        capacity_pages = report.capacity_pages,
                        elapsed_us = started.elapsed().as_micros() as u64,
                        "profile paged_kv_past_import",
                    );
                } else {
                    debug!(
                        layers = report.layers,
                        entries = report.entries,
                        capacity_pages = report.capacity_pages,
                        elapsed_us = started.elapsed().as_micros() as u64,
                        "paged KV past import completed"
                    );
                }
                true
            }
            Ok(_) => true,
            Err(err) => {
                warn!(
                    error = %err,
                    elapsed_us = started.elapsed().as_micros() as u64,
                    "paged KV past import failed; keeping contiguous fallback authoritative"
                );
                false
            }
        }
    }

    fn import_paged_kv_batch_past(
        &mut self,
        batch: &[String],
        kv_lens: &[usize],
        original_max_kv: usize,
    ) -> AnyhowResult<PagedKvNativeAppendReport> {
        if batch.len() != kv_lens.len() {
            bail!(
                "paged KV past import metadata mismatch: batch={} kv_lens={}",
                batch.len(),
                kv_lens.len()
            );
        }

        for (row, seq_id) in batch.iter().enumerate() {
            self.sync_paged_kv_for_sequence(seq_id, kv_lens[row]);
        }

        let plan = self.build_paged_kv_batch_past_import_plan(batch, kv_lens, original_max_kv)?;
        if plan.is_empty() {
            return Ok(PagedKvNativeAppendReport::default());
        }

        let buffers = self.model.get_kv_cache_buffers();
        if buffers.len() != self.num_layers {
            bail!(
                "paged KV past import expected {} layers, model returned {}",
                self.num_layers,
                buffers.len()
            );
        }

        let store = self
            .paged_kv_gpu_store
            .as_mut()
            .context("paged KV GPU store is not initialized")?;
        let layers = store
            .copy_layers_from_cache_buffers(&buffers, &plan)
            .context("import existing K/V into paged KV GPU store")?;
        let capacity_pages = store.capacity_pages();

        for (row, seq_id) in batch.iter().enumerate() {
            if let Some(seq) = self.sequences.get_mut(seq_id) {
                seq.paged_kv.mark_gpu_resident(kv_lens[row]);
            }
        }
        self.refresh_paged_kv_stats();

        Ok(PagedKvNativeAppendReport {
            layers,
            entries: plan.entries(),
            capacity_pages,
        })
    }

    fn build_paged_kv_batch_past_import_plan(
        &self,
        batch: &[String],
        kv_lens: &[usize],
        original_max_kv: usize,
    ) -> AnyhowResult<PagedKvNativeAppendPlan> {
        let mut plan = PagedKvNativeAppendPlan::default();
        for (row, seq_id) in batch.iter().enumerate() {
            let seq = self
                .sequences
                .get(seq_id)
                .with_context(|| format!("missing sequence {seq_id} for paged KV import"))?;
            let kv_len = kv_lens[row];
            if kv_len > original_max_kv {
                bail!("row {row} kv length {kv_len} exceeds batch source width {original_max_kv}");
            }
            let resident_len = seq.paged_kv.gpu_resident_token_len().min(kv_len);
            let source_start = original_max_kv - kv_len;
            for token_index in resident_len..kv_len {
                let (page_id, token_offset) = seq.paged_kv.page_slot(token_index)?;
                plan.push(page_id, token_offset, row, source_start + token_index)?;
            }
        }
        Ok(plan)
    }

    pub(super) fn maybe_validate_paged_kv_shadow_gather(
        &self,
        batch: &[String],
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
    ) {
        if !self.paged_kv_shadow_validate {
            return;
        }

        let started = Instant::now();
        match self.validate_paged_kv_shadow_gather(batch, seq_kv_caches) {
            Ok(report) => info!(
                layers = report.layers,
                sequences = report.sequences,
                values_compared = report.values_compared,
                elapsed_us = started.elapsed().as_micros() as u64,
                "paged KV shadow gather matched direct packing"
            ),
            Err(err) => warn!(
                error = %err,
                elapsed_us = started.elapsed().as_micros() as u64,
                "paged KV shadow gather validation failed"
            ),
        }
    }

    fn validate_paged_kv_shadow_gather(
        &self,
        batch: &[String],
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
    ) -> AnyhowResult<PagedKvShadowGatherReport> {
        if batch.len() != seq_kv_caches.len() {
            bail!(
                "batch/cache count mismatch: batch={}, caches={}",
                batch.len(),
                seq_kv_caches.len()
            );
        }

        let layout = self.paged_kv_allocator.layout();
        if layout.num_layers == 0 || layout.num_kv_heads == 0 || layout.head_dim == 0 {
            bail!("paged KV layout is empty: {:?}", layout);
        }

        let sequences: Vec<&Sequence> = batch
            .iter()
            .map(|seq_id| {
                self.sequences
                    .get(seq_id)
                    .with_context(|| format!("missing sequence {seq_id} for paged KV shadow"))
            })
            .collect::<AnyhowResult<_>>()?;
        let page_sequences: Vec<&super::paged_kv::PagedKvSequence> =
            sequences.iter().map(|seq| &seq.paged_kv).collect();
        let kv_lens: Vec<usize> = seq_kv_caches
            .iter()
            .enumerate()
            .map(|(row, caches)| sequence_kv_cache_len(row, caches))
            .collect::<AnyhowResult<_>>()?;
        let max_len = kv_lens.iter().copied().max().unwrap_or(0);
        let layer_limit = self
            .paged_kv_shadow_max_layers
            .min(layout.num_layers)
            .min(self.num_layers);
        if max_len == 0 || layer_limit == 0 {
            return Ok(PagedKvShadowGatherReport {
                layers: 0,
                sequences: batch.len(),
                values_compared: 0,
            });
        }

        for (row, (sequence, &kv_len)) in sequences.iter().zip(kv_lens.iter()).enumerate() {
            if sequence.paged_kv.token_len() != kv_len {
                bail!(
                    "row {row} paged token length mismatch: paged={}, cache={kv_len}",
                    sequence.paged_kv.token_len()
                );
            }
        }

        let mut values_compared = 0usize;
        for layer in 0..layer_limit {
            let mut key_rows = Vec::with_capacity(seq_kv_caches.len());
            let mut value_rows = Vec::with_capacity(seq_kv_caches.len());
            for (row, caches) in seq_kv_caches.iter().enumerate() {
                match caches.get(layer).and_then(|cache| cache.as_ref()) {
                    Some((key, value)) => {
                        key_rows.push(tensor_head_major_f32_values(
                            key,
                            layout,
                            kv_lens[row],
                            row,
                            layer,
                            PagedKvPlane::Key,
                        )?);
                        value_rows.push(tensor_head_major_f32_values(
                            value,
                            layout,
                            kv_lens[row],
                            row,
                            layer,
                            PagedKvPlane::Value,
                        )?);
                    }
                    None if kv_lens[row] == 0 => {
                        key_rows.push(Vec::new());
                        value_rows.push(Vec::new());
                    }
                    None => bail!(
                        "row {row} has cache length {} but missing layer {layer}",
                        kv_lens[row]
                    ),
                }
            }

            let gathered_key = gather_head_major_layer_via_pages(
                self.paged_kv_allocator.block_size(),
                layout,
                &page_sequences,
                layer,
                PagedKvPlane::Key,
                &key_rows,
                max_len,
            )?;
            let direct_key =
                build_right_aligned_head_major_batch(layout, &kv_lens, max_len, &key_rows)?;
            values_compared +=
                ensure_shadow_values_match(layer, PagedKvPlane::Key, &gathered_key, &direct_key)?;

            let gathered_value = gather_head_major_layer_via_pages(
                self.paged_kv_allocator.block_size(),
                layout,
                &page_sequences,
                layer,
                PagedKvPlane::Value,
                &value_rows,
                max_len,
            )?;
            let direct_value =
                build_right_aligned_head_major_batch(layout, &kv_lens, max_len, &value_rows)?;
            values_compared += ensure_shadow_values_match(
                layer,
                PagedKvPlane::Value,
                &gathered_value,
                &direct_value,
            )?;
        }

        Ok(PagedKvShadowGatherReport {
            layers: layer_limit,
            sequences: batch.len(),
            values_compared,
        })
    }
}
