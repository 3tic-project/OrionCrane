use super::*;

fn layout() -> PagedKvLayout {
    PagedKvLayout {
        num_layers: 2,
        num_kv_heads: 4,
        head_dim: 8,
        dtype_size_bytes: 2,
    }
}

#[test]
fn bytes_per_block_counts_k_and_v_for_all_layers() {
    assert_eq!(layout().bytes_per_block(16), 2 * 2 * 4 * 16 * 8 * 2);
}

#[test]
fn ensure_token_len_allocates_pages_and_fragmentation() {
    let mut allocator = PagedKvAllocator::new(16, layout());
    let mut seq = PagedKvSequence::new(16);

    let update = allocator.ensure_token_len(&mut seq, 33);

    assert_eq!(update.allocated_pages, 3);
    assert_eq!(update.reused_pages, 0);
    assert_eq!(seq.token_len(), 33);
    assert_eq!(seq.reserved_tokens(), 48);
    assert_eq!(seq.fragmentation_tokens(), 15);
    assert_eq!(seq.blocks()[0].start_token, 0);
    assert_eq!(seq.blocks()[1].start_token, 16);
    assert_eq!(seq.blocks()[2].len, 1);
    assert_eq!(allocator.snapshot().live_pages, 3);
}

#[test]
fn released_pages_are_reused() {
    let mut allocator = PagedKvAllocator::new(8, layout());
    let mut first = PagedKvSequence::new(8);
    let mut second = PagedKvSequence::new(8);

    allocator.ensure_token_len(&mut first, 9);
    assert_eq!(allocator.release_sequence(&mut first), 2);
    let update = allocator.ensure_token_len(&mut second, 8);

    assert_eq!(update.allocated_pages, 0);
    assert_eq!(update.reused_pages, 1);
    assert_eq!(allocator.snapshot().free_pages, 1);
    assert_eq!(allocator.snapshot().live_pages, 1);
}

#[test]
fn page_slot_maps_token_to_page_and_offset() {
    let mut allocator = PagedKvAllocator::new(8, layout());
    let mut seq = PagedKvSequence::new(8);

    allocator.ensure_token_len(&mut seq, 17);

    assert_eq!(seq.page_slot(0).unwrap(), (1, 0));
    assert_eq!(seq.page_slot(7).unwrap(), (1, 7));
    assert_eq!(seq.page_slot(8).unwrap(), (2, 0));
    assert_eq!(seq.page_slot(16).unwrap(), (3, 0));
    assert!(matches!(
        seq.page_slot(17),
        Err(PagedKvStorageError::InvalidTokenIndex { .. })
    ));
}

#[test]
fn release_sequence_immediately_detaches_all_pages() {
    let mut allocator = PagedKvAllocator::new(8, layout());
    let mut seq = PagedKvSequence::new(8);

    allocator.ensure_token_len(&mut seq, 17);
    seq.mark_gpu_resident(17);
    let released = allocator.release_sequence(&mut seq);

    assert_eq!(released, 3);
    assert_eq!(seq.token_len(), 0);
    assert_eq!(seq.gpu_resident_token_len(), 0);
    assert!(seq.blocks().is_empty());
    assert_eq!(allocator.snapshot().live_pages, 0);
    assert_eq!(allocator.snapshot().free_pages, 3);
}

#[test]
fn gpu_resident_len_is_clamped_when_token_len_shrinks() {
    let mut allocator = PagedKvAllocator::new(8, layout());
    let mut seq = PagedKvSequence::new(8);

    allocator.ensure_token_len(&mut seq, 17);
    seq.mark_gpu_resident(17);
    allocator.ensure_token_len(&mut seq, 9);

    assert_eq!(seq.token_len(), 9);
    assert_eq!(seq.gpu_resident_token_len(), 9);
}

#[test]
fn native_copy_plan_records_absolute_source_tokens() {
    let mut plan = PagedKvNativeAppendPlan::default();

    plan.push(3, 5, 2, 19).unwrap();
    plan.push(4, 0, 1, 20).unwrap();

    assert_eq!(plan.entries(), 2);
    assert_eq!(plan.page_ids, vec![3, 4]);
    assert_eq!(plan.token_offsets, vec![5, 0]);
    assert_eq!(plan.row_indices, vec![2, 1]);
    assert_eq!(plan.source_token_indices, vec![19, 20]);
}

#[test]
fn gather_plan_records_right_aligned_targets() {
    let mut allocator = PagedKvAllocator::new(4, layout());
    let mut full = PagedKvSequence::new(4);
    let mut short = PagedKvSequence::new(4);
    allocator.ensure_token_len(&mut full, 5);
    allocator.ensure_token_len(&mut short, 2);

    let plan = PagedKvGatherPlan::from_optional_sequences(4, &[Some(&full), None, Some(&short)], 5)
        .unwrap();

    assert_eq!(plan.entries(), 7);
    assert_eq!(plan.row_indices, vec![0, 0, 0, 0, 0, 2, 2]);
    assert_eq!(plan.target_token_indices, vec![0, 1, 2, 3, 4, 3, 4]);
    assert_eq!(plan.token_offsets, vec![0, 1, 2, 3, 0, 0, 1]);
    assert_eq!(plan.page_ids, vec![1, 1, 1, 1, 2, 3, 3]);
}

#[test]
fn batch_page_table_matches_flashinfer_style_metadata() {
    let mut allocator = PagedKvAllocator::new(4, layout());
    let mut first = PagedKvSequence::new(4);
    let mut second = PagedKvSequence::new(4);
    let empty = PagedKvSequence::new(4);
    allocator.ensure_token_len(&mut first, 5);
    allocator.ensure_token_len(&mut second, 8);

    let table = PagedKvBatchPageTable::from_sequences(4, &[&first, &empty, &second]).unwrap();

    assert_eq!(table.indptr, vec![0, 2, 2, 4]);
    assert_eq!(table.indices, vec![1, 2, 3, 4]);
    assert_eq!(table.last_page_lens, vec![1, 0, 4]);
    assert_eq!(table.seq_lens, vec![5, 0, 8]);
}

#[test]
fn optional_batch_page_table_keeps_inactive_rows_empty() {
    let mut allocator = PagedKvAllocator::new(4, layout());
    let mut first = PagedKvSequence::new(4);
    let mut second = PagedKvSequence::new(4);
    allocator.ensure_token_len(&mut first, 5);
    allocator.ensure_token_len(&mut second, 1);

    let table =
        PagedKvBatchPageTable::from_optional_sequences(4, &[Some(&first), None, Some(&second)])
            .unwrap();

    assert_eq!(table.indptr, vec![0, 2, 2, 3]);
    assert_eq!(table.indices, vec![1, 2, 3]);
    assert_eq!(table.last_page_lens, vec![1, 0, 1]);
    assert_eq!(table.seq_lens, vec![5, 0, 1]);
}

fn head_major_values(layout: PagedKvLayout, seq_len: usize, offset: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(layout.num_kv_heads * seq_len * layout.head_dim);
    for head in 0..layout.num_kv_heads {
        for token in 0..seq_len {
            for dim in 0..layout.head_dim {
                values.push(offset + head as f32 * 0.13 + token as f32 * 0.07 + dim as f32 * 0.03);
            }
        }
    }
    values
}

fn contiguous_decode_attention(
    layout: PagedKvLayout,
    num_heads: usize,
    past_k: &[Vec<f32>],
    past_v: &[Vec<f32>],
    current_k: &[f32],
    current_v: &[f32],
    query: &[f32],
    seq_lens: &[usize],
) -> Vec<f32> {
    let n_rep = num_heads / layout.num_kv_heads;
    let scale = 1.0f32 / (layout.head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_lens.len() * num_heads * layout.head_dim];
    for (row, &seq_len) in seq_lens.iter().enumerate() {
        for head in 0..num_heads {
            let kv_head = head / n_rep;
            let q_base = (row * num_heads + head) * layout.head_dim;
            let mut scores = Vec::with_capacity(seq_len + 1);
            for token in 0..=seq_len {
                let mut dot = 0.0f32;
                for dim in 0..layout.head_dim {
                    let key = if token == seq_len {
                        current_k[(row * layout.num_kv_heads + kv_head) * layout.head_dim + dim]
                    } else {
                        past_k[row]
                            [kv_head * seq_len * layout.head_dim + token * layout.head_dim + dim]
                    };
                    dot += query[q_base + dim] * key;
                }
                scores.push(dot * scale);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denom: f32 = scores.iter().map(|score| (*score - max_score).exp()).sum();
            for token in 0..=seq_len {
                let weight = (scores[token] - max_score).exp() / denom;
                for dim in 0..layout.head_dim {
                    let value = if token == seq_len {
                        current_v[(row * layout.num_kv_heads + kv_head) * layout.head_dim + dim]
                    } else {
                        past_v[row]
                            [kv_head * seq_len * layout.head_dim + token * layout.head_dim + dim]
                    };
                    output[(row * num_heads + head) * layout.head_dim + dim] += weight * value;
                }
            }
        }
    }
    output
}

fn paged_decode_attention_reference(
    store: &PagedKvPageStore<f32>,
    sequences: &[&PagedKvSequence],
    layer: usize,
    layout: PagedKvLayout,
    num_heads: usize,
    current_k: &[f32],
    current_v: &[f32],
    query: &[f32],
) -> Vec<f32> {
    let n_rep = num_heads / layout.num_kv_heads;
    let scale = 1.0f32 / (layout.head_dim as f32).sqrt();
    let mut output = vec![0.0f32; sequences.len() * num_heads * layout.head_dim];
    for (row, sequence) in sequences.iter().enumerate() {
        for head in 0..num_heads {
            let kv_head = head / n_rep;
            let q_base = (row * num_heads + head) * layout.head_dim;
            let mut max_score = f32::NEG_INFINITY;
            let mut denom = 0.0f32;
            let mut acc = vec![0.0f32; layout.head_dim];
            for token in 0..=sequence.token_len() {
                let mut dot = 0.0f32;
                let past_key = if token == sequence.token_len() {
                    None
                } else {
                    Some(
                        store
                            .read_token(sequence, token, layer, PagedKvPlane::Key)
                            .unwrap(),
                    )
                };
                let past_value = if token == sequence.token_len() {
                    None
                } else {
                    Some(
                        store
                            .read_token(sequence, token, layer, PagedKvPlane::Value)
                            .unwrap(),
                    )
                };
                for dim in 0..layout.head_dim {
                    let key = if let Some(key) = past_key.as_ref() {
                        key[kv_head * layout.head_dim + dim]
                    } else {
                        current_k[(row * layout.num_kv_heads + kv_head) * layout.head_dim + dim]
                    };
                    dot += query[q_base + dim] * key;
                }
                let score = dot * scale;
                let next_max = max_score.max(score);
                let alpha = (max_score - next_max).exp();
                let beta = (score - next_max).exp();
                denom = denom * alpha + beta;
                for dim in 0..layout.head_dim {
                    let value = if let Some(value) = past_value.as_ref() {
                        value[kv_head * layout.head_dim + dim]
                    } else {
                        current_v[(row * layout.num_kv_heads + kv_head) * layout.head_dim + dim]
                    };
                    acc[dim] = acc[dim] * alpha + beta * value;
                }
                max_score = next_max;
            }
            for dim in 0..layout.head_dim {
                output[(row * num_heads + head) * layout.head_dim + dim] = acc[dim] / denom;
            }
        }
    }
    output
}

#[test]
fn paged_attention_reference_matches_contiguous_gqa_sdpa() {
    let layout = PagedKvLayout {
        num_layers: 2,
        num_kv_heads: 2,
        head_dim: 3,
        dtype_size_bytes: 4,
    };
    let block_size = 3;
    let num_heads = 4;
    let mut allocator = PagedKvAllocator::new(block_size, layout);
    let mut first = PagedKvSequence::new(block_size);
    let mut second = PagedKvSequence::new(block_size);
    allocator.ensure_token_len(&mut first, 5);
    allocator.ensure_token_len(&mut second, 2);

    let past_k = vec![
        head_major_values(layout, first.token_len(), 0.1),
        head_major_values(layout, second.token_len(), 0.4),
    ];
    let past_v = vec![
        head_major_values(layout, first.token_len(), 1.1),
        head_major_values(layout, second.token_len(), 1.4),
    ];
    let mut store = PagedKvPageStore::<f32>::new(block_size, layout);
    store
        .write_sequence_layer_kv_head_major(&first, 1, &past_k[0], &past_v[0])
        .unwrap();
    store
        .write_sequence_layer_kv_head_major(&second, 1, &past_k[1], &past_v[1])
        .unwrap();

    let rows = 2;
    let current_k: Vec<f32> = (0..rows * layout.num_kv_heads * layout.head_dim)
        .map(|idx| 0.2 + idx as f32 * 0.017)
        .collect();
    let current_v: Vec<f32> = (0..rows * layout.num_kv_heads * layout.head_dim)
        .map(|idx| 1.2 + idx as f32 * 0.019)
        .collect();
    let query: Vec<f32> = (0..rows * num_heads * layout.head_dim)
        .map(|idx| 0.3 + idx as f32 * 0.011)
        .collect();

    let contiguous = contiguous_decode_attention(
        layout,
        num_heads,
        &past_k,
        &past_v,
        &current_k,
        &current_v,
        &query,
        &[first.token_len(), second.token_len()],
    );
    let paged = paged_decode_attention_reference(
        &store,
        &[&first, &second],
        1,
        layout,
        num_heads,
        &current_k,
        &current_v,
        &query,
    );

    for (lhs, rhs) in paged.iter().zip(contiguous.iter()) {
        assert!((lhs - rhs).abs() < 1.0e-5, "paged={lhs} contiguous={rhs}");
    }
}

#[test]
fn reset_when_idle_drops_free_high_water_page_ids() {
    let mut allocator = PagedKvAllocator::new(8, layout());
    let mut first = PagedKvSequence::new(8);
    let mut second = PagedKvSequence::new(8);

    allocator.ensure_token_len(&mut first, 17);
    assert_eq!(allocator.release_sequence(&mut first), 3);
    assert_eq!(allocator.reset_when_idle(), 3);

    let update = allocator.ensure_token_len(&mut second, 1);

    assert_eq!(update.allocated_pages, 1);
    assert_eq!(update.reused_pages, 0);
    assert_eq!(second.blocks()[0].id, 1);
    assert_eq!(allocator.snapshot().free_pages, 0);
}

#[test]
fn compact_sequences_rewrites_live_page_ids_and_clears_residency() {
    let mut allocator = PagedKvAllocator::new(8, layout());
    let mut released = PagedKvSequence::new(8);
    let mut live = PagedKvSequence::new(8);
    let mut next = PagedKvSequence::new(8);

    allocator.ensure_token_len(&mut released, 17);
    allocator.ensure_token_len(&mut live, 9);
    live.mark_gpu_resident(9);
    assert_eq!(allocator.release_sequence(&mut released), 3);
    assert_eq!(live.blocks()[0].id, 4);
    assert_eq!(live.blocks()[1].id, 5);

    let report = allocator.compact_sequences(std::iter::once(&mut live));

    assert_eq!(report.live_pages, 2);
    assert_eq!(report.moved_pages, 2);
    assert_eq!(report.dropped_free_pages, 3);
    assert_eq!(live.blocks()[0].id, 1);
    assert_eq!(live.blocks()[1].id, 2);
    assert_eq!(live.gpu_resident_token_len(), 0);
    assert_eq!(allocator.snapshot().free_pages, 0);

    let update = allocator.ensure_token_len(&mut next, 1);
    assert_eq!(update.allocated_pages, 1);
    assert_eq!(next.blocks()[0].id, 3);
}

fn storage_layout() -> PagedKvLayout {
    PagedKvLayout {
        num_layers: 2,
        num_kv_heads: 2,
        head_dim: 3,
        dtype_size_bytes: 2,
    }
}

#[test]
fn page_store_round_trips_tokens_across_block_boundary() {
    let mut allocator = PagedKvAllocator::new(4, storage_layout());
    let mut sequence = PagedKvSequence::new(4);
    allocator.ensure_token_len(&mut sequence, 5);

    let mut store = PagedKvPageStore::<u32>::new(4, storage_layout());
    store
        .write_token(
            &sequence,
            3,
            1,
            PagedKvPlane::Key,
            &[30, 31, 32, 33, 34, 35],
        )
        .unwrap();
    store
        .write_token(
            &sequence,
            4,
            1,
            PagedKvPlane::Value,
            &[40, 41, 42, 43, 44, 45],
        )
        .unwrap();

    assert_eq!(
        store
            .read_token(&sequence, 3, 1, PagedKvPlane::Key)
            .unwrap(),
        vec![30, 31, 32, 33, 34, 35]
    );
    assert_eq!(
        store
            .read_token(&sequence, 4, 1, PagedKvPlane::Value)
            .unwrap(),
        vec![40, 41, 42, 43, 44, 45]
    );
}

#[test]
fn page_store_gathers_right_aligned_batch_layout() {
    let layout = PagedKvLayout {
        num_layers: 1,
        num_kv_heads: 2,
        head_dim: 2,
        dtype_size_bytes: 2,
    };
    let mut allocator = PagedKvAllocator::new(2, layout);
    let mut short = PagedKvSequence::new(2);
    let mut full = PagedKvSequence::new(2);
    let empty = PagedKvSequence::new(2);
    allocator.ensure_token_len(&mut short, 3);
    allocator.ensure_token_len(&mut full, 5);

    let mut store = PagedKvPageStore::<u32>::new(2, layout);
    for token_index in 0..short.token_len() {
        let base = 10 + token_index as u32 * 10;
        store
            .write_token(
                &short,
                token_index,
                0,
                PagedKvPlane::Key,
                &[base, base + 1, base + 2, base + 3],
            )
            .unwrap();
    }
    for token_index in 0..full.token_len() {
        let base = 100 + token_index as u32 * 10;
        store
            .write_token(
                &full,
                token_index,
                0,
                PagedKvPlane::Key,
                &[base, base + 1, base + 2, base + 3],
            )
            .unwrap();
    }

    let gathered = store
        .gather_layer_right_aligned(&[&short, &full, &empty], 0, 5, PagedKvPlane::Key)
        .unwrap();

    let mut expected = vec![0u32; 3 * 2 * 5 * 2];
    for token_index in 0..short.token_len() {
        let base = 10 + token_index as u32 * 10;
        put_expected_token(
            &mut expected,
            0,
            2 + token_index,
            5,
            &[base, base + 1, base + 2, base + 3],
        );
    }
    for token_index in 0..full.token_len() {
        let base = 100 + token_index as u32 * 10;
        put_expected_token(
            &mut expected,
            1,
            token_index,
            5,
            &[base, base + 1, base + 2, base + 3],
        );
    }
    assert_eq!(gathered, expected);
}

#[test]
fn page_store_imports_head_major_layer_layout() {
    let layout = PagedKvLayout {
        num_layers: 1,
        num_kv_heads: 2,
        head_dim: 2,
        dtype_size_bytes: 2,
    };
    let mut allocator = PagedKvAllocator::new(2, layout);
    let mut short = PagedKvSequence::new(2);
    let mut full = PagedKvSequence::new(2);
    allocator.ensure_token_len(&mut short, 3);
    allocator.ensure_token_len(&mut full, 4);

    let mut store = PagedKvPageStore::<u32>::new(2, layout);
    store
        .write_sequence_layer_kv_head_major(
            &short,
            0,
            &[10, 11, 20, 21, 30, 31, 12, 13, 22, 23, 32, 33],
            &[110, 111, 120, 121, 130, 131, 112, 113, 122, 123, 132, 133],
        )
        .unwrap();
    store
        .write_sequence_layer_head_major(
            &full,
            0,
            PagedKvPlane::Key,
            &[
                200, 201, 210, 211, 220, 221, 230, 231, 202, 203, 212, 213, 222, 223, 232, 233,
            ],
        )
        .unwrap();

    let gathered_k = store
        .gather_layer_right_aligned(&[&short, &full], 0, 4, PagedKvPlane::Key)
        .unwrap();
    let gathered_v = store
        .gather_layer_right_aligned(&[&short], 0, 4, PagedKvPlane::Value)
        .unwrap();

    let mut expected_k = vec![0u32; 2 * 2 * 4 * 2];
    put_expected_token(&mut expected_k, 0, 1, 4, &[10, 11, 12, 13]);
    put_expected_token(&mut expected_k, 0, 2, 4, &[20, 21, 22, 23]);
    put_expected_token(&mut expected_k, 0, 3, 4, &[30, 31, 32, 33]);
    put_expected_token(&mut expected_k, 1, 0, 4, &[200, 201, 202, 203]);
    put_expected_token(&mut expected_k, 1, 1, 4, &[210, 211, 212, 213]);
    put_expected_token(&mut expected_k, 1, 2, 4, &[220, 221, 222, 223]);
    put_expected_token(&mut expected_k, 1, 3, 4, &[230, 231, 232, 233]);

    let mut expected_v = vec![0u32; 2 * 4 * 2];
    put_expected_token(&mut expected_v, 0, 1, 4, &[110, 111, 112, 113]);
    put_expected_token(&mut expected_v, 0, 2, 4, &[120, 121, 122, 123]);
    put_expected_token(&mut expected_v, 0, 3, 4, &[130, 131, 132, 133]);

    assert_eq!(gathered_k, expected_k);
    assert_eq!(gathered_v, expected_v);
}

#[test]
fn page_store_head_major_import_rejects_wrong_length() {
    let layout = PagedKvLayout {
        num_layers: 1,
        num_kv_heads: 2,
        head_dim: 2,
        dtype_size_bytes: 2,
    };
    let mut allocator = PagedKvAllocator::new(2, layout);
    let mut sequence = PagedKvSequence::new(2);
    allocator.ensure_token_len(&mut sequence, 3);
    let mut store = PagedKvPageStore::<u32>::new(2, layout);

    let err = store
        .write_sequence_layer_head_major(&sequence, 0, PagedKvPlane::Key, &[1, 2, 3])
        .unwrap_err();

    assert_eq!(
        err,
        PagedKvStorageError::LayerValueLenMismatch {
            expected: 12,
            actual: 3
        }
    );
}

#[test]
fn head_major_page_gather_matches_direct_right_align() {
    let layout = PagedKvLayout {
        num_layers: 1,
        num_kv_heads: 2,
        head_dim: 2,
        dtype_size_bytes: 2,
    };
    let mut allocator = PagedKvAllocator::new(2, layout);
    let mut short = PagedKvSequence::new(2);
    let mut full = PagedKvSequence::new(2);
    allocator.ensure_token_len(&mut short, 3);
    allocator.ensure_token_len(&mut full, 4);
    let values = vec![
        vec![10, 11, 20, 21, 30, 31, 12, 13, 22, 23, 32, 33],
        vec![
            200, 201, 210, 211, 220, 221, 230, 231, 202, 203, 212, 213, 222, 223, 232, 233,
        ],
    ];
    let sequences = [&short, &full];

    let via_pages =
        gather_head_major_layer_via_pages(2, layout, &sequences, 0, PagedKvPlane::Key, &values, 4)
            .unwrap();
    let direct = build_right_aligned_head_major_batch(layout, &[3, 4], 4, &values).unwrap();

    assert_eq!(via_pages, direct);
}

#[test]
fn page_store_reset_sequence_pages_removes_stale_reused_values() {
    let layout = PagedKvLayout {
        num_layers: 1,
        num_kv_heads: 1,
        head_dim: 1,
        dtype_size_bytes: 2,
    };
    let mut allocator = PagedKvAllocator::new(2, layout);
    let mut first = PagedKvSequence::new(2);
    let mut second = PagedKvSequence::new(2);
    let mut store = PagedKvPageStore::<u32>::new(2, layout);

    allocator.ensure_token_len(&mut first, 2);
    store
        .write_token(&first, 0, 0, PagedKvPlane::Key, &[99])
        .unwrap();
    allocator.release_sequence(&mut first);
    allocator.ensure_token_len(&mut second, 1);
    store.reset_sequence_pages(&second);

    assert_eq!(second.blocks()[0].len, 1);
    assert_eq!(
        store.read_token(&second, 0, 0, PagedKvPlane::Key).unwrap(),
        vec![0]
    );
}

#[test]
fn page_store_round_trips_boundary_lengths() {
    let layout = PagedKvLayout {
        num_layers: 1,
        num_kv_heads: 1,
        head_dim: 1,
        dtype_size_bytes: 2,
    };
    let block_size = 4;
    for token_len in [
        1,
        block_size - 1,
        block_size,
        block_size + 1,
        2 * block_size + 1,
    ] {
        let mut allocator = PagedKvAllocator::new(block_size, layout);
        let mut sequence = PagedKvSequence::new(block_size);
        allocator.ensure_token_len(&mut sequence, token_len);
        let mut store = PagedKvPageStore::<u32>::new(block_size, layout);
        for token_index in 0..token_len {
            store
                .write_token(
                    &sequence,
                    token_index,
                    0,
                    PagedKvPlane::Value,
                    &[token_index as u32 + 1],
                )
                .unwrap();
        }

        let gathered = store
            .gather_layer_right_aligned(&[&sequence], 0, token_len, PagedKvPlane::Value)
            .unwrap();
        assert_eq!(gathered, (1..=token_len as u32).collect::<Vec<_>>());
    }
}

fn put_expected_token(
    output: &mut [u32],
    row: usize,
    token_index: usize,
    max_len: usize,
    values: &[u32],
) {
    let num_heads = 2;
    let head_dim = 2;
    let row_stride = num_heads * max_len * head_dim;
    for head in 0..num_heads {
        let src = head * head_dim;
        let dst = row * row_stride + head * max_len * head_dim + token_index * head_dim;
        output[dst..dst + head_dim].copy_from_slice(&values[src..src + head_dim]);
    }
}
