extern "C" __global__ void paged_kv_append_bf16(
    __nv_bfloat16       *__restrict__ pages,
    const __nv_bfloat16 *__restrict__ full_k,
    const __nv_bfloat16 *__restrict__ full_v,
    const uint32_t      *__restrict__ page_ids,
    const uint32_t      *__restrict__ token_offsets,
    const uint32_t      *__restrict__ row_indices,
    const uint32_t      *__restrict__ source_token_indices,
    const int entries,
    const int layer,
    const int src_width,
    const int num_layers,
    const int block_size,
    const int num_kv_heads,
    const int head_dim
) {
    const int entry = blockIdx.x;
    if (entry >= entries) return;

    const int token_width = num_kv_heads * head_dim;
    const uint32_t page_id = page_ids[entry];
    if (page_id == 0) return;

    const int page = (int)page_id - 1;
    const int token_offset = (int)token_offsets[entry];
    const int row = (int)row_indices[entry];
    const int src_token = (int)source_token_indices[entry];
    if (src_token < 0 || src_token >= src_width) return;

    const int page_values = num_layers * 2 * block_size * token_width;
    const int layer_stride = 2 * block_size * token_width;
    const int plane_stride = block_size * token_width;
    const int page_base = page * page_values + layer * layer_stride + token_offset * token_width;
    const int src_row_base = row * num_kv_heads * src_width * head_dim;

    for (int idx = threadIdx.x; idx < token_width; idx += blockDim.x) {
        const int head = idx / head_dim;
        const int dim = idx - head * head_dim;
        const int src = src_row_base + head * src_width * head_dim + src_token * head_dim + dim;
        pages[page_base + idx] = full_k[src];
        pages[page_base + plane_stride + idx] = full_v[src];
    }
}

extern "C" __global__ void batch_kv_append_bf16_with_offset(
    __nv_bfloat16       *__restrict__ dst_k,
    __nv_bfloat16       *__restrict__ dst_v,
    const __nv_bfloat16 *__restrict__ src_k,
    const __nv_bfloat16 *__restrict__ src_v,
    const uint32_t      *__restrict__ append_offset,
    const int batch_size,
    const int dst_width,
    const int num_kv_heads,
    const int head_dim
) {
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const int dst_token = (int)(*append_offset);
    if (dst_token < 0 || dst_token >= dst_width) return;

    const int token_width = num_kv_heads * head_dim;
    const int64_t src_row_base = (int64_t)row * token_width;
    const int64_t dst_row_base = (int64_t)row * num_kv_heads * dst_width * head_dim;

    for (int idx = threadIdx.x; idx < token_width; idx += blockDim.x) {
        const int head = idx / head_dim;
        const int dim = idx - head * head_dim;
        const int64_t src = src_row_base + idx;
        const int64_t dst = dst_row_base + (int64_t)head * dst_width * head_dim + (int64_t)dst_token * head_dim + dim;
        dst_k[dst] = src_k[src];
        dst_v[dst] = src_v[src];
    }
}

extern "C" __global__ void batch_kv_copy_ragged_bf16(
    __nv_bfloat16       *__restrict__ dst_k,
    __nv_bfloat16       *__restrict__ dst_v,
    const __nv_bfloat16 *__restrict__ src_k,
    const __nv_bfloat16 *__restrict__ src_v,
    const uint32_t      *__restrict__ kv_lens,
    const int batch_size,
    const int src_width,
    const int dst_width,
    const int num_kv_heads,
    const int head_dim
) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= batch_size || token >= src_width) return;

    const int kv_len = (int)kv_lens[row];
    if (kv_len <= 0 || kv_len > src_width) return;
    const int offset = src_width - kv_len;
    if (token < offset) return;

    const int token_width = num_kv_heads * head_dim;
    const int64_t src_row_base = (int64_t)row * num_kv_heads * src_width * head_dim;
    const int64_t dst_row_base = (int64_t)row * num_kv_heads * dst_width * head_dim;

    for (int idx = threadIdx.x; idx < token_width; idx += blockDim.x) {
        const int head = idx / head_dim;
        const int dim = idx - head * head_dim;
        const int64_t src = src_row_base + (int64_t)head * src_width * head_dim + (int64_t)token * head_dim + dim;
        const int64_t dst = dst_row_base + (int64_t)head * dst_width * head_dim + (int64_t)token * head_dim + dim;
        dst_k[dst] = src_k[src];
        dst_v[dst] = src_v[src];
    }
}

extern "C" __global__ void paged_kv_gather_bf16(
    const __nv_bfloat16 *__restrict__ pages,
    __nv_bfloat16       *__restrict__ output,
    const uint32_t      *__restrict__ page_ids,
    const uint32_t      *__restrict__ token_offsets,
    const uint32_t      *__restrict__ row_indices,
    const uint32_t      *__restrict__ target_token_indices,
    const int entries,
    const int layer,
    const int plane,
    const int max_len,
    const int num_layers,
    const int block_size,
    const int num_kv_heads,
    const int head_dim
) {
    const int entry = blockIdx.x;
    if (entry >= entries) return;

    const int token_width = num_kv_heads * head_dim;
    const uint32_t page_id = page_ids[entry];
    if (page_id == 0) return;

    const int page = (int)page_id - 1;
    const int token_offset = (int)token_offsets[entry];
    const int row = (int)row_indices[entry];
    const int target_token = (int)target_token_indices[entry];
    if (target_token < 0 || target_token >= max_len) return;

    const int page_values = num_layers * 2 * block_size * token_width;
    const int layer_stride = 2 * block_size * token_width;
    const int plane_stride = block_size * token_width;
    const int page_base = page * page_values + layer * layer_stride + plane * plane_stride + token_offset * token_width;
    const int dst_row_base = row * num_kv_heads * max_len * head_dim;

    for (int idx = threadIdx.x; idx < token_width; idx += blockDim.x) {
        const int head = idx / head_dim;
        const int dim = idx - head * head_dim;
        const int dst = dst_row_base + head * max_len * head_dim + target_token * head_dim + dim;
        output[dst] = pages[page_base + idx];
    }
}

extern "C" __global__ void paged_attention_decode_bf16(
    const __nv_bfloat16 *__restrict__ pages,
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ current_k,
    const __nv_bfloat16 *__restrict__ current_v,
    __nv_bfloat16       *__restrict__ output,
    const uint32_t      *__restrict__ indptr,
    const uint32_t      *__restrict__ indices,
    const uint32_t      *__restrict__ last_page_lens,
    const uint32_t      *__restrict__ seq_lens,
    const int batch_size,
    const int layer,
    const int num_layers,
    const int block_size,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const float scale
) {
    // Split the sequence scan across multiple warps. The previous one-warp
    // implementation made every block execute the whole context serially and
    // limited Ada occupancy because a single-warp block still consumes a block
    // scheduling slot. Each warp computes an independent online-softmax state;
    // warp 0 merges those states and writes the final head.
    static constexpr int MAX_PAGED_ATTN_WARPS = 8;
    static constexpr int MAX_PAGED_ATTN_HEAD_DIM = 256;
    __shared__ float partial_max[MAX_PAGED_ATTN_WARPS];
    __shared__ float partial_sum[MAX_PAGED_ATTN_WARPS];
    __shared__ float partial_weight[MAX_PAGED_ATTN_WARPS];
    __shared__ float partial_acc[MAX_PAGED_ATTN_WARPS * MAX_PAGED_ATTN_HEAD_DIM];

    const int row = blockIdx.x;
    const int head = blockIdx.y;
    if (row >= batch_size || head >= num_heads || num_kv_heads <= 0 || head_dim > 256) return;

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;
    if (num_warps <= 0 || num_warps > MAX_PAGED_ATTN_WARPS) return;
    const int n_rep = max(num_heads / num_kv_heads, 1);
    const int kv_head = min(head / n_rep, num_kv_heads - 1);
    const int seq_len = (int)seq_lens[row];
    const int page_start = (int)indptr[row];
    const int page_end = (int)indptr[row + 1];
    const int page_count = max(page_end - page_start, 0);
    const int last_page_len = (int)last_page_lens[row];
    const int token_width = num_kv_heads * head_dim;
    const int64_t page_values = (int64_t)num_layers * 2 * block_size * token_width;
    const int64_t layer_stride = (int64_t)2 * block_size * token_width;
    const int64_t plane_stride = (int64_t)block_size * token_width;
    const __nv_bfloat16 *q_row = q + ((int64_t)row * num_heads + head) * head_dim;
    const __nv_bfloat16 *cur_k_row = current_k + ((int64_t)row * num_kv_heads + kv_head) * head_dim;
    const __nv_bfloat16 *cur_v_row = current_v + ((int64_t)row * num_kv_heads + kv_head) * head_dim;

    float q_vals[8];
    float acc_vals[8];
    int dims[8];
    const int values_per_lane = (head_dim + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        const int dim = lane * values_per_lane + slot;
        dims[slot] = dim;
        q_vals[slot] = 0.0f;
        acc_vals[slot] = 0.0f;
    }
    if (head_dim == 128) {
        // Qwen3's head is exactly four BF16 values per lane. One 64-bit
        // read replaces four scalar global loads and matches the layout used
        // by the reference megakernel attention implementation.
        const uint2 packed_q = __ldg(reinterpret_cast<const uint2 *>(q_row + lane * 4));
        const __nv_bfloat16 *q4 = reinterpret_cast<const __nv_bfloat16 *>(&packed_q);
#pragma unroll
        for (int slot = 0; slot < 4; ++slot) {
            q_vals[slot] = __bfloat162float(q4[slot]);
        }
    } else {
#pragma unroll
        for (int slot = 0; slot < 8; ++slot) {
            const int dim = dims[slot];
            q_vals[slot] = dim < head_dim ? __bfloat162float(q_row[dim]) : 0.0f;
        }
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (int token = warp; token <= seq_len; token += num_warps) {
        const bool is_current = token == seq_len;
        int64_t base = 0;
        bool valid = is_current;
        if (!is_current) {
            const int page_slot = token / block_size;
            const int token_offset = token - page_slot * block_size;
            if (page_slot < page_count) {
                const int page_id = (int)indices[page_start + page_slot];
                const bool inside_last = page_slot + 1 != page_count
                    || last_page_len <= 0
                    || token_offset < last_page_len;
                if (page_id > 0 && inside_last) {
                    const int page = page_id - 1;
                    base = (int64_t)page * page_values
                        + (int64_t)layer * layer_stride
                        + (int64_t)token_offset * token_width
                        + (int64_t)kv_head * head_dim;
                    valid = true;
                }
            }
        }

        if (!valid) continue;

        const __nv_bfloat16 *key_row = is_current ? cur_k_row : pages + base;
        float local_dot = 0.0f;
        if (head_dim == 128) {
            const uint2 packed_k = __ldg(reinterpret_cast<const uint2 *>(key_row + lane * 4));
            const __nv_bfloat16 *k4 = reinterpret_cast<const __nv_bfloat16 *>(&packed_k);
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                local_dot += q_vals[slot] * __bfloat162float(k4[slot]);
            }
        } else {
#pragma unroll
            for (int slot = 0; slot < 8; ++slot) {
                const int dim = dims[slot];
                if (dim < head_dim) {
                    local_dot += q_vals[slot] * __bfloat162float(key_row[dim]);
                }
            }
        }
        const float dot = __shfl_sync(0xffffffff, warp_reduce_sum_f32(local_dot), 0);
        const float score = dot * scale;
        float alpha;
        float beta;
        if (score > running_max) {
            alpha = isinf(running_max) ? 0.0f : __expf(running_max - score);
            beta = 1.0f;
            running_max = score;
        } else {
            alpha = 1.0f;
            beta = __expf(score - running_max);
        }
        running_sum = running_sum * alpha + beta;

        const __nv_bfloat16 *value_row = is_current ? cur_v_row : pages + base + plane_stride;
        if (head_dim == 128) {
            const uint2 packed_v = __ldg(reinterpret_cast<const uint2 *>(value_row + lane * 4));
            const __nv_bfloat16 *v4 = reinterpret_cast<const __nv_bfloat16 *>(&packed_v);
#pragma unroll
            for (int slot = 0; slot < 4; ++slot) {
                acc_vals[slot] = acc_vals[slot] * alpha
                    + beta * __bfloat162float(v4[slot]);
            }
        } else {
#pragma unroll
            for (int slot = 0; slot < 8; ++slot) {
                const int dim = dims[slot];
                if (dim < head_dim) {
                    acc_vals[slot] = acc_vals[slot] * alpha
                        + beta * __bfloat162float(value_row[dim]);
                }
            }
        }
    }

    if (lane == 0) {
        partial_max[warp] = running_max;
        partial_sum[warp] = running_sum;
    }
#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        const int dim = dims[slot];
        if (dim < head_dim) {
            partial_acc[warp * MAX_PAGED_ATTN_HEAD_DIM + dim] = acc_vals[slot];
        }
    }
    __syncthreads();

    // The merge factor is identical for every output dimension. Compute it
    // once per source warp instead of evaluating expf in every output lane.
    if (threadIdx.x == 0) {
        float merged_max = -INFINITY;
#pragma unroll
        for (int source_warp = 0; source_warp < MAX_PAGED_ATTN_WARPS; ++source_warp) {
            if (source_warp < num_warps) {
                merged_max = fmaxf(merged_max, partial_max[source_warp]);
            }
        }
        float merged_sum = 0.0f;
#pragma unroll
        for (int source_warp = 0; source_warp < MAX_PAGED_ATTN_WARPS; ++source_warp) {
            if (source_warp < num_warps) {
                const float scale = __expf(partial_max[source_warp] - merged_max);
                partial_weight[source_warp] = scale;
                merged_sum += partial_sum[source_warp] * scale;
            }
        }
        const float inv_denom = 1.0f / fmaxf(merged_sum, 1.0e-20f);
#pragma unroll
        for (int source_warp = 0; source_warp < MAX_PAGED_ATTN_WARPS; ++source_warp) {
            if (source_warp < num_warps) {
                partial_weight[source_warp] *= inv_denom;
            }
        }
    }
    __syncthreads();

    if (warp != 0) return;

#pragma unroll
    for (int slot = 0; slot < 8; ++slot) {
        const int dim = dims[slot];
        if (dim >= head_dim) continue;
        float merged_acc = 0.0f;
#pragma unroll
        for (int source_warp = 0; source_warp < MAX_PAGED_ATTN_WARPS; ++source_warp) {
            if (source_warp < num_warps) {
                merged_acc += partial_acc[
                    source_warp * MAX_PAGED_ATTN_HEAD_DIM + dim
                ] * partial_weight[source_warp];
            }
        }
        output[((int64_t)row * num_heads + head) * head_dim + dim] =
            __float2bfloat16(merged_acc);
    }
}

extern "C" __global__ void paged_kv_zero_pages_bf16(
    __nv_bfloat16  *__restrict__ pages,
    const uint32_t *__restrict__ page_ids,
    const int num_pages,
    const int page_values
) {
    const int entry = blockIdx.x;
    if (entry >= num_pages) return;
    const uint32_t page_id = page_ids[entry];
    if (page_id == 0) return;

    const int page = (int)page_id - 1;
    __nv_bfloat16 *base = pages + page * page_values;
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);
    for (int idx = threadIdx.x; idx < page_values; idx += blockDim.x) {
        base[idx] = zero;
    }
}
