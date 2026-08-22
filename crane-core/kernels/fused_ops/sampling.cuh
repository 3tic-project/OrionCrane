// =====================================================================

extern "C" __global__ void gpu_argmax_bf16_phase1(
    const __nv_bfloat16 *__restrict__ logits,  // [vocab_size]
    float               *__restrict__ block_max_vals,
    int32_t             *__restrict__ block_max_idxs,
    const int vocab_size
) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int bid = blockIdx.x;
    const int num_blocks = gridDim.x;

    // Each block handles a strided chunk
    int chunk = (vocab_size + num_blocks - 1) / num_blocks;
    int start = bid * chunk;
    int end   = min(start + chunk, vocab_size);

    float local_max = -INFINITY;
    int   local_idx = -1;

    for (int i = start + tid; i < end; i += block_size) {
        float v = __bfloat162float(logits[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Warp reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
        int   other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_val > local_max) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    // Cross-warp reduce
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    __shared__ float s_max_vals[32];
    __shared__ int   s_max_idxs[32];

    if (lane_id == 0) {
        s_max_vals[warp_id] = local_max;
        s_max_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < num_warps) {
        local_max = s_max_vals[lane_id];
        local_idx = s_max_idxs[lane_id];

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
            int   other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            block_max_vals[bid] = local_max;
            block_max_idxs[bid] = local_idx;
        }
    }
}

extern "C" __global__ void gpu_argmax_phase2(
    const float   *__restrict__ block_max_vals,
    const int32_t *__restrict__ block_max_idxs,
    int32_t       *__restrict__ output_token,
    const int num_blocks
) {
    const int tid = threadIdx.x;

    float best_val = -INFINITY;
    int   best_idx = -1;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float v = block_max_vals[i];
        if (v > best_val) {
            best_val = v;
            best_idx = block_max_idxs[i];
        }
    }

    // Warp reduce
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
        int   other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val) {
            best_val = other_val;
            best_idx = other_idx;
        }
    }

    // Cross-warp reduce
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    __shared__ float s_vals[32];
    __shared__ int   s_idxs[32];

    if (lane_id == 0) {
        s_vals[warp_id] = best_val;
        s_idxs[warp_id] = best_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x / WARP_SIZE;
        best_val = (lane_id < num_warps) ? s_vals[lane_id] : -INFINITY;
        best_idx = (lane_id < num_warps) ? s_idxs[lane_id] : -1;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
            int   other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
            if (other_val > best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            *output_token = best_idx;
        }
    }
}

// =====================================================================
// 5. Paged KV copy — copy batch K/V tokens into page storage
//
// pages layout:
//   [page][layer][K/V][block_token][kv_head][head_dim]
// source layout:
//   [batch][kv_head][src_width][head_dim]
//
// Metadata arrays are entry-major. One entry maps one source token for one
// batch row to one destination page slot. Page ids are 1-based to match the
// engine allocator; the kernel converts them to 0-based page indices.
// =====================================================================

extern "C" __global__ void gpu_argmax_batch_bf16(
    const __nv_bfloat16 *__restrict__ logits,  // [batch_size, vocab_size]
    uint32_t            *__restrict__ output_tokens,
    const int batch_size,
    const int vocab_size
) {
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const __nv_bfloat16 *row_logits = logits + (int64_t)row * vocab_size;

    float local_max = -INFINITY;
    int local_idx = 0;

    for (int i = tid; i < vocab_size; i += block_size) {
        float v = __bfloat162float(row_logits[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_val > local_max) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    __shared__ float s_vals[32];
    __shared__ int s_idxs[32];

    if (lane_id == 0) {
        s_vals[warp_id] = local_max;
        s_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane_id < num_warps) ? s_vals[lane_id] : -INFINITY;
        local_idx = (lane_id < num_warps) ? s_idxs[lane_id] : 0;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        if (lane_id == 0) {
            output_tokens[row] = (uint32_t)local_idx;
        }
    }
}

extern "C" __global__ void gpu_apply_repetition_penalty_batch_bf16(
    __nv_bfloat16       *__restrict__ logits,          // [batch_size, vocab_size], updated in-place
    const uint32_t      *__restrict__ recent_tokens,   // [batch_size, max_recent]
    const uint32_t      *__restrict__ recent_lengths,  // [batch_size]
    const float         *__restrict__ penalties,       // [batch_size]
    const int batch_size,
    const int vocab_size,
    const int max_recent
) {
    const int row = blockIdx.x;
    if (row >= batch_size) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    __nv_bfloat16 *row_logits = logits + (int64_t)row * vocab_size;
    const uint32_t *row_recent = recent_tokens + (int64_t)row * max_recent;
    const int recent_len = min((int)recent_lengths[row], max_recent);
    const float penalty = penalties[row];

    if (penalty == 1.0f || recent_len <= 0) return;

    for (int i = tid; i < recent_len; i += block_size) {
        const uint32_t token = row_recent[i];
        if ((int)token >= vocab_size) continue;

        bool first_occurrence = true;
        for (int j = 0; j < i; ++j) {
            if (row_recent[j] == token) {
                first_occurrence = false;
                break;
            }
        }
        if (!first_occurrence) continue;

        float v = __bfloat162float(row_logits[token]);
        v = (v >= 0.0f) ? (v / penalty) : (v * penalty);
        row_logits[token] = __float2bfloat16(v);
    }
}

__device__ __forceinline__ uint64_t splitmix64_next(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float uniform01_from_seed(uint64_t seed) {
    const uint64_t bits = splitmix64_next(seed) >> 40;
    return fminf(((float)bits + 0.5f) * (1.0f / 16777216.0f), 0.99999994f);
}

// Map a non-NaN BF16 bit pattern to an unsigned key whose natural ordering is
// the same as floating-point ordering.  A two-byte radix histogram can then
// find the exact kth-largest BF16 value without maintaining a 64-entry sorted
// list for every thread while scanning the vocabulary.
__device__ __forceinline__ uint16_t ordered_bf16_key(__nv_bfloat16 value) {
    const uint16_t bits = __bfloat16_as_ushort(value);
    return (bits & 0x8000u) ? (uint16_t)~bits : (uint16_t)(bits ^ 0x8000u);
}

__device__ __forceinline__ bool bf16_is_nan_bits(__nv_bfloat16 value) {
    const uint16_t bits = __bfloat16_as_ushort(value);
    return (bits & 0x7f80u) == 0x7f80u && (bits & 0x007fu) != 0;
}

// Retained as an opt-in A/B fallback for validating the radix-select sampler.
extern "C" __global__ void gpu_sample_topk_topp_batch_bf16_legacy(
    const __nv_bfloat16 *__restrict__ logits,
    uint32_t            *__restrict__ output_tokens,
    const float         *__restrict__ temperatures,
    const uint32_t      *__restrict__ top_ks,
    const float         *__restrict__ top_ps,
    const uint64_t      *__restrict__ seeds,
    const int batch_size,
    const int vocab_size,
    const int max_top_k
) {
    const int row = blockIdx.x;
    if (row >= batch_size || max_top_k <= 0 || max_top_k > 64) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    int k = (int)top_ks[row];
    if (k <= 0) {
        if (tid == 0) output_tokens[row] = 0;
        return;
    }
    k = min(k, max_top_k);
    k = min(k, vocab_size);

    float vals[64];
    uint32_t idx[64];
#pragma unroll
    for (int j = 0; j < 64; ++j) {
        vals[j] = -INFINITY;
        idx[j] = 0;
    }

    const __nv_bfloat16 *row_logits = logits + (int64_t)row * vocab_size;
    for (int token = tid; token < vocab_size; token += block_size) {
        const float value = __bfloat162float(row_logits[token]);
        if (!isnan(value)) topk_insert(value, (uint32_t)token, vals, idx, k);
    }

    extern __shared__ uint8_t smem_sample_legacy[];
    float *block_vals = (float *)smem_sample_legacy;
    uint32_t *block_idx =
        (uint32_t *)(block_vals + (uint32_t)block_size * max_top_k);
    const int base = tid * max_top_k;
    for (int j = 0; j < k; ++j) {
        block_vals[base + j] = vals[j];
        block_idx[base + j] = idx[j];
    }
    for (int j = k; j < max_top_k; ++j) {
        block_vals[base + j] = -INFINITY;
        block_idx[base + j] = 0;
    }
    __syncthreads();

    if (tid != 0) return;

    float best_vals[64];
    uint32_t best_idx[64];
#pragma unroll
    for (int j = 0; j < 64; ++j) {
        best_vals[j] = -INFINITY;
        best_idx[j] = 0;
    }
    for (int thread = 0; thread < block_size; ++thread) {
        const int thread_base = thread * max_top_k;
        for (int j = 0; j < k; ++j) {
            topk_insert(
                block_vals[thread_base + j],
                block_idx[thread_base + j],
                best_vals,
                best_idx,
                k
            );
        }
    }

    const float temperature = temperatures[row];
    if (temperature <= 0.0f || k == 1) {
        output_tokens[row] = best_idx[0];
        return;
    }

    const float inv_temp = 1.0f / fmaxf(temperature, 1.0e-6f);
    const float max_score = best_vals[0] * inv_temp;
    float probs[64];
    float total = 0.0f;
    for (int j = 0; j < k; ++j) {
        probs[j] = expf(best_vals[j] * inv_temp - max_score);
        total += probs[j];
    }

    const float top_p = top_ps[row];
    int cutoff = k;
    if (top_p > 0.0f && top_p < 1.0f && total > 0.0f) {
        const float threshold = top_p * total;
        float running = 0.0f;
        cutoff = 1;
        for (int j = 0; j < k; ++j) {
            running += probs[j];
            cutoff = j + 1;
            if (running >= threshold) break;
        }
    }

    float sample_total = 0.0f;
    for (int j = 0; j < cutoff; ++j) sample_total += probs[j];
    if (sample_total <= 0.0f || !isfinite(sample_total)) {
        output_tokens[row] = best_idx[0];
        return;
    }

    const float sample = uniform01_from_seed(seeds[row]) * sample_total;
    float running = 0.0f;
    int chosen = cutoff - 1;
    for (int j = 0; j < cutoff; ++j) {
        running += probs[j];
        if (sample <= running) {
            chosen = j;
            break;
        }
    }
    output_tokens[row] = best_idx[chosen];
}

extern "C" __global__ void gpu_sample_topk_topp_batch_bf16(
    const __nv_bfloat16 *__restrict__ logits,       // [batch_size, vocab_size]
    uint32_t            *__restrict__ output_tokens,
    const float         *__restrict__ temperatures, // [batch_size]
    const uint32_t      *__restrict__ top_ks,       // [batch_size], 0 means inactive
    const float         *__restrict__ top_ps,       // [batch_size]
    const uint64_t      *__restrict__ seeds,        // [batch_size]
    const int batch_size,
    const int vocab_size,
    const int max_top_k
) {
    const int row = blockIdx.x;
    if (row >= batch_size || max_top_k <= 0 || max_top_k > 64) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    int k = (int)top_ks[row];
    if (k <= 0) {
        if (tid == 0) output_tokens[row] = 0;
        return;
    }
    k = min(k, max_top_k);
    k = min(k, vocab_size);

    const __nv_bfloat16 *row_logits = logits + (int64_t)row * vocab_size;

    __shared__ uint32_t histogram[256];
    __shared__ uint32_t high_threshold;
    __shared__ uint32_t cutoff_key;
    __shared__ uint32_t strict_count;
    __shared__ uint32_t strict_written;
    __shared__ uint32_t ties_written;
    __shared__ float best_vals[64];
    __shared__ uint32_t best_idx[64];

    // Keep the launch width tunable.  Wider blocks expose more of a single
    // vocabulary row to the GPU (important when batch_size is well below the
    // SM count), while the histogram itself remains fixed at 256 bins.
    for (int bin = tid; bin < 256; bin += block_size) histogram[bin] = 0;
    __syncthreads();

    // First radix pass: histogram the high byte.  Warp-aggregated atomics
    // avoid serializing all lanes when logits cluster in a small value range.
    for (int base = 0; base < vocab_size; base += block_size) {
        const int token = base + tid;
        const bool in_range = token < vocab_size;
        const __nv_bfloat16 value = in_range ? row_logits[token] : __float2bfloat16(0.0f);
        const bool valid = in_range && !bf16_is_nan_bits(value);
        const unsigned active = __ballot_sync(0xffffffffu, valid);
        if (valid) {
            const uint32_t bin = (uint32_t)(ordered_bf16_key(value) >> 8);
            const unsigned peers = __match_any_sync(active, bin);
            if ((threadIdx.x & 31) == (__ffs((int)peers) - 1)) {
                atomicAdd(&histogram[bin], (uint32_t)__popc(peers));
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t above = 0;
        uint32_t threshold = 0;
        for (int bin = 255; bin >= 0; --bin) {
            const uint32_t count = histogram[bin];
            if (above + count >= (uint32_t)k) {
                threshold = (uint32_t)bin;
                break;
            }
            above += count;
        }
        high_threshold = threshold;
        // Retain this count: it is exactly the number of candidates above the
        // selected high-byte bucket, so there is no reason to scan the full
        // vocabulary again after the low-byte pass.
        strict_count = above;
    }
    __syncthreads();
    for (int bin = tid; bin < 256; bin += block_size) histogram[bin] = 0;
    __syncthreads();

    // Second radix pass only touches values in the selected high-byte bucket.
    for (int base = 0; base < vocab_size; base += block_size) {
        const int token = base + tid;
        const bool in_range = token < vocab_size;
        const __nv_bfloat16 value = in_range ? row_logits[token] : __float2bfloat16(0.0f);
        const bool valid = in_range && !bf16_is_nan_bits(value);
        const uint32_t key = valid ? (uint32_t)ordered_bf16_key(value) : 0;
        const bool in_bucket = valid && (key >> 8) == high_threshold;
        const unsigned active = __ballot_sync(0xffffffffu, in_bucket);
        if (in_bucket) {
            const uint32_t bin = key & 0xffu;
            const unsigned peers = __match_any_sync(active, bin);
            if ((threadIdx.x & 31) == (__ffs((int)peers) - 1)) {
                atomicAdd(&histogram[bin], (uint32_t)__popc(peers));
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        const uint32_t need = (uint32_t)k - strict_count;
        uint32_t above = 0;
        uint32_t threshold = 0;
        for (int bin = 255; bin >= 0; --bin) {
            const uint32_t count = histogram[bin];
            if (above + count >= need) {
                threshold = (uint32_t)bin;
                break;
            }
            above += count;
        }
        cutoff_key = (high_threshold << 8) | threshold;
        strict_count += above;
        strict_written = 0;
        ties_written = 0;
    }
    __syncthreads();

    // One final vocabulary pass gathers both strict winners and enough cutoff
    // ties. The two independent counters make their destination ranges
    // disjoint even though blocks encounter the two classes in arbitrary
    // token order. Ties use one atomic reservation per warp.
    for (int base_token = 0; base_token < vocab_size; base_token += block_size) {
        const int token = base_token + tid;
        const bool in_range = token < vocab_size;
        const __nv_bfloat16 value =
            in_range ? row_logits[token] : __float2bfloat16(0.0f);
        const bool valid = in_range && !bf16_is_nan_bits(value);
        const uint32_t key = valid ? (uint32_t)ordered_bf16_key(value) : 0;

        if (valid && key > cutoff_key) {
            const uint32_t slot = atomicAdd(&strict_written, 1u);
            if (slot < strict_count) {
                best_vals[slot] = __bfloat162float(value);
                best_idx[slot] = (uint32_t)token;
            }
        }

        const bool wanted = valid && key == cutoff_key;
        const unsigned peers = __ballot_sync(0xffffffffu, wanted);
        if (wanted) {
            const int lane = threadIdx.x & 31;
            const int leader = __ffs((int)peers) - 1;
            uint32_t base = 0;
            if (lane == leader) base = atomicAdd(&ties_written, (uint32_t)__popc(peers));
            base = __shfl_sync(peers, base, leader);
            const uint32_t rank = (uint32_t)__popc(peers & ((1u << lane) - 1u));
            const uint32_t slot = strict_count + base + rank;
            if (slot < (uint32_t)k) {
                best_vals[slot] = __bfloat162float(value);
                best_idx[slot] = (uint32_t)token;
            }
        }
    }
    __syncthreads();

    if (tid != 0) return;

    // Sort only the final <=64 candidates. Stable token-id tie breaking keeps
    // deterministic seeds deterministic even when BF16 logits are equal.
    for (int i = 1; i < k; ++i) {
        const float value = best_vals[i];
        const uint32_t token = best_idx[i];
        int j = i;
        while (j > 0
               && (value > best_vals[j - 1]
                   || (value == best_vals[j - 1] && token < best_idx[j - 1]))) {
            best_vals[j] = best_vals[j - 1];
            best_idx[j] = best_idx[j - 1];
            --j;
        }
        best_vals[j] = value;
        best_idx[j] = token;
    }

    const float temperature = temperatures[row];
    if (temperature <= 0.0f || k == 1) {
        output_tokens[row] = best_idx[0];
        return;
    }

    const float inv_temp = 1.0f / fmaxf(temperature, 1.0e-6f);
    const float max_score = best_vals[0] * inv_temp;
    float probs[64];
    float total = 0.0f;
    for (int j = 0; j < k; ++j) {
        probs[j] = expf(best_vals[j] * inv_temp - max_score);
        total += probs[j];
    }

    float top_p = top_ps[row];
    int cutoff = k;
    if (top_p > 0.0f && top_p < 1.0f && total > 0.0f) {
        const float threshold = top_p * total;
        float running = 0.0f;
        cutoff = 1;
        for (int j = 0; j < k; ++j) {
            running += probs[j];
            cutoff = j + 1;
            if (running >= threshold) break;
        }
    }

    float sample_total = 0.0f;
    for (int j = 0; j < cutoff; ++j) sample_total += probs[j];
    if (sample_total <= 0.0f || !isfinite(sample_total)) {
        output_tokens[row] = best_idx[0];
        return;
    }

    const float u = uniform01_from_seed(seeds[row]) * sample_total;
    float running = 0.0f;
    int chosen = cutoff - 1;
    for (int j = 0; j < cutoff; ++j) {
        running += probs[j];
        if (u <= running) {
            chosen = j;
            break;
        }
    }
    output_tokens[row] = best_idx[chosen];
}
