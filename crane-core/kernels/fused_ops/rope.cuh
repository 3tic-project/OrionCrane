// =====================================================================
// Decode-time indexed RoPE for BF16 Qwen3 tensors.
//
// x layout:      [batch, heads, 1, head_dim]
// cos/sin table: [max_position, head_dim / 2], f32
// positions:     [batch], u32
// output layout: [batch, heads, 1, head_dim]
// =====================================================================

extern "C" __global__ void fused_rope_indexed_bf16(
    const __nv_bfloat16 *__restrict__ x,
    const float         *__restrict__ cos,
    const float         *__restrict__ sin,
    const uint32_t      *__restrict__ positions,
    __nv_bfloat16       *__restrict__ output,
    const int batch_size,
    const int num_heads,
    const int head_dim,
    const int max_position
) {
    const int half_dim = head_dim / 2;
    const int total = batch_size * num_heads * half_dim;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int dim = idx % half_dim;
        const int head_idx = idx / half_dim;
        const int head = head_idx % num_heads;
        const int row = head_idx / num_heads;
        if (row >= batch_size || head >= num_heads) continue;

        const int pos = min((int)positions[row], max_position - 1);
        const int64_t base = ((int64_t)row * num_heads + head) * head_dim;
        const int64_t table = (int64_t)pos * half_dim + dim;
        const float c = cos[table];
        const float s = sin[table];
        const float x0 = __bfloat162float(x[base + dim]);
        const float x1 = __bfloat162float(x[base + half_dim + dim]);
        output[base + dim] = __float2bfloat16(x0 * c - x1 * s);
        output[base + half_dim + dim] = __float2bfloat16(x0 * s + x1 * c);
    }
}

// =====================================================================
// Decode-time QK RMSNorm + indexed RoPE for BF16 Qwen3 tensors.
//
// q layout:      [batch, q_heads, 1, head_dim]
// k layout:      [batch, kv_heads, 1, head_dim]
// weights:       [head_dim]
// cos/sin table: [max_position, head_dim / 2], f32
// positions:     [batch], u32
// output layouts match q/k.
// =====================================================================

extern "C" __global__ void fused_qk_norm_rope_indexed_bf16(
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ q_weight,
    const __nv_bfloat16 *__restrict__ k_weight,
    const float         *__restrict__ cos,
    const float         *__restrict__ sin,
    const uint32_t      *__restrict__ positions,
    __nv_bfloat16       *__restrict__ q_output,
    __nv_bfloat16       *__restrict__ k_output,
    const int batch_size,
    const int q_heads,
    const int kv_heads,
    const int head_dim,
    const int max_position,
    const float eps
) {
    const int total_heads = q_heads + kv_heads;
    const int row = blockIdx.x;
    if (row >= batch_size * total_heads) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int half_dim = head_dim / 2;
    const int batch = row / total_heads;
    const int head_in_row = row - batch * total_heads;
    const bool is_q = head_in_row < q_heads;
    const int head = is_q ? head_in_row : head_in_row - q_heads;
    const __nv_bfloat16 *src = is_q ? q : k;
    const __nv_bfloat16 *weight = is_q ? q_weight : k_weight;
    __nv_bfloat16 *dst = is_q ? q_output : k_output;
    const int heads = is_q ? q_heads : kv_heads;
    const int64_t base = ((int64_t)batch * heads + head) * head_dim;

    float sum_sq = 0.0f;
    for (int col = tid; col < head_dim; col += block_size) {
        const float v = __bfloat162float(src[base + col]);
        sum_sq += v * v;
    }

    sum_sq = warp_reduce_sum_f32(sum_sq);
    __shared__ float s_partial[32];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = block_size / WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum_f32(sum_sq);
        if (lane_id == 0) s_partial[0] = sum_sq;
    }
    __syncthreads();

    const float scale = rsqrtf(s_partial[0] / (float)head_dim + eps);
    const int pos = min((int)positions[batch], max_position - 1);
    const int64_t table = (int64_t)pos * half_dim;

    for (int dim = tid; dim < half_dim; dim += block_size) {
        const float c = cos[table + dim];
        const float s = sin[table + dim];
        const float w0 = __bfloat162float(weight[dim]);
        const float w1 = __bfloat162float(weight[half_dim + dim]);
        const float x0 = __bfloat162float(src[base + dim]) * scale * w0;
        const float x1 = __bfloat162float(src[base + half_dim + dim]) * scale * w1;
        dst[base + dim] = __float2bfloat16(x0 * c - x1 * s);
        dst[base + half_dim + dim] = __float2bfloat16(x0 * s + x1 * c);
    }
}
