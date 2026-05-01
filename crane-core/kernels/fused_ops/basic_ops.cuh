//
//    dst[row, col] = (x[row, col] / rms) * weight[col]
//    where rms = sqrt(mean(x²) + eps)
//
//    Identical to candle's rmsnorm but with explicit bf16 I/O and
//    handles up to 16384 columns per warp-tree reduction.
// =====================================================================

extern "C" __global__ void fused_rmsnorm_bf16(
    const __nv_bfloat16 *__restrict__ x,      // [rows, cols]
    __nv_bfloat16       *__restrict__ dst,     // [rows, cols]
    const __nv_bfloat16 *__restrict__ weight,  // [cols]
    const int ncols,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Phase 1: compute sum of squares
    float sum_sq = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        float v = __bfloat162float(x[row * ncols + col]);
        sum_sq += v * v;
    }

    // Warp reduce
    sum_sq = warp_reduce_sum_f32(sum_sq);

    // Cross-warp reduce via shared memory
    __shared__ float s_partial[32];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum_f32(sum_sq);
        if (lane_id == 0) s_partial[0] = sum_sq;
    }
    __syncthreads();

    float scale = rsqrtf(s_partial[0] / (float)ncols + eps);

    // Phase 2: normalize and write output
    for (int col = tid; col < ncols; col += block_size) {
        float v = __bfloat162float(x[row * ncols + col]);
        float w = __bfloat162float(weight[col]);
        dst[row * ncols + col] = __float2bfloat16(v * scale * w);
    }
}

// =====================================================================
// 2. Fused SiLU(gate) * up — one pass over 2 * intermediate_size
//
//    Input:  gate_up [rows, 2*intermediate_size]  (gate||up concatenated)
//    Output: dst     [rows, intermediate_size]
//    dst[i] = silu(gate_up[i]) * gate_up[i + intermediate_size]
//
//    Saves 2 kernel launches (separate silu + mul) and 1 intermediate
//    tensor allocation.
// =====================================================================

extern "C" __global__ void fused_silu_mul_bf16(
    const __nv_bfloat16 *__restrict__ gate_up,  // [rows, 2*intermediate_size]
    __nv_bfloat16       *__restrict__ dst,       // [rows, intermediate_size]
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const __nv_bfloat16 *gate_row = gate_up + row * 2 * intermediate_size;
    const __nv_bfloat16 *up_row   = gate_row + intermediate_size;
    __nv_bfloat16       *dst_row  = dst + row * intermediate_size;

    for (int i = tid; i < intermediate_size; i += block_size) {
        float g = __bfloat162float(gate_row[i]);
        float u = __bfloat162float(up_row[i]);
        dst_row[i] = __float2bfloat16(fast_silu(g) * u);
    }
}

// f16 variant
extern "C" __global__ void fused_silu_mul_f16(
    const __half *__restrict__ gate_up,
    __half       *__restrict__ dst,
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const __half *gate_row = gate_up + row * 2 * intermediate_size;
    const __half *up_row   = gate_row + intermediate_size;
    __half       *dst_row  = dst + row * intermediate_size;

    for (int i = tid; i < intermediate_size; i += block_size) {
        float g = __half2float(gate_row[i]);
        float u = __half2float(up_row[i]);
        dst_row[i] = __float2half(fast_silu(g) * u);
    }
}

// f32 variant
extern "C" __global__ void fused_silu_mul_f32(
    const float *__restrict__ gate_up,
    float       *__restrict__ dst,
    const int intermediate_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float *gate_row = gate_up + row * 2 * intermediate_size;
    const float *up_row   = gate_row + intermediate_size;
    float       *dst_row  = dst + row * intermediate_size;

    for (int i = tid; i < intermediate_size; i += block_size) {
        float g = gate_row[i];
        float u = up_row[i];
        dst_row[i] = fast_silu(g) * u;
    }
}

// =====================================================================
// 3. Fused residual_add + RMSNorm — one read of hidden, write norm + residual
//
//    residual_out[row] = residual[row] + hidden[row]
//    dst[row] = rmsnorm(residual_out[row]) * weight
//
//    Eliminates the separate add kernel + RMSNorm kernel + extra read.
// =====================================================================

extern "C" __global__ void fused_add_rmsnorm_bf16(
    const __nv_bfloat16 *__restrict__ residual,  // [rows, cols]
    const __nv_bfloat16 *__restrict__ hidden,    // [rows, cols] — value to add
    __nv_bfloat16       *__restrict__ residual_out, // [rows, cols]
    __nv_bfloat16       *__restrict__ dst,        // [rows, cols] — normalized output
    const __nv_bfloat16 *__restrict__ weight,     // [cols]
    const int ncols,
    const float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int row_offset = row * ncols;

    // Phase 1: add residual, compute sum of squares
    float sum_sq = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        float r = __bfloat162float(residual[row_offset + col]);
        float h = __bfloat162float(hidden[row_offset + col]);
        float v = r + h;
        residual_out[row_offset + col] = __float2bfloat16(v);
        sum_sq += v * v;
    }

    // Warp + cross-warp reduce
    sum_sq = warp_reduce_sum_f32(sum_sq);
    __shared__ float s_partial[32];
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = block_size / WARP_SIZE;

    if (lane_id == 0) s_partial[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum_f32(sum_sq);
        if (lane_id == 0) s_partial[0] = sum_sq;
    }
    __syncthreads();

    float scale = rsqrtf(s_partial[0] / (float)ncols + eps);

    // Phase 2: normalize from updated residual
    for (int col = tid; col < ncols; col += block_size) {
        float v = __bfloat162float(residual_out[row_offset + col]);
        float w = __bfloat162float(weight[col]);
        dst[row_offset + col] = __float2bfloat16(v * scale * w);
    }
}

// =====================================================================
// 5. Fused residual_add in-place: residual += hidden
//    Simple element-wise kernel to avoid candle's tensor add overhead.
// =====================================================================

extern "C" __global__ void fused_residual_add_bf16(
    __nv_bfloat16       *__restrict__ residual,  // [n] — updated in-place
    const __nv_bfloat16 *__restrict__ hidden,    // [n]
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r = __bfloat162float(residual[idx]);
        float h = __bfloat162float(hidden[idx]);
        residual[idx] = __float2bfloat16(r + h);
    }
}
