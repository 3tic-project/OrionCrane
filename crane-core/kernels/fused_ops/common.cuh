/**
 * Fused CUDA kernels for Crane transformer inference.
 *
 * Targets: sm_80+ (Ampere & newer, bf16 support)
 *
 * Kernels:
 *   1. fused_rmsnorm_residual_bf16  — RMSNorm + residual save
 *   2. fused_silu_mul_bf16          — SiLU(gate) * up  (one pass)
 *   3. fused_add_rmsnorm_bf16       — residual_add + RMSNorm (one pass)
 *   4. gpu_argmax_bf16              — GPU-side argmax over vocab (greedy decode)
 *   5. paged_kv_append_bf16         — copy batch K/V tokens into page storage
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// =====================================================================
// Helpers
// =====================================================================

static constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum_f32(float val, float *shared) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    val = warp_reduce_sum_f32(val);
    if (lane_id == 0) shared[warp_id] = val;
    __syncthreads();

    val = (warp_id == 0 && lane_id < num_warps) ? shared[lane_id] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum_f32(val);
    if (tid == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

// Fast SiLU: x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float fast_silu(float x) {
    return x / (1.0f + expf(-x));
}

// =====================================================================
