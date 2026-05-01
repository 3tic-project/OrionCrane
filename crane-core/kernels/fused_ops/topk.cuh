// =====================================================================
// 6. GPU TopK — two-stage block reduction (k ≤ 64)
//
//    Stage 1: Each block processes `items_per_block` elements from
//             the input, producing per-block top-k.
//    Stage 2: Single block merges all per-block results → final top-k
//             indices output.
// =====================================================================

static inline __device__ void topk_insert(
    float v,
    uint32_t i,
    float * vals,
    uint32_t * idx,
    const int k
) {
    if (v <= vals[k - 1]) return;
    int p = k - 1;
    while (p > 0 && v > vals[p - 1]) {
        vals[p] = vals[p - 1];
        idx[p] = idx[p - 1];
        --p;
    }
    vals[p] = v;
    idx[p] = i;
}

extern "C" __global__ void topk_stage1_f32(
    const float * x,
    const uint32_t n,
    const uint32_t k,
    const uint32_t items_per_block,
    float * out_vals,
    uint32_t * out_idx
) {
    const uint32_t start = blockIdx.x * items_per_block;
    const uint32_t end = min(n, start + items_per_block);
    if (start >= end) return;

    float vals[64];
    uint32_t idx[64];
#pragma unroll
    for (int j = 0; j < 64; ++j) {
        vals[j] = -INFINITY;
        idx[j] = 0;
    }

    for (uint32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        float v = x[i];
        topk_insert(v, i, vals, idx, (int)k);
    }

    extern __shared__ uint8_t smem[];
    float * block_vals = (float *)smem;
    uint32_t * block_idx = (uint32_t *)(block_vals + (uint32_t)blockDim.x * k);

    const uint32_t base = (uint32_t)threadIdx.x * k;
    for (uint32_t j = 0; j < k; ++j) {
        block_vals[base + j] = vals[j];
        block_idx[base + j] = idx[j];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float bvals[64];
        uint32_t bidx[64];
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            bvals[j] = -INFINITY;
            bidx[j] = 0;
        }
        for (uint32_t t = 0; t < (uint32_t)blockDim.x; ++t) {
            const uint32_t tb = t * k;
            for (uint32_t j = 0; j < k; ++j) {
                topk_insert(block_vals[tb + j], block_idx[tb + j], bvals, bidx, (int)k);
            }
        }
        const uint32_t out_base = blockIdx.x * k;
        for (uint32_t j = 0; j < k; ++j) {
            out_vals[out_base + j] = bvals[j];
            out_idx[out_base + j] = bidx[j];
        }
    }
}

extern "C" __global__ void topk_stage2_f32(
    const float * in_vals,
    const uint32_t * in_idx,
    const uint32_t m,
    const uint32_t k,
    uint32_t * out_idx
) {
    float vals[64];
    uint32_t idx[64];
    #pragma unroll
    for (int j = 0; j < 64; ++j) {
        vals[j] = -INFINITY;
        idx[j] = 0;
    }

    for (uint32_t i = threadIdx.x; i < m; i += blockDim.x) {
        float v = in_vals[i];
        uint32_t id = in_idx[i];
        topk_insert(v, id, vals, idx, (int)k);
    }

    extern __shared__ uint8_t smem2[];
    float * block_vals = (float *)smem2;
    uint32_t * block_idx = (uint32_t *)(block_vals + (uint32_t)blockDim.x * k);

    const uint32_t base = (uint32_t)threadIdx.x * k;
    for (uint32_t j = 0; j < k; ++j) {
        block_vals[base + j] = vals[j];
        block_idx[base + j] = idx[j];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float bvals[64];
        uint32_t bidx[64];
#pragma unroll
        for (int j = 0; j < 64; ++j) {
            bvals[j] = -INFINITY;
            bidx[j] = 0;
        }
        for (uint32_t t = 0; t < (uint32_t)blockDim.x; ++t) {
            const uint32_t tb = t * k;
            for (uint32_t j = 0; j < k; ++j) {
                topk_insert(block_vals[tb + j], block_idx[tb + j], bvals, bidx, (int)k);
            }
        }
        for (uint32_t j = 0; j < k; ++j) {
            out_idx[j] = bidx[j];
        }
    }
}

