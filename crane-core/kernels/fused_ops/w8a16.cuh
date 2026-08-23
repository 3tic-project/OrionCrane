/**
 * Weight-only INT8 linear layer for BF16 activations.
 *
 * The packed weight is row-major [N, K]. Each byte stores a signed symmetric
 * q8 value biased by 128. One FP32 scale is stored per output row. Activations
 * remain BF16 throughout the API; weights are decoded to BF16 in shared memory
 * and accumulated with BF16 Tensor Core MMA into FP32 accumulators.
 */

#include <mma.h>

// Vectorized conversion adapted from the Apache-2.0 AllSpark W8A16 kernel in
// vLLM. Four uint8 values are converted to BF16 and the 128 storage bias is
// removed without four scalar integer-to-float conversion sequences.
__device__ __forceinline__ void w8a16_u8x4_to_bf16x4(
    const uint32_t packed, nv_bfloat162 *output) {
    float fp32[4];
    uint32_t *bits = reinterpret_cast<uint32_t *>(fp32);
    asm volatile(
        "prmt.b32 %0, %4, 0x4B000000, 0x7650;"
        "prmt.b32 %1, %4, 0x4B000000, 0x7651;"
        "prmt.b32 %2, %4, 0x4B000000, 0x7652;"
        "prmt.b32 %3, %4, 0x4B000000, 0x7653;"
        : "=r"(bits[0]), "=r"(bits[1]), "=r"(bits[2]), "=r"(bits[3])
        : "r"(packed));
    fp32[0] -= 8388736.0f;
    fp32[1] -= 8388736.0f;
    fp32[2] -= 8388736.0f;
    fp32[3] -= 8388736.0f;

    uint32_t *bf16 = reinterpret_cast<uint32_t *>(output);
    asm volatile(
        "prmt.b32 %0, %2, %3, 0x7632;"
        "prmt.b32 %1, %4, %5, 0x7632;"
        : "=r"(bf16[0]), "=r"(bf16[1])
        : "r"(bits[0]), "r"(bits[1]), "r"(bits[2]), "r"(bits[3]));
}

extern "C" __global__ void w8a16_linear_bf16(
    const nv_bfloat16 *__restrict__ x,
    const uint8_t *__restrict__ weight,
    const float *__restrict__ scales,
    nv_bfloat16 *__restrict__ output,
    int m,
    int k,
    int n) {
    constexpr int TILE = 16;

    // One warp computes one 16x16 output tile. Keeping one warp per block
    // provides enough independent blocks for the decode-sized N dimensions
    // used by Qwen3 while avoiding cross-warp synchronization.
    __shared__ nv_bfloat16 x_tile[TILE * TILE];
    __shared__ nv_bfloat16 w_tile[TILE * TILE];
    __shared__ float out_tile[TILE * TILE];

    const int lane = threadIdx.x;
    const int m0 = blockIdx.y * TILE;
    const int n0 = blockIdx.x * TILE;

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a,
        TILE,
        TILE,
        TILE,
        nv_bfloat16,
        nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b,
        TILE,
        TILE,
        TILE,
        nv_bfloat16,
        nvcuda::wmma::col_major>
        b_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator,
        TILE,
        TILE,
        TILE,
        float>
        acc_frag;
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < k; k0 += TILE) {
        // Each lane loads half of one activation row as a single 128-bit
        // transaction. Qwen3 projection K dimensions are multiples of 16.
        const int row = lane / 2;
        const int half_row = lane % 2;
        const int x_row = m0 + row;
        nv_bfloat16 *x_dst = x_tile + row * TILE + half_row * 8;
        if (x_row < m) {
            *reinterpret_cast<uint4 *>(x_dst) = *reinterpret_cast<const uint4 *>(
                x + x_row * k + k0 + half_row * 8);
        } else {
            *reinterpret_cast<uint4 *>(x_dst) = make_uint4(0, 0, 0, 0);
        }

        // Stored as [N, K]. WMMA's column-major B view represents the
        // mathematical [K, N] matrix without a global-memory transpose.
        const int w_row = n0 + row;
        uint2 packed = make_uint2(0x80808080u, 0x80808080u);
        if (w_row < n) {
            packed = *reinterpret_cast<const uint2 *>(
                weight + w_row * k + k0 + half_row * 8);
        }
        nv_bfloat162 converted[4];
        w8a16_u8x4_to_bf16x4(packed.x, converted);
        w8a16_u8x4_to_bf16x4(packed.y, converted + 2);
        const nv_bfloat162 scale = __bfloat162bfloat162(
            w_row < n ? __float2bfloat16(scales[w_row]) : __float2bfloat16(0.0f));
#pragma unroll
        for (int i = 0; i < 4; ++i) converted[i] = __hmul2(converted[i], scale);
        *reinterpret_cast<uint4 *>(w_tile + row * TILE + half_row * 8) =
            *reinterpret_cast<uint4 *>(converted);
        __syncwarp();

        nvcuda::wmma::load_matrix_sync(a_frag, x_tile, TILE);
        nvcuda::wmma::load_matrix_sync(b_frag, w_tile, TILE);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncwarp();
    }

    nvcuda::wmma::store_matrix_sync(
        out_tile, acc_frag, TILE, nvcuda::wmma::mem_row_major);
    __syncwarp();

#pragma unroll
    for (int i = lane; i < TILE * TILE; i += WARP_SIZE) {
        const int tile_row = i / TILE;
        const int tile_col = i % TILE;
        const int out_row = m0 + tile_row;
        const int out_col = n0 + tile_col;
        if (out_row < m && out_col < n) {
            output[out_row * n + out_col] = __float2bfloat16(out_tile[i]);
        }
    }
}

extern "C" __global__ void w8a16_linear_bf16_splitk(
    const nv_bfloat16 *__restrict__ x,
    const uint8_t *__restrict__ weight,
    const float *__restrict__ scales,
    float *__restrict__ partial,
    int m,
    int k,
    int n,
    int split_k) {
    constexpr int TILE = 16;
    __shared__ nv_bfloat16 x_tile[TILE * TILE];
    __shared__ nv_bfloat16 w_tile[TILE * TILE];
    __shared__ float out_tile[TILE * TILE];

    const int lane = threadIdx.x;
    const int m0 = blockIdx.y * TILE;
    const int n0 = blockIdx.x * TILE;
    const int split = blockIdx.z;
    const int k_tiles = k / TILE;
    const int tiles_per_split = (k_tiles + split_k - 1) / split_k;
    const int first_tile = split * tiles_per_split;
    const int last_tile = min(first_tile + tiles_per_split, k_tiles);

    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_a, TILE, TILE, TILE, nv_bfloat16,
        nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::matrix_b, TILE, TILE, TILE, nv_bfloat16,
        nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<
        nvcuda::wmma::accumulator, TILE, TILE, TILE, float> acc_frag;
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    for (int tile = first_tile; tile < last_tile; ++tile) {
        const int k0 = tile * TILE;
        const int row = lane / 2;
        const int half_row = lane % 2;
        const int x_row = m0 + row;
        nv_bfloat16 *x_dst = x_tile + row * TILE + half_row * 8;
        if (x_row < m) {
            *reinterpret_cast<uint4 *>(x_dst) = *reinterpret_cast<const uint4 *>(
                x + x_row * k + k0 + half_row * 8);
        } else {
            *reinterpret_cast<uint4 *>(x_dst) = make_uint4(0, 0, 0, 0);
        }

        const int w_row = n0 + row;
        uint2 packed = make_uint2(0x80808080u, 0x80808080u);
        if (w_row < n) {
            packed = *reinterpret_cast<const uint2 *>(
                weight + w_row * k + k0 + half_row * 8);
        }
        nv_bfloat162 converted[4];
        w8a16_u8x4_to_bf16x4(packed.x, converted);
        w8a16_u8x4_to_bf16x4(packed.y, converted + 2);
        const nv_bfloat162 scale = __bfloat162bfloat162(
            w_row < n ? __float2bfloat16(scales[w_row]) : __float2bfloat16(0.0f));
#pragma unroll
        for (int i = 0; i < 4; ++i) converted[i] = __hmul2(converted[i], scale);
        *reinterpret_cast<uint4 *>(w_tile + row * TILE + half_row * 8) =
            *reinterpret_cast<uint4 *>(converted);
        __syncwarp();

        nvcuda::wmma::load_matrix_sync(a_frag, x_tile, TILE);
        nvcuda::wmma::load_matrix_sync(b_frag, w_tile, TILE);
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncwarp();
    }

    nvcuda::wmma::store_matrix_sync(
        out_tile, acc_frag, TILE, nvcuda::wmma::mem_row_major);
    __syncwarp();

    const size_t split_offset = static_cast<size_t>(split) * m * n;
#pragma unroll
    for (int i = lane; i < TILE * TILE; i += WARP_SIZE) {
        const int tile_row = i / TILE;
        const int tile_col = i % TILE;
        const int out_row = m0 + tile_row;
        const int out_col = n0 + tile_col;
        if (out_row < m && out_col < n) {
            partial[split_offset + out_row * n + out_col] = out_tile[i];
        }
    }
}

extern "C" __global__ void w8a16_splitk_reduce_bf16(
    const float *__restrict__ partial,
    nv_bfloat16 *__restrict__ output,
    int elements,
    int split_k) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float sum = 0.0f;
    for (int split = 0; split < split_k; ++split) {
        sum += partial[static_cast<size_t>(split) * elements + index];
    }
    output[index] = __float2bfloat16(sum);
}

extern "C" __global__ void w8a16_dequantize_bf16(
    const uint8_t *__restrict__ weight,
    const float *__restrict__ scales,
    nv_bfloat16 *__restrict__ output,
    int elements,
    int k) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const int q = static_cast<int>(weight[index]) - 128;
    output[index] = __float2bfloat16(static_cast<float>(q) * scales[index / k]);
}
