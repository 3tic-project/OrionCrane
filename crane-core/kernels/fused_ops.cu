/**
 * Fused CUDA kernels for Crane transformer inference.
 *
 * Targets: sm_80+ (Ampere & newer, bf16 support)
 *
 * This file is intentionally only an aggregation unit. Kernel definitions are
 * grouped under `kernels/fused_ops/` so paged-KV and paged-attention work can
 * evolve without growing a single monolithic CUDA source file. Keep this unit
 * touched when changing exported kernels in included headers so PTX cache refreshes.
 * Sampling currently uses an exact BF16 radix-select path for batched top-k/top-p,
 * scans the vocabulary in three radix/gather passes with a tunable block width,
 * keeps its u32 output device-resident for the next decode round, and retains
 * the previous insertion-list kernel for opt-in A/B validation.
 */

#include "fused_ops/common.cuh"
#include "fused_ops/basic_ops.cuh"
#include "fused_ops/topk.cuh"
#include "fused_ops/sampling.cuh"
#include "fused_ops/paged_kv.cuh"
#include "fused_ops/rope.cuh"
