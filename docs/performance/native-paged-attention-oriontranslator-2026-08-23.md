# Native paged attention experiment (OrionTranslator, 2026-08-23)

## Baseline implementation

The existing decode kernel assigned one 32-thread warp to each query head and scanned every past token serially. For this Qwen3-1.7B model (`16` query heads, `8` KV heads, head dimension `128`), it also used scalar BF16 loads and precise exponentials. Page-backed attention was already disabled by default.

The current engine additionally keeps the contiguous batched KV cache up to date and copies generated KV into pages after a round. Consequently, enabling the attention kernel replaces only contiguous GQA attention; it does not yet remove the contiguous KV path.

## Changes evaluated

- Split each head's sequence scan across up to eight warps and merge independent online-softmax states in shared memory.
- Specialized the 128-wide Qwen3 head path to load four BF16 values per lane with one 64-bit transaction, following the layout used by `ref/qwen_megakernel`.
- Replaced redundant precise exponentials with one fast exponential per online-softmax update.
- Made `CRANE_PAGED_KV_ATTENTION_THREADS` tunable (`32..256`, warp aligned); the experimental default is 256.
- Cache page-table metadata and skip unchanged `indptr`, `indices`, last-page-length, and sequence-length H2D uploads.

## Profile

Nsight Systems on GPU0, an 8-request OrionTranslator decode workload, showed the original-shaped paged kernel dominating the trace:

| Kernel | GPU time | Instances | Mean |
|---|---:|---:|---:|
| `paged_attention_decode_bf16` | 76.2% / 1.801 s | 1,316 | 1.369 ms |
| leading CUTLASS BF16 GEMM | 8.8% / 0.208 s | 5,423 | 38.4 us |
| leading cuBLAS BF16 GEMV | 5.9% / 0.139 s | 3,961 | 35.1 us |
| `paged_kv_append_bf16` | 0.2% / 4.2 ms | 1,428 | 3.0 us |
| `paged_kv_gather_bf16` | 0.1% / 2.3 ms | 280 | 8.1 us |

This rules out page append/gather as the primary regression. The serial/scalar attention computation itself is the bottleneck.

## End-to-end result

All tests used the OrionTranslator context-plus-glossary prompt and GPU0. Correctness passed with 16/16 greedy JSONL and 64/64 production-sampling JSONL.

The kernel variants improved the 16-request, fixed-64-token experiment from 131.6 tok/s (one warp) to 151.4 tok/s (four warps plus vector loads) and 210.3 tok/s (eight warps plus vector loads). However, internal decode throughput remained below the contiguous implementation, and the full production workload exposed the regression clearly:

| Path | Requests | Completion tokens | Wall | Completion tok/s | Valid JSONL |
|---|---:|---:|---:|---:|---:|
| contiguous GQA | 64 | 16,994 | 18.908 s | 898.8 | 64/64 |
| optimized paged attention, 256 threads | 64 | 16,940 | 45.095 s | 375.7 | 64/64 |

The optimized paged path is 58.2% slower on the actual workload.

## Decision and next design

Keep `CRANE_PAGED_KV_ATTENTION` disabled by default and do not include this branch in the recommended integration stack. The optimized kernel and metadata cache remain isolated on `perf/native-paged-attention` for continued long-context experiments.

A competitive native implementation needs a structural change rather than further tuning of this kernel:

1. Make pages authoritative during decode and write current K/V directly from each attention layer, eliminating the duplicate contiguous KV update and post-round append.
2. Use a partitioned FlashAttention-style kernel (or FlashInfer-compatible implementation) with tensor-core QK work, GQA KV reuse, and a second-stage partition reduction for long contexts.
3. Materialize contiguous KV only for an explicit fallback or prefix-cache export, not every decode round.

