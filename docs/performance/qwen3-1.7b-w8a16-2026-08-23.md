# Qwen3-1.7B W8A16 exploration (2026-08-23)

## Scope and definition

- Model: `Orion-Qwen3-1.7B_SFT_v2608/checkpoint-55191`.
- GPU: RTX 4090 (Ada, sm_89), GPU0 only, 24 GB VRAM, 450 W power limit.
- Workload: the OrionTranslator `context-glossary` prompt, 15 Japanese lines
  per request, JSONLINE simplified-Chinese output.
- W8A16 here means persistent symmetric INT8 weights, BF16 activations and
  FP32 accumulation. Activations are never dynamically quantized.
- Transformer projections are quantized per output channel. Tied token
  embedding/lm-head, norms and all KV state remain BF16.

Candle's GGUF Q8_0 path is not used as the W8A16 implementation: it casts the
activation to F32 and CUDA QMatMul dynamically converts it to Q8_1. It is a
useful W8A8 comparison, but it does not satisfy the definition above.

The implementation is opt-in with `CRANE_QWEN3_W8A16=1`. It is intentionally
not a default because this first kernel establishes the memory/quality
trade-off but does not yet beat the production BF16 path end to end.

## Implementation

- Checkpoint BF16 projections are merged first (QKV and gate+up), then
  quantized one matrix at a time to bound host startup memory.
- Each output row uses `scale = max(abs(weight)) / 127`; signed values in
  `[-127, 127]` are stored as U8 with a bias of 128.
- Small-M decode uses a BF16 Tensor Core WMMA kernel. INT8 bytes are unpacked
  to BF16 in shared memory with vectorized `prmt` conversion and accumulated
  into FP32.
- Decode uses shape-dependent split-K plus an FP32 reduction. The split scan
  covered 1/2/4/8/16/32 slices for all four 1.7B projection shapes and
  `M=1/16/32`.
- For `M > 32`, the persistent INT8 weight is restored once to a temporary
  BF16 matrix and passed to cuBLAS. This prevents the direct kernel from
  rereading a weight tile once per 16 activation rows during prefill.
- `CRANE_W8A16_CUBLAS_M_THRESHOLD` can override the default threshold of 32.

## Correctness and memory

CUDA tests compare direct W8A16 against the scalar CPU definition at
`M=1/7/16/31/32`, including output edge tiles, and compare CUDA/CPU prefill
dequantization bit-for-bit. Both tests pass.

The real checkpoint packs 112 merged projection matrices in 6.82 seconds:

| Storage | Projection weights |
| --- | ---: |
| BF16 | 2.62 GiB |
| INT8 + FP32 row scales | 1.31 GiB |
| Saving | **1.31 GiB (49.9%)** |

Observed idle server memory fell from about 3.9 GB to 2.58 GB. With a 4G
server limit, Crane reported a 2.5G W8A16 model baseline and completed a real
OrionTranslator request without CUDA or numerical errors.

Translation-format validation remained stable:

| Run | BF16 valid | W8A16 valid | BF16 output chars | W8A16 output chars |
| --- | ---: | ---: | ---: | ---: |
| Production sampling, 64 requests | 64/64 | 64/64 | 27,869 | 27,961 |
| Greedy control, 32 requests | 32/32 | 32/32 | 13,825 | 13,866--13,915 |

The two repeated W8A16 greedy runs differ from the BF16 output size by
0.30--0.65%. This verifies the service contract and basic translation
stability; it is not a substitute for a BLEU/COMET regression set.

## Projection microbenchmark

The table uses the best scanned split-K for each shape. Times include Candle
output/workspace allocation, and event tracking was still enabled in this
standalone unit benchmark; the server disables per-tensor event tracking.

| M | Projection (K x N) | Best split | W8A16 | BF16 | W8/BF16 speed |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | QKV (2048 x 4096) | 8 | 10.11 us | 11.39 us | **1.13x** |
| 1 | O (2048 x 2048) | 8 | 7.97 us | 7.78 us | 0.98x |
| 1 | gate+up (2048 x 12288) | 4 | 19.67 us | 21.77 us | **1.11x** |
| 1 | down (6144 x 2048) | 16 | 12.75 us | 15.53 us | **1.22x** |
| 16 | QKV | 4 | 12.51 us | 13.12 us | **1.05x** |
| 16 | O | 8 | 9.75 us | 12.59 us | **1.29x** |
| 16 | gate+up | 2 | 26.93 us | 28.36 us | **1.05x** |
| 16 | down | 16 | 19.13 us | 29.21 us | **1.53x** |
| 32 | QKV | 2 | 18.88 us | 12.95 us | 0.69x |
| 32 | O | 4 | 12.69 us | 10.15 us | 0.80x |
| 32 | gate+up | 4 | 60.61 us | 24.33 us | 0.40x |
| 32 | down | 4 | 30.58 us | 19.01 us | 0.62x |

The naive unsplit prototype took 73--217 us at M=1. Split-K reduced that to
9--29 us by filling otherwise-idle SMs, but the M=32 path still lacks a
multistage global-to-shared pipeline and enough weight reuse.

## OrionTranslator A/B

Production settings are 64 measured requests, four warmups, client/server
concurrency 32, natural EOS, temperature 0.7, top-p 0.9 and top-k 20.

| Metric | BF16 | W8A16 | Change |
| --- | ---: | ---: | ---: |
| Valid JSONL | 64/64 | 64/64 | unchanged |
| Completion tok/s | 1081.2 | 616.3 | -43.0% |
| Wall time | 15.69 s | 27.73 s | +76.8% |
| Avg batch forward | 58.95 ms | 54.09 ms | **-8.2%** |
| Total batch forward | 4.72 s | 4.33 s | **-8.2%** |
| Avg prefill forward | 12.79 ms | 16.14 ms | +26.2% |
| Total batch decode | 11.65 s | 12.31 s | +5.7% |
| Batch sampling time | 5.64 s | 6.78 s | +20.1% |
| Queue wait, summed over requests | 2.30 s | 18.97 s | +16.67 s |

This aligned pair has nearly identical work counts: 17,088 vs 16,962 measured
completion tokens, 80 batch-decode calls and four sequential-decode calls in
both runs. The projection forward aggregate is 8.2% faster, but slower prefill,
sampling and admission/queue behavior dominate the service result. This is a
real production-workload regression, not just an artifact of a different
number of decode batches.

A deterministic 32-request greedy control tells a narrower but repeatable
story: BF16 measured 907.7 tok/s in 9.28 s, while two aligned W8A16 runs
measured 1056.5--1057.7 tok/s in 8.00--8.02 s, a roughly 16.4% improvement.
The divergence between greedy and production sampling is why the implementation
must remain opt-in even though the small-M kernels can beat BF16.

## Nsight evidence

Nsight Systems 2023.4.4 captured the full projection sweep:

| Kernel | Calls | Average | GPU-time share |
| --- | ---: | ---: | ---: |
| `w8a16_linear_bf16_splitk` | 3180 | 26.00 us | 65.6% |
| `w8a16_linear_bf16` | 636 | 40.75 us | 20.6% |
| `w8a16_splitk_reduce_bf16` | 3180 | 2.38 us | 6.0% |

The standalone sweep also issued 7,681 async allocations and frees. Persistent
per-layer split workspaces can remove that host/API work. Nsight Compute 2024.1
could not inject because its library requires the driver symbol
`cuDeviceRegisterAsyncNotification`; the installed 550.90.07 driver does not
export it.

## Decision and next optimization

The implementation may live on the main branch as a default-off experimental
backend. It proves that strict W8A16 halves persistent projection storage and
preserves the translation contract, but it is not ready to replace BF16
defaults. Production continues to require explicit `CRANE_QWEN3_W8A16=1` to
enable it.

The next kernel should port the mature AllSpark design already present in the
vLLM reference tree:

1. Offline N32K16 weight and scale reorder, eliminating row-major gather work.
2. Four-stage `cp.async` global-to-shared pipeline.
3. Register/`ldmatrix` unpack and `mma.m16n8k16` rather than shared-memory WMMA
   staging for every K tile.
4. Persistent split-K workspaces with fused reduction for small M.
5. Keep dequantize+cuBLAS above an empirically tuned M threshold.
6. Store packed tensors in an offline cache or quantized checkpoint so startup
   does not spend 6.8 seconds round-tripping BF16 weights through the CPU.

After that port, rerun both deterministic and sampled OrionTranslator pairs and
add a held-out translation quality set before considering W8A16 as a default.
