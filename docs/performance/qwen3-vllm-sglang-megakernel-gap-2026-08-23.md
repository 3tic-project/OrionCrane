# Qwen3 inference gap audit: Crane vs vLLM, SGLang and MegaKernel

Date: 2026-08-23

## Conclusion

Crane already has most of the obvious Qwen3 operator fusions. The remaining
production gap is primarily an execution-system problem, not a missing SiLU or
RMSNorm kernel:

1. The KV page store is a shadow of a second, authoritative contiguous cache.
2. Decode shapes and addresses are not stable enough to make CUDA Graph useful
   on the ragged production workload.
3. Candle issues hundreds of thousands of stream-ordered allocations and
   kernel launches, then the token D2H synchronizes that queued work every
   autoregressive round.
4. BF16 GEMM/GEMV still accounts for three quarters of GPU kernel time.

The recommended dependency order is therefore:

1. make paged KV authoritative and write K/V directly by slot;
2. use persistent, padded batch metadata and production CUDA Graph replay;
3. add one-step overlap for token readback and CPU scheduling;
4. tune BF16 skinny GEMM/cuBLASLt and port the second-generation packed W8A16
   kernel in parallel;
5. only then spend time on the sampler kernel or a batch-1 MegaKernel path.

The current BF16 defaults should not change based on this audit. The existing
W8A16 backend remains an opt-in experiment.

## Scope

- Model: `Orion-Qwen3-1.7B_SFT_v2608/checkpoint-55191`, BF16, 28 layers,
  hidden size 2048, intermediate size 6144, 16 query heads and 8 KV heads.
- GPU: GPU0 only, RTX 4090 24 GB, sm_89, 450 W power limit.
- Workload: OrionTranslator `context-glossary`, 15 Japanese lines per request,
  JSONLINE simplified-Chinese output, temperature 0.7, top-p 0.9, top-k 20.
- Production reference: 64 measured requests, four warmups, concurrency 32,
  16,962 completion tokens, 1081.2 tok/s and 15.69 s wall time.
- Comparison trees: `ref/vllm-main`, `ref/sglang-main`, and
  `ref/qwen_megakernel` at commit `5030e15`.

The Nsight Systems audit used 32 measured requests plus four warmups under
tracing. Tracing reduced observed throughput to 767.5 tok/s, so its wall time
is not used as a new performance baseline. Kernel shares and API counts are
used to locate work.

## Correcting the host timing attribution

The engine's `total_batch_decode_sampling_time_us` is not sampler GPU time.
Model forward enqueues asynchronous CUDA work; the sampler's synchronous D2H
then waits for that work. Consequently the host timer around sampling absorbs
much of the preceding forward. The non-profiled 64-request result reports
5.64 s in this field, but Nsight measures the radix sampler and repetition
penalty at only about 2.25% of GPU kernel time.

Any future tuning decision must use CUDA events or a CUDA timeline for GPU
operator attribution. The existing host field remains useful as the location
of the synchronization barrier, not as a sampler-kernel measurement.

## Nsight evidence

The dense interval from the first through last batched sampling kernel spans
13.089 s and contains 9.133 s of GPU kernels:

| Group | Launches | GPU time | Kernel-time share |
| --- | ---: | ---: | ---: |
| GEMM/GEMV | 207,029 | 6878.6 ms | **75.32%** |
| Attention math | 76,837 | 567.8 ms | 6.22% |
| Fused model ops | 148,178 | 475.0 ms | 5.20% |
| Copy/cast | 180,301 | 417.5 ms | 4.57% |
| Other | 59,493 | 323.7 ms | 3.54% |
| Shadow paged-KV | 10,952 | 264.6 ms | 2.90% |
| Sampling | 2,525 | 205.4 ms | 2.25% |

The same interval records:

| CUDA API | Calls | Host API time |
| --- | ---: | ---: |
| `cuMemcpyDtoHAsync_v2` | 1,262 | 2436.0 ms |
| `cuLaunchKernel` | 645,091 | 2257.1 ms |
| `cuMemAllocAsync` | 858,611 | 867.4 ms |
| `cuMemFreeAsync` | 858,554 | 648.6 ms |
| `cuMemcpyHtoDAsync_v2` | 178,304 | 649.0 ms |

The D2H payload itself takes only about 1--2 microseconds; the roughly 1.93 ms
average API duration is queued-forward synchronization. API times overlap GPU
execution and must not be added to wall time, but they expose the serialized
per-round structure.

There are 1.454 s of positive GPU gaps no larger than 100 microseconds inside
the dense interval. Most are launch/allocation/metadata bubbles between short
kernels. Production CUDA Graph and persistent buffers can attack this work;
optimizing the 2.25% sampler cannot.

Low GPU power is consistent with this profile. Small-M decode repeatedly reads
weights and crosses host launch barriers, so it is bandwidth/latency limited
rather than a large Tensor Core workload. Driving the card toward 450 W is not
an optimization objective; reducing milliseconds per useful token is.

## What Crane already matches

The current Qwen3 path already includes:

- merged QKV and merged gate/up projections;
- fused SiLU-times-up;
- fused residual add plus RMSNorm between decoder layers;
- fused BF16 Q/K RMSNorm plus indexed RoPE in decode;
- cached full RoPE cosine/sine tables;
- grouped GQA decode instead of per-head host loops;
- device-resident sampled tokens for the next decode input;
- persistent sampler metadata/output buffers;
- adaptive decode quantum and request-arrival coalescing;
- optional prefix cache, paged attention, CUDA Graph and strict W8A16 paths.

These correspond to the main dense-Qwen fusions in
`vllm/model_executor/models/qwen3.py` and
`sglang/srt/models/qwen2.py`. Reimplementing them again has little value.

## Ranked remaining work

| Priority | Work | Evidence / plausible benefit | Main dependency or risk |
| --- | --- | --- | --- |
| P0 | Authoritative paged KV with direct slot writes | Removes the 2.90% shadow path, contiguous setup/extract, and duplicate KV memory; unlocks the larger graph win | Large cache-ownership refactor; paged attention must beat the contiguous GQA path |
| P0 | Persistent padded metadata plus production CUDA Graph | 645k launches, 858k alloc/free pairs, and 1.454 s of short GPU gaps; stable greedy graphs already delivered about +8% | Requires stable page/input/output addresses and inactive-row sentinels |
| P0 | One-step overlap scheduler and async token readback | 1,262 round barriers spend 2.436 s waiting on queued work | EOS rollback, repetition history, extraction and streaming need a look-ahead state machine |
| P1 | Shape-tuned BF16 linear backend | GEMM/GEMV is 75.32%; a 10% projection improvement is material end to end | Must scan M=1..32 and all four Qwen3 projection shapes on sm_89 |
| P1 | N32K16 multistage W8A16 | Existing W8 reduces aggregate forward by 8.2% and halves projection storage, but production regresses | Needs AllSpark-style packing, `cp.async`, `ldmatrix`, direct MMA and persistent split-K workspace |
| P1 | Fuse QKV epilogue, QK norm/RoPE and KV scatter | Reduces Q/K/V materialization, copies, allocations and launches | Coupled to the authoritative page layout and graph-safe metadata |
| P2 | Chunked/mixed prefill and page-backed radix cache | Improves admission fairness and reusable-prefix workloads | Normal Orion context-first template has zero useful prefix hits |
| P2 | Multi-CTA/rejection top-k/top-p sampler | Current one-block-per-row sampler is underfilled at small ragged batches | Whole category is only 2.25% GPU time on the target workload |
| P3 | MegaKernel-style batch-1 tail | Can collapse launches and fuse LM-head argmax for singleton latency | Only 71 sequential tokens occurred in the profiled run; 4090/Qwen3-1.7B differs fundamentally from 5090/Qwen3-0.6B |
| Defer | MTP/speculative decode | Can reduce target-model steps only with a compatible, accurate draft | This checkpoint has no MTP head/config; prior MTP experiments were removed and high-batch draft bandwidth is unfavorable |

The percentages above are shares, not guaranteed speedups. Dependencies and
overlap mean their benefits are not additive.

## 1. Make paged KV the only KV cache

Crane currently updates the model's contiguous per-sequence/batch KV cache and
also appends/gathers an engine-owned page store. Paged attention is attempted
only after contiguous `update_kv_cache`, so pages cannot eliminate the
contiguous lifecycle. The current page-attention kernel also assigns one warp
to each `(row, query head)` and has no split-K/partitioned long-context path.

vLLM instead computes a persistent slot mapping and calls
`PagedAttention.write_to_paged_cache` / `reshape_and_cache`; its block tables
and slot-mapping buffers are persistent and use a pad slot for graph rows.
SGLang similarly passes token-to-KV-pool locations directly and plans
FlashInfer with `kv_indptr`, `kv_indices`, and `last_page_len` against reusable
workspace buffers.

The Crane transition should be staged:

1. allocate one page pool per layer and make page IDs the sequence state;
2. scatter newly produced K/V directly from the QKV epilogue using a device
   slot mapping;
3. read decode attention only from pages;
4. expose page-table views to prefix cache and graph metadata;
5. retain the old contiguous path only as an A/B correctness fallback, then
   remove the shadow append/gather/extract loop.

The page attention kernel should use a FlashInfer-style partitioned online
softmax for long contexts and a separately tuned short-context kernel. The
existing kernel's measured short/medium regression means merely changing
ownership without replacing attention is insufficient.

## 2. Stabilize Candle execution and capture ragged decode

Candle currently obtains a fresh `CudaSlice` for nearly every output tensor.
The CUDA async pool may recycle physical pages, but Crane still pays the host
`cuMemAllocAsync`/`cuMemFreeAsync` calls and receives changing tensor objects.
This makes graph capture difficult and contributes to launch bubbles.

vLLM's `BlockTables` preallocates GPU block tables, slot mappings and padded
input buffers. Its graph dispatcher chooses the smallest compatible captured
descriptor. SGLang preallocates FlashInfer workspaces and per-bucket KV
metadata and supports full, breakable and piecewise graphs.

Crane's current graph prototype requires an exact batch-size bucket, uses a
fixed contiguous-cache width in the graph key, and rejects paged attention.
That is why it helps stable greedy power-of-two batches but regresses the
natural-EOS production workload.

After authoritative pages, the production graph should:

- pad a real batch to the next configured bucket with a null page/slot;
- keep input IDs, positions, sequence lengths, block tables, slot mappings,
  logits and sampler output at stable addresses;
- update metadata with one or a few compact asynchronous copies/kernels;
- capture model forward while leaving top-k/top-p outside initially;
- pre-capture common buckets during warmup instead of capturing on the first
  live request;
- eventually capture the sampler or overlap it with output processing.

An eager-only Candle allocation cache/arena is also worth prototyping. It must
be explicitly single-stream or event-safe, bounded by bytes, exact-size or
size-classed, and disabled during graph retention. A general unbounded cache in
`CudaSlice::drop` would be unsafe and could silently consume the KV budget.

## 3. Pipeline token output instead of changing the copy alone

Crane already retains the sampled token on device for the next input, but the
CPU immediately performs a synchronous readback to update history, detect EOS
and stream output. The prior reusable pinned-D2H experiment was 7.4% slower at
the median because it still waited immediately.

SGLang's overlap scheduler keeps a one-step delay and uses pinned asynchronous
copies. vLLM's `AsyncOutput` starts output copy before work that can overlap it.
The corresponding Crane design is:

1. sampler writes current tokens to a persistent device buffer;
2. enqueue D2H to a ring of pinned host buffers and record an event;
3. launch the next decode from the device tokens without waiting;
4. process the previous round's EOS/history/network output while that forward
   runs;
5. discard or trim one speculative KV token for rows that stopped.

This is a scheduler/state-machine change. Re-enabling the old pinned copy
without the pipeline should not be repeated.

## 4. Improve the dominant linear path

The BF16 path uses Candle's `cublasGemmStridedBatchedEx` with the default Tensor
Op algorithm for all shapes. The profile shows repeated CUTLASS WMMA kernels;
there is no per-shape cuBLASLt heuristic cache or dedicated sm_89 skinny GEMM.

Two complementary experiments are justified:

- add a cuBLASLt backend that benchmarks/caches algorithms and workspaces for
  `(M,K,N,dtype)` during warmup, including fused residual/bias epilogues where
  applicable;
- port/adapt a vectorized skinny GEMM for M=1..16, similar in intent to vLLM's
  CuTeDSL `ShapeDynamicSkinnyGemm`, and dispatch to cuBLASLt above its crossover.

The scan must cover QKV 2048x4096, O 2048x2048, gate+up 2048x12288 and down
6144x2048 for every observed batch row count, not just M=1. End-to-end
OrionTranslator throughput, not an isolated GEMM, selects the dispatch table.

For W8A16, the first implementation proved the memory and small-M opportunity.
The next kernel should follow vLLM's AllSpark reference: offline N32K16 reorder,
multistage `cp.async`, register/`ldmatrix` unpack, direct `mma.sync`, persistent
split-K workspace and fused reduction. A packed on-disk cache should remove
startup quantization. Ada has no native INT8-by-BF16 MMA, so the kernel still
dequantizes weight tiles to BF16 before Tensor Core MMA.

## 5. Fuse around the projection boundaries

Crane fuses QK normalization and RoPE only after the merged QKV output has been
materialized and split. With the final page layout known, a QKV epilogue can:

- write Q in the attention layout;
- normalize and rotate Q/K;
- scatter K/V directly to their page slots;
- avoid separate contiguous Q/K/V copies and temporary tensors.

Likewise, SGLang's norm API can pass a quantized linear target so norm and
activation staging are fused. The same idea applies to Crane W8A16: fused
add+RMSNorm should write the activation layout consumed by the packed linear
kernel, rather than round-tripping a standalone normalized tensor.

## 6. Sampling is real but secondary

The exact BF16 radix sampler performs three full vocabulary scans in one block
per row, followed by a serial sort of at most 64 candidates. At small ragged
batches, four rows mean only four resident blocks across 128 SMs. A two-stage
multi-CTA radix select or FlashInfer-style rejection sampler would improve this
kernel, and a warp-level final sort would remove its serial tail.

However, sampler plus repetition penalty is only 2.25% of GPU kernel time here.
Even deleting it entirely cannot produce the benefit available from stable
graphs or a faster linear backend. It should be optimized after the execution
pipeline, unless a new workload returns logprobs or uses much larger top-k.

## 7. Scheduling and prefix reuse

vLLM uses a token-budget scheduler with chunked prefill; SGLang supports mixed
chunked prefill/decode and page-backed radix scheduling. Crane prefills one
whole prompt at a time and then runs a decode quantum.

Chunked/mixed prefill can reduce head-of-line blocking and keep decode batches
healthy when long prompts arrive continuously. It is less important in the
fixed synchronized benchmark, where prefill forward is a small fraction of
the decode workload, but should follow the page-table refactor.

The required OrionTranslator template starts with changing context, so the
normal `context-glossary` benchmark has zero useful shared prefix hits. A
page-backed radix cache is still superior to copied contiguous entries for
other request mixes, but it cannot turn a common suffix into a prefix hit.
Prompt reordering is outside this optimization scope because it would change
the required OrionTranslator template.

## 8. What to borrow from MegaKernel

`ref/qwen_megakernel` is a valuable latency design, but not a drop-in backend.
It hardcodes Qwen3-0.6B, batch 1, sequence length limits, CUDA 12.8/sm_120 and
RTX 5090 assumptions. Its 1036 tok/s result uses a persistent 128-block decode
kernel, atomic grid barriers, L2 weight prefetch during attention, and a fused
LM-head argmax. The RTX 5090 has a much larger L2 than the RTX 4090, while the
1.7B model's weights cannot reside in either cache as a whole.

Reusable ideas are:

- a batch-1/tail-specific persistent decode path;
- overlap attention with L2 prefetch of the next projection;
- redundant very-cheap normalization when it removes a grid barrier;
- fused LM-head plus argmax for greedy generation;
- multiple device-resident generation steps when no CPU policy is required.

The target production run used only 71 sequential tokens under profiling, so
even a perfect batch-1 path has little throughput impact at concurrency 32.
Build it only after the shared production path, or for a separate low-latency
service objective.

## Delivery gates

Each major item should pass the same gates before becoming a default:

1. GPU0 is idle and is the only visible device.
2. OrionTranslator `context-glossary` output remains valid JSONLINE for every
   request; add BLEU/COMET or a held-out adjudicated set for quantization.
3. Compare at least 64 measured requests, four warmups and concurrency 32 with
   natural EOS, plus a fixed-length control.
4. Report completion tok/s, wall time, TTFT/P95, forwarded/inactive rows, VRAM,
   page and graph hit rates.
5. Use Nsight kernel/API attribution; do not interpret the host sampling timer
   as sampler GPU time.
6. Keep the previous path available behind an opt-in flag until two repeated
   runs show a stable win.

