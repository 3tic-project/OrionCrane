# Qwen3-4B OrionTranslator CUDA optimization (2026-08-23)

## Scope

- Model: `Orion-Qwen3-4B_SFT_v2608/checkpoint-40000`, BF16.
- GPU: RTX 4090, GPU0 only, 24 GB physical VRAM, 450 W power limit.
- Workload: `benchmarks/oriontranslator_bench.py`, `context-glossary` prompt mode,
  15 Japanese input lines per request, OrionTranslator JSONLINE output contract,
  temperature 0.7, top-p 0.9, top-k 20.
- Production run: 64 measured requests, 4 warmups, client concurrency 32,
  natural EOS, server GPU limit 22G.

## Production A/B

| Configuration | Valid JSONL | Completion tok/s | Wall time | P95 latency | Avg TTFT | Observed / reported VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous default: q16 + native paged shadow | 64/64 | 622.2 | 26.99 s | 15.38 s | 207 ms | 23.4 GiB / card full |
| Disable paged shadow, q16 | 64/64 | 743.3 | 22.83 s | 15.38 s | 191 ms | 16.0–16.9 GiB |
| Disable paged shadow, q32 | 64/64 | 799.0 | 21.15 s | 10.90 s | 479 ms | 16.5 GiB |
| Shadow off, q32, 1 ms arrival coalescing | 64/64 | **826.7** | **20.56 s** | **10.15 s** | 636 ms | 16.9 GiB |

The best measured production throughput is 32.9% above the old default. The
shadow-cache change alone contributes 19.5% and removes roughly 7 GiB of peak
VRAM pressure.

The paged copy is useful when its gather path saves many small-model cache
materializations. With paged attention disabled, however, it remains a second
copy of the authoritative contiguous KV. Qwen3-4B at 32-way concurrency projects
a 4.5 GiB 1024-token shadow working set, over 25% of the 13.6 GiB dynamic budget.
The new default therefore enables it only when that projection fits; explicit
`CRANE_PAGED_KV_NATIVE_APPEND=0/1` still wins over the heuristic.

## Decode quantum and arrival coalescing

Fixed 128-token output was used to remove natural-EOS length variance while
choosing scheduler parameters:

| Configuration | Completion tok/s | Wall time | Avg effective batch rows/round |
| --- | ---: | ---: | ---: |
| q8, no wait | 415.7 | 19.71 s | 9.4 |
| q16, no wait | 436.6 | 18.76 s | 12.6 |
| q32, no wait | 452.6 | 18.10 s | 12.2 |
| q32, 500 us wait | 543.5 | 15.07 s | 15.7 |
| q32, 1000 us wait, run 1 | **685.6** | **11.95 s** | 20.8 |
| q32, 1000 us wait, repeat | 614.1 | 13.34 s | 15.9 |
| q32, 2000 us wait | 467.8 | 17.51 s | 20.3, but a long singleton tail |

One millisecond is the stable throughput compromise. Its measured cumulative
wait was only 32–65 ms per complete benchmark, but it avoids locking q32 onto
the first few HTTP requests that reach the engine. It can be disabled with
`CRANE_SCHED_WAIT_BATCH_US=0` for latency-sensitive single-request service.

The adaptive runtime tier for 10–16 GiB of free post-load budget now selects
`max_concurrent=32, decode_tokens_per_seq=32`. This is the tier selected
automatically by the 4B model on the tested 24 GB card. The >=16 GiB tier keeps
q16 because it drives 64 concurrent rows, where shorter quanta control ragged
tails and prefill admission.

## Nsight result

Nsight Systems 2024.1.1 successfully captured the q32, shadow-off path. GPU
kernel time was dominated by:

| Kernel group | GPU time share |
| --- | ---: |
| Main BF16 CUTLASS/cuBLAS GEMM | 41.2% |
| Fused BF16 GEMM | 9.0% |
| cuBLAS GEMV variants | 10.3% |
| BF16 copies | 4.4% |
| fused add + RMSNorm | 2.5% |
| batched top-k/top-p sampling | 0.8% |

The profile supports increasing useful GEMM batch width rather than spending
more work on the already-small sampling kernel. Host CUDA API time is dominated
by per-autoregressive-round D2H synchronization; removing it safely requires a
future device-resident stopping/repetition-history design, not a local copy
tweak.

## Rejected experiments

- q64: 606.0 tok/s with P95 15.96 s; long quanta amplify ragged and singleton
  tails.
- `max_concurrent=40`: reached 429 W and about 19.7 GB, but fixed-workload
  throughput was only 547.6 tok/s. Higher power was not higher useful output.
- Candle BF16 reduced-precision GEMM accumulation: 697.8 tok/s; forward time
  was unchanged and tail latency regressed. The experiment was removed.
- Forcing singleton requests through batch decode: 486.9 tok/s on the fixed
  workload; contiguous setup/extract costs outweighed the fused sampler win.
- Current native paged attention and CUDA Graph experiments remain opt-in; the
  previously validated eager contiguous path is faster for this translation
  workload.

## Recommended launch

No scheduler or paged-KV environment overrides are required:

```bash
CUDA_VISIBLE_DEVICES=0 RUST_LOG=warn ./target/release/crane-oai \
  --model-path /path/to/Orion-Qwen3-4B/checkpoint-40000 \
  --port 9633 \
  --gpu-memory-limit 22G
```

Startup should report `max_concurrent=32`, `decode_tokens_per_seq=32`, and
`adaptive paged-KV shadow-cache default enabled=false`. Explicit CLI values
continue to override the adaptive tier.
