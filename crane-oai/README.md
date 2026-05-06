# crane-oai

`crane-oai` is a Qwen3-only API server. It exposes OpenAI-compatible and SGLang-compatible text generation endpoints backed by the Qwen3 inference path in `crane-core`.

## Build

```bash
cargo build -p crane-oai --release
cargo build -p crane-oai --release --features cuda
```

## Run

```bash
./target/release/crane-oai \
  --model-path /path/to/Qwen3-1.7B \
  --model-type qwen3 \
  --port 8000
```

Useful options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--model-path` | required | Qwen3 model directory or GGUF file. |
| `--model-type` | `auto` | `auto` or `qwen3`; non-Qwen3 models are rejected. |
| `--format` | `auto` | `auto`, `safetensors`, or `gguf`. |
| `--cpu` | false | Force CPU execution. |
| `--max-concurrent` | auto (VRAM-tiered) | Max running decode sequences. See *Adaptive defaults* below. |
| `--decode-tokens-per-seq` | auto (VRAM-tiered) | Decode rounds per sequence before switching. See *Adaptive defaults* below. |
| `--max-seq-len` | 2800 | Prompt plus completion limit; 0 means unlimited. |
| `--gpu-memory-limit` | unset (full device VRAM) | Absolute size such as `8G` or fraction such as `0.7`. |

### Adaptive defaults

`--max-concurrent` and `--decode-tokens-per-seq` auto-tune based on the
*effective GPU budget*, resolved in this order:

1. If `--gpu-memory-limit` is set, the budget is `min(--gpu-memory-limit,
   free VRAM)`.
2. Otherwise the budget is the *available* (free) device VRAM at startup,
   queried after model load + warmup. This adapts to whatever other processes
   are already using the card.
3. On CPU or when CUDA is unavailable, defaults fall back to the middle tier.

| GPU budget | `--max-concurrent` | `--decode-tokens-per-seq` |
| --- | --- | --- |
| `< 8G`        | 6  | 16 |
| `8G .. 18G`   | 16 | 16 |
| `>= 18G`      | 28 | 32 |
| unknown / CPU | 16 | 16 |

Pass either flag explicitly to override the auto value. Both the GPU memory
snapshot and the resolved values are logged at startup:

```
GPU memory at startup: free=20.0G / total=24.0G
Adaptive defaults: budget=20.0G (source=free VRAM) max_concurrent=28 ...
```

### Performance defaults

The shipped defaults are tuned for the typical OpenAI-style translation /
short-completion workload on Qwen3 1.7B BF16. **You should not need to set
any environment variables for production deployments** — running

```bash
./target/release/crane-oai \
  --model-path /path/to/Qwen3-1.7B \
  --max-concurrent 32 \
  --gpu-memory-limit 0.85 \
  --max-seq-len 2048
```

reproduces the best-known throughput on Qwen3 1.7B (validated 2026-04-30,
RTX-class GPU, BLOCKS=200, MC=32: **6.19 s / 916 chars·s⁻¹, 100/100 ok**).

The defaults are pinned to the best-performing values for every tunable that
materially affects throughput:

| Setting | Default | Why |
| --- | --- | --- |
| `--decode-tokens-per-seq` | **auto: 16 / 16 / 32** | VRAM-tiered (see *Adaptive defaults* above). 16 is best on small/medium GPUs; 32 wins on ≥18G where the larger batch amortises setup cost. |
| `--max-concurrent` | **auto: 6 / 16 / 28** | VRAM-tiered. Larger budgets fit more concurrent sequences before the KV-cache pressure gate kicks in. |
| `--max-seq-len` | **2800** | Covers typical OpenAI-style chat / translation contexts (prompt + completion). Set to 0 for unlimited. |
| `CRANE_PAGED_KV_NATIVE_APPEND` | **on** (CUDA BF16) | Source of the Round 9 win; collapses per-token KV materialisation kernels. |
| `CRANE_PAGED_KV_GATHER_EXTRACT` | **on** (CUDA BF16) | One-shot gather kernels per layer instead of per-row per-layer. |
| `CRANE_PAGED_KV_ATTENTION` | **off** | Current paged attention kernel regresses on short/medium contexts; eager GQA wins. |
| `CRANE_PAGED_KV_BATCHED_SETUP` | **off** | Opt-in M2 batched KV setup path. Correctness validated on the Orion Qwen3 translation probe; keep opt-in while broader workloads are profiled. |
| `CRANE_BATCH_KV_RAGGED_COPY` | **on** (CUDA BF16) | Uses one fused BF16 kernel per layer for ragged batched-setup workspace copies. |
| `CRANE_CUDA_GRAPH_DECODE` | **off** | Eager forward is at parity or ~1% faster on the validated workload. |
| `CRANE_CUDA_GRAPH_DECODE_CAPTURE` | **off** | Requires the master switch; opt-in only. |
| `CRANE_CUDA_GRAPH_DECODE_WIDTH_BUCKET` | **on** | Safe with capture off; ~6–10% lift when capture is on. Leave on. |
| `CRANE_IDLE_CUDA_MEM_TRIM_SECS` | **120** | After the engine is fully idle for this many seconds, synchronize CUDA and call `cuMemPoolTrimTo(pool, 0)` so request-local high-water allocations can return to the driver. Set `0` to disable. |

Only `--max-concurrent` and `--gpu-memory-limit` are deployment-specific and
should be tuned to the available VRAM and target concurrency.

## Endpoints

OpenAI-compatible:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/tokenize`
- `POST /v1/detokenize`

SGLang-compatible:

- `POST /generate`
- `GET /model_info`
- `GET /server_info`
- `GET /health_generate`
- `POST /flush_cache`
- `POST /abort_request`

## Chat Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3",
    "messages": [
      {"role": "user", "content": "Translate to English: 今天天气很好。"}
    ],
    "max_tokens": 128,
    "temperature": 0.0
  }'
```

## Notes

- The active API surface is text generation, tokenization, model metadata, and server health.
- Qwen3 chat templates are loaded from `tokenizer_config.json` when available, with a Qwen3 fallback template.
- On CUDA BF16, paged-KV native append and GPU gather extraction are enabled by default. Decode-only paged attention is available but **off by default** because the current kernel regresses end-to-end throughput on Qwen3 short/medium translation contexts compared to the contiguous GQA path. Set `CRANE_PAGED_KV_ATTENTION=1` to opt in (and optionally tune `CRANE_PAGED_KV_ATTENTION_MIN_SEQ_LEN` for profiling); set `CRANE_PAGED_KV_NATIVE_APPEND=0` to return to the contiguous KV fallback.

## Qwen3 Runtime Flags

| Variable | Default | Purpose |
| --- | --- | --- |
| `CRANE_PAGED_KV_NATIVE_APPEND` | on for CUDA BF16 | Copy/import past and generated K/V into GPU pages. |
| `CRANE_PAGED_KV_GATHER_EXTRACT` | on for CUDA BF16 | Gather page-backed K/V into per-sequence state after a batch step. |
| `CRANE_PAGED_KV_ATTENTION` | **off** (opt-in) | Allow GPU page-backed BF16 decode attention when rows are resident and the heuristic passes. Currently regresses throughput on Qwen3 short/medium contexts; enable explicitly only when profiling the paged attention kernel. |
| `CRANE_PAGED_KV_ATTENTION_MIN_SEQ_LEN` | `1024` | Minimum max past length in the batch before the current paged attention kernel is used. |
| `CRANE_PAGED_KV_ATTENTION_MIN_ACTIVE_ROWS` | `1` | Minimum live rows before paged attention is considered. |
| `CRANE_PAGED_KV_BLOCK_SIZE` | `16` | Tokens per KV page. |
| `CRANE_PAGED_KV_PRESSURE_RESERVE_MB` | `512` | Memory headroom reserved near the GPU memory limit. |
| `CRANE_PAGED_KV_SHADOW_VALIDATE` | off | Debug-only page-store gather validation. |
| `CRANE_PROFILE` | off | Emit per-stage structured timing logs for short profiling runs. |
| `CRANE_PAGED_KV_BATCHED_SETUP` | **off** (opt-in) | M2 batched KV setup path. Publishes page-gathered batched KV for the next setup and falls back to per-row materialization when disabled. Validated on the Orion Qwen3 translation probe; keep opt-in until more workload coverage is collected. |
| `CRANE_BATCH_KV_RAGGED_COPY` | on for CUDA BF16 | Replaces `narrow + contiguous + slice_set` loops in ragged batched setup with a right-aligned BF16 copy kernel. Set `0` only for profiling the legacy rowwise path. |
| `CRANE_IDLE_CUDA_MEM_TRIM_SECS` | `120` | When no requests are active, wait this many seconds, clear request-local workspaces, synchronize CUDA, and trim the CUDA async memory pool. This returns idle pool reservations but keeps model weights/context resident; `0` disables it. |

## CUDA Graph Flags (advanced, opt-in)

Graph capture is disabled by default because the eager path is at parity or
slightly faster on the validated translation workload. The flags below are
useful when batch shapes are very stable (long completions, fixed batch
widths). All are read at server start.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CRANE_CUDA_GRAPH_DECODE` | off | Master switch for the CUDA Graph decode path. |
| `CRANE_CUDA_GRAPH_DECODE_CAPTURE` | off | Capture and replay decode graphs (requires the master switch). |
| `CRANE_CUDA_GRAPH_DECODE_WIDTH_BUCKET` | **on** | Bucket the cache width to the next power of two so successive batches share captured graphs. Safe with capture off; provides ~6–10% throughput when capture is on. |
| `CRANE_CUDA_GRAPH_DECODE_CAPTURE_SAMPLING` | off | Capture greedy argmax inside the decode graph. Only fires for greedy, no-penalty workloads. |
| `CRANE_CUDA_GRAPH_DECODE_BUCKETS` | adaptive | Override the batch-size buckets used for graph capture (e.g. `1,2,4,8,16,32`). |
| `CRANE_CUDA_GRAPH_DECODE_MAX_REPLAYS` | unbounded | Evict and recapture a graph after this many replays (debugging). |
| `CRANE_DISABLE_GPU_MEM_HARD_CHECK` | off | Bypass the hard KV-cache VRAM check; required by some CUDA Graph configurations on tight VRAM. |

Recommended graph-on configuration (only enable after validating on your
workload):

```bash
CRANE_CUDA_GRAPH_DECODE=1 \
CRANE_CUDA_GRAPH_DECODE_CAPTURE=1 \
CRANE_DISABLE_GPU_MEM_HARD_CHECK=1 \
./target/release/crane-oai --model-path ... --port 8000
```
