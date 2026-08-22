# Async token D2H experiment (OrionTranslator, 2026-08-23)

## Change under test

- Added reusable readback-oriented pinned host allocations to the local cudarc fork.
- Fixed `CudaStream::memcpy_dtoh` to copy the source length when the reusable host buffer has spare capacity.
- Added a reusable pinned token buffer for batched greedy and top-k/top-p sampling.
- `CRANE_PINNED_TOKEN_D2H=1` enables the path. It remains disabled by default.

The copy is enqueued asynchronously and completion is tracked by the pinned buffer's CUDA event. The CPU must still wait before it can update repetition-penalty history, detect EOS, and send the token.

## Correctness

The normal OrionTranslator context-plus-glossary workload completed 64/64 requests with valid JSONL while the pinned path was enabled. The CUDA build and release check also passed.

## A/B result

GPU0, RTX 4090, production sampling (`temperature=0.7`, `top_p=0.9`, `top_k=20`), 64 requests, concurrency 32, decode quantum 16. The fixed-length runs use 128 generated tokens per request to remove EOS-length variance.

| Path | Run 1 tok/s | Run 2 tok/s | Run 3 tok/s | Median tok/s |
|---|---:|---:|---:|---:|
| pageable `clone_dtoh` | 497.9 | 564.7 | 824.6 | 564.7 |
| reusable pinned D2H | 484.0 | 592.9 | 523.1 | 523.1 |

The host was noisy across runs, but the pinned path did not show a repeatable gain; median completion throughput was 7.4% lower. A separate natural-EOS pair was effectively neutral/noisy (814.1 versus 833.4 tok/s).

## Decision

Do not enable pinned token D2H by default. A 4-256 byte copy followed by an immediate CPU wait has too little transfer work to amortize pinned-memory/event overhead. The reusable cudarc primitive and opt-in sampler path are retained on the experiment branch for future pipelining work, but are not part of the recommended integration stack.

The next viable version of this optimization must overlap readback with useful GPU work: launch the following decode forward from the sampler's device token buffer, copy the current token on a dedicated stream, then consume the CPU token before the next sampling launch. That requires a one-token look-ahead state machine and careful rollback/extraction handling for EOS rows; simply changing host allocation type is insufficient.
