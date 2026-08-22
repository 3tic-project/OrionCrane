# Ragged scheduling quantum: OrionTranslator A/B (2026-08-23)

## Finding

The existing lazy-row policy is correct for Qwen3-1.7B: a profiled 32-request
run forwarded 8,844 batch rows for 8,380 live tokens, only 5.25% inactive-row
work. Forward time was also nearly flat from batch 2 through batch 19, so
eagerly rebuilding a smaller batch whenever a row finishes would trade a small
amount of compute for additional page gather, KV setup, and extraction.

The larger issue was the scheduling quantum. With 32 tokens per sequence, an
early batch of only 2/6/10 rows occupied the engine for a full quantum before
newly arrived requests could prefill. A 16-token quantum forms useful decode
cohorts sooner while retaining enough rounds to amortize KV setup. Eight tokens
caused excessive setup/gather churn.

## GPU results

RTX 4090 GPU0, Qwen3-1.7B BF16, OrionTranslator `context-glossary`, 15 lines
per request, client concurrency 32. CUDA Graph capture and paged attention were
off; native paged append/gather remained on.

### Deterministic greedy control (32 requests)

| Metric | Quantum 32 | Quantum 16 | Change |
| --- | ---: | ---: | ---: |
| wall time | 9.709 s | 9.180 s | -5.4% |
| completion throughput | 868.1 tok/s | 917.6 tok/s | +5.7% |
| batch decode time | 7.765 s | 7.045 s | -9.3% |
| average TTFT | 106.9 ms | 78.7 ms | -26.4% |
| valid JSONL | 32/32 | 32/32 | unchanged |

### Production sampling (64 requests)

| Metric | Quantum 32 | Quantum 16 | Change |
| --- | ---: | ---: | ---: |
| wall time | 16.571 s | 14.452 s | -12.8% |
| completion throughput | 1023.6 tok/s | 1167.3 tok/s | +14.0% |
| average TTFT | 230.3 ms | 151.0 ms | -34.4% |
| valid JSONL | 64/64 | 64/64 | unchanged |

A 32-request production run at quantum 8 took 10.638 s versus 8.454 s at
quantum 16. A 500 us scheduler wait captured zero additional arrivals across
49 waits and remains disabled. The production default is therefore 16 across
all VRAM tiers; VRAM still determines `max_concurrent` independently.

`/v1/stats` now reports batch decode rounds, forwarded rows, inactive rows, and
the inactive-row ratio so future workloads can revisit the lazy-compaction
decision using direct evidence.
