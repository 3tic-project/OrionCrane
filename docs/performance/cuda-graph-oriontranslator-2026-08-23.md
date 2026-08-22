# CUDA Graph decode: OrionTranslator A/B (2026-08-23)

Hardware and workload:

- GPU0: RTX 4090 24 GB, no foreign compute process
- Qwen3-1.7B BF16 checkpoint
- OrionTranslator `context-glossary` prompt, 15 lines/request
- 32 concurrent clients, 32 measured requests, two warmups
- Exact 1/2/4/8/16/32/64 graph buckets

## Deterministic greedy control

| Path | Wall time | Completion tok/s | JSONL valid | Engine batch-decode time |
| --- | ---: | ---: | ---: | ---: |
| eager | 9.881 s | 852.7 | 32/32 | 7.365 s |
| graph before bucket gating | 10.034 s | 839.1 | 32/32 | 7.523 s |
| graph with bucket gating | 9.173 s | 919.6 | 32/32 | 6.102 s |

The old graph path forced every ragged batch through fixed-width metadata and
stable host uploads even when its size had no graph bucket. Restricting the
fixed-width path to configured buckets recovered the eager device-token path
for all other batch sizes. It also removed duplicate fallback accounting: the
final run reports 352 no-bucket misses and 352 fallbacks, rather than 2x.

## Production sampling decision

The real OrionTranslator settings (`temperature=0.7`, `top_p=0.9`, `top_k=20`)
do not use captured greedy sampling. In a 64-request run, graph capture remained
slower in total batch-decode time (12.521 s vs 11.418 s) and increased wall-time
variance as active rows became ragged. All 64 outputs remained valid JSONL, so
this is a performance decision rather than a correctness fallback.

CUDA Graph therefore remains opt-in. The bucket gate is retained because it is
a strict improvement for graph users and has no effect while the default master
switch is off.
