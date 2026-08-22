# OrionTranslator prompt prefix KV cache (2026-08-23)

## Workload constraint

The Orion SFT prompt order is dynamic context, optional glossary, instruction,
then translation JSONL. Consequently the normal `context-glossary` workload
does not have a useful prefix: in a 32-request GPU run the admission controller
recorded 34 lookups, zero entries, zero hits, and 32/32 valid outputs. It adds
no persistent GPU memory to this dominant workload.

Context-free requests with the same glossary do share a useful prefix. Crane
therefore admits an immutable KV entry only when another live request actually
shares at least 256 tokens. At least one suffix token is always recomputed so
fresh logits are available. Entries are exact-width device copies, capped by a
four-entry/256 MiB LRU budget. With an explicit GPU memory limit, that byte cap
is automatically reduced to at most one quarter of the request-KV budget.

## GPU A/B

RTX 4090 GPU0, Qwen3-1.7B BF16, 16 requests, concurrency 8, 15 lines/request,
Orion sampling (`temperature=0.7`, `top_p=0.9`, `top_k=20`). The benchmark used
`--glossary-repeat 16`, producing a 678-token shared prefix.

| Metric | Cache off | Cache on | Change |
| --- | ---: | ---: | ---: |
| average prefill forward | 20.275 ms | 10.840 ms | -46.5% |
| average complete prefill step | 49.332 ms | 25.959 ms | -47.4% |
| wall time | 9.087 s | 8.183 s | -10.0% |
| completion throughput | 466.6 tok/s | 521.5 tok/s | +11.8% |
| valid JSONL | 16/16 | 16/16 | unchanged |

The entry used 77,758,464 bytes (74.2 MiB); 16 requests reused 10,848 prompt
tokens. A separate 67-token experiment did not improve prefill latency, which
is why the production minimum is 256 rather than 64.
