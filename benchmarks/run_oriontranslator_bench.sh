#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_path=${MODEL_PATH:-/mnt/Shared_05_disk/home/zjp/ss33/Orion-Qwen3-1.7B_SFT_v2608/checkpoint-55191}
port=${PORT:-9633}
max_concurrent=${MAX_CONCURRENT:-64}
gpu_memory_limit=${GPU_MEMORY_LIMIT:-22G}
decode_tokens_per_seq=${DECODE_TOKENS_PER_SEQ:-}
result_path=${RESULT_PATH:-"$repo_dir/outputs/oriontranslator-$(date +%Y%m%d-%H%M%S).json"}
server_log=${SERVER_LOG:-"$repo_dir/outputs/oriontranslator-server-$(date +%Y%m%d-%H%M%S).log"}

mapfile -t foreign_pids < <(
  nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF && $1 != "[Not" { print $1 }'
)
if ((${#foreign_pids[@]} > 0)); then
  echo "GPU0 is busy (compute PIDs: ${foreign_pids[*]}), refusing to start benchmark" >&2
  exit 2
fi

mkdir -p "$repo_dir/outputs"
if [[ ${SKIP_BUILD:-0} != 1 ]]; then
  env PATH=/usr/local/cuda/bin:$PATH cargo build --release --features cuda -p crane-oai \
    --manifest-path "$repo_dir/Cargo.toml"
fi

server_args=(
  --model-path "$model_path"
  --max-concurrent "$max_concurrent"
  --port "$port"
  --gpu-memory-limit "$gpu_memory_limit"
)
if [[ -n $decode_tokens_per_seq ]]; then
  server_args+=(--decode-tokens-per-seq "$decode_tokens_per_seq")
fi

CUDA_VISIBLE_DEVICES=0 RUST_LOG=${RUST_LOG:-info} \
  "$repo_dir/target/release/crane-oai" "${server_args[@]}" \
  >"$server_log" 2>&1 &
server_pid=$!
cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid"
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "server exited during startup; see $server_log" >&2
    exit 1
  fi
  if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null

python3 "$repo_dir/benchmarks/oriontranslator_bench.py" \
  --endpoint "http://127.0.0.1:$port/v1/chat/completions" \
  --requests "${REQUESTS:-64}" \
  --warmup-requests "${WARMUP_REQUESTS:-4}" \
  --concurrency "${CONCURRENCY:-32}" \
  --batch-lines "${BATCH_LINES:-15}" \
  --prompt-mode "${PROMPT_MODE:-context-glossary}" \
  --output "$result_path" \
  "$@"

echo "result: $result_path"
echo "server log: $server_log"
