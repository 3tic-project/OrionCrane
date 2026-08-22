#!/usr/bin/env python3
"""Reproducible OpenAI-compatible benchmark for OrionTranslator prompts.

The prompt builder and sampling defaults mirror OrionTranslator's `alnilam`
client instead of using a generic chat prompt. The script intentionally uses
only the Python standard library so it runs on inference hosts unchanged.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import math
import pathlib
import statistics
import time
import urllib.error
import urllib.request


CORPUS = (
    "窓の外では、細い雨が街灯の光を銀色に染めていた。",
    "遅かったじゃない、と彼女は振り返らずに言った。",
    "僕は濡れた鞄を机の脇に置き、深く息を吐いた。",
    "図書館の時計は、もう九時を少し回っている。",
    "この時間まで残っている生徒は、僕たち二人だけだった。",
    "例の手紙、読んだ？",
    "彼女の指先には、古びた封筒が挟まれていた。",
    "差出人の欄は空白で、消印さえ見当たらない。",
    "それでも僕には、その文字に見覚えがあった。",
    "三年前に姿を消した姉と、よく似た筆跡だった。",
    "偶然だよ、と答えた声は自分でも驚くほど弱かった。",
    "彼女はようやくこちらを向き、静かに首を横に振った。",
    "偶然なら、どうして震えているの？",
    "言われて初めて、封筒を持つ手が震えていることに気づいた。",
    "雨音が急に大きくなり、窓ガラスを激しく叩いた。",
    "手紙には、今夜十時に旧校舎へ来いとだけ書かれていた。",
    "罠かもしれない。それでも行かないという選択肢はなかった。",
    "私も行く、と彼女は当然のように鞄を肩へ掛けた。",
    "危険だから駄目だと言っても、聞いてはくれないだろう。",
    "僕たちは明かりを消し、誰もいない廊下へ踏み出した。",
    "旧校舎へ続く渡り廊下は、夜になると別の場所のようだった。",
    "床板が鳴るたびに、遠い記憶が胸の奥で目を覚ます。",
    "姉も最後の夜、この道を歩いたのだろうか。",
    "ねえ、あそこに誰かいる。",
    "彼女が指した先で、白い影が角を曲がった。",
    "待って、と叫んで駆け出した瞬間、校内放送が鳴り響いた。",
    "雑音の向こうから、聞き覚えのある声が僕の名前を呼んだ。",
    "足が止まり、呼吸の仕方さえ分からなくなる。",
    "その声は三年前と同じ調子で、早く逃げて、と繰り返した。",
    "次の瞬間、背後で重い扉がひとりでに閉まった。",
    "非常灯が赤く瞬き、廊下の影がゆっくり形を変えていく。",
    "彼女は僕の腕をつかみ、理科準備室へ飛び込んだ。",
    "棚の薬品瓶が揺れ、甘いような刺激臭が鼻を刺した。",
    "ここから中庭へ出られるはず、と彼女は小窓を押し開けた。",
    "冷たい風と一緒に、一枚の写真が室内へ舞い込んできた。",
    "写真には、旧校舎の前に立つ姉と幼い僕が写っていた。",
    "裏返すと、今日の日付と短い伝言が赤い字で記されている。",
    "過去を変えたいなら、時計塔の鐘を止めろ。",
    "意味を考える暇もなく、廊下から足音が近づいてきた。",
    "僕たちは顔を見合わせ、同時に小窓から雨の中へ飛び出した。",
)

GLOSSARY = "术语表：\n姉→姐姐\n旧校舎→旧校舍\n時計塔→钟楼\n理科準備室→理科准备室\n"


@dataclasses.dataclass(frozen=True)
class RequestResult:
    request_id: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    output_chars: int
    valid_jsonl: bool
    error: str | None = None


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def select_lines(start: int, count: int) -> list[str]:
    return [CORPUS[(start + offset) % len(CORPUS)] for offset in range(count)]


def build_prompt(request_id: int, batch_lines: int, mode: str) -> str:
    texts = select_lines(request_id * 7, batch_lines)
    context = select_lines(request_id * 7 - 10, 10) if "context" in mode else []
    glossary = GLOSSARY if "glossary" in mode else None
    parts: list[str] = []
    if context:
        parts.extend(("\n".join(context), "\n\n"))
    if glossary is not None:
        parts.extend((glossary, "\n"))
    if context and glossary is not None:
        instruction = "参考上文和术语表，将以下文本翻译为简体中文，使用JSONLINE格式输出翻译结果，只需输出翻译结果：\n"
    elif context:
        instruction = "参考上文信息，将以下文本翻译为简体中文，使用JSONLINE格式输出翻译结果，只需输出翻译结果：\n"
    elif glossary is not None:
        instruction = "参考术语表中的译法，将以下文本翻译为简体中文，使用JSONLINE格式输出翻译结果，只需输出翻译结果：\n"
    else:
        instruction = "将以下文本翻译为简体中文，使用JSONLINE格式输出翻译结果，只需输出翻译结果，不要额外解释：\n"
    parts.append(instruction)
    parts.extend(
        json.dumps({str(index + 1): text}, ensure_ascii=False, separators=(",", ":")) + "\n"
        for index, text in enumerate(texts)
    )
    return "".join(parts)


def estimate_max_tokens(request_id: int, batch_lines: int, mode: str) -> int:
    texts = select_lines(request_id * 7, batch_lines)
    context = select_lines(request_id * 7 - 10, 10) if "context" in mode else []
    glossary_chars = len(GLOSSARY) if "glossary" in mode else 0
    source_budget = sum(len(text) for text in texts) * 2
    structure_budget = batch_lines * 48 + 512
    context_margin = min(sum(len(text) for text in context) // 4, 1_500)
    glossary_margin = min(glossary_chars // 8, 1_000)
    return min(
        12_000,
        max(1_024, source_budget + structure_budget + context_margin + glossary_margin),
    )


def parse_jsonl_count(content: str) -> int:
    text = content.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    count = 0
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and len(value) == 1:
            count += 1
    return count


def send_request(args: argparse.Namespace, request_id: int) -> RequestResult:
    prompt = build_prompt(request_id, args.batch_lines, args.prompt_mode)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens
        or estimate_max_tokens(request_id, args.batch_lines, args.prompt_mode),
        "stream": False,
    }
    request = urllib.request.Request(
        args.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = json.load(response)
        latency = time.perf_counter() - started
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        return RequestResult(
            request_id=request_id,
            latency_s=latency,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            output_chars=len(content),
            valid_jsonl=parse_jsonl_count(content) == args.batch_lines,
        )
    except (OSError, KeyError, ValueError, urllib.error.HTTPError) as error:
        return RequestResult(
            request_id=request_id,
            latency_s=time.perf_counter() - started,
            prompt_tokens=0,
            completion_tokens=0,
            output_chars=0,
            valid_jsonl=False,
            error=str(error),
        )


def run_round(args: argparse.Namespace, count: int, offset: int) -> list[RequestResult]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(send_request, args, offset + index) for index in range(count)]
        return [future.result() for future in concurrent.futures.as_completed(futures)]


def fetch_engine_stats(endpoint: str, timeout: float) -> dict[str, object] | None:
    base = endpoint.split("/v1/", 1)[0].rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/v1/stats", timeout=timeout) as response:
            value = json.load(response)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, urllib.error.HTTPError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9633/v1/chat/completions")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--batch-lines", type=int, default=15)
    parser.add_argument(
        "--prompt-mode",
        choices=("plain", "glossary", "context", "context-glossary"),
        default="context-glossary",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0 or args.batch_lines <= 0:
        parser.error("requests, concurrency, and batch-lines must be positive")
    if args.warmup_requests:
        warmup = run_round(args, args.warmup_requests, 10_000)
        errors = [result.error for result in warmup if result.error]
        if errors:
            raise SystemExit(f"warmup failed: {errors[0]}")
    started = time.perf_counter()
    results = run_round(args, args.requests, 0)
    wall_s = time.perf_counter() - started
    successful = [result for result in results if result.error is None]
    latencies = [result.latency_s for result in successful]
    completion_tokens = sum(result.completion_tokens for result in successful)
    output_chars = sum(result.output_chars for result in successful)
    engine_stats = fetch_engine_stats(args.endpoint, min(args.timeout, 10.0))
    report = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "summary": {
            "wall_s": wall_s,
            "requests_ok": len(successful),
            "requests_failed": len(results) - len(successful),
            "jsonl_valid": sum(result.valid_jsonl for result in successful),
            "prompt_tokens": sum(result.prompt_tokens for result in successful),
            "completion_tokens": completion_tokens,
            "output_chars": output_chars,
            "completion_tokens_per_s": completion_tokens / wall_s,
            "output_chars_per_s": output_chars / wall_s,
            "latency_mean_s": statistics.fmean(latencies) if latencies else 0.0,
            "latency_p50_s": percentile(latencies, 0.50),
            "latency_p95_s": percentile(latencies, 0.95),
            "latency_p99_s": percentile(latencies, 0.99),
        },
        "engine_stats": engine_stats,
        "errors": [dataclasses.asdict(result) for result in results if result.error],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if len(successful) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
