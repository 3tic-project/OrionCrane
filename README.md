# OrionCrane

OrionCrane 是一个面向 **Qwen3** 系列模型的高吞吐 Rust 推理引擎，提供 OpenAI 兼容与 SGLang 兼容的 HTTP API。

并将 CUDA 相关依赖切换为 **dynamic-loading** 模式，因此发布产物在构建时不需要把 CUDA 运行库静态或动态链接进可执行文件，而是在运行时按需加载。

- Linux 下会在运行时加载 `libcuda.so`、`libcublas.so`、`libcurand.so`、`libnvrtc.so` 等。
- Windows 下会在运行时加载 `nvcuda.dll`、`cublas*.dll`、`curand*.dll`、`nvrtc*.dll` 等。
- 目标机器仍然需要安装正确的 NVIDIA 驱动，并让对应 CUDA 运行库可被系统找到。

## 项目特点

- 提供可直接接入的 OpenAI 兼容与 SGLang 兼容接口。
- 内置 paged-KV、GPU gather-extract、显存压力控制等推理优化路径。
- `--max-concurrent` 与 `--decode-tokens-per-seq` 会根据启动时可用显存自动调整。
- CUDA 采用动态加载，Linux 上 `ldd` 不会直接看到 CUDA 运行库依赖，便于单独打包部署。
- 当前默认参数针对 Qwen3 1.7B BF16 的短文本/翻译类负载做过实测调优。（单卡4090最高吞吐量可达2900tokens/s）

## 目录结构

```text
OrionCrane/
├── Cargo.toml          # 工作区配置，以及 vendored 依赖的 [patch.crates-io]
├── crane-core/         # Qwen3 模型加载、KV cache、采样、CUDA kernel 等核心逻辑
├── crane-oai/          # HTTP 服务端，可执行入口
└── lib/
    ├── candle/         # mayocream/candle 的 cuda-dynamic-loading 分支，已裁剪
    ├── ug/             # mayocream/ug 的 cuda-dynamic-loading 分支，已裁剪
    └── cudarc/         # 本地 vendored cudarc，供 candle 侧使用
```

## 构建说明

### macOS

适用于 CPU / Metal 路径：

```bash
cargo build --release -p crane-oai
```

### Linux + CUDA

Linux 下构建 CUDA 版本时，仍然需要本机提供 CUDA Toolkit，用于编译 `candle-kernels`。
但最终产物运行时走的是 dynamic-loading，不会把 CUDA 运行库直接链接进二进制。

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
cargo build --release -p crane-oai --features cuda
```

可用下面的命令确认最终二进制没有直接链接 CUDA 运行库：

```bash
ldd target/release/crane-oai | grep -E 'cuda|cublas|curand|nvrtc' \
  || echo 'no CUDA libs linked (dlopen at runtime)'
```

### Windows + CUDA

Windows 端建议使用 **MSVC toolchain**。推荐准备：

- Rust stable
- Visual Studio Build Tools 2022（含 C++ 工具链）
- CUDA Toolkit 12.x
- 已安装的 NVIDIA 驱动

PowerShell 构建示例：

```powershell
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:Path = "$env:CUDA_HOME\bin;" + $env:Path
cargo build --release -p crane-oai --features cuda
```

Windows 运行时注意：

- `crane-oai.exe` 本身不应直接依赖 CUDA DLL。
- 但运行时必须能找到 NVIDIA 驱动与对应 CUDA DLL。
- 最常见的做法是：
  - 让这些 DLL 位于系统 `PATH` 中，或
  - 将所需 CUDA DLL 放到 `crane-oai.exe` 同目录。

如果你计划把 OrionCrane 打包成单独项目在 Windows 分发，推荐至少做一次本机 smoke test，确认 DLL 搜索路径与驱动版本匹配。

## 运行示例

### Linux / macOS

```bash
./target/release/crane-oai \
  --model-path /path/to/Qwen3-1.7B \
  --model-type qwen3 \
  --host 0.0.0.0 \
  --port 8000
```

### Windows PowerShell

```powershell
.\target\release\crane-oai.exe `
  --model-path D:\models\Qwen3-1.7B `
  --model-type qwen3 `
  --host 0.0.0.0 `
  --port 8000
```

启动后日志会打印当前显存快照与自动选择出的默认参数，例如：

```text
GPU memory at startup: free=19.3G / total=23.5G
Adaptive defaults: budget=19.3G (source=free VRAM) max_concurrent=28 (auto=28, user=None) decode_tokens_per_seq=32 (auto=32, user=None)
```

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-path` | 必填 | Qwen3 模型目录，或 GGUF 文件路径。 |
| `--model-type` | `auto` | 支持 `auto` 或 `qwen3`。非 Qwen3 会被拒绝。 |
| `--format` | `auto` | `auto`、`safetensors`、`gguf`。 |
| `--cpu` | false | 强制走 CPU。 |
| `--max-concurrent` | 自动 | 最大并发解码序列数。 |
| `--decode-tokens-per-seq` | 自动 | 每轮单序列解码 token 数。 |
| `--max-seq-len` | 2800 | 提示词与输出总长度上限，`0` 表示不限制。 |
| `--gpu-memory-limit` | 不设置 | 可以写绝对值如 `8G`，或比例如 `0.85`。 |

CUDA 版本默认启用空闲显存整理：当 engine 完全空闲 `120s` 后，会清理请求级 workspace，并调用 `cuMemPoolTrimTo(pool, 0)` 将 CUDA async memory pool 的空闲保留显存还给驱动。可用 `CRANE_IDLE_CUDA_MEM_TRIM_SECS=0` 关闭，或设置更短秒数用于验证。该机制不会卸载模型权重，因此目标是回到“模型加载后的启动基线”附近，而不是进程退出后的 0 占用。

## 自适应默认参数

`--max-concurrent` 与 `--decode-tokens-per-seq` 会根据“有效 GPU 预算”自动计算：

1. 如果显式指定了 `--gpu-memory-limit`，则预算为 `min(--gpu-memory-limit, 启动时空闲显存)`。
2. 如果未指定，则预算为模型加载并 warmup 之后的“启动时空闲显存”。
3. 如果是 CPU 或无法获取显存信息，则回退到中档默认值。

| GPU 预算 | `--max-concurrent` | `--decode-tokens-per-seq` |
| --- | --- | --- |
| `< 8G` | 6 | 16 |
| `8G .. 18G` | 16 | 16 |
| `>= 18G` | 28 | 32 |
| unknown / CPU | 16 | 16 |

如果你手动传入 `--max-concurrent` 或 `--decode-tokens-per-seq`，则手动值优先。

## 接口列表

### OpenAI 兼容接口

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `GET /v1/models/{model_id}`
- `POST /v1/tokenize`
- `POST /v1/detokenize`

### SGLang 兼容接口

- `POST /generate`
- `GET /model_info`
- `GET /server_info`
- `GET /health_generate`
- `POST /flush_cache`
- `POST /abort_request`

### 管理接口

- `GET /health`
- `GET /v1/stats`


## 进阶运行开关

当前默认值已经针对常见的 Qwen3 短文本推理负载做过调优，生产环境通常不需要额外设置环境变量。

完整的 paged-KV / CUDA Graph 相关开关，请参考：

- [crane-oai/README.md](crane-oai/README.md)

其中包括：

- `CRANE_PAGED_KV_NATIVE_APPEND`，默认开启（CUDA BF16）
- `CRANE_PAGED_KV_GATHER_EXTRACT`，默认开启（CUDA BF16）
- `CRANE_PAGED_KV_ATTENTION`，默认关闭，仅建议分析性能时手动开启
- `CRANE_CUDA_GRAPH_DECODE` 系列开关，默认关闭

当前 `lib/candle/Cargo.toml` 与 `lib/ug/Cargo.toml` 已按 OrionCrane 的实际用途做过裁剪，只保留本项目需要的 workspace member。

## 致谢

本项目直接使用或参考了以下开源项目，在此表示感谢：

- [Crane](https://github.com/lucasjinreal/Crane)：本项目的上游来源。OrionCrane 基于其现有推理引擎继续裁剪、重组并演进为独立仓库。
- [candle](https://github.com/huggingface/candle)：提供张量、模型执行与 CUDA/Metal 相关基础能力。本项目使用了其 fork 后的 dynamic-loading 变体。
- [ug](https://github.com/LaurentMazare/ug)：提供底层张量/编译相关能力，本项目使用了其 dynamic-loading 分支以配合 CUDA 运行时动态加载。
- [cudarc](https://github.com/coreylowman/cudarc)：提供 Rust 侧 CUDA 驱动与相关库封装，是本项目实现 CUDA dynamic-loading 的关键基础组件。
- [vLLM](https://github.com/vllm-project/vllm)：在 continuous batching、调度器设计、CUDA Graph bucket 化、KV/block 管理等方向上提供了明确的算法与工程实现参考。
- [SGLang](https://github.com/sgl-project/sglang)：本项目既兼容其原生接口，也参考了其请求结构、调度策略和部分显存/KV 管理思路。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：paged attention 元数据组织、paged decode/prefill 接口形态与后续优化路线。
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)：attention kernel、online softmax 以及高性能注意力实现方向。
