mod chat_template;
mod engine;
mod handlers;
mod openai_api;
mod sglang_api;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::{
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use clap::Parser;
use tracing::info;

use chat_template::ChatTemplateProcessor;
use engine::model_factory::{ModelFormat, ModelType};
use engine::{EngineHandle, InferenceEngine, MemoryConfig};
use openai_api::ErrorResponse;

#[derive(Parser, Debug)]
#[command(
    name = "crane-oai",
    about = "Qwen3-only OpenAI and SGLang compatible API server with continuous batching"
)]
struct Args {
    /// Path to Qwen3 model directory or GGUF file
    #[arg(long)]
    model_path: String,

    /// Model architecture: auto or qwen3. Non-Qwen3 models are rejected.
    #[arg(long, default_value = "auto")]
    model_type: String,

    /// Model name to report in API responses (defaults to directory name)
    #[arg(long)]
    model_name: Option<String>,

    /// Host to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Use CPU even if GPU is available
    #[arg(long)]
    cpu: bool,

    /// Max concurrent sequences in decode phase. If unset, auto-tunes from
    /// the post-load GPU memory budget: <6G → 6, <10G → 16,
    /// <16G → 32, ≥16G → 64.
    #[arg(long)]
    max_concurrent: Option<usize>,

    /// Tokens to decode per sequence before switching (higher = fewer KV swaps).
    /// If unset, auto-tunes from the post-load GPU memory budget:
    /// <10G → 16, ≥10G → 32.
    #[arg(long)]
    decode_tokens_per_seq: Option<usize>,

    /// Model weight format: auto, safetensors, or gguf
    #[arg(long, default_value = "auto")]
    format: String,

    /// Maximum sequence length (prompt + completion tokens).
    /// Limits KV cache growth per sequence. 0 = unlimited (model default).
    #[arg(long, default_value_t = 2800)]
    max_seq_len: usize,

    /// GPU memory limit. Accepts absolute sizes like 8G/5120M or utilization fractions like 0.7.
    #[arg(long)]
    gpu_memory_limit: Option<String>,
}

pub struct AppState {
    pub engine: EngineHandle,
    pub model_name: String,
    pub tokenizer: tokenizers::Tokenizer,
    pub chat_template: Box<dyn ChatTemplateProcessor>,
    pub eos_token_id: Vec<u32>,
    pub server_start_time: u64,
    pub model_path: String,
    pub model_type_name: String,
    pub dtype_name: String,
    pub device_name: String,
    pub host: String,
    pub port: u16,
    pub max_concurrent: usize,
    pub decode_tokens_per_seq: usize,
    pub max_seq_len: usize,
    pub gpu_memory_limit: String,
}

pub fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1}G", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0}M", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{}B", bytes)
    }
}

pub fn make_error(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: openai_api::ErrorDetail {
                message: msg.to_string(),
                r#type: "invalid_request_error".into(),
                code: None,
            },
        }),
    )
}

/// Effective GPU budget used to derive adaptive runtime defaults.
/// Effective GPU budget used to derive adaptive runtime defaults.
///
/// Resolution order (first non-zero wins):
/// 1. `--gpu-memory-limit` (parsed into `memory_config.gpu_memory_limit_bytes`)
/// 2. *Available* (free) device VRAM at startup, queried after model load + warmup.
///    This adapts to whatever other processes are already using the GPU.
/// 3. 0 (unknown / CPU) — falls back to middle-tier defaults.
///
/// Also returns `(free, total)` for diagnostic logging.
fn effective_gpu_budget_bytes(
    memory_config: &MemoryConfig,
    _device: &crane_core::models::Device,
) -> (u64, u64, u64) {
    let (free, total) = query_gpu_free_total(_device);
    if memory_config.gpu_memory_limit_bytes > 0 {
        // User-specified limit caps the budget but never exceeds what's free.
        let budget = if free > 0 {
            memory_config.gpu_memory_limit_bytes.min(free)
        } else {
            memory_config.gpu_memory_limit_bytes
        };
        return (budget, free, total);
    }
    (free, free, total)
}

fn query_gpu_free_total(_device: &crane_core::models::Device) -> (u64, u64) {
    #[cfg(feature = "cuda")]
    {
        if let crane_core::models::Device::Cuda(_) = _device {
            if let Ok((free, total)) =
                candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
            {
                return (free as u64, total as u64);
            }
        }
    }
    (0, 0)
}

/// Adaptive `(max_concurrent, decode_tokens_per_seq)` defaults.
///
/// Tiers (gpu budget, inclusive lower bound):
/// | post-load budget | max_concurrent | decode_tokens_per_seq |
/// | ---------------- | -------------- | --------------------- |
/// | < 6G             |  6             | 16                    |
/// | 6G  ..< 10G      | 16             | 16                    |
/// | 10G ..< 16G      | 32             | 32                    |
/// | >= 16G           | 64             | 32                    |
/// | unknown / CPU    | 16             | 16  (middle tier)     |
fn adaptive_runtime_defaults(budget_bytes: u64) -> (usize, usize) {
    const G: u64 = 1u64 << 30;
    if budget_bytes == 0 {
        return (16, 16);
    }
    if budget_bytes < 6 * G {
        (6, 16)
    } else if budget_bytes < 10 * G {
        (16, 16)
    } else if budget_bytes < 16 * G {
        (32, 32)
    } else {
        (64, 32)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    info!("Loading Qwen3 model from: {}", args.model_path);

    let device = if args.cpu {
        crane_core::models::Device::Cpu
    } else {
        #[cfg(feature = "cuda")]
        {
            crane_core::models::Device::new_cuda_with_stream(0)
                .or_else(|_| crane_core::models::Device::cuda_if_available(0))?
        }
        #[cfg(not(feature = "cuda"))]
        {
            #[cfg(target_os = "macos")]
            {
                crane_core::models::Device::new_metal(0).unwrap_or(crane_core::models::Device::Cpu)
            }
            #[cfg(not(target_os = "macos"))]
            {
                crane_core::models::Device::Cpu
            }
        }
    };

    #[cfg(feature = "cuda")]
    let dtype = if args.cpu {
        crane_core::models::DType::F32
    } else {
        crane_core::models::DType::BF16
    };
    #[cfg(not(feature = "cuda"))]
    let dtype = crane_core::models::DType::F32;

    let device_name = format!("{:?}", device);
    let dtype_name = format!("{:?}", dtype);
    info!("Device: {}, dtype: {}", device_name, dtype_name);

    let requested_model_type = ModelType::from_str(&args.model_type);
    let resolved_type = engine::model_factory::resolve(requested_model_type, &args.model_path)?;
    let format = ModelFormat::from_str(&args.format);

    let mut backend = engine::model_factory::create_backend(
        resolved_type,
        &args.model_path,
        &device,
        &dtype,
        format,
    )?;

    info!("Qwen3 model loaded successfully (format: {:?})", format,);

    backend.warmup();
    info!("Model warmed up");

    let tokenizer = backend.tokenizer().clone();
    let eos_token_id = backend.eos_token_id();
    let chat_template = engine::model_factory::create_chat_template(&args.model_path);

    let mut memory_config =
        MemoryConfig::parse(args.max_seq_len, args.gpu_memory_limit.as_deref(), &device);
    memory_config.record_baseline(&device);
    let baseline_gpu = memory_config.baseline_gpu_bytes;

    // Resolve adaptive defaults for max_concurrent / decode_tokens_per_seq
    // from the effective GPU memory budget. The budget is the parsed
    // `--gpu-memory-limit` if set, otherwise the total device VRAM. This
    // lets a single binary run sensibly on 8G / 16G / 24G+ cards without
    // the operator hand-tuning each knob.
    let (effective_budget_bytes, gpu_free_bytes, gpu_total_bytes) =
        effective_gpu_budget_bytes(&memory_config, &device);
    let (auto_max_concurrent, auto_decode_tokens) =
        adaptive_runtime_defaults(effective_budget_bytes);
    let max_concurrent = args.max_concurrent.unwrap_or(auto_max_concurrent);
    let decode_tokens_per_seq = args.decode_tokens_per_seq.unwrap_or(auto_decode_tokens);
    if gpu_total_bytes > 0 {
        info!(
            "GPU memory at startup: free={} / total={}",
            format_bytes(gpu_free_bytes),
            format_bytes(gpu_total_bytes),
        );
    }
    info!(
        "Adaptive defaults: budget={} (source={}) max_concurrent={} (auto={}, user={:?}) decode_tokens_per_seq={} (auto={}, user={:?})",
        if effective_budget_bytes == 0 {
            "unknown".to_string()
        } else {
            format_bytes(effective_budget_bytes)
        },
        if memory_config.gpu_memory_limit_bytes > 0 {
            "min(--gpu-memory-limit, free)"
        } else if gpu_free_bytes > 0 {
            "free VRAM"
        } else {
            "fallback"
        },
        max_concurrent,
        auto_max_concurrent,
        args.max_concurrent,
        decode_tokens_per_seq,
        auto_decode_tokens,
        args.decode_tokens_per_seq,
    );
    info!(
        "Memory config: max_seq_len={}, gpu_limit={}, baseline_gpu={}",
        if memory_config.max_seq_len == 0 {
            "unlimited".to_string()
        } else {
            memory_config.max_seq_len.to_string()
        },
        if memory_config.gpu_memory_limit_bytes == 0 {
            "unlimited".to_string()
        } else {
            format_bytes(memory_config.gpu_memory_limit_bytes)
        },
        format_bytes(baseline_gpu),
    );

    let (engine, handle) = InferenceEngine::new(
        backend,
        max_concurrent,
        decode_tokens_per_seq,
        memory_config,
    );

    std::thread::Builder::new()
        .name("inference-engine".into())
        .spawn(move || engine.run())
        .expect("Failed to spawn engine thread");
    info!(
        "Inference engine started (max_concurrent={}, decode_tokens_per_seq={})",
        max_concurrent, decode_tokens_per_seq,
    );

    let model_name = args.model_name.unwrap_or_else(|| {
        std::path::Path::new(&args.model_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| resolved_type.display_name().to_string())
    });

    let gpu_memory_limit_display = args
        .gpu_memory_limit
        .clone()
        .unwrap_or_else(|| "unlimited".to_string());

    let state = Arc::new(AppState {
        engine: handle,
        model_name: model_name.clone(),
        tokenizer,
        chat_template,
        eos_token_id,
        server_start_time: now_epoch(),
        model_path: args.model_path.clone(),
        model_type_name: resolved_type.display_name().to_string(),
        dtype_name,
        device_name,
        host: args.host.clone(),
        port: args.port,
        max_concurrent,
        decode_tokens_per_seq,
        max_seq_len: args.max_seq_len,
        gpu_memory_limit: gpu_memory_limit_display,
    });

    let app = build_router(state.clone());

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let local_addr = listener.local_addr()?;

    let sep = "=".repeat(60);
    let sep2 = "-".repeat(60);
    println!("\n  {sep}");
    println!("  crane-oai v{} ready", env!("CARGO_PKG_VERSION"));
    println!("  {sep}");
    println!(
        "  Model   : {} ({})",
        model_name,
        resolved_type.display_name()
    );
    println!(
        "  Device  : {} | dtype: {}",
        state.device_name, state.dtype_name
    );
    println!("  Listen  : http://{local_addr}");
    if args.max_seq_len > 0 || state.gpu_memory_limit != "unlimited" {
        let seq_str = if args.max_seq_len == 0 {
            "unlimited".to_string()
        } else {
            args.max_seq_len.to_string()
        };
        println!(
            "  Memory  : seq_len={seq_str} gpu_limit={}",
            state.gpu_memory_limit
        );
    }
    println!(
        "  Batch   : max_concurrent={} decode_tokens_per_seq={}",
        max_concurrent, decode_tokens_per_seq
    );
    println!("  {sep2}");
    println!("  OpenAI-compatible API");
    println!("    POST  http://{local_addr}/v1/chat/completions");
    println!("    POST  http://{local_addr}/v1/completions");
    println!("    GET   http://{local_addr}/v1/models");
    println!("    POST  http://{local_addr}/v1/tokenize");
    println!("    POST  http://{local_addr}/v1/detokenize");
    println!("  {sep2}");
    println!("  SGLang-compatible API");
    println!("    POST  http://{local_addr}/generate");
    println!("    GET   http://{local_addr}/model_info");
    println!("    GET   http://{local_addr}/server_info");
    println!("    GET   http://{local_addr}/health_generate");
    println!("    POST  http://{local_addr}/flush_cache");
    println!("    POST  http://{local_addr}/abort_request");
    println!("  {sep2}");
    println!("  Management");
    println!("    GET   http://{local_addr}/health");
    println!("    GET   http://{local_addr}/v1/stats");
    println!("  {sep}\n");

    axum::serve(listener, app).await?;

    Ok(())
}

fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(handlers::common::health))
        .route("/v1/stats", get(handlers::common::stats))
        .route(
            "/v1/chat/completions",
            post(handlers::openai::chat_completions),
        )
        .route("/v1/completions", post(handlers::openai::completions))
        .route("/v1/models", get(handlers::openai::list_models))
        .route(
            "/v1/models/{model_id}",
            get(handlers::openai::retrieve_model),
        )
        .route("/v1/tokenize", post(handlers::openai::tokenize))
        .route("/v1/detokenize", post(handlers::openai::detokenize))
        .route("/tokenize", post(handlers::openai::tokenize))
        .route("/detokenize", post(handlers::openai::detokenize))
        .route("/generate", post(handlers::sglang::generate))
        .route("/model_info", get(handlers::sglang::model_info))
        .route("/server_info", get(handlers::sglang::server_info))
        .route("/health_generate", get(handlers::sglang::health_generate))
        .route(
            "/flush_cache",
            get(handlers::sglang::flush_cache).post(handlers::sglang::flush_cache),
        )
        .route("/abort_request", post(handlers::sglang::abort_request))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_epoch_is_reasonable() {
        let ts = now_epoch();
        assert!(ts > 1_577_836_800);
    }

    #[test]
    fn make_error_returns_correct_status() {
        let (status, Json(body)) = make_error(StatusCode::BAD_REQUEST, "test error");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body.error.message, "test error");
        assert_eq!(body.error.r#type, "invalid_request_error");
        assert!(body.error.code.is_none());
    }

    #[test]
    fn make_error_internal() {
        let (status, Json(body)) = make_error(StatusCode::INTERNAL_SERVER_ERROR, "boom");
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body.error.message, "boom");
    }

    #[test]
    fn make_error_service_unavailable() {
        let (status, _) = make_error(StatusCode::SERVICE_UNAVAILABLE, "overloaded");
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn adaptive_defaults_follow_post_load_memory_tiers() {
        const G: u64 = 1 << 30;
        assert_eq!(adaptive_runtime_defaults(0), (16, 16));
        assert_eq!(adaptive_runtime_defaults(6 * G - 1), (6, 16));
        assert_eq!(adaptive_runtime_defaults(6 * G), (16, 16));
        assert_eq!(adaptive_runtime_defaults(10 * G - 1), (16, 16));
        assert_eq!(adaptive_runtime_defaults(10 * G), (32, 32));
        assert_eq!(adaptive_runtime_defaults(16 * G - 1), (32, 32));
        assert_eq!(adaptive_runtime_defaults(16 * G), (64, 32));
        assert_eq!(adaptive_runtime_defaults(80 * G), (64, 32));
    }
}
