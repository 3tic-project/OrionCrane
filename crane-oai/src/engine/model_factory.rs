//! Qwen3-only model factory.
//!
//! Auto-detection accepts Qwen3 configs and otherwise fails early with a clear
//! error.

use anyhow::Result;
use candle_core::{DType, Device};
use serde::Deserialize;
use std::path::Path;

use super::backend::{ModelBackend, Qwen3Backend};
use crate::chat_template::{AutoChatTemplate, ChatTemplateProcessor, Qwen3ChatTemplate};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    Auto,
    Qwen3,
}

impl ModelType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "qwen3" => Self::Qwen3,
            _ => Self::Auto,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Qwen3 => "qwen3",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelFormat {
    Auto,
    Safetensors,
    Gguf,
}

impl ModelFormat {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "safetensors" => Self::Safetensors,
            "gguf" => Self::Gguf,
            _ => Self::Auto,
        }
    }
}

#[derive(Deserialize, Default)]
struct HfConfig {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
}

pub fn detect_model_type(model_path: &str) -> Result<ModelType> {
    let path = Path::new(model_path);
    let config_path = if path.is_file() {
        path.parent().map(|p| p.join("config.json"))
    } else {
        Some(path.join("config.json"))
    };

    if let Some(config_path) = config_path {
        if let Ok(data) = std::fs::read(&config_path) {
            let config: HfConfig = serde_json::from_slice(&data)?;

            if config
                .model_type
                .as_deref()
                .map(|model_type| model_type.eq_ignore_ascii_case("qwen3"))
                .unwrap_or(false)
            {
                return Ok(ModelType::Qwen3);
            }

            if config
                .architectures
                .as_ref()
                .map(|architectures| {
                    architectures
                        .iter()
                        .any(|arch| arch.to_lowercase().contains("qwen3"))
                })
                .unwrap_or(false)
            {
                return Ok(ModelType::Qwen3);
            }
        }
    }

    if model_path.to_lowercase().contains("qwen3") {
        Ok(ModelType::Qwen3)
    } else {
        anyhow::bail!(
            "only Qwen3 inference is supported; pass --model-type qwen3 or use a Qwen3 model path"
        )
    }
}

pub fn resolve(model_type: ModelType, model_path: &str) -> Result<ModelType> {
    match model_type {
        ModelType::Auto => detect_model_type(model_path),
        ModelType::Qwen3 => Ok(ModelType::Qwen3),
    }
}

pub fn create_backend(
    model_type: ModelType,
    model_path: &str,
    device: &Device,
    dtype: &DType,
    format: ModelFormat,
) -> Result<Box<dyn ModelBackend>> {
    let model_type = resolve(model_type, model_path)?;
    tracing::info!("Creating backend: {:?}", model_type);

    match model_type {
        ModelType::Qwen3 => {
            let qwen3_format = match format {
                ModelFormat::Safetensors => crane_core::models::qwen3::ModelFormat::Safetensors,
                ModelFormat::Gguf => crane_core::models::qwen3::ModelFormat::Gguf,
                ModelFormat::Auto => crane_core::models::qwen3::ModelFormat::Auto,
            };
            Ok(Box::new(Qwen3Backend::new_with_format(
                model_path,
                device,
                dtype,
                qwen3_format,
            )?))
        }
        ModelType::Auto => unreachable!(),
    }
}

pub fn create_chat_template(model_path: &str) -> Box<dyn ChatTemplateProcessor> {
    match AutoChatTemplate::new(model_path) {
        Ok(template) => Box::new(template),
        Err(err) => {
            tracing::warn!("Failed to load chat template: {err}; using Qwen3 fallback");
            Box::new(Qwen3ChatTemplate)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_type_from_str_accepts_qwen3() {
        assert_eq!(ModelType::from_str("qwen3"), ModelType::Qwen3);
        assert_eq!(ModelType::from_str("QWEN3"), ModelType::Qwen3);
    }

    #[test]
    fn model_type_from_str_non_qwen3_is_auto() {
        assert_eq!(ModelType::from_str("other"), ModelType::Auto);
        assert_eq!(ModelType::from_str("unsupported"), ModelType::Auto);
    }

    #[test]
    fn model_format_from_str() {
        assert_eq!(
            ModelFormat::from_str("safetensors"),
            ModelFormat::Safetensors
        );
        assert_eq!(ModelFormat::from_str("gguf"), ModelFormat::Gguf);
        assert_eq!(ModelFormat::from_str("other"), ModelFormat::Auto);
    }

    #[test]
    fn detect_path_heuristic_qwen3() {
        let result = detect_model_type("/models/Qwen3-1.7B").unwrap();
        assert_eq!(result, ModelType::Qwen3);
    }

    #[test]
    fn detect_path_heuristic_rejects_non_qwen3() {
        let result = detect_model_type("/models/unsupported-model");
        assert!(result.is_err());
    }

    #[test]
    fn resolve_explicit_qwen3() {
        let result = resolve(ModelType::Qwen3, "/models/whatever").unwrap();
        assert_eq!(result, ModelType::Qwen3);
    }
}
