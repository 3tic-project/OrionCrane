//! Chat template formatting.
//!
//! Converts OpenAI-style `[{role, content}]` messages into the prompt string
//! expected by each model family.  Two strategies:
//!
//! * **[`AutoChatTemplate`]** — uses the Jinja `chat_template` from
//!   `tokenizer_config.json` (works for Qwen, Llama, Mistral, …).
//! * **[`Qwen3ChatTemplate`]** — hardcoded Qwen3 fallback template.

use crate::openai_api::ChatMessage;
use crane_core::autotokenizer::AutoTokenizer;

// ─────────────────────────────────────────────────────────────
//  Trait
// ─────────────────────────────────────────────────────────────

/// Formats chat messages into a model-specific prompt string.
pub trait ChatTemplateProcessor: Send + Sync {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String, String>;
}

// ─────────────────────────────────────────────────────────────
//  AutoChatTemplate (Jinja-based)
// ─────────────────────────────────────────────────────────────

/// Uses [`AutoTokenizer`]'s Jinja `chat_template` from `tokenizer_config.json`.
pub struct AutoChatTemplate {
    tokenizer: AutoTokenizer,
}

impl AutoChatTemplate {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let tokenizer = AutoTokenizer::from_pretrained(model_path, None)
            .map_err(|e| format!("Failed to load AutoTokenizer: {e}"))?;
        Ok(Self { tokenizer })
    }
}

impl ChatTemplateProcessor for AutoChatTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String, String> {
        // Build the list of {role, content} values expected by the Jinja template.
        let template_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.text_content(),
                })
            })
            .collect();

        self.tokenizer
            .apply_chat_template(&template_messages, true)
            .map_err(|e| format!("Chat template error: {e}"))
    }
}

// ─────────────────────────────────────────────────────────────
//  Qwen3ChatTemplate (hardcoded fallback)
// ─────────────────────────────────────────────────────────────

/// Hardcoded fallback chat template for Qwen3 chat models.
pub struct Qwen3ChatTemplate;

impl ChatTemplateProcessor for Qwen3ChatTemplate {
    fn apply(&self, messages: &[ChatMessage]) -> Result<String, String> {
        let mut result = String::new();
        for msg in messages {
            match msg.role.as_str() {
                "system" | "user" | "assistant" => {
                    result.push_str("<|im_start|>");
                    result.push_str(&msg.role);
                    result.push('\n');
                    result.push_str(&msg.text_content());
                    result.push_str("<|im_end|>\n");
                }
                _ => {}
            }
        }

        result.push_str("<|im_start|>assistant\n");
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai_api::ChatMessage;
    use crate::openai_api::ChatMessageContent;

    fn make_messages(pairs: &[(&str, &str)]) -> Vec<ChatMessage> {
        pairs
            .iter()
            .map(|(role, content)| ChatMessage {
                role: role.to_string(),
                content: ChatMessageContent::Text(content.to_string()),
            })
            .collect()
    }

    // ── Qwen3ChatTemplate ──

    #[test]
    fn qwen3_basic_user_message() {
        let tmpl = Qwen3ChatTemplate;
        let msgs = make_messages(&[("user", "Hello")]);
        let result = tmpl.apply(&msgs).unwrap();

        assert!(result.starts_with("<|im_start|>user\nHello<|im_end|>\n"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn qwen3_system_message_prepended() {
        let tmpl = Qwen3ChatTemplate;
        let msgs = make_messages(&[("system", "You are helpful"), ("user", "Hi")]);
        let result = tmpl.apply(&msgs).unwrap();

        assert!(result.contains("<|im_start|>system\nYou are helpful<|im_end|>\n"));
        assert!(result.contains("<|im_start|>user\nHi<|im_end|>\n"));
    }

    #[test]
    fn qwen3_multi_turn() {
        let tmpl = Qwen3ChatTemplate;
        let msgs = make_messages(&[
            ("user", "Hello"),
            ("assistant", "Hi!"),
            ("user", "How are you?"),
        ]);
        let result = tmpl.apply(&msgs).unwrap();

        assert!(result.contains("<|im_start|>assistant\nHi!<|im_end|>\n"));
        assert!(result.contains("How are you?"));
    }

    #[test]
    fn qwen3_empty_messages() {
        let tmpl = Qwen3ChatTemplate;
        let msgs: Vec<ChatMessage> = vec![];
        let result = tmpl.apply(&msgs).unwrap();

        assert_eq!(result, "<|im_start|>assistant\n");
    }

    #[test]
    fn qwen3_unknown_role_skipped() {
        let tmpl = Qwen3ChatTemplate;
        let msgs = make_messages(&[
            ("user", "Hello"),
            ("tool", "some tool output"),
            ("user", "Next"),
        ]);
        let result = tmpl.apply(&msgs).unwrap();
        assert!(!result.contains("some tool output"));
    }

    // ── ChatTemplateProcessor trait ──

    #[test]
    fn qwen3_implements_trait() {
        let proc: Box<dyn ChatTemplateProcessor> = Box::new(Qwen3ChatTemplate);
        let msgs = make_messages(&[("user", "test")]);
        assert!(proc.apply(&msgs).is_ok());
    }
}
