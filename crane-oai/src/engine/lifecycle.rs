//! Request lifecycle helpers for the continuous-batching engine.
//!
//! This module owns response emission, error completion, normal completion,
//! and cleanup so the main engine loop can stay focused on scheduling and model execution.

use std::sync::atomic::Ordering;

use tracing::{debug, error, info, warn};

use super::{sequence, EngineResponse, InferenceEngine};

impl InferenceEngine {
    pub(super) fn send_token(&mut self, seq_id: &str, token_id: u32) {
        let text = if let Some(stream) = self.token_streams.get_mut(seq_id) {
            match stream.next_token(token_id) {
                Ok(Some(text)) => text,
                Ok(None) => return,
                Err(err) => {
                    warn!(id = %seq_id, "Token decode error: {err}");
                    return;
                }
            }
        } else {
            return;
        };

        if let Some(seq) = self.sequences.get(seq_id) {
            if seq
                .response_tx
                .send(EngineResponse::Token { text, token_id })
                .is_err()
            {
                debug!(id = %seq_id, "Response channel closed (client disconnected)");
            }
        }
    }

    pub(super) fn send_error(&mut self, seq_id: &str, msg: &str) {
        error!(id = %seq_id, "Engine error: {msg}");
        if let Some(seq) = self.sequences.get(seq_id) {
            let _ = seq.response_tx.send(EngineResponse::Error(msg.to_string()));
        }
        self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
        self.cleanup_sequence(seq_id);
    }

    pub(super) fn finish_sequence(&mut self, seq_id: &str) {
        let remaining = self
            .token_streams
            .get(seq_id)
            .and_then(|stream| stream.decode_rest().ok().flatten())
            .unwrap_or_default();

        if !remaining.is_empty() {
            if let Some(seq) = self.sequences.get(seq_id) {
                let _ = seq.response_tx.send(EngineResponse::Token {
                    text: remaining,
                    token_id: 0,
                });
            }
        }

        if let Some(seq) = self.sequences.get(seq_id) {
            let generated_ids = &seq.tokens[seq.prompt_len..];
            let completion_tokens = seq.num_generated();
            let full_text = self
                .model
                .tokenizer()
                .decode(generated_ids, true)
                .unwrap_or_default();
            let finish_reason = seq.finish_reason().to_string();

            info!(
                id = %seq_id,
                prompt_tokens = seq.prompt_len,
                completion_tokens,
                finish_reason = %finish_reason,
                "Sequence finished",
            );

            let _ = seq.response_tx.send(EngineResponse::Finished {
                full_text,
                prompt_tokens: seq.prompt_len,
                completion_tokens,
                finish_reason,
            });

            self.stats
                .total_completion_tokens
                .fetch_add(completion_tokens as u64, Ordering::Relaxed);
            self.stats
                .completed_requests
                .fetch_add(1, Ordering::Relaxed);
        }

        self.cleanup_sequence(seq_id);
    }

    pub(super) fn cleanup_sequence(&mut self, seq_id: &str) {
        let freed = if self.active_seq_id.as_deref() == Some(seq_id) {
            self.model.active_kv_cache_bytes()
        } else if let Some(seq) = self.sequences.get(seq_id) {
            sequence::kv_cache_bytes(&seq.kv_caches)
        } else {
            0
        };
        self.tracked_kv_bytes = self.tracked_kv_bytes.saturating_sub(freed);
        self.stats
            .tracked_kv_cache_bytes
            .store(self.tracked_kv_bytes, Ordering::Relaxed);
        self.release_paged_kv_for_sequence(seq_id);

        self.sequences.remove(seq_id);
        self.token_streams.remove(seq_id);
        self.scheduler.remove(seq_id);

        if self.active_seq_id.as_deref() == Some(seq_id) {
            self.active_seq_id = None;
        }
        self.model.clear_kv_cache();

        if self.scheduler.effective_max_running.is_some() && self.scheduler.waiting.is_empty() {
            debug!("Eviction cap lifted (no waiting sequences, load subsided)");
            self.scheduler.effective_max_running = None;
        } else if let Some(cap) = self.scheduler.effective_max_running {
            // A sequence just finished — we freed KV memory. Relax the cap by
            // one slot so the scheduler can admit ONE more pending sequence.
            // The next prefill is still gated by `is_over_kv_budget`, which
            // will re-evict if the new admission actually exceeds the budget,
            // but we no longer leave the engine permanently stuck at batch=1
            // under sustained load (the previous behaviour: cap was only
            // lifted when `waiting.is_empty()`, which under continuous traffic
            // never happened, so a single transient eviction would cap the
            // engine at one running sequence forever and starve batched
            // decode entirely).
            let new_cap = cap.saturating_add(1);
            if new_cap >= self.scheduler.max_running {
                debug!("Eviction cap fully lifted (catching up with load)");
                self.scheduler.effective_max_running = None;
            } else {
                debug!(cap = new_cap, "Eviction cap relaxed by 1 after cleanup");
                self.scheduler.effective_max_running = Some(new_cap);
            }
        }

        debug!(id = %seq_id, "Sequence cleaned up");
    }
}
