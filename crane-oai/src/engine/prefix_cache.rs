//! Admission-controlled prompt-prefix KV cache.
//!
//! Entries are created only after the engine observes another queued prompt
//! sharing a sufficiently long prefix. This matters for OrionTranslator:
//! context+glossary prompts begin with changing context, while context-free
//! batches and retry requests can share a large glossary/instruction prefix.

use std::collections::VecDeque;

use candle_core::{Result, Tensor};

use super::sequence::kv_cache_bytes;

pub type LayerKvCaches = Vec<Option<(Tensor, Tensor)>>;

struct PrefixCacheEntry {
    tokens: Vec<u32>,
    caches: LayerKvCaches,
    bytes: u64,
}

pub struct PrefixCache {
    enabled: bool,
    min_tokens: usize,
    max_entries: usize,
    max_bytes: u64,
    used_bytes: u64,
    entries: VecDeque<PrefixCacheEntry>,
}

impl PrefixCache {
    pub fn new(enabled: bool, min_tokens: usize, max_entries: usize, max_bytes: u64) -> Self {
        Self {
            enabled,
            min_tokens: min_tokens.max(1),
            max_entries: max_entries.max(1),
            max_bytes,
            used_bytes: 0,
            entries: VecDeque::new(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled && self.max_bytes > 0
    }

    pub fn min_tokens(&self) -> usize {
        self.min_tokens
    }

    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }

    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Restore the longest cached prefix, moving the entry to MRU position.
    pub fn lookup(&mut self, prompt: &[u32]) -> Option<(usize, LayerKvCaches)> {
        if !self.enabled() || prompt.len() <= self.min_tokens {
            return None;
        }
        let index = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| {
                entry.tokens.len() < prompt.len() && prompt.starts_with(&entry.tokens)
            })
            .max_by_key(|(_, entry)| entry.tokens.len())
            .map(|(index, _)| index)?;
        let entry = self.entries.remove(index)?;
        let hit = (entry.tokens.len(), entry.caches.clone());
        self.entries.push_back(entry);
        Some(hit)
    }

    /// Copy an exact-width prefix out of model-owned growable KV buffers.
    /// The contiguous copies are immutable and safe to share: restoring one
    /// forces the model to grow before appending the uncached suffix.
    pub fn insert(
        &mut self,
        tokens: &[u32],
        model_caches: &[Option<(Tensor, Tensor)>],
    ) -> Result<bool> {
        if !self.enabled() || tokens.len() < self.min_tokens {
            return Ok(false);
        }
        if self.entries.iter().any(|entry| entry.tokens == tokens) {
            return Ok(false);
        }

        let prefix_len = tokens.len();
        let mut caches = Vec::with_capacity(model_caches.len());
        for cache in model_caches {
            let copied = match cache {
                Some((k, v)) if k.dim(2)? >= prefix_len && v.dim(2)? >= prefix_len => Some((
                    k.narrow(2, 0, prefix_len)?.force_contiguous()?,
                    v.narrow(2, 0, prefix_len)?.force_contiguous()?,
                )),
                _ => return Ok(false),
            };
            caches.push(copied);
        }
        let bytes = kv_cache_bytes(&caches);
        if bytes == 0 || bytes > self.max_bytes {
            return Ok(false);
        }

        while self.entries.len() >= self.max_entries
            || self.used_bytes.saturating_add(bytes) > self.max_bytes
        {
            let Some(evicted) = self.entries.pop_front() else {
                break;
            };
            self.used_bytes = self.used_bytes.saturating_sub(evicted.bytes);
        }
        self.entries.push_back(PrefixCacheEntry {
            tokens: tokens.to_vec(),
            caches,
            bytes,
        });
        self.used_bytes = self.used_bytes.saturating_add(bytes);
        Ok(true)
    }
}

pub fn common_prefix_len(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn caches(seq_len: usize) -> LayerKvCaches {
        let tensor =
            Tensor::zeros((1, 1, seq_len, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        vec![Some((tensor.clone(), tensor))]
    }

    #[test]
    fn common_prefix_stops_at_first_difference() {
        assert_eq!(common_prefix_len(&[1, 2, 3, 4], &[1, 2, 9, 4]), 2);
    }

    #[test]
    fn lookup_uses_longest_prefix_and_never_entire_prompt() {
        let mut cache = PrefixCache::new(true, 2, 4, 1 << 20);
        cache.insert(&[1, 2], &caches(4)).unwrap();
        cache.insert(&[1, 2, 3], &caches(4)).unwrap();
        assert_eq!(cache.lookup(&[1, 2, 3, 4]).unwrap().0, 3);
        // The entire three-token entry is ineligible without cached logits,
        // so lookup safely falls back to the shorter two-token entry.
        assert_eq!(cache.lookup(&[1, 2, 3]).unwrap().0, 2);
    }

    #[test]
    fn insertion_honors_entry_limit() {
        let mut cache = PrefixCache::new(true, 2, 1, 1 << 20);
        cache.insert(&[1, 2], &caches(3)).unwrap();
        cache.insert(&[4, 5], &caches(3)).unwrap();
        assert_eq!(cache.len(), 1);
        assert!(cache.lookup(&[1, 2, 3]).is_none());
        assert!(cache.lookup(&[4, 5, 6]).is_some());
    }
}
