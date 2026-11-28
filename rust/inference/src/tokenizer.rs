//! tiktoken-compatible byte-level BPE for inference.
//!
//! The training-side tokenizer is `rustbpe/` (unchanged). It is built as a PyO3 extension
//! module, so it can't be linked into a native binary as-is. What we reuse is its
//! *artifact*: the split regex and the mergeable ranks it produces (the same data the
//! Python side hands to `tiktoken.Encoding`). `export_checkpoint.py` writes them to
//! `tokenizer.json`; this module loads them and runs the same merge algorithm, so ids
//! match tiktoken's `encode_ordinary` exactly.

use ahash::AHashMap;
use base64::Engine as _;
use fancy_regex::Regex;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct TokenizerJson {
    pattern: String,
    /// [[base64(token bytes), rank], ...]
    mergeable_ranks: Vec<(String, u32)>,
    special_tokens: std::collections::BTreeMap<String, u32>,
    #[serde(default = "default_bos")]
    bos_token: String,
}

fn default_bos() -> String {
    "<|bos|>".into()
}

pub struct Tokenizer {
    regex: Regex,
    ranks: AHashMap<Vec<u8>, u32>,
    decoder: Vec<Option<Vec<u8>>>,
    special: AHashMap<String, u32>,
    special_decoder: AHashMap<u32, String>,
    bos_id: u32,
}

impl Tokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path.as_ref())?;
        Self::from_json(&s)
    }

    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        let j: TokenizerJson = serde_json::from_str(s)?;
        let b64 = base64::engine::general_purpose::STANDARD;
        let mut ranks = AHashMap::with_capacity(j.mergeable_ranks.len());
        let mut max_id = 0u32;
        for (tok, rank) in &j.mergeable_ranks {
            ranks.insert(b64.decode(tok)?, *rank);
            max_id = max_id.max(*rank);
        }
        for &id in j.special_tokens.values() {
            max_id = max_id.max(id);
        }
        let mut decoder = vec![None; max_id as usize + 1];
        for (bytes, &id) in &ranks {
            decoder[id as usize] = Some(bytes.clone());
        }
        let special: AHashMap<String, u32> = j.special_tokens.into_iter().collect();
        let special_decoder = special.iter().map(|(k, &v)| (v, k.clone())).collect();
        let bos_id = *special
            .get(&j.bos_token)
            .ok_or_else(|| anyhow::anyhow!("bos token {} not in special tokens", j.bos_token))?;
        Ok(Self { regex: Regex::new(&j.pattern)?, ranks, decoder, special, special_decoder, bos_id })
    }

    pub fn vocab_size(&self) -> usize {
        self.decoder.len()
    }

    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }

    pub fn special_id(&self, name: &str) -> Option<u32> {
        self.special.get(name).copied()
    }

    /// Equivalent to `tiktoken.Encoding.encode_ordinary` (special tokens are not parsed).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for m in self.regex.find_iter(text) {
            let piece = m.expect("regex match failed").as_str().as_bytes();
            if let Some(&id) = self.ranks.get(piece) {
                out.push(id);
            } else {
                self.bpe(piece, &mut out);
            }
        }
        out
    }

    /// tiktoken's `byte_pair_merge`: repeatedly merge the adjacent pair whose
    /// concatenation has the lowest rank.
    fn bpe(&self, piece: &[u8], out: &mut Vec<u32>) {
        // parts[i] = start offset of the i-th current token
        let mut parts: Vec<usize> = (0..=piece.len()).collect();
        let rank_of = |parts: &[usize], i: usize| -> u32 {
            if i + 2 < parts.len() {
                self.ranks.get(&piece[parts[i]..parts[i + 2]]).copied().unwrap_or(u32::MAX)
            } else {
                u32::MAX
            }
        };
        let mut pair_ranks: Vec<u32> = (0..parts.len()).map(|i| rank_of(&parts, i)).collect();
        loop {
            let (mut best, mut best_i) = (u32::MAX, usize::MAX);
            for (i, &r) in pair_ranks.iter().enumerate().take(parts.len().saturating_sub(2)) {
                if r < best {
                    best = r;
                    best_i = i;
                }
            }
            if best == u32::MAX {
                break;
            }
            parts.remove(best_i + 1);
            pair_ranks.remove(best_i + 1);
            pair_ranks[best_i] = rank_of(&parts, best_i);
            if best_i > 0 {
                pair_ranks[best_i - 1] = rank_of(&parts, best_i - 1);
            }
        }
        for w in parts.windows(2) {
            out.push(self.ranks[&piece[w[0]..w[1]]]);
        }
    }

    /// Raw bytes of the tokens (special tokens rendered as their names).
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        for &id in ids {
            if let Some(Some(b)) = self.decoder.get(id as usize) {
                out.extend_from_slice(b);
            } else if let Some(name) = self.special_decoder.get(&id) {
                out.extend_from_slice(name.as_bytes());
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }
}

/// Incremental UTF-8 decoder for streaming: emits only complete characters, so a
/// multi-byte character split across tokens is not printed as replacement chars.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        self.pending.extend(tok.decode_bytes(&[id]));
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_owned();
                self.pending.clear();
                s
            }
            Err(e) => {
                let valid = e.valid_up_to();
                // An invalid (not just incomplete) sequence: flush lossily.
                if e.error_len().is_some() {
                    let s = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    return s;
                }
                let s = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                s
            }
        }
    }

    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_level() -> Tokenizer {
        let b64 = base64::engine::general_purpose::STANDARD;
        let mut ranks: Vec<(String, u32)> = (0..256u32).map(|i| (b64.encode([i as u8]), i)).collect();
        ranks.push((b64.encode(b"ab"), 256));
        ranks.push((b64.encode(b"abc"), 257));
        let j = serde_json::json!({
            "pattern": r"\s+|\S+",
            "mergeable_ranks": ranks,
            "special_tokens": {"<|bos|>": 258},
        });
        Tokenizer::from_json(&j.to_string()).unwrap()
    }

    #[test]
    fn merges_and_roundtrip() {
        let t = byte_level();
        assert_eq!(t.encode("abc"), vec![257]);
        assert_eq!(t.encode("abd"), vec![256, b'd' as u32]);
        let s = "hello abc ✓";
        assert_eq!(t.decode(&t.encode(s)), s);
        assert_eq!(t.bos_id(), 258);
        assert_eq!(t.decode(&[258]), "<|bos|>");
    }

    #[test]
    fn stream_decoder_handles_split_utf8() {
        let t = byte_level();
        let ids = t.encode("✓"); // 3 bytes -> 3 byte tokens
        let mut d = StreamDecoder::default();
        let out: String = ids.iter().map(|&i| d.push(&t, i)).collect();
        assert_eq!(out, "✓");
    }
}
