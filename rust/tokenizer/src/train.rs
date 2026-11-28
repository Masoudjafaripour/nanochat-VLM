//! BPE training. A pure-Rust port of `rustbpe/src/lib.rs` (left unchanged there):
//! same regex pre-split, same incremental merge loop with a lazy max-heap, same
//! deterministic tie-break, so the learned merges are identical. Differences: no PyO3,
//! and text is fed in batches from Rust (`Trainer::feed`) instead of a Python iterator.

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use rayon::prelude::*;
use std::cmp::Ordering;

/// nanochat's split pattern (`nanochat_vlm/tokenizer.py: SPLIT_PATTERN`).
pub const NANOCHAT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// `nanochat_vlm/tokenizer_vlm.py: SPECIAL_TOKENS_VLM`.
pub const SPECIAL_TOKENS_VLM: [&str; 10] = [
    "<|bos|>", "<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>", "<|image|>",
];

type Pair = (u32, u32);

struct Word {
    ids: Vec<u32>,
}

impl Word {
    /// Merge all non-overlapping `pair` occurrences; return pair-count deltas for this word.
    fn merge_pair(&mut self, (a, b): Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let n = self.ids.len();
        let mut out = Vec::with_capacity(n);
        let mut deltas = Vec::with_capacity(6);
        let mut i = 0;
        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                if let Some(&x) = out.last() {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if i + 2 < n {
                    let y = self.ids[i + 2];
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }
                out.push(new_id);
                i += 2;
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }
        self.ids = out;
        deltas
    }
}

#[derive(PartialEq, Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
    pos: Vec<usize>,
}

impl Ord for MergeJob {
    // Max-heap by count; ties -> smaller pair first (same as rustbpe).
    fn cmp(&self, o: &Self) -> Ordering {
        self.count.cmp(&o.count).then_with(|| o.pair.cmp(&self.pair))
    }
}

impl PartialOrd for MergeJob {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Streaming trainer: `feed` text batches (parallel regex split + counting), then `train`.
pub struct Trainer {
    regex: Regex,
    pattern: String,
    counts: AHashMap<CompactString, i32>,
}

impl Trainer {
    pub fn new(pattern: &str) -> anyhow::Result<Self> {
        Ok(Self { regex: Regex::new(pattern)?, pattern: pattern.to_string(), counts: AHashMap::new() })
    }

    pub fn feed<S: AsRef<str> + Sync>(&mut self, texts: &[S]) {
        let re = &self.regex;
        let local = texts
            .par_iter()
            .map(|s| {
                let mut m: AHashMap<CompactString, i32> = AHashMap::new();
                for mat in re.find_iter(s.as_ref()) {
                    *m.entry(CompactString::from(mat.expect("regex match failed").as_str())).or_default() += 1;
                }
                m
            })
            .reduce(AHashMap::new, |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_default() += v;
                }
                a
            });
        for (k, v) in local {
            *self.counts.entry(k).or_default() += v;
        }
    }

    /// Learn `vocab_size - 256` merges. Returns merges as (pair, new_id) in id order.
    pub fn train(self, vocab_size: u32) -> TrainedBpe {
        assert!(vocab_size >= 256, "vocab_size must be at least 256");
        let num_merges = vocab_size - 256;
        let (mut words, mut counts) = (Vec::with_capacity(self.counts.len()), Vec::with_capacity(self.counts.len()));
        for (chunk, c) in self.counts {
            words.push(Word { ids: chunk.as_bytes().iter().map(|&b| b as u32).collect() });
            counts.push(c);
        }

        // Initial pair counts and where each pair occurs (parallel).
        let (mut pair_counts, where_) = words
            .par_iter()
            .enumerate()
            .map(|(i, w)| {
                let (mut pc, mut wh) = (AHashMap::<Pair, i32>::new(), AHashMap::<Pair, AHashSet<usize>>::new());
                for p in w.ids.windows(2).map(|w| (w[0], w[1])) {
                    *pc.entry(p).or_default() += counts[i];
                    wh.entry(p).or_default().insert(i);
                }
                (pc, wh)
            })
            .reduce(
                || (AHashMap::new(), AHashMap::new()),
                |(mut a, mut aw), (b, bw)| {
                    for (k, v) in b {
                        *a.entry(k).or_default() += v;
                    }
                    for (k, s) in bw {
                        aw.entry(k).or_default().extend(s);
                    }
                    (a, aw)
                },
            );
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, pos) in where_ {
            let c = pair_counts.get(&pair).copied().unwrap_or(0);
            if c > 0 {
                heap.push(MergeJob { pair, count: c as u64, pos: pos.into_iter().collect() });
            }
        }

        let mut merges = Vec::with_capacity(num_merges as usize);
        while (merges.len() as u32) < num_merges {
            let Some(mut top) = heap.pop() else { break };
            let current = pair_counts.get(&top.pair).copied().unwrap_or(0);
            if top.count != current as u64 {
                // stale entry: refresh lazily
                top.count = current as u64;
                if top.count > 0 {
                    heap.push(top);
                }
                continue;
            }
            if top.count == 0 {
                break;
            }
            let new_id = 256 + merges.len() as u32;
            merges.push((top.pair, new_id));
            let mut updates: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            for &wi in &top.pos {
                for (pair, delta) in words[wi].merge_pair(top.pair, new_id) {
                    let d = delta * counts[wi];
                    if d != 0 {
                        *pair_counts.entry(pair).or_default() += d;
                        if delta > 0 {
                            updates.entry(pair).or_default().insert(wi);
                        }
                    }
                }
            }
            for (pair, pos) in updates {
                let c = pair_counts.get(&pair).copied().unwrap_or(0);
                if c > 0 {
                    heap.push(MergeJob { pair, count: c as u64, pos: pos.into_iter().collect() });
                }
            }
            if merges.len() % 1000 == 0 {
                log::info!("{}/{} merges", merges.len(), num_merges);
            }
        }
        TrainedBpe { pattern: self.pattern, merges }
    }
}

pub struct TrainedBpe {
    pub pattern: String,
    pub merges: Vec<(Pair, u32)>,
}

impl TrainedBpe {
    /// Token bytes -> id for every non-special token (what tiktoken calls mergeable ranks).
    pub fn mergeable_ranks(&self) -> Vec<(Vec<u8>, u32)> {
        let mut bytes: Vec<Vec<u8>> = (0..256u32).map(|b| vec![b as u8]).collect();
        for &((a, b), id) in &self.merges {
            let mut m = bytes[a as usize].clone();
            m.extend_from_slice(&bytes[b as usize]);
            debug_assert_eq!(id as usize, bytes.len());
            bytes.push(m);
        }
        bytes.into_iter().enumerate().map(|(i, b)| (b, i as u32)).collect()
    }

    /// `tokenizer.json` (the format `Tokenizer::from_json` and export_checkpoint.py use).
    /// Special tokens get ids right after the mergeable ranks, like tokenizer.py.
    pub fn to_json(&self, special_tokens: &[&str]) -> String {
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD;
        let ranks = self.mergeable_ranks();
        let offset = ranks.len() as u32;
        let ranks: Vec<(String, u32)> = ranks.into_iter().map(|(b, i)| (b64.encode(b), i)).collect();
        let special: serde_json::Map<String, serde_json::Value> =
            special_tokens.iter().enumerate().map(|(i, s)| (s.to_string(), (offset + i as u32).into())).collect();
        serde_json::json!({
            "pattern": self.pattern,
            "mergeable_ranks": ranks,
            "special_tokens": special,
            "bos_token": special_tokens.first().copied().unwrap_or("<|bos|>"),
        })
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tokenizer;

    #[test]
    fn learns_most_frequent_pair_first_and_roundtrips() {
        let mut t = Trainer::new(NANOCHAT_PATTERN).unwrap();
        t.feed(&["aaab aaab aaab", "hello hello world"]);
        let bpe = t.train(270);
        assert_eq!(bpe.merges[0].0, (b'a' as u32, b'a' as u32), "'aa' is the most frequent pair");
        let tok = Tokenizer::from_json(&bpe.to_json(&SPECIAL_TOKENS_VLM)).unwrap();
        assert_eq!(tok.special_id("<|image|>"), Some(bpe.merges.len() as u32 + 256 + 9));
        for s in ["aaab hello", "unseen ✓ text 123"] {
            assert_eq!(tok.decode(&tok.encode(s)), s);
        }
        assert!(tok.encode("aaab").len() < 4, "merges are applied");
    }

    #[test]
    fn deterministic_and_stops_when_no_pairs_left() {
        let run = || {
            let mut t = Trainer::new(NANOCHAT_PATTERN).unwrap();
            t.feed(&["the cat the hat the bat"; 3]);
            t.train(1000).merges
        };
        let (a, b) = (run(), run());
        assert_eq!(a, b);
        assert!(a.len() < 744, "tiny corpus runs out of pairs early");
    }
}
