//! Next-token sampling. Same semantics as `engine.sample_next_token`:
//! temperature 0 -> argmax; otherwise optional top-k, then softmax(logits / T).
//!
//! Note: seeded sampling is reproducible within Rust but does not reproduce
//! `torch.multinomial`'s random stream; only greedy decoding is token-identical to Python.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_seed() -> u64 {
    42
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self { temperature: 0.0, top_k: None, top_p: None, seed: 42 }
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: StdRng,
    scratch: Vec<(f32, u32)>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let rng = StdRng::seed_from_u64(params.seed);
        Self { params, rng, scratch: Vec::new() }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.params.temperature <= 0.0 {
            return argmax(logits);
        }
        let t = self.params.temperature;
        let s = &mut self.scratch;
        s.clear();
        s.extend(logits.iter().enumerate().map(|(i, &l)| (l, i as u32)));
        if let Some(k) = self.params.top_k {
            let k = k.clamp(1, s.len());
            s.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
            s.truncate(k);
        }
        let max = s.iter().map(|x| x.0).fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0f32;
        for x in s.iter_mut() {
            x.0 = ((x.0 - max) / t).exp();
            total += x.0;
        }
        if let Some(p) = self.params.top_p {
            s.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            let mut acc = 0.0;
            let mut keep = s.len();
            for (i, x) in s.iter().enumerate() {
                acc += x.0 / total;
                if acc >= p {
                    keep = i + 1;
                    break;
                }
            }
            s.truncate(keep);
            total = s.iter().map(|x| x.0).sum();
        }
        let mut r = self.rng.gen::<f32>() * total;
        for x in s.iter() {
            r -= x.0;
            if r <= 0.0 {
                return x.1;
            }
        }
        s.last().map(|x| x.1).unwrap_or(0)
    }
}

/// First index of the maximum, like torch.argmax.
pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_topk1_agree() {
        let l = [0.1, 3.0, -1.0, 3.0, 2.0];
        assert_eq!(argmax(&l), 1);
        let mut s = Sampler::new(SamplingParams { temperature: 1.0, top_k: Some(1), ..Default::default() });
        for _ in 0..10 {
            let t = s.sample(&l);
            assert!(t == 1 || t == 3);
        }
    }

    #[test]
    fn seeded_is_deterministic() {
        let l: Vec<f32> = (0..100).map(|i| (i as f32 * 0.37).sin()).collect();
        let p = SamplingParams { temperature: 0.8, top_k: Some(20), top_p: Some(0.9), seed: 7 };
        let a: Vec<u32> = { let mut s = Sampler::new(p.clone()); (0..20).map(|_| s.sample(&l)).collect() };
        let b: Vec<u32> = { let mut s = Sampler::new(p); (0..20).map(|_| s.sample(&l)).collect() };
        assert_eq!(a, b);
    }
}
