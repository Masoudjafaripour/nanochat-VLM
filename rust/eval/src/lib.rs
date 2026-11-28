//! Base-model evaluation on the Rust engine (CPU).
//!
//! * [`bpb`]: bits per byte on pre-tokenized shards (`nanochat_vlm/loss_eval.py`)
//! * [`core`]: the CORE metric (`nanochat_vlm/core_eval.py` + `src/base_eval.py`),
//!   with the same prompt rendering, data shuffle and few-shot sampling (bit-exact
//!   port of Python's `random`) so scores are comparable with the Python numbers.

pub mod pyrandom;

use candle_core::{DType, Tensor, D};
use vlm_inference::model::Gpt;
use vlm_inference::tokenizer::Tokenizer;

/// Per-position cross-entropy and argmax for ids (B, T): position t predicts ids[t+1].
/// Returns (losses (B, T-1), predictions (B, T)) as nested Vecs.
pub fn forward_losses(model: &Gpt, ids: &[Vec<u32>]) -> anyhow::Result<(Vec<Vec<f32>>, Vec<Vec<u32>>)> {
    let (b, t) = (ids.len(), ids[0].len());
    let flat: Vec<u32> = ids.concat();
    let x = Tensor::from_vec(flat, (b, t), &model.device)?;
    let logits = model.forward(&x, None, false)?.to_dtype(DType::F32)?; // (B, T, V)
    let logp = candle_nn::ops::log_softmax(&logits, D::Minus1)?;
    let preds: Vec<Vec<u32>> = logits.argmax(D::Minus1)?.to_vec2()?;
    let targets = Tensor::from_vec(ids.iter().flat_map(|r| r[1..].to_vec()).collect::<Vec<u32>>(), (b, t - 1, 1), &model.device)?;
    let nll = logp.narrow(1, 0, t - 1)?.contiguous()?.gather(&targets, 2)?.squeeze(2)?.neg()?;
    Ok((nll.to_vec2()?, preds))
}

/// tok_train.py's token_bytes: UTF-8 length of the *lossy* decode of each token; 0 for specials.
pub fn token_bytes(tok: &Tokenizer, vocab_size: usize) -> Vec<u32> {
    (0..vocab_size as u32)
        .map(|id| if tok.is_special(id) { 0 } else { tok.decode(&[id]).len() as u32 })
        .collect()
}

pub mod bpb {
    use super::*;

    /// `evaluate_bpb` over `steps` windows of (B, T) from a token stream (same windows as
    /// `pretokenized_loader.py`, rank 0). Returns (bpb, nats, bytes).
    pub fn evaluate(model: &Gpt, tokens: &[u32], b: usize, t: usize, steps: usize, tb: &[u32]) -> anyhow::Result<(f64, f64, u64)> {
        let (mut nats, mut bytes) = (0f64, 0u64);
        let windows = (tokens.len().saturating_sub(1)) / (b * t);
        anyhow::ensure!(windows >= steps, "only {windows} windows of {b}x{t} available, asked for {steps}");
        for w in 0..steps {
            let chunk = &tokens[w * b * t..w * b * t + b * t + 1];
            // inputs row r = chunk[r*T .. r*T+T], targets shifted by one; append the
            // last target so each row is T+1 long and forward_losses scores all T.
            let rows: Vec<Vec<u32>> = (0..b).map(|r| chunk[r * t..r * t + t + 1].to_vec()).collect();
            let (losses, _) = forward_losses_rows(model, &rows)?;
            for (r, row) in rows.iter().enumerate() {
                for (i, &y) in row[1..].iter().enumerate() {
                    let nb = tb[y as usize];
                    if nb > 0 {
                        nats += losses[r][i] as f64;
                        bytes += nb as u64;
                    }
                }
            }
        }
        Ok((nats / (std::f64::consts::LN_2 * bytes as f64), nats, bytes))
    }

    /// Like `forward_losses` but rows are T+1 long and only the first T positions are fed.
    fn forward_losses_rows(model: &Gpt, rows: &[Vec<u32>]) -> anyhow::Result<(Vec<Vec<f32>>, ())> {
        let t = rows[0].len() - 1;
        let inputs: Vec<u32> = rows.iter().flat_map(|r| r[..t].to_vec()).collect();
        let x = Tensor::from_vec(inputs, (rows.len(), t), &model.device)?;
        let logp = candle_nn::ops::log_softmax(&model.forward(&x, None, false)?.to_dtype(DType::F32)?, D::Minus1)?;
        let y = Tensor::from_vec(rows.iter().flat_map(|r| r[1..].to_vec()).collect::<Vec<u32>>(), (rows.len(), t, 1), &model.device)?;
        Ok((logp.gather(&y, 2)?.squeeze(2)?.neg()?.to_vec2()?, ()))
    }
}

pub mod core;
