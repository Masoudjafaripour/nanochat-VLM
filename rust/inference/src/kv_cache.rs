//! KV cache for autoregressive decoding.
//!
//! Python reference: `nanochat_vlm.engine.KVCache`. That cache preallocates one
//! (L, 2, B, H_kv, T, D) tensor and grows it in multiples of 1024 positions.
//!
//! Two strategies live here so we can measure the difference instead of assuming it:
//! * `Preallocated`: one buffer per layer, written in place with `slice_set`. Capacity
//!   is reserved once per forward pass and doubles when exceeded (amortized O(1) appends).
//! * `Concat`: the textbook `cat([k_cache, k_new])` every step. That is O(T) copies per
//!   step, O(T^2) per generation. Kept as a baseline for the benchmarks.

use candle_core::{DType, Device, Result, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvStrategy {
    Preallocated,
    Concat,
}

pub struct KvCache {
    strategy: KvStrategy,
    /// Per layer (K, V). Preallocated: (B, H_kv, capacity, D). Concat: (B, H_kv, pos, D).
    layers: Vec<Option<(Tensor, Tensor)>>,
    /// Positions stored (equal for all layers between forward passes).
    pos: usize,
    capacity: usize,
    batch: usize,
    n_kv_head: usize,
    head_dim: usize,
    dtype: DType,
    device: Device,
}

impl KvCache {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        strategy: KvStrategy,
        n_layer: usize,
        batch: usize,
        n_kv_head: usize,
        head_dim: usize,
        initial_capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> Self {
        Self {
            strategy,
            layers: vec![None; n_layer],
            pos: 0,
            capacity: initial_capacity.max(1),
            batch,
            n_kv_head,
            head_dim,
            dtype,
            device: device.clone(),
        }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn strategy(&self) -> KvStrategy {
        self.strategy
    }

    /// Forget all positions. Preallocated buffers are kept for reuse.
    pub fn reset(&mut self) {
        self.pos = 0;
        if self.strategy == KvStrategy::Concat {
            self.layers.iter_mut().for_each(|l| *l = None);
        }
    }

    /// Bytes currently held for K and V across all layers.
    pub fn allocated_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|(k, v)| (k.elem_count() + v.elem_count()) * self.dtype.size_in_bytes())
            .sum()
    }

    /// Make room for `t_new` more positions in every layer. Called once per forward
    /// pass, before the first layer, so all layers grow together.
    pub fn reserve(&mut self, t_new: usize) -> Result<()> {
        if self.strategy != KvStrategy::Preallocated {
            return Ok(());
        }
        let need = self.pos + t_new;
        let allocated = self.layers.iter().all(|l| l.is_some());
        if allocated && need <= self.capacity {
            return Ok(());
        }
        let mut cap = self.capacity;
        while cap < need {
            cap *= 2;
        }
        let shape = (self.batch, self.n_kv_head, cap, self.head_dim);
        for l in self.layers.iter_mut() {
            let nk = Tensor::zeros(shape, self.dtype, &self.device)?;
            let nv = Tensor::zeros(shape, self.dtype, &self.device)?;
            if let Some((k, v)) = l.as_ref() {
                if self.pos > 0 {
                    nk.slice_set(&k.narrow(2, 0, self.pos)?.contiguous()?, 2, 0)?;
                    nv.slice_set(&v.narrow(2, 0, self.pos)?.contiguous()?, 2, 0)?;
                }
            }
            *l = Some((nk, nv));
        }
        self.capacity = cap;
        Ok(())
    }

    /// Store `k`, `v` of shape (B, H_kv, T_new, D) for `layer` and return (K, V) over all
    /// positions so far. `pos` advances after the last layer, as in engine.py.
    pub fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let t_new = k.dim(2)?;
        let t1 = self.pos + t_new;
        let out = match self.strategy {
            KvStrategy::Concat => {
                let (nk, nv) = match &self.layers[layer] {
                    Some((pk, pv)) => (Tensor::cat(&[pk, k], 2)?, Tensor::cat(&[pv, v], 2)?),
                    None => (k.contiguous()?, v.contiguous()?),
                };
                self.layers[layer] = Some((nk.clone(), nv.clone()));
                (nk, nv)
            }
            KvStrategy::Preallocated => {
                let (bk, bv) = self.layers[layer]
                    .as_ref()
                    .expect("KvCache::reserve must be called before append");
                bk.slice_set(&k.contiguous()?, 2, self.pos)?;
                bv.slice_set(&v.contiguous()?, 2, self.pos)?;
                (bk.narrow(2, 0, t1)?, bv.narrow(2, 0, t1)?)
            }
        };
        if layer == self.layers.len() - 1 {
            self.pos = t1;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(strategy: KvStrategy) -> (Vec<f32>, usize) {
        let dev = Device::Cpu;
        let mut c = KvCache::new(strategy, 2, 1, 1, 2, 2, DType::F32, &dev);
        let mut last = None;
        for step in 0..5usize {
            let t = if step == 0 { 3 } else { 1 };
            c.reserve(t).unwrap();
            for layer in 0..2 {
                let base = (step * 10 + layer) as f32;
                let k = Tensor::full(base, (1, 1, t, 2), &dev).unwrap();
                let (kk, _) = c.append(layer, &k, &k).unwrap();
                last = Some(kk);
            }
        }
        assert_eq!(c.pos(), 7);
        (last.unwrap().flatten_all().unwrap().to_vec1().unwrap(), c.allocated_bytes())
    }

    #[test]
    fn strategies_agree_and_grow() {
        let (a, bytes_a) = run(KvStrategy::Preallocated);
        let (b, bytes_b) = run(KvStrategy::Concat);
        assert_eq!(a, b);
        assert_eq!(a.len(), 14);
        // Preallocated rounds capacity up (2 -> 4 -> 8); concat stores exactly 7 positions.
        assert_eq!(bytes_a, 2 * 2 * 8 * 2 * 4);
        assert_eq!(bytes_b, 2 * 2 * 7 * 2 * 4);
    }
}
