//! Fused causal attention kernel for CPU.
//!
//! Why: profiling (`vlm-bench ops`) showed that the composed candle-op attention
//! (`q @ k^T`, mask add, softmax, `@ v`) dominated prefill on CPU. Candle runs the batched
//! per-head matmuls one small GEMM at a time, materializes the (T x T) mask and scores,
//! and needs contiguous copies of the KV cache view plus the repeated KV heads for MQA/GQA.
//!
//! This kernel computes, for each (batch, head, query row) in parallel (rayon):
//!     softmax(q . K[:limit]^T * scale) @ V[:limit],   limit = Tk - Tq + i + 1
//! reading K/V through their strides, so a narrowed KV-cache buffer is used in place and
//! query head h reads KV head h / (H / H_kv) directly (no repeat_interleave copy).
//! Softmax uses max subtraction with f32 accumulation, like PyTorch's CPU SDPA.
//!
//! GPU devices keep using the composed candle-op path in `model.rs`.

use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape};
use rayon::prelude::*;

pub struct CausalAttention {
    pub scale: f32,
}

struct View<'a> {
    data: &'a [f32],
    off: usize,
    s0: usize, // batch stride
    s1: usize, // head stride
    s2: usize, // time stride
}

fn view<'a>(s: &'a CpuStorage, l: &Layout, name: &str) -> Result<(View<'a>, [usize; 4])> {
    let data = match s {
        CpuStorage::F32(v) => v.as_slice(),
        _ => candle_core::bail!("fused attention: {name} must be f32"),
    };
    let dims: [usize; 4] = l.dims().try_into().map_err(|_| candle_core::Error::Msg(format!("{name} must be 4D")))?;
    let st = l.stride();
    if st[3] != 1 {
        candle_core::bail!("fused attention: {name} last dim must be contiguous");
    }
    Ok((View { data, off: l.start_offset(), s0: st[0], s1: st[1], s2: st[2] }, dims))
}

impl CustomOp3 for CausalAttention {
    fn name(&self) -> &'static str {
        "fused-causal-attention"
    }

    fn cpu_fwd(
        &self,
        sq: &CpuStorage,
        lq: &Layout,
        sk: &CpuStorage,
        lk: &Layout,
        sv: &CpuStorage,
        lv: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (q, [b, h, tq, d]) = view(sq, lq, "q")?;
        let (k, [bk, hkv, tk, dk]) = view(sk, lk, "k")?;
        let (v, [bv, hv, tv, dv]) = view(sv, lv, "v")?;
        if bk != b || bv != b || dk != d || dv != d || hv != hkv || tv != tk || h % hkv != 0 || tk < tq {
            candle_core::bail!("fused attention: incompatible shapes q {:?} k {:?} v {:?}", lq.dims(), lk.dims(), lv.dims());
        }
        let rep = h / hkv;
        let prefix = tk - tq;
        let scale = self.scale;
        let mut out = vec![0f32; b * h * tq * d];
        out.par_chunks_mut(d).enumerate().for_each_init(
            || Vec::with_capacity(tk),
            |scores, (row, o)| {
                let i = row % tq;
                let hh = (row / tq) % h;
                let bb = row / (tq * h);
                let kh = hh / rep;
                let qo = q.off + bb * q.s0 + hh * q.s1 + i * q.s2;
                let qr = &q.data[qo..qo + d];
                let kbase = k.off + bb * k.s0 + kh * k.s1;
                let vbase = v.off + bb * v.s0 + kh * v.s1;
                let limit = prefix + i + 1; // causal: keys 0..=prefix+i
                scores.clear();
                let mut max = f32::NEG_INFINITY;
                for j in 0..limit {
                    let ko = kbase + j * k.s2;
                    let kr = &k.data[ko..ko + d];
                    let s = dot(qr, kr) * scale;
                    max = max.max(s);
                    scores.push(s);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = 1.0 / sum;
                o.fill(0.0);
                for (j, &p) in scores.iter().enumerate() {
                    let vo = vbase + j * v.s2;
                    axpy(p * inv, &v.data[vo..vo + d], o);
                }
            },
        );
        Ok((CpuStorage::F32(out), Shape::from((b, h, tq, d))))
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // 8 independent accumulators so the compiler can vectorize.
    let mut acc = [0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        for l in 0..8 {
            acc[l] += a[c * 8 + l] * b[c * 8 + l];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
fn axpy(alpha: f32, x: &[f32], y: &mut [f32]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}
