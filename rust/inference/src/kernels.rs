//! Small fused, rayon-parallel CPU kernels for element-wise hot spots found by
//! `vlm-bench ops`: in candle 0.9, `relu().sqr()` on a (T, 4C) activation took ~2.7 ms
//! (two single-threaded passes plus an allocation), and the fused `rms_norm` is single-threaded.

use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape};
use rayon::prelude::*;

fn contiguous_f32<'a>(s: &'a CpuStorage, l: &Layout, op: &str) -> Result<&'a [f32]> {
    let data = match s {
        CpuStorage::F32(v) => v.as_slice(),
        _ => candle_core::bail!("{op}: expected f32"),
    };
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(&data[a..b]),
        None => candle_core::bail!("{op}: input must be contiguous"),
    }
}

const CHUNK: usize = 16 * 1024;

/// y = relu(x)^2 (nanochat's MLP activation), one parallel pass.
pub struct ReluSquare;

impl CustomOp1 for ReluSquare {
    fn name(&self) -> &'static str {
        "relu-square"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let x = contiguous_f32(s, l, "relu-square")?;
        let mut out = vec![0f32; x.len()];
        out.par_chunks_mut(CHUNK).zip(x.par_chunks(CHUNK)).for_each(|(o, i)| {
            for (o, &v) in o.iter_mut().zip(i) {
                let r = v.max(0.0);
                *o = r * r;
            }
        });
        Ok((CpuStorage::F32(out), l.shape().clone()))
    }
}

/// Parameter-free RMSNorm over the last dim, rows in parallel:
/// y = x / sqrt(mean(x^2) + eps)   (matches F.rms_norm without weight)
pub struct RmsNormNoWeight {
    pub eps: f32,
}

impl CustomOp1 for RmsNormNoWeight {
    fn name(&self) -> &'static str {
        "rms-norm-no-weight"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let x = contiguous_f32(s, l, "rms-norm")?;
        let d = *l.dims().last().unwrap();
        let mut out = vec![0f32; x.len()];
        let eps = self.eps;
        // Group rows so each task does a reasonable amount of work.
        let rows_per_task = (CHUNK / d).max(1);
        out.par_chunks_mut(d * rows_per_task).zip(x.par_chunks(d * rows_per_task)).for_each(|(o, i)| {
            for (orow, irow) in o.chunks_mut(d).zip(i.chunks(d)) {
                let ss: f32 = irow.iter().map(|v| v * v).sum();
                let inv = 1.0 / (ss / d as f32 + eps).sqrt();
                for (o, &v) in orow.iter_mut().zip(irow) {
                    *o = v * inv;
                }
            }
        });
        Ok((CpuStorage::F32(out), l.shape().clone()))
    }
}
