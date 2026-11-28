//! Bit-exact port of Pillow's separable convolution resampler (`libImaging/Resample.c`)
//! for 8-bit-per-channel images.
//!
//! Why port instead of using the `image` crate's resize? The Python pipeline does
//! `torchvision.transforms.Resize((224, 224))` on a PIL image, which calls
//! `PIL.Image.resize(..., BILINEAR)`. Pillow uses an antialiased (support-scaled)
//! filter with fixed-point arithmetic and an intermediate uint8 rounding between the
//! horizontal and vertical passes. Any other implementation drifts by a few uint8
//! levels per pixel, which breaks numerical-equivalence tests downstream. This module
//! reproduces Pillow's integer arithmetic exactly so Rust and Python produce
//! identical pixels for losslessly decoded inputs (PNG/BMP).

/// Resampling filter. Only the filters the Python pipeline can use are implemented.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// `PIL.Image.BILINEAR` (torchvision default for PIL inputs).
    Bilinear,
    /// `PIL.Image.BICUBIC` (CLIP/SigLIP-style preprocessing).
    Bicubic,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Bilinear => 1.0,
            Filter::Bicubic => 2.0,
        }
    }

    #[inline]
    fn eval(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Filter::Bilinear => {
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Filter::Bicubic => {
                const A: f64 = -0.5;
                if x < 1.0 {
                    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * A
                } else {
                    0.0
                }
            }
        }
    }
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Per-output-pixel window: first input index and number of taps.
struct Coeffs {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    /// Fixed-point coefficients, `ksize` per output pixel.
    kk: Vec<i32>,
}

/// `precompute_coeffs` + `normalize_coeffs_8bpc` from Pillow.
fn precompute_coeffs(in_size: usize, out_size: usize, filter: Filter) -> Coeffs {
    let in0 = 0.0f64;
    let in1 = in_size as f64;
    let scale = (in1 - in0) / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = filter.support() * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;

    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    let mut pre = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = in0 + (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // C `(int)` casts truncate toward zero.
        let mut xmin = (center - support + 0.5) as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        let n = (xmax - xmin).max(0) as usize;
        let mut ww = 0.0f64;
        for (x, p) in pre.iter_mut().enumerate().take(n) {
            let w = filter.eval((x as f64 + xmin as f64 - center + 0.5) * ss);
            *p = w;
            ww += w;
        }
        let k = &mut kk[xx * ksize..(xx + 1) * ksize];
        for x in 0..n {
            let v = if ww != 0.0 { pre[x] / ww } else { pre[x] };
            let scaled = v * (1u32 << PRECISION_BITS) as f64;
            k[x] = if v < 0.0 {
                (-0.5 + scaled) as i32
            } else {
                (0.5 + scaled) as i32
            };
        }
        bounds.push((xmin as usize, n));
    }
    Coeffs { ksize, bounds, kk }
}

#[inline]
fn clip8(v: i32) -> u8 {
    let s = v >> PRECISION_BITS;
    s.clamp(0, 255) as u8
}

/// Horizontal pass over an interleaved `channels`-channel uint8 buffer.
fn resample_horizontal(src: &[u8], w: usize, h: usize, ch: usize, out_w: usize, c: &Coeffs) -> Vec<u8> {
    let mut out = vec![0u8; out_w * h * ch];
    let init = 1i32 << (PRECISION_BITS - 1);
    for y in 0..h {
        let row = &src[y * w * ch..(y + 1) * w * ch];
        let orow = &mut out[y * out_w * ch..(y + 1) * out_w * ch];
        for xx in 0..out_w {
            let (xmin, n) = c.bounds[xx];
            let k = &c.kk[xx * c.ksize..xx * c.ksize + n];
            for cc in 0..ch {
                let mut ss = init;
                for (x, &kv) in k.iter().enumerate() {
                    ss = ss.wrapping_add(row[(x + xmin) * ch + cc] as i32 * kv);
                }
                orow[xx * ch + cc] = clip8(ss);
            }
        }
    }
    out
}

/// Vertical pass over an interleaved `channels`-channel uint8 buffer.
fn resample_vertical(src: &[u8], w: usize, ch: usize, out_h: usize, c: &Coeffs) -> Vec<u8> {
    let stride = w * ch;
    let mut out = vec![0u8; stride * out_h];
    let init = 1i32 << (PRECISION_BITS - 1);
    for yy in 0..out_h {
        let (ymin, n) = c.bounds[yy];
        let k = &c.kk[yy * c.ksize..yy * c.ksize + n];
        let orow = &mut out[yy * stride..(yy + 1) * stride];
        for (i, o) in orow.iter_mut().enumerate() {
            let mut ss = init;
            for (y, &kv) in k.iter().enumerate() {
                ss = ss.wrapping_add(src[(y + ymin) * stride + i] as i32 * kv);
            }
            *o = clip8(ss);
        }
    }
    out
}

/// Resize an interleaved uint8 image exactly like `PIL.Image.resize((out_w, out_h), filter)`.
///
/// Pillow runs the horizontal pass first, then the vertical pass, and skips a pass when
/// that dimension is unchanged. We do the same, so outputs match bit for bit.
pub fn resize_u8(
    src: &[u8],
    w: usize,
    h: usize,
    channels: usize,
    out_w: usize,
    out_h: usize,
    filter: Filter,
) -> Vec<u8> {
    assert_eq!(src.len(), w * h * channels, "buffer size does not match dimensions");
    assert!(out_w > 0 && out_h > 0, "output size must be positive");
    let mut buf: Vec<u8>;
    let mut cur: &[u8] = src;
    let mut cur_w = w;
    if out_w != w {
        let c = precompute_coeffs(w, out_w, filter);
        buf = resample_horizontal(cur, cur_w, h, channels, out_w, &c);
        cur_w = out_w;
        cur = &buf;
    }
    if out_h != h {
        let c = precompute_coeffs(h, out_h, filter);
        let v = resample_vertical(cur, cur_w, channels, out_h, &c);
        return v;
    }
    buf = cur.to_vec();
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_when_same_size() {
        let src: Vec<u8> = (0..(4 * 3 * 3)).map(|v| v as u8).collect();
        assert_eq!(resize_u8(&src, 4, 3, 3, 4, 3, Filter::Bilinear), src);
    }

    #[test]
    fn constant_image_stays_constant() {
        let src = vec![137u8; 50 * 31 * 3];
        for f in [Filter::Bilinear, Filter::Bicubic] {
            let out = resize_u8(&src, 50, 31, 3, 17, 64, f);
            assert!(out.iter().all(|&v| v == 137), "{f:?}");
        }
    }

    #[test]
    fn coefficients_sum_to_one() {
        for (i, o) in [(100usize, 7usize), (7, 100), (224, 224), (513, 224)] {
            let c = precompute_coeffs(i, o, Filter::Bilinear);
            for xx in 0..o {
                let (_, n) = c.bounds[xx];
                let s: i64 = c.kk[xx * c.ksize..xx * c.ksize + n].iter().map(|&v| v as i64).sum();
                assert!((s - (1i64 << PRECISION_BITS)).abs() <= n as i64, "{i}->{o} at {xx}: {s}");
            }
        }
    }
}
