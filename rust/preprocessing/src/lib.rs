//! Visual preprocessing for nanochat-VLM.
//!
//! Reproduces the Python reference pipeline in `test/test_projector.py`:
//!
//! ```text
//! PIL.Image.open(p).convert("RGB")             -> decode_rgb8 / load_rgb8
//! transforms.Resize((224, 224))                -> resize::resize_u8 (Pillow-exact bilinear)
//! transforms.ToTensor()                        -> to_chw_f32 (u8 / 255, HWC -> CHW)
//! (optional) transforms.Normalize(mean, std)   -> normalize_chw
//! unfold/permute/reshape into 16x16 patches    -> patchify (num_patches, C*P*P)
//! ```
//!
//! All stages are pure CPU code on contiguous buffers, so they can be called from the
//! data pipeline, the inference engine, the PyO3 bindings and the WASM demo alike.

pub mod resize;

use rayon::prelude::*;
pub use resize::Filter;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum PreprocessError {
    #[error("failed to read image {path}: {source}")]
    Io { path: String, source: std::io::Error },
    #[error("failed to decode image: {0}")]
    Decode(#[from] image::ImageError),
    #[error("invalid config: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, PreprocessError>;

/// An 8-bit RGB image stored row-major, interleaved (HWC), like a PIL "RGB" image.
#[derive(Clone, Debug)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// Preprocessing configuration. Defaults match `test/test_projector.py`.
#[derive(Clone, Debug)]
pub struct PreprocessConfig {
    pub image_size: (usize, usize), // (height, width), like transforms.Resize((h, w))
    pub patch_size: usize,
    pub filter: Filter,
    /// Per-channel mean/std for transforms.Normalize. `None` = ToTensor only.
    pub mean_std: Option<([f32; 3], [f32; 3])>,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self { image_size: (224, 224), patch_size: 16, filter: Filter::Bilinear, mean_std: None }
    }
}

impl PreprocessConfig {
    pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
    pub const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
    pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];

    pub fn validate(&self) -> Result<()> {
        let (h, w) = self.image_size;
        let p = self.patch_size;
        if p == 0 || h == 0 || w == 0 {
            return Err(PreprocessError::Config("sizes must be positive".into()));
        }
        if h % p != 0 || w % p != 0 {
            return Err(PreprocessError::Config(format!(
                "image size {h}x{w} not divisible by patch size {p}"
            )));
        }
        Ok(())
    }

    pub fn num_patches(&self) -> usize {
        (self.image_size.0 / self.patch_size) * (self.image_size.1 / self.patch_size)
    }

    /// Flattened patch dimension = C * P * P (the projector's `vision_dim`).
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }
}

/// Decode encoded image bytes (PNG/JPEG) to RGB8. Alpha is dropped and grayscale is
/// replicated, matching `PIL.Image.convert("RGB")`.
pub fn decode_rgb8(bytes: &[u8]) -> Result<RgbImage> {
    let img = image::load_from_memory(bytes)?.to_rgb8();
    let (w, h) = img.dimensions();
    Ok(RgbImage { width: w as usize, height: h as usize, data: img.into_raw() })
}

pub fn load_rgb8(path: impl AsRef<Path>) -> Result<RgbImage> {
    let p = path.as_ref();
    let bytes = std::fs::read(p).map_err(|e| PreprocessError::Io { path: p.display().to_string(), source: e })?;
    decode_rgb8(&bytes)
}

pub fn resize(img: &RgbImage, height: usize, width: usize, filter: Filter) -> RgbImage {
    let data = resize::resize_u8(&img.data, img.width, img.height, 3, width, height, filter);
    RgbImage { width, height, data }
}

/// `transforms.ToTensor()`: HWC uint8 -> CHW f32 in [0, 1].
pub fn to_chw_f32(img: &RgbImage) -> Vec<f32> {
    let hw = img.width * img.height;
    let mut out = vec![0f32; 3 * hw];
    for (i, px) in img.data.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * hw + i] = px[c] as f32 / 255.0;
        }
    }
    out
}

/// `transforms.Normalize(mean, std)` in place on a CHW buffer.
pub fn normalize_chw(chw: &mut [f32], hw: usize, mean: [f32; 3], std: [f32; 3]) {
    for c in 0..3 {
        let (m, s) = (mean[c], std[c]);
        for v in &mut chw[c * hw..(c + 1) * hw] {
            *v = (*v - m) / s;
        }
    }
}

/// Split a CHW image into non-overlapping patches.
///
/// Equivalent to
/// `x.unfold(1,P,P).unfold(2,P,P).permute(1,2,0,3,4).reshape(nh*nw, C*P*P)`:
/// patches are row-major over the grid, and each patch vector is laid out (c, py, px).
pub fn patchify(chw: &[f32], channels: usize, h: usize, w: usize, p: usize) -> Vec<f32> {
    assert_eq!(chw.len(), channels * h * w);
    assert!(h % p == 0 && w % p == 0);
    let (nh, nw) = (h / p, w / p);
    let pd = channels * p * p;
    let mut out = vec![0f32; nh * nw * pd];
    for gy in 0..nh {
        for gx in 0..nw {
            let dst = &mut out[(gy * nw + gx) * pd..(gy * nw + gx + 1) * pd];
            for c in 0..channels {
                for py in 0..p {
                    let src_off = c * h * w + (gy * p + py) * w + gx * p;
                    let dst_off = c * p * p + py * p;
                    dst[dst_off..dst_off + p].copy_from_slice(&chw[src_off..src_off + p]);
                }
            }
        }
    }
    out
}

/// Output of the full pipeline for one image.
#[derive(Clone, Debug)]
pub struct Patches {
    pub num_patches: usize,
    pub patch_dim: usize,
    /// Row-major (num_patches, patch_dim).
    pub data: Vec<f32>,
}

/// Full pipeline on an already-decoded image.
pub fn preprocess_image(img: &RgbImage, cfg: &PreprocessConfig) -> Result<Patches> {
    cfg.validate()?;
    let (h, w) = cfg.image_size;
    let resized = if img.height == h && img.width == w { img.clone() } else { resize(img, h, w, cfg.filter) };
    let mut chw = to_chw_f32(&resized);
    if let Some((mean, std)) = cfg.mean_std {
        normalize_chw(&mut chw, h * w, mean, std);
    }
    let data = patchify(&chw, 3, h, w, cfg.patch_size);
    Ok(Patches { num_patches: cfg.num_patches(), patch_dim: cfg.patch_dim(), data })
}

pub fn preprocess_bytes(bytes: &[u8], cfg: &PreprocessConfig) -> Result<Patches> {
    preprocess_image(&decode_rgb8(bytes)?, cfg)
}

pub fn preprocess_path(path: impl AsRef<Path>, cfg: &PreprocessConfig) -> Result<Patches> {
    preprocess_image(&load_rgb8(path)?, cfg)
}

/// Preprocess many images in parallel (rayon). Output is one contiguous
/// (batch, num_patches, patch_dim) buffer, in input order.
pub fn preprocess_batch<P: AsRef<Path> + Sync>(paths: &[P], cfg: &PreprocessConfig) -> Result<Vec<f32>> {
    cfg.validate()?;
    let per = cfg.num_patches() * cfg.patch_dim();
    let mut out = vec![0f32; paths.len() * per];
    out.par_chunks_mut(per).zip(paths.par_iter()).try_for_each(|(dst, p)| {
        let patches = preprocess_path(p, cfg)?;
        dst.copy_from_slice(&patches.data);
        Ok::<(), PreprocessError>(())
    })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchify_matches_manual_layout() {
        // 1 channel, 4x4 image, 2x2 patches: values = index.
        let chw: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let p = patchify(&chw, 1, 4, 4, 2);
        assert_eq!(p, vec![0., 1., 4., 5., 2., 3., 6., 7., 8., 9., 12., 13., 10., 11., 14., 15.]);
    }

    #[test]
    fn default_config_shapes() {
        let cfg = PreprocessConfig::default();
        assert_eq!(cfg.num_patches(), 196);
        assert_eq!(cfg.patch_dim(), 768);
        let img = RgbImage { width: 300, height: 200, data: vec![255; 300 * 200 * 3] };
        let out = preprocess_image(&img, &cfg).unwrap();
        assert_eq!(out.data.len(), 196 * 768);
        assert!(out.data.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn rejects_bad_patch_size() {
        let cfg = PreprocessConfig { patch_size: 15, ..Default::default() };
        assert!(cfg.validate().is_err());
    }
}
