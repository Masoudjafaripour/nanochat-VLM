//! `config.json` written by `rust/scripts/export_checkpoint.py`.

use serde::{Deserialize, Serialize};
use std::path::Path;
use vlm_preprocessing::{Filter, PreprocessConfig};

/// Mirrors `nanochat_vlm.gpt.GPTConfig`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GptConfig {
    pub sequence_len: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
}

impl GptConfig {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head");
        anyhow::ensure!(
            self.n_kv_head <= self.n_head && self.n_head % self.n_kv_head == 0,
            "n_head must be a multiple of n_kv_head"
        );
        anyhow::ensure!(self.head_dim() % 2 == 0, "head_dim must be even for rotary embeddings");
        Ok(())
    }
}

/// Mirrors `nanochat_vlm.projector.VisionProjector(vision_dim, lm_dim)`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProjectorConfig {
    pub vision_dim: usize,
    pub lm_dim: usize,
    #[serde(default = "default_ln_eps")]
    pub layer_norm_eps: f64,
}

fn default_ln_eps() -> f64 {
    1e-5
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PreprocessJson {
    pub image_size: [usize; 2],
    pub patch_size: usize,
    #[serde(default = "default_filter")]
    pub filter: String,
    #[serde(default)]
    pub mean: Option<[f32; 3]>,
    #[serde(default)]
    pub std: Option<[f32; 3]>,
}

fn default_filter() -> String {
    "bilinear".into()
}

impl Default for PreprocessJson {
    fn default() -> Self {
        Self { image_size: [224, 224], patch_size: 16, filter: default_filter(), mean: None, std: None }
    }
}

impl PreprocessJson {
    pub fn to_config(&self) -> anyhow::Result<PreprocessConfig> {
        let filter = match self.filter.as_str() {
            "bilinear" => Filter::Bilinear,
            "bicubic" => Filter::Bicubic,
            f => anyhow::bail!("unknown filter {f}"),
        };
        let mean_std = match (self.mean, self.std) {
            (Some(m), Some(s)) => Some((m, s)),
            (None, None) => None,
            _ => anyhow::bail!("mean and std must be given together"),
        };
        let cfg = PreprocessConfig {
            image_size: (self.image_size[0], self.image_size[1]),
            patch_size: self.patch_size,
            filter,
            mean_std,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelConfig {
    pub model: GptConfig,
    #[serde(default)]
    pub projector: Option<ProjectorConfig>,
    #[serde(default)]
    pub preprocess: PreprocessJson,
    /// Rotary base (gpt.py: `_precompute_rotary_embeddings(base=10000)`).
    #[serde(default = "default_rope_base")]
    pub rope_base: f64,
    /// Logit soft-capping (gpt.py: `softcap = 15`).
    #[serde(default = "default_softcap")]
    pub softcap: f64,
    /// gpt.py keeps cos/sin in bfloat16; we round them the same way for parity.
    #[serde(default = "default_true")]
    pub rope_bf16: bool,
}

fn default_rope_base() -> f64 {
    10000.0
}
fn default_softcap() -> f64 {
    15.0
}
fn default_true() -> bool {
    true
}

impl ModelConfig {
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path.as_ref())?;
        let cfg: Self = serde_json::from_str(&s)?;
        cfg.model.validate()?;
        if let Some(p) = &cfg.projector {
            anyhow::ensure!(p.lm_dim == cfg.model.n_embd, "projector lm_dim != n_embd");
        }
        Ok(cfg)
    }
}
