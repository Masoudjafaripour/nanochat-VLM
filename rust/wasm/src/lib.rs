//! WebAssembly bindings: run nanochat-VLM fully client-side.
//!
//! The same preprocessing and inference crates as the native build are compiled to
//! `wasm32-unknown-unknown`. Limits (see rust/README.md): single-threaded (no rayon
//! threads without cross-origin isolation + atomics), WASM SIMD only if built with
//! `+simd128`, and the 4 GiB wasm32 address space caps model size (fp32 weights are
//! loaded fully into memory).

use candle_core::{DType, Device};
use vlm_inference::tokenizer::StreamDecoder;
use vlm_inference::{Engine, GenerateOptions, SamplingParams};
use wasm_bindgen::prelude::*;

fn js_err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Decode + resize + patchify encoded image bytes with the default config
/// (224x224, 16x16 patches). Returns a flat Float32Array of length 196*768.
#[wasm_bindgen]
pub fn preprocess(image: &[u8]) -> Result<Vec<f32>, JsError> {
    let p = vlm_preprocessing::preprocess_bytes(image, &Default::default()).map_err(js_err)?;
    Ok(p.data)
}

#[wasm_bindgen]
pub struct Model {
    engine: Engine,
}

#[wasm_bindgen]
impl Model {
    /// `config` = config.json text, `weights` = model.safetensors bytes, `tokenizer` = tokenizer.json text.
    #[wasm_bindgen(constructor)]
    pub fn new(config: &str, weights: Vec<u8>, tokenizer: &str) -> Result<Model, JsError> {
        let engine = Engine::from_buffers(config, weights, Some(tokenizer), &Device::Cpu, DType::F32).map_err(js_err)?;
        Ok(Model { engine })
    }

    #[wasm_bindgen(getter)]
    pub fn has_projector(&self) -> bool {
        self.engine.model.projector.is_some()
    }

    /// Generate a reply. `on_token(text_delta)` is called for each token; return `false`
    /// from it to stop. Returns generation stats as a JSON string.
    pub fn generate(
        &self,
        prompt: &str,
        image: Option<Vec<u8>>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
        seed: u64,
        on_token: &js_sys::Function,
    ) -> Result<String, JsError> {
        let patches = match image {
            Some(bytes) => Some(vlm_preprocessing::preprocess_bytes(&bytes, &self.engine.preprocess).map_err(js_err)?),
            None => None,
        };
        let (segs, stop) = self.engine.chat_prompt(prompt, patches).map_err(js_err)?;
        let opts = GenerateOptions {
            max_new_tokens,
            sampling: SamplingParams { temperature, top_k, top_p: None, seed },
            stop_tokens: stop.clone(),
            ..Default::default()
        };
        let tok = self.engine.tokenizer().map_err(js_err)?;
        let mut dec = StreamDecoder::default();
        let (_, stats) = self
            .engine
            .generate(&segs, &opts, |id| {
                if stop.contains(&id) {
                    return true;
                }
                let s = dec.push(tok, id);
                on_token
                    .call1(&JsValue::NULL, &JsValue::from_str(&s))
                    .map(|r| r.as_bool() != Some(false))
                    .unwrap_or(false)
            })
            .map_err(js_err)?;
        serde_json::to_string(&stats).map_err(js_err)
    }
}
