//! Python wrappers for the Rust inference engine and tokenizer.

use candle_core::DType;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use vlm_inference::{tokenizer::Tokenizer, Engine, GenerateOptions, KvStrategy, SamplingParams, Segment};
use vlm_preprocessing::Patches;

fn rt<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Python segments: list of `list[int]` (token ids) or float32 ndarray (num_patches, patch_dim).
fn parse_segments(segments: &Bound<'_, PyList>) -> PyResult<Vec<Segment>> {
    let mut out = Vec::with_capacity(segments.len());
    for item in segments.iter() {
        if let Ok(arr) = item.extract::<PyReadonlyArray2<f32>>() {
            let a = arr.as_array();
            let (n, d) = a.dim();
            out.push(Segment::Image(Patches { num_patches: n, patch_dim: d, data: a.iter().copied().collect() }));
        } else if let Ok(ids) = item.extract::<Vec<u32>>() {
            out.push(Segment::Tokens(ids));
        } else {
            return Err(PyValueError::new_err("segment must be list[int] or float32 ndarray (N, patch_dim)"));
        }
    }
    Ok(out)
}

#[pyclass(name = "Engine", module = "nanochat_vlm_rs")]
pub struct PyEngine {
    inner: Engine,
}

#[pymethods]
impl PyEngine {
    /// Load an export directory written by rust/scripts/export_checkpoint.py (CPU, float32).
    /// `attn`: "fused" (default, CPU kernels) or "composed" (plain candle ops).
    #[new]
    #[pyo3(signature = (export_dir, attn="fused"))]
    fn new(py: Python<'_>, export_dir: String, attn: &str) -> PyResult<Self> {
        let imp = match attn {
            "fused" => vlm_inference::AttnImpl::Fused,
            "composed" => vlm_inference::AttnImpl::Composed,
            s => return Err(PyValueError::new_err(format!("unknown attn {s}"))),
        };
        let mut inner = py
            .allow_threads(|| Engine::load(&export_dir, &candle_core::Device::Cpu, DType::F32))
            .map_err(rt)?;
        inner.model.attn_impl = imp;
        Ok(Self { inner })
    }

    /// Full-sequence logits without KV cache -> float32 (T, vocab).
    fn logits<'py>(&self, py: Python<'py>, segments: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let segs = parse_segments(segments)?;
        let (t, v, data) = py
            .allow_threads(|| -> anyhow::Result<_> {
                let l = self.inner.logits(&segs)?;
                let (t, v) = l.dims2()?;
                Ok((t, v, l.flatten_all()?.to_vec1::<f32>()?))
            })
            .map_err(rt)?;
        let arr = numpy::ndarray::Array2::from_shape_vec((t, v), data).map_err(rt)?;
        Ok(arr.into_pyarray(py))
    }

    /// Returns (tokens, stats_dict).
    #[pyo3(signature = (segments, max_new_tokens=32, temperature=0.0, top_k=None, top_p=None, seed=42,
                        stop_tokens=vec![], kv_strategy="preallocated", prefill_chunk=None))]
    #[allow(clippy::too_many_arguments)]
    fn generate<'py>(
        &self,
        py: Python<'py>,
        segments: &Bound<'py, PyList>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        seed: u64,
        stop_tokens: Vec<u32>,
        kv_strategy: &str,
        prefill_chunk: Option<usize>,
    ) -> PyResult<(Vec<u32>, Bound<'py, PyDict>)> {
        let segs = parse_segments(segments)?;
        let kv_strategy = match kv_strategy {
            "preallocated" => KvStrategy::Preallocated,
            "concat" => KvStrategy::Concat,
            s => return Err(PyValueError::new_err(format!("unknown kv_strategy {s}"))),
        };
        let opts = GenerateOptions {
            max_new_tokens,
            sampling: SamplingParams { temperature, top_k, top_p, seed },
            stop_tokens,
            kv_strategy,
            prefill_chunk,
        };
        let (toks, stats) = py.allow_threads(|| self.inner.generate(&segs, &opts, |_| true)).map_err(rt)?;
        let json = serde_json::to_string(&stats).map_err(rt)?;
        let d = py.import("json")?.call_method1("loads", (json,))?.downcast_into::<PyDict>()?;
        Ok((toks, d))
    }

    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let json = serde_json::to_string(&self.inner.model.cfg).map_err(rt)?;
        py.import("json")?.call_method1("loads", (json,))
    }
}

#[pyclass(name = "Tokenizer", module = "nanochat_vlm_rs")]
pub struct PyTokenizer {
    pub(crate) inner: std::sync::Arc<Tokenizer>,
}

#[pymethods]
impl PyTokenizer {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        Ok(Self { inner: std::sync::Arc::new(Tokenizer::from_file(path).map_err(rt)?) })
    }

    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        Ok(Self { inner: std::sync::Arc::new(Tokenizer::from_json(s).map_err(rt)?) })
    }

    fn encode(&self, py: Python<'_>, text: &str) -> Vec<u32> {
        py.allow_threads(|| self.inner.encode(text))
    }

    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<u32>> {
        py.allow_threads(|| texts.iter().map(|t| self.inner.encode(t)).collect())
    }

    fn decode(&self, ids: Vec<u32>) -> String {
        self.inner.decode(&ids)
    }

    fn special_id(&self, name: &str) -> Option<u32> {
        self.inner.special_id(name)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

/// Train a BPE tokenizer (Rust port of rustbpe) and return tokenizer.json text.
#[pyfunction]
#[pyo3(signature = (texts, vocab_size, pattern=None, special_tokens=None))]
pub fn train_tokenizer(
    py: Python<'_>,
    texts: Vec<String>,
    vocab_size: u32,
    pattern: Option<String>,
    special_tokens: Option<Vec<String>>,
) -> PyResult<String> {
    use vlm_inference::tokenizer::{Trainer, NANOCHAT_PATTERN, SPECIAL_TOKENS_VLM};
    let pat = pattern.unwrap_or_else(|| NANOCHAT_PATTERN.to_string());
    let special: Vec<String> = special_tokens.unwrap_or_else(|| SPECIAL_TOKENS_VLM.iter().map(|s| s.to_string()).collect());
    py.allow_threads(|| -> anyhow::Result<String> {
        let mut t = Trainer::new(&pat)?;
        t.feed(&texts);
        let refs: Vec<&str> = special.iter().map(|s| s.as_str()).collect();
        Ok(t.train(vocab_size).to_json(&refs))
    })
    .map_err(rt)
}
