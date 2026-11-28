//! Python iterator over the Rust multimodal data pipeline.

use crate::inference::PyTokenizer;
use crate::PyPreprocessConfig;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;
use vlm_data::{Loader, LoaderConfig, Manifest, TextEncoder};

/// Streams collated batches from a JSONL manifest (`{"image": ..., "text": ...}` per line).
///
/// Each item is a dict of NumPy arrays: `patches` float32 (B, N, D), `input_ids` int64 (B, L)
/// right-padded with `pad_id`, `lengths` int64 (B,), `indices` int64 (B,), plus `epoch`.
/// Decoding, resizing, patchify and tokenization all run on Rust threads without the GIL.
#[pyclass(name = "DataLoader", module = "nanochat_vlm_rs", unsendable)]
pub struct PyDataLoader {
    loader: Option<Loader>,
    #[pyo3(get)]
    batches_per_epoch: usize,
    #[pyo3(get)]
    num_samples: usize,
}

#[pymethods]
impl PyDataLoader {
    #[new]
    #[pyo3(signature = (manifest, tokenizer, config=None, batch_size=8, num_workers=4, prefetch=4,
                        shuffle=true, seed=0, rank=0, world_size=1, max_text_tokens=128, pad_id=0,
                        drop_last=true, epochs=Some(1), skip_errors=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        manifest: String,
        tokenizer: &PyTokenizer,
        config: Option<&PyPreprocessConfig>,
        batch_size: usize,
        num_workers: usize,
        prefetch: usize,
        shuffle: bool,
        seed: u64,
        rank: usize,
        world_size: usize,
        max_text_tokens: usize,
        pad_id: u32,
        drop_last: bool,
        epochs: Option<usize>,
        skip_errors: bool,
    ) -> PyResult<Self> {
        let m = Arc::new(Manifest::open(&manifest).map_err(|e| PyRuntimeError::new_err(e.to_string()))?);
        let tok = tokenizer.inner.clone();
        let encode: TextEncoder = Arc::new(move |s: &str| tok.encode(s));
        let cfg = LoaderConfig {
            batch_size, num_workers, prefetch, shuffle, seed, rank, world_size,
            max_text_tokens, pad_id, drop_last, epochs, skip_errors,
        };
        let pre = config.map(|c| c.inner.clone()).unwrap_or_default();
        let num_samples = m.len();
        let loader = Loader::new(m, pre, encode, cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let batches_per_epoch = loader.batches_per_epoch;
        Ok(Self { loader: Some(loader), batches_per_epoch, num_samples })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let loader = self.loader.as_mut().ok_or_else(|| PyStopIteration::new_err(()))?;
        let next = py.allow_threads(|| loader.next());
        let b = match next {
            None => {
                self.loader = None;
                return Err(PyStopIteration::new_err(()));
            }
            Some(Err(e)) => return Err(PyRuntimeError::new_err(e.to_string())),
            Some(Ok(b)) => b,
        };
        let d = PyDict::new(py);
        let patches = numpy::ndarray::Array3::from_shape_vec((b.size, b.num_patches, b.patch_dim), b.patches)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let ids: Vec<i64> = b.input_ids.iter().map(|&x| x as i64).collect();
        let ids = numpy::ndarray::Array2::from_shape_vec((b.size, b.max_len), ids)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        d.set_item("patches", patches.into_pyarray(py))?;
        d.set_item("input_ids", ids.into_pyarray(py))?;
        let lengths: Vec<i64> = b.lengths.iter().map(|&x| x as i64).collect();
        let indices: Vec<i64> = b.indices.iter().map(|&x| x as i64).collect();
        d.set_item("lengths", PyArray1::from_vec(py, lengths))?;
        d.set_item("indices", PyArray1::from_vec(py, indices))?;
        d.set_item("epoch", b.epoch)?;
        Ok(d)
    }

    /// (samples processed, samples skipped because they failed to load)
    fn stats(&self) -> (usize, usize) {
        use std::sync::atomic::Ordering::Relaxed;
        self.loader
            .as_ref()
            .map(|l| (l.stats.samples_ok.load(Relaxed), l.stats.samples_skipped.load(Relaxed)))
            .unwrap_or((0, 0))
    }
}
