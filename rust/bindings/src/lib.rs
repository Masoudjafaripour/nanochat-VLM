//! Python module `nanochat_vlm_rs`.
//!
//! Exposes the Rust preprocessing pipeline to Python as NumPy arrays so the existing
//! PyTorch code can call it without any changes (`torch.from_numpy(...)` is zero-copy).
//! Heavy work runs with the GIL released.

use numpy::{IntoPyArray, PyArray2, PyArray3};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use vlm_preprocessing as pp;

mod data;
mod inference;

fn to_py_err(e: pp::PreprocessError) -> PyErr {
    match e {
        pp::PreprocessError::Io { .. } => PyIOError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

pub(crate) fn parse_filter(name: &str) -> PyResult<pp::Filter> {
    match name {
        "bilinear" => Ok(pp::Filter::Bilinear),
        "bicubic" => Ok(pp::Filter::Bicubic),
        other => Err(PyValueError::new_err(format!("unknown filter '{other}' (use bilinear|bicubic)"))),
    }
}

/// Preprocessing settings. Defaults match `test/test_projector.py`
/// (Resize((224,224)) + ToTensor + 16x16 patches, no Normalize).
#[pyclass(name = "PreprocessConfig", module = "nanochat_vlm_rs")]
#[derive(Clone)]
pub struct PyPreprocessConfig {
    pub(crate) inner: pp::PreprocessConfig,
}

#[pymethods]
impl PyPreprocessConfig {
    #[new]
    #[pyo3(signature = (image_size=(224, 224), patch_size=16, filter="bilinear", mean=None, std=None))]
    fn new(
        image_size: (usize, usize),
        patch_size: usize,
        filter: &str,
        mean: Option<[f32; 3]>,
        std: Option<[f32; 3]>,
    ) -> PyResult<Self> {
        let mean_std = match (mean, std) {
            (Some(m), Some(s)) => Some((m, s)),
            (None, None) => None,
            _ => return Err(PyValueError::new_err("mean and std must be given together")),
        };
        let inner = pp::PreprocessConfig { image_size, patch_size, filter: parse_filter(filter)?, mean_std };
        inner.validate().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn num_patches(&self) -> usize {
        self.inner.num_patches()
    }

    #[getter]
    fn patch_dim(&self) -> usize {
        self.inner.patch_dim()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

fn cfg_or_default(cfg: Option<&PyPreprocessConfig>) -> pp::PreprocessConfig {
    cfg.map(|c| c.inner.clone()).unwrap_or_default()
}

/// Decode + convert("RGB") + resize. Returns uint8 (H, W, 3), like `np.asarray(pil_img)`.
#[pyfunction]
#[pyo3(signature = (path, height, width, filter="bilinear"))]
fn load_and_resize<'py>(
    py: Python<'py>,
    path: String,
    height: usize,
    width: usize,
    filter: &str,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let f = parse_filter(filter)?;
    let img = py
        .allow_threads(|| pp::load_rgb8(&path).map(|i| pp::resize(&i, height, width, f)))
        .map_err(to_py_err)?;
    let arr = numpy::ndarray::Array3::from_shape_vec((height, width, 3), img.data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Full pipeline for one image file -> float32 (num_patches, patch_dim).
#[pyfunction]
#[pyo3(signature = (path, config=None))]
fn preprocess<'py>(
    py: Python<'py>,
    path: String,
    config: Option<&PyPreprocessConfig>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let cfg = cfg_or_default(config);
    let p = py.allow_threads(|| pp::preprocess_path(&path, &cfg)).map_err(to_py_err)?;
    let arr = numpy::ndarray::Array2::from_shape_vec((p.num_patches, p.patch_dim), p.data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Full pipeline for encoded image bytes (PNG/JPEG) -> float32 (num_patches, patch_dim).
#[pyfunction]
#[pyo3(signature = (data, config=None))]
fn preprocess_bytes<'py>(
    py: Python<'py>,
    data: Vec<u8>,
    config: Option<&PyPreprocessConfig>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let cfg = cfg_or_default(config);
    let p = py.allow_threads(|| pp::preprocess_bytes(&data, &cfg)).map_err(to_py_err)?;
    let arr = numpy::ndarray::Array2::from_shape_vec((p.num_patches, p.patch_dim), p.data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Parallel (rayon) pipeline for many files -> float32 (batch, num_patches, patch_dim).
#[pyfunction]
#[pyo3(signature = (paths, config=None))]
fn preprocess_batch<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    config: Option<&PyPreprocessConfig>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let cfg = cfg_or_default(config);
    let n = paths.len();
    let data = py.allow_threads(|| pp::preprocess_batch(&paths, &cfg)).map_err(to_py_err)?;
    let arr = numpy::ndarray::Array3::from_shape_vec((n, cfg.num_patches(), cfg.patch_dim()), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

#[pymodule]
fn nanochat_vlm_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPreprocessConfig>()?;
    m.add_class::<inference::PyEngine>()?;
    m.add_class::<inference::PyTokenizer>()?;
    m.add_class::<data::PyDataLoader>()?;
    m.add_function(wrap_pyfunction!(inference::train_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_resize, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_batch, m)?)?;
    Ok(())
}
