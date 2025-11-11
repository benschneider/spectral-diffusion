use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SpectralError {
    #[error("Unsupported tensor dtype: {0}")]
    UnsupportedDtype(String),

    #[error("Unsupported tensor device: {0}")]
    UnsupportedDevice(String),

    #[error("Tensor shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    #[error("FFT planning failed: {0}")]
    FFTPlanError(String),

    #[error("CUDA operation failed: {0}")]
    CudaError(String),

    #[error("DLPack conversion failed: {0}")]
    DLPackError(String),

    #[error("Backend not available: {0}")]
    BackendUnavailable(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),
}

impl From<SpectralError> for PyErr {
    fn from(err: SpectralError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

impl From<PyErr> for SpectralError {
    fn from(err: PyErr) -> Self {
        SpectralError::ConfigError(err.to_string())
    }
}

impl From<pyo3::PyDowncastError<'_>> for SpectralError {
    fn from(err: pyo3::PyDowncastError) -> Self {
        SpectralError::ConfigError(format!("PyDowncastError: {}", err))
    }
}

pub type Result<T> = std::result::Result<T, SpectralError>;