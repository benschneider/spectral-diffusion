use pyo3::prelude::*;
use pyo3::PyResult;
use std::sync::Arc;

mod error;
mod tensor;
mod fft;
mod backends;

use crate::error::SpectralError;
use crate::tensor::DeviceTensor;
use crate::fft::SpectralProcessor;
use crate::backends::FFTBackend;

/// Main PyO3 module
#[pymodule]
fn spectral_core(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SpectralCore>()?;
    Ok(())
}

/// Core spectral processing interface
#[pyclass]
pub struct SpectralCore {
    processor: Arc<SpectralProcessor>,
}

#[pymethods]
impl SpectralCore {
    #[new]
    fn new() -> PyResult<Self> {
        let backend = FFTBackend::detect_best()?;
        let processor = SpectralProcessor::new(backend)?;
        Ok(Self {
            processor: Arc::new(processor),
        })
    }

    /// 2D FFT using numpy arrays
    fn fft2(&self, array: &PyAny) -> PyResult<PyObject> {
        let tensor = DeviceTensor::from_numpy(array)?;
        let result = self.processor.fft2(&tensor)?;
        Ok(result.to_numpy(array.py())?)
    }

    /// 2D inverse FFT using numpy arrays
    fn ifft2(&self, array: &PyAny) -> PyResult<PyObject> {
        let tensor = DeviceTensor::from_numpy(array)?;
        let result = self.processor.ifft2(&tensor)?;
        Ok(result.to_numpy(array.py())?)
    }

    /// Fused FFT → filter → iFFT operation
    fn fft_filter2(&self, x_array: &PyAny, h_array: &PyAny) -> PyResult<PyObject> {
        let x = DeviceTensor::from_numpy(x_array)?;
        let h = DeviceTensor::from_numpy(h_array)?;
        let result = self.processor.fft_filter2(&x, &h)?;
        Ok(result.to_numpy(x_array.py())?)
    }

    /// Check if CUDA is available
    #[staticmethod]
    fn is_cuda_available() -> bool {
        // TODO: Implement CUDA detection
        false
    }

    /// Get available backends
    #[staticmethod]
    fn available_backends() -> Vec<String> {
        vec![
            "cpu_pocketfft".to_string(),
            #[cfg(feature = "fftw")]
            "cpu_fftw".to_string(),
            #[cfg(feature = "mkl")]
            "cpu_mkl".to_string(),
            #[cfg(feature = "cuda")]
            "fused_cuda".to_string(),
        ]
    }
}