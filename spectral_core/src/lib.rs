use pyo3::prelude::*;
use pyo3::PyResult;
use pyo3::buffer::PyBuffer;
use std::sync::Arc;

mod error;
mod tensor;
mod fft;
mod backends;

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

    /// Batch 2D FFT using numpy arrays (FFI optimization)
    fn fft2_batch(&self, arrays: Vec<&PyAny>) -> PyResult<Vec<PyObject>> {
        let mut results = Vec::with_capacity(arrays.len());
        for array in arrays {
            let tensor = DeviceTensor::from_numpy(array)?;
            let result = self.processor.fft2(&tensor)?;
            results.push(result.to_numpy(array.py())?);
        }
        Ok(results)
    }

    /// 2D FFT returning DLPack capsule directly (zero-copy return)
    fn fft2_dlpack(&self, dlpack_capsule: &PyAny) -> PyResult<PyObject> {
        // Extract shape from the capsule by creating a temporary buffer view
        let buffer: PyBuffer<f32> = PyBuffer::get(dlpack_capsule)?;
        let data_len = buffer.len_bytes() / std::mem::size_of::<f32>();

        // Assume 2D tensor - extract height/width from buffer info
        // For now, assume square tensors or extract from buffer if possible
        // TODO: Extract actual shape from DLPack metadata
        let width = (data_len as f64).sqrt() as usize;
        let height = data_len / width;

        if height * width != data_len {
            return Err(pyo3::exceptions::PyValueError::new_err("Cannot determine 2D shape from buffer size"));
        }

        // Perform FFT and return DLPack capsule
        Ok(self.processor.fft2_dlpack_shaped(dlpack_capsule, height, width)?)
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
        #[cfg(feature = "cuda")]
        {
            FFTBackend::is_cuda_available()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Get the best available backend
    #[staticmethod]
    fn best_backend() -> String {
        match FFTBackend::detect_best() {
            Ok(FFTBackend::CpuPocketfft) => "cpu_pocketfft".to_string(),
            #[cfg(feature = "fftw")]
            Ok(FFTBackend::CpuFftw) => "cpu_fftw".to_string(),
            #[cfg(feature = "mkl")]
            Ok(FFTBackend::CpuMkl) => "cpu_mkl".to_string(),
            #[cfg(feature = "cuda")]
            Ok(FFTBackend::CudaFused) => "fused_cuda".to_string(),
            Err(_) => "cpu_pocketfft".to_string(),
        }
    }

    /// Get available backends
    #[staticmethod]
    fn available_backends() -> Vec<String> {
        let mut backends = vec!["cpu_pocketfft".to_string()];
        #[cfg(feature = "fftw")]
        if FFTBackend::is_fftw_available() {
            backends.push("cpu_fftw".to_string());
        }
        #[cfg(feature = "mkl")]
        if FFTBackend::is_mkl_available() {
            backends.push("cpu_mkl".to_string());
        }
        #[cfg(feature = "cuda")]
        if FFTBackend::is_cuda_available() {
            backends.push("fused_cuda".to_string());
        }
        backends
    }


    /// 2D FFT using DLPack capsules with explicit shape (hybrid zero-copy)
    fn fft2_dlpack_shaped(&self, dlpack_capsule: &PyAny, height: usize, width: usize) -> PyResult<PyObject> {
        Ok(self.processor.fft2_dlpack_shaped(dlpack_capsule, height, width)?)
    }

    /// 2D inverse FFT using DLPack capsules (zero-copy)
    fn ifft2_dlpack(&self, dlpack_capsule: &PyAny) -> PyResult<PyObject> {
        let tensor = DeviceTensor::from_dlpack(dlpack_capsule)?;
        let result = self.processor.ifft2(&tensor)?;
        Ok(result.to_dlpack(dlpack_capsule.py())?)
    }

    /// Fused FFT → filter → iFFT using DLPack capsules
    fn fft_filter2_dlpack(&self, x_capsule: &PyAny, h_capsule: &PyAny) -> PyResult<PyObject> {
        let x = DeviceTensor::from_dlpack(x_capsule)?;
        let h = DeviceTensor::from_dlpack(h_capsule)?;
        let result = self.processor.fft_filter2(&x, &h)?;
        Ok(result.to_dlpack(x_capsule.py())?)
    }
}