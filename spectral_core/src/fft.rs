use crate::backends::{FFTBackend, PlanCache};
use crate::error::{Result, SpectralError};
use crate::tensor::DeviceTensor;
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Mutex;
use pyo3::prelude::*;
use pyo3::buffer::PyBuffer;
#[cfg(feature = "fftw")]
use fftw::plan::*;
#[cfg(feature = "fftw")]
use fftw::types::*;
#[cfg(feature = "fftw")]
use std::collections::HashMap;
#[cfg(feature = "fftw")]
use std::sync::{Arc, RwLock};
#[cfg(feature = "fftw")]
use once_cell::sync::Lazy;

/// Main spectral processing interface
pub struct SpectralProcessor {
    backend: FFTBackend,
    planner: Mutex<FftPlanner<f32>>,
    plan_cache: Mutex<PlanCache<Box<dyn std::any::Any + Send + Sync>>>,
    #[cfg(feature = "fftw")]
    fftw_forward_cache: Mutex<std::collections::HashMap<(usize, usize), fftw::plan::C2CPlan32>>,
    #[cfg(feature = "fftw")]
    fftw_inverse_cache: Mutex<std::collections::HashMap<(usize, usize), fftw::plan::C2CPlan32>>,
}

#[cfg(feature = "fftw")]
static FFTW_FORWARD_CACHE: Lazy<RwLock<HashMap<(usize, usize), Arc<Mutex<C2CPlan32>>>>> = Lazy::new(|| RwLock::new(HashMap::new()));

#[cfg(feature = "fftw")]
static FFTW_INVERSE_CACHE: Lazy<RwLock<HashMap<(usize, usize), Arc<Mutex<C2CPlan32>>>>> = Lazy::new(|| RwLock::new(HashMap::new()));

impl SpectralProcessor {
    pub fn new(backend: FFTBackend) -> Result<Self> {
        Ok(Self {
            backend,
            planner: Mutex::new(FftPlanner::new()),
            plan_cache: Mutex::new(PlanCache::new(100)), // Cache up to 100 plans
            #[cfg(feature = "fftw")]
            fftw_forward_cache: Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "fftw")]
            fftw_inverse_cache: Mutex::new(std::collections::HashMap::new()),
        })
    }

    pub fn fft2(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        match &self.backend {
            FFTBackend::CpuPocketfft => self.fft2_pocketfft(input),
            #[cfg(feature = "fftw")]
            FFTBackend::CpuFftw => self.fft2_fftw(input),
            #[cfg(feature = "mkl")]
            FFTBackend::CpuMkl => self.fft2_mkl(input),
            #[cfg(feature = "cuda")]
            FFTBackend::CudaFused => self.fft2_cuda(input),
        }
    }

    pub fn ifft2(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        match &self.backend {
            FFTBackend::CpuPocketfft => self.ifft2_pocketfft(input),
            #[cfg(feature = "fftw")]
            FFTBackend::CpuFftw => self.ifft2_fftw(input),
            #[cfg(feature = "mkl")]
            FFTBackend::CpuMkl => self.ifft2_mkl(input),
            #[cfg(feature = "cuda")]
            FFTBackend::CudaFused => self.ifft2_cuda(input),
        }
    }

    pub fn fft_filter2(&self, input: &DeviceTensor, filter: &DeviceTensor) -> Result<DeviceTensor> {
        // For now, implement as separate operations
        // TODO: Fuse into single kernel for better performance
        let fft_input = self.fft2(input)?;
        let filtered = self.apply_filter(&fft_input, filter)?;
        self.ifft2(&filtered)
    }

    /// Zero-copy FFT2 using PyTorch DLPack capsule with explicit shape (hybrid zero-copy)
    pub fn fft2_dlpack_shaped(&self, dlpack_capsule: &PyAny, height: usize, width: usize) -> Result<PyObject> {
        let start_total = std::time::Instant::now();

        // Extract tensor info from DLPack
        let start_dlpack = std::time::Instant::now();

        // Get the underlying buffer directly using PyBuffer
        let buffer: PyBuffer<f32> = PyBuffer::get(dlpack_capsule)?;
        let data_len = buffer.len_bytes() / std::mem::size_of::<f32>();
        let data_ptr = buffer.buf_ptr() as *const f32;

        let expected_len = height * width;
        if data_len != expected_len {
            return Err(SpectralError::ShapeMismatch {
                expected: format!("{} elements", expected_len),
                actual: format!("{} elements", data_len),
            });
        }
        let dlpack_time = start_dlpack.elapsed();

        // Perform true zero-copy FFT
        let result = self.fft2_zerocopy_buffer(data_ptr, height, width, dlpack_capsule.py())?;

        let total_time = start_total.elapsed();
        eprintln!("DLPack FFT2 {}x{}: dlpack_setup={:.3}ms, fft_compute={:.3}ms, total={:.3}ms",
                  height, width,
                  dlpack_time.as_secs_f64() * 1000.0,
                  (total_time - dlpack_time).as_secs_f64() * 1000.0,
                  total_time.as_secs_f64() * 1000.0);

        Ok(result)
    }

    /// True zero-copy FFT2 implementation using raw buffer
    fn fft2_zerocopy_buffer(&self, input_ptr: *const f32, height: usize, width: usize, py: Python) -> Result<PyObject> {
        let cache_key = (height, width);

        // Get or create cached plan
        let plan_arc = {
            let cache = FFTW_FORWARD_CACHE.read().unwrap();
            if let Some(plan) = cache.get(&cache_key) {
                Arc::clone(plan)
            } else {
                drop(cache);
                let mut cache = FFTW_FORWARD_CACHE.write().unwrap();
                if let Some(plan) = cache.get(&cache_key) {
                    Arc::clone(plan)
                } else {
                    let new_plan = C2CPlan::aligned(&[height, width], Sign::Forward, Flag::MEASURE)?;
                    let plan_arc = Arc::new(Mutex::new(new_plan));
                    cache.insert(cache_key, Arc::clone(&plan_arc));
                    plan_arc
                }
            }
        };

        // Create a slice from the raw pointer
        let input_len = height * width;
        let input_data = unsafe { std::slice::from_raw_parts(input_ptr, input_len) };

        // Convert input to complex format
        let mut complex_data: Vec<c32> = input_data.iter()
            .map(|&x| c32::new(x, 0.0))
            .collect();

        // Execute FFT using cached plan - RELEASE GIL during computation
        py.allow_threads(|| {
            let mut plan = plan_arc.lock().unwrap();
            // Use a temporary buffer to avoid aliasing issues
            let mut temp_buffer = complex_data.clone();
            plan.c2c(&mut complex_data, &mut temp_buffer)?;
            complex_data = temp_buffer;
            Ok::<_, SpectralError>(())
        })?;

        // Convert back to real (magnitude for now)
        let real_data: Vec<f32> = complex_data.iter()
            .map(|c| c.norm())
            .collect();

        // Return as DLPack capsule for zero-copy
        Ok(DeviceTensor {
            data: real_data,
            shape: vec![height, width],
        }.to_dlpack(py)?)
    }

    /// Zero-copy iFFT2 using DLPack capsule
    pub fn ifft2_dlpack(&self, dlpack_capsule: &PyAny) -> Result<PyObject> {
        // For now, convert through numpy - TODO: implement true zero-copy
        let tensor = DeviceTensor::from_dlpack(dlpack_capsule)?;
        let result = self.ifft2(&tensor)?;
        result.to_dlpack(dlpack_capsule.py())
    }

    /// Zero-copy FFT filtering using DLPack capsules
    pub fn fft_filter2_dlpack(&self, x_capsule: &PyAny, h_capsule: &PyAny) -> Result<PyObject> {
        // For now, convert through numpy - TODO: implement true zero-copy
        let x_tensor = DeviceTensor::from_dlpack(x_capsule)?;
        let h_tensor = DeviceTensor::from_dlpack(h_capsule)?;
        let result = self.fft_filter2(&x_tensor, &h_tensor)?;
        result.to_dlpack(x_capsule.py())
    }

    fn fft2_pocketfft(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        if input.shape.len() != 2 {
            return Err(SpectralError::ShapeMismatch {
                expected: "2D".to_string(),
                actual: format!("{}D", input.shape.len()),
            });
        }

        let height = input.shape[0];
        let width = input.shape[1];

        // Convert to complex
        let mut complex_data: Vec<Complex<f32>> = input.data.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        // Perform 2D FFT using rustfft (row-wise then column-wise)
        let mut planner = self.planner.lock().unwrap();

        // FFT along rows
        for row in 0..height {
            let row_start = row * width;
            let row_end = row_start + width;
            let row_slice = &mut complex_data[row_start..row_end];
            let fft = planner.plan_fft_forward(width);
            fft.process(row_slice);
        }

        // FFT along columns (transpose operation)
        for col in 0..width {
            let mut col_data: Vec<Complex<f32>> = (0..height)
                .map(|row| complex_data[row * width + col])
                .collect();
            let fft = planner.plan_fft_forward(height);
            fft.process(&mut col_data);
            for row in 0..height {
                complex_data[row * width + col] = col_data[row];
            }
        }

        // Convert back to real (magnitude for now - TODO: proper complex handling)
        let real_data: Vec<f32> = complex_data.iter()
            .map(|c| c.norm())
            .collect();

        Ok(DeviceTensor {
            data: real_data,
            shape: input.shape.clone(),
        })
    }

    fn ifft2_pocketfft(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        if input.shape.len() != 2 {
            return Err(SpectralError::ShapeMismatch {
                expected: "2D".to_string(),
                actual: format!("{}D", input.shape.len()),
            });
        }

        let height = input.shape[0];
        let width = input.shape[1];

        // Convert to complex (assuming real input for now)
        let mut complex_data: Vec<Complex<f32>> = input.data.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        // Perform 2D iFFT using rustfft (column-wise then row-wise)
        let mut planner = self.planner.lock().unwrap();

        // iFFT along columns first
        for col in 0..width {
            let mut col_data: Vec<Complex<f32>> = (0..height)
                .map(|row| complex_data[row * width + col])
                .collect();
            let fft = planner.plan_fft_inverse(height);
            fft.process(&mut col_data);
            for row in 0..height {
                complex_data[row * width + col] = col_data[row];
            }
        }

        // iFFT along rows
        for row in 0..height {
            let row_start = row * width;
            let row_end = row_start + width;
            let row_slice = &mut complex_data[row_start..row_end];
            let fft = planner.plan_fft_inverse(width);
            fft.process(row_slice);
        }

        // Normalize and take real part
        let scale = 1.0 / (height * width) as f32;
        let real_data: Vec<f32> = complex_data.iter()
            .map(|c| (c.re * scale))
            .collect();

        Ok(DeviceTensor {
            data: real_data,
            shape: input.shape.clone(),
        })
    }

    fn apply_filter(&self, spectrum: &DeviceTensor, filter: &DeviceTensor) -> Result<DeviceTensor> {
        if spectrum.shape != filter.shape {
            return Err(SpectralError::ShapeMismatch {
                expected: format!("{:?}", spectrum.shape),
                actual: format!("{:?}", filter.shape),
            });
        }

        // Element-wise multiplication (for now, simple filtering)
        let filtered_data: Vec<f32> = spectrum.data.iter()
            .zip(filter.data.iter())
            .map(|(&s, &f)| s * f)
            .collect();

        Ok(DeviceTensor {
            data: filtered_data,
            shape: spectrum.shape.clone(),
        })
    }

    #[cfg(feature = "fftw")]
    fn fft2_fftw(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        if input.shape.len() != 2 {
            return Err(SpectralError::ShapeMismatch {
                expected: "2D".to_string(),
                actual: format!("{}D", input.shape.len()),
            });
        }

        let height = input.shape[0] as usize;
        let width = input.shape[1] as usize;
        let cache_key = (height, width);

        // Get or create cached plan
        let plan_arc = {
            let cache = FFTW_FORWARD_CACHE.read().unwrap();
            if let Some(plan) = cache.get(&cache_key) {
                Arc::clone(plan)
            } else {
                drop(cache);
                let mut cache = FFTW_FORWARD_CACHE.write().unwrap();
                if let Some(plan) = cache.get(&cache_key) {
                    Arc::clone(plan)
                } else {
                    let new_plan = C2CPlan::aligned(&[height, width], Sign::Forward, Flag::MEASURE)?;
                    let plan_arc = Arc::new(Mutex::new(new_plan));
                    cache.insert(cache_key, Arc::clone(&plan_arc));
                    plan_arc
                }
            }
        };

        // Convert to complex format for FFTW
        let mut complex_data: Vec<c32> = input.data.iter()
            .map(|&x| c32::new(x, 0.0))
            .collect();

        // Execute FFT using cached plan
        let mut temp = complex_data.clone();
        {
            let mut plan = plan_arc.lock().unwrap();
            plan.c2c(&mut complex_data, &mut temp)?;
        }
        complex_data = temp;

        // Convert back to real (magnitude for now)
        let real_data: Vec<f32> = complex_data.iter()
            .map(|c| c.norm())
            .collect();

        Ok(DeviceTensor {
            data: real_data,
            shape: input.shape.clone(),
        })
    }

    #[cfg(feature = "fftw")]
    fn ifft2_fftw(&self, input: &DeviceTensor) -> Result<DeviceTensor> {
        if input.shape.len() != 2 {
            return Err(SpectralError::ShapeMismatch {
                expected: "2D".to_string(),
                actual: format!("{}D", input.shape.len()),
            });
        }

        let height = input.shape[0] as usize;
        let width = input.shape[1] as usize;
        let cache_key = (height, width);

        // Get or create cached plan
        let plan_arc = {
            let cache = FFTW_INVERSE_CACHE.read().unwrap();
            if let Some(plan) = cache.get(&cache_key) {
                Arc::clone(plan)
            } else {
                drop(cache);
                let mut cache = FFTW_INVERSE_CACHE.write().unwrap();
                if let Some(plan) = cache.get(&cache_key) {
                    Arc::clone(plan)
                } else {
                    let new_plan = C2CPlan::aligned(&[height, width], Sign::Backward, Flag::MEASURE)?;
                    let plan_arc = Arc::new(Mutex::new(new_plan));
                    cache.insert(cache_key, Arc::clone(&plan_arc));
                    plan_arc
                }
            }
        };

        // Convert to complex (assuming real input)
        let mut complex_data: Vec<c32> = input.data.iter()
            .map(|&x| c32::new(x, 0.0))
            .collect();

        // Execute inverse FFT using cached plan
        let mut temp = complex_data.clone();
        {
            let mut plan = plan_arc.lock().unwrap();
            plan.c2c(&mut complex_data, &mut temp)?;
        }
        complex_data = temp;

        // Normalize and take real part
        let scale = 1.0 / (height * width) as f32;
        let real_data: Vec<f32> = complex_data.iter()
            .map(|c| c.re * scale)
            .collect();

        Ok(DeviceTensor {
            data: real_data,
            shape: input.shape.clone(),
        })
    }

    #[cfg(not(feature = "fftw"))]
    fn fft2_fftw(&self, _input: &DeviceTensor) -> Result<DeviceTensor> {
        Err(SpectralError::BackendUnavailable("FFTW not enabled".to_string()))
    }

    #[cfg(not(feature = "fftw"))]
    fn ifft2_fftw(&self, _input: &DeviceTensor) -> Result<DeviceTensor> {
        Err(SpectralError::BackendUnavailable("FFTW not enabled".to_string()))
    }

    #[cfg(feature = "mkl")]
    fn fft2_mkl(&self, _input: &DeviceTensor) -> Result<DeviceTensor> {
        Err(SpectralError::BackendUnavailable("MKL not implemented".to_string()))
    }

    #[cfg(feature = "mkl")]
    fn ifft2_mkl(&self, _input: &DeviceTensor) -> Result<DeviceTensor> {
        Err(SpectralError::BackendUnavailable("MKL not implemented".to_string()))
    }

    #[cfg(not(feature = "mkl"))]
    fn fft2_mkl(&self, _input: &DeviceTensor) -> Result<DeviceTensor> {
        Err(SpectralError::BackendUnavailable("MKL not enabled".to_string()))
    }

    #[cfg(not(feature = "mkl"))]
    fn ifft2_mkl(&self, _input: &DeviceTensor) -> Result<DeviceTensor> {
        Err(SpectralError::BackendUnavailable("MKL not enabled".to_string()))
    }

    #[cfg(feature = "cuda")]
    fn fft2_cuda(&self, _input: &DeviceTensor) -> Result<DeviceTensor> {
        Err(SpectralError::BackendUnavailable("CUDA not implemented".to_string()))
    }

    #[cfg(feature = "cuda")]
    fn ifft2_cuda(&self, _input: &DeviceTensor) -> Result<DeviceTensor> {
        Err(SpectralError::BackendUnavailable("CUDA not implemented".to_string()))
    }

    #[cfg(not(feature = "cuda"))]
    fn fft2_cuda(&self, _input: &DeviceTensor) -> Result<DeviceTensor> {
        Err(SpectralError::BackendUnavailable("CUDA not enabled".to_string()))
    }

    #[cfg(not(feature = "cuda"))]
    fn ifft2_cuda(&self, _input: &DeviceTensor) -> Result<DeviceTensor> {
        Err(SpectralError::BackendUnavailable("CUDA not enabled".to_string()))
    }
}