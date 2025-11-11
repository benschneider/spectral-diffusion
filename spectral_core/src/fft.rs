use crate::backends::{FFTBackend, PlanCache, PlanKey};
use crate::error::{Result, SpectralError};
use crate::tensor::DeviceTensor;
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::Mutex;

/// Main spectral processing interface
pub struct SpectralProcessor {
    backend: FFTBackend,
    planner: Mutex<FftPlanner<f32>>,
    plan_cache: Mutex<PlanCache<Box<dyn std::any::Any + Send + Sync>>>,
}

impl SpectralProcessor {
    pub fn new(backend: FFTBackend) -> Result<Self> {
        Ok(Self {
            backend,
            planner: Mutex::new(FftPlanner::new()),
            plan_cache: Mutex::new(PlanCache::new(100)), // Cache up to 100 plans
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

    // Placeholder implementations for other backends
    #[cfg(feature = "fftw")]
    fn fft2_fftw(&self, _input: &DeviceTensor) -> Result<DeviceTensor> {
        Err(SpectralError::BackendUnavailable("FFTW not implemented".to_string()))
    }

    #[cfg(feature = "fftw")]
    fn ifft2_fftw(&self, _input: &DeviceTensor) -> Result<DeviceTensor> {
        Err(SpectralError::BackendUnavailable("FFTW not implemented".to_string()))
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