use crate::error::{Result, SpectralError};
use std::collections::HashMap;

/// FFT backend enumeration
#[derive(Debug, Clone)]
pub enum FFTBackend {
    CpuPocketfft,
    #[cfg(feature = "fftw")]
    CpuFftw,
    #[cfg(feature = "mkl")]
    CpuMkl,
    #[cfg(feature = "cuda")]
    CudaFused,
}

impl FFTBackend {
    /// Detect the best available backend
    pub fn detect_best() -> Result<Self> {
        // Priority order: CUDA > MKL > FFTW > PocketFFT
        #[cfg(feature = "cuda")]
        if Self::is_cuda_available() {
            return Ok(Self::CudaFused);
        }

        #[cfg(feature = "mkl")]
        if Self::is_mkl_available() {
            return Ok(Self::CpuMkl);
        }

        #[cfg(feature = "fftw")]
        if Self::is_fftw_available() {
            return Ok(Self::CpuFftw);
        }

        // PocketFFT is always available
        Ok(Self::CpuPocketfft)
    }

    /// Check if CUDA is available
    #[cfg(feature = "cuda")]
    fn is_cuda_available() -> bool {
        // TODO: Implement CUDA detection
        false
    }

    #[cfg(not(feature = "cuda"))]
    fn is_cuda_available() -> bool {
        false
    }

    /// Check if MKL is available
    #[cfg(feature = "mkl")]
    fn is_mkl_available() -> bool {
        // TODO: Implement MKL detection
        true
    }

    #[cfg(not(feature = "mkl"))]
    fn is_mkl_available() -> bool {
        false
    }

    /// Check if FFTW is available
    #[cfg(feature = "fftw")]
    fn is_fftw_available() -> bool {
        // TODO: Implement FFTW detection
        true
    }

    #[cfg(not(feature = "fftw"))]
    fn is_fftw_available() -> bool {
        false
    }
}

/// Plan cache key
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PlanKey {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub norm_mode: String,
    pub forward: bool,
}

/// FFT plan cache
pub struct PlanCache<T> {
    cache: HashMap<PlanKey, T>,
    max_size: usize,
}

impl<T> PlanCache<T> {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
        }
    }

    pub fn get(&self, key: &PlanKey) -> Option<&T> {
        self.cache.get(key)
    }

    pub fn insert(&mut self, key: PlanKey, value: T) {
        if self.cache.len() >= self.max_size {
            // Simple LRU: remove oldest (for now, just don't insert)
            // TODO: Implement proper LRU
            return;
        }
        self.cache.insert(key, value);
    }
}