#![cfg_attr(feature = "simd_avx2", feature(portable_simd))]

pub mod api_c;
pub mod dlpack;
pub mod fft2d;
pub mod fused;
pub mod planner;
pub mod simd;
pub mod types;
pub mod workspace;
#[cfg(feature = "python")]
mod pybindings;

use planner::{PlanEntry, GLOBAL_PLANNER};
use std::sync::Arc;
use types::{FftDirection, Result};

pub struct RifftHandle {
    planner: Arc<planner::Planner>,
    simd_enabled: bool,
}

impl RifftHandle {
    pub fn new() -> Self {
        Self {
            planner: GLOBAL_PLANNER.clone(),
            simd_enabled: cfg!(feature = "simd_avx2"),
        }
    }

    fn plan(&self, height: usize, width: usize, direction: FftDirection) -> Result<Arc<PlanEntry>> {
        self.planner.plan(height, width, direction, self.simd_enabled)
    }

    pub fn fft2d_forward(
        &self,
        data: &mut [types::Complex],
        height: usize,
        width: usize,
    ) -> Result<()> {
        let plan = self.plan(height, width, FftDirection::Forward)?;
        fft2d::execute(&plan, data)
    }

    pub fn fft2d_inverse(
        &self,
        data: &mut [types::Complex],
        height: usize,
        width: usize,
    ) -> Result<()> {
        let plan = self.plan(height, width, FftDirection::Inverse)?;
        fft2d::execute(&plan, data)
    }

    pub fn fft_filter_ifft(
        &self,
        data: &mut [types::Complex],
        filter: &[types::Complex],
        height: usize,
        width: usize,
    ) -> Result<()> {
        let forward = self.plan(height, width, FftDirection::Forward)?;
        let inverse = self.plan(height, width, FftDirection::Inverse)?;
        fused::fft_filter_ifft(&forward, &inverse, data, filter)
    }
}

pub fn get_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

pub fn get_backend_name() -> &'static str {
    types::BACKEND_NAME
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _internal(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pybindings::register(py, m)
}
