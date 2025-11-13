use num_complex::Complex32;
use std::fmt;

pub type Complex = Complex32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FftDirection {
    Forward,
    Inverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorDType {
    Complex32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanKey {
    pub height: usize,
    pub width: usize,
    pub direction: FftDirection,
    pub dtype: TensorDType,
    pub simd_enabled: bool,
}

impl fmt::Display for PlanKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}x{}::{:?}::{:?}::simd{}",
            self.height,
            self.width,
            self.direction,
            self.dtype,
            if self.simd_enabled { 1 } else { 0 }
        )
    }
}

#[derive(thiserror::Error, Debug)]
pub enum RifftError {
    #[error("input tensor is not contiguous")]
    NonContiguous,
    #[error("invalid tensor shape; expected {expected} elements but received {got}")]
    ShapeMismatch { expected: usize, got: usize },
    #[error("unsupported dtype")]
    UnsupportedDType,
    #[error("workspace not available for {0}")]
    WorkspaceUnavailable(String),
    #[error("planner error: {0}")]
    Planner(String),
    #[error("dlpack error: {0}")]
    DlPack(String),
}

pub type Result<T> = std::result::Result<T, RifftError>;

pub const BACKEND_NAME: &str = "rustfft+rayon+simd";
