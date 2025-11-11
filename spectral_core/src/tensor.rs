use crate::error::{Result, SpectralError};
use pyo3::prelude::*;

/// Simplified tensor representation for CPU f32 only
#[derive(Debug, Clone)]
pub struct DeviceTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl DeviceTensor {
    /// Create from numpy array
    pub fn from_numpy(array: &PyAny) -> Result<Self> {
        // Extract shape
        let shape_py: Vec<i64> = array.getattr("shape")?.extract()?;
        let shape: Vec<usize> = shape_py.into_iter().map(|x| x as usize).collect();

        // Extract data as f32
        let data: Vec<f32> = array.extract()?;

        Ok(Self { data, shape })
    }

    /// Convert back to numpy array
    pub fn to_numpy(&self, py: Python) -> Result<PyObject> {
        Ok(numpy::PyArray::from_vec(py, self.data.clone())
            .reshape(self.shape.clone())?
            .into_py(py))
    }

    /// Get total number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}