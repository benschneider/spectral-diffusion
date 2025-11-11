use crate::error::{Result, SpectralError};
use pyo3::prelude::*;
use dlpack;

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

        // Extract data as contiguous f32 array
        let pyarray = array.downcast::<numpy::PyArray<f32, numpy::IxDyn>>()?;
        let data: Vec<f32> = unsafe { pyarray.as_array().iter().cloned().collect() };

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

    /// Create from DLPack capsule (zero-copy)
    pub fn from_dlpack(dlpack_capsule: &PyAny) -> Result<Self> {
        // For now, fall back to numpy extraction
        // TODO: Implement true zero-copy DLPack access
        Self::from_numpy(dlpack_capsule)
    }

    /// Convert to DLPack capsule (zero-copy)
    pub fn to_dlpack(&self, py: Python) -> Result<PyObject> {
        // For now, create numpy array and return its __dlpack__ capsule
        // TODO: Implement true zero-copy DLPack creation
        let numpy_array = self.to_numpy(py)?;
        let capsule = numpy_array.as_ref(py).getattr("__dlpack__")?.call((), None)?;
        Ok(capsule.into())
    }
}