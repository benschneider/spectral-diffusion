use crate::dlpack::{self, DlManagedTensorHandle, DLManagedTensor};
use crate::RifftHandle;
use num_complex::Complex32;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};
use std::ffi::CStr;
use std::os::raw::c_void;

#[pyclass(name = "Handle")]
pub struct PyHandle {
    inner: RifftHandle,
}

#[pymethods]
impl PyHandle {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RifftHandle::new(),
        }
    }

    fn fft2<'py>(
        &mut self,
        py: Python<'py>,
        capsule: &Bound<'py, PyCapsule>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        transform_capsule(py, capsule, |data, h, w| self.inner.fft2d_forward(data, h, w))
    }

    fn ifft2<'py>(
        &mut self,
        py: Python<'py>,
        capsule: &Bound<'py, PyCapsule>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        transform_capsule(py, capsule, |data, h, w| self.inner.fft2d_inverse(data, h, w))
    }

    fn fft_filter_ifft<'py>(
        &mut self,
        py: Python<'py>,
        data_capsule: &Bound<'py, PyCapsule>,
        filter_capsule: &Bound<'py, PyCapsule>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        transform_two_capsules(py, data_capsule, filter_capsule, |data, filter, h, w| {
            self.inner.fft_filter_ifft(data, filter, h, w)
        })
    }
}

pub fn register(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyHandle>()?;
    module.add("__version__", crate::get_version())?;
    Ok(())
}

fn transform_capsule<'py, F>(
    py: Python<'py>,
    capsule: &Bound<'py, PyCapsule>,
    f: F,
) -> PyResult<Bound<'py, PyCapsule>>
where
    F: FnOnce(&mut [Complex32], usize, usize) -> crate::types::Result<()>,
{
    let handle = take_capsule(py, capsule)?;
    let height = handle.height();
    let width = handle.width();
    let slice = unsafe { handle.as_mut_slice() };
    f(slice, height, width).map_err(py_err)?;
    make_capsule(py, handle)
}

fn transform_two_capsules<'py, F>(
    py: Python<'py>,
    data_capsule: &Bound<'py, PyCapsule>,
    filter_capsule: &Bound<'py, PyCapsule>,
    f: F,
) -> PyResult<Bound<'py, PyTuple>>
where
    F: FnOnce(&mut [Complex32], &[Complex32], usize, usize) -> crate::types::Result<()>,
{
    let data_handle = take_capsule(py, data_capsule)?;
    let filter_handle = take_capsule(py, filter_capsule)?;
    if data_handle.len() != filter_handle.len() {
        return Err(PyRuntimeError::new_err("filter and data size mismatch"));
    }
    let height = data_handle.height();
    let width = data_handle.width();
    let data_slice = unsafe { data_handle.as_mut_slice() };
    let filter_slice = unsafe { filter_handle.as_mut_slice() };
    f(data_slice, filter_slice, height, width).map_err(py_err)?;
    let data_capsule_obj = make_capsule(py, data_handle)?;
    let filter_capsule_obj = make_capsule(py, filter_handle)?;
    let tuple = PyTuple::new_bound(py, &[data_capsule_obj.into_any(), filter_capsule_obj.into_any()]);
    Ok(tuple)
}

fn take_capsule<'py>(
    py: Python<'py>,
    capsule: &Bound<'py, PyCapsule>,
) -> PyResult<DlManagedTensorHandle> {
    let ptr = capsule.pointer();
    if ptr.is_null() {
        return Err(PyRuntimeError::new_err("capsule pointer was null"));
    }
    let status = unsafe {
        pyo3::ffi::PyCapsule_SetName(capsule.as_ptr(), used_dlpack_name().as_ptr())
    };
    if status != 0 {
        return Err(PyErr::fetch(py));
    }
    unsafe { dlpack::from_dlpack_capsule(ptr) }.map_err(py_err)
}

fn make_capsule<'py>(
    py: Python<'py>,
    handle: DlManagedTensorHandle,
) -> PyResult<Bound<'py, PyCapsule>> {
    let raw = handle.into_raw() as *mut c_void;
    let capsule_ptr = unsafe {
        pyo3::ffi::PyCapsule_New(raw, dlpack_name().as_ptr(), Some(drop_dlpack_capsule))
    };
    if capsule_ptr.is_null() {
        return Err(PyErr::fetch(py));
    }
    let any = unsafe { Bound::from_owned_ptr(py, capsule_ptr) };
    any.downcast_into::<PyCapsule>().map_err(|err| err.into())
}

fn dlpack_name() -> &'static CStr {
    unsafe { CStr::from_bytes_with_nul_unchecked(b"dltensor\0") }
}

fn used_dlpack_name() -> &'static CStr {
    unsafe { CStr::from_bytes_with_nul_unchecked(b"used_dltensor\0") }
}

fn py_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

unsafe extern "C" fn drop_dlpack_capsule(obj: *mut pyo3::ffi::PyObject) {
    if obj.is_null() {
        return;
    }
    let ptr = pyo3::ffi::PyCapsule_GetPointer(obj, std::ptr::null());
    if ptr.is_null() {
        unsafe { pyo3::ffi::PyErr_Clear() };
        return;
    }
    // Reconstruct handle and drop immediately to invoke deleter.
    let _ = dlpack::DlManagedTensorHandle::from_raw(ptr as *mut DLManagedTensor);
}
