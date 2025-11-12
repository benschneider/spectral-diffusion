#![allow(clippy::too_many_arguments)]

#[cfg(not(feature = "fftw"))]
compile_error!("The spectral_core crate requires the `fftw` feature to be enabled.");

use once_cell::sync::Lazy;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::collections::HashMap;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Arc, Mutex, OnceLock};
use std::{env, ptr, slice};

#[cfg(feature = "fftw")]
use fftw_sys::{
    fftw_cleanup_threads, fftw_complex, fftw_destroy_plan, fftw_execute_dft, fftw_init_threads,
    fftw_plan, fftw_plan_dft_2d, fftw_plan_with_nthreads, FFTW_BACKWARD, FFTW_FORWARD,
    FFTW_MEASURE,
};
#[cfg(feature = "fftw")]
use fftw_sys::{
    fftwf_cleanup_threads, fftwf_complex, fftwf_destroy_plan, fftwf_execute_dft,
    fftwf_init_threads, fftwf_plan, fftwf_plan_dft_2d, fftwf_plan_with_nthreads,
};

const DLTENSOR_NAME: &[u8] = b"dltensor\0";
const USED_DLTENSOR_NAME: &[u8] = b"used_dltensor\0";
const DEVICE_TYPE_CPU: i32 = 1; // kDLCPU
const DTYPE_CODE_FLOAT: u8 = 2; // kDLFloat
const DTYPE_CODE_COMPLEX: u8 = 5; // kDLComplex

#[repr(C)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: usize,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Precision {
    F32,
    F64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TransformKind {
    Forward,
    Inverse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PlanKey {
    height: i64,
    width: i64,
    kind: TransformKind,
}

struct BorrowedTensor {
    handle: *mut DLManagedTensor,
    data_ptr: *mut c_void,
    shape: Vec<i64>,
    dtype: DLDataType,
    strides: Option<Vec<i64>>,
}

impl BorrowedTensor {
    unsafe fn from_capsule(capsule: &PyCapsule) -> PyResult<Self> {
        let ptr = pyo3::ffi::PyCapsule_GetPointer(
            capsule.as_ptr(),
            DLTENSOR_NAME.as_ptr() as *const c_char,
        );
        if ptr.is_null() {
            return Err(PyValueError::new_err(
                "Capsule does not contain a DLPack tensor",
            ));
        }
        if pyo3::ffi::PyCapsule_SetName(
            capsule.as_ptr(),
            USED_DLTENSOR_NAME.as_ptr() as *const c_char,
        ) != 0
        {
            return Err(PyRuntimeError::new_err(
                "Failed to mark capsule as consumed",
            ));
        }
        if pyo3::ffi::PyCapsule_SetDestructor(capsule.as_ptr(), None) != 0 {
            return Err(PyRuntimeError::new_err(
                "Failed to clear capsule destructor",
            ));
        }

        let handle = ptr as *mut DLManagedTensor;
        let tensor = &(*handle).dl_tensor;

        if tensor.device.device_type != DEVICE_TYPE_CPU {
            return Err(PyTypeError::new_err("Only CPU tensors are supported"));
        }
        if tensor.ndim != 2 {
            return Err(PyValueError::new_err("Expected a 2D tensor"));
        }

        let mut shape = Vec::with_capacity(tensor.ndim as usize);
        for i in 0..tensor.ndim {
            shape.push(*tensor.shape.add(i as usize));
        }

        let strides = if tensor.strides.is_null() {
            None
        } else {
            let mut v = Vec::with_capacity(tensor.ndim as usize);
            for i in 0..tensor.ndim {
                v.push(*tensor.strides.add(i as usize));
            }
            Some(v)
        };

        let byte_offset = tensor.byte_offset;
        let data_ptr = (tensor.data as *mut u8).add(byte_offset) as *mut c_void;

        Ok(Self {
            handle,
            data_ptr,
            shape,
            dtype: tensor.dtype,
            strides,
        })
    }

    fn len(&self) -> usize {
        self.shape.iter().product::<i64>() as usize
    }

    fn is_contiguous(&self) -> bool {
        if let Some(strides) = &self.strides {
            if strides.is_empty() {
                return true;
            }
            let mut expected = 1i64;
            for (dim, stride) in self.shape.iter().rev().zip(strides.iter().rev()) {
                if *stride != expected {
                    return false;
                }
                expected *= *dim;
            }
            true
        } else {
            true
        }
    }
}

impl Drop for BorrowedTensor {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                if let Some(deleter) = (*self.handle).deleter {
                    deleter(self.handle);
                }
                self.handle = ptr::null_mut();
            }
        }
    }
}

#[repr(C)]
struct OutputTensor32 {
    tensor: DLManagedTensor,
    buffer: Vec<fftwf_complex>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

#[repr(C)]
struct OutputTensor64 {
    tensor: DLManagedTensor,
    buffer: Vec<fftw_complex>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

impl OutputTensor32 {
    fn new(shape: &[i64]) -> PyResult<Box<Self>> {
        let len = shape.iter().product::<i64>() as usize;
        let buffer = vec![[0f32; 2]; len];
        let strides = compute_strides(shape);
        let mut boxed = Box::new(Self {
            tensor: DLManagedTensor {
                dl_tensor: DLTensor {
                    data: ptr::null_mut(),
                    device: DLDevice {
                        device_type: DEVICE_TYPE_CPU,
                        device_id: 0,
                    },
                    ndim: shape.len() as i32,
                    dtype: DLDataType {
                        code: DTYPE_CODE_COMPLEX,
                        bits: 64,
                        lanes: 1,
                    },
                    shape: ptr::null_mut(),
                    strides: ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: ptr::null_mut(),
                deleter: Some(drop_output32),
            },
            buffer,
            shape: shape.to_vec(),
            strides,
        });

        let tensor_ptr: *mut Self = &mut *boxed;
        boxed.tensor.manager_ctx = tensor_ptr as *mut c_void;
        boxed.tensor.dl_tensor.data = boxed.buffer.as_mut_ptr() as *mut c_void;
        boxed.tensor.dl_tensor.shape = boxed.shape.as_mut_ptr();
        boxed.tensor.dl_tensor.strides = boxed.strides.as_mut_ptr();
        Ok(boxed)
    }

    fn data_ptr(&mut self) -> *mut fftwf_complex {
        self.buffer.as_mut_ptr()
    }
}

impl OutputTensor64 {
    fn new(shape: &[i64]) -> PyResult<Box<Self>> {
        let len = shape.iter().product::<i64>() as usize;
        let buffer = vec![[0f64; 2]; len];
        let strides = compute_strides(shape);
        let mut boxed = Box::new(Self {
            tensor: DLManagedTensor {
                dl_tensor: DLTensor {
                    data: ptr::null_mut(),
                    device: DLDevice {
                        device_type: DEVICE_TYPE_CPU,
                        device_id: 0,
                    },
                    ndim: shape.len() as i32,
                    dtype: DLDataType {
                        code: DTYPE_CODE_COMPLEX,
                        bits: 128,
                        lanes: 1,
                    },
                    shape: ptr::null_mut(),
                    strides: ptr::null_mut(),
                    byte_offset: 0,
                },
                manager_ctx: ptr::null_mut(),
                deleter: Some(drop_output64),
            },
            buffer,
            shape: shape.to_vec(),
            strides,
        });

        let tensor_ptr: *mut Self = &mut *boxed;
        boxed.tensor.manager_ctx = tensor_ptr as *mut c_void;
        boxed.tensor.dl_tensor.data = boxed.buffer.as_mut_ptr() as *mut c_void;
        boxed.tensor.dl_tensor.shape = boxed.shape.as_mut_ptr();
        boxed.tensor.dl_tensor.strides = boxed.strides.as_mut_ptr();
        Ok(boxed)
    }

    fn data_ptr(&mut self) -> *mut fftw_complex {
        self.buffer.as_mut_ptr()
    }
}

unsafe extern "C" fn drop_output32(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    let holder = managed as *mut OutputTensor32;
    drop(Box::from_raw(holder));
}

unsafe extern "C" fn drop_output64(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    let holder = managed as *mut OutputTensor64;
    drop(Box::from_raw(holder));
}

unsafe extern "C" fn capsule_destructor32(capsule: *mut pyo3::ffi::PyObject) {
    capsule_destructor_impl(capsule);
}

unsafe extern "C" fn capsule_destructor64(capsule: *mut pyo3::ffi::PyObject) {
    capsule_destructor_impl(capsule);
}

unsafe fn capsule_destructor_impl(capsule: *mut pyo3::ffi::PyObject) {
    if capsule.is_null() {
        return;
    }
    let mut ptr =
        pyo3::ffi::PyCapsule_GetPointer(capsule, USED_DLTENSOR_NAME.as_ptr() as *const c_char);
    if ptr.is_null() {
        pyo3::ffi::PyErr_Clear();
        ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, DLTENSOR_NAME.as_ptr() as *const c_char);
        if ptr.is_null() {
            pyo3::ffi::PyErr_Clear();
            return;
        }
    }
    let tensor = ptr as *mut DLManagedTensor;
    if let Some(deleter) = (*tensor).deleter {
        deleter(tensor);
    }
}

struct Plan32 {
    plan: fftwf_plan,
}

struct Plan64 {
    plan: fftw_plan,
}

static PLAN_CACHE_F32: Lazy<Mutex<HashMap<PlanKey, Arc<Plan32>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static PLAN_CACHE_F64: Lazy<Mutex<HashMap<PlanKey, Arc<Plan64>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static FFTWF_THREAD_STATUS: OnceLock<Result<(), String>> = OnceLock::new();
static FFTW_THREAD_STATUS: OnceLock<Result<(), String>> = OnceLock::new();

fn ensure_fftwf_threads() -> PyResult<()> {
    let result = FFTWF_THREAD_STATUS.get_or_init(|| unsafe {
        if fftwf_init_threads() == 0 {
            Err("fftwf_init_threads failed".to_string())
        } else {
            Ok(())
        }
    });
    match result {
        Ok(_) => Ok(()),
        Err(msg) => Err(PyRuntimeError::new_err(msg.clone())),
    }
}

fn ensure_fftw_threads() -> PyResult<()> {
    let result = FFTW_THREAD_STATUS.get_or_init(|| unsafe {
        if fftw_init_threads() == 0 {
            Err("fftw_init_threads failed".to_string())
        } else {
            Ok(())
        }
    });
    match result {
        Ok(_) => Ok(()),
        Err(msg) => Err(PyRuntimeError::new_err(msg.clone())),
    }
}

fn thread_count() -> i32 {
    static THREADS: OnceLock<i32> = OnceLock::new();
    *THREADS.get_or_init(|| {
        env::var("SPECTRAL_BRIDGE_THREADS")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
            .filter(|v| *v > 0)
            .unwrap_or_else(|| num_cpus::get().max(1) as i32)
    })
}

fn compute_strides(shape: &[i64]) -> Vec<i64> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![0; shape.len()];
    strides[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn get_plan_f32(key: PlanKey) -> PyResult<Arc<Plan32>> {
    let mut cache = PLAN_CACHE_F32
        .lock()
        .map_err(|_| PyRuntimeError::new_err("FFTW f32 plan cache poisoned"))?;
    if let Some(plan) = cache.get(&key) {
        return Ok(plan.clone());
    }
    ensure_fftwf_threads()?;
    let threads = thread_count();
    let plan = unsafe {
        fftwf_plan_with_nthreads(threads);
        let len = (key.height * key.width) as usize;
        let mut scratch_in = vec![[0f32; 2]; len];
        let mut scratch_out = vec![[0f32; 2]; len];
        let plan = fftwf_plan_dft_2d(
            key.height as c_int,
            key.width as c_int,
            scratch_in.as_mut_ptr(),
            scratch_out.as_mut_ptr(),
            match key.kind {
                TransformKind::Forward => FFTW_FORWARD,
                TransformKind::Inverse => FFTW_BACKWARD,
            },
            FFTW_MEASURE,
        );
        if plan.is_null() {
            return Err(PyRuntimeError::new_err("fftwf_plan_dft_2d failed"));
        }
        plan
    };
    let arc = Arc::new(Plan32 { plan });
    cache.insert(key, arc.clone());
    Ok(arc)
}

fn get_plan_f64(key: PlanKey) -> PyResult<Arc<Plan64>> {
    let mut cache = PLAN_CACHE_F64
        .lock()
        .map_err(|_| PyRuntimeError::new_err("FFTW f64 plan cache poisoned"))?;
    if let Some(plan) = cache.get(&key) {
        return Ok(plan.clone());
    }
    ensure_fftw_threads()?;
    let threads = thread_count();
    let plan = unsafe {
        fftw_plan_with_nthreads(threads);
        let len = (key.height * key.width) as usize;
        let mut scratch_in = vec![[0f64; 2]; len];
        let mut scratch_out = vec![[0f64; 2]; len];
        let plan = fftw_plan_dft_2d(
            key.height as c_int,
            key.width as c_int,
            scratch_in.as_mut_ptr(),
            scratch_out.as_mut_ptr(),
            match key.kind {
                TransformKind::Forward => FFTW_FORWARD,
                TransformKind::Inverse => FFTW_BACKWARD,
            },
            FFTW_MEASURE,
        );
        if plan.is_null() {
            return Err(PyRuntimeError::new_err("fftw_plan_dft_2d failed"));
        }
        plan
    };
    let arc = Arc::new(Plan64 { plan });
    cache.insert(key, arc.clone());
    Ok(arc)
}

fn dtype_precision(dtype: DLDataType) -> PyResult<Precision> {
    match (dtype.code, dtype.bits) {
        (DTYPE_CODE_FLOAT, 32) | (DTYPE_CODE_COMPLEX, 64) => Ok(Precision::F32),
        (DTYPE_CODE_FLOAT, 64) | (DTYPE_CODE_COMPLEX, 128) => Ok(Precision::F64),
        _ => Err(PyTypeError::new_err("Unsupported dtype")),
    }
}

fn scale_inverse32(buffer: &mut [[f32; 2]], norm: f32) {
    for value in buffer.iter_mut() {
        value[0] *= norm;
        value[1] *= norm;
    }
}

fn scale_inverse64(buffer: &mut [[f64; 2]], norm: f64) {
    for value in buffer.iter_mut() {
        value[0] *= norm;
        value[1] *= norm;
    }
}

fn fft2_impl(py: Python<'_>, tensor: BorrowedTensor, kind: TransformKind) -> PyResult<PyObject> {
    if !tensor.is_contiguous() {
        return Err(PyValueError::new_err("Input tensor must be contiguous"));
    }
    let height = tensor.shape[0];
    let width = tensor.shape[1];
    let precision = dtype_precision(tensor.dtype)?;
    match precision {
        Precision::F32 => fft2_f32(py, tensor, height, width, kind),
        Precision::F64 => fft2_f64(py, tensor, height, width, kind),
    }
}

fn fft2_f32(
    py: Python<'_>,
    tensor: BorrowedTensor,
    height: i64,
    width: i64,
    kind: TransformKind,
) -> PyResult<PyObject> {
    let len = tensor.len();
    let plan = get_plan_f32(PlanKey {
        height,
        width,
        kind,
    })?;
    let mut output = OutputTensor32::new(&tensor.shape)?;
    let in_ptr = match (tensor.dtype.code, tensor.dtype.bits) {
        (DTYPE_CODE_COMPLEX, 64) => tensor.data_ptr as *mut fftwf_complex,
        (DTYPE_CODE_FLOAT, 32) => {
            let src = unsafe { slice::from_raw_parts(tensor.data_ptr as *const f32, len) };
            let mut temp = vec![[0f32; 2]; len];
            for (idx, &value) in src.iter().enumerate() {
                temp[idx][0] = value;
                temp[idx][1] = 0.0;
            }
            execute_plan_f32(py, &plan, temp.as_mut_ptr(), output.data_ptr())?;
            return finalize_capsule32(py, output);
        }
        _ => return Err(PyTypeError::new_err("Unsupported dtype for f32 FFT")),
    };
    execute_plan_f32(py, &plan, in_ptr, output.data_ptr())?;
    if let TransformKind::Inverse = kind {
        let scale = 1.0f32 / (height as f32 * width as f32);
        scale_inverse32(&mut output.buffer, scale);
    }
    finalize_capsule32(py, output)
}

fn execute_plan_f32(
    py: Python<'_>,
    plan: &Arc<Plan32>,
    in_ptr: *mut fftwf_complex,
    out_ptr: *mut fftwf_complex,
) -> PyResult<()> {
    unsafe {
        py.allow_threads(|| {
            fftwf_execute_dft(plan.plan, in_ptr, out_ptr);
        });
    }
    Ok(())
}

fn fft2_f64(
    py: Python<'_>,
    tensor: BorrowedTensor,
    height: i64,
    width: i64,
    kind: TransformKind,
) -> PyResult<PyObject> {
    let len = tensor.len();
    let plan = get_plan_f64(PlanKey {
        height,
        width,
        kind,
    })?;
    let mut output = OutputTensor64::new(&tensor.shape)?;
    let in_ptr = match (tensor.dtype.code, tensor.dtype.bits) {
        (DTYPE_CODE_COMPLEX, 128) => tensor.data_ptr as *mut fftw_complex,
        (DTYPE_CODE_FLOAT, 64) => {
            let src = unsafe { slice::from_raw_parts(tensor.data_ptr as *const f64, len) };
            let mut temp = vec![[0f64; 2]; len];
            for (idx, &value) in src.iter().enumerate() {
                temp[idx][0] = value;
                temp[idx][1] = 0.0;
            }
            execute_plan_f64(py, &plan, temp.as_mut_ptr(), output.data_ptr())?;
            return finalize_capsule64(py, output);
        }
        _ => return Err(PyTypeError::new_err("Unsupported dtype for f64 FFT")),
    };
    execute_plan_f64(py, &plan, in_ptr, output.data_ptr())?;
    if let TransformKind::Inverse = kind {
        let scale = 1.0f64 / (height as f64 * width as f64);
        scale_inverse64(&mut output.buffer, scale);
    }
    finalize_capsule64(py, output)
}

fn execute_plan_f64(
    py: Python<'_>,
    plan: &Arc<Plan64>,
    in_ptr: *mut fftw_complex,
    out_ptr: *mut fftw_complex,
) -> PyResult<()> {
    unsafe {
        py.allow_threads(|| {
            fftw_execute_dft(plan.plan, in_ptr, out_ptr);
        });
    }
    Ok(())
}

fn finalize_capsule32(py: Python<'_>, holder: Box<OutputTensor32>) -> PyResult<PyObject> {
    unsafe {
        let holder_ptr = Box::into_raw(holder);
        let tensor_ptr = &mut (*holder_ptr).tensor as *mut DLManagedTensor;
        let capsule_ptr = pyo3::ffi::PyCapsule_New(
            tensor_ptr as *mut c_void,
            DLTENSOR_NAME.as_ptr() as *const c_char,
            Some(capsule_destructor32),
        );
        if capsule_ptr.is_null() {
            drop(Box::from_raw(holder_ptr));
            return Err(PyRuntimeError::new_err("Failed to create DLPack capsule"));
        }
        Ok(PyObject::from_owned_ptr(py, capsule_ptr))
    }
}

fn finalize_capsule64(py: Python<'_>, holder: Box<OutputTensor64>) -> PyResult<PyObject> {
    unsafe {
        let holder_ptr = Box::into_raw(holder);
        let tensor_ptr = &mut (*holder_ptr).tensor as *mut DLManagedTensor;
        let capsule_ptr = pyo3::ffi::PyCapsule_New(
            tensor_ptr as *mut c_void,
            DLTENSOR_NAME.as_ptr() as *const c_char,
            Some(capsule_destructor64),
        );
        if capsule_ptr.is_null() {
            drop(Box::from_raw(holder_ptr));
            return Err(PyRuntimeError::new_err("Failed to create DLPack capsule"));
        }
        Ok(PyObject::from_owned_ptr(py, capsule_ptr))
    }
}

#[pyfunction]
fn fft2_dlpack(py: Python<'_>, capsule: &PyCapsule) -> PyResult<PyObject> {
    let tensor = unsafe { BorrowedTensor::from_capsule(capsule)? };
    fft2_impl(py, tensor, TransformKind::Forward)
}

#[pyfunction]
fn ifft2_dlpack(py: Python<'_>, capsule: &PyCapsule) -> PyResult<PyObject> {
    let tensor = unsafe { BorrowedTensor::from_capsule(capsule)? };
    fft2_impl(py, tensor, TransformKind::Inverse)
}

#[pyfunction]
fn fft2_batch_dlpack(py: Python<'_>, capsules: Vec<&PyCapsule>) -> PyResult<Vec<PyObject>> {
    let mut outputs = Vec::with_capacity(capsules.len());
    for capsule in capsules {
        let tensor = unsafe { BorrowedTensor::from_capsule(capsule)? };
        let result = fft2_impl(py, tensor, TransformKind::Forward)?;
        outputs.push(result);
    }
    Ok(outputs)
}

#[pyfunction]
fn fftw_thread_count() -> usize {
    thread_count() as usize
}

#[pymodule]
fn spectral_core(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft2_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(ifft2_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(fft2_batch_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(fftw_thread_count, m)?)?;
    // Ensure FFTW threading is initialised eagerly so diagnostics report correctly
    ensure_fftwf_threads()?;
    ensure_fftw_threads()?;
    // Respect the configured thread count
    let threads = thread_count();
    unsafe {
        fftwf_plan_with_nthreads(threads);
        fftw_plan_with_nthreads(threads);
    }
    Ok(())
}

impl Drop for Plan32 {
    fn drop(&mut self) {
        unsafe {
            if !self.plan.is_null() {
                fftwf_destroy_plan(self.plan);
            }
            fftwf_cleanup_threads();
        }
    }
}

impl Drop for Plan64 {
    fn drop(&mut self) {
        unsafe {
            if !self.plan.is_null() {
                fftw_destroy_plan(self.plan);
            }
            fftw_cleanup_threads();
        }
    }
}
