use crate::types::{Result, RifftError};
use libc::c_void;
use num_complex::Complex32;
use std::os::raw::{c_int, c_ulonglong};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDevice {
    pub device_type: c_int,
    pub device_id: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DLTensor {
    pub data: *mut c_void,
    pub device: DLDevice,
    pub ndim: c_int,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: c_ulonglong,
}

pub type DlDeleter = Option<unsafe extern "C" fn(*mut DLManagedTensor)>;

#[repr(C)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut c_void,
    pub deleter: DlDeleter,
}

unsafe impl Send for DLManagedTensor {}
unsafe impl Sync for DLManagedTensor {}

pub struct DlManagedTensorHandle {
    ptr: *mut DLManagedTensor,
    len: usize,
    height: usize,
    width: usize,
}

unsafe impl Send for DlManagedTensorHandle {}
unsafe impl Sync for DlManagedTensorHandle {}

impl DlManagedTensorHandle {
    pub unsafe fn from_raw(ptr: *mut DLManagedTensor) -> Result<Self> {
        if ptr.is_null() {
            return Err(RifftError::DlPack("received null pointer".into()));
        }
        let tensor = &*ptr;
        let dl = &tensor.dl_tensor;
        if dl.ndim < 2 {
            return Err(RifftError::DlPack("expected at least 2 dimensions".into()));
        }
        let dtype = dl.dtype;
        let is_complex32 = dtype.code == 5 && dtype.bits == 64;
        if !is_complex32 {
            return Err(RifftError::UnsupportedDType);
        }
        let ndim = dl.ndim as isize;
        let shape_slice = std::slice::from_raw_parts(dl.shape, ndim as usize);
        let height = shape_slice[shape_slice.len() - 2] as usize;
        let width = shape_slice[shape_slice.len() - 1] as usize;
        let len = shape_slice.iter().fold(1usize, |acc, dim| acc * (*dim as usize));
        if !dl.strides.is_null() {
            let strides = std::slice::from_raw_parts(dl.strides, ndim as usize);
            if strides[strides.len() - 1] != 1 {
                return Err(RifftError::NonContiguous);
            }
            if strides[strides.len() - 2] as usize != width {
                return Err(RifftError::NonContiguous);
            }
        }
        Ok(Self {
            ptr,
            len,
            height,
            width,
        })
    }

    pub unsafe fn as_mut_slice(&self) -> &mut [Complex32] {
        let tensor = &*self.ptr;
        let dl = &tensor.dl_tensor;
        let data = (dl.data as *mut Complex32).add(dl.byte_offset as usize / 8);
        std::slice::from_raw_parts_mut(data, self.len)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn into_raw(self) -> *mut DLManagedTensor {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }
}

impl Drop for DlManagedTensorHandle {
    fn drop(&mut self) {
        unsafe {
            if let Some(deleter) = (*self.ptr).deleter {
                deleter(self.ptr);
            }
        }
    }
}

pub unsafe fn from_dlpack_capsule(ptr: *mut c_void) -> Result<DlManagedTensorHandle> {
    DlManagedTensorHandle::from_raw(ptr as *mut DLManagedTensor)
}

pub unsafe fn to_dlpack_capsule(slice: &mut [Complex32], height: usize, width: usize) -> *mut DLManagedTensor {
    extern "C" fn release(ptr: *mut DLManagedTensor) {
        unsafe {
            if !(*ptr).manager_ctx.is_null() {
                let _ = Box::from_raw((*ptr).manager_ctx as *mut [i64; 2]);
            }
            let _ = Box::from_raw(ptr);
        }
    }

    let shape_box = Box::new([height as i64, width as i64]);
    let shape_ptr = Box::into_raw(shape_box);
    let tensor = Box::new(DLManagedTensor {
        dl_tensor: DLTensor {
            data: slice.as_mut_ptr() as *mut c_void,
            device: DLDevice {
                device_type: 1,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: 2,
                bits: 32,
                lanes: 1,
            },
            shape: shape_ptr as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        },
        manager_ctx: shape_ptr as *mut c_void,
        deleter: Some(release),
    });
    Box::into_raw(tensor)
}
