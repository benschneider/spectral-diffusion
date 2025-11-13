use crate::types::Result;
use crate::RifftHandle;
use libc::c_char;
use num_complex::Complex32;
use once_cell::sync::Lazy;
use std::ffi::CString;
use std::os::raw::c_int;

const SUCCESS: c_int = 0;
const FAILURE: c_int = -1;

#[no_mangle]
pub extern "C" fn riff_create_handle() -> *mut RifftHandle {
    Box::into_raw(Box::new(RifftHandle::new()))
}

#[no_mangle]
pub unsafe extern "C" fn riff_free_handle(handle: *mut RifftHandle) {
    if handle.is_null() {
        return;
    }
    drop(Box::from_raw(handle));
}

#[no_mangle]
pub unsafe extern "C" fn riff_fft2d_forward(
    handle: *mut RifftHandle,
    data: *mut Complex32,
    height: usize,
    width: usize,
) -> c_int {
    call_fft(handle, data, height, width, |handle, slice| {
        handle.fft2d_forward(slice, height, width)
    })
}

#[no_mangle]
pub unsafe extern "C" fn riff_fft2d_inverse(
    handle: *mut RifftHandle,
    data: *mut Complex32,
    height: usize,
    width: usize,
) -> c_int {
    call_fft(handle, data, height, width, |handle, slice| {
        handle.fft2d_inverse(slice, height, width)
    })
}

#[no_mangle]
pub unsafe extern "C" fn riff_fft2d_fused_filter(
    handle: *mut RifftHandle,
    data: *mut Complex32,
    filter: *const Complex32,
    height: usize,
    width: usize,
) -> c_int {
    if filter.is_null() {
        return FAILURE;
    }
    call_fft(handle, data, height, width, |handle, slice| {
        let total = height * width;
        let filter_slice = std::slice::from_raw_parts(filter, total);
        handle.fft_filter_ifft(slice, filter_slice, height, width)
    })
}

static VERSION_CSTR: Lazy<CString> =
    Lazy::new(|| CString::new(env!("CARGO_PKG_VERSION")).expect("valid version"));
static BACKEND_CSTR: Lazy<CString> =
    Lazy::new(|| CString::new(crate::types::BACKEND_NAME).expect("valid backend name"));

#[no_mangle]
pub extern "C" fn riff_get_version() -> *const c_char {
    VERSION_CSTR.as_ptr()
}

#[no_mangle]
pub extern "C" fn riff_get_backend_name() -> *const c_char {
    BACKEND_CSTR.as_ptr()
}

unsafe fn call_fft<F>(
    handle: *mut RifftHandle,
    data: *mut Complex32,
    height: usize,
    width: usize,
    f: F,
) -> c_int
where
    F: FnOnce(&mut RifftHandle, &mut [Complex32]) -> Result<()>,
{
    if handle.is_null() || data.is_null() {
        return FAILURE;
    }
    let len = height * width;
    let slice = std::slice::from_raw_parts_mut(data, len);
    let handle_ref = &mut *handle;
    match f(handle_ref, slice) {
        Ok(_) => SUCCESS,
        Err(err) => {
            log::error!("RIFFT call failed: {:?}", err);
            FAILURE
        }
    }
}
