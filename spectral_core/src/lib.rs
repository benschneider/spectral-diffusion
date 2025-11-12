use std::ffi::c_void;

/// Simple C API for spectral operations
/// This provides a clean interface without PyO3 complexity

/// Initialize the spectral core (returns 0 on success)
#[no_mangle]
pub extern "C" fn spectral_init() -> i32 {
    0  // Success
}

/// Cleanup the spectral core
#[no_mangle]
pub extern "C" fn spectral_cleanup() {
    // Cleanup code here
}

/// Get version string
#[no_mangle]
pub extern "C" fn spectral_version() -> *const std::os::raw::c_char {
    c"0.3.0".as_ptr()
}

/// Get available backends (returns null-terminated string)
#[no_mangle]
pub extern "C" fn spectral_backends() -> *const std::os::raw::c_char {
    c"cpu_basic".as_ptr()
}

/// Simple test function that returns 42
#[no_mangle]
pub extern "C" fn spectral_test() -> i32 {
    42
}
