#![cfg_attr(feature = "simd_avx2", feature(portable_simd))]

mod fft;
mod plan;

use fft::{Dimensions, FftEngine, PhaseTimings};
use once_cell::sync::Lazy;
use rustfft::num_complex::Complex32;
use std::ffi::CString;
use std::os::raw::{c_char, c_float};
use std::slice;
use std::sync::Mutex;

const VERSION: &str = "0.5.0-rustfft-opt";
const BACKEND: &str = "cpu_rustfft_optimized";

#[repr(C)]
pub struct SDContext {
    engine: FftEngine,
}

impl SDContext {
    fn new(max_height: usize, max_width: usize) -> Self {
        Self {
            engine: FftEngine::new(max_height, max_width),
        }
    }
}

#[derive(Default, Debug)]
struct TimingStats {
    plan_ns: u128,
    conversion_in_ns: u128,
    conversion_out_ns: u128,
    row_fft_ns: u128,
    col_fft_ns: u128,
    transpose_ns: u128,
    transpose_back_ns: u128,
    elementwise_ns: u128,
    total_calls: u64,
}

impl TimingStats {
    fn record(&mut self, timings: &PhaseTimings) {
        self.plan_ns += timings.plan.as_nanos();
        self.conversion_in_ns += timings.conversion_in.as_nanos();
        self.conversion_out_ns += timings.conversion_out.as_nanos();
        self.row_fft_ns += timings.row_fft.as_nanos();
        self.col_fft_ns += timings.col_fft.as_nanos();
        self.transpose_ns += timings.transpose.as_nanos();
        self.transpose_back_ns += timings.transpose_back.as_nanos();
        self.elementwise_ns += timings.elementwise.as_nanos();
        self.total_calls += 1;
    }

    fn reset(&mut self) {
        *self = TimingStats::default();
    }
}

static TIMING_STATS: Lazy<Mutex<TimingStats>> = Lazy::new(|| Mutex::new(TimingStats::default()));
static VERSION_CSTR: Lazy<CString> = Lazy::new(|| CString::new(VERSION).unwrap());
static BACKEND_CSTR: Lazy<CString> = Lazy::new(|| CString::new(BACKEND).unwrap());
static NO_DATA_CSTR: Lazy<CString> = Lazy::new(|| CString::new("no_timing_data").unwrap());

fn record_timings(timings: PhaseTimings) {
    if let Ok(mut stats) = TIMING_STATS.lock() {
        stats.record(&timings);
    }
}

#[no_mangle]
pub extern "C" fn spectral_init() -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn spectral_cleanup() {}

#[no_mangle]
pub extern "C" fn spectral_version() -> *const c_char {
    VERSION_CSTR.as_ptr()
}

#[no_mangle]
pub extern "C" fn spectral_backends() -> *const c_char {
    BACKEND_CSTR.as_ptr()
}

#[no_mangle]
pub extern "C" fn spectral_test() -> i32 {
    42
}

#[no_mangle]
pub extern "C" fn spectral_fft2_marker() -> c_float {
    1e-9
}

#[no_mangle]
pub extern "C" fn sd_ctx_new(max_height: i32, max_width: i32) -> *mut SDContext {
    if max_height <= 0 || max_width <= 0 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(SDContext::new(
        max_height as usize,
        max_width as usize,
    )))
}

#[no_mangle]
pub extern "C" fn sd_ctx_free(ctx: *mut SDContext) {
    if ctx.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(ctx));
    }
}

#[no_mangle]
pub extern "C" fn sd_fft2_f32(
    ctx: *mut SDContext,
    input_ptr: *const f32,
    output_ptr: *mut f32,
    height: i32,
    width: i32,
) -> i32 {
    if ctx.is_null() || input_ptr.is_null() || output_ptr.is_null() || height <= 0 || width <= 0 {
        return -1;
    }
    let dims = Dimensions {
        height: height as usize,
        width: width as usize,
    };
    let ctx = unsafe { &mut *ctx };
    if !ctx.engine.supports(dims) {
        return -2;
    }

    let len = dims.elements();
    let input = unsafe { slice::from_raw_parts(input_ptr, len) };
    let output = unsafe { slice::from_raw_parts_mut(output_ptr as *mut Complex32, len) };
    let timings = ctx.engine.fft2_real_to_complex(input, output, dims);
    record_timings(timings);
    0
}

#[no_mangle]
pub extern "C" fn spectral_ifft2_f32(
    ctx: *mut SDContext,
    input_ptr: *const f32,
    output_ptr: *mut f32,
    height: i32,
    width: i32,
) -> i32 {
    if ctx.is_null() || input_ptr.is_null() || output_ptr.is_null() || height <= 0 || width <= 0 {
        return -1;
    }
    let dims = Dimensions {
        height: height as usize,
        width: width as usize,
    };
    let ctx = unsafe { &mut *ctx };
    if !ctx.engine.supports(dims) {
        return -2;
    }

    let len = dims.elements();
    let input = unsafe { slice::from_raw_parts(input_ptr as *const Complex32, len) };
    let output = unsafe { slice::from_raw_parts_mut(output_ptr as *mut Complex32, len) };
    let timings = ctx.engine.ifft2_complex(input, output, dims);
    record_timings(timings);
    0
}

#[no_mangle]
pub extern "C" fn sd_fft_filter_ifft_f32(
    ctx: *mut SDContext,
    signal_ptr: *const f32,
    kernel_ptr: *const f32,
    output_ptr: *mut f32,
    height: i32,
    width: i32,
) -> i32 {
    if ctx.is_null()
        || signal_ptr.is_null()
        || kernel_ptr.is_null()
        || output_ptr.is_null()
        || height <= 0
        || width <= 0
    {
        return -1;
    }
    let dims = Dimensions {
        height: height as usize,
        width: width as usize,
    };
    let ctx = unsafe { &mut *ctx };
    if !ctx.engine.supports(dims) {
        return -2;
    }

    let len = dims.elements();
    let signal = unsafe { slice::from_raw_parts(signal_ptr, len) };
    let kernel = unsafe { slice::from_raw_parts(kernel_ptr, len) };
    let output = unsafe { slice::from_raw_parts_mut(output_ptr, len) };
    let timings = ctx.engine.fft_filter_ifft(signal, kernel, output, dims);
    record_timings(timings);
    0
}

#[no_mangle]
pub extern "C" fn spectral_get_timing_stats() -> *const c_char {
    if let Ok(stats) = TIMING_STATS.lock() {
        if stats.total_calls == 0 {
            return NO_DATA_CSTR.as_ptr();
        }
        let calls = stats.total_calls;
        let fft_compute = avg_seconds(stats.row_fft_ns + stats.col_fft_ns, calls);
        let data_transfer = avg_seconds(stats.conversion_in_ns + stats.conversion_out_ns, calls);
        let data_movement = avg_seconds(stats.transpose_ns + stats.transpose_back_ns, calls);
        let plan = avg_seconds(stats.plan_ns, calls);
        let elementwise = avg_seconds(stats.elementwise_ns, calls);
        let payload = format!(
            "fft_compute={:.6e}, data_transfer={:.6e}, data_movement={:.6e}, plan={:.6e}, elementwise={:.6e}, calls={}",
            fft_compute,
            data_transfer,
            data_movement,
            plan,
            elementwise,
            calls
        );
        return leak_string(payload);
    }
    NO_DATA_CSTR.as_ptr()
}

#[no_mangle]
pub extern "C" fn spectral_reset_timing_stats() {
    if let Ok(mut stats) = TIMING_STATS.lock() {
        stats.reset();
    }
}

fn leak_string(payload: String) -> *const c_char {
    match CString::new(payload) {
        Ok(cstr) => cstr.into_raw() as *const c_char,
        Err(_) => NO_DATA_CSTR.as_ptr(),
    }
}

fn avg_seconds(ns: u128, calls: u64) -> f64 {
    if calls == 0 {
        0.0
    } else {
        ns as f64 / calls as f64 / 1_000_000_000.0
    }
}
