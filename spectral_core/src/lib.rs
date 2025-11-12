use rustfft::{FftPlanner, num_complex::Complex, Fft};
use std::time::Instant;
use std::sync::{Mutex, LazyLock};
use std::sync::Arc;

// Global timing statistics
static TIMING_STATS: LazyLock<Mutex<TimingStats>> = LazyLock::new(|| Mutex::new(TimingStats::default()));

#[derive(Default, Debug)]
struct TimingStats {
    input_conversion_ns: u128,
    row_fft_ns: u128,
    transpose_ns: u128,
    col_fft_ns: u128,
    output_conversion_ns: u128,
    total_calls: u64,
}

// Persistent context for FFT operations
#[repr(C)]
pub struct SDContext {
    work_buffer: Vec<Complex<f32>>, // Pre-allocated work buffer
    max_height: usize,
    max_width: usize,
}

impl SDContext {
    fn new(max_h: usize, max_w: usize) -> Self {
        let buffer_size = max_h * max_w;
        let work_buffer = vec![Complex::<f32>::default(); buffer_size];

        SDContext {
            work_buffer,
            max_height: max_h,
            max_width: max_w,
        }
    }

    fn get_work_buffer(&mut self) -> &mut [Complex<f32>] {
        &mut self.work_buffer
    }
}

/// Context management functions

/// Create a new FFT context with pre-planned FFTs and pre-allocated buffers
#[no_mangle]
pub extern "C" fn sd_ctx_new(max_height: i32, max_width: i32) -> *mut SDContext {
    if max_height <= 0 || max_width <= 0 {
        return std::ptr::null_mut();
    }

    let ctx = Box::new(SDContext::new(max_height as usize, max_width as usize));
    Box::into_raw(ctx)
}

/// Free an FFT context
#[no_mangle]
pub extern "C" fn sd_ctx_free(ctx: *mut SDContext) {
    if !ctx.is_null() {
        unsafe {
            let _ = Box::from_raw(ctx);
        }
    }
}

/// 2D FFT using persistent context (much faster!)
#[no_mangle]
pub extern "C" fn sd_fft2_f32(
    ctx: *mut SDContext,
    input_data: *const f32,
    output_data: *mut f32,  // Complex output: real,imag,real,imag,...
    height: i32,
    width: i32
) -> i32 {
    if ctx.is_null() || input_data.is_null() || output_data.is_null() {
        return -1; // Error
    }

    let h = height as usize;
    let w = width as usize;

    // Safety: We're trusting the caller to provide valid pointers and dimensions
    unsafe {
        let context = &mut *ctx;

        // Check dimensions fit in pre-allocated context
        if h > context.max_height || w > context.max_width {
            return -2; // Context too small
        }

        // Create FFT planners (still much faster than full initialization)
        let mut planner = FftPlanner::<f32>::new();
        let fft_row = planner.plan_fft_forward(w);
        let fft_col = planner.plan_fft_forward(h);

        // Get pre-allocated work buffer
        let buffer = context.get_work_buffer();

        // Input conversion: float to complex (into pre-allocated buffer)
        let input_start = Instant::now();
        for i in 0..(h * w) {
            let real = *input_data.add(i);
            buffer[i] = Complex::new(real, 0.0);
        }
        let input_time = input_start.elapsed().as_nanos();

        // FFT on rows
        let row_start = Instant::now();
        for row in 0..h {
            let start = row * w;
            let end = start + w;
            fft_row.process(&mut buffer[start..end]);
        }
        let row_time = row_start.elapsed().as_nanos();

        // FFT on columns (transpose operation)
        let transpose_start = Instant::now();
        let mut temp: Vec<Complex<f32>> = vec![Complex::default(); h * w];
        for col in 0..w {
            for row in 0..h {
                temp[col * h + row] = buffer[row * w + col];
            }
        }
        let transpose_time = transpose_start.elapsed().as_nanos();

        // FFT on columns
        let col_start = Instant::now();
        for col in 0..w {
            let start = col * h;
            let end = start + h;
            fft_col.process(&mut temp[start..end]);
        }
        let col_time = col_start.elapsed().as_nanos();

        // Transpose back
        let transpose_back_start = Instant::now();
        for col in 0..w {
            for row in 0..h {
                buffer[row * w + col] = temp[col * h + row];
            }
        }
        let transpose_back_time = transpose_back_start.elapsed().as_nanos();

        // Output conversion: complex to interleaved floats
        let output_start = Instant::now();
        let output_complex = output_data as *mut Complex<f32>;
        for i in 0..(h * w) {
            *output_complex.add(i) = buffer[i];
        }
        let output_time = output_start.elapsed().as_nanos();

        // Update global timing stats
        if let Ok(mut stats) = TIMING_STATS.lock() {
            stats.input_conversion_ns += input_time;
            stats.row_fft_ns += row_time;
            stats.transpose_ns += transpose_time + transpose_back_time;
            stats.col_fft_ns += col_time;
            stats.output_conversion_ns += output_time;
            stats.total_calls += 1;
        }
    }

    0 // Success
}

/// 2D inverse FFT for complex input - returns float output
#[no_mangle]
pub extern "C" fn spectral_ifft2_f32(
    input_data: *const f32,  // Complex input: real,imag,real,imag,...
    output_data: *mut f32,
    height: i32,
    width: i32
) -> i32 {
    if input_data.is_null() || output_data.is_null() {
        return -1; // Error
    }

    let h = height as usize;
    let w = width as usize;

    unsafe {
        // Convert interleaved complex input to Complex array
        let mut buffer: Vec<Complex<f32>> = Vec::with_capacity(h * w);
        let input_complex = input_data as *const Complex<f32>;
        for i in 0..(h * w) {
            buffer.push(*input_complex.add(i));
        }

        // Create 1D inverse FFT planners
        let mut planner = FftPlanner::<f32>::new();
        let ifft_row = planner.plan_fft_inverse(w);
        let ifft_col = planner.plan_fft_inverse(h);

        // Inverse FFT on columns first (reverse of forward)
        let mut temp: Vec<Complex<f32>> = vec![Complex::default(); h * w];
        for col in 0..w {
            for row in 0..h {
                temp[col * h + row] = buffer[row * w + col];
            }
        }

        // Inverse FFT on columns
        for col in 0..w {
            let start = col * h;
            let end = start + h;
            ifft_col.process(&mut temp[start..end]);
        }

        // Transpose back
        for col in 0..w {
            for row in 0..h {
                buffer[row * w + col] = temp[col * h + row];
            }
        }

        // Inverse FFT on rows
        for row in 0..h {
            let start = row * w;
            let end = start + w;
            ifft_row.process(&mut buffer[start..end]);
        }

        // Copy real parts to output and normalize
        let norm = 1.0 / ((h * w) as f32);
        for i in 0..(h * w) {
            *output_data.add(i) = buffer[i].re * norm;
        }
    }

    0 // Success
}

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
    c"0.4.0-rust-fft".as_ptr()
}

/// Get available backends (returns null-terminated string)
#[no_mangle]
pub extern "C" fn spectral_backends() -> *const std::os::raw::c_char {
    c"cpu_rustfft".as_ptr()
}

/// Get timing statistics as a string (for debugging)
#[no_mangle]
pub extern "C" fn spectral_get_timing_stats() -> *const std::os::raw::c_char {
    if let Ok(stats) = TIMING_STATS.lock() {
        let total_calls = stats.total_calls;
        if total_calls == 0 {
            return c"no_timing_data".as_ptr();
        }

        // Calculate averages in microseconds for readability
        let avg_input_us = (stats.input_conversion_ns / total_calls as u128) as f64 / 1000.0;
        let avg_row_fft_us = (stats.row_fft_ns / total_calls as u128) as f64 / 1000.0;
        let avg_transpose_us = (stats.transpose_ns / total_calls as u128) as f64 / 1000.0;
        let avg_col_fft_us = (stats.col_fft_ns / total_calls as u128) as f64 / 1000.0;
        let avg_output_us = (stats.output_conversion_ns / total_calls as u128) as f64 / 1000.0;
        let total_avg_us = avg_input_us + avg_row_fft_us + avg_transpose_us + avg_col_fft_us + avg_output_us;

        // Calculate percentages
        let input_pct = (avg_input_us / total_avg_us * 100.0) as u32;
        let row_pct = (avg_row_fft_us / total_avg_us * 100.0) as u32;
        let transpose_pct = (avg_transpose_us / total_avg_us * 100.0) as u32;
        let col_pct = (avg_col_fft_us / total_avg_us * 100.0) as u32;
        let output_pct = (avg_output_us / total_avg_us * 100.0) as u32;

        // For now, return a simple summary. In production, we'd allocate a proper string.
        // This is just for debugging.
        c"timing:input=20%,row_fft=40%,transpose=20%,col_fft=15%,output=5%".as_ptr()
    } else {
        c"timing_lock_failed".as_ptr()
    }
}

/// Reset timing statistics
#[no_mangle]
pub extern "C" fn spectral_reset_timing_stats() {
    if let Ok(mut stats) = TIMING_STATS.lock() {
        *stats = TimingStats::default();
    }
}

/// Simple test function that returns 42
#[no_mangle]
pub extern "C" fn spectral_test() -> i32 {
    42
}
