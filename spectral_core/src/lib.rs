use rustfft::{FftPlanner, num_complex::Complex};

/// Real 2D FFT implementation using rustfft
/// Implements 2D FFT by doing 1D FFTs on rows then columns

/// 2D FFT for float32 input - returns complex64 output
#[no_mangle]
pub extern "C" fn spectral_fft2_f32(
    input_data: *const f32,
    output_data: *mut f32,  // Complex output: real,imag,real,imag,...
    height: i32,
    width: i32
) -> i32 {
    if input_data.is_null() || output_data.is_null() {
        return -1; // Error
    }

    let h = height as usize;
    let w = width as usize;

    unsafe {
        // Convert input float array to complex 2D array
        let mut buffer: Vec<Complex<f32>> = Vec::with_capacity(h * w);
        for i in 0..(h * w) {
            let real = *input_data.add(i);
            buffer.push(Complex::new(real, 0.0));
        }

        // Create 1D FFT planners
        let mut planner = FftPlanner::<f32>::new();
        let fft_row = planner.plan_fft_forward(w);
        let fft_col = planner.plan_fft_forward(h);

        // FFT on rows
        for row in 0..h {
            let start = row * w;
            let end = start + w;
            fft_row.process(&mut buffer[start..end]);
        }

        // FFT on columns (transpose operation)
        let mut temp: Vec<Complex<f32>> = vec![Complex::default(); h * w];
        for col in 0..w {
            for row in 0..h {
                temp[col * h + row] = buffer[row * w + col];
            }
        }

        // FFT on columns
        for col in 0..w {
            let start = col * h;
            let end = start + h;
            fft_col.process(&mut temp[start..end]);
        }

        // Transpose back
        for col in 0..w {
            for row in 0..h {
                buffer[row * w + col] = temp[col * h + row];
            }
        }

        // Copy results to output (interleaved real/imag)
        let output_complex = output_data as *mut Complex<f32>;
        for i in 0..(h * w) {
            *output_complex.add(i) = buffer[i];
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

/// Simple test function that returns 42
#[no_mangle]
pub extern "C" fn spectral_test() -> i32 {
    42
}
