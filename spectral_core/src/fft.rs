use crate::plan::{Plan2D, PlanCache, PlanDirection, SmallPlanCache};
use num_cpus;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rustfft::num_complex::Complex32;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cell::RefCell;
use std::env;
use std::ptr::NonNull;
use std::slice;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

const ALIGN_BYTES: usize = 64;
const BLOCK_SIZE_SMALL: usize = 32;
const BLOCK_SIZE_LARGE: usize = 64;
const PARALLEL_ROW_THRESHOLD: usize = 8;
const PARALLEL_COL_THRESHOLD: usize = 8;
const DEFAULT_TRANSPOSE_MIN_DIM: usize = usize::MAX;

thread_local! {
    static TLS_COMPLEX: RefCell<Vec<Complex32>> = RefCell::new(Vec::new());
}

#[derive(Clone, Copy)]
struct RawMutPtr<T>(*mut T);

#[derive(Clone, Copy)]
struct RawConstPtr<T>(*const T);

unsafe impl<T: Send> Send for RawMutPtr<T> {}
unsafe impl<T: Send> Sync for RawMutPtr<T> {}

unsafe impl<T: Sync> Send for RawConstPtr<T> {}
unsafe impl<T: Sync> Sync for RawConstPtr<T> {}

#[derive(Clone, Copy)]
struct BlockTransposer {
    src: RawConstPtr<Complex32>,
    dst: RawMutPtr<Complex32>,
    rows: usize,
    cols: usize,
    block: usize,
    col_blocks: usize,
}

impl BlockTransposer {
    fn run(&self, block_idx: usize) {
        let rb = block_idx / self.col_blocks;
        let cb = block_idx % self.col_blocks;
        let row_start = rb * self.block;
        let col_start = cb * self.block;
        let row_end = (row_start + self.block).min(self.rows);
        let col_end = (col_start + self.block).min(self.cols);
        for r in row_start..row_end {
            for c in col_start..col_end {
                let src_idx = r * self.cols + c;
                let dst_idx = c * self.rows + r;
                unsafe {
                    *self.dst.0.add(dst_idx) = *self.src.0.add(src_idx);
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Dimensions {
    pub height: usize,
    pub width: usize,
}

impl Dimensions {
    pub fn elements(&self) -> usize {
        self.height * self.width
    }
}

#[derive(Default, Clone, Copy)]
pub struct PhaseTimings {
    pub plan: Duration,
    pub conversion_in: Duration,
    pub conversion_out: Duration,
    pub row_fft: Duration,
    pub col_fft: Duration,
    pub transpose: Duration,
    pub transpose_back: Duration,
    pub elementwise: Duration,
}

impl PhaseTimings {
    pub fn total(&self) -> Duration {
        self.plan
            + self.conversion_in
            + self.conversion_out
            + self.row_fft
            + self.col_fft
            + self.transpose
            + self.transpose_back
            + self.elementwise
    }
}

fn resolve_plan(height: usize, width: usize, direction: PlanDirection) -> (Arc<Plan2D>, Duration) {
    if let Some(plan) = SmallPlanCache::global().get(height, width, direction) {
        (plan, Duration::ZERO)
    } else {
        PlanCache::global().get_or_build(height, width, direction)
    }
}

pub struct FftEngine {
    max_height: usize,
    max_width: usize,
    work: AlignedBuffer<Complex32>,
    scratch: AlignedBuffer<Complex32>,
    filter: AlignedBuffer<Complex32>,
    transpose_enabled: bool,
}

impl FftEngine {
    pub fn new(max_height: usize, max_width: usize) -> Self {
        let total = max_height.max(1) * max_width.max(1);
        Self {
            max_height,
            max_width,
            work: AlignedBuffer::with_capacity(total),
            scratch: AlignedBuffer::with_capacity(total),
            filter: AlignedBuffer::with_capacity(total),
            transpose_enabled: transpose_enabled(),
        }
    }

    #[inline]
    pub fn supports(&self, dims: Dimensions) -> bool {
        dims.height <= self.max_height && dims.width <= self.max_width
    }

    #[inline]
    fn use_transpose_for(&self, dims: Dimensions) -> bool {
        if !self.transpose_enabled {
            return false;
        }
        dims.height.max(dims.width) >= transpose_threshold()
    }

    pub fn fft2_real_to_complex(
        &mut self,
        input: &[f32],
        output: &mut [Complex32],
        dims: Dimensions,
    ) -> PhaseTimings {
        debug_assert_eq!(input.len(), dims.elements());
        debug_assert_eq!(output.len(), dims.elements());

        let mut timings = PhaseTimings::default();
        let (plan, elapsed) = resolve_plan(dims.height, dims.width, PlanDirection::Forward);
        timings.plan += elapsed;

        let total = dims.elements();
        let use_transpose = self.use_transpose_for(dims);
        self.work.ensure_len(total);
        self.scratch.ensure_len(total);

        let work = self.work.as_mut_slice();
        let scratch = self.scratch.as_mut_slice();
        timings.conversion_in += convert_real_to_complex(input, work);
        execute_fft(use_transpose, work, scratch, dims, &plan, &mut timings);
        timings.conversion_out += copy_complex(work, output);

        log_debug("fft2", dims, &timings);
        timings
    }

    pub fn ifft2_complex(
        &mut self,
        input: &[Complex32],
        output: &mut [Complex32],
        dims: Dimensions,
    ) -> PhaseTimings {
        debug_assert_eq!(input.len(), dims.elements());
        debug_assert_eq!(output.len(), dims.elements());

        let mut timings = PhaseTimings::default();
        let (plan, elapsed) = resolve_plan(dims.height, dims.width, PlanDirection::Inverse);
        timings.plan += elapsed;

        let total = dims.elements();
        let use_transpose = self.use_transpose_for(dims);
        self.work.ensure_len(total);
        self.scratch.ensure_len(total);
        let work = self.work.as_mut_slice();
        let scratch = self.scratch.as_mut_slice();
        work.copy_from_slice(input);

        execute_fft(use_transpose, work, scratch, dims, &plan, &mut timings);

        let scale = 1.0 / dims.elements() as f32;
        timings.elementwise += scale_complex(work, scale);
        timings.conversion_out += copy_complex(work, output);

        log_debug("ifft2", dims, &timings);
        timings
    }

    pub fn fft_filter_ifft(
        &mut self,
        signal: &[f32],
        kernel: &[f32],
        output: &mut [f32],
        dims: Dimensions,
    ) -> PhaseTimings {
        debug_assert_eq!(signal.len(), dims.elements());
        debug_assert_eq!(kernel.len(), dims.elements());
        debug_assert_eq!(output.len(), dims.elements());

        let mut timings = PhaseTimings::default();
        let (forward_plan, f_elapsed) =
            resolve_plan(dims.height, dims.width, PlanDirection::Forward);
        let (inverse_plan, i_elapsed) =
            resolve_plan(dims.height, dims.width, PlanDirection::Inverse);
        timings.plan += f_elapsed + i_elapsed;

        let total = dims.elements();
        let use_transpose = self.use_transpose_for(dims);
        self.work.ensure_len(total);
        self.scratch.ensure_len(total);

        self.filter.ensure_len(total);
        let filter_slice = self.filter.as_mut_slice();
        let work = self.work.as_mut_slice();
        let scratch = self.scratch.as_mut_slice();
        timings.conversion_in += convert_real_to_complex(signal, work);
        timings.conversion_in += convert_real_to_complex(kernel, filter_slice);

        execute_fft(
            use_transpose,
            filter_slice,
            scratch,
            dims,
            &forward_plan,
            &mut timings,
        );
        execute_fft(
            use_transpose,
            work,
            scratch,
            dims,
            &forward_plan,
            &mut timings,
        );

        timings.elementwise += multiply_inplace(work, filter_slice);
        execute_fft(
            use_transpose,
            work,
            scratch,
            dims,
            &inverse_plan,
            &mut timings,
        );

        let scale = 1.0 / dims.elements() as f32;
        timings.elementwise += scale_complex(work, scale);
        timings.conversion_out += complex_to_real_scaled(work, output, 1.0);
        log_debug("fft_filter_ifft", dims, &timings);
        timings
    }
}

fn execute_fft(
    use_transpose: bool,
    data: &mut [Complex32],
    scratch: &mut [Complex32],
    dims: Dimensions,
    plan: &Plan2D,
    timings: &mut PhaseTimings,
) {
    configure_rayon_pool();

    timings.row_fft += process_rows(data, dims.width, plan.row_plan.clone());
    if use_transpose {
        timings.transpose += blocked_transpose(&*data, scratch, dims.height, dims.width);
        timings.col_fft += process_rows(scratch, dims.height, plan.col_plan.clone());
        timings.transpose_back += blocked_transpose(scratch, data, dims.width, dims.height);
    } else {
        timings.col_fft += process_columns_tls(data, dims, plan.col_plan.clone());
    }
}

fn convert_real_to_complex(input: &[f32], output: &mut [Complex32]) -> Duration {
    let start = Instant::now();
    #[cfg(feature = "simd_avx2")]
    {
        simd::real_to_complex(input, output);
    }
    #[cfg(not(feature = "simd_avx2"))]
    {
        for (dst, &value) in output.iter_mut().zip(input.iter()) {
            dst.re = value;
            dst.im = 0.0;
        }
    }
    start.elapsed()
}

fn complex_to_real_scaled(input: &[Complex32], output: &mut [f32], scale: f32) -> Duration {
    let start = Instant::now();
    #[cfg(feature = "simd_avx2")]
    {
        simd::complex_to_real_scaled(input, output, scale);
    }
    #[cfg(not(feature = "simd_avx2"))]
    {
        for (dst, value) in output.iter_mut().zip(input.iter()) {
            *dst = value.re * scale;
        }
    }
    start.elapsed()
}

fn copy_complex(input: &[Complex32], output: &mut [Complex32]) -> Duration {
    let start = Instant::now();
    output.copy_from_slice(input);
    start.elapsed()
}

fn multiply_inplace(lhs: &mut [Complex32], rhs: &[Complex32]) -> Duration {
    let start = Instant::now();
    #[cfg(feature = "simd_avx2")]
    {
        simd::mul_complex_inplace(lhs, rhs);
    }
    #[cfg(not(feature = "simd_avx2"))]
    {
        for (a, b) in lhs.iter_mut().zip(rhs.iter()) {
            let re = a.re * b.re - a.im * b.im;
            let im = a.re * b.im + a.im * b.re;
            a.re = re;
            a.im = im;
        }
    }
    start.elapsed()
}

fn scale_complex(data: &mut [Complex32], scale: f32) -> Duration {
    let start = Instant::now();
    #[cfg(feature = "simd_avx2")]
    {
        simd::scale_complex(data, scale);
    }
    #[cfg(not(feature = "simd_avx2"))]
    {
        for value in data.iter_mut() {
            value.re *= scale;
            value.im *= scale;
        }
    }
    start.elapsed()
}

fn process_rows(
    data: &mut [Complex32],
    width: usize,
    plan: std::sync::Arc<dyn rustfft::Fft<f32> + Send + Sync>,
) -> Duration {
    let start = Instant::now();
    let rows = if width == 0 { 0 } else { data.len() / width };
    if rows >= PARALLEL_ROW_THRESHOLD {
        data.par_chunks_mut(width).for_each(|row| {
            plan.process(row);
        });
    } else {
        for row in data.chunks_mut(width) {
            plan.process(row);
        }
    }
    start.elapsed()
}

fn process_columns_tls(
    data: &mut [Complex32],
    dims: Dimensions,
    plan: std::sync::Arc<dyn rustfft::Fft<f32> + Send + Sync>,
) -> Duration {
    let start = Instant::now();
    let height = dims.height;
    let width = dims.width;
    let stride = width;
    let data_ptr = RawMutPtr(data.as_mut_ptr());

    let process_col = |col: usize, data_ptr: RawMutPtr<Complex32>| {
        TLS_COMPLEX.with(|cell| {
            let mut buffer = cell.borrow_mut();
            if buffer.len() < height {
                buffer.resize(height, Complex32::default());
            }
            let slice = &mut buffer[..height];
            for row in 0..height {
                unsafe {
                    slice[row] = *data_ptr.0.add(row * stride + col);
                }
            }
            plan.process(slice);
            for row in 0..height {
                unsafe {
                    *data_ptr.0.add(row * stride + col) = slice[row];
                }
            }
        });
    };

    if width >= PARALLEL_COL_THRESHOLD {
        let ptr = data_ptr;
        (0..width)
            .into_par_iter()
            .for_each(move |col| process_col(col, ptr));
    } else {
        for col in 0..width {
            process_col(col, data_ptr);
        }
    }

    start.elapsed()
}

fn blocked_transpose(
    src: &[Complex32],
    dst: &mut [Complex32],
    rows: usize,
    cols: usize,
) -> Duration {
    let start = Instant::now();
    let max_dim = rows.max(cols);
    let block = if max_dim >= 512 {
        BLOCK_SIZE_LARGE
    } else {
        BLOCK_SIZE_SMALL
    };
    let block = block.max(1);
    let row_blocks = (rows + block - 1) / block;
    let col_blocks = (cols + block - 1) / block;
    let total_blocks = row_blocks * col_blocks;

    let transposer = BlockTransposer {
        src: RawConstPtr(src.as_ptr()),
        dst: RawMutPtr(dst.as_mut_ptr()),
        rows,
        cols,
        block,
        col_blocks,
    };

    if max_dim >= 512 {
        (0..total_blocks)
            .into_par_iter()
            .for_each(|idx| transposer.run(idx));
    } else {
        for block_idx in 0..total_blocks {
            transposer.run(block_idx);
        }
    }
    start.elapsed()
}

#[derive(Debug)]
struct AlignedBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}

impl<T> AlignedBuffer<T> {
    fn with_capacity(capacity: usize) -> Self {
        let ptr = if capacity == 0 {
            NonNull::dangling()
        } else {
            allocate_aligned::<T>(capacity)
        };
        Self {
            ptr,
            len: capacity,
            capacity,
        }
    }

    fn ensure_len(&mut self, len: usize) {
        if len > self.capacity {
            if self.capacity != 0 {
                unsafe { deallocate_aligned::<T>(self.ptr, self.capacity) };
            }
            self.ptr = allocate_aligned::<T>(len);
            self.capacity = len;
        }
        self.len = len;
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        if self.capacity != 0 {
            unsafe { deallocate_aligned::<T>(self.ptr, self.capacity) };
        }
    }
}

unsafe impl<T: Send> Send for AlignedBuffer<T> {}
unsafe impl<T: Sync> Sync for AlignedBuffer<T> {}

fn allocate_aligned<T>(len: usize) -> NonNull<T> {
    let layout = Layout::from_size_align(
        std::mem::size_of::<T>() * len,
        ALIGN_BYTES.max(std::mem::align_of::<T>()),
    )
    .expect("invalid layout");
    let ptr = unsafe { alloc_zeroed(layout) };
    NonNull::new(ptr as *mut T).expect("allocation failed")
}

unsafe fn deallocate_aligned<T>(ptr: NonNull<T>, len: usize) {
    if len == 0 {
        return;
    }
    let layout = Layout::from_size_align(
        std::mem::size_of::<T>() * len,
        ALIGN_BYTES.max(std::mem::align_of::<T>()),
    )
    .expect("invalid layout");
    dealloc(ptr.as_ptr() as *mut u8, layout);
}

fn configure_rayon_pool() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        let requested = env::var("RUSTFFT_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or_else(num_cpus::get);
        if let Err(err) = ThreadPoolBuilder::new()
            .num_threads(requested)
            .build_global()
        {
            if debug_enabled() {
                eprintln!("RUSTFFT thread pool already initialised: {err}");
            }
        }
        requested
    })
}

fn transpose_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| match env::var("RUSTFFT_TRANSPOSE") {
        Ok(value) => value != "0" && !value.eq_ignore_ascii_case("false"),
        Err(_) => true,
    })
}

fn debug_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| match env::var("RUSTFFT_DEBUG") {
        Ok(value) => value == "1" || value.eq_ignore_ascii_case("true"),
        Err(_) => false,
    })
}

fn transpose_threshold() -> usize {
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        env::var("RUSTFFT_TRANSPOSE_MIN")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_TRANSPOSE_MIN_DIM)
    })
}

fn log_debug(label: &str, dims: Dimensions, timings: &PhaseTimings) {
    if !debug_enabled() {
        return;
    }
    let h = dims.height;
    let w = dims.width;
    eprintln!(
        "[RUSTFFT] {label} {h}x{w}: total={:.3}ms plan={:.3} conv_in={:.3} conv_out={:.3} row={:.3} col={:.3} trans={:.3} trans_back={:.3} mul={:.3}",
        timings.total().as_secs_f64() * 1000.0,
        timings.plan.as_secs_f64() * 1000.0,
        timings.conversion_in.as_secs_f64() * 1000.0,
        timings.conversion_out.as_secs_f64() * 1000.0,
        timings.row_fft.as_secs_f64() * 1000.0,
        timings.col_fft.as_secs_f64() * 1000.0,
        timings.transpose.as_secs_f64() * 1000.0,
        timings.transpose_back.as_secs_f64() * 1000.0,
        timings.elementwise.as_secs_f64() * 1000.0,
    );
}

#[cfg(feature = "simd_avx2")]
mod simd {
    use super::*;
    use core::simd::{simd_swizzle, Simd};

    const COMPLEX_LANES: usize = 4;
    const FLOAT_LANES: usize = COMPLEX_LANES * 2;

    pub fn real_to_complex(input: &[f32], output: &mut [Complex32]) {
        let floats = input.len();
        let dst = complex_as_f32_mut(output);
        let mut i = 0;
        let mut o = 0;
        while i + COMPLEX_LANES <= floats {
            let re = Simd::<f32, COMPLEX_LANES>::from_slice(&input[i..i + COMPLEX_LANES]);
            let zeros = Simd::<f32, COMPLEX_LANES>::splat(0.0);
            let interleaved = simd_swizzle!(re, zeros, [0, 4, 1, 5, 2, 6, 3, 7]);
            dst[o..o + FLOAT_LANES].copy_from_slice(&interleaved.to_array());
            i += COMPLEX_LANES;
            o += FLOAT_LANES;
        }
        while i < floats {
            output[i].re = input[i];
            output[i].im = 0.0;
            i += 1;
        }
    }

    pub fn complex_to_real_scaled(input: &[Complex32], output: &mut [f32], scale: f32) {
        let floats = input.len();
        let src = complex_as_f32(input);
        let mut i = 0;
        let mut o = 0;
        while o + COMPLEX_LANES <= floats {
            let chunk = Simd::<f32, FLOAT_LANES>::from_slice(&src[i..i + FLOAT_LANES]);
            let re = simd_swizzle!(chunk, [0, 2, 4, 6]);
            let scaled = re * Simd::splat(scale);
            output[o..o + COMPLEX_LANES].copy_from_slice(&scaled.to_array());
            i += FLOAT_LANES;
            o += COMPLEX_LANES;
        }
        while o < floats {
            output[o] = input[o].re * scale;
            o += 1;
        }
    }

    pub fn mul_complex_inplace(lhs: &mut [Complex32], rhs: &[Complex32]) {
        let len = lhs.len().min(rhs.len());
        let lhs_f32 = complex_as_f32_mut(lhs);
        let rhs_f32 = complex_as_f32(rhs);
        let mut idx = 0;
        while idx + FLOAT_LANES <= len * 2 {
            let a = Simd::<f32, FLOAT_LANES>::from_slice(&lhs_f32[idx..idx + FLOAT_LANES]);
            let b = Simd::<f32, FLOAT_LANES>::from_slice(&rhs_f32[idx..idx + FLOAT_LANES]);
            let are = simd_swizzle!(a, [0, 2, 4, 6]);
            let aim = simd_swizzle!(a, [1, 3, 5, 7]);
            let bre = simd_swizzle!(b, [0, 2, 4, 6]);
            let bim = simd_swizzle!(b, [1, 3, 5, 7]);
            let out_re = are * bre - aim * bim;
            let out_im = are * bim + aim * bre;
            let interleaved = simd_swizzle!(out_re, out_im, [0, 4, 1, 5, 2, 6, 3, 7]);
            lhs_f32[idx..idx + FLOAT_LANES].copy_from_slice(&interleaved.to_array());
            idx += FLOAT_LANES;
        }
        let mut complex_idx = idx / 2;
        while complex_idx < len {
            let a = lhs[complex_idx];
            let b = rhs[complex_idx];
            lhs[complex_idx].re = a.re * b.re - a.im * b.im;
            lhs[complex_idx].im = a.re * b.im + a.im * b.re;
            complex_idx += 1;
        }
    }

    pub fn scale_complex(data: &mut [Complex32], scale: f32) {
        let slice = complex_as_f32_mut(data);
        let mut idx = 0;
        let lanes = Simd::<f32, FLOAT_LANES>::splat(scale);
        while idx + FLOAT_LANES <= slice.len() {
            let chunk = Simd::<f32, FLOAT_LANES>::from_slice(&slice[idx..idx + FLOAT_LANES]);
            let scaled = chunk * lanes;
            slice[idx..idx + FLOAT_LANES].copy_from_slice(&scaled.to_array());
            idx += FLOAT_LANES;
        }
        let mut complex_idx = idx / 2;
        while complex_idx < data.len() {
            data[complex_idx].re *= scale;
            data[complex_idx].im *= scale;
            complex_idx += 1;
        }
    }

    fn complex_as_f32(data: &[Complex32]) -> &[f32] {
        unsafe { slice::from_raw_parts(data.as_ptr() as *const f32, data.len() * 2) }
    }

    fn complex_as_f32_mut(data: &mut [Complex32]) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, data.len() * 2) }
    }
}
