use crate::planner::PlanEntry;
use crate::types::{Complex, Result};
use crate::workspace;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use rayon::ThreadPool;
use std::env;

pub fn execute(plan: &PlanEntry, data: &mut [Complex]) -> Result<()> {
    let plane = plan.len;
    assert_eq!(data.len() % plane, 0, "data must contain whole planes");
    let batch = data.len() / plane;
    let width = plan.key.width;
    let height = plan.key.height;

    RAYON_POOL.install(|| {
        for b in 0..batch {
            let plane_slice = &mut data[b * plane..(b + 1) * plane];
            row_fft(plan, plane_slice, width);
            col_fft(plan, plane_slice, width, height);
        }
    });
    Ok(())
}

fn row_fft(plan: &PlanEntry, plane: &mut [Complex], width: usize) {
    plane
        .par_chunks_mut(width)
        .for_each(|row| plan.row_fft.process(row));
}

fn col_fft(plan: &PlanEntry, plane: &mut [Complex], width: usize, height: usize) {
    let plane_addr = plane.as_mut_ptr() as usize;
    (0..width).into_par_iter().for_each(move |col| {
        let plane_ptr = plane_addr as *mut Complex;
        workspace::with_tls_scratch(height, move |scratch| {
            for row in 0..height {
                unsafe {
                    scratch[row] = *plane_ptr.add(row * width + col);
                }
            }
            plan.col_fft.process(&mut scratch[..height]);
            for row in 0..height {
                unsafe {
                    *plane_ptr.add(row * width + col) = scratch[row];
                }
            }
        });
    });
}

static RAYON_POOL: Lazy<ThreadPool> = Lazy::new(|| {
    let builder = rayon::ThreadPoolBuilder::new();
    let builder = match env::var("RUSTFFT_THREADS").ok().and_then(|v| v.parse().ok()) {
        Some(threads) if threads > 0 => builder.num_threads(threads),
        _ => builder,
    };
    builder.build().expect("failed to build Rayon pool")
});
