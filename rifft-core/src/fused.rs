use crate::fft2d;
use crate::planner::PlanEntry;
use crate::simd;
use crate::types::{Result, RifftError};

pub fn fft_filter_ifft(
    plan_forward: &PlanEntry,
    plan_inverse: &PlanEntry,
    data: &mut [crate::types::Complex],
    filter: &[crate::types::Complex],
) -> Result<()> {
    if data.len() != filter.len() {
        return Err(RifftError::ShapeMismatch {
            expected: data.len(),
            got: filter.len(),
        });
    }
    fft2d::execute(plan_forward, data)?;
    let mut filter_freq = filter.to_vec();
    fft2d::execute(plan_forward, &mut filter_freq)?;
    simd::complex_mul_inplace(data, &filter_freq);
    fft2d::execute(plan_inverse, data)?;
    let norm = plan_forward.len as f32;
    if norm > 0.0 {
        for value in data.iter_mut() {
            value.re /= norm;
            value.im /= norm;
        }
    }
    Ok(())
}
