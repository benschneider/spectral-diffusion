use num_complex::Complex32;

#[cfg(feature = "simd_avx2")]
use core::simd::Simd;

#[cfg(feature = "simd_avx2")]
type F32x4 = Simd<f32, 4>;

pub fn complex_mul_inplace(a: &mut [Complex32], b: &[Complex32]) {
    assert_eq!(a.len(), b.len());
    #[cfg(feature = "simd_avx2")]
    {
        let mut i = 0;
        while i + 4 <= a.len() {
            let are = F32x4::from_array([
                a[i].re,
                a[i + 1].re,
                a[i + 2].re,
                a[i + 3].re,
            ]);
            let aim = F32x4::from_array([
                a[i].im,
                a[i + 1].im,
                a[i + 2].im,
                a[i + 3].im,
            ]);
            let bre = F32x4::from_array([
                b[i].re,
                b[i + 1].re,
                b[i + 2].re,
                b[i + 3].re,
            ]);
            let bim = F32x4::from_array([
                b[i].im,
                b[i + 1].im,
                b[i + 2].im,
                b[i + 3].im,
            ]);
            let out_re = are * bre - aim * bim;
            let out_im = are * bim + aim * bre;
            let re_arr = out_re.to_array();
            let im_arr = out_im.to_array();
            for lane in 0..4 {
                a[i + lane].re = re_arr[lane];
                a[i + lane].im = im_arr[lane];
            }
            i += 4;
        }
        for k in i..a.len() {
            a[k] *= b[k];
        }
    }
    #[cfg(not(feature = "simd_avx2"))]
    {
        for (lhs, rhs) in a.iter_mut().zip(b.iter()) {
            *lhs *= *rhs;
        }
    }
}
