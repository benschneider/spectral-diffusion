use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use num_complex::Complex32;
use rand::Rng;
use rifft_core::RifftHandle;

fn bench_fused(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let size = 512usize;
    let plane = size * size;
    let template: Vec<_> = (0..plane)
        .map(|_| Complex32::new(rng.gen::<f32>(), rng.gen::<f32>()))
        .collect();
    let filter: Vec<_> = (0..plane)
        .map(|_| Complex32::new(rng.gen::<f32>(), rng.gen::<f32>()))
        .collect();
    let mut handle = RifftHandle::new();
    c.bench_function("fused_fft_filter_ifft_512", |b| {
        b.iter_batched(
            || template.clone(),
            |mut data| {
                handle
                    .fft_filter_ifft(&mut data, &filter, size, size)
                    .unwrap();
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_fused);
criterion_main!(benches);
