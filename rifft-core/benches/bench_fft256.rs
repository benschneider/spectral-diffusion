use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use num_complex::Complex32;
use rand::Rng;
use rifft_core::RifftHandle;

fn bench_fft256(c: &mut Criterion) {
    run_fft_bench(c, 256, "fft2d_256");
}

fn run_fft_bench(c: &mut Criterion, size: usize, label: &str) {
    let mut rng = rand::thread_rng();
    let template: Vec<_> = (0..size * size)
        .map(|_| Complex32::new(rng.gen::<f32>(), rng.gen::<f32>()))
        .collect();
    let mut handle = RifftHandle::new();
    c.bench_function(label, |b| {
        b.iter_batched(
            || template.clone(),
            |mut data| {
                handle.fft2d_forward(&mut data, size, size).unwrap();
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_fft256);
criterion_main!(benches);
