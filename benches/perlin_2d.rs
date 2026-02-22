use criterion::{Criterion, criterion_group, criterion_main};
use quick_noise::perlin::Perlin;

fn perlin_2d_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("perlin_noise_2d");
    for scale in [64.0, 48.0, 32.0, 24.0] {
        group.bench_function(format!("scale: {scale}"), |b| {
            let mut perlin = Perlin::new(0);
            let mut i = 0;
            b.iter(|| {
                i = i + 1 & 0xFFFFFF;
                perlin.noise_2d((i, i).into(), 1, scale, 1.0, 2.0, 0.5, 1, 0.0)
            });
        });
    }
}

criterion_group!(benches, perlin_2d_benchmark);
criterion_main!(benches);
