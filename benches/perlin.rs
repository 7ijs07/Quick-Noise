use criterion::{Criterion, criterion_group, criterion_main};
use quick_noise::perlin::{Perlin, PerlinMap, PerlinVol};
const SCALES: [f32; 11] = [64.0, 48.0, 32.0, 24.0, 16.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0];

fn perlin_2d_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("perlin_noise_2d");
    for scale in SCALES {
        group.bench_function(format!("scale: {scale}"), |b| {
            let mut perlin = Perlin::new(0);
            let mut array = PerlinMap::new_uninit();
            let mut i = 0;
            b.iter(|| {
                i = i + 1 & 0xFFFFFF;
                perlin.noise_2d(&mut array, (i, i).into(), 1, scale, 1.0, 2.0, 0.5, 1, 0.0)
            });
        });
    }
}

fn perlin_3d_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("perlin_noise_3d");
    for scale in SCALES {
        group.bench_function(format!("scale: {scale}"), |b| {
            let mut perlin = Perlin::new(0);
            let mut array = PerlinVol::new_uninit();
            let mut i = 0;
            b.iter(|| {
                i = i + 1 & 0xFFFFFF;
                perlin.noise_3d(
                    &mut array,
                    (i, i, i).into(),
                    1,
                    scale,
                    1.0,
                    2.0,
                    0.5,
                    1,
                    0.0,
                )
            });
        });
    }
}

criterion_group!(benches, perlin_2d_benchmark, perlin_2d_benchmark);
criterion_main!(benches);
