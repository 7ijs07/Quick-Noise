use crate::noise::perlin::Perlin;
use std::time::Instant;
use std::hint::black_box;

pub fn profile_perlin_2d_call(octaves: u32, scale: f32, lacunarity: f32, persistence: f32) {
    const NUM_LOOPS: usize = 10000000;
    const SAMPLE_SIZE: usize = 1024;

    let mut perlin = Perlin::new(0);

    let start = Instant::now();
    for i in 0..NUM_LOOPS {
        black_box(perlin.noise_2d((i as i32, i as i32).into(), octaves, scale, 1.0, lacunarity, persistence, 1, 0.0)[i & 0xFF]);
    }
    let elapsed = start.elapsed();
    let ms_elapsed = elapsed.as_millis();

    let total = NUM_LOOPS * SAMPLE_SIZE;
    let elapsed_per_loop = elapsed.as_nanos() as u64 / NUM_LOOPS as u64;
    let samples_per_second = (total as f64 / elapsed.as_secs_f64()) as u64;

    println!(
        "-+- Completed 2D Perlin Noise Profile -+-\n\
        -> {total} samples in {ms_elapsed} ms!\n\
        -> {elapsed_per_loop} ns per {SAMPLE_SIZE} samples!\n\
        -> {samples_per_second} samples per second!"
    );
}

pub fn bench_perlin_2d() {
    println!("-+-+- Perlin 2D Noise Bench -+-+-");

    profile_perlin_2d_call_internal(1, 64.0, 10000000);
    profile_perlin_2d_call_internal(1, 48.0, 10000000);
    profile_perlin_2d_call_internal(1, 32.0, 10000000);
    profile_perlin_2d_call_internal(1, 24.0, 10000000);
    profile_perlin_2d_call_internal(1, 16.0, 10000000);
    profile_perlin_2d_call_internal(1, 12.0, 7500000);
    profile_perlin_2d_call_internal(1, 8.0, 5000000);
    profile_perlin_2d_call_internal(1, 6.0, 5000000);
    profile_perlin_2d_call_internal(1, 4.0, 2500000);
    profile_perlin_2d_call_internal(1, 3.0, 2500000);
    profile_perlin_2d_call_internal(1, 2.0, 1000000);

    println!("-+-+- Completed Perlin 2D Bench -+-+-");
}

fn profile_perlin_2d_call_internal(octaves: u32, scale: f32, num_loops: usize) {
    const SAMPLE_SIZE: usize = 1024;
    
    let mut perlin = Perlin::new(0);

    let start = Instant::now();
    for i in 0..num_loops {
        black_box(perlin.noise_2d((i as i32, i as i32).into(), octaves, scale, 1.0, 2.0, 0.5, 1, 0.0)[i & 0xFF]);
    }
    let elapsed = start.elapsed();

    let elapsed_per_loop = elapsed.as_nanos() as u64 / num_loops as u64;
    println!("Scale {scale} -> {elapsed_per_loop} ns per {SAMPLE_SIZE} samples!");
}