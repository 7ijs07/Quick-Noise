use std::{
    hint::black_box,
    num::{NonZero, NonZeroU32},
    thread,
    time::{Duration, Instant},
};

use quick_noise::perlin::Perlin;
use rand::{RngExt, SeedableRng};

use crate::lcg::Lcg64;

mod lcg;
mod quick_noise_gen;

#[derive(Debug)]
struct NoiseConfig {
    /// Each index increses by 32 samples, so adjacent positions generate adjacent batches
    pub pos: (i32, i32),
    pub scale: f32,
    pub octaves: NonZero<u32>,
    pub lacunarity: f32,
}

impl NoiseConfig {
    pub fn random(rng: &mut Lcg64, single_octave: bool) -> Self {
        let pos = (
            rng.random_range(-10_000..10_000),
            rng.random_range(-10_000..10_000),
        );

        let scale = rng.random_range(4.0..32.0);

        let octaves = if single_octave {
            unsafe { NonZeroU32::new_unchecked(1) }
        } else {
            unsafe { NonZeroU32::new_unchecked(rng.random_range(1..=8)) }
        };

        let lacunarity = rng.random_range(0f32..1f32);

        return Self {
            pos,
            scale,
            octaves,
            lacunarity,
        };
    }
}

trait NoiseGenerator {
    type Output2D: Default;
    type Output3D: Default;

    fn new_with_seed(seed: u64) -> Self;

    fn generate_2d(&mut self, output: &mut Self::Output2D, config: &NoiseConfig) -> usize;
}

const TEST_TIME: Duration = Duration::from_secs(10);

fn test_2d<T: NoiseGenerator>(
    config_seed: u64,
    noise_seed: u64,
    single_octave: bool,
) -> (Duration, usize) {
    let mut generator = T::new_with_seed(noise_seed);
    let mut output = T::Output2D::default();
    let mut config_rng = Lcg64::from_seed(config_seed.to_ne_bytes());

    let mut samples = 0usize;
    let start = Instant::now();

    while start.elapsed() < TEST_TIME {
        let config = NoiseConfig::random(&mut config_rng, single_octave);

        samples += generator.generate_2d(&mut output, &config);

        black_box(&output);
    }

    return (start.elapsed(), samples);
}

fn main() {
    let multi_handle = thread::spawn(|| {
        let (duration, samples) = test_2d::<Perlin>(0, 0, false);
        let sps_billion = (samples as f64 / duration.as_secs_f64() / 1e9);
        (duration, samples, sps_billion)
    });

    let single_handle = thread::spawn(|| {
        let (duration, samples) = test_2d::<Perlin>(0, 0, true);
        let sps_billion = (samples as f64 / duration.as_secs_f64() / 1e9);
        (duration, samples, sps_billion)
    });

    let (multi_duration, multi_samples, multi_sps) = multi_handle.join().unwrap();
    let (single_duration, single_samples, single_sps) = single_handle.join().unwrap();

    println!(
        "Multi-octave generated {} samples in {:.0?}, {:.3} billion samples per second",
        multi_samples, multi_duration, multi_sps
    );
    println!(
        "Single-octave generated {} samples in {:.0?}, {:.3} billion samples per second",
        single_samples, single_duration, single_sps
    );
}
