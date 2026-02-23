use std::{hint::black_box, num::{NonZero, NonZeroU32}, process::Output, time::{Duration, Instant}};

use quick_noise::perlin::Perlin;
use rand::{RngExt, SeedableRng, rngs::StdRng};

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
    pub fn random(rng: &mut Lcg64) -> Self {
        let pos = (
            rng.random_range(-10_000..10_000),
            rng.random_range(-10_000..10_000),
        );

        let scale = rng.random_range(2.0..100.0);

        let octaves = NonZeroU32::new(rng.random_range(1..=8)).unwrap();

        let lacunarity = rng.random_range(1f32..4f32);

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

fn test_single_2d<T: NoiseGenerator>(config_seed: u64, noise_seed: u64) -> (Duration, usize) {
    let mut generator = T::new_with_seed(noise_seed);
    let mut output = T::Output2D::default();
    let mut config_rng = Lcg64::from_seed(config_seed.to_ne_bytes());

    let mut samples = 0usize;
    let start = Instant::now();

    while start.elapsed() < TEST_TIME {
        let config = NoiseConfig::random(&mut config_rng);
        println!("{config:#?}");

        samples += generator.generate_2d(&mut output, &config);

        black_box(&output);
    }

    return (start.elapsed(), samples);
}

fn main() {
    let (duration, samples) = test_single_2d::<Perlin>(0, 0);

    println!("Generated {} samples in {:?}", samples, duration);
}
