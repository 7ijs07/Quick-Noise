use std::{
    hint::black_box,
    num::{NonZero, NonZeroU32},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use quick_noise::perlin::Perlin;
use rand::{Rng, RngExt, SeedableRng};

use crate::{fast_noise_2_gen::PerlinFastNoise2, lcg::Lcg64};

mod fast_noise_2_gen;
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
    #[inline(always)]
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
    type Output2D;
    type Output3D;

    fn create_output_2d() -> Self::Output2D;

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
    let mut output = T::create_output_2d();
    let mut config_rng = Lcg64::from_seed(config_seed.to_ne_bytes());

    let mut samples = 0usize;
    let mut elapsed = Duration::ZERO;

    while elapsed < TEST_TIME {
        let config = NoiseConfig::random(&mut config_rng, single_octave);
        
        let start = Instant::now();
        for _ in 0..20 {
            samples += generator.generate_2d(&mut output, &config);
            black_box(&output);
        }
        elapsed += start.elapsed();
    }

    return (elapsed, samples);
}

fn test_2d_thread<T: NoiseGenerator>(config_seed: u64, noise_seed: u64, single_octave: bool) -> JoinHandle<(Duration, usize, f64)> {
    return thread::spawn(move || {
        let (duration, samples) = test_2d::<T>(config_seed, noise_seed, single_octave);
        let sps_billion = samples as f64 / duration.as_secs_f64() / 1e9;
        (duration, samples, sps_billion)
    });
}

fn print_results_2d((duration, samples, sps_billion): (Duration, usize, f64), single_octave: bool, name: &str) {
    let octave = if single_octave {
        "Single-octave"
    } else {
        "Multi-octave"
    };
    
    println!(
        "{} {} generated {} samples in {:.0?}, {:.3} billion samples per second",
        octave, name, samples, duration, sps_billion
    );
}

fn main() {
    let mut rng = rand::rng();
    let config_seed = rng.next_u64();
    let noise_seed = rng.next_u64();

    let handle_qn_single = test_2d_thread::<Perlin>(config_seed, noise_seed, true);
    let handle_qn_multi = test_2d_thread::<Perlin>(config_seed, noise_seed, false);
    let handle_fn2_single = test_2d_thread::<PerlinFastNoise2>(config_seed, noise_seed, true);
    let handle_fn2_multi = test_2d_thread::<PerlinFastNoise2>(config_seed, noise_seed, false);

    print_results_2d(handle_qn_single.join().unwrap(), true, "Quick Noise");
    print_results_2d(handle_qn_multi.join().unwrap(), false, "Quick Noise");
    print_results_2d(handle_fn2_single.join().unwrap(), true, "Fast Noise 2");
    print_results_2d(handle_fn2_multi.join().unwrap(), false, "Fast Noise 2");
}
