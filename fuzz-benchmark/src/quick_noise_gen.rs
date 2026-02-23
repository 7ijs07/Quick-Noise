use std::mem::MaybeUninit;

use quick_noise::{
    math::vec::Vec2,
    perlin::{Perlin, PerlinMap, PerlinVol},
};

use crate::NoiseGenerator;

impl NoiseGenerator for Perlin {
    type Output2D = PerlinMap;
    type Output3D = PerlinVol;

    #[inline(always)]
    fn create_output_2d() -> Self::Output2D {
        return unsafe { MaybeUninit::uninit().assume_init() };
    }

    #[inline(always)]
    fn new_with_seed(seed: u64) -> Self {
        return Self::new(seed as i64);
    }

    #[inline(always)]
    fn generate_2d(&mut self, output: &mut Self::Output2D, config: &crate::NoiseConfig) -> usize {
        self.uniform_grid_2d(
            output,
            Vec2 {
                x: config.pos.0,
                y: config.pos.1,
            },
            config.octaves.get(),
            config.scale,
            1f32,
            config.lacunarity,
            1f32,
            0,
            0f32,
        );
        return 1024;
    }
}
