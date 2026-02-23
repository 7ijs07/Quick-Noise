use fastnoise2::Node;

use crate::NoiseGenerator;

pub struct PerlinFastNoise2 {
    seed: u64,
    node: Node,
}

impl NoiseGenerator for PerlinFastNoise2 {
    type Output2D = [f32; 1024];

    type Output3D = [f32; 32768];

    fn create_output_2d() -> Self::Output2D {
        return [0f32; 1024];
    }

    fn new_with_seed(seed: u64) -> Self {
        return PerlinFastNoise2 {
            seed,
            node: Node::from_name("Perlin").unwrap(),
        };
    }

    fn generate_2d(&mut self, output: &mut Self::Output2D, config: &crate::NoiseConfig) -> usize {
        for i in 0..config.octaves.get() {
            let scale = config.scale / (i as f32 * config.lacunarity);

            unsafe {
                self.node.gen_uniform_grid_2d_unchecked(
                    output,
                    (config.pos.0 * 32) as f32,
                    (config.pos.0 * 32) as f32,
                    32,
                    32,
                    1f32 / scale,
                    1f32 / scale,
                    self.seed as i32,
                );
            }
        }

        return 1024;
    }
}
