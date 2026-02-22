use crate::noise::perlin::Perlin;
use crate::noise::perlin::constants::*;
use crate::noise::perlin::containers::*;
use crate::simd::arch_simd::{ArchSimd, NUM_SIMD_REG, SimdInfo};
use std::simd::StdFloat;

impl Perlin {
    // #[inline(never)]
    pub(super) fn compute_noise_from_vecs_2d<const INITIALIZE: bool>(
        gradients: &PerlinContainer2D,
        x_frac_start: f32,
        x_increment: f32,
        x_chunk_size: usize,
        interpolations: &PerlinVecPair,
        x_start_index: usize,
        weight: f32,
        result: &mut PerlinMap,
    ) {
        let weight_vec = ArchSimd::splat(weight);
        let x_weighted_increment_vec = ArchSimd::splat(x_increment * weight);
        let x_upper_increment = ArchSimd::splat(x_frac_start);
        let x_lower_increment = ArchSimd::splat(x_frac_start - 1.0);

        const NUM_BLOCKS_POSSIBLE: usize = NUM_SIMD_REG / 8;
        const MAX_NUM_BLOCKS: usize = ROW_SIZE / f32::LANES;
        const NUM_BLOCKS: usize = NUM_BLOCKS_POSSIBLE.min(MAX_NUM_BLOCKS);

        for y_it in (0..ROW_SIZE).step_by(f32::LANES * NUM_BLOCKS) {
            // Set up registers per block. Initialization is just to keep Rust happy. Compiler will optimize away.
            let mut base_lerps_top: [ArchSimd<f32>; NUM_BLOCKS] = Default::default();
            let mut base_lerps_dif: [ArchSimd<f32>; NUM_BLOCKS] = Default::default();
            let mut x_offset_lerps_top: [ArchSimd<f32>; NUM_BLOCKS] = Default::default();
            let mut x_offset_lerps_dif: [ArchSimd<f32>; NUM_BLOCKS] = Default::default();

            // These blocked loops will get entirely unrolled by the compiler.
            for block in 0..NUM_BLOCKS {
                // Load gradients into registers.
                let y_lerp = interpolations.y.load_simd(y_it + f32::LANES * block);
                let x_tl: ArchSimd<f32> = gradients.tl().x.load_simd(y_it + f32::LANES * block);
                let x_tr: ArchSimd<f32> = gradients.tr().x.load_simd(y_it + f32::LANES * block);
                let x_bl: ArchSimd<f32> = gradients.bl().x.load_simd(y_it + f32::LANES * block);
                let x_br: ArchSimd<f32> = gradients.br().x.load_simd(y_it + f32::LANES * block);
                let y_tl: ArchSimd<f32> = gradients.tl().y.load_simd(y_it + f32::LANES * block);
                let y_tr: ArchSimd<f32> = gradients.tr().y.load_simd(y_it + f32::LANES * block);
                let y_bl: ArchSimd<f32> = gradients.bl().y.load_simd(y_it + f32::LANES * block);
                let y_br: ArchSimd<f32> = gradients.br().y.load_simd(y_it + f32::LANES * block);

                // Compute base dot products.
                let prod_sum_tl = x_tl.mul_add(x_upper_increment, y_tl);
                let prod_sum_tr = x_tr.mul_add(x_upper_increment, y_tr);
                let prod_sum_bl = x_bl.mul_add(x_lower_increment, y_bl);
                let prod_sum_br = x_br.mul_add(x_lower_increment, y_br);

                // Base interpolation.
                base_lerps_top[block] =
                    y_lerp.mul_add(prod_sum_tr - prod_sum_tl, prod_sum_tl) * weight_vec;
                let base_lerp_bottom =
                    y_lerp.mul_add(prod_sum_br - prod_sum_bl, prod_sum_bl) * weight_vec;
                base_lerps_dif[block] = base_lerp_bottom - base_lerps_top[block];

                // Offset interpolation.
                x_offset_lerps_top[block] =
                    y_lerp.mul_add(x_tr - x_tl, x_tl) * x_weighted_increment_vec;
                let x_offset_lerp_bottom =
                    y_lerp.mul_add(x_br - x_bl, x_bl) * x_weighted_increment_vec;
                x_offset_lerps_dif[block] = x_offset_lerp_bottom - x_offset_lerps_top[block];
            }

            for x_it in x_start_index..x_start_index + x_chunk_size {
                let x_lerp = ArchSimd::splat(unsafe { interpolations.x.get_unchecked(x_it) });

                for block in 0..NUM_BLOCKS {
                    let index: usize = x_it * ROW_SIZE + y_it + block * f32::LANES;

                    // Final interpolation.
                    let output = x_lerp.mul_add(base_lerps_dif[block], base_lerps_top[block]);

                    // Store if Initialize, Add to and store if not.
                    let val = if INITIALIZE {
                        output
                    } else {
                        output + result.load_simd(index)
                    };
                    result.store_simd(index, val);

                    // Accumulate interpolation variables.
                    base_lerps_dif[block] += x_offset_lerps_dif[block];
                    base_lerps_top[block] += x_offset_lerps_top[block];
                }
            }
        }
    }
}
