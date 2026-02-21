use crate::noise::perlin::Perlin;
use crate::noise::perlin::constants::*;
use crate::noise::perlin::containers::*;
use std::simd::StdFloat;
use crate::simd::arch_simd::{ArchSimd, SimdInfo};

// === Helper Macros ===

macro_rules! compute_lerps_2d {(
        $gradients:expr,
        $interpolations:expr,
        $x_upper_increment:expr,
        $x_lower_increment:expr,
        $x_weighted_increment_vec:expr,
        $weight_vec:expr,
        $y_it:expr
    ) => {{
        // Load gradients into registers.
        let y_lerp = $interpolations.y.load_simd($y_it);
        let x_tl = $gradients.tl().x.load_simd($y_it);
        let x_tr = $gradients.tr().x.load_simd($y_it);
        let x_bl = $gradients.bl().x.load_simd($y_it);
        let x_br = $gradients.br().x.load_simd($y_it);
        let y_tl = $gradients.tl().y.load_simd($y_it);
        let y_tr = $gradients.tr().y.load_simd($y_it);
        let y_bl = $gradients.bl().y.load_simd($y_it);
        let y_br = $gradients.br().y.load_simd($y_it);

        // Compute base dot products.
        let prod_sum_tl = x_tl.mul_add($x_upper_increment, y_tl);
        let prod_sum_tr = x_tr.mul_add($x_upper_increment, y_tr);
        let prod_sum_bl = x_bl.mul_add($x_lower_increment, y_bl);
        let prod_sum_br = x_br.mul_add($x_lower_increment, y_br);

        // Base interpolation.
        let base_lerp_top = y_lerp.mul_add(prod_sum_tr - prod_sum_tl, prod_sum_tl) * $weight_vec;
        let base_lerp_bottom = y_lerp.mul_add(prod_sum_br - prod_sum_bl, prod_sum_bl) * $weight_vec;
        let base_lerp_dif = base_lerp_bottom - base_lerp_top;

        // Compute offset dot x offset dot products.
        let x_offset_tl = x_tl * $x_weighted_increment_vec;
        let x_offset_tr = x_tr * $x_weighted_increment_vec;
        let x_offset_bl = x_bl * $x_weighted_increment_vec;
        let x_offset_br = x_br * $x_weighted_increment_vec;

        // Offset interpolation.
        let x_offset_lerp_top = y_lerp.mul_add(x_offset_tr - x_offset_tl, x_offset_tl);
        let x_offset_lerp_bottom = y_lerp.mul_add(x_offset_br - x_offset_bl, x_offset_bl);
        let x_offset_lerp_dif = x_offset_lerp_bottom - x_offset_lerp_top;

        (base_lerp_top, base_lerp_dif, x_offset_lerp_top, x_offset_lerp_dif)
    }};
}

macro_rules! compute_results_2d {
        ($result:expr,
        $x_lerp:expr,
        $base_lerp_top:expr,
        $base_lerp_dif:expr, 
        $x_offset_lerp_top:expr,
        $x_offset_lerp_dif:expr,
        $index:expr,
        $initialize:expr
    ) => {{
        // Final interpolation.
        let output = $x_lerp.mul_add($base_lerp_dif, $base_lerp_top);

        // Store if $Initialize, Add to and store if not.
        let val = if $initialize { output } else { output + $result.load_simd($index) };
        $result.store_simd($index, val);

        // Accumulate interpolation variables.
        $base_lerp_dif += $x_offset_lerp_dif;
        $base_lerp_top += $x_offset_lerp_top;
    }};
}

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
        let x_weighted_increment_vec = ArchSimd::splat(x_increment) * weight_vec;
        let x_upper_increment = ArchSimd::splat(x_frac_start);
        let x_lower_increment = ArchSimd::splat(x_frac_start - 1.0);

        // Do 4 passes in 1 to fill 32 available registers.
        #[cfg(target_feature = "neon")]
        {
            for y_it in (0..ROW_SIZE).step_by(f32::LANES * 4) {
                let (mut base_lerp_top_1, mut base_lerp_dif_1, x_offset_lerp_top_1, x_offset_lerp_dif_1) =
                    compute_lerps_2d!(gradients, interpolations, x_upper_increment, x_lower_increment, x_weighted_increment_vec, weight_vec, y_it);
                let (mut base_lerp_top_2, mut base_lerp_dif_2, x_offset_lerp_top_2, x_offset_lerp_dif_2) =
                    compute_lerps_2d!(gradients, interpolations, x_upper_increment, x_lower_increment, x_weighted_increment_vec, weight_vec, y_it + f32::LANES);
                let (mut base_lerp_top_3, mut base_lerp_dif_3, x_offset_lerp_top_3, x_offset_lerp_dif_3) =
                    compute_lerps_2d!(gradients, interpolations, x_upper_increment, x_lower_increment, x_weighted_increment_vec, weight_vec, y_it + f32::LANES * 2);
                let (mut base_lerp_top_4, mut base_lerp_dif_4, x_offset_lerp_top_4, x_offset_lerp_dif_4) =
                    compute_lerps_2d!(gradients, interpolations, x_upper_increment, x_lower_increment, x_weighted_increment_vec, weight_vec, y_it + f32::LANES * 3);
    
                for x_it in x_start_index..x_start_index + x_chunk_size {
                    let x_lerp = ArchSimd::splat(unsafe { interpolations.x.get_unchecked(x_it) } );
                    compute_results_2d!(result, x_lerp, base_lerp_top_1, base_lerp_dif_1, x_offset_lerp_top_1, x_offset_lerp_dif_1, x_it * ROW_SIZE + y_it, INITIALIZE);
                    compute_results_2d!(result, x_lerp, base_lerp_top_2, base_lerp_dif_2, x_offset_lerp_top_2, x_offset_lerp_dif_2, x_it * ROW_SIZE + y_it + f32::LANES, INITIALIZE);
                    compute_results_2d!(result, x_lerp, base_lerp_top_3, base_lerp_dif_3, x_offset_lerp_top_3, x_offset_lerp_dif_3, x_it * ROW_SIZE + y_it + f32::LANES * 2, INITIALIZE);
                    compute_results_2d!(result, x_lerp, base_lerp_top_4, base_lerp_dif_4, x_offset_lerp_top_4, x_offset_lerp_dif_4, x_it * ROW_SIZE + y_it + f32::LANES * 3, INITIALIZE);
                }
            }
        }

        // Do 2 passes in 1 to fill 16 available registers.
        // AVX512f would finish in 2 passes, so it also takes this route. 
        #[cfg(not(target_feature = "neon"))]
        {
            for y_it in (0..ROW_SIZE).step_by(f32::LANES * 2) {
                let (mut base_lerp_top_1, mut base_lerp_dif_1, x_offset_lerp_top_1, x_offset_lerp_dif_1) =
                    compute_lerps_2d!(gradients, interpolations, x_upper_increment, x_lower_increment, x_weighted_increment_vec, weight_vec, y_it);
                let (mut base_lerp_top_2, mut base_lerp_dif_2, x_offset_lerp_top_2, x_offset_lerp_dif_2) =
                    compute_lerps_2d!(gradients, interpolations, x_upper_increment, x_lower_increment, x_weighted_increment_vec, weight_vec, y_it + f32::LANES);
    
                for x_it in x_start_index..x_start_index + x_chunk_size {
                    let x_lerp = ArchSimd::splat(unsafe { interpolations.x.get_unchecked(x_it) } );
                    compute_results_2d!(result, x_lerp, base_lerp_top_1, base_lerp_dif_1, x_offset_lerp_top_1, x_offset_lerp_dif_1, x_it * ROW_SIZE + y_it, INITIALIZE);
                    compute_results_2d!(result, x_lerp, base_lerp_top_2, base_lerp_dif_2, x_offset_lerp_top_2, x_offset_lerp_dif_2, x_it * ROW_SIZE + y_it + f32::LANES, INITIALIZE);
                }
            }
        }
    }
}
