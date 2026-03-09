use image::imageops::FilterType::Triangle;

use crate::simplex::Simplex;
use crate::simd::simd_array::SimdArray;
use crate::simd::arch_simd::{ArchSimd};
use crate::simd::simd_traits::*;
use std::f32::consts::SQRT_2;

const SQRT_3: f32 = 1.7320508075688772935274;
const SKEW: f32 = (SQRT_3 - 1.0) / 2.0;
const UNSKEW: f32 = (3.0 - SQRT_3) / 6.0;

const SCALED_SQRT: f32 = (SQRT_2 / 2.0) * 70.0;
pub const X_GRADIENTS_2D: [f32; 8] = [
    70.0,
    SCALED_SQRT,
    0.0,
   -SCALED_SQRT,
   -70.0,
   -SCALED_SQRT,
    0.0,
    SCALED_SQRT,
];

pub const Y_GRADIENTS_2D: [f32; 8] = [
    0.0,
    SCALED_SQRT,
    70.0,
    SCALED_SQRT,
    0.0,
   -SCALED_SQRT,
   -70.0,
   -SCALED_SQRT,
];


// pub const X_GRADIENTS_2D: [f32; 8] = [
//     1.4142135623730951,
//     1.0000000000000000,
//     0.0000000000000000,
//    -1.0000000000000000,
//    -1.4142135623730951,
//    -1.0000000000000000,
//     0.0000000000000000,
//     1.0000000000000000,
// ];

// pub const Y_GRADIENTS_2D: [f32; 8] = [
//     0.0000000000000000,
//     1.0000000000000000,
//     1.4142135623730951,
//     1.0000000000000000,
//     0.0000000000000000,
//    -1.0000000000000000,
//    -1.4142135623730951,
//    -1.0000000000000000,
// ];

impl Simplex {
    pub fn batched_2d(
        &mut self,
        output: &mut SimdArray<f32, 1024>,
        x_array: &SimdArray<f32, 1024>,
        y_array: &SimdArray<f32, 1024>,
        freq: f32,
        weight_coef: f32,
        channel_seed: u64,
        octave_offset: f32,
    ) {
        // Constants.
        let skew: ArchSimd<f32> = ArchSimd::splat(SKEW);
        let unskew: ArchSimd<f32> = ArchSimd::splat(UNSKEW);
        let subbed_unskew: ArchSimd<f32> = ArchSimd::splat(UNSKEW - 1.0);
        let hi_skew_offset: ArchSimd<f32> = ArchSimd::splat(2.0 * UNSKEW - 1.0);
        let half: ArchSimd<f32> = ArchSimd::splat(0.5);
        let zero: ArchSimd<f32> = ArchSimd::splat(0.0);

        // Hash constants.
        const BYTE_SHUFFLE: [u8; 64] = [
            3,0,2,1, 7,4,6,5, 11,8,10,9, 15,12,14,13,
            3,0,2,1, 7,4,6,5, 11,8,10,9, 15,12,14,13,
            3,0,2,1, 7,4,6,5, 11,8,10,9, 15,12,14,13,
            3,0,2,1, 7,4,6,5, 11,8,10,9, 15,12,14,13,
        ];
        
        let shuffle_indices = ArchSimd::<u8>::load(&BYTE_SHUFFLE[..]);
        let channel_seed = ArchSimd::splat(self.random_gen.channel_seed as u32);
        let prime = ArchSimd::splat(0x85ebca6b_u32 as u32);
    
        // Frequency constant.
        let freq_vec = ArchSimd::<f32>::splat(freq);

        for i in (0..1024).step_by(ArchSimd::<f32>::LANES) {

            // Load and scale: 4
            let x_vec = x_array.load_simd(i);
            let y_vec = y_array.load_simd(i);

            let x_scaled = x_vec * freq_vec;
            let y_scaled = y_vec * freq_vec;

            // Gridpoints and distances: 19
            let s = (x_scaled + y_scaled) * skew;
            let x_grid = (x_scaled + s).floor();
            let y_grid = (y_scaled + s).floor();

            let unskew_sub = (x_grid + y_grid) * unskew;
            let x_dist_lo = x_scaled - x_grid + unskew_sub;
            let y_dist_lo = y_scaled - y_grid + unskew_sub;
            let triangle_mask = x_dist_lo.simd_gt(y_dist_lo);

            let x_dist_mi_offset = unskew.blend_32(subbed_unskew, triangle_mask);
            let y_dist_mi_offset = subbed_unskew.blend_32(unskew, triangle_mask);
            let x_dist_mi = x_dist_lo + x_dist_mi_offset;
            let y_dist_mi = y_dist_lo + y_dist_mi_offset;
            let x_dist_hi = x_dist_lo + hi_skew_offset;
            let y_dist_hi = y_dist_lo + hi_skew_offset;

            // Hash: 19
            let x1: ArchSimd<u32> = x_grid.cast_int_trunc().raw_cast() * channel_seed;
            let y1: ArchSimd<u32> = y_grid.cast_int_trunc().raw_cast() * channel_seed;
            let x2 = x1 + channel_seed;
            let y2 = y1 + channel_seed;

            let x1_shuf = x1.permute_8(shuffle_indices) ^ prime;
            let y1_shuf = y1.permute_8(shuffle_indices) ^ prime;
            let x2_shuf = x2.permute_8(shuffle_indices) ^ prime;
            let y2_shuf = y2.permute_8(shuffle_indices) ^ prime;

            let mix_lo = x1_shuf * y1_shuf;
            let mix_mi_1 = x1_shuf * y2_shuf;
            let mix_mi_2 = x2_shuf * y1_shuf;
            let mix_hi = x2_shuf * y2_shuf;

            let mix_mi = mix_mi_1.blend_32(mix_mi_2, triangle_mask.raw_cast());

            // Gradient lookup: 9
            let indices_lo = mix_lo >> 29;
            let indices_mi = mix_mi >> 29;
            let indices_hi = mix_hi >> 29;

            let x_grads_lo = indices_lo.gather(&X_GRADIENTS_2D);
            let y_grads_lo = indices_lo.gather(&Y_GRADIENTS_2D);
            let x_grads_mi = indices_mi.gather(&X_GRADIENTS_2D);
            let y_grads_mi = indices_mi.gather(&Y_GRADIENTS_2D);
            let x_grads_hi = indices_hi.gather(&X_GRADIENTS_2D);
            let y_grads_hi = indices_hi.gather(&Y_GRADIENTS_2D);
            
            // Sum of products: 27
            let t_lo = (half - x_dist_lo.mul_add(x_dist_lo, y_dist_lo * y_dist_lo)).max(zero);
            let t_mi = (half - x_dist_mi.mul_add(x_dist_mi, y_dist_mi * y_dist_mi)).max(zero);
            let t_hi = (half - x_dist_hi.mul_add(x_dist_hi, y_dist_hi * y_dist_hi)).max(zero);

            let t2_lo = t_lo * t_lo;
            let t2_mi = t_mi * t_mi;
            let t2_hi = t_hi * t_hi;

            let t4_lo = t2_lo * t2_lo;
            let t4_mi = t2_mi * t2_mi;
            let t4_hi = t2_hi * t2_hi;

            let dot_lo = x_grads_lo.mul_add(x_dist_lo, y_grads_lo * y_dist_lo);
            let dot_mi = x_grads_mi.mul_add(x_dist_mi, y_grads_mi * y_dist_mi);
            let dot_hi = x_grads_hi.mul_add(x_dist_hi, y_grads_hi * y_dist_hi);

            let result = t4_lo.mul_add(dot_lo, t4_mi.mul_add(dot_mi, t4_hi * dot_hi));

            // Store: 1
            output.store_simd(i, result);
        }
    }
}
