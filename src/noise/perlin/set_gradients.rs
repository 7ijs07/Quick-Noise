use crate::noise::perlin::Perlin;
use crate::noise::perlin::constants::*;
use std::simd::num::SimdUint;
use crate::simd::arch_simd::{ArchSimd, SimdInfo};
use crate::simd::simd_array::SimdArray;

impl Perlin {
    // #[inline(never)]
    pub(super) fn set_gradients_2d (
        &mut self,
        left: &mut PerlinVecPair,
        right: &mut PerlinVecPair,
        x_start: i32,
        y_start: i32,
        y_next_index_offset: f32,
        y_scale: f32,
        y_num_loops: u32,
        y_distances: &PerlinVec,
    ) {
        let iota_vec = ArchSimd::from_array(std::array::from_fn(|i| i as i32));
        let mut y_vec = ArchSimd::splat(y_start) + iota_vec;
        let x_vec = ArchSimd::splat(x_start);

        // Temporary buffer to store indices for gradient values.
        let mut grad_array = SimdArray::<u32, ROW_SIZE>::new_uninit();

        // Peel first vectorized mix.
        let grad_vec: ArchSimd<u32> = self.random_gen.mix_i32_simd_pair(x_vec, y_vec).cast() & ArchSimd::splat(0xF as u32);
        grad_array.store_simd(0 as usize, grad_vec);
        
        // Main vectorized bit mixing loop.
        let lane_increment = ArchSimd::splat(f32::LANES as i32);
        for i in (f32::LANES..y_num_loops as usize).step_by(f32::LANES) {
            y_vec += lane_increment;
            let grad_vec: ArchSimd<u32> = self.random_gen.mix_i32_simd_pair(x_vec, y_vec) & ArchSimd::splat(0xF as u32);
            grad_array.store_simd(i as usize, grad_vec);
        }

        let mut arrays = [
            &mut left.x, &mut left.y,
            &mut right.x, &mut right.y,
        ];

        // Loop through the y chunks.
        let mut cur_index: u32 = 0;
        let mut y_it_scale: f32 = 0.0;
        for y_it in 0..y_num_loops {

            // Find range of gradients to set.
            let next_index_exact: f32 = y_it_scale + y_next_index_offset;
            let next_index: u32 = unsafe { next_index_exact.to_int_unchecked::<u32>().min(ROW_SIZE as u32) }; // Never negative or NaN.
            let set_amount: u32 = next_index - cur_index;

            // Set all gradients at once with masked stores.
            unsafe {
                let l = grad_array.get_unchecked(y_it as usize) as usize;
                let r = grad_array.get_unchecked(y_it as usize + 1) as usize;

                debug_assert!(l < 32);
                debug_assert!(r < 32);
                let values = [
                    GRADIENTS_2D.get_unchecked(l).x, GRADIENTS_2D.get_unchecked(l).y,
                    GRADIENTS_2D.get_unchecked(r).x, GRADIENTS_2D.get_unchecked(r).y,
                ];

                PerlinVec::multiset_many::<4>(&mut arrays, &values, cur_index as usize, set_amount as isize);
            }

            cur_index = next_index;
            y_it_scale += y_scale;
        }

        // Compute y dot products (Better to do here since these dot products get reused and operate per element).
        left.y *= *y_distances;
        right.y = right.y.mul_sub(*y_distances, right.y); // equivalent to -> right.y *= y_distances - 1.0
    }
}
