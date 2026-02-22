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
        let grad_vec: ArchSimd<u32> = self.random_gen.mix_i32_simd_pair(x_vec, y_vec) & ArchSimd::splat(0xF as u32);
        grad_array.store_simd(0 as usize, grad_vec);
        
        // Main vectorized bit mixing loop.
        let lane_increment = ArchSimd::splat(f32::LANES as i32);
        for i in (f32::LANES..y_num_loops as usize + 1).step_by(f32::LANES) {
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
        let mut y_next_index_exact: f32 = y_next_index_offset;
        for y_it in 0..y_num_loops {

            // Find range of gradients to set.
            debug_assert!(y_next_index_exact >= 0.0 && y_next_index_exact.is_finite());
            let y_next_index: u32 = unsafe { y_next_index_exact.to_int_unchecked::<u32>().min(ROW_SIZE as u32) }; // Never negative or NaN.
            let set_amount: u32 = y_next_index - cur_index;

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

            if y_next_index == 32 { break; }

            cur_index = y_next_index;
            y_next_index_exact += y_scale;
        }

        // Compute y dot products (Better to do here since these dot products get reused and operate per element).
        left.y *= *y_distances;
        right.y = right.y.mul_sub(*y_distances, right.y); // equivalent to -> right.y *= y_distances - 1.0
    }

    #[inline(never)]
    pub(super) fn set_gradients_3d (
        &mut self,
        lf: &mut PerlinVecTriple,
        rf: &mut PerlinVecTriple,
        lb: &mut PerlinVecTriple,
        rb: &mut PerlinVecTriple,
        x_start: i32,
        y_start: i32,
        z_start: i32,
        z_next_index_offset: f32,
        z_scale: f32,
        z_num_loops: u32,
        z_distances: &PerlinVec,
    ) {
        let iota_vec = ArchSimd::from_array(std::array::from_fn(|i| i as i32));

        let x_front_vec = ArchSimd::splat(x_start);
        let x_back_vec = ArchSimd::splat(x_start + 1);
        let y_vec = ArchSimd::splat(y_start);
        let mut z_vec = ArchSimd::splat(z_start) + iota_vec;

        // Temporary buffer to store indices for gradient values.
        let mut front_grad_array = SimdArray::<u32, ROW_SIZE>::new_uninit();
        let mut back_grad_array = SimdArray::<u32, ROW_SIZE>::new_uninit();

        // Peel first vectorized mix.
        let front_grad_vec: ArchSimd<u32> = self.random_gen.mix_i32_simd_triple(x_front_vec, y_vec, z_vec) % ArchSimd::splat(12);
        let back_grad_vec: ArchSimd<u32> = self.random_gen.mix_i32_simd_triple(x_back_vec , y_vec, z_vec) % ArchSimd::splat(12);
        front_grad_array.store_simd(0 as usize, front_grad_vec);
        back_grad_array.store_simd(0 as usize, back_grad_vec);
        
        // Main vectorized bit mixing loop.
        let lane_increment = ArchSimd::splat(f32::LANES as i32);
        for i in (f32::LANES..z_num_loops as usize + 1).step_by(f32::LANES) {
            z_vec += lane_increment;
            let front_grad_vec: ArchSimd<u32> = self.random_gen.mix_i32_simd_triple(x_front_vec, y_vec, z_vec) % ArchSimd::splat(12);
            let back_grad_vec: ArchSimd<u32> = self.random_gen.mix_i32_simd_triple(x_back_vec , y_vec, z_vec) % ArchSimd::splat(12);
            front_grad_array.store_simd(i as usize, front_grad_vec);
            back_grad_array.store_simd(i as usize, back_grad_vec);
        }

        let mut arrays = [
            &mut lf.x, &mut lf.y, &mut lf.z,
            &mut rf.x, &mut rf.y, &mut rf.z,
            &mut lb.x, &mut lb.y, &mut lb.z,
            &mut rb.x, &mut rb.y, &mut rb.z,
        ];

        // Loop through the y chunks.
        let mut z_cur_index: u32 = 0;
        let mut z_next_index_exact: f32 = z_next_index_offset;
        for z_it in 0..z_num_loops {

            // Find range of gradients to set.
            debug_assert!(z_next_index_exact >= 0.0 && z_next_index_exact.is_finite());
            let z_next_index: u32 = unsafe { z_next_index_exact.to_int_unchecked::<u32>().min(ROW_SIZE as u32) }; // Never negative or NaN.
            let set_amount: u32 = z_next_index - z_cur_index;

            // Set all gradients at once with masked stores.
            unsafe {
                let lf_grad = front_grad_array.get_unchecked(z_it as usize) as usize;
                let rf_grad = front_grad_array.get_unchecked(z_it as usize + 1) as usize;
                let lb_grad = back_grad_array.get_unchecked(z_it as usize) as usize;
                let rb_grad = back_grad_array.get_unchecked(z_it as usize + 1) as usize;

                debug_assert!(lf_grad < 32);
                debug_assert!(rf_grad < 32);
                debug_assert!(lb_grad < 32);
                debug_assert!(rb_grad < 32);
                let values = [
                    GRADIENTS_3D.get_unchecked(lf_grad).x, GRADIENTS_3D.get_unchecked(lf_grad).y, GRADIENTS_3D.get_unchecked(lf_grad).z,
                    GRADIENTS_3D.get_unchecked(rf_grad).x, GRADIENTS_3D.get_unchecked(rf_grad).y, GRADIENTS_3D.get_unchecked(rf_grad).z,
                    GRADIENTS_3D.get_unchecked(lb_grad).x, GRADIENTS_3D.get_unchecked(lb_grad).y, GRADIENTS_3D.get_unchecked(lb_grad).z,
                    GRADIENTS_3D.get_unchecked(rb_grad).x, GRADIENTS_3D.get_unchecked(rb_grad).y, GRADIENTS_3D.get_unchecked(rb_grad).z,
                ];

                PerlinVec::multiset_many::<12>(&mut arrays, &values, z_cur_index as usize, set_amount as isize);
            }

            if z_next_index == 32 { break; }

            z_cur_index = z_next_index;
            z_next_index_exact += z_scale;
        }

        // Compute z dot products (Better to do here since these dot products get reused and operate per element).
        lf.z *= *z_distances;
        rf.z = rf.z.mul_sub(*z_distances, rf.z);
        lb.z *= *z_distances;
        rb.z = rb.z.mul_sub(*z_distances, rb.z);
    }
}
