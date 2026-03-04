use crate::math::random::Random;
use crate::math::vec::{Vec2, Vec3};
use crate::noise::perlin::constants::*;
use crate::noise::perlin::containers::*;
use crate::perlin::Perlin;
use crate::simd::simd_array::SimdArray;
use crate::simd::arch_simd::{ArchSimd, ArchMask};
use std::simd::StdFloat;
use crate::simd::simd_traits::*;
use std::simd::num::SimdFloat;
use crate::simd::simd_vec::core::SimdVec;
use crate::simd::architectures::families::Avx2Family;
use std::simd::num::SimdInt;
// use crate::simd::arch_simd::;

// std::simd implementation.
// impl Perlin {
//     pub fn batched_2d(
//         &mut self,
//         output: &mut SimdArray<f32, 1024>,
//         x_array: &SimdArray<f32, 1024>,
//         y_array: &SimdArray<f32, 1024>,
//         octave: &Octave2D,
//         weight_coef: f32,
//         channel_seed: u64,
//         octave_offset: f32,
//     ) where ArchSimd<f32>: StdFloat
//     {
//         let freq = ArchSimd::<f32>::splat(octave.scale.x);

//         let int_one_vec = ArchSimd::<i32>::splat(1);
//         let float_one_vec = ArchSimd::<f32>::splat(1.0);

//         for i in (0..1024).step_by(f32::LANES) {
//             let x_vec = x_array.load_simd(i);
//             let y_vec = y_array.load_simd(i);

//             let x_scaled = x_vec * freq;
//             let y_scaled = y_vec * freq;

//             let x_scaled_floored = x_scaled.floor();
//             let y_scaled_floored = y_scaled.floor();

//             let x_grid_lo: ArchSimd<i32> = unsafe { x_scaled_floored.floor().to_int_unchecked() };
//             let y_grid_lo: ArchSimd<i32> = unsafe { y_scaled_floored.floor().to_int_unchecked() };

//             let x_dist_lo = x_scaled - x_scaled_floored;
//             let y_dist_lo = y_scaled - y_scaled_floored;
//             let x_dist_hi = x_dist_lo - float_one_vec;
//             let y_dist_hi = y_dist_lo - float_one_vec;

//             let x_lerp = x_dist_lo.quintic_lerp();
//             let y_lerp = y_dist_lo.quintic_lerp();

//             let (mix_tl, mix_tr, mix_bl, mix_br) =
//                 self.random_gen.mix_u32_four_group(x_grid_lo.cast(), y_grid_lo.cast());

//             let indices_tl = mix_tl >> 29;
//             let indices_tr = mix_tr >> 29;
//             let indices_bl = mix_bl >> 29;
//             let indices_br = mix_br >> 29;

//             let x_grads_tl = ArchSimd::dyn_swizzle_u32(&X_GRADIENTS_2D, indices_tl);
//             let y_grads_tl = ArchSimd::dyn_swizzle_u32(&Y_GRADIENTS_2D, indices_tl);
//             let x_grads_tr = ArchSimd::dyn_swizzle_u32(&X_GRADIENTS_2D, indices_tr);
//             let y_grads_tr = ArchSimd::dyn_swizzle_u32(&Y_GRADIENTS_2D, indices_tr);
//             let x_grads_bl = ArchSimd::dyn_swizzle_u32(&X_GRADIENTS_2D, indices_bl);
//             let y_grads_bl = ArchSimd::dyn_swizzle_u32(&Y_GRADIENTS_2D, indices_bl);
//             let x_grads_br = ArchSimd::dyn_swizzle_u32(&X_GRADIENTS_2D, indices_br);
//             let y_grads_br = ArchSimd::dyn_swizzle_u32(&Y_GRADIENTS_2D, indices_br);

//             let prod_tl = x_grads_tl.mul_add(x_dist_lo, y_grads_tl * y_dist_lo);
//             let prod_tr = x_grads_tr.mul_add(x_dist_lo, y_grads_tr * y_dist_hi);
//             let top_lerp = y_lerp.mul_add(prod_tr - prod_tl, prod_tl);

//             let prod_bl = x_grads_bl.mul_add(x_dist_hi, y_grads_bl * y_dist_lo);
//             let prod_br = x_grads_br.mul_add(x_dist_hi, y_grads_br * y_dist_hi);
//             let bottom_lerp = y_lerp.mul_add(prod_br - prod_bl, prod_bl);

//             let result = x_lerp.mul_add(bottom_lerp - top_lerp, top_lerp);
            
//             output.store_simd(i, result);
//         }
//     }
// }

// Custom simd wrapper implementation.
impl Perlin {
    pub fn batched_2d(
        &mut self,
        output: &mut SimdArray<f32, 1024>,
        x_array: &SimdArray<f32, 1024>,
        y_array: &SimdArray<f32, 1024>,
        octave: &Octave2D,
        weight_coef: f32,
        channel_seed: u64,
        octave_offset: f32,
    ) {
        let freq = ArchSimd::<f32>::splat(octave.scale.x);

        let float_one_vec = ArchSimd::<f32>::splat(1.0);

        let x_grads = ArchSimd::load(&X_GRADIENTS_2D[..]);
        let y_grads = ArchSimd::load(&Y_GRADIENTS_2D[..]);

        for i in (0..1024).step_by(ArchSimd::<f32>::LANES) {
            let x_vec = unsafe { ArchSimd::load(&x_array.data.assume_init_ref()[i..]) };
            let y_vec = unsafe { ArchSimd::load(&y_array.data.assume_init_ref()[i..]) };

            let x_scaled = x_vec * freq;
            let y_scaled = y_vec * freq;

            let x_scaled_floored = x_scaled.floor();
            let y_scaled_floored = y_scaled.floor();

            let x_grid_lo = x_scaled_floored.cast_int_trunc();
            let y_grid_lo = y_scaled_floored.cast_int_trunc();

            let x_dist_lo = x_scaled - x_scaled_floored;
            let y_dist_lo = y_scaled - y_scaled_floored;
            let x_dist_hi = x_dist_lo - float_one_vec;
            let y_dist_hi = y_dist_lo - float_one_vec;

            let x_lerp = x_dist_lo.quintic_lerp();
            let y_lerp = y_dist_lo.quintic_lerp();

            let (mix_tl, mix_tr, mix_bl, mix_br) =
                self.random_gen.mix_u32_four_group(x_grid_lo.raw_cast(), y_grid_lo.raw_cast());

            let indices_tl = mix_tl >> 29;
            let indices_tr = mix_tr >> 29;
            let indices_bl = mix_bl >> 29;
            let indices_br = mix_br >> 29;

            let x_grads_tl = x_grads.runtime_permute(indices_tl);
            let y_grads_tl = y_grads.runtime_permute(indices_tl);
            let x_grads_tr = x_grads.runtime_permute(indices_tr);
            let y_grads_tr = y_grads.runtime_permute(indices_tr);
            let x_grads_bl = x_grads.runtime_permute(indices_bl);
            let y_grads_bl = y_grads.runtime_permute(indices_bl);
            let x_grads_br = x_grads.runtime_permute(indices_br);
            let y_grads_br = y_grads.runtime_permute(indices_br);

            let prod_tl = x_grads_tl.mul_add(x_dist_lo, y_grads_tl * y_dist_lo);
            let prod_tr = x_grads_tr.mul_add(x_dist_lo, y_grads_tr * y_dist_hi);
            let top_lerp = y_lerp.mul_add(prod_tr - prod_tl, prod_tl);

            let prod_bl = x_grads_bl.mul_add(x_dist_hi, y_grads_bl * y_dist_lo);
            let prod_br = x_grads_br.mul_add(x_dist_hi, y_grads_br * y_dist_hi);
            let bottom_lerp = y_lerp.mul_add(prod_br - prod_bl, prod_bl);

            let result = x_lerp.mul_add(bottom_lerp - top_lerp, top_lerp);
            
            unsafe { result.store_aligned(&mut output.data.assume_init_mut()[i..]); }
        }
    }
}

// ISSUE: ICE compiler error when using const generics for custom length.
// impl Perlin {
//     pub fn batched_2d(
//         &mut self,
//         output: &mut SimdArray<f32, 1024>,
//         x_array: &SimdArray<f32, 1024>,
//         y_array: &SimdArray<f32, 1024>,
//         octave: &Octave2D,
//         weight_coef: f32,
//         channel_seed: u64,
//         octave_offset: f32,
//     ) where ArchSimd<f32>: StdFloat
//     {
//         let freq = ArchSimd::<f32>::splat(octave.scale.x);

//         let int_one_vec = ArchSimd::<i32>::splat(1);
//         let float_one_vec = ArchSimd::<f32>::splat(1.0);

//         let sign_bit = ArchSimd::<u32>::splat(0x80000000);

//         for i in (0..1024).step_by(f32::LANES) {
//             let x_vec = x_array.load_simd(i);
//             let y_vec = y_array.load_simd(i);

//             let x_scaled = x_vec * freq;
//             let y_scaled = y_vec * freq;

//             let x_scaled_floored = x_scaled.floor();
//             let y_scaled_floored = y_scaled.floor();

//             let x_grid_lo = unsafe { x_scaled_floored.floor().to_int_unchecked() };
//             let y_grid_lo = unsafe { y_scaled_floored.floor().to_int_unchecked() };
//             let x_grid_hi = x_grid_lo + int_one_vec;
//             let y_grid_hi = y_grid_lo + int_one_vec;

//             let x_dist_lo = x_scaled - x_scaled_floored;
//             let y_dist_lo = y_scaled - y_scaled_floored;
//             let x_dist_hi = x_dist_lo - float_one_vec;
//             let y_dist_hi = y_dist_lo - float_one_vec;

//             let x_lerp = x_dist_lo.quintic_lerp();
//             let y_lerp = y_dist_lo.quintic_lerp();

//             let (mix_tl, mix_tr, mix_bl, mix_br) =
//                 self.random_gen.mix_u32_four_group(x_grid_lo.cast(), x_grid_hi.cast(), y_grid_lo.cast(), y_grid_hi.cast());

//             let indices_tl = mix_tl;
//             let indices_tr = mix_tr;
//             let indices_bl = mix_bl;
//             let indices_br = mix_br;

//             let x_dist_lo_raw: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_lo) };
//             let y_dist_lo_raw: ArchSimd<u32> = unsafe { std::mem::transmute(y_dist_lo) };
//             let x_dist_hi_raw: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_hi) };
//             let y_dist_hi_raw: ArchSimd<u32> = unsafe { std::mem::transmute(y_dist_hi) };

//             let x_prod_tl: ArchSimd<f32> = unsafe { std::mem::transmute((indices_tl & sign_bit) ^ x_dist_lo_raw) };
//             let y_prod_tl: ArchSimd<f32> = unsafe { std::mem::transmute(((indices_tl << 1) & sign_bit) ^ y_dist_lo_raw) };
//             let x_prod_tr: ArchSimd<f32> = unsafe { std::mem::transmute((indices_tr & sign_bit) ^ x_dist_lo_raw) };
//             let y_prod_tr: ArchSimd<f32> = unsafe { std::mem::transmute(((indices_tr << 1) & sign_bit) ^ y_dist_hi_raw) };
//             let x_prod_bl: ArchSimd<f32> = unsafe { std::mem::transmute((indices_bl & sign_bit) ^ x_dist_hi_raw) };
//             let y_prod_bl: ArchSimd<f32> = unsafe { std::mem::transmute(((indices_bl << 1) & sign_bit) ^ y_dist_lo_raw) };
//             let x_prod_br: ArchSimd<f32> = unsafe { std::mem::transmute((indices_br & sign_bit) ^ x_dist_hi_raw) };
//             let y_prod_br: ArchSimd<f32> = unsafe { std::mem::transmute(((indices_br << 1) & sign_bit) ^ y_dist_hi_raw) };

//             let prod_tl = x_prod_tl + y_prod_tl;
//             let prod_tr = x_prod_tr + y_prod_tr;
//             let prod_bl = x_prod_bl + y_prod_bl;
//             let prod_br = x_prod_br + y_prod_br;

//             let top_lerp = y_lerp.mul_add(prod_tr - prod_tl, prod_tl);
//             let bottom_lerp = y_lerp.mul_add(prod_br - prod_bl, prod_bl);

//             let result = x_lerp.mul_add(bottom_lerp - top_lerp, top_lerp);
            
//             output.store_simd(i, result);
//         }
//     }
// }


// impl Perlin {
//     pub fn batched_2d(
//         &mut self,
//         output: &mut SimdArray<f32, 1024>,
//         x_array: &SimdArray<f32, 1024>,
//         y_array: &SimdArray<f32, 1024>,
//         octave: &Octave2D,
//         weight_coef: f32,
//         channel_seed: u64,
//         octave_offset: f32,
//     ) where ArchSimd<f32>: StdFloat
//     {
//         let freq = ArchSimd::<f32>::splat(octave.scale.x);

//         let int_one_vec = ArchSimd::<i32>::splat(1);
//         let float_one_vec = ArchSimd::<f32>::splat(1.0);
//         let rem_mask = ArchSimd::<u32>::splat(0xF);

//         let x_prime = ArchSimd::<i32>::splat(501125321);
//         let y_prime = ArchSimd::<i32>::splat(1136930381);

//         // let entry_mask = ArchSimd::<u32>::splat(0xC0000000);
//         let exp_bit = ArchSimd::<u32>::splat(0x40000000);
//         let exit_mask = ArchSimd::<u32>::splat(0x7F000000);
//         let sign_bit = ArchSimd::<u32>::splat(0x80000000);
//         let next_bit = ArchSimd::<u32>::splat(0x40000000);
//         let ones_exp = ArchSimd::<u32>::splat(0x3F800000);

//         // 00 -> 11111111111
//         // 10 -> 01111111111
//         // 01 -> 00111111111
//         // 11 -> 10111111111

//         // 0 0 -> 0100000000 -> 00000000000000000
//         // 1 0 -> 1100000000 -> 10000000000000000
//         // 0 1 -> 1011111111 -> 00111111100000000
//         // 1 1 -> 1011111111 -> 10111111100000000

//         for i in (0..1024).step_by(f32::LANES) {
//             let x_vec = x_array.load_simd(i);
//             let y_vec = y_array.load_simd(i);

//             let x_scaled = x_vec * freq;
//             let y_scaled = y_vec * freq;

//             let x_scaled_floored = x_scaled.floor();
//             let y_scaled_floored = y_scaled.floor();

//             let x_grid_lo = unsafe { x_scaled_floored.floor().to_int_unchecked() };
//             let y_grid_lo = unsafe { y_scaled_floored.floor().to_int_unchecked() };
//             let x_grid_hi = x_grid_lo + int_one_vec;
//             let y_grid_hi = y_grid_lo + int_one_vec;

//             // let x_grid_lo_primed = x_grid_lo * x_prime;
//             // let y_grid_lo_primed = y_grid_lo * y_prime;
//             // let x_grid_hi_primed = x_grid_hi * x_prime;
//             // let y_grid_hi_primed = y_grid_hi * y_prime;

//             let x_dist_lo = x_scaled - x_scaled_floored;
//             let y_dist_lo = y_scaled - y_scaled_floored;
//             let x_dist_hi = x_dist_lo - float_one_vec;
//             let y_dist_hi = y_dist_lo - float_one_vec;

//             let x_lerp = x_dist_lo.quintic_lerp();
//             let y_lerp = y_dist_lo.quintic_lerp();

//             let (mix_tl, mix_tr, mix_bl, mix_br) =
//                 self.random_gen.mix_u32_four_group(x_grid_lo.cast(), x_grid_hi.cast(), y_grid_lo.cast(), y_grid_hi.cast());

//             let indices_tl = mix_tl;
//             let indices_tr = mix_tr;
//             let indices_bl = mix_bl;
//             let indices_br = mix_br;

//             // std::hint::black_box(indices_tl);
//             // std::hint::black_box(indices_tr);
//             // std::hint::black_box(indices_bl);
//             // std::hint::black_box(indices_br);

//             // let indices_tl = self.random_gen.mix_i32_simd_pair_fast(x_grid_lo_primed, y_grid_lo_primed) >> 28;
//             // let indices_tr = self.random_gen.mix_i32_simd_pair_fast(x_grid_lo_primed, y_grid_hi_primed) >> 28;
//             // let indices_bl = self.random_gen.mix_i32_simd_pair_fast(x_grid_hi_primed, y_grid_lo_primed) >> 28;
//             // let indices_br = self.random_gen.mix_i32_simd_pair_fast(x_grid_hi_primed, y_grid_hi_primed) >> 28;

//             // let x_grads_tl = ArchSimd::gather_u32(&X_GRADIENTS_2D, indices_tl);
//             // let y_grads_tl = ArchSimd::gather_u32(&Y_GRADIENTS_2D, indices_tl);
//             // let x_grads_tr = ArchSimd::gather_u32(&X_GRADIENTS_2D, indices_tr);
//             // let y_grads_tr = ArchSimd::gather_u32(&Y_GRADIENTS_2D, indices_tr);
//             // let x_grads_bl = ArchSimd::gather_u32(&X_GRADIENTS_2D, indices_bl);
//             // let y_grads_bl = ArchSimd::gather_u32(&Y_GRADIENTS_2D, indices_bl);
//             // let x_grads_br = ArchSimd::gather_u32(&X_GRADIENTS_2D, indices_br);
//             // let y_grads_br = ArchSimd::gather_u32(&Y_GRADIENTS_2D, indices_br);

//             // let x_dist_lo_raw: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_lo) };
//             // let y_dist_lo_raw: ArchSimd<u32> = unsafe { std::mem::transmute(y_dist_lo) };
//             // let x_dist_hi_raw: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_hi) };
//             // let y_dist_hi_raw: ArchSimd<u32> = unsafe { std::mem::transmute(y_dist_hi) };

//             // let x_prod_tl: ArchSimd<f32> = unsafe { std::mem::transmute((indices_tl & sign_bit) ^ x_dist_lo_raw) };
//             // let y_prod_tl: ArchSimd<f32> = unsafe { std::mem::transmute(((indices_tl << 1) & sign_bit) ^ y_dist_lo_raw) };
//             // let x_prod_tr: ArchSimd<f32> = unsafe { std::mem::transmute((indices_tr & sign_bit) ^ x_dist_lo_raw) };
//             // let y_prod_tr: ArchSimd<f32> = unsafe { std::mem::transmute(((indices_tr << 1) & sign_bit) ^ y_dist_hi_raw) };
//             // let x_prod_bl: ArchSimd<f32> = unsafe { std::mem::transmute((indices_bl & sign_bit) ^ x_dist_hi_raw) };
//             // let y_prod_bl: ArchSimd<f32> = unsafe { std::mem::transmute(((indices_bl << 1) & sign_bit) ^ y_dist_lo_raw) };
//             // let x_prod_br: ArchSimd<f32> = unsafe { std::mem::transmute((indices_br & sign_bit) ^ x_dist_hi_raw) };
//             // let y_prod_br: ArchSimd<f32> = unsafe { std::mem::transmute(((indices_br << 1) & sign_bit) ^ y_dist_hi_raw) };

//             // let x_dist_sum: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_lo + x_dist_hi) };
//             // let x_dist_dif: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_lo - x_dist_hi) };

//             // let unsigned_prod_tl = (indices_tl & next_bit).simd_eq(next_bit).select(x_dist_sum, x_dist_dif);
//             // let unsigned_prod_tr = (indices_tr & next_bit).simd_eq(next_bit).select(x_dist_sum, x_dist_dif);
//             // let unsigned_prod_bl = (indices_bl & next_bit).simd_eq(next_bit).select(x_dist_sum, x_dist_dif);
//             // let unsigned_prod_br = (indices_br & next_bit).simd_eq(next_bit).select(x_dist_sum, x_dist_dif);

//             // let prod_tl: ArchSimd<f32> = unsafe { std::mem::transmute((indices_tl & sign_bit) ^ unsigned_prod_tl) };
//             // let prod_tr: ArchSimd<f32> = unsafe { std::mem::transmute((indices_tr & sign_bit) ^ unsigned_prod_tr) };
//             // let prod_bl: ArchSimd<f32> = unsafe { std::mem::transmute((indices_bl & sign_bit) ^ unsigned_prod_bl) };
//             // let prod_br: ArchSimd<f32> = unsafe { std::mem::transmute((indices_br & sign_bit) ^ unsigned_prod_br) };

//             // tl uses x_dist_lo, y_dist_lo
//             let sum_lo_lo: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_lo + y_dist_lo) };
//             let dif_lo_lo: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_lo - y_dist_lo) };

//             // tr uses x_dist_lo, y_dist_hi
//             let sum_lo_hi: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_lo + y_dist_hi) };
//             let dif_lo_hi: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_lo - y_dist_hi) };

//             // bl uses x_dist_hi, y_dist_lo
//             let sum_hi_lo: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_hi + y_dist_lo) };
//             let dif_hi_lo: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_hi - y_dist_lo) };

//             // br uses x_dist_hi, y_dist_hi
//             let sum_hi_hi: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_hi + y_dist_hi) };
//             let dif_hi_hi: ArchSimd<u32> = unsafe { std::mem::transmute(x_dist_hi - y_dist_hi) };

//             let unsigned_prod_tl = (mix_tl & next_bit).simd_eq(next_bit).select(sum_lo_lo, dif_lo_lo);
//             let unsigned_prod_tr = (mix_tr & next_bit).simd_eq(next_bit).select(sum_lo_hi, dif_lo_hi);
//             let unsigned_prod_bl = (mix_bl & next_bit).simd_eq(next_bit).select(sum_hi_lo, dif_hi_lo);
//             let unsigned_prod_br = (mix_br & next_bit).simd_eq(next_bit).select(sum_hi_hi, dif_hi_hi);

//             let prod_tl: ArchSimd<f32> = unsafe { std::mem::transmute((mix_tl & sign_bit) ^ unsigned_prod_tl) };
//             let prod_tr: ArchSimd<f32> = unsafe { std::mem::transmute((mix_tr & sign_bit) ^ unsigned_prod_tr) };
//             let prod_bl: ArchSimd<f32> = unsafe { std::mem::transmute((mix_bl & sign_bit) ^ unsigned_prod_bl) };
//             let prod_br: ArchSimd<f32> = unsafe { std::mem::transmute((mix_br & sign_bit) ^ unsigned_prod_br) };

//             // println!("x_grads_tl: {:?}, tl_hash: {:?}, after sign bit mask: {:?}", x_grads_tl, indices_tl, indices_tl & sign_bit);

//             // let prod_tl = x_prod_tl + y_prod_tl;
//             // let prod_tr = x_prod_tr + y_prod_tr;
//             // let prod_bl = x_prod_bl + y_prod_bl;
//             // let prod_br = x_prod_br + y_prod_br;

//             let top_lerp = y_lerp.mul_add(prod_tr - prod_tl, prod_tl);
//             let bottom_lerp = y_lerp.mul_add(prod_br - prod_bl, prod_bl);

//             let result = x_lerp.mul_add(bottom_lerp - top_lerp, top_lerp);
            
//             output.store_simd(i, result);
//         }
//     }
// }