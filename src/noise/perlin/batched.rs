// use crate::math::random::Random;
// use crate::math::vec::{Vec2, Vec3};
use crate::noise::perlin::constants::*;
use crate::noise::perlin::containers::*;
use crate::perlin::Perlin;
use crate::simd::simd_array::SimdArray;
use crate::simd::arch_simd::{ArchSimd};
use crate::simd::simd_traits::*;
// use crate::simd::simd_vec::core::SimdVec;
// use crate::simd::architectures::families::Avx2Family;

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
        // Constants.
        let six: ArchSimd<f32> = ArchSimd::splat(6.0);
        let ten: ArchSimd<f32> = ArchSimd::splat(10.0);
        let fifteen: ArchSimd<f32> = ArchSimd::splat(15.0);
        let one: ArchSimd<f32> = ArchSimd::splat(1.0);

        // Hash constants.
        const BYTE_SHUFFLE: [u8; 64] = [
            3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1,
            3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1,
            3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1,
            3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1,
        ];
        let shuffle_indices = ArchSimd::<u8>::load(&BYTE_SHUFFLE[..]);
        let channel_seed = ArchSimd::splat(self.random_gen.channel_seed as u32);
        let prime = ArchSimd::splat(0x85ebca6b_u32 as u32);
    
        // Frequency constant.
        let freq = ArchSimd::<f32>::splat(octave.scale.x);

        for i in (0..1024).step_by(ArchSimd::<f32>::LANES) {

            // Load and scale: 4
            let x_vec = x_array.load_simd(i);
            let y_vec = y_array.load_simd(i);

            let x_scaled = x_vec * freq;
            let y_scaled = y_vec * freq;

            // Gridpoints and distances: 8
            let x_scaled_floored = x_scaled.floor();
            let y_scaled_floored = y_scaled.floor();

            let x_grid_lo = x_scaled_floored.cast_int_trunc();
            let y_grid_lo = y_scaled_floored.cast_int_trunc();

            let x_dist_lo = x_scaled - x_scaled_floored;
            let y_dist_lo = y_scaled - y_scaled_floored;
            let x_dist_hi = x_dist_lo - one;
            let y_dist_hi = y_dist_lo - one;

            // Lerp fade calculation: 10 
            let t = x_dist_lo;
            let s = y_dist_lo;
            let x_lerp = t * t * t * t.mul_add(t.mul_sub(six, fifteen), ten);
            let y_lerp = s * s * s * s.mul_add(s.mul_sub(six, fifteen), ten);

            // Hash: 16
            let x1: ArchSimd<u32> = x_grid_lo.raw_cast() * channel_seed;
            let y1: ArchSimd<u32> = y_grid_lo.raw_cast() * channel_seed;
            let x2 = x1 + channel_seed;
            let y2 = y1 + channel_seed;

            let x1_shuf = x1.permute_8(shuffle_indices) ^ prime;
            let y1_shuf = y1.permute_8(shuffle_indices) ^ prime;
            let x2_shuf = x2.permute_8(shuffle_indices) ^ prime;
            let y2_shuf = y2.permute_8(shuffle_indices) ^ prime;

            let mix_tl = x1_shuf * y1_shuf;
            let mix_tr = x1_shuf * y2_shuf;
            let mix_bl = x2_shuf * y1_shuf;
            let mix_br = x2_shuf * y2_shuf;

            // Permute Gather: 12
            let indices_tl = mix_tl >> 29;
            let indices_tr = mix_tr >> 29;
            let indices_bl = mix_bl >> 29;
            let indices_br = mix_br >> 29;

            let x_grads_tl = indices_tl.gather(&X_GRADIENTS_2D);
            let y_grads_tl = indices_tl.gather(&Y_GRADIENTS_2D);
            let x_grads_tr = indices_tr.gather(&X_GRADIENTS_2D);
            let y_grads_tr = indices_tr.gather(&Y_GRADIENTS_2D);
            let x_grads_bl = indices_bl.gather(&X_GRADIENTS_2D);
            let y_grads_bl = indices_bl.gather(&Y_GRADIENTS_2D);
            let x_grads_br = indices_br.gather(&X_GRADIENTS_2D);
            let y_grads_br = indices_br.gather(&Y_GRADIENTS_2D);

            let prod_tl = x_grads_tl.mul_add(x_dist_lo, y_grads_tl * y_dist_lo);
            let prod_tr = x_grads_tr.mul_add(x_dist_lo, y_grads_tr * y_dist_hi);
            let top_lerp = y_lerp.mul_add(prod_tr - prod_tl, prod_tl);

            let prod_bl = x_grads_bl.mul_add(x_dist_hi, y_grads_bl * y_dist_lo);
            let prod_br = x_grads_br.mul_add(x_dist_hi, y_grads_br * y_dist_hi);
            let bottom_lerp = y_lerp.mul_add(prod_br - prod_bl, prod_bl);

            let result = x_lerp.mul_add(bottom_lerp - top_lerp, top_lerp);

            // // Interpolation: 14
            // let prod_tl = x_grads_tl.mul_add(x_dist_lo, y_grads_tl * y_dist_lo);
            // let prod_tr = x_grads_tr.mul_add(x_dist_lo, y_grads_tr * y_dist_hi);
            // let top_lerp = y_lerp.mul_add(prod_tr - prod_tl, prod_tl);

            // let prod_bl = x_grads_bl.mul_add(x_dist_hi, y_grads_bl * y_dist_lo);
            // let prod_br = x_grads_br.mul_add(x_dist_hi, y_grads_br * y_dist_hi);
            // let bottom_lerp = y_lerp.mul_add(prod_br - prod_bl, prod_bl);

            // let result = x_lerp.mul_add(bottom_lerp - top_lerp, top_lerp);
            
            // Store: 1
            unsafe { result.store_aligned(&mut output.data.assume_init_mut()[i..]); }
        }
    }

    pub fn batched_3d(
        &mut self,
        output: &mut SimdArray<f32, 32768>,
        x_array: &SimdArray<f32, 32768>,
        y_array: &SimdArray<f32, 32768>,
        z_array: &SimdArray<f32, 32768>,
        octave: &Octave3D,
        weight_coef: f32,
        channel_seed: u64,
        octave_offset: f32,
    ) {
        // Constants.
        let six: ArchSimd<f32> = ArchSimd::splat(6.0);
        let ten: ArchSimd<f32> = ArchSimd::splat(10.0);
        let fifteen: ArchSimd<f32> = ArchSimd::splat(15.0);
        let one: ArchSimd<f32> = ArchSimd::splat(1.0);
        let zero: ArchSimd<f32> = ArchSimd::splat(0.0);
        let one_int: ArchSimd<u32> = ArchSimd::splat(1);
        let sixteen_int: ArchSimd<u32> = ArchSimd::splat(16);

        let c1: ArchSimd<u32> = ArchSimd::splat(0x30FF20AA);
        let c2: ArchSimd<u32> = ArchSimd::splat(0xFF0FCA0C);
        let c3: ArchSimd<u32> = ArchSimd::splat(0xCFF08CC0);

        // Hash constants.
        const BYTE_SHUFFLE: [u8; 64] = [
            3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1,
            3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1,
            3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1,
            3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1,
        ];

        let shuffle_indices = ArchSimd::<u8>::load(&BYTE_SHUFFLE[..]);
        let channel_seed = ArchSimd::splat(self.random_gen.channel_seed as u32);
        let prime = ArchSimd::splat(0x85ebca6b_u32 as u32);
    
        // Frequency constant.
        let freq = ArchSimd::<f32>::splat(octave.scale.x);

        for i in (0..32768).step_by(ArchSimd::<f32>::LANES) {

            // Load and scale: 6
            let x_vec = x_array.load_simd(i);
            let y_vec = y_array.load_simd(i);
            let z_vec = z_array.load_simd(i);

            let x_scaled = x_vec * freq;
            let y_scaled = y_vec * freq;
            let z_scaled = z_vec * freq;

            // Gridpoints and distances: 12
            let x_scaled_floored = x_scaled.floor();
            let y_scaled_floored = y_scaled.floor();
            let z_scaled_floored = z_scaled.floor();

            let x_grid_lo = x_scaled_floored.cast_int_trunc();
            let y_grid_lo = y_scaled_floored.cast_int_trunc();
            let z_grid_lo = z_scaled_floored.cast_int_trunc();

            let x_dist_lo = x_scaled - x_scaled_floored;
            let y_dist_lo = y_scaled - y_scaled_floored;
            let z_dist_lo = z_scaled - z_scaled_floored;
            let x_dist_hi = x_dist_lo - one;
            let y_dist_hi = y_dist_lo - one;
            let z_dist_hi = z_dist_lo - one;

            // Lerp fade calculation: 15
            let t = x_dist_lo;
            let s = y_dist_lo;
            let u = z_dist_lo;
            let x_lerp = t * t * t * t.mul_add(t.mul_sub(six, fifteen), ten);
            let y_lerp = s * s * s * s.mul_add(s.mul_sub(six, fifteen), ten);
            let z_lerp = u * u * u * u.mul_add(u.mul_sub(six, fifteen), ten);

            // Hash: 26
            let x1: ArchSimd<u32> = x_grid_lo.raw_cast() * channel_seed;
            let y1: ArchSimd<u32> = y_grid_lo.raw_cast() * channel_seed;
            let z1: ArchSimd<u32> = z_grid_lo.raw_cast() * channel_seed;
            let x2 = x1 + channel_seed;
            let y2 = y1 + channel_seed;
            let z2 = z1 + channel_seed;

            let x1_shuf = x1.permute_8(shuffle_indices) ^ prime;
            let y1_shuf = y1.permute_8(shuffle_indices) ^ prime;
            let z1_shuf = z1.permute_8(shuffle_indices) ^ prime;
            let x2_shuf = x2.permute_8(shuffle_indices) ^ prime;
            let y2_shuf = y2.permute_8(shuffle_indices) ^ prime;
            let z2_shuf = z2.permute_8(shuffle_indices) ^ prime;

            let mix_tlf = x1_shuf * y1_shuf ^ z1_shuf;
            let mix_trf = x1_shuf * y2_shuf ^ z1_shuf;
            let mix_blf = x2_shuf * y1_shuf ^ z1_shuf;
            let mix_brf = x2_shuf * y2_shuf ^ z1_shuf;
            let mix_tlb = x1_shuf * y1_shuf ^ z2_shuf;
            let mix_trb = x1_shuf * y2_shuf ^ z2_shuf;
            let mix_blb = x2_shuf * y1_shuf ^ z2_shuf;
            let mix_brb = x2_shuf * y2_shuf ^ z2_shuf;

            // Products: 176
            let indices_tlf = mix_tlf >> 28;
            let indices_trf = mix_trf >> 28;
            let indices_blf = mix_blf >> 28;
            let indices_brf = mix_brf >> 28;
            let indices_tlb = mix_tlb >> 28;
            let indices_trb = mix_trb >> 28;
            let indices_blb = mix_blb >> 28;
            let indices_brb = mix_brb >> 28;

            let x_grads_tlf = c1 >> indices_tlf;
            let x_grads_trf = c1 >> indices_trf;
            let x_grads_blf = c1 >> indices_blf;
            let x_grads_brf = c1 >> indices_brf;
            let x_grads_tlb = c1 >> indices_tlb;
            let x_grads_trb = c1 >> indices_trb;
            let x_grads_blb = c1 >> indices_blb;
            let x_grads_brb = c1 >> indices_brb;
            let y_grads_tlf = c2 >> indices_tlf;
            let y_grads_trf = c2 >> indices_trf;
            let y_grads_blf = c2 >> indices_blf;
            let y_grads_brf = c2 >> indices_brf;
            let y_grads_tlb = c2 >> indices_tlb;
            let y_grads_trb = c2 >> indices_trb;
            let y_grads_blb = c2 >> indices_blb;
            let y_grads_brb = c2 >> indices_brb;
            let z_grads_tlf = c3 >> indices_tlf;
            let z_grads_trf = c3 >> indices_trf;
            let z_grads_blf = c3 >> indices_blf;
            let z_grads_brf = c3 >> indices_brf;
            let z_grads_tlb = c3 >> indices_tlb;
            let z_grads_trb = c3 >> indices_trb;
            let z_grads_blb = c3 >> indices_blb;
            let z_grads_brb = c3 >> indices_brb;

            let x_one_prod_tlf: ArchSimd<f32> = ((x_grads_tlf << 31) ^ x_dist_lo.raw_cast()).raw_cast();
            let x_one_prod_trf: ArchSimd<f32> = ((x_grads_trf << 31) ^ x_dist_lo.raw_cast()).raw_cast();
            let x_one_prod_blf: ArchSimd<f32> = ((x_grads_blf << 31) ^ x_dist_lo.raw_cast()).raw_cast();
            let x_one_prod_brf: ArchSimd<f32> = ((x_grads_brf << 31) ^ x_dist_lo.raw_cast()).raw_cast();
            let x_one_prod_tlb: ArchSimd<f32> = ((x_grads_tlb << 31) ^ x_dist_hi.raw_cast()).raw_cast();
            let x_one_prod_trb: ArchSimd<f32> = ((x_grads_trb << 31) ^ x_dist_hi.raw_cast()).raw_cast();
            let x_one_prod_blb: ArchSimd<f32> = ((x_grads_blb << 31) ^ x_dist_hi.raw_cast()).raw_cast();
            let x_one_prod_brb: ArchSimd<f32> = ((x_grads_brb << 31) ^ x_dist_hi.raw_cast()).raw_cast();
            let y_one_prod_tlf: ArchSimd<f32> = ((y_grads_tlf << 31) ^ y_dist_lo.raw_cast()).raw_cast();
            let y_one_prod_trf: ArchSimd<f32> = ((y_grads_trf << 31) ^ y_dist_lo.raw_cast()).raw_cast();
            let y_one_prod_blf: ArchSimd<f32> = ((y_grads_blf << 31) ^ y_dist_hi.raw_cast()).raw_cast();
            let y_one_prod_brf: ArchSimd<f32> = ((y_grads_brf << 31) ^ y_dist_hi.raw_cast()).raw_cast();
            let y_one_prod_tlb: ArchSimd<f32> = ((y_grads_tlb << 31) ^ y_dist_lo.raw_cast()).raw_cast();
            let y_one_prod_trb: ArchSimd<f32> = ((y_grads_trb << 31) ^ y_dist_lo.raw_cast()).raw_cast();
            let y_one_prod_blb: ArchSimd<f32> = ((y_grads_blb << 31) ^ y_dist_hi.raw_cast()).raw_cast();
            let y_one_prod_brb: ArchSimd<f32> = ((y_grads_brb << 31) ^ y_dist_hi.raw_cast()).raw_cast();
            let z_one_prod_tlf: ArchSimd<f32> = ((z_grads_tlf << 31) ^ z_dist_lo.raw_cast()).raw_cast();
            let z_one_prod_trf: ArchSimd<f32> = ((z_grads_trf << 31) ^ z_dist_hi.raw_cast()).raw_cast();
            let z_one_prod_blf: ArchSimd<f32> = ((z_grads_blf << 31) ^ z_dist_lo.raw_cast()).raw_cast();
            let z_one_prod_brf: ArchSimd<f32> = ((z_grads_brf << 31) ^ z_dist_hi.raw_cast()).raw_cast();
            let z_one_prod_tlb: ArchSimd<f32> = ((z_grads_tlb << 31) ^ z_dist_lo.raw_cast()).raw_cast();
            let z_one_prod_trb: ArchSimd<f32> = ((z_grads_trb << 31) ^ z_dist_hi.raw_cast()).raw_cast();
            let z_one_prod_blb: ArchSimd<f32> = ((z_grads_blb << 31) ^ z_dist_lo.raw_cast()).raw_cast();
            let z_one_prod_brb: ArchSimd<f32> = ((z_grads_brb << 31) ^ z_dist_hi.raw_cast()).raw_cast();

            let x_zero_mask_tlf = ((x_grads_tlf >> sixteen_int) & one_int).simd_eq(one_int);
            let x_zero_mask_trf = ((x_grads_trf >> sixteen_int) & one_int).simd_eq(one_int);
            let x_zero_mask_blf = ((x_grads_blf >> sixteen_int) & one_int).simd_eq(one_int);
            let x_zero_mask_brf = ((x_grads_brf >> sixteen_int) & one_int).simd_eq(one_int);
            let x_zero_mask_tlb = ((x_grads_tlb >> sixteen_int) & one_int).simd_eq(one_int);
            let x_zero_mask_trb = ((x_grads_trb >> sixteen_int) & one_int).simd_eq(one_int);
            let x_zero_mask_blb = ((x_grads_blb >> sixteen_int) & one_int).simd_eq(one_int);
            let x_zero_mask_brb = ((x_grads_brb >> sixteen_int) & one_int).simd_eq(one_int);
            let y_zero_mask_tlf = ((y_grads_tlf >> sixteen_int) & one_int).simd_eq(one_int);
            let y_zero_mask_trf = ((y_grads_trf >> sixteen_int) & one_int).simd_eq(one_int);
            let y_zero_mask_blf = ((y_grads_blf >> sixteen_int) & one_int).simd_eq(one_int);
            let y_zero_mask_brf = ((y_grads_brf >> sixteen_int) & one_int).simd_eq(one_int);
            let y_zero_mask_tlb = ((y_grads_tlb >> sixteen_int) & one_int).simd_eq(one_int);
            let y_zero_mask_trb = ((y_grads_trb >> sixteen_int) & one_int).simd_eq(one_int);
            let y_zero_mask_blb = ((y_grads_blb >> sixteen_int) & one_int).simd_eq(one_int);
            let y_zero_mask_brb = ((y_grads_brb >> sixteen_int) & one_int).simd_eq(one_int);
            let z_zero_mask_tlf = ((z_grads_tlf >> sixteen_int) & one_int).simd_eq(one_int);
            let z_zero_mask_trf = ((z_grads_trf >> sixteen_int) & one_int).simd_eq(one_int);
            let z_zero_mask_blf = ((z_grads_blf >> sixteen_int) & one_int).simd_eq(one_int);
            let z_zero_mask_brf = ((z_grads_brf >> sixteen_int) & one_int).simd_eq(one_int);
            let z_zero_mask_tlb = ((z_grads_tlb >> sixteen_int) & one_int).simd_eq(one_int);
            let z_zero_mask_trb = ((z_grads_trb >> sixteen_int) & one_int).simd_eq(one_int);
            let z_zero_mask_blb = ((z_grads_blb >> sixteen_int) & one_int).simd_eq(one_int);
            let z_zero_mask_brb = ((z_grads_brb >> sixteen_int) & one_int).simd_eq(one_int);

            let x_prod_tlf = zero.blend_32(x_one_prod_tlf, x_zero_mask_tlf.raw_cast());
            let x_prod_trf = zero.blend_32(x_one_prod_trf, x_zero_mask_trf.raw_cast());
            let x_prod_blf = zero.blend_32(x_one_prod_blf, x_zero_mask_blf.raw_cast());
            let x_prod_brf = zero.blend_32(x_one_prod_brf, x_zero_mask_brf.raw_cast());
            let x_prod_tlb = zero.blend_32(x_one_prod_tlb, x_zero_mask_tlb.raw_cast());
            let x_prod_trb = zero.blend_32(x_one_prod_trb, x_zero_mask_trb.raw_cast());
            let x_prod_blb = zero.blend_32(x_one_prod_blb, x_zero_mask_blb.raw_cast());
            let x_prod_brb = zero.blend_32(x_one_prod_brb, x_zero_mask_brb.raw_cast());
            let y_prod_tlf = zero.blend_32(y_one_prod_tlf, y_zero_mask_tlf.raw_cast());
            let y_prod_trf = zero.blend_32(y_one_prod_trf, y_zero_mask_trf.raw_cast());
            let y_prod_blf = zero.blend_32(y_one_prod_blf, y_zero_mask_blf.raw_cast());
            let y_prod_brf = zero.blend_32(y_one_prod_brf, y_zero_mask_brf.raw_cast());
            let y_prod_tlb = zero.blend_32(y_one_prod_tlb, y_zero_mask_tlb.raw_cast());
            let y_prod_trb = zero.blend_32(y_one_prod_trb, y_zero_mask_trb.raw_cast());
            let y_prod_blb = zero.blend_32(y_one_prod_blb, y_zero_mask_blb.raw_cast());
            let y_prod_brb = zero.blend_32(y_one_prod_brb, y_zero_mask_brb.raw_cast());
            let z_prod_tlf = zero.blend_32(z_one_prod_tlf, z_zero_mask_tlf.raw_cast());
            let z_prod_trf = zero.blend_32(z_one_prod_trf, z_zero_mask_trf.raw_cast());
            let z_prod_blf = zero.blend_32(z_one_prod_blf, z_zero_mask_blf.raw_cast());
            let z_prod_brf = zero.blend_32(z_one_prod_brf, z_zero_mask_brf.raw_cast());
            let z_prod_tlb = zero.blend_32(z_one_prod_tlb, z_zero_mask_tlb.raw_cast());
            let z_prod_trb = zero.blend_32(z_one_prod_trb, z_zero_mask_trb.raw_cast());
            let z_prod_blb = zero.blend_32(z_one_prod_blb, z_zero_mask_blb.raw_cast());
            let z_prod_brb = zero.blend_32(z_one_prod_brb, z_zero_mask_brb.raw_cast());

            // Interpolation: 30
            let prod_tlf = x_prod_tlf + y_prod_tlf + z_prod_tlf;
            let prod_blf = x_prod_blf + y_prod_blf + z_prod_blf;
            let prod_brf = x_prod_brf + y_prod_brf + z_prod_brf;
            let prod_trf = x_prod_trf + y_prod_trf + z_prod_trf;
            let prod_tlb = x_prod_tlb + y_prod_tlb + z_prod_tlb;
            let prod_trb = x_prod_trb + y_prod_trb + z_prod_trb;
            let prod_blb = x_prod_blb + y_prod_blb + z_prod_blb;
            let prod_brb = x_prod_brb + y_prod_brb + z_prod_brb;
            
            let lerp_tf = z_lerp.mul_add(prod_trf - prod_tlf, prod_tlf);
            let lerp_bf = z_lerp.mul_add(prod_brf - prod_blf, prod_blf);
            let lerp_tb = z_lerp.mul_add(prod_trb - prod_tlb, prod_tlb);
            let lerp_bb = z_lerp.mul_add(prod_brb - prod_blb, prod_blb);

            let lerp_front = y_lerp.mul_add(lerp_bf - lerp_tf, lerp_tf);
            let lerp_back = y_lerp.mul_add(lerp_bb - lerp_tb, lerp_tb);

            let result = x_lerp.mul_add(lerp_back - lerp_front, lerp_front);
            
            // Store: 1
            unsafe { result.store_aligned(&mut output.data.assume_init_mut()[i..]); }
        }
    }
}
