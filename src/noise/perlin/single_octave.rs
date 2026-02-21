use crate::noise::perlin::Perlin;
use crate::noise::perlin::constants::*;
use crate::noise::perlin::containers::*;
use crate::math::vec::{Vec2, Vec3};

impl Perlin {
    // #[inline(never)]
    pub(super) fn single_octave_2d<const INITIALIZE: bool>(
        &mut self,
        result: &mut PerlinMap,
        pos: Vec2<i32>,
        octave: &Octave2D,
        weight_coef: f32,
        channel_seed: u64,
        octave_offset: f32,
    ) {
        let increment: Vec2<f32> = 1.0 / octave.scale;
        let block_pos: Vec2<i32> = pos * 32;
        let weight: f32 = octave.weight * weight_coef;

        // Get the starting gradient coordinates and how far the first sample is to the next one.
        let grid_start: Vec2<i32> = ((block_pos + 1).as_f32() * increment + LO_EPSILON as f32).floor().as_i32();
        let frac_start: Vec2<f32> = (block_pos.as_f32() * increment - grid_start.as_f32()).float_max(Vec2::splat(0.0));

        // Get the distances from the gradient gridpoints.
        let distances: PerlinVecPair = PerlinVecPair {
            x: PerlinVec::iota_custom(frac_start.x + LO_EPSILON as f32, increment.x).fract(),
            y: PerlinVec::iota_custom(frac_start.y + LO_EPSILON as f32, increment.y).fract(),
        };

        // Quintic lerp the distances to get the fade factor.
        let interpolations: PerlinVecPair = PerlinVecPair {
            x: distances.x.quintic_lerp(),
            y: distances.y.quintic_lerp(),
        };

        // Set the channel for random number generation based on the octave scale and selected channel.
        // Note: Octave offset does not currently work.
        self.random_gen.set_channel(channel_seed ^ (octave.scale + octave_offset).sum() as u64);

        // Identify the number of loops to iterate through (better compiler optimization when known).
        let num_loops: Vec2<u32> = (frac_start + increment * ROW_SIZE as f32).ceil().as_u32();

        // Get the amount that next index fraction needs to increase by each iteration.
        let next_index_offset: Vec2<f32> = (1.0 - frac_start) * octave.scale + HI_EPSILON as f32;

        // Initialize gradient vectors.
        let mut d_vecs: PerlinContainer2D = PerlinContainer2D::new_uninit();

        // Set the top gradients.
        let (tl, tr) = d_vecs.tl_tr_mut();
        self.set_gradients_2d(tl, tr, grid_start.x, grid_start.y, next_index_offset.y, octave.scale.y, num_loops.y, &distances.y);

        // Iterate through single x chunks but full y chunks.
        let mut x_cur_index: u32 = 0;
        let mut x_it_scale: f32 = 0.0;
        for x_it in 0..num_loops.x {

            // Identify the current range of x gradients.
            let x_next_index_exact = x_it_scale + next_index_offset.x;
            
            debug_assert!(x_next_index_exact >= 0.0 && x_next_index_exact.is_finite());
            let x_next_index: u32 = unsafe { x_next_index_exact.to_int_unchecked::<u32>().min(ROW_SIZE as u32) as u32 };

            let x_cur_frac_start = unsafe { distances.x.get_unchecked(x_cur_index as usize) };
            let x_chunk_size: u32 = x_next_index - x_cur_index;

            // Set bottom gradients.
            let (bl, br) = d_vecs.bl_br_mut();
            self.set_gradients_2d(bl, br, grid_start.x + x_it as i32 + 1, grid_start.y, next_index_offset.y, octave.scale.y, num_loops.y, &distances.y);
        
            // Perform dot products on x and trilinear interpolation (with quintic fade).
            Self::compute_noise_from_vecs_2d::<INITIALIZE>(
                &d_vecs, x_cur_frac_start, increment.x,
                x_chunk_size as usize, &interpolations, x_cur_index as usize, weight, result
            );

            // Reuse the top and bottom gradients.
            d_vecs.swap_top_bottom();

            // Early exit case. Maybe num_loops calculation can be adjusted to remove this?
            if x_next_index == 32 { break; }

            x_cur_index = x_next_index;
            x_it_scale += octave.scale.x;
        }
    }
}