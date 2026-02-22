use crate::math::vec::{Vec2, Vec3};
use crate::simd::simd_array::SimdArray;
use std::sync::LazyLock;

pub const ROW_SIZE: usize = 32;
pub const MAP_SIZE: usize = 1024;
pub const VOL_SIZE: usize = 32768;

pub const NUM_DIRECTIONS: usize = 16;
pub const LO_EPSILON: f64 = 1e-4;
pub const HI_EPSILON: f64 = 1.0 - 1e-4;

pub const GRADIENTS_3D_LOOKUP: [(f64, f64, f64); 12] = [
    (1.0, 1.0, 0.0),
    (-1.0, 1.0, 0.0),
    (1.0, -1.0, 0.0),
    (-1.0, -1.0, 0.0),
    (1.0, 0.0, 1.0),
    (-1.0, 0.0, 1.0),
    (1.0, 0.0, -1.0),
    (-1.0, 0.0, -1.0),
    (0.0, 1.0, 1.0),
    (0.0, -1.0, 1.0),
    (0.0, 1.0, -1.0),
    (0.0, -1.0, -1.0),
];

pub type PerlinVec = SimdArray<f32, ROW_SIZE>;
pub type PerlinMap = SimdArray<f32, MAP_SIZE>;
pub type PerlinVol = SimdArray<f32, VOL_SIZE>;

pub type PerlinVecPair = Vec2<PerlinVec>;
pub type PerlinVecTriple = Vec3<PerlinVec>;

// pub static GRADIENTS_2D: LazyLock<[Vec2<f64>; NUM_DIRECTIONS]> = LazyLock::new(|| {
//     let mut gradients: [Vec2<f64>; NUM_DIRECTIONS] = [Vec2::<f64> {x: 0.0, y: 0.0}; NUM_DIRECTIONS];

//     let scale: f64 = 2.0_f64.sqrt();
//     for i in 0..NUM_DIRECTIONS {
//         let angle: f64 = (2.0 * PI) * (i as f64 / NUM_DIRECTIONS as f64);
//         let x: f64 = angle.cos() * scale;
//         let y: f64 = angle.sin() * scale;
//         gradients[i] = Vec2::<f64> {x, y};
//     }

//     gradients
// });

pub const GRADIENTS_2D: [Vec2<f32>; 16] = [
    Vec2 {
        x: 1.4142135623730951,
        y: 0.0000000000000000,
    },
    Vec2 {
        x: 1.3065629648763766,
        y: 0.5411961001338174,
    },
    Vec2 {
        x: 1.0000000000000002,
        y: 1.0000000000000002,
    },
    Vec2 {
        x: 0.5411961001338176,
        y: 1.3065629648763766,
    },
    Vec2 {
        x: 0.0000000000000001,
        y: 1.4142135623730951,
    },
    Vec2 {
        x: -0.5411961001338172,
        y: 1.3065629648763768,
    },
    Vec2 {
        x: -0.9999999999999998,
        y: 1.0000000000000004,
    },
    Vec2 {
        x: -1.3065629648763764,
        y: 0.5411961001338180,
    },
    Vec2 {
        x: -1.4142135623730951,
        y: 0.0000000000000000,
    },
    Vec2 {
        x: -1.3065629648763768,
        y: -0.5411961001338169,
    },
    Vec2 {
        x: -1.0000000000000004,
        y: -0.9999999999999998,
    },
    Vec2 {
        x: -0.5411961001338183,
        y: -1.3065629648763764,
    },
    Vec2 {
        x: -0.0000000000000002,
        y: -1.4142135623730951,
    },
    Vec2 {
        x: 0.5411961001338166,
        y: -1.3065629648763771,
    },
    Vec2 {
        x: 0.9999999999999996,
        y: -1.0000000000000007,
    },
    Vec2 {
        x: 1.3065629648763771,
        y: -0.5411961001338161,
    },
];

pub static GRADIENTS_3D: LazyLock<[Vec3<f64>; NUM_DIRECTIONS]> = LazyLock::new(|| {
    let mut gradients: [Vec3<f64>; NUM_DIRECTIONS] =
        [Vec3::<f64>::new(0.0, 0.0, 0.0); NUM_DIRECTIONS];

    for i in (0..NUM_DIRECTIONS).step_by(12) {
        for j in 0..12 {
            gradients[i] = GRADIENTS_3D_LOOKUP[j].into();
        }
    }

    gradients
});
