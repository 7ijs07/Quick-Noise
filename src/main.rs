#![feature(portable_simd)]
#![feature(generic_const_exprs)]

mod simd {
    pub mod arch_simd;
    pub mod simd_array;
}

mod math {
    pub mod random;
    pub mod vec;
}

mod noise {
    pub mod perlin;
}

use crate::noise::perlin::Perlin;

fn main() {
    // Perlin::profile_noise_2d();

    let mut perlin = Perlin::new(0);
    perlin.write_height_map(String::from("testyTest.png"), 32, 4, 64.0, 2.0, 0.5);
    // perlin.write_height_map(String::from("testyTest.png"), 32, 6, 128.0, 2.0, 0.5);
}
