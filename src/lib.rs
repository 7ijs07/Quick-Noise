#![feature(const_cmp, const_trait_impl, generic_const_exprs)]
#![feature(portable_simd)]

mod simd {
    pub mod arch_simd;
    pub mod simd_array;
}

mod math {
    pub mod random;
    pub mod vec;
}

mod testing {
    pub mod profiler;
}

pub mod emit {
    pub mod grayscale;
}

mod noise;
pub use noise::*;
