#![feature(const_cmp, const_trait_impl, generic_const_exprs, associated_type_defaults)]
#![feature(portable_simd)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_unsafe)]
#![allow(unused_variables)]

pub mod simd;

pub mod math {
    pub mod random;
    pub mod vec;
}

pub mod testing {
    pub mod profiler;
}

pub mod emit {
    pub mod grayscale;
}

mod noise;
pub use noise::*;
