use std::simd::{Simd, SimdElement};
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};
use std::marker::PhantomData;

// Static dispatch for identifying lane sizes and number of simd registers.

cfg_if::cfg_if! {
    // x86_64
    if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
        pub const SIMD_WIDTH: usize = 64;
        pub const NUM_SIMD_REG: usize = 32;
    } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
        pub const SIMD_WIDTH: usize = 32;
        pub const NUM_SIMD_REG: usize = 16;
    } else if #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))] {
        pub const SIMD_WIDTH: usize = 16;
        pub const NUM_SIMD_REG: usize = 16;
    }

    // aarch64
    else if #[cfg(all(target_arch = "aarch64", target_feature = "sve"))] {
        pub const SIMD_WIDTH: usize = 32;
        pub const NUM_SIMD_REG: usize = 32;
    } else if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
        pub const SIMD_WIDTH: usize = 16;
        pub const NUM_SIMD_REG: usize = 32;
    }

    // wasm
    else if #[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))] {
        pub const SIMD_WIDTH: usize = 16;
        pub const NUM_SIMD_REG: usize = 16;
    }

    // riscv
    else if #[cfg(all(any(target_arch = "riscv64", target_arch = "riscv32"), target_feature = "v"))] {
        pub const SIMD_WIDTH: usize = 32;
        pub const NUM_SIMD_REG: usize = 32;
    }

    // fallback
    else {
        pub const SIMD_WIDTH: usize = 4;
        pub const NUM_SIMD_REG: usize = 8;
    }
}

pub type ArchSimd<T: SimdInfo> = Simd<T, { T::LANES }>;

pub trait SimdInfo: SimdElement {
    const LANES: usize;
}

impl<T: SimdElement> SimdInfo for T {
    const LANES: usize = SIMD_WIDTH / std::mem::size_of::<T>();
}
