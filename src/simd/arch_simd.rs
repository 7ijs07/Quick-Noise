use std::simd::{Simd, SimdElement};
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};
use std::marker::PhantomData;

// Static dispatch for identifying lane sizes and number of simd registers.

cfg_if::cfg_if!{
    if #[cfg(target_feature = "avx512f")] {
        const SIMD_WIDTH: usize = 64;
        const NUM_SIMD_REG: usize = 32;
    } else if #[cfg(target_feature = "avx2")] {
        const SIMD_WIDTH: usize = 32;
        const NUM_SIMD_REG: usize = 16;
    } else if #[cfg(target_feature = "sse2")] {
        const SIMD_WIDTH: usize = 16;
        const NUM_SIMD_REG: usize = 16;
    } else if #[cfg(target_feature = "neon")] {
        const SIMD_WIDTH: usize = 16;
        const NUM_SIMD_REG: usize = 32;
    } else {
        const SIMD_WIDTH: usize = 1;
        const NUM_SIMD_REG: usize = 8;
    }
}

pub type ArchSimd<T: SimdInfo> = Simd<T, { T::LANES }>;

pub trait SimdInfo: SimdElement {
    const LANES: usize;
}

impl<T: SimdElement> SimdInfo for T {
    const LANES: usize = SIMD_WIDTH / std::mem::size_of::<T>();
}
