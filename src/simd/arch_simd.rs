use std::simd::{Simd, SimdElement};
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};
use std::marker::PhantomData;

// Static dispatch for identifying lane sizes and number of simd registers.

#[cfg(target_feature = "avx512f")]
const SIMD_WIDTH: usize = 64;
#[cfg(target_feature = "avx512f")]
const NUM_SIMD_REG: usize = 32;

#[cfg(target_feature = "avx2")]
const SIMD_WIDTH: usize = 32;
#[cfg(target_feature = "avx2")]
const NUM_SIMD_REG: usize = 16;

#[cfg(target_feature = "neon")]
const SIMD_WIDTH: usize = 16;
#[cfg(target_feature = "neon")]
const NUM_SIMD_REG: usize = 32;

#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
const SIMD_WIDTH: usize = 16;
#[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
const NUM_SIMD_REG: usize = 16;

#[cfg(not(any(target_feature = "avx2", target_feature = "sse2")))]
const SIMD_WIDTH: usize = 1;
#[cfg(not(any(target_feature = "avx2", target_feature = "sse2")))]
const NUM_SIMD_REG: usize = 8;

pub type ArchSimd<T: SimdInfo> = Simd<T, { T::LANES }>;

pub trait SimdInfo: SimdElement {
    const LANES: usize;
}

impl<T: SimdElement> SimdInfo for T {
    const LANES: usize = SIMD_WIDTH / std::mem::size_of::<T>();
}
