use crate::simd::architectures::arch_impl::SimdFamily;
use crate::simd::architectures::intrinsics::avx2::Avx2;
use crate::simd::architectures::intrinsics::avx512::{Avx512, Avx512Mask};
use crate::simd::architectures::intrinsics::sse::Sse;
use std::fmt::Debug;
use crate::simd::traits::Array;

#[derive(Copy, Clone)]
pub struct SseFamily;
impl SimdFamily for SseFamily {
    const SIMD_WIDTH: usize = 16;
    type Vec = Sse;
    type Mask = Sse;

    type Array64<T: Debug + Copy> = [T; 2];
    type Array32<T: Debug + Copy> = [T; 4];
    type Array16<T: Debug + Copy> = [T; 8];
    type Array8<T: Debug + Copy> = [T; 16];
}

#[derive(Copy, Clone)]
pub struct Avx2Family;
impl SimdFamily for Avx2Family {
    const SIMD_WIDTH: usize = 32;
    type Vec = Avx2;
    type Mask = Avx2;

    type Array64<T: Debug + Copy> = [T; 4];
    type Array32<T: Debug + Copy> = [T; 8];
    type Array16<T: Debug + Copy> = [T; 16];
    type Array8<T: Debug + Copy> = [T; 32];
}

#[derive(Copy, Clone)]
pub struct Avx512Family;
impl SimdFamily for Avx512Family {
    const SIMD_WIDTH: usize = 64;
    type Vec = Avx512;
    type Mask = Avx512Mask;

    type Array64<T: Debug + Copy> = [T; 8];
    type Array32<T: Debug + Copy> = [T; 16];
    type Array16<T: Debug + Copy> = [T; 32];
    type Array8<T: Debug + Copy> = [T; 64];
}
