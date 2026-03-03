use crate::simd::architectures::arch_impl::SimdFamily;
use crate::simd::architectures::intrinsics::avx2::Avx2;
use crate::simd::architectures::intrinsics::avx512::{Avx512, Avx512Mask};
use crate::simd::architectures::intrinsics::sse::Sse;

#[derive(Copy, Clone)]
pub struct SseFamily;
impl SimdFamily for SseFamily {
    const SIMD_WIDTH: usize = 16;
    type Vec = Sse;
    type Mask = Sse;
}

#[derive(Copy, Clone)]
pub struct Avx2Family;
impl SimdFamily for Avx2Family {
    const SIMD_WIDTH: usize = 32;
    type Vec = Avx2;
    type Mask = Avx2;
}

#[derive(Copy, Clone)]
pub struct Avx512Family;
impl SimdFamily for Avx512Family {
    const SIMD_WIDTH: usize = 64;
    type Vec = Avx512;
    type Mask = Avx512Mask;
}
