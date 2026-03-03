use std::arch::x86_64::*;
use crate::simd::simd_vec::SimdTag;

#[derive(Clone, Copy)]
pub struct F64AVX2(pub __m256d);
#[derive(Clone, Copy)]
pub struct F32AVX2(pub __m256);
#[derive(Clone, Copy)]
pub struct I64AVX2(pub __m256i);
#[derive(Clone, Copy)]
pub struct U64AVX2(pub __m256i);
#[derive(Clone, Copy)]
pub struct I32AVX2(pub __m256i);
#[derive(Clone, Copy)]
pub struct U32AVX2(pub __m256i);
#[derive(Clone, Copy)]
pub struct I16AVX2(pub __m256i);
#[derive(Clone, Copy)]
pub struct U16AVX2(pub __m256i);
#[derive(Clone, Copy)]
pub struct I8AVX2(pub __m256i);
#[derive(Clone, Copy)]
pub struct U8AVX2(pub __m256i);

pub struct Mask64AVX2(pub __m256i);
pub struct Mask32AVX2(pub __m256i);
pub struct Mask16AVX2(pub __m256i);
pub struct Mask8AVX2(pub __m256i);

impl SimdTag for f64 {
    type Vec = F64AVX2;
    type Mask = Mask64AVX2;
    const LANES: usize = 4;
    const SIMD_WIDTH: usize = 32;
}

impl SimdTag for f32 {
    type Vec = F32AVX2;
    type Mask = Mask32AVX2;
    const LANES: usize = 8;
    const SIMD_WIDTH: usize = 32;
}

impl SimdTag for i64 {
    type Vec = I64AVX2;
    type Mask = Mask64AVX2;
    const LANES: usize = 4;
    const SIMD_WIDTH: usize = 32;
}

impl SimdTag for u64 {
    type Vec = U64AVX2;
    type Mask = Mask64AVX2;
    const LANES: usize = 4;
    const SIMD_WIDTH: usize = 32;
}

impl SimdTag for i32 {
    type Vec = I32AVX2;
    type Mask = Mask32AVX2;
    const LANES: usize = 8;
    const SIMD_WIDTH: usize = 32;
}

impl SimdTag for u32 {
    type Vec = U32AVX2;
    type Mask = Mask32AVX2;
    const LANES: usize = 8;
    const SIMD_WIDTH: usize = 32;
}

impl SimdTag for i16 {
    type Vec = I16AVX2;
    type Mask = Mask16AVX2;
    const LANES: usize = 16;
    const SIMD_WIDTH: usize = 32;
}

impl SimdTag for u16 {
    type Vec = U16AVX2;
    type Mask = Mask16AVX2;
    const LANES: usize = 16;
    const SIMD_WIDTH: usize = 32;
}

impl SimdTag for i8 {
    type Vec = I8AVX2;
    type Mask = Mask8AVX2;
    const LANES: usize = 32;
    const SIMD_WIDTH: usize = 32;
}

impl SimdTag for u8 {
    type Vec = U8AVX2;
    type Mask = Mask8AVX2;
    const LANES: usize = 32;
    const SIMD_WIDTH: usize = 32;
}
