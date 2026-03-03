use std::ops::*;
use std::arch::x86_64::*;
use crate::simd::avx2::tags::*;
use crate::simd::traits::*;

// f64.
add_simple_trait_op_avx2!(F64AVX2, Add, add, _mm256_add_pd);
add_simple_trait_op_avx2!(F64AVX2, Sub, sub, _mm256_sub_pd);
add_simple_trait_op_avx2!(F64AVX2, Mul, mul, _mm256_mul_pd);
add_simple_trait_op_avx2!(F64AVX2, Div, div, _mm256_div_pd);
add_simple_assign_trait_op_avx2!(F64AVX2, AddAssign, add_assign, _mm256_add_pd);
add_simple_assign_trait_op_avx2!(F64AVX2, SubAssign, sub_assign, _mm256_sub_pd);
add_simple_assign_trait_op_avx2!(F64AVX2, MulAssign, mul_assign, _mm256_mul_pd);
add_simple_assign_trait_op_avx2!(F64AVX2, DivAssign, div_assign, _mm256_div_pd);
add_load_ops_avx2!(F64AVX2, f64, f64, _mm256_load_pd, _mm256_loadu_pd);
add_store_ops_avx2!(F64AVX2, f64, f64, _mm256_store_pd, _mm256_storeu_pd);
add_float_equality_ops_avx2!(F64AVX2, Mask64AVX2, _mm256_cmp_pd);
add_splat_avx2!(F64AVX2, f64, _mm256_set1_pd);
add_masked_load_avx2!(F64AVX2, Mask64AVX2, f64, f64, _mm256_maskload_pd);
add_masked_store_avx2!(F64AVX2, Mask64AVX2, f64, f64, _mm256_maskstore_pd);
add_blend!(F64AVX2, Mask64AVX2, _mm256_blendv_pd);
add_simple_trait_op_3_args_avx2!(F64AVX2, SimdMulAdd, mul_add, _mm256_fmadd_pd);
add_simple_trait_op_3_args_avx2!(F64AVX2, SimdMulSub, mul_sub, _mm256_fmsub_pd);
add_simple_trait_op_3_args_avx2!(F64AVX2, SimdNegMulAdd, negated_mul_add, _mm256_fnmadd_pd);
add_simple_trait_op_3_args_avx2!(F64AVX2, SimdNegMulSub, negated_mul_sub, _mm256_fnmsub_pd);
add_round_op!(F64AVX2, SimdFloor, floor, _mm256_round_pd, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
add_round_op!(F64AVX2, SimdCeil, ceil, _mm256_round_pd, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
add_round_op!(F64AVX2, SimdRound, round, _mm256_round_pd, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
add_const_extract_op_avx2!(F64AVX2, i64, f64, f64, _mm256_extract_epi64);
add_zero_op_avx2!(F64AVX2);
// add_float_cast_op_avx2!(F64AVX2, I64AVX2, _mm256_cvttpd_epi64, _mm256_cvtpd_epi64);

// f32.
add_simple_trait_op_avx2!(F32AVX2, Add, add, _mm256_add_ps);
add_simple_trait_op_avx2!(F32AVX2, Sub, sub, _mm256_sub_ps);
add_simple_trait_op_avx2!(F32AVX2, Mul, mul, _mm256_mul_ps);
add_simple_trait_op_avx2!(F32AVX2, Div, div, _mm256_div_ps);
add_simple_assign_trait_op_avx2!(F32AVX2, AddAssign, add_assign, _mm256_add_ps);
add_simple_assign_trait_op_avx2!(F32AVX2, SubAssign, sub_assign, _mm256_sub_ps);
add_simple_assign_trait_op_avx2!(F32AVX2, MulAssign, mul_assign, _mm256_mul_ps);
add_simple_assign_trait_op_avx2!(F32AVX2, DivAssign, div_assign, _mm256_div_ps);
add_load_ops_avx2!(F32AVX2, f32, f32, _mm256_load_ps, _mm256_loadu_ps);
add_store_ops_avx2!(F32AVX2, f32, f32, _mm256_store_ps, _mm256_storeu_ps);
add_float_equality_ops_avx2!(F32AVX2, Mask32AVX2, _mm256_cmp_ps);
add_splat_avx2!(F32AVX2, f32, _mm256_set1_ps);
add_masked_load_avx2!(F32AVX2, Mask32AVX2, f32, f32, _mm256_maskload_ps);
add_masked_store_avx2!(F32AVX2, Mask32AVX2, f32, f32, _mm256_maskstore_ps);
add_runtime_permute!(F32AVX2, U32AVX2, u32, _mm256_permutevar8x32_ps);
add_blend!(F32AVX2, Mask32AVX2, _mm256_blendv_ps);
add_simple_trait_op_3_args_avx2!(F32AVX2, SimdMulAdd, mul_add, _mm256_fmadd_ps);
add_simple_trait_op_3_args_avx2!(F32AVX2, SimdMulSub, mul_sub, _mm256_fmsub_ps);
add_simple_trait_op_3_args_avx2!(F32AVX2, SimdNegMulAdd, negated_mul_add, _mm256_fnmadd_ps);
add_simple_trait_op_3_args_avx2!(F32AVX2, SimdNegMulSub, negated_mul_sub, _mm256_fnmsub_ps);
add_round_op!(F32AVX2, SimdFloor, floor, _mm256_round_ps, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
add_round_op!(F32AVX2, SimdCeil, ceil, _mm256_round_ps, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
add_round_op!(F32AVX2, SimdRound, round, _mm256_round_ps, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
add_const_extract_op_avx2!(F32AVX2, i32, f32, f32, _mm256_extract_epi32);
add_zero_op_avx2!(F32AVX2);
add_float_cast_op_avx2!(F32AVX2, I32AVX2, i32, _mm256_cvttps_epi32, _mm256_cvtps_epi32);

// i64.
add_integer_ops_avx2!(I64AVX2, i64);
add_simple_trait_op_avx2!(I64AVX2, Add, add, _mm256_add_epi64);
add_simple_trait_op_avx2!(I64AVX2, Sub, sub, _mm256_sub_epi64);
add_simple_assign_trait_op_avx2!(I64AVX2, AddAssign, add_assign, _mm256_add_epi64);
add_simple_assign_trait_op_avx2!(I64AVX2, SubAssign, sub_assign, _mm256_sub_epi64);
add_shift_op_avx2!(I64AVX2, Shl, shl, i64, _mm256_sllv_epi64, _mm256_set1_epi64x);
// add_const_shift_op_avx2!(I64AVX2, Shl, shl, i64, _mm256_slli_epi64);
add_shift_op_avx2!(I64AVX2, Shr, shr, i64, _mm256_srav_epi64, _mm256_set1_epi64x);
add_shift_assign_op_avx2!(I64AVX2, ShlAssign, shl_assign, i64, _mm256_sllv_epi64, _mm256_set1_epi64x);
add_shift_assign_op_avx2!(I64AVX2, ShrAssign, shr_assign, i64, _mm256_srav_epi64, _mm256_set1_epi64x);
add_const_extract_op_avx2!(I64AVX2, i64, i64, i64, _mm256_extract_epi64);
add_integer_equality_op_avx2!(I64AVX2, Mask64AVX2, compare_eq, _mm256_cmpeq_epi64);
add_integer_equality_op_avx2!(I64AVX2, Mask64AVX2, compare_gt, _mm256_cmpgt_epi64);
add_splat_avx2!(I64AVX2, i64, _mm256_set1_epi64x);
add_masked_load_avx2!(I64AVX2, Mask64AVX2, i64, i64, _mm256_maskload_epi64);
add_masked_store_avx2!(I64AVX2, Mask64AVX2, i64, i64, _mm256_maskstore_epi64);
add_blend!(I64AVX2, Mask64AVX2, _mm256_blendv_pd);

// u64.
add_integer_ops_avx2!(U64AVX2, u64);
add_simple_trait_op_avx2!(U64AVX2, Add, add, _mm256_add_epi64);
add_simple_trait_op_avx2!(U64AVX2, Sub, sub, _mm256_sub_epi64);
add_simple_assign_trait_op_avx2!(U64AVX2, AddAssign, add_assign, _mm256_add_epi64);
add_simple_assign_trait_op_avx2!(U64AVX2, SubAssign, sub_assign, _mm256_sub_epi64);
add_shift_op_avx2!(U64AVX2, Shl, shl, i64, _mm256_sllv_epi64, _mm256_set1_epi64x);
add_shift_op_avx2!(U64AVX2, Shr, shr, i64, _mm256_srlv_epi64, _mm256_set1_epi64x);
add_shift_assign_op_avx2!(U64AVX2, ShlAssign, shl_assign, i64, _mm256_sllv_epi64, _mm256_set1_epi64x);
add_shift_assign_op_avx2!(U64AVX2, ShrAssign, shr_assign, i64, _mm256_srlv_epi64, _mm256_set1_epi64x);
add_const_extract_op_avx2!(U64AVX2, i64, u64, u64, _mm256_extract_epi64);
add_integer_equality_op_avx2!(U64AVX2, Mask64AVX2, compare_eq, _mm256_cmpeq_epi64);
add_integer_equality_op_avx2!(U64AVX2, Mask64AVX2, compare_gt, _mm256_cmpgt_epi64);
add_splat_avx2!(U64AVX2, u64, _mm256_set1_epi64x);
add_masked_load_avx2!(U64AVX2, Mask64AVX2, u64, i64, _mm256_maskload_epi64);
add_masked_store_avx2!(U64AVX2, Mask64AVX2, u64, i64, _mm256_maskstore_epi64);
add_blend!(U64AVX2, Mask64AVX2, _mm256_blendv_pd);

// i32.
add_integer_ops_avx2!(I32AVX2, i32);
add_simple_trait_op_avx2!(I32AVX2, Add, add, _mm256_add_epi32);
add_simple_trait_op_avx2!(I32AVX2, Sub, sub, _mm256_sub_epi32);
add_simple_trait_op_avx2!(I32AVX2, Mul, mul, _mm256_mullo_epi32);
add_simple_assign_trait_op_avx2!(I32AVX2, AddAssign, add_assign, _mm256_add_epi32);
add_simple_assign_trait_op_avx2!(I32AVX2, SubAssign, sub_assign, _mm256_sub_epi32);
add_simple_assign_trait_op_avx2!(I32AVX2, MulAssign, mul_assign, _mm256_mullo_epi32);
add_shift_op_avx2!(I32AVX2, Shl, shl, i32, _mm256_sllv_epi32, _mm256_set1_epi32);
add_shift_op_avx2!(I32AVX2, Shr, shr, i32, _mm256_srav_epi32, _mm256_set1_epi32);
add_shift_assign_op_avx2!(I32AVX2, ShlAssign, shl_assign, i32, _mm256_sllv_epi32, _mm256_set1_epi32);
add_shift_assign_op_avx2!(I32AVX2, ShrAssign, shr_assign, i32, _mm256_srav_epi32, _mm256_set1_epi32);
add_const_extract_op_avx2!(I32AVX2, i32, i32, i32, _mm256_extract_epi32);
add_integer_equality_op_avx2!(I32AVX2, Mask32AVX2, compare_eq, _mm256_cmpeq_epi32);
add_integer_equality_op_avx2!(I32AVX2, Mask32AVX2, compare_gt, _mm256_cmpgt_epi32);
add_splat_avx2!(I32AVX2, i32, _mm256_set1_epi32);
add_masked_load_avx2!(I32AVX2, Mask32AVX2, i32, i32, _mm256_maskload_epi32);
add_masked_store_avx2!(I32AVX2, Mask32AVX2, i32, i32, _mm256_maskstore_epi32);
add_runtime_permute!(I32AVX2, U32AVX2, u32, _mm256_permutevar8x32_epi32);
add_blend!(I32AVX2, Mask32AVX2, _mm256_blendv_ps);
add_integer_cast_op_avx2!(I32AVX2, F32AVX2, f32, _mm256_cvtepi32_ps);

// u32.
add_integer_ops_avx2!(U32AVX2, u32);
add_simple_trait_op_avx2!(U32AVX2, Add, add, _mm256_add_epi32);
add_simple_trait_op_avx2!(U32AVX2, Sub, sub, _mm256_sub_epi32);
add_simple_trait_op_avx2!(U32AVX2, Mul, mul, _mm256_mullo_epi32);
add_simple_assign_trait_op_avx2!(U32AVX2, AddAssign, add_assign, _mm256_add_epi32);
add_simple_assign_trait_op_avx2!(U32AVX2, SubAssign, sub_assign, _mm256_sub_epi32);
add_simple_assign_trait_op_avx2!(U32AVX2, MulAssign, mul_assign, _mm256_mullo_epi32);
add_shift_op_avx2!(U32AVX2, Shl, shl, i32, _mm256_sllv_epi32, _mm256_set1_epi32);
add_shift_op_avx2!(U32AVX2, Shr, shr, i32, _mm256_srlv_epi32, _mm256_set1_epi32);
add_shift_assign_op_avx2!(U32AVX2, ShlAssign, shl_assign, i32, _mm256_sllv_epi32, _mm256_set1_epi32);
add_shift_assign_op_avx2!(U32AVX2, ShrAssign, shr_assign, i32, _mm256_srlv_epi32, _mm256_set1_epi32);
add_const_extract_op_avx2!(U32AVX2, i32, u32, u32, _mm256_extract_epi32);
add_integer_equality_op_avx2!(U32AVX2, Mask32AVX2, compare_eq, _mm256_cmpeq_epi32);
add_integer_equality_op_avx2!(U32AVX2, Mask32AVX2, compare_gt, _mm256_cmpgt_epi32);
add_splat_avx2!(U32AVX2, u32, _mm256_set1_epi32);
add_masked_load_avx2!(U32AVX2, Mask32AVX2, u32, i32, _mm256_maskload_epi32);
add_masked_store_avx2!(U32AVX2, Mask32AVX2, u32, i32, _mm256_maskstore_epi32);
add_runtime_permute!(U32AVX2, U32AVX2, u32, _mm256_permutevar8x32_epi32);
add_blend!(U32AVX2, Mask32AVX2, _mm256_blendv_ps);

// i16.
add_integer_ops_avx2!(I16AVX2, i16);
add_simple_trait_op_avx2!(I16AVX2, Add, add, _mm256_add_epi16);
add_simple_trait_op_avx2!(I16AVX2, Sub, sub, _mm256_sub_epi16);
add_simple_trait_op_avx2!(I16AVX2, Mul, mul, _mm256_mullo_epi16);
add_simple_assign_trait_op_avx2!(I16AVX2, AddAssign, add_assign, _mm256_add_epi16);
add_simple_assign_trait_op_avx2!(I16AVX2, SubAssign, sub_assign, _mm256_sub_epi16);
add_simple_assign_trait_op_avx2!(I16AVX2, MulAssign, mul_assign, _mm256_mullo_epi16);
add_shift_op_avx2!(I16AVX2, Shl, shl, i16, _mm256_sllv_epi16, _mm256_set1_epi16);
add_shift_op_avx2!(I16AVX2, Shr, shr, i16, _mm256_srav_epi16, _mm256_set1_epi16);
add_shift_assign_op_avx2!(I16AVX2, ShlAssign, shl_assign, i16, _mm256_sllv_epi16, _mm256_set1_epi16);
add_shift_assign_op_avx2!(I16AVX2, ShrAssign, shr_assign, i16, _mm256_srav_epi16, _mm256_set1_epi16);
add_const_extract_op_avx2!(I16AVX2, i32, i32, i16, _mm256_extract_epi16);
add_integer_equality_op_avx2!(I16AVX2, Mask16AVX2, compare_eq, _mm256_cmpeq_epi16);
add_integer_equality_op_avx2!(I16AVX2, Mask16AVX2, compare_gt, _mm256_cmpgt_epi16);
add_splat_avx2!(I16AVX2, i16, _mm256_set1_epi16);

// u16.
add_integer_ops_avx2!(U16AVX2, u16);
add_simple_trait_op_avx2!(U16AVX2, Add, add, _mm256_add_epi16);
add_simple_trait_op_avx2!(U16AVX2, Sub, sub, _mm256_sub_epi16);
add_simple_trait_op_avx2!(U16AVX2, Mul, mul, _mm256_mullo_epi16);
add_simple_assign_trait_op_avx2!(U16AVX2, AddAssign, add_assign, _mm256_add_epi16);
add_simple_assign_trait_op_avx2!(U16AVX2, SubAssign, sub_assign, _mm256_sub_epi16);
add_simple_assign_trait_op_avx2!(U16AVX2, MulAssign, mul_assign, _mm256_mullo_epi16);
add_shift_op_avx2!(U16AVX2, Shl, shl, i16, _mm256_sllv_epi16, _mm256_set1_epi16);
add_shift_op_avx2!(U16AVX2, Shr, shr, i16, _mm256_srlv_epi16, _mm256_set1_epi16);
add_shift_assign_op_avx2!(U16AVX2, ShlAssign, shl_assign, i16, _mm256_sllv_epi16, _mm256_set1_epi16);
add_shift_assign_op_avx2!(U16AVX2, ShrAssign, shr_assign, i16, _mm256_srlv_epi16, _mm256_set1_epi16);
add_const_extract_op_avx2!(U16AVX2, i32, u32, u16, _mm256_extract_epi16);
add_integer_equality_op_avx2!(U16AVX2, Mask16AVX2, compare_eq, _mm256_cmpeq_epi16);
add_integer_equality_op_avx2!(U16AVX2, Mask16AVX2, compare_gt, _mm256_cmpgt_epi16);
add_splat_avx2!(U16AVX2, u16, _mm256_set1_epi16);

// i8.
add_integer_ops_avx2!(I8AVX2, i8);
add_simple_trait_op_avx2!(I8AVX2, Add, add, _mm256_add_epi8);
add_simple_trait_op_avx2!(I8AVX2, Sub, sub, _mm256_sub_epi8);
add_simple_assign_trait_op_avx2!(I8AVX2, AddAssign, add_assign, _mm256_add_epi8);
add_simple_assign_trait_op_avx2!(I8AVX2, SubAssign, sub_assign, _mm256_sub_epi8);
add_const_extract_op_avx2!(I8AVX2, i32, i32, i8, _mm256_extract_epi8);
add_integer_equality_op_avx2!(I8AVX2, Mask8AVX2, compare_eq, _mm256_cmpeq_epi8);
add_integer_equality_op_avx2!(I8AVX2, Mask8AVX2, compare_gt, _mm256_cmpgt_epi8);
add_splat_avx2!(I8AVX2, i8, _mm256_set1_epi8);

// u8.
add_integer_ops_avx2!(U8AVX2, u8);
add_simple_trait_op_avx2!(U8AVX2, Add, add, _mm256_add_epi8);
add_simple_trait_op_avx2!(U8AVX2, Sub, sub, _mm256_sub_epi8);
add_simple_assign_trait_op_avx2!(U8AVX2, AddAssign, add_assign, _mm256_add_epi8);
add_simple_assign_trait_op_avx2!(U8AVX2, SubAssign, sub_assign, _mm256_sub_epi8);
add_const_extract_op_avx2!(U8AVX2, i32, u32, u8, _mm256_extract_epi8);
add_integer_equality_op_avx2!(U8AVX2, Mask8AVX2, compare_eq, _mm256_cmpeq_epi8);
add_integer_equality_op_avx2!(U8AVX2, Mask8AVX2, compare_gt, _mm256_cmpgt_epi8);
add_splat_avx2!(U8AVX2, u8, _mm256_set1_epi8);

// 64-bit Mask.
add_integer_ops_avx2!(Mask64AVX2, i64);
add_integer_equality_op_avx2!(Mask64AVX2, Mask64AVX2, compare_eq, _mm256_cmpeq_epi64);
add_mask_splat_avx2!(Mask64AVX2);

// 32-bit Mask.
add_integer_ops_avx2!(Mask32AVX2, i32);
add_integer_equality_op_avx2!(Mask32AVX2, Mask32AVX2, compare_eq, _mm256_cmpeq_epi32);
add_mask_splat_avx2!(Mask32AVX2);

// 16-bit Mask.
add_integer_ops_avx2!(Mask16AVX2, i16);
add_integer_equality_op_avx2!(Mask16AVX2, Mask16AVX2, compare_eq, _mm256_cmpeq_epi16);
add_mask_splat_avx2!(Mask16AVX2);

// 8-bit Mask.
add_integer_ops_avx2!(Mask8AVX2, i8);
add_integer_equality_op_avx2!(Mask8AVX2, Mask8AVX2, compare_eq, _mm256_cmpeq_epi8);
add_mask_splat_avx2!(Mask8AVX2);
