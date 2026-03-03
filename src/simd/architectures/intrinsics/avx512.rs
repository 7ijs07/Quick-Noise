    use std::arch::x86_64::*;
    use crate::simd::architectures::arch_impl::*;
    use std::mem::{transmute, transmute_copy};
    use crate::simd::architectures::macros::*;

    #[derive(Copy, Clone)]
    #[repr(transparent)]
    pub struct Avx512(pub __m512i);
    impl SimdArch for Avx512 {
    }

    #[derive(Copy, Clone)]
    pub struct Avx512Mask(pub __mmask64);
    impl MaskArch for Avx512Mask {}



    impl SimdAddImpl for Avx512 {
        fn f64_add(self, rhs: Self) -> Self { self_from_op!(_mm512_add_pd, self, rhs) }
        fn f32_add(self, rhs: Self) -> Self { self_from_op!(_mm512_add_ps, self, rhs) }
        fn i64_add(self, rhs: Self) -> Self { self_from_op!(_mm512_add_epi64, self, rhs) }
        fn i32_add(self, rhs: Self) -> Self { self_from_op!(_mm512_add_epi32, self, rhs) }
        fn i16_add(self, rhs: Self) -> Self { self_from_op!(_mm512_add_epi16, self, rhs) }
        fn i8_add(self, rhs: Self) -> Self { self_from_op!(_mm512_add_epi8, self, rhs) }
    }

    impl SimdSubImpl for Avx512 {
        fn f64_sub(self, rhs: Self) -> Self { self_from_op!(_mm512_sub_pd, self, rhs) }
        fn f32_sub(self, rhs: Self) -> Self { self_from_op!(_mm512_sub_ps, self, rhs) }
        fn i64_sub(self, rhs: Self) -> Self { self_from_op!(_mm512_sub_epi64, self, rhs) }
        fn i32_sub(self, rhs: Self) -> Self { self_from_op!(_mm512_sub_epi32, self, rhs) }
        fn i16_sub(self, rhs: Self) -> Self { self_from_op!(_mm512_sub_epi16, self, rhs) }
        fn i8_sub(self, rhs: Self) -> Self { self_from_op!(_mm512_sub_epi8, self, rhs) }
    }

    impl SimdMulImpl for Avx512 {
        fn f64_mul(self, rhs: Self) -> Self { self_from_op!(_mm512_mul_pd, self, rhs) }
        fn f32_mul(self, rhs: Self) -> Self { self_from_op!(_mm512_mul_ps, self, rhs) }
        fn i32_mul(self, rhs: Self) -> Self { self_from_op!(_mm512_mullo_epi32, self, rhs) }
        fn i16_mul(self, rhs: Self) -> Self { self_from_op!(_mm512_mullo_epi16, self, rhs) }
    }

    impl SimdDivImpl for Avx512 {
        fn f64_div(self, rhs: Self) -> Self { self_from_op!(_mm512_div_pd, self, rhs) }
        fn f32_div(self, rhs: Self) -> Self { self_from_op!(_mm512_div_ps, self, rhs) }
    }

    impl SimdBitwiseImpl for Avx512 {
        fn and(self, rhs: Self) -> Self { self_from_op!(_mm512_and_si512, self, rhs) }
        fn or(self, rhs: Self) -> Self { self_from_op!(_mm512_or_si512, self, rhs) }
        fn xor(self, rhs: Self) -> Self { self_from_op!(_mm512_xor_si512, self, rhs) }
        fn not(self) -> Self { Self(self.xor(Self::splat_32(!0)).0) }
        fn and_not(self, rhs: Self) -> Self { self_from_op!(_mm512_andnot_si512, self, rhs) }
    }

    impl SimdShiftImpl for Avx512 {
        fn sllv_64(self, rhs: Self) -> Self { self_from_op!(_mm512_sllv_epi64, self, rhs) }
        fn srlv_64(self, rhs: Self) -> Self { self_from_op!(_mm512_srlv_epi64, self, rhs) }
        fn srav_64(self, rhs: Self) -> Self { self_from_op!(_mm512_srav_epi64, self, rhs) }
        fn sllv_32(self, rhs: Self) -> Self { self_from_op!(_mm512_sllv_epi32, self, rhs) }
        fn srlv_32(self, rhs: Self) -> Self { self_from_op!(_mm512_srlv_epi32, self, rhs) }
        fn srav_32(self, rhs: Self) -> Self { self_from_op!(_mm512_srav_epi32, self, rhs) }
        fn sllv_16(self, rhs: Self) -> Self { self_from_op!(_mm512_sllv_epi16, self, rhs) }
        fn srlv_16(self, rhs: Self) -> Self { self_from_op!(_mm512_srlv_epi16, self, rhs) }
        fn srav_16(self, rhs: Self) -> Self { self_from_op!(_mm512_srav_epi16, self, rhs) }
    }

    impl SimdLoadImpl for Avx512 {
        type MaskType = Avx512Mask;
        fn load_aligned<T>(ptr: *const T) -> Self { self_from_op!(_mm512_load_si512, ptr) }
        fn load_unaligned<T>(ptr: *const T) -> Self { self_from_op!(_mm512_loadu_si512, ptr) }
        fn masked_load_64<T>(ptr: *const T, mask: Self::MaskType) -> Self { self_from_op!(_mm512_mask_load_epi64, Self::zero(), ptr, mask) }
        fn masked_load_32<T>(ptr: *const T, mask: Self::MaskType) -> Self { self_from_op!(_mm512_mask_load_epi32, Self::zero(), ptr, mask) }
    }

    impl SimdStoreImpl for Avx512 {
        type MaskType = Avx512Mask;
        fn store_aligned<T>(self, ptr: *mut T) { execute_intrinsic!(_mm512_store_si512, ptr, self); }
        fn store_unaligned<T>(self, ptr: *mut T) { execute_intrinsic!(_mm512_storeu_si512, ptr, self); }
        fn masked_store_64<T>(self, ptr: *mut T, mask: Self::MaskType) { execute_intrinsic!(_mm512_mask_store_epi64, ptr, mask, self); }
        fn masked_store_32<T>(self, ptr: *mut T, mask: Self::MaskType) { execute_intrinsic!(_mm512_mask_store_epi32, ptr, mask, self); }
    }

    impl SimdZeroImpl for Avx512 {
        fn zero() -> Self { self_from_op!(_mm512_setzero_si512,) }
    }

    impl SimdFloatCastsImpl for Avx512 {
        fn float_to_int_trunc(self) -> Self { self_from_op!(_mm512_cvttps_epi32, self) }
        fn float_to_int_round(self) -> Self { self_from_op!(_mm512_cvtps_epi32, self) }
    }

    impl SimdIntCastsImpl for Avx512 {
        fn int_to_float(self) -> Self { self_from_op!(_mm512_cvtepi32_ps, self) }
    }

    impl SimdPermuteImpl for Avx512 {
        // type BlockVec = Sse;
        fn permute_32(self, rhs: Self) -> Self { self_from_op!(_mm512_permutexvar_epi32, self, rhs) }
        fn permute_8(self, rhs: Self) -> Self { self_from_op!(_mm512_shuffle_epi8, self, rhs) }
    }

    impl SimdVariableBlendImpl for Avx512 {
        fn vblend_64(self, other: Self, mask: Self) -> Self { self_from_op!(_mm512_mask_blend_pd, mask, self, other) }
        fn vblend_32(self, other: Self, mask: Self) -> Self { self_from_op!(_mm512_mask_blend_ps, mask, self, other) }
        fn vblend_8(self, other: Self, mask: Self) -> Self { self_from_op!(_mm512_mask_blend_epi8, mask, self, other) }
    }

    impl SimdMulAddImpl for Avx512 {
        fn mul_add_f64(self, mult: Self, add: Self) -> Self { self_from_op!(_mm512_fmadd_pd, self, mult, add) }
        fn mul_sub_f64(self, mult: Self, sub: Self) -> Self { self_from_op!(_mm512_fmsub_pd, self, mult, sub) }
        fn negated_mul_add_f64(self, mult: Self, add: Self) -> Self { self_from_op!(_mm512_fnmadd_pd, self, mult, add) }
        fn negated_mul_sub_f64(self, mult: Self, sub: Self) -> Self { self_from_op!(_mm512_fnmsub_pd, self, mult, sub) }
        fn mul_add_f32(self, mult: Self, add: Self) -> Self { self_from_op!(_mm512_fmadd_ps, self, mult, add) }
        fn mul_sub_f32(self, mult: Self, sub: Self) -> Self { self_from_op!(_mm512_fmsub_ps, self, mult, sub) }
        fn negated_mul_add_f32(self, mult: Self, add: Self) -> Self { self_from_op!(_mm512_fnmadd_ps, self, mult, add) }
        fn negated_mul_sub_f32(self, mult: Self, sub: Self) -> Self { self_from_op!(_mm512_fnmsub_ps, self, mult, sub) }
    }

    impl SimdRoundImpl for Avx512 {
        fn round_f64(self) -> Self { self_from_const_op!(_mm512_roundscale_pd, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC, self) }
        fn round_f32(self) -> Self { self_from_const_op!(_mm512_roundscale_ps, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC, self) }
        fn floor_f64(self) -> Self { self_from_const_op!(_mm512_roundscale_pd, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC, self) }
        fn floor_f32(self) -> Self { self_from_const_op!(_mm512_roundscale_ps, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC, self) }
        fn ceil_f64(self) -> Self { self_from_const_op!(_mm512_roundscale_pd, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC, self) }
        fn ceil_f32(self) -> Self { self_from_const_op!(_mm512_roundscale_ps, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC, self) }
    }

    impl SimdOrdImpl for Avx512 {
        type MaskType = Avx512Mask;
        fn cmp_f64_eq(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_const_intrinsic!(_mm512_cmp_pd_mask, _CMP_EQ_OQ, self, rhs) as u64) }
        fn cmp_f64_lt(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_const_intrinsic!(_mm512_cmp_pd_mask, _CMP_LT_OQ, self, rhs) as u64) }
        fn cmp_f64_le(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_const_intrinsic!(_mm512_cmp_pd_mask, _CMP_LE_OQ, self, rhs) as u64) }
        fn cmp_f64_gt(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_const_intrinsic!(_mm512_cmp_pd_mask, _CMP_GT_OQ, self, rhs) as u64) }
        fn cmp_f64_ge(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_const_intrinsic!(_mm512_cmp_pd_mask, _CMP_GE_OQ, self, rhs) as u64) }
        fn cmp_f64_neq(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_const_intrinsic!(_mm512_cmp_pd_mask, _CMP_NEQ_OQ, self, rhs) as u64) }
        fn cmp_f32_eq(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_const_intrinsic!(_mm512_cmp_ps_mask, _CMP_EQ_OQ, self, rhs) as u64) }
        fn cmp_f32_lt(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_const_intrinsic!(_mm512_cmp_ps_mask, _CMP_LT_OQ, self, rhs) as u64) }
        fn cmp_f32_le(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_const_intrinsic!(_mm512_cmp_ps_mask, _CMP_LE_OQ, self, rhs) as u64) }
        fn cmp_f32_gt(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_const_intrinsic!(_mm512_cmp_ps_mask, _CMP_GT_OQ, self, rhs) as u64) }
        fn cmp_f32_ge(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_const_intrinsic!(_mm512_cmp_ps_mask, _CMP_GE_OQ, self, rhs) as u64) }
        fn cmp_f32_neq(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_const_intrinsic!(_mm512_cmp_ps_mask, _CMP_NEQ_OQ, self, rhs) as u64) }
        fn cmp_i64_eq(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_intrinsic!(_mm512_cmpeq_epi64_mask, self, rhs) as u64) }
        fn cmp_i64_gt(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_intrinsic!(_mm512_cmpgt_epi64_mask, self, rhs) as u64) }
        fn cmp_i32_eq(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_intrinsic!(_mm512_cmpeq_epi32_mask, self, rhs) as u64) }
        fn cmp_i32_gt(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_intrinsic!(_mm512_cmpgt_epi32_mask, self, rhs) as u64) }
        fn cmp_i16_eq(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_intrinsic!(_mm512_cmpeq_epi16_mask, self, rhs) as u64) }
        fn cmp_i16_gt(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_intrinsic!(_mm512_cmpgt_epi16_mask, self, rhs) as u64) }
        fn cmp_i8_eq(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_intrinsic!(_mm512_cmpeq_epi8_mask, self, rhs) as u64) }
        fn cmp_i8_gt(self, rhs: Self) -> Self::MaskType { Avx512Mask(execute_intrinsic!(_mm512_cmpgt_epi8_mask, self, rhs) as u64) }
    }

    // TODO: Make a custom trait for handling this transmutation into i*.
    impl SimdSplatImpl for Avx512 {
        fn splat_64<T>(val: T) -> Self { self_from_op!(_mm512_set1_epi64, val) }
        fn splat_32<T>(val: T) -> Self { self_from_op!(_mm512_set1_epi32, val) }
        fn splat_16<T>(val: T) -> Self { self_from_op!(_mm512_set1_epi16, val) }
        fn splat_8<T>(val: T) -> Self { self_from_op!(_mm512_set1_epi8, val) }
    }

    impl SimdBitwiseImpl for Avx512Mask {
        fn and(self, rhs: Self) -> Self { self_from_op!(_kand_mask64, self, rhs) }
        fn or(self, rhs: Self) -> Self { self_from_op!(_kor_mask64, self, rhs) }
        fn xor(self, rhs: Self) -> Self { self_from_op!(_kxor_mask64, self, rhs) }
        fn not(self) -> Self { self_from_op!(_knot_mask64, self) }
        fn and_not(self, rhs: Self) -> Self { self_from_op!(_kandn_mask64, self, rhs) }
    }
