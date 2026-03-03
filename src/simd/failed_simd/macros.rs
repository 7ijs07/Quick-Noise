#[macro_export]
macro_rules! add_simple_op_avx2 {
    ($struct:ident, $method:ident, $intrinsic:ident) => {
        impl $struct {
            #[inline(always)]
            pub fn $method(self, rhs: Self) -> Self {
                unsafe { $struct($intrinsic(self.0, rhs.0)) }
            }
        }
    };
}

#[macro_export]
macro_rules! add_simple_op_3_args_avx2 {
    ($struct:ident, $method:ident, $intrinsic:ident) => {
        impl $struct {
            #[inline(always)]
            pub fn $method(self, a: Self, b: Self) -> Self {
                unsafe { $struct($intrinsic(self.0, a.0, b.0)) }
            }
        }
    };
}

#[macro_export]
macro_rules! add_simple_trait_op_avx2 {
    ($struct:ident, $trait:ident, $method:ident, $intrinsic:ident) => {
        impl $trait for $struct {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                unsafe { $struct($intrinsic(self.0, rhs.0)) }
            }
        }
    };
}

#[macro_export]
macro_rules! add_simple_trait_op_3_args_avx2 {
    ($struct:ident, $trait:ident, $method:ident, $intrinsic:ident) => {
        impl $trait for $struct {
            type Output = Self;
            #[inline(always)]
            fn $method(self, a: Self, b: Self) -> Self {
                unsafe { $struct($intrinsic(self.0, a.0, b.0)) }
            }
        }
    };
}

#[macro_export]
macro_rules! add_simple_generic_trait_op_avx2 {
    ($struct:ident, $trait:ident, $type:ty, $method:ident, $intrinsic:ident) => {
        impl $trait for $struct {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                unsafe { $struct($intrinsic(self.0, rhs.0)) }
            }
        }
    };
}

#[macro_export]
macro_rules! add_simple_assign_trait_op_avx2 {
    ($struct:ident, $trait:ident, $method:ident, $intrinsic:ident) => {
        impl $trait for $struct {
            #[inline(always)]
            fn $method(&mut self, rhs: Self) {
                unsafe { self.0 = $intrinsic(self.0, rhs.0); }
            }
        }
    };
}

#[macro_export]
macro_rules! add_shift_op_avx2 {
    ($struct:ident, $trait:ident, $method:ident, $shift_type:ident, $shift_intrinsic:ident, $set_intrinsic:ident) => {
        impl $trait<$shift_type> for $struct {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: $shift_type) -> Self {
                unsafe {
                    let shift = $set_intrinsic(rhs);
                    $struct($shift_intrinsic(self.0, shift))
                }
            }
        }
    };
}

// #[macro_export]
// macro_rules! add_const_shift_op_avx2 {
//     ($struct:ident, $trait:ident, $method:ident, $shift_type:ident, $shift_intrinsic:ident) => {
//         impl $trait<$shift_type> for $struct {
//             type Output = Self;
//             #[inline(always)]
//             fn $method(self, rhs: $shift_type) -> Self {
//                 unsafe {
//                     $struct($shift_intrinsic(self.0, shift))
//                 }
//             }
//         }
//     };
// }

#[macro_export]
macro_rules! add_shift_assign_op_avx2 {
    ($struct:ident, $trait:ident, $method:ident, $shift_type:ident, $shift_intrinsic:ident, $set_intrinsic:ident) => {
        impl $trait<$shift_type> for $struct {
            #[inline(always)]
            fn $method(&mut self, rhs: $shift_type) {
                unsafe {
                    let shift = $set_intrinsic(rhs);
                    self.0 = $shift_intrinsic(self.0, shift);
                }
            }
        }
    };
}

#[macro_export]
macro_rules! add_const_extract_op_avx2 {
    ($struct:ident, $src_type:ty, $dst_type:ty, $true_type:ty, $extract_intrinsic:ident) => {
        impl SimdConstExtract<$true_type> for $struct {
            #[allow(unnecessary_transmutes)]
            #[inline(always)]
            fn const_extract<const N: i32>(self) -> $true_type {
                unsafe {
                    std::mem::transmute::<$src_type, $dst_type>(
                        $extract_intrinsic(std::mem::transmute(self.0), N))
                    as $true_type
                }
            }
        }
    }
}

#[macro_export]
macro_rules! add_load_ops_avx2 {
    ($struct:ident, $arg_type:ty, $ptr_type:ty, $aligned_intrinsic:ident, $unaligned_intrinsic:ident) => {
        impl SimdLoad<$arg_type> for $struct {
            #[inline(always)]
            unsafe fn load_aligned(ptr: *const $arg_type) -> Self {
                unsafe { Self($aligned_intrinsic(ptr as *const $ptr_type)) }
            }
            #[inline(always)]
            unsafe fn load_unaligned(ptr: *const $arg_type) -> Self {
                unsafe { Self($unaligned_intrinsic(ptr as *const $ptr_type)) }
            }
        }
    }
}

#[macro_export]
macro_rules! add_masked_load_avx2 {
    ($simd_struct:ident, $mask_struct:ident, $arg_type:ty, $ptr_type:ty, $mask_intrinsic:ident) => {
        impl SimdMaskedLoad<$arg_type> for $simd_struct {
            type Mask = $mask_struct;
            #[inline(always)]
            unsafe fn masked_load(ptr: *const $arg_type, mask: $mask_struct) -> Self {
                unsafe { Self($mask_intrinsic(ptr as *const $ptr_type, mask.0)) }
            }
        }
    }
}

#[macro_export]
macro_rules! add_store_ops_avx2 {
    ($simd_struct:ident, $arg_type:ty, $ptr_type:ty, $aligned_intrinsic:ident, $unaligned_intrinsic:ident) => {
        impl SimdStore<$arg_type> for $simd_struct {
            #[inline(always)]
            unsafe fn store_aligned(self, ptr: *mut $arg_type) {
                unsafe { $aligned_intrinsic(ptr as *mut $ptr_type, self.0); }
            }
            #[inline(always)]
            unsafe fn store_unaligned(self, ptr: *mut $arg_type) {
                unsafe { $unaligned_intrinsic(ptr as *mut $ptr_type, self.0); }
            }
        }
    }
}

#[macro_export]
macro_rules! add_masked_store_avx2 {
    ($simd_struct:ident, $mask_struct:ident, $arg_type:ty, $ptr_type:ty, $mask_intrinsic:ident) => {
        impl SimdMaskedStore<$arg_type> for $simd_struct {
            type Mask = $mask_struct;
            #[inline(always)]
            unsafe fn masked_store(self, ptr: *mut $arg_type, mask: $mask_struct) {
                unsafe { $mask_intrinsic(ptr as *mut $ptr_type, mask.0, self.0); }
            }
        }
    }
}

#[macro_export]
macro_rules! add_integer_equality_op_avx2 {
    ($simd_struct:ident, $mask_struct:ident, $op:ident, $op_intrinsic:ident) => {
        impl $simd_struct {
            #[inline(always)]
            pub fn $op(self, other: Self) -> $mask_struct {
                unsafe { $mask_struct($op_intrinsic(self.0, other.0)) }
            }
        }
    }
}

#[macro_export]
macro_rules! add_float_equality_op_avx2 {
    ($simd_struct:ident, $mask_struct:ident, $cmp_intrinsic:ident, $op:ident, $op_code:ident) => {
        impl $simd_struct {
            #[inline(always)]
            pub fn $op(self, other: Self) -> $mask_struct {
                unsafe { $mask_struct(
                    std::mem::transmute($cmp_intrinsic(self.0, other.0, $op_code))
                ) }
            }
        }
    }
}

#[macro_export]
macro_rules! add_splat_avx2 {
    ($struct:ident, $arg_type:ty, $intrinsic:ident) => {
        impl SimdSplat<$arg_type> for $struct {
            #[allow(unnecessary_transmutes)]
            #[inline(always)]
            fn splat(val: $arg_type) -> Self {
                unsafe { Self($intrinsic(std::mem::transmute(val))) }
            }
        }
    }
}

#[macro_export]
macro_rules! add_mask_splat_avx2 {
    ($struct:ident) => {
        impl $struct {
            #[inline(always)]
            pub fn splat(val: bool) -> Self {
                unsafe { Self(_mm256_set1_epi32(-(val as i32))) }
            }
        }
    }
}

#[macro_export]
macro_rules! add_runtime_permute {
    ($simd_struct:ident, $indices_struct:ident, $indices_type:ty, $intrinsic:ident) => {
        impl SimdRuntimePermute for $simd_struct {
            type Output = Self;
            type Indices = $indices_struct;
            type IndiciesType = $indices_type;
            #[inline(always)]
            fn runtime_permute(self, indices: $indices_struct) -> Self {
                unsafe { Self($intrinsic(self.0, indices.0)) }
            }
        }
    }
}

#[macro_export]
macro_rules! add_blend {
    ($simd_struct:ident, $mask_struct:ident, $intrinsic:ident) => {
        impl $simd_struct {
            #[allow(unnecessary_transmutes)]
            #[inline(always)]
            pub fn blend(self, other: Self, mask: $mask_struct) -> Self {
                unsafe {
                    Self(std::mem::transmute(
                        $intrinsic(
                            std::mem::transmute(self.0),
                            std::mem::transmute(other.0),
                            std::mem::transmute(mask.0)
                        )
                    ))
                }
            }
        }
    }
}

#[macro_export]
macro_rules! add_round_op {
    ($simd_struct:ident, $trait:ident, $op:ident, $intrinsic:ident, $op_code:expr) => {
        impl $trait for $simd_struct {
            type Output = Self;
            fn $op(self) -> Self {
                unsafe { Self($intrinsic(std::mem::transmute(self.0), $op_code)) }
            }
        }
    }
}

#[macro_export]
macro_rules! add_zero_op_avx2 {
    ($simd_struct:ident) => {
        impl SimdZero for $simd_struct {
            type Output = Self;
            fn zero() -> Self {
                unsafe { Self(std::mem::transmute(_mm256_setzero_si256())) }
            }
        }
    }
}

#[macro_export]
macro_rules! add_integer_cast_op_avx2 {
    ($src_struct:ident, $cast_struct:ident, $cast_type:ty, $intrinsic:ident) => {
        impl SimdIntToFloat<$cast_type> for $src_struct {
            #[inline(always)]
            fn cast_float(self) -> $cast_struct {
                unsafe { $cast_struct($intrinsic(self.0)) }
            }
        }
    };
}

#[macro_export]
macro_rules! add_float_cast_op_avx2 {
    ($src_struct:ident, $cast_struct:ident, $cast_type:ty, $trunc_intrinsic:ident, $round_intrinsic:ident) => {
        impl SimdFloatToInt<$cast_type> for $src_struct {
            #[inline(always)]
            fn cast_int_trunc(self) -> $cast_struct {
                unsafe { $cast_struct($trunc_intrinsic(self.0)) }
            }
            #[inline(always)]
            fn cast_int_round(self) -> $cast_struct {
                unsafe { $cast_struct($round_intrinsic(self.0)) }
            }
        }
    };
}

// #[macro_export]
// macro_rules! add_byte_shuffle {
//     ($src_struct:ident, $cast_struct:ident)
// }

#[macro_export]
macro_rules! add_float_equality_ops_avx2 {
    ($simd_struct:ident, $mask_struct:ident, $cmp_intrinsic:ident) => {
        add_float_equality_op_avx2!($simd_struct, $mask_struct, $cmp_intrinsic, compare_eq, _CMP_EQ_OQ);
        add_float_equality_op_avx2!($simd_struct, $mask_struct, $cmp_intrinsic, compare_lt, _CMP_LT_OQ);
        add_float_equality_op_avx2!($simd_struct, $mask_struct, $cmp_intrinsic, compare_le, _CMP_LE_OQ);
        add_float_equality_op_avx2!($simd_struct, $mask_struct, $cmp_intrinsic, compare_gt, _CMP_GT_OQ);
        add_float_equality_op_avx2!($simd_struct, $mask_struct, $cmp_intrinsic, compare_ge, _CMP_GE_OQ);
        add_float_equality_op_avx2!($simd_struct, $mask_struct, $cmp_intrinsic, compare_neq, _CMP_NEQ_OQ);
    }
}

#[macro_export]
macro_rules! add_integer_ops_avx2 {
    ($struct:ident, $type:ty) => {
        add_simple_trait_op_avx2!($struct, BitAnd, bitand, _mm256_and_si256);
        add_simple_trait_op_avx2!($struct, BitOr, bitor, _mm256_or_si256);
        add_simple_trait_op_avx2!($struct, BitXor, bitxor, _mm256_xor_si256);
        add_simple_assign_trait_op_avx2!($struct, BitAndAssign, bitand_assign, _mm256_and_si256);
        add_simple_assign_trait_op_avx2!($struct, BitOrAssign, bitor_assign, _mm256_or_si256);
        add_simple_assign_trait_op_avx2!($struct, BitXorAssign, bitxor_assign, _mm256_xor_si256);
        add_simple_generic_trait_op_avx2!($struct, BitAndNot, $type, andnot, _mm256_andnot_si256);
        // add_simple_op_avx2!($struct, andnot, _mm256_andnot_si256);

        add_load_ops_avx2!($struct, $type, __m256i, _mm256_load_si256, _mm256_loadu_si256);
        add_store_ops_avx2!($struct, $type, __m256i, _mm256_store_si256, _mm256_storeu_si256);
        add_zero_op_avx2!($struct);
    }
}

pub use crate::add_simple_op_avx2;
pub use crate::add_integer_cast_op_avx2;
pub use crate::add_simple_op_3_args_avx2;
pub use crate::add_simple_trait_op_avx2;
pub use crate::add_simple_trait_op_3_args_avx2;
pub use crate::add_simple_assign_trait_op_avx2;
pub use crate::add_shift_op_avx2;
// pub use crate::add_const_shift_op_avx2;
pub use crate::add_shift_assign_op_avx2;
pub use crate::add_integer_ops_avx2;
pub use crate::add_const_extract_op_avx2;
pub use crate::add_load_ops_avx2;
pub use crate::add_store_ops_avx2;
pub use crate::add_integer_equality_op_avx2;
pub use crate::add_float_equality_op_avx2;
pub use crate::add_float_equality_ops_avx2;
pub use crate::add_splat_avx2;
pub use crate::add_mask_splat_avx2;
pub use crate::add_masked_load_avx2;
pub use crate::add_masked_store_avx2;
pub use crate::add_runtime_permute;
pub use crate::add_blend;
pub use crate::add_round_op;
