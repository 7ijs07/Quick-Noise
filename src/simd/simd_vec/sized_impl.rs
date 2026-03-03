use crate::simd::architectures::arch_impl::*;
use crate::simd::traits::*;
use crate::simd::simd_vec::core::SimdVec;
use crate::simd::simd_mask::core::SimdMask;
use crate::simd::simd_traits::*;

impl<T: SimdElement, F: SimdFamily> SimdVec<T, F> {
    #[inline(always)]
    pub fn splat(val: T) -> Self {
        Self::new (
            match T::BIT_SIZE {
                BitSize::Size64 => F::Vec::splat_64(val),
                BitSize::Size32 => F::Vec::splat_32(val),
                BitSize::Size16 => F::Vec::splat_16(val),
                BitSize::Size8 => F::Vec::splat_8(val),
            }
        )
    }
}

impl<T: SimdWideType, F: SimdFamily> SimdMaskedLoad<T, F> for SimdVec<T, F> {
    #[inline(always)]
    fn masked_load(slice: &[T], mask: SimdMask<T, F>) -> Self {
        Self::new(
            match T::BIT_SIZE {
                BitSize::Size64 => F::Vec::masked_load_64(slice.as_ptr(), mask.data),
                BitSize::Size32 => F::Vec::masked_load_32(slice.as_ptr(), mask.data),
                _ => unreachable!()
            }
        )
    }
}

impl<T: SimdWideType, F: SimdFamily> SimdMaskedStore<T, F> for SimdVec<T, F> {
    #[inline(always)]
    fn masked_store(self, slice: &mut [T], mask: SimdMask<T, F>) {
        match T::BIT_SIZE {
            BitSize::Size64 => F::Vec::masked_store_64(self.data, slice.as_mut_ptr(), mask.data),
            BitSize::Size32 => F::Vec::masked_store_32(self.data, slice.as_mut_ptr(), mask.data),
            _ => unreachable!()
        }
    }
}

impl<T: SimdElement + SimdElement<BitWidthType = B32>, F: SimdFamily> SimdVec<T, F> {
    #[inline(always)]
    pub fn runtime_permute(self, indices: SimdVec<u32, F>) -> Self {
        Self::new(self.data.permute_32(indices.data))
    }
}

impl<T: SimdElement + SimdElement<BitWidthType = B8>, F: SimdFamily> SimdVec<T, F> {
    #[inline(always)]
    pub fn runtime_permute_bytes(self, indices: Self) -> Self {
        Self::new(self.data.permute_8(indices.data))
    }
}