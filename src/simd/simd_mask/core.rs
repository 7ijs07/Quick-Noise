use crate::simd::architectures::arch_impl::SimdFamily;
use crate::simd::traits::*;
use std::fmt::Debug;
use std::fmt;
use std::ops::*;
use num_traits::NumCast;
use crate::simd::traits::*;
use std::marker::PhantomData;
use crate::simd::architectures::arch_impl::MaskArch;

#[derive(Clone, Copy)]
pub struct SimdMask<T, F: SimdFamily> {
    pub(crate) data: F::Mask,
    pub(crate) _marker: PhantomData<T>,
}

impl<T: SimdElement, F: SimdFamily> SimdMask<T, F> {
    pub(crate) fn new(data: F::Mask) -> Self {
        Self { data, _marker: PhantomData }
    }
}