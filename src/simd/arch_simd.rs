use std::simd::{Simd, SimdElement};

// Static dispatch for identifying lane sizes and number of simd registers.

pub const trait SimdConstants {
    const WIDTH: usize;
    const NUM_REG: usize;
    const LANES: usize = Self::WIDTH / std::mem::size_of::<f32>();
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Avx512;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Avx2;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Sse2;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Sve;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Neon;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Simd128;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct V;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fallback;

cfg_if::cfg_if! {
    // x86_64
    if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
        pub type Current = Avx512;
    } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))] {
        pub type Current = Avx2;
    } else if #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))] {
        pub type Current = Sse2;
    }

    // aarch64
    else if #[cfg(all(target_arch = "aarch64", target_feature = "sve"))] {
        pub type Current = Sve;
    } else if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
        pub type Current = Neon;
    }

    // wasm
    else if #[cfg(all(any(target_arch = "wasm32", target_arch = "wasm64"), target_feature = "simd128"))] {
        pub type Current = Simd128;
    }

    // riscv
    else if #[cfg(all(any(target_arch = "riscv64", target_arch = "riscv32"), target_feature = "v"))] {
        pub type Current = V;
    }

    // fallback
    else {
        pub type Current = Fallback;
    }
}

impl const SimdConstants for Avx512 {
    const WIDTH: usize = 64;
    const NUM_REG: usize = 32;
}

impl const SimdConstants for Avx2 {
    const WIDTH: usize = 32;
    const NUM_REG: usize = 16;
}

impl const SimdConstants for Sse2 {
    const WIDTH: usize = 16;
    const NUM_REG: usize = 16;
}

impl const SimdConstants for Sve {
    const WIDTH: usize = 32;
    const NUM_REG: usize = 32;
}

impl const SimdConstants for Neon {
    const WIDTH: usize = 16;
    const NUM_REG: usize = 32;
}

impl const SimdConstants for Simd128 {
    const WIDTH: usize = 16;
    const NUM_REG: usize = 16;
}

impl const SimdConstants for V {
    const WIDTH: usize = 32;
    const NUM_REG: usize = 32;
}

impl const SimdConstants for Fallback {
    const WIDTH: usize = 4;
    const NUM_REG: usize = 8;
}

pub type ArchSimd<SC: const SimdConstants, T: const SimdInfo<SC>> = Simd<T, { SC::LANES }>;

pub const trait SimdInfo<SC: const SimdConstants>: SimdElement {}

impl<SC: const SimdConstants> const SimdInfo<SC> for f32 {}

impl<SC: const SimdConstants> const SimdInfo<SC> for u32 {}

impl<SC: const SimdConstants> const SimdInfo<SC> for i32 {}

// impl<SC: const SimdConstants<Double>> const SimdInfo<Double, SC> for f64 {}

// impl<SC: const SimdConstants<Double>> const SimdInfo<Double, SC> for u64 {}

// impl<SC: const SimdConstants<Double>> const SimdInfo<Double, SC> for i64 {}
