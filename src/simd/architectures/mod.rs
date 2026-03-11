pub mod intrinsics {
    pub mod avx2;
    pub mod sse;
    pub mod avx512;

    #[cfg(target_arch = "aarch64")]
    pub mod neon;
}

#[macro_use]
pub mod macros;
pub mod families;
pub mod arch_impl;