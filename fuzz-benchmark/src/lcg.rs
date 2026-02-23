use std::convert::Infallible;

use rand::{Rng, SeedableRng, TryRng};

#[derive(Debug, Clone)]
pub struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    const A: u64 = 6364136223846793005;
    const C: u64 = 1;

    #[inline(always)]
    fn next_state(&mut self) {
        self.state = self.state.wrapping_mul(Self::A).wrapping_add(Self::C);
    }
}

impl TryRng for Lcg64 {
    type Error = Infallible;

    #[inline(always)]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        self.next_state();
        return Ok((self.state >> 32) as u32);
    }

    #[inline(always)]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        self.next_state();
        return Ok(self.state);
    }

    #[inline(always)]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        let mut chunks = dst.chunks_exact_mut(4);

        for chunk in &mut chunks {
            let rand = self.next_u32().to_ne_bytes();
            chunk.copy_from_slice(&rand);
        }

        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let rand = self.next_u32().to_ne_bytes();
            remainder.copy_from_slice(&rand[..remainder.len()]);
        }

        return Ok(());
    }
}

impl SeedableRng for Lcg64 {
    type Seed = [u8; 8];

    #[inline(always)]
    fn from_seed(seed: Self::Seed) -> Self {
        let state = u64::from_ne_bytes(seed);
        return Self { state };
    }

    #[inline(always)]
    fn seed_from_u64(seed: u64) -> Self {
        return Self { state: seed };
    }
}
