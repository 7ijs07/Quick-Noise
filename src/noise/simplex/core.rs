use crate::math::random::Random;

pub struct Simplex {
    pub(super) random_gen: Random,
}

impl Simplex {
    pub fn new(seed: u64) -> Self {
        Self { random_gen: Random::new(seed) }
    }
}