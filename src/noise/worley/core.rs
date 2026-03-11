use crate::math::random::Random;

pub struct Worley {
    pub(super) random_gen: Random,
}

impl Worley {
    pub fn new(seed: u64) -> Self {
        Self { random_gen: Random::new(seed) }
    }
}