use crate::math::random::Random;

pub struct Value {
    pub(super) random_gen: Random,
}

impl Value {
    pub fn new(seed: u64) -> Self {
        Self { random_gen: Random::new(seed) }
    }
}