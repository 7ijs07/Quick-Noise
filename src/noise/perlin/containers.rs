use crate::math::vec::{Vec2, Vec3};
use crate::noise::perlin::constants::{PerlinVec, PerlinVecPair, PerlinVecTriple};

#[derive(Copy, Clone)]
pub struct Octave2D {
    pub scale: Vec2<f32>,
    pub weight: f32,
}

impl Octave2D {
    pub fn new(scale: Vec2<f32>, weight: f32) -> Self {
        Self { scale, weight }
    }

    pub fn splat(scale: f32, weight: f32) -> Self {
        Self { scale: Vec2::<f32>::new(scale, scale), weight }
    }
}

impl From<(f32, f32)> for Octave2D {
    fn from((scale, weight): (f32, f32)) -> Self {
        Octave2D::new((scale, scale).into(), weight)
    }
}

impl From<((f32, f32), f32)> for Octave2D {
    fn from(((x_scale, y_scale), weight): ((f32, f32), f32)) -> Self {
        Octave2D::new((x_scale, y_scale).into(), weight)
    }
}

impl From<&Octave2D> for Octave2D {
    fn from(octave: &Octave2D) -> Self {
        octave.clone()
    }
}

#[derive(Copy, Clone)]
pub struct Octave3D {
    pub scale: Vec3<f32>,
    pub weight: f32,
}

impl Octave3D {
    pub fn new(scale: Vec3<f32>, weight: f32) -> Self {
        Self { scale, weight }
    }

    pub fn splat(scale: f32, weight: f32) -> Self {
        Self { scale: Vec3::<f32>::new(scale, scale, scale), weight }
    }
}

pub struct PerlinContainer2D {
    vecs: [PerlinVecPair; 4],
    tl: usize, // Top left.
    tr: usize, // Top right.
    bl: usize, // Bottom left.
    br: usize, // Bottom right.
}

impl PerlinContainer2D {
    pub fn new_uninit() -> Self {
        PerlinContainer2D {
            vecs: [
                PerlinVecPair::new(PerlinVec::new_uninit(), PerlinVec::new_uninit()),
                PerlinVecPair::new(PerlinVec::new_uninit(), PerlinVec::new_uninit()),
                PerlinVecPair::new(PerlinVec::new_uninit(), PerlinVec::new_uninit()),
                PerlinVecPair::new(PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            ],
            tl: 0,
            tr: 1,
            bl: 2,
            br: 3,
        }
    }

    pub fn tl(&self) -> &PerlinVecPair { unsafe { &self.vecs.get_unchecked(self.tl) } }
    pub fn tr(&self) -> &PerlinVecPair { unsafe { &self.vecs.get_unchecked(self.tr) } }
    pub fn bl(&self) -> &PerlinVecPair { unsafe { &self.vecs.get_unchecked(self.bl) } }
    pub fn br(&self) -> &PerlinVecPair { unsafe { &self.vecs.get_unchecked(self.br) } }

    pub fn tl_tr_mut(&mut self) -> (&mut PerlinVecPair, &mut PerlinVecPair) {
        debug_assert!(self.tl < self.tr);
        debug_assert!(self.tr < self.vecs.len());
        unsafe {
            let ptr = self.vecs.as_mut_ptr();
            (&mut *ptr.add(self.tl), &mut *ptr.add(self.tr))
        }
    }

    pub fn bl_br_mut(&mut self) -> (&mut PerlinVecPair, &mut PerlinVecPair) {
        debug_assert!(self.bl < self.br);
        debug_assert!(self.br < self.vecs.len());
        unsafe {
            let ptr = self.vecs.as_mut_ptr();
            (&mut *ptr.add(self.bl), &mut *ptr.add(self.br))
        }
    }

    pub fn swap_top_bottom(&mut self) {
        std::mem::swap(&mut self.tl, &mut self.bl);
        std::mem::swap(&mut self.tr, &mut self.br);
    }
}

pub struct PerlinContainer3D {
    tlf: PerlinVecTriple, // Top left front.
    trf: PerlinVecTriple, // Top right front.
    blf: PerlinVecTriple, // Bottom left front.
    brf: PerlinVecTriple, // Bottom right front.
    tlb: PerlinVecTriple, // Top left back.
    trb: PerlinVecTriple, // Top right back.
    blb: PerlinVecTriple, // Bottom left back.
    brb: PerlinVecTriple, // Bottom right back.
}

impl PerlinContainer3D {
    pub fn new_uninit() -> Self {
        PerlinContainer3D {
            tlf: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            trf: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            blf: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            brf: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            tlb: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            trb: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            blb: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            brb: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
        }
    }
}
