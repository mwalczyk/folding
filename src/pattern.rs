use crate::fold_specification::FoldSpecification;

use cgmath::Vector3;

/// The Huzita-Hatori Axioms.
///
/// Reference: `https://en.wikipedia.org/wiki/Huzita%E2%80%93Hatori_axioms`
pub enum Axiom {
    Axiom1,
    Axiom2,
    Axiom3,
    Axiom4,
    Axiom5,
    Axiom6,
    Axiom7,
}

pub trait Pattern {
    fn generate() -> FoldSpecification;
}

/// A struct representing a flat, "uncreased" piece of paper.
struct Uncreased {
}

impl Uncreased {
    pub fn apply_axiom(&mut self, axiom: Axiom) {
        unimplemented!();
    }
}

/// A struct representing a semigeneralized Miura-ori (SGMO) with an
/// arbitrary cross section.
struct MiuraOri {
    // The piecewise linear line segment that forms the cross-section of the resulting Miura-ori
    generating_line: Vec<Vector3<f32>>,

    // The number of times the crease pattern is repeated vertically
    repetitions: usize,
}

impl MiuraOri {
    pub fn new() {
        unimplemented!();
    }
}

impl Pattern for MiuraOri {
    fn generate() -> FoldSpecification {
        unimplemented!();
    }
}

struct RegularPolygon {
    sides: usize,
    radius: f32,
}

/// A struct representing a single, untiled simple flat twist (SFT), i.e.
/// a twist that is flat-foldable.
struct SimpleFlatTwist {
    central_polygon: RegularPolygon,
}