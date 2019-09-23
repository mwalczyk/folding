use cgmath::Vector3;
use std::str::FromStr;

pub type Color = Vector3<f32>;

#[derive(Clone, Copy, Debug)]
pub enum Assignment {
    // A mountain fold
    M,

    // A valley fold
    V,

    // A facet (flat) fold
    F,

    // A border edge (i.e. no folding)
    B,
}

impl Assignment {
    pub fn get_target_angle(&self) -> f32 {
        match *self {
            Assignment::M => -std::f32::consts::PI,
            Assignment::V => std::f32::consts::PI,
            _ => 0.0,
        }
    }

    pub fn get_color(&self) -> Color {
        match *self {
            Assignment::M => Vector3::new(1.0, 0.0, 0.0),
            Assignment::V => Vector3::new(0.0, 0.0, 1.0),
            Assignment::F => Vector3::new(1.0, 1.0, 0.0),
            Assignment::B => Vector3::new(0.0, 1.0, 0.0),
        }
    }
}

impl FromStr for Assignment {
    type Err = ();

    fn from_str(s: &str) -> Result<Assignment, ()> {
        match s {
            "M" => Ok(Assignment::M),
            "V" => Ok(Assignment::V),
            "F" => Ok(Assignment::F),
            "B" => Ok(Assignment::B),
            _ => Err(()),
        }
    }
}
