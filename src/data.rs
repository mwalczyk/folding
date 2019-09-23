use crate::assignment::Assignment;

use cgmath::{Vector3, Zero};

#[derive(Clone, Copy, Debug)]
pub struct VertexData {
    // The position of this vertex in 3-space
    pub position: Vector3<f32>,

    // The current velocity of this vertex
    pub velocity: Vector3<f32>,

    // The current acceleration of this vertex
    pub acceleration: Vector3<f32>,

    // The current cumulative forces acting on this vertex
    pub force: Vector3<f32>,

    // The mass of this vertex
    pub mass: f32,
}

impl VertexData {
    pub fn new(position: Vector3<f32>) -> VertexData {
        VertexData {
            position,
            velocity: Vector3::zero(),
            acceleration: Vector3::zero(),
            force: Vector3::zero(),
            mass: 1.0,
        }
    }

    pub fn reset_force(&mut self) {
        self.force = Vector3::zero();
    }

    pub fn integrate(&mut self, timestep: f32) {
        self.acceleration = self.force / self.mass;
        self.acceleration *= timestep;

        self.velocity += self.acceleration;
        self.velocity *= timestep;

        self.position += self.velocity;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EdgeData {
    // The crease assignments of this edge (i.e. "mountain," "valley," etc.)
    pub assignment: Assignment,

    // The dihedral angle made between the adjacent faces of this edge in the crease pattern, in the previous frame
    pub last_angle: f32,

    // The target fold angle of this edge in the crease pattern
    pub target_angle: f32,

    // The starting (rest) length of this edge in the crease pattern
    pub nominal_length: f32,

    // `k_axial` for this edge in the crease pattern
    pub axial_coefficient: f32,
}

impl EdgeData {
    pub fn new(
        assignment: Assignment,
        last_angle: f32,
        target_angle: f32,
        nominal_length: f32,
        axial_coefficient: f32,
    ) -> EdgeData {
        EdgeData {
            assignment,
            last_angle,
            target_angle,
            nominal_length,
            axial_coefficient,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FaceData {
    // The normal vector of this face
    pub normal: Vector3<f32>,

    // The centroid of this face
    pub centroid: Vector3<f32>,

    // The surface area of this face
    pub area: f32,
}

impl FaceData {
    pub fn new() -> FaceData {
        FaceData {
            normal: Vector3::unit_y(),
            centroid: Vector3::zero(),
            area: 0.0,
        }
    }
}
