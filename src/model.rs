use crate::graphics::mesh::Mesh;
use crate::half_edge::HalfEdgeMesh;

use cgmath::{InnerSpace, Vector3, Zero};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::Hash;
use std::io::prelude::*;
use std::path::Path;
use std::str::FromStr;

#[derive(Serialize, Deserialize, Debug)]
pub struct FoldSpecification {
    #[serde(default)]
    file_spec: i32,

    #[serde(default)]
    file_creator: String,

    #[serde(default)]
    file_author: String,

    #[serde(default)]
    frame_title: String,

    #[serde(default)]
    frame_classes: Vec<String>,

    #[serde(default)]
    frame_attributes: Vec<String>,

    #[serde(default)]
    frame_unit: String,

    #[serde(rename(deserialize = "vertices_coords"))]
    vertices: Vec<Vec<f32>>,

    #[serde(rename(deserialize = "edges_vertices"))]
    edges: Vec<Vec<i32>>,

    #[serde(rename(deserialize = "edges_assignment"))]
    assignments: Vec<String>,

    #[serde(rename(deserialize = "faces_vertices"))]
    faces: Vec<Vec<i32>>,

    // Renamed and removed all occurrences of `null` -> replaced with `10.0`
    #[serde(rename(deserialize = "edges_foldAngles"))]
    fold_angles: Vec<f32>,
}

impl FoldSpecification {
    pub fn from_file(path: &Path) -> std::io::Result<FoldSpecification> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let spec: FoldSpecification = serde_json::from_str(&contents).expect("Failed to load JSON");

        Ok(spec)
    }
}

type Color = Vector3<f32>;

#[derive(Clone, Copy, Debug)]
enum Assignment {
    M,
    V,
    F,
    B,
}

impl Assignment {
    fn get_color(&self) -> Color {
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

struct HalfEdge {
    indices: [usize; 2],
    nominal_length: f32,
    faces: [usize; 3],
}

type Vertex = Vector3<f32>;
type Edge = [usize; 2];
type Face = [usize; 3];

#[derive(Default)]
pub struct Model {
    // The vertices of this model
    vertices: Vec<Vertex>,

    // The edges (creases) of this model
    edges: Vec<Edge>,

    // The faces (facets) of this model, which are assumed to be triangular
    faces: Vec<Face>,

    // The crease assignments of all of the edges of this model (i.e. "mountain," "valley," etc.)
    assignments: Vec<Assignment>,

    //
    neighbors: HashMap<usize, HashSet<usize>>,

    // The dihedral angle made between the adjacent faces at each edge in the crease pattern, in the previous frame
    last_angles: Vec<f32>,

    // The target fold angle at each edge in the crease pattern
    target_angles: Vec<f32>,

    // The starting (rest) length of each edge in the crease pattern
    nominal_lengths: Vec<f32>,

    // `k_axial` for each edge in the crease pattern
    axial_coefficients: Vec<f32>,

    // The normal vectors of all of the faces of this model
    normals: Vec<Vector3<f32>>,

    // The surface areas of all of the faces of this model
    areas: Vec<f32>,

    // The mass of each vertex of this model
    masses: Vec<f32>,

    // The stiffness coefficient along facet creases
    k_facet: f32,

    // The stiffness coefficient along foldable (M or V) creases
    k_fold: f32,

    // Young's constant
    ea: f32,

    // A multiplier that strengthens or weakens the axial forces
    zeta: f32,

    // The number of steps (per frame) of the physics simulation
    iterations: usize,

    timestep: f32,

    forces: Vec<Vector3<f32>>,
    positions: Vec<Vector3<f32>>,
    velocities: Vec<Vector3<f32>>,
    accelerations: Vec<Vector3<f32>>,

    // The render-able mesh
    pub mesh: Mesh,
}

impl Model {
    pub fn from_specification(spec: &FoldSpecification, scale: f32) -> Model {
        let mut vertices = vec![];
        let mut edges = vec![];
        let mut faces = vec![];
        let mut assignments = vec![];
        let mut neighbors: HashMap<_, HashSet<_>> = HashMap::new();
        let mut last_angles = vec![];
        let mut target_angles = vec![];
        let mut nominal_lengths = vec![];
        let mut axial_coefficients = vec![];
        let k_facet = 0.7;
        let k_fold = 0.7;
        let ea = 20.0;
        let zeta = 0.1;
        let iterations = 1;

        // Reformat and add vertices
        for vertex_coordinates in spec.vertices.iter() {
            assert_eq!(vertex_coordinates.len(), 3);
            vertices.push(Vector3::new(
                vertex_coordinates[0] * scale,
                vertex_coordinates[1] * scale,
                vertex_coordinates[2] * scale,
            ));
        }
        assert_eq!(vertices.len(), spec.vertices.len());

        // Add edges and edge assignments
        assert_eq!(spec.edges.len(), spec.assignments.len());
        for (edge_indices, assignment_str) in spec.edges.iter().zip(spec.assignments.iter()) {
            assert_eq!(edge_indices.len(), 2);
            edges.push([edge_indices[0] as usize, edge_indices[1] as usize]);

            // TODO: does it matter if the edge indices are sorted, per edge?
            // ...

            if let Ok(assignment) = assignment_str.parse::<Assignment>() {
                assignments.push(assignment);
            } else {
                println!("Error parsing crease assignment string: defaulting to F");
                assignments.push(Assignment::F);
            }
        }

        // Add faces (performing triangulation if necessary)
        for face_indices in spec.faces.iter() {
            if face_indices.len() > 3 {
                // TODO
                panic!("Attempting to process facet with more than 3 vertices");
            }
            assert_eq!(face_indices.len(), 3);
            faces.push([
                face_indices[0] as usize,
                face_indices[1] as usize,
                face_indices[2] as usize,
            ]);
        }

        // Create neighborhood map
        for edge_indices in edges.iter() {
            if let Some(v) = neighbors.get_mut(&edge_indices[0]) {
                v.insert(edge_indices[1]);
            } else {
                neighbors.insert(edge_indices[0], HashSet::new());
            }

            if let Some(v) = neighbors.get_mut(&edge_indices[1]) {
                v.insert(edge_indices[0]);
            } else {
                neighbors.insert(edge_indices[1], HashSet::new());
            }
        }

        // Calculate the target angle of each crease
        for assignment in assignments.iter() {
            // TODO: this could be done in the for-loop above (when we parse the assignments)
            // ...

            // TODO: remember, `null` values in the JSON need to be dealt with, but
            //    for now we have manually edited them
            // ...

            target_angles.push(match assignment {
                Assignment::M => -std::f32::consts::PI,
                Assignment::V => std::f32::consts::PI,
                _ => 0.0,
            });

            // TODO: what should this be?
            last_angles.push(0.0);
        }

        // Calculate vertex masses
        let masses_min = 1.0; // TODO
        println!("Vertex with the smallest mass ({} units)\n", masses_min);

        // Calculate nominal (rest) lengths for each edge
        let mut omega_max = 0.0;
        for edge_indices in edges.iter() {
            // First, calculate the length of this edge
            let l_0 = (vertices[edge_indices[0]] - vertices[edge_indices[1]]).magnitude();
            nominal_lengths.push(l_0);

            // Then, calculate the axial coefficient for this edge
            let k_axial = 20.0f32 / l_0;
            axial_coefficients.push(k_axial);

            let omega = (k_axial / masses_min).sqrt();
            if omega > omega_max {
                omega_max = omega;
            }
        }

        // Calculate the timestep for the physics simulation
        let timestep_reduction = 1.2;
        let timestep = (1.0 / (2.0 * std::f32::consts::PI * omega_max)) * timestep_reduction;
        println!("Setting simulation timestep to {}\n", timestep);

        // Some extra sanity checks
        assert_eq!(edges.len(), assignments.len());
        assert_eq!(edges.len(), last_angles.len());
        assert_eq!(edges.len(), target_angles.len());
        assert_eq!(edges.len(), nominal_lengths.len());
        assert_eq!(edges.len(), axial_coefficients.len());

        // Debug printing
        println!("vertices: {:?}\n", vertices);
        println!("edges: {:?}\n", edges);
        println!("faces: {:?}\n", faces);
        println!("assignments: {:?}\n", assignments);
        println!("neighbors: {:?}\n", neighbors);
        println!("last angles: {:?}\n", last_angles);
        println!("target_angles: {:?}\n", target_angles);
        println!("nominal lengths: {:?}\n", nominal_lengths);
        println!("axial coefficients: {:?}\n", axial_coefficients);

        // Finally, build the mesh for rendering
        let mut positions = vec![];
        let mut colors = vec![];
        for (edge_indices, assignment) in edges.iter().zip(assignments.iter()) {
            // Push back the two endpoints
            positions.push(vertices[edge_indices[0]]);
            positions.push(vertices[edge_indices[1]]);

            // Push back colors
            colors.push(assignment.get_color());
            colors.push(assignment.get_color());
        }
        let mesh = Mesh::new(&positions, Some(&colors), None, None);

        // Set initial physics params
        let normals = vec![Vector3::zero(); faces.len()];
        let areas = vec![0.0; faces.len()];
        let masses = vec![1.0; vertices.len()];
        let forces = vec![Vector3::zero(); vertices.len()];
        let positions = vertices.clone();
        let velocities = vec![Vector3::zero(); vertices.len()];
        let accelerations = vec![Vector3::zero(); vertices.len()];

        let hem = HalfEdgeMesh::from_faces(&faces, &vertices);

        Model {
            vertices,
            edges,
            faces,
            assignments,
            neighbors,
            last_angles,
            target_angles,
            nominal_lengths,
            axial_coefficients,
            normals,
            areas,
            masses,
            k_facet,
            k_fold,
            ea,
            zeta,
            iterations,
            timestep,
            forces,
            positions,
            velocities,
            accelerations,
            mesh,
        }
    }

    fn draw_mesh(&self) {
        unimplemented!();
    }

    fn draw_normals(&self) {
        unimplemented!();
    }

    pub fn step_simulation(&mut self) {
        for _ in 0..self.iterations {
            // Reset forces
            self.forces = vec![Vector3::zero(); self.vertices.len()];

            // First, calculate new face normals / areas
            self.update_face_data();

            // Apply 3 different types of forces
            self.apply_axial_constraints();
            self.apply_crease_constraints();
            self.apply_face_constraints();

            // Integrate accelerations, velocities, and positions
            self.integrate();
        }

        // Send new position data to the GPU
        self.update_mesh();
    }

    fn update_face_data(&mut self) {
        for (face_index, face_indices) in self.faces.iter().enumerate() {
            // Remember that face indices are always stored in a CCW
            // winding order, as described in the .FOLD specification
            let p0 = self.get_vertex(face_indices[0]);
            let p1 = self.get_vertex(face_indices[1]);
            let p2 = self.get_vertex(face_indices[2]);
            let centroid = (*p0 + *p1 + *p2) / 3.0;

            let mut u = *p1 - *p0;
            let mut v = *p2 - *p0;

            // Calculate the area of this (triangular) face before normalizing `u` and `v`
            let area = u.cross(v).magnitude() * 0.5;

            u = u.normalize();
            v = v.normalize();
            let normal = u.cross(v).normalize();

            self.normals[face_index] = normal;
            self.areas[face_index] = area;

            // To draw the normals, create a line segment from `centroid` to the
            // point `centroid + normal`
            // ...
        }
    }

    fn update_mesh(&mut self) {
        self.mesh.set_positions(&self.positions);
    }

    fn apply_axial_constraints(&mut self) {
        for (vertex_index, vertex) in self.vertices.iter().enumerate() {
            let mut force_axial = Vector3::zero();
            let mut force_damping = Vector3::zero();

            for neighbor_index in self.neighbors.get(&vertex_index).unwrap().iter() {
                // The indices of the current vertex + the current neighbor together
                let axial_indices = [vertex_index, *neighbor_index];

                // The index of the edge above within the model's list of edges
                let pos = self
                    .edges
                    .iter()
                    .position(|&edge_indices| edge_indices == axial_indices)
                    .unwrap();
                // TODO: `pos` above probably needs to be sorted

                let mut direction = *vertex - self.vertices[*neighbor_index];
                let l = direction.magnitude();
                let l_0 = self.nominal_lengths[pos];
                let k_axial = self.ea / l_0;
                direction = direction.normalize();

                // Accumulate axial force from this neighbor
                force_axial += -k_axial * (l - l_0) * direction;

                // Accumulate damping force
                let c = 2.0 * self.zeta * (k_axial * self.masses[vertex_index]).sqrt();
                force_damping +=
                    c * (self.velocities[*neighbor_index] - self.velocities[vertex_index]);
            }

            // Accumulate both forces calculated above in the inner for-loop
            self.forces[vertex_index] += force_axial + force_damping;
        }
    }

    fn apply_crease_constraints(&mut self) {
        unimplemented!();
    }

    fn apply_face_constraints(&mut self) {
        unimplemented!();
    }

    fn integrate(&mut self) {
        for (vertex_index, a) in self.accelerations.iter_mut().enumerate() {
            *a = self.forces[vertex_index] / self.masses[vertex_index];
            *a *= self.timestep;
        }

        for (vertex_index, v) in self.velocities.iter_mut().enumerate() {
            *v += self.accelerations[vertex_index];
            *v *= self.timestep;
        }

        for (vertex_index, p) in self.positions.iter_mut().enumerate() {
            *p += self.velocities[vertex_index];
        }
    }

    fn get_vertex(&self, index: usize) -> &Vertex {
        &self.vertices[index]
    }
}
