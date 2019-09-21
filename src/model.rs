use crate::fold_specification::FoldSpecification;
use crate::graphics::mesh::Mesh;
use crate::half_edge::{FaceIndex, HalfEdgeIndex, HalfEdgeMesh, VertexIndex};

use cgmath::{InnerSpace, Vector3, Zero};
use std::collections::{HashMap, HashSet};

use std::str::FromStr;

type Color = Vector3<f32>;

fn cot(x: f32) -> f32 {
    1.0 / x.tan()
}

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

#[derive(Clone, Copy, Debug)]
struct SimulationParameters {
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
}

impl SimulationParameters {
    pub fn new() -> SimulationParameters {
        SimulationParameters {
            k_facet: 0.7,
            k_fold: 0.7,
            ea: 20.0,
            zeta: 0.1,
            iterations: 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct EdgeData {
    // The crease assignments of this edge (i.e. "mountain," "valley," etc.)
    assignment: Assignment,

    // The dihedral angle made between the adjacent faces of this edge in the crease pattern, in the previous frame
    last_angle: f32,

    // The target fold angle of this edge in the crease pattern
    target_angle: f32,

    // The starting (rest) length of this edge in the crease pattern
    nominal_length: f32,

    // `k_axial` for this edge in the crease pattern
    axial_coefficient: f32,
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
struct FaceData {
    // The normal vector of this face
    normal: Vector3<f32>,

    // The centroid of this face
    centroid: Vector3<f32>,

    // The surface area of this face
    area: f32,
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

pub struct Model {
    vertices: Vec<Vector3<f32>>,
    faces: Vec<[usize; 3]>,
    edges: Vec<[usize; 2]>,

    // Data associated with each edge
    edge_data: Vec<EdgeData>,

    // Data associated with each face
    face_data: Vec<FaceData>,

    neighbors: HashMap<usize, HashSet<usize>>,

    opposites: HashMap<usize, Vec<(usize, usize)>>,

    // The mass of each vertex of this model
    masses: Vec<f32>,

    // Various parameters for the simulation (described above)
    params: SimulationParameters,

    // The simulation timestep
    timestep: f32,

    // The forces acting on each vertex
    forces: Vec<Vector3<f32>>,

    // The position of each (dynamic) vertex
    positions: Vec<Vector3<f32>>,

    // The velocity of each (dynamic) vertex
    velocities: Vec<Vector3<f32>>,

    // The acceleration of each (dynamic) vertex
    accelerations: Vec<Vector3<f32>>,

    // The render-able mesh
    mesh_body: Mesh,

    // The mesh for drawing debug normals
    mesh_debug: Mesh,
}

impl Model {
    pub fn from_specification(spec: &FoldSpecification, scale: f32) -> Model {
        let mut base_vertices = vec![];
        let mut base_edges = vec![];
        let mut base_faces = vec![];
        let mut edge_data = vec![];
        let mut face_data = vec![];

        // Set initial physics params
        let params = SimulationParameters::new();

        // Add vertices
        for vertex_coordinates in spec.vertices.iter() {
            assert_eq!(vertex_coordinates.len(), 3);
            base_vertices.push(Vector3::new(
                vertex_coordinates[0] * scale,
                vertex_coordinates[1] * scale,
                vertex_coordinates[2] * scale,
            ));
        }
        assert_eq!(base_vertices.len(), spec.vertices.len());

        // TODO: calculate vertex masses and find the minimum
        // ...
        let mut masses_min = 1.0;
        let mut omega_max = 0.0;

        // Add edges
        for (edge_index, edge_indices) in spec.edges.iter().enumerate() {
            assert_eq!(edge_indices.len(), 2);
            base_edges.push([edge_indices[0] as usize, edge_indices[1] as usize]);

            // Find this edge's assignment
            let assignment_str = &spec.assignments[edge_index];
            let assignment = assignment_str.parse::<Assignment>().unwrap();

            // Find this edge's target fold angle
            let target_angle = match assignment {
                Assignment::M => -std::f32::consts::PI,
                Assignment::V => std::f32::consts::PI,
                _ => 0.0,
            };

            // Find this edge's nominal (starting) length
            let v0 = base_vertices[edge_indices[0] as usize];
            let v1 = base_vertices[edge_indices[1] as usize];
            let l_0 = (v0 - v1).magnitude();
            assert!(l_0.abs() > 0.0);

            // Find this edge's axial coefficient
            let k_axial = params.ea / l_0;

            // Used for calculating the simulation timestep below
            let omega = (k_axial / masses_min).sqrt();
            if omega > omega_max {
                omega_max = omega;
            }

            edge_data.push(EdgeData::new(assignment, 0.0, target_angle, l_0, k_axial));
        }
        assert_eq!(base_edges.len(), spec.edges.len());
        assert_eq!(base_edges.len(), edge_data.len());

        // Add faces
        for (face_index, face_indices) in spec.faces.iter().enumerate() {
            if face_indices.len() > 3 {
                // TODO: triangulate face, if necessary
                panic!("Attempting to process facet with more than 3 vertices");
            }
            assert_eq!(face_indices.len(), 3);
            base_faces.push([
                face_indices[0] as usize,
                face_indices[1] as usize,
                face_indices[2] as usize,
            ]);

            face_data.push(FaceData::new());
        }
        assert_eq!(base_faces.len(), spec.faces.len());
        assert_eq!(base_faces.len(), face_data.len());

        // Create neighborhood map, which tells us the indices of all of the vertices (the neighbors)
        // that surround each vertex
        let mut neighbors: HashMap<_, HashSet<_>> = HashMap::new();
        for edge_indices in base_edges.iter() {
            let (a, b) = (edge_indices[0], edge_indices[1]);

            if let Some(v) = neighbors.get_mut(&a) {
                v.insert(b);
            } else {
                neighbors.insert(a, HashSet::new());
                neighbors.get_mut(&a).unwrap().insert(b);
            }

            if let Some(v) = neighbors.get_mut(&b) {
                v.insert(a);
            } else {
                neighbors.insert(b, HashSet::new());
                neighbors.get_mut(&b).unwrap().insert(a);
            }
        }

        // Each entry looks like:
        //
        // edge_index: {(face_index, vertex_index), (face_index, vertex_index), ... }
        let mut opposites: HashMap<_, Vec<_>> = HashMap::new();

        for (edge_index, edge_indices) in base_edges.iter().enumerate() {
            println!("Examing edge #{}: {:?}", edge_index, edge_indices);

            for (face_index, face_indices) in base_faces.iter().enumerate() {
                let edge_set: HashSet<usize> = edge_indices.iter().cloned().collect();
                assert_eq!(edge_set.len(), 2);
                let face_set: HashSet<usize> = face_indices.iter().cloned().collect();
                assert_eq!(face_set.len(), 3);

                // Is this edge part of this face?
                if edge_set.is_subset(&face_set) {
                    println!("\tEdge is part of face #{}: {:?}", face_index, face_indices);

                    // The difference should always yield a list with a single element
                    let difference: Vec<_> = face_set.difference(&edge_set).collect();
                    assert_eq!(difference.len(), 1);

                    // The third vertex in this face that is opposite this edge
                    let opposite = *difference[0];

                    if let Some(v) = opposites.get_mut(&edge_index) {
                        v.push((face_index, opposite));
                    } else {
                        opposites.insert(edge_index, Vec::new());
                        opposites
                            .get_mut(&edge_index)
                            .unwrap()
                            .push((face_index, opposite));
                    }
                }
            }
        }
        println!("{:?}", opposites);

        let mut line_positions = vec![];
        let mut colors = vec![];
        for (edge_indices, data) in base_edges.iter().zip(edge_data.iter()) {
            // Push back the two endpoints
            line_positions.push(base_vertices[edge_indices[0]]);
            line_positions.push(base_vertices[edge_indices[1]]);

            // Push back colors
            colors.push(data.assignment.get_color());
            colors.push(data.assignment.get_color());
        }

        // Construct a renderable, GPU mesh from the adjacency information provided by the half-edge mesh
        let mesh_body = Mesh::new(&line_positions, Some(&colors), None, None);
        let mesh_debug = Mesh::new(
            &vec![Vector3::zero(); base_faces.len() * 2],
            None,
            None,
            None,
        );

        // Calculate the timestep for the physics simulation
        let timestep_reduction = 0.1;
        let timestep = (1.0 / (2.0 * std::f32::consts::PI * omega_max)) * timestep_reduction;
        println!("Setting simulation timestep to {}\n", timestep);

        let masses = vec![1.0; base_vertices.len()];
        let forces = vec![Vector3::zero(); base_vertices.len()];
        let mut positions = base_vertices.clone();
        for p in positions.iter_mut() {
            *p *= 1.4;
        }
        let velocities = vec![Vector3::zero(); base_vertices.len()];
        let accelerations = vec![Vector3::zero(); base_vertices.len()];

        Model {
            vertices: base_vertices,
            faces: base_faces,
            edges: base_edges,
            edge_data,
            face_data,
            neighbors,
            opposites,
            masses,
            params,
            timestep,
            forces,
            positions,
            velocities,
            accelerations,
            mesh_body,
            mesh_debug,
        }
    }

    pub fn draw_mesh(&mut self) {
        self.mesh_body.draw(gl::LINES);
    }

    pub fn draw_normals(&self) {
        self.mesh_debug.draw(gl::LINES);
    }

    pub fn step_simulation(&mut self) {
        for _ in 0..self.params.iterations {
            // Reset forces
            self.forces = vec![Vector3::zero(); self.positions.len()];

            // Calculate new face normals / areas
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
            let p0 = self.positions[face_indices[0]];
            let p1 = self.positions[face_indices[1]];
            let p2 = self.positions[face_indices[2]];
            let centroid = (p0 + p1 + p2) / 3.0;

            let mut u = p1 - p0;
            let mut v = p2 - p0;

            // Calculate the area of this (triangular) face before normalizing `u` and `v`
            let area = u.cross(v).magnitude() * 0.5;

            u = u.normalize();
            v = v.normalize();
            let normal = u.cross(v).normalize();

            self.face_data[face_index].normal = normal;
            self.face_data[face_index].area = area;
            self.face_data[face_index].centroid = centroid;

            // To draw the normals, create a line segment from `centroid` to the
            // point `centroid + normal`
            // ...
        }
    }

    fn apply_axial_constraints(&mut self) {
        for (vertex_index, vertex) in self.positions.iter().enumerate() {
            let mut force_axial = Vector3::zero();
            let mut force_damping = Vector3::zero();

            for neighbor_index in self.neighbors.get(&vertex_index).unwrap().iter() {
                // The indices of the current vertex + the current neighbor together
                let axial_indices = [vertex_index, *neighbor_index];

                // The index of the edge above within the model's list of edges
                let mut pos = 0;
                let mut found = false;
                for (edge_index, edge_indices) in self.edges.iter().enumerate() {
                    let mut rev = axial_indices.clone();
                    rev.reverse();

                    if *edge_indices == axial_indices || *edge_indices == rev {
                        pos = edge_index;
                        found = true;
                        break;
                    }
                }
                if !found {
                    panic!("Couldn't find matching edge");
                }

                let mut direction = *vertex - self.positions[*neighbor_index];
                let l = direction.magnitude();
                let l_0 = self.edge_data[pos].nominal_length;
                let k_axial = self.params.ea / l_0;
                direction = direction.normalize();

                // Accumulate axial force from this neighbor
                force_axial += -k_axial * (l - l_0) * direction;

                // Accumulate damping force
                let c = 2.0 * self.params.zeta * (k_axial * self.masses[vertex_index]).sqrt();
                force_damping +=
                    c * (self.velocities[*neighbor_index] - self.velocities[vertex_index]);
            }

            // Accumulate both forces calculated above in the inner for-loop
            self.forces[vertex_index] += force_axial + force_damping;
        }
    }

    fn apply_crease_constraints(&mut self) {
        for (edge_index, edge_indices) in self.edges.iter().enumerate() {
            // Don't process border edges, which only have a single face neighbor
            if self.opposites.get(&edge_index).unwrap().len() == 2 {

                let opposite = self.opposites.get(&edge_index).unwrap();

                // The indices of the two faces adjacent to this edge
                let face_0 = opposite[0].0;
                let face_1 = opposite[1].0;

                // The indices of the two vertices opposite this edge, across each of the two
                // adjacent faces
                let out_0 = opposite[0].1;
                let out_1 = opposite[1].1;

                // The indices of the two vertices that form this edge
                let p0 = edge_indices[0];
                let p1 = edge_indices[1];

                // The base of each face triangle (the length of this crease)
                let b = (self.positions[p0] - self.positions[p1]).magnitude();

                // Grab the normal vectors of the two adjacent faces
                let n_0 = self.face_data[face_0].normal;
                let n_1 = self.face_data[face_1].normal;

                // Grab the surface areas of the two adjacent faces
                let a_0 = self.face_data[face_0].area;
                let a_1 = self.face_data[face_1].area;

                // Calculate the altitudes of the two adjacent (triangular) faces
                let h_0 = 2.0 * (a_0 / b);
                let h_1 = 2.0 * (a_1 / b);

                // Interior angles on triangle `face_0`
                let p0_out_0 = (self.positions[p0] - self.positions[out_0]).magnitude();
                let p1_out_0 = (self.positions[p1] - self.positions[out_0]).magnitude();

                let angle_p0_out_0 = if (h_0 / p0_out_0).abs() > 1.0 {
                    std::f32::consts::FRAC_PI_2 // 90 degrees
                } else {
                   (h_0 / p0_out_0).asin()
                };
                let angle_p1_out_0 = if (h_0 / p1_out_0).abs() > 1.0 {
                    std::f32::consts::FRAC_PI_2 // 90 degrees
                } else {
                    (h_0 / p1_out_0).asin()
                };

                // Interior angles on triangle `face_1`
                let p0_out_1 = (self.positions[p0] - self.positions[out_1]).magnitude();
                let p1_out_1 = (self.positions[p1] - self.positions[out_1]).magnitude();

                let angle_p0_out_1 = if (h_1 / p0_out_1).abs() > 1.0 {
                    std::f32::consts::FRAC_PI_2 // 90 degrees
                } else {
                    (h_1 / p0_out_1).asin()
                };
                let angle_p1_out_1 = if (h_1 / p1_out_1).abs() > 1.0 {
                    std::f32::consts::FRAC_PI_2 // 90 degrees
                } else {
                    (h_1 / p1_out_1).asin()
                };

                // The "edge" vector along the creased edge
                let mut crease_vector = (self.positions[p1] - self.positions[p0]).normalize();
                let mut dot_normals = n_0.dot(n_1);

                // Clamp to range -1..1
                dot_normals = dot_normals.min(1.0).max(-1.0);

                // https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates
                let mut dihedral = ((n_0.cross(crease_vector)).dot(n_1)).atan2(dot_normals);

                let assignment = self.edge_data[edge_index].assignment;
                let l_0 = self.edge_data[edge_index].nominal_length;
                let fold_percent = 0.85;

                let target_angle = self.edge_data[edge_index].target_angle * fold_percent;

                let amount = match assignment {
                    Assignment::M => {
                        let k_crease = l_0 * self.params.k_fold;
                        if dihedral > 0.0 {
                            dihedral *= -1.0
                        }
                        -k_crease * (dihedral - target_angle)
                    },
                    Assignment::V => {
                        let k_crease = l_0 * self.params.k_fold;
                        if dihedral < 0.0 {
                            dihedral *= -1.0;
                        }
                        -k_crease * (dihedral - target_angle)
                    },
                    Assignment::F => {
                        let k_crease = l_0 * self.params.k_facet;
                        -k_crease * (dihedral - target_angle)
                    },
                    _ => 0.0,
                };

                // Forces on the two "outer" points
                self.forces[out_0] += amount * (n_0 / h_0);
                self.forces[out_1] += amount * (n_1 / h_1);

                // Forces on the first "hinge" joint, `p0`
                let coeff1 = -cot(angle_p1_out_0) / (cot(angle_p0_out_0) + cot(angle_p1_out_0));
                let coeff2 = -cot(angle_p1_out_1) / (cot(angle_p0_out_1) + cot(angle_p1_out_1));
                self.forces[p0] += amount * (coeff1 * (n_0 / h_0) + coeff2 * (n_1 / h_1));

                // Forces on the second "hinge" joint, `p1`
                let coeff1 = -cot(angle_p0_out_0) / (cot(angle_p1_out_0) + cot(angle_p0_out_0));
                let coeff2 = -cot(angle_p0_out_1) / (cot(angle_p1_out_1) + cot(angle_p0_out_1));
                self.forces[p1] += amount * (coeff1 * (n_0 / h_0) + coeff2 * (n_1 / h_1));
            }
        }
    }

    fn apply_face_constraints(&mut self) {
       
    }

    fn integrate(&mut self) {
        for (i, a) in self.accelerations.iter_mut().enumerate() {
            *a = self.forces[i] / self.masses[i];
            *a *= self.timestep;
        }

        for (i, v) in self.velocities.iter_mut().enumerate() {
            *v += self.accelerations[i];
            *v *= self.timestep;
        }

        for (i, p) in self.positions.iter_mut().enumerate() {
            *p += self.velocities[i];
        }
    }

    fn update_mesh(&mut self) {
        // Transfer face data to GPU for rendering
        let mut normal_debug_lines = vec![];
        let length = 50.0;
        for data in self.face_data.iter() {
            normal_debug_lines.push(data.centroid);
            normal_debug_lines.push(data.centroid + data.normal * length);
        }
        self.mesh_debug.set_positions(&normal_debug_lines);

        let mut line_positions = vec![];
        let mut colors = vec![];
        for (edge_indices, data) in self.edges.iter().zip(self.edge_data.iter()) {
            // Push back the two endpoints
            line_positions.push(self.positions[edge_indices[0]]);
            line_positions.push(self.positions[edge_indices[1]]);

            // Push back colors
            colors.push(data.assignment.get_color());
            colors.push(data.assignment.get_color());
        }
        self.mesh_body = Mesh::new(&line_positions, Some(&colors), None, None);
    }
}
