use crate::fold_specification::FoldSpecification;
use crate::graphics::mesh::Mesh;
use crate::half_edge::{FaceIndex, HalfEdgeIndex, HalfEdgeMesh, VertexIndex};

use cgmath::{InnerSpace, Vector3, Zero};
use std::collections::{HashMap, HashSet};

use std::str::FromStr;

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

    // The surface area of this face
    area: f32,
}

impl FaceData {
    pub fn new() -> FaceData {
        FaceData {
            normal: Vector3::unit_y(),
            area: 0.0,
        }
    }
}

pub struct Model {
    // The CPU-side half-edge mesh representation used for adjacency queries
    half_edge_mesh: HalfEdgeMesh,

    // Data associated with each edge
    edge_data: Vec<EdgeData>,

    // Data associated with each face
    face_data: Vec<FaceData>,

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
    renderable_mesh: Mesh,
}

impl Model {
    pub fn from_specification(spec: &FoldSpecification, scale: f32) -> Model {
        let mut base_vertices = vec![];
        let mut base_edges = vec![];
        let mut base_faces = vec![];
        let mut edge_data = vec![];
        let mut face_data = vec![];

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

        // Add edges
        for edge_indices in spec.edges.iter() {
            assert_eq!(edge_indices.len(), 2);
            base_edges.push([edge_indices[0] as usize, edge_indices[1] as usize]);
        }
        assert_eq!(base_edges.len(), spec.edges.len());

        // Add faces
        for face_indices in spec.faces.iter() {
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
        }
        assert_eq!(base_faces.len(), spec.faces.len());

        // Calculate vertex masses
        let masses_min = 1.0; // TODO
        println!("Vertex with the smallest mass ({} units)\n", masses_min);

        // Calculate nominal (rest) lengths for each edge
        let mut omega_max = 0.0;

        // Construct half-edge representation of this model
        let half_edge_mesh = HalfEdgeMesh::from_faces(&base_faces, &base_vertices).unwrap();

        for (i, half_edge) in half_edge_mesh.get_half_edges().iter().enumerate() {
            let half_edge_indices =
                half_edge_mesh.get_adjacent_vertices_to_half_edge(HalfEdgeIndex(i));

            // Find the index of the (base) edge of this model that matches this half-edge (i.e.
            // they start / end at the same vertex indices)
            for (base_edge_index, base_edge_indices) in base_edges.iter().enumerate() {
                if (half_edge_indices[0] == VertexIndex(base_edge_indices[0])
                    && half_edge_indices[1] == VertexIndex(base_edge_indices[1]))
                    || (half_edge_indices[0] == VertexIndex(base_edge_indices[1])
                        && half_edge_indices[1] == VertexIndex(base_edge_indices[0]))
                {
                    // Find this edge's assignment
                    let assignment_str = &spec.assignments[base_edge_index];
                    let assignment = assignment_str.parse::<Assignment>().unwrap();

                    let target_angle = match assignment {
                        Assignment::M => -std::f32::consts::PI,
                        Assignment::V => std::f32::consts::PI,
                        _ => 0.0,
                    };

                    // First, calculate the length of this edge
                    let v0 = half_edge_mesh.get_vertex(half_edge_indices[0]);
                    let v1 = half_edge_mesh.get_vertex(half_edge_indices[1]);
                    let l_0 = (v0.get_coordinates() - v1.get_coordinates()).magnitude();
                    assert!(l_0.abs() > 0.0);

                    // Then, calculate the axial coefficient for this edge
                    let k_axial = 20.0f32 / l_0;

                    let omega = (k_axial / masses_min).sqrt();
                    if omega > omega_max {
                        omega_max = omega;
                    }
                    println!(
                        "Half-edge with vertex indices {:?} assigned: {:?}",
                        half_edge_indices, assignment_str
                    );

                    edge_data.push(EdgeData::new(assignment, 0.0, target_angle, l_0, k_axial));
                }
            }
        }
        assert_eq!(half_edge_mesh.get_half_edges().len(), edge_data.len());
        println!("Edge data: {:?}\n", edge_data);
        println!("Vertices:");
        for v in half_edge_mesh.get_vertices().iter() {
            println!("{:?}", v.get_coordinates());
        }

        // Construct a renderable, GPU mesh from the adjacency information provided by the half-edge mesh
        let renderable_mesh = Mesh::new(&half_edge_mesh.gather_triangles(), None, None, None);

        // Create colors for rendering
        let mut colors = vec![];
        for data in edge_data.iter() {
            // Push back colors: each edge corresponds to 2 vertices
            colors.push(data.assignment.get_color());
            colors.push(data.assignment.get_color());
        }
        // TODO: use the colors

        // Set initial physics params
        face_data = vec![FaceData::new(); half_edge_mesh.get_faces().len()];

        let masses = vec![1.0; half_edge_mesh.get_vertices().len()];

        let params = SimulationParameters::new();

        // Calculate the timestep for the physics simulation
        let timestep_reduction = 1.2;
        let timestep = (1.0 / (2.0 * std::f32::consts::PI * omega_max)) * timestep_reduction;
        println!("Setting simulation timestep to {}\n", timestep);

        let forces = vec![Vector3::zero(); half_edge_mesh.get_vertices().len()];
        let positions = half_edge_mesh
            .get_vertices()
            .iter()
            .map(|&v| *v.get_coordinates() * 1.2)
            .collect();
        let velocities = vec![Vector3::zero(); half_edge_mesh.get_vertices().len()];
        let accelerations = vec![Vector3::zero(); half_edge_mesh.get_vertices().len()];

        Model {
            half_edge_mesh,
            edge_data,
            face_data,
            masses,
            params,
            timestep,
            forces,
            positions,
            velocities,
            accelerations,
            renderable_mesh,
        }
    }

    pub fn draw_mesh(&mut self) {
        self.renderable_mesh.draw(gl::TRIANGLES);
    }

    pub fn draw_normals(&self) {
        unimplemented!();
    }

    pub fn step_simulation(&mut self) {
        for _ in 0..self.params.iterations {
            // Reset forces
            self.forces = vec![Vector3::zero(); self.half_edge_mesh.get_vertices().len()];

            // First, calculate new face normals / areas
            self.update_face_data();

            // Apply 3 different types of forces
            self.apply_axial_constraints();
            // TODO: self.apply_crease_constraints();
            // TODO: self.apply_face_constraints();

            //println!("{:?}", self.forces[0]);

            // Integrate accelerations, velocities, and positions
            self.integrate();
        }

        // Send new position data to the GPU
        self.update_mesh();
    }

    fn update_face_data(&mut self) {
        for i in 0..self.half_edge_mesh.get_faces().len() {
            let face_indices = self
                .half_edge_mesh
                .get_adjacent_vertices_to_face(FaceIndex(i));

            // Remember that face indices are always stored in a CCW
            // winding order, as described in the .FOLD specification
            let p0 = self
                .half_edge_mesh
                .get_vertex(face_indices[0])
                .get_coordinates();
            let p1 = self
                .half_edge_mesh
                .get_vertex(face_indices[1])
                .get_coordinates();
            let p2 = self
                .half_edge_mesh
                .get_vertex(face_indices[2])
                .get_coordinates();
            let _centroid = (*p0 + *p1 + *p2) / 3.0;

            let mut u = *p1 - *p0;
            let mut v = *p2 - *p0;

            // Calculate the area of this (triangular) face before normalizing `u` and `v`
            let area = u.cross(v).magnitude() * 0.5;

            u = u.normalize();
            v = v.normalize();
            let normal = u.cross(v).normalize();

            self.face_data[i].normal = normal;
            self.face_data[i].area = area;

            // To draw the normals, create a line segment from `centroid` to the
            // point `centroid + normal`
            // ...
        }
    }

    fn apply_axial_constraints(&mut self) {
        for (i, vertex) in self.half_edge_mesh.get_vertices().iter().enumerate() {
            // 0, 1, 2, ..., 13

            let mut force_axial = Vector3::zero();
            let mut force_damping = Vector3::zero();

            let neighbors = self
                .half_edge_mesh
                .get_adjacent_vertices_to_vertex(VertexIndex(i));

            for neighbor in neighbors.iter() {


                let p0 = self.positions[i];
                let p1 = self.positions[neighbor.0];

                //println!("axial from vertices #{} -> #{}", i, neighbor.0);

                // Force from the second (neighbor) vertex towards the first vertex
                let mut direction = p0 - p1;
                let l = direction.magnitude();
                let half_edge_index = vertex.get_half_edge();

                // TODO: we have to find the index of the half-edge corresponding to the
                //  edge <i, neighbor>, then grab THIS nominal length
                let mut l_0 = self.edge_data[half_edge_index.0].nominal_length;

                for (idx, he) in self.half_edge_mesh.get_half_edges().iter().enumerate() {

                    let edges = self.half_edge_mesh.get_adjacent_vertices_to_half_edge(HalfEdgeIndex(idx));

                    if (edges[0].0 == i && edges[1] == *neighbor) || (edges[1].0 == i && edges[0] == *neighbor) {
                        l_0 = self.edge_data[idx].nominal_length;
                        break;
                    }
                }


                let k_axial = self.params.ea / l_0;
                direction = direction.normalize();

                // Accumulate axial force from this neighbor
                force_axial += -k_axial * (l - l_0) * direction * 0.1;

                // Accumulate damping force
                let c = 2.0 * self.params.zeta * (k_axial * self.masses[i]).sqrt();
                force_damping += c * (self.velocities[i] - self.velocities[neighbor.0]);
            }

            self.forces[i] += force_axial + force_damping;
        }
    }

    fn apply_crease_constraints(&mut self) {
        unimplemented!();
    }

    fn apply_face_constraints(&mut self) {
        unimplemented!();
    }

    fn integrate(&mut self) {
        for (i, a) in self.accelerations.iter_mut().enumerate() {
            *a = self.forces[i] / self.masses[i];
            *a *= self.timestep;
            //println!("{:?}", *a);
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
        // Update the vertices of the half-edge mesh
        assert_eq!(self.half_edge_mesh.get_vertices().len(), self.positions.len());

        for (vertex, p) in self
            .half_edge_mesh
            .get_vertices_mut()
            .iter_mut()
            .zip(self.positions.iter())
        {
            vertex.set_coordinates(p);
        }
        //println!("{:?}", self.positions);

        // Transfer triangles to GPU for rendering
        self.renderable_mesh
            .set_positions(&self.half_edge_mesh.gather_triangles());
    }
}
