extern crate gl;
extern crate glutin;
extern crate serde;

mod graphics;

use crate::graphics::program::Program;
use crate::graphics::mesh::Mesh;
use cgmath::{InnerSpace, Vector3, Zero};
use gl::types::*;
use glutin::dpi::LogicalSize;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::str::FromStr;

#[derive(Serialize, Deserialize, Debug)]
struct FoldSpecification {
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
    fn from_file(path: &Path) -> std::io::Result<FoldSpecification> {
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

type Vertex = Vector3<f32>;
type Edge = [usize; 2];
type Face = [usize; 3];

#[derive(Default)]
struct Model {
    // The vertices of this model
    vertices: Vec<Vertex>,

    // The edges (creases) of this model
    edges: Vec<Edge>,

    // The faces (facets) of this model, which are assumed to be triangular
    faces: Vec<Face>,

    // The crease assignments of all of the edges of this model (i.e. "mountain," "valley," etc.)
    assignments: Vec<Assignment>,

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

    mesh: Mesh,
}

impl Model {
    fn from_specification(spec: &FoldSpecification, scale: f32) -> Model {
        let mut vertices = vec![];
        let mut edges = vec![];
        let mut faces = vec![];
        let mut assignments = vec![];
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

        // Calculate vertex masses
        let masses = vec![1.0; vertices.len()];
        let masses_min = 1.0; // TODO
        println!("Vertex with the smallest mass ({} units)\n", masses_min);

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
        assert_eq!(vertices.len(), masses.len());
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

        Model {
            vertices,
            edges,
            faces,
            assignments,
            last_angles,
            target_angles,
            nominal_lengths,
            axial_coefficients,
            k_facet,
            k_fold,
            ea,
            zeta,
            iterations,
            mesh,
            ..Default::default()
        }
    }

    fn update_mesh(&mut self) {
        unimplemented!();
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

    fn step_simulation(&mut self) {
        // First, calculate new face normals / areas
        self.update_face_data();

        // Then, step the physics simulation
        for _ in 0..self.iterations {}
    }

    fn get_vertex(&self, index: usize) -> &Vertex {
        &self.vertices[index]
    }
}

fn main() {
    let size = LogicalSize::new(720.0, 720.0);
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new().with_title("folding").with_dimensions(size);
    let gl_window = glutin::ContextBuilder::new()
        .build_windowed(window, &events_loop)
        .unwrap();

    let gl_window = unsafe { gl_window.make_current() }.unwrap();
    gl::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _);

    // Load the origami model
    let spec = FoldSpecification::from_file(Path::new("bird_base.fold")).unwrap();
    let model = Model::from_specification(&spec, 1000.0);

    // Main rendering loop
    events_loop.run_forever(|event| {
        use glutin::{ControlFlow, Event, WindowEvent};

        if let Event::WindowEvent { event, .. } = event {
            if let WindowEvent::CloseRequested = event {
                return ControlFlow::Break;
            }
        }

        unsafe {
            // Clear the screen to black
            gl::ClearColor(0.12, 0.1, 0.1, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            // Draw a triangle from the 3 vertices
            //gl::DrawArrays(gl::TRIANGLES, 0, 3);
        }

        gl_window.swap_buffers().unwrap();

        ControlFlow::Continue
    });
}
