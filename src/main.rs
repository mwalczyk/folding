extern crate serde;

use cgmath::{InnerSpace, Vector3, Zero};
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

#[derive(Clone, Copy, Debug)]
enum Assignment {
    M,
    V,
    F,
    B,
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

    // The normal vectors of all of the faces of this model
    normals: Vec<Vector3<f32>>,

    // The surface areas of all of the faces of this model
    areas: Vec<f32>,
}

impl Model {
    fn from_specification(spec: &FoldSpecification) -> Model {
        let mut vertices = vec![];
        let mut edges = vec![];
        let mut faces = vec![];
        let mut assignments = vec![];

        // Reformat and add vertices
        for vertex_coordinates in spec.vertices.iter() {
            assert_eq!(vertex_coordinates.len(), 3);
            vertices.push(Vector3::new(
                vertex_coordinates[0],
                vertex_coordinates[1],
                vertex_coordinates[2],
            ));
        }

        assert_eq!(spec.edges.len(), spec.assignments.len());

        // Add edges
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
        assert_eq!(edges.len(), assignments.len());

        // Add faces, performing triangulation if necessarily
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

        println!("{:?}", vertices);
        println!("{:?}", edges);
        println!("{:?}", faces);
        println!("{:?}", assignments);

        // Initialize other simulations vars to some sensible defaults
        let masses = vec![1.0; vertices.len()];
        let masses_min = 1.0; // TODO
        println!("Vertex with the smallest mass ({} units)", masses_min);

        // The dihedral angle occurring between the facets at each crease,
        // in the previous frame
        let last_angles = vec![0.0; edges.len()];

        let mut target_angles = vec![];
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
        }

        // Some extra sanity checks
        assert_eq!(vertices.len(), masses.len());
        assert_eq!(edges.len(), assignments.len());
        assert_eq!(edges.len(), target_angles.len());

        Model {
            vertices,
            edges,
            faces,
            assignments,
            ..Default::default()
        }
    }

    fn update(&mut self) {
        let mut updated_normals = vec![];
        let mut updated_areas = vec![];

        for face_indices in self.faces.iter() {
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

            updated_normals.push(normal);
            updated_areas.push(area);

            // To draw the normals, create a line segment from `centroid` to the
            // point `centroid + normal`
            // ...
        }

        // Move
        self.normals = updated_normals;
        self.areas = updated_areas;
    }

    fn step_simulation(&mut self) {
        // First, calculate new face normals / areas
        self.update();

        // Then, step the physics simulation
        let k_facet = 0.7;
        let k_fold = 0.7;
        let ea = 20.0;
        let zeta = 0.1;
        let iterations = 1;

        for _ in 0..iterations {}
    }

    fn get_vertex(&self, index: usize) -> &Vertex {
        &self.vertices[index]
    }
}

fn main() {
    let spec = FoldSpecification::from_file(Path::new("bird_base.fold")).unwrap();
    let model = Model::from_specification(&spec);
}
