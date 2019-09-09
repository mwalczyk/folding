extern crate serde;

use cgmath::{Vector3, Zero};
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
    fn from_file(path: &Path) -> std::io::Result<()> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let json: FoldSpecification = serde_json::from_str(&contents).expect("Failed to load JSON");
        println!("{:?}", json);

        Ok(())
    }
}

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
    fn from_specification(spec: &FoldSpecification) {
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
            assignments.push(assignment_str.parse::<Assignment>().unwrap());
        }
        assert_eq!(edges.len(), assignments.len());

        // Add faces, triangulated
        for face_indices in spec.faces.iter() {
            assert_eq!(face_indices.len(), 3);
            faces.push([
                face_indices[0] as usize,
                face_indices[1] as usize,
                face_indices[2] as usize,
            ]);
        }
    }

    fn step_simulation(&mut self) {}

    fn get_vertex(&self, index: usize) -> () {}
}

fn main() {
    let fold_specification = FoldSpecification::from_file(Path::new("bird_base.fold"));
}
