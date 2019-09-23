use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
pub struct FoldSpecification {
    #[serde(default)]
    pub file_spec: i32,

    #[serde(default)]
    pub file_creator: String,

    #[serde(default)]
    pub file_author: String,

    #[serde(default)]
    pub frame_title: String,

    #[serde(default)]
    pub frame_classes: Vec<String>,

    #[serde(default)]
    pub frame_attributes: Vec<String>,

    #[serde(default)]
    pub frame_unit: String,

    #[serde(rename(deserialize = "vertices_coords"))]
    pub vertices: Vec<Vec<f32>>,

    #[serde(rename(deserialize = "edges_vertices"))]
    pub edges: Vec<Vec<i32>>,

    #[serde(rename(deserialize = "edges_assignment"))]
    pub assignments: Vec<String>,

    #[serde(rename(deserialize = "faces_vertices"))]
    pub faces: Vec<Vec<i32>>,

    // Renamed and removed all occurrences of `null`
    #[serde(rename(deserialize = "edges_foldAngles"))]
    pub fold_angles: Vec<f32>,
}

impl FoldSpecification {
    pub fn from_file(path: &Path) -> std::io::Result<FoldSpecification> {
        let mut file = File::open(path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        Ok(serde_json::from_str(&contents).expect("Failed to load JSON"))
    }
}
