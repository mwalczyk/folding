use cgmath::Vector3;

/// Assuming a triangle mesh (i.e. one where all faces are triangles), each face
/// can be represented as a 3-tuple of vertex indices in CCW winding order. The
/// edges of this face, then, are the unique pairs of indices we generate as we traverse
/// the face in a CCW direction.
///
/// For example, say a particular face is specified by the vertex indices `[0, 2, 4]`.
/// This function would return an array of arrays:
///
/// ```
/// [
///     [0, 2],
///     [2, 4],
///     [4, 0],
/// ]
/// ```
fn get_edge_indices_for_face_indices(face_indices: &[usize; 3]) -> [[usize; 2]; 3] {
    // A (triangular) face has 3 unique edges:
    [
        [face_indices[0], face_indices[1]], // 1st edge
        [face_indices[1], face_indices[2]], // 2nd edge
        [face_indices[2], face_indices[0]], // 3rd edge
    ]
}

pub struct _HalfEdgeIndex {
    inner: usize,
}
pub struct _VertexIndex {
    inner: usize,
}
pub struct _FaceIndex {
    inner: usize,
}

/// A simple half-edge data structure.
///
/// Note that face indices are assumed to be in CCW winding
/// order.
///
/// Reference: `http://kaba.hilvi.org/homepage/blog/halfedge/halfedge.htm`
#[derive(Clone, Copy, Debug, Default)]
pub struct HalfEdge {
    // The ID of the previous half-edge
    pub prev_id: usize,

    // The ID of the next half-edge
    pub next_id: usize,

    // The ID of the half-edge that runs parallel to this one but in the opposite direction
    pub pair_id: usize,

    // The ID of the vertex from which this half-edge originates
    pub origin_vertex_id: usize,

    // The ID of the face that is adjacent to this half-edge
    pub face_id: usize,
}

impl HalfEdge {
    pub fn new() -> HalfEdge {
        HalfEdge {
            prev_id: 0,
            next_id: 0,
            pair_id: 0,
            origin_vertex_id: 0,
            face_id: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    // The XYZ-coordinates of this vertex, in 3-space
    pub coordinates: Vector3<f32>,

    // The ID of the half-edge that this vertex "belongs" to (i.e. is the origin vertex of)
    pub half_edge_id: usize,
}

impl Vertex {
    pub fn new(coordinates: Vector3<f32>, half_edge_id: usize) -> Vertex {
        Vertex {
            coordinates,
            half_edge_id,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Face {
    // The ID of one of the 3 half-edges that surround this face (the other 2 can be found
    // by simply following pointers between adjacent half-edges)
    pub half_edge_id: usize,
}

impl Face {
    pub fn new(half_edge_id: usize) -> Face {
        Face { half_edge_id }
    }
}

#[derive(Clone, Debug)]
pub struct HalfEdgeMesh {
    half_edges: Vec<HalfEdge>,
    vertices: Vec<Vertex>,
    faces: Vec<Face>,
}

impl HalfEdgeMesh {
    /// Build a list of half-edges from the given face description.
    ///
    /// Reference: `https://stackoverflow.com/questions/15365471/initializing-half-edge-data-structure-from-vertices`
    pub fn from_faces(base_faces: &Vec<[usize; 3]>, base_vertices: &Vec<Vector3<f32>>) -> HalfEdgeMesh {
        let mut half_edges = vec![];
        let mut vertices = vec![];
        let mut faces = vec![];

        for (face_index, face_indices) in base_faces.iter().enumerate() {
            // Create a new half-edge for each edge of this face
            for (edge_index, edge_indices) in get_edge_indices_for_face_indices(face_indices)
                .iter()
                .enumerate()
            {
                assert!(edge_index < 3);

                // `face_index` ranges from 0..number of faces in the mesh
                // `edge_index` loops between 0, 1, 2
                //
                // `current_index` (below) is the index of this new half-edge within the parent
                // `HalfEdgeMesh`'s data store
                let face_offset = face_index * 3;

                let current_id = edge_index + face_offset;

                let prev_id = if edge_index == 0 {
                    2 + face_offset
                } else {
                    (edge_index - 1) + face_offset
                };

                let next_id = (edge_index + 1) % 3 + face_offset;

                // CCW winding order show below:
                //
                // 1
                // |\
                // | \
                // |  \
                // |   \
                // |    \
                // |     \
                // 2------0
                let mut half_edge = HalfEdge::new();

                // Make sure to add per-face offsets
                half_edge.prev_id = prev_id;
                half_edge.next_id = next_id;
                half_edge.pair_id = 0;
                half_edge.origin_vertex_id = edge_indices[0];
                half_edge.face_id = 0;

                // Add a new, tracked vertex to the mesh
                vertices.push(Vertex::new(
                    base_vertices[half_edge.origin_vertex_id],
                    current_id,
                ));

                // Add a new, tracked face to the mesh (only do this once per set of 3 half-edges)
                if edge_index == 0 {
                    faces.push(Face::new(current_id));
                }

                // Finally, add this half-edge to the mesh
                half_edges.push(half_edge);
            }
        }
        println!("{:?}", vertices.len());

        // Some quick sanity checks
        assert_eq!(half_edges.len(), base_faces.len() * 3);
        assert_eq!(vertices.len(), base_faces.len() * 3);
        assert_eq!(faces.len(), base_faces.len());

        // We have to do this because of the borrow checker in the inner for-loop
        let half_edges_clone = half_edges.clone();

        // All that's left is to set each half-edge's `pair_id` indices
        for (half_edge_index, half_edge) in half_edges.iter_mut().enumerate() {
            let mut found = false;

            // Find the index of the half-edge whose origin vertex index is the same as this one's,
            // then get its previous half-edge
            for (other_index, other_edge) in half_edges_clone.iter().enumerate() {
                if other_index != half_edge_index
                    && other_edge.origin_vertex_id == half_edge.origin_vertex_id
                {
                    // This is `half_edge`'s pair
                    half_edge.pair_id = other_edge.prev_id;
                    found = true;
                    break;
                }
            }

            if !found {
                panic!("Pair half-edge not found for half-edge {:?}", half_edge);
            }
        }

        for (half_edge_index, half_edge) in half_edges.iter().enumerate() {
            println!("ID: {}, {:?}", half_edge_index, half_edge);
        }

        HalfEdgeMesh {
            half_edges,
            vertices,
            faces,
        }
    }

    pub fn get_edges_connected_to_vertex(&self, vertex_index: usize) {
        unimplemented!();
    }

    pub fn get_faces_connected_to_edge(&self, half_edge_index: usize) {
        unimplemented!();
    }

    pub fn get_both_edge_indices(&self, half_edge_index: usize) -> [usize; 2] {
        unimplemented!();
    }
}
