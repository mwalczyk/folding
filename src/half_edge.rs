use cgmath::Vector3;

pub type EdgeId = usize;
pub type VertexId = usize;
pub type FaceId = usize;

fn get_edge_indices_for_face_indices(face_indices: &[usize; 3]) -> [[usize; 2]; 3] {
    [
        [face_indices[0], face_indices[1]], // 1st edge
        [face_indices[1], face_indices[2]], // 2nd edge
        [face_indices[2], face_indices[0]], // 3rd edge
    ]
}

/// A simple half-edge data structure.
///
/// Note that face indices are assumed to be in CCW winding
/// order.
///
/// Reference: `http://kaba.hilvi.org/homepage/blog/halfedge/halfedge.htm`
#[derive(Default)]
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

pub struct Vertex {
    pub coordinates: Vector3<f32>,
    pub half_edge_id: usize,
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

pub struct HalfEdgeMesh {
    half_edges: Vec<HalfEdge>,
    vertices: Vec<Vertex>,
}

impl HalfEdgeMesh {
    /// Reference: `https://stackoverflow.com/questions/15365471/initializing-half-edge-data-structure-from-vertices`
    pub fn from_faces(faces: &Vec<[usize; 3]>) {
        // Build a list of half-edges from the given face description
        let mut half_edges = vec![];

        for (face_index, face_indices) in faces.iter().enumerate() {

            // Create a new half-edge for each edge of this face
            for (edge_index, edge_indices) in get_edge_indices_for_face_indices(face_indices)
                .iter()
                .enumerate()
            {
                let mut half_edge = HalfEdge::new();

                // `face_index` ranges from 0 -> number of faces in the mesh
                // `edge_index` loops between 0, 1, 2
                //
                // `current_index` (below) is the index of this new half-edge within the parent
                // `HalfEdgeMesh`'s data store
                let _current_id = face_index + edge_index;

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

                // Make sure to add per-face offsets
                half_edge.prev_id = if edge_index == 0 { 2 + face_index } else { edge_index - 1 + face_index };
                half_edge.next_id = (edge_index + 1) % 3 + face_index;
                half_edge.pair_id = 0;
                half_edge.origin_vertex_id = edge_indices[0];
                half_edge.face_id = 0;

                // Add this half-edge to the mesh
                half_edges.push(half_edge);
            }
        }

        // All that's left is to set each half-edge's `pair_id` indices
        for (half_edge_index, half_edge) in half_edges.iter_mut().enumerate() {

            // Find the index of the half-edge whose origin vertex index is the same as this one's,
            // then get its previous half-edge
            for (other_index, other_edge) in half_edges.iter() {
                if other_index != half_edge_index && other_edge.origin_vertex_id == half_edge.origin_vertex_id {
                    // This is `half_edge`'s pair
                    half_edge.pair_id = other_edge.prev_id;
                }
            }
        }
    }

    pub fn get_edges_connected_to_vertex(&self, vertex_index: usize) {
        unimplemented!();
    }

    pub fn get_faces_connected_to_edge(&self, edge_index: usize) {
        unimplemented!();
    }
}
