use cgmath::Vector3;
use core::fmt;

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

/// Define different "types" of indices using the `newtype` pattern.
///
/// Reference: `https://doc.rust-lang.org/1.0.0/style/features/types/newtype.html`
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct HalfEdgeIndex(usize);
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VertexIndex(usize);
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceIndex(usize);

/// A simple half-edge data structure.
///
/// Note that face indices are assumed to be in CCW winding
/// order.
///
/// Reference: `http://kaba.hilvi.org/homepage/blog/halfedge/halfedge.htm`
#[derive(Clone, Copy, Debug)]
pub struct HalfEdge {
    // The ID of the previous half-edge
    pub prev_id: HalfEdgeIndex,

    // The ID of the next half-edge
    pub next_id: HalfEdgeIndex,

    // The ID of the half-edge that runs parallel to this one but in the opposite direction
    pub pair_id: HalfEdgeIndex,

    // The ID of the vertex from which this half-edge originates
    pub origin_vertex_id: VertexIndex,

    // The ID of the face that is adjacent to this half-edge
    pub face_id: Option<FaceIndex>,
}

impl HalfEdge {
    pub fn new() -> HalfEdge {
        HalfEdge {
            prev_id: HalfEdgeIndex(0),
            next_id: HalfEdgeIndex(0),
            pair_id: HalfEdgeIndex(0),
            origin_vertex_id: VertexIndex(0),
            face_id: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    // The XYZ-coordinates of this vertex, in 3-space
    pub coordinates: Vector3<f32>,

    // The ID of the half-edge that this vertex "belongs" to (i.e. is the origin vertex of)
    pub half_edge_id: HalfEdgeIndex,
}

impl Vertex {
    pub fn new(coordinates: Vector3<f32>, half_edge_id: HalfEdgeIndex) -> Vertex {
        Vertex {
            coordinates,
            half_edge_id,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Face {
    // The ID of one of the 3 half-edges that surround this face
    pub half_edge_id: HalfEdgeIndex,
}

impl Face {
    pub fn new(half_edge_id: HalfEdgeIndex) -> Face {
        Face { half_edge_id }
    }
}

#[derive(Clone)]
pub struct HalfEdgeMesh {
    half_edges: Vec<HalfEdge>,
    vertices: Vec<Vertex>,
    faces: Vec<Face>,
}

impl HalfEdgeMesh {
    /// Build a list of half-edges from the given face description.
    ///
    /// Reference: `https://stackoverflow.com/questions/15365471/initializing-half-edge-data-structure-from-vertices`
    pub fn from_faces(
        base_faces: &Vec<[usize; 3]>,
        base_vertices: &Vec<Vector3<f32>>,
    ) -> HalfEdgeMesh {
        println!("Building half-edges from {} faces and {} vertices", base_faces.len(), base_vertices.len());
        let mut half_edges = vec![];
        let mut vertices = vec![];
        let mut faces = vec![];

        for (face_index, face_indices) in base_faces.iter().enumerate() {
            for (edge_index, edge_indices) in get_edge_indices_for_face_indices(face_indices)
                .iter()
                .enumerate()
            {
                // Create a new half-edge for each existing edge of this face (there should always be 3)
                println!("Examining edge {:?}", edge_indices);
                assert!(edge_index < 3);

                // `face_index` ranges from 0 -> # of faces in the mesh
                // `edge_index` loops between 0, 1, 2 (assuming triangle faces)
                //
                // `current_id` is the index of the new half-edge in the list of half-edges
                let face_offset = face_index * 3;

                let current_id = edge_index + face_offset;
                assert_eq!(current_id, half_edges.len());

                let prev_id = if edge_index == 0 {
                    2 + face_offset
                } else {
                    (edge_index - 1) + face_offset
                };

                let next_id = (edge_index + 1) % 3 + face_offset;

                // CCW winding order show below:
                //
                // 0
                // |\
                // | \
                // |  \
                // 1---2
                let mut half_edge = HalfEdge::new();

                // Make sure to add per-face offsets
                half_edge.prev_id = HalfEdgeIndex(prev_id);
                half_edge.next_id = HalfEdgeIndex(next_id);
                half_edge.pair_id = HalfEdgeIndex(0); // This will be set below
                half_edge.origin_vertex_id = VertexIndex(edge_indices[0]);
                half_edge.face_id = Some(FaceIndex(face_index));

                // TODO: vertices shouldn't be added more than once...

                // Add a new, tracked vertex to the mesh
                vertices.push(Vertex::new(
                    base_vertices[half_edge.origin_vertex_id.0],
                    HalfEdgeIndex(current_id),
                ));

                // Add a new, tracked face to the mesh (only do this once per set of 3 half-edges)
                if edge_index == 0 {
                    faces.push(Face::new(HalfEdgeIndex(current_id)));
                }

                // Finally, add this half-edge to the mesh
                half_edges.push(half_edge);
            }
        }

        // Some quick sanity checks
        assert_eq!(half_edges.len(), base_faces.len() * 3);
        assert_eq!(vertices.len(), base_faces.len() * 3);
        assert_eq!(faces.len(), base_faces.len());

        let mut border_edges = vec![];

        // Now, find each half-edge's pair (or "twin" / "opposite"): if one is not found, this means
        // that the half-edge is on the border (i.e. boundary) of the mesh, and a new, "dummy" half-edge
        // will need to be created alongside it
        for i in 0..half_edges.len() {
            let mut found_pair = false;

            for j in 0..half_edges.len() {

                if i != j {

                    if half_edges[i].origin_vertex_id == half_edges[half_edges[j].next_id.0].origin_vertex_id &&
                        half_edges[half_edges[i].next_id.0].origin_vertex_id == half_edges[j].origin_vertex_id
                    {
                        // The vertex from which this half-edge originates from is the same as the other one's next
                        // (the conditions above should uniquely identify this edge)
                        half_edges[i].pair_id = HalfEdgeIndex(j);
                        found_pair = true;
                        break;
                    }
                }
            }

            if !found_pair {
                // This must be a border edge: create a new, "dummy" pair edge, whose
                // face pointer is null, i.e. `None`
                let mut border_edge = HalfEdge::new();

                border_edge.prev_id = HalfEdgeIndex(0); // This will be set later
                border_edge.next_id = HalfEdgeIndex(0); // This will be set later
                border_edge.pair_id = HalfEdgeIndex(i);
                border_edge.origin_vertex_id = half_edges[half_edges[i].next_id.0].origin_vertex_id;
                border_edge.face_id = None;

                half_edges[i].pair_id = HalfEdgeIndex(half_edges.len() + border_edges.len());

                border_edges.push(border_edge);
            }
        }

        println!("\n{} border edges found in total\n", border_edges.len());

        // Now, assign next / previous pointers for the newly created half-edges along the border
        for i in 0..border_edges.len() {
            let mut found_next = false;
            let mut found_prev = false;

            for j in 0..border_edges.len() {
                if i != j {
                    if half_edges[border_edges[i].pair_id.0].origin_vertex_id == border_edges[j].origin_vertex_id
                    {
                        // The vertex from which this half-edge's pair originates from is the same as the other one
                        border_edges[i].next_id = HalfEdgeIndex(half_edges.len() + j);
                        found_next = true;
                    } else if border_edges[i].origin_vertex_id == half_edges[border_edges[j].pair_id.0].origin_vertex_id
                    {
                        // The vertex from which this half-edge originates from is the same as the other one's pair
                        border_edges[i].prev_id = HalfEdgeIndex(half_edges.len() + j);
                        found_prev = true;
                    }
                }
            }

            if !found_next || !found_prev {
                panic!("Could not find next (or maybe, previous) half-edge corresponding to border half-edge #{}", i);
            }
        }
        half_edges.extend_from_slice(&border_edges);

        HalfEdgeMesh {
            half_edges,
            vertices,
            faces,
        }
    }

    /// Returns the half-edge at `index`.
    pub fn get_half_edge(&self, index: HalfEdgeIndex) -> &HalfEdge {
        &self.half_edges[index.0]
    }

    /// Returns the vertex at `index`.
    pub fn get_vertex(&self, index: VertexIndex) -> &Vertex {
        &self.vertices[index.0]
    }

    /// Returns the face at `index`.
    pub fn get_face(&self, index: FaceIndex) -> &Face {
        &self.faces[index.0]
    }

    /// Returns the indices of the two vertices that are adjacent to this half-edge (
    /// i.e. the two vertices that make this edge).
    ///
    /// Reference: `http://www.sccg.sk/~samuelcik/dgs/half_edge.pdf`
    pub fn get_adjacent_vertices(&self, index: HalfEdgeIndex) -> [VertexIndex; 2] {
        // Returns:
        // [0] edge -> vertex
        // [1] edge -> pair -> vertex
        [
            self.get_half_edge(index).origin_vertex_id,
            self.get_half_edge(self.get_half_edge(index).pair_id).origin_vertex_id,
        ]
    }

    /// Returns the indices of the two faces that are adjacent to this half-edge.
    pub fn get_adjacent_faces(&self, index: HalfEdgeIndex) -> [Option<FaceIndex>; 2] {
        // Returns:
        // [0] edge -> face
        // [1] edge -> pair -> face
        [
            self.get_half_edge(index).face_id,
            self.get_half_edge(self.get_half_edge(index).pair_id).face_id,
        ]
    }

    /// Returns the indices of all half-edges that originate from the vertex at `index`.
    pub fn get_all_edges_adjacent_to(&self, index: VertexIndex) -> Vec<HalfEdgeIndex> {
        let vertex = self.get_vertex(index);
        let start = vertex.half_edge_id;
        let mut current = start;

        let mut indices = vec![current];

        loop {
            // Get the current half-edge's pair's next
            current = self.get_half_edge(self.get_half_edge(current).pair_id).next_id;
            if current == start {
                break;
            }

            indices.push(current);
        }

        indices
    }

    pub fn get_all_faces_adjacent_to(&self, index: HalfEdgeIndex) {
        unimplemented!();
    }
}


impl fmt::Debug for HalfEdgeMesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (half_edge_index, half_edge) in self.half_edges.iter().enumerate() {
            if let None = half_edge.face_id {
                write!(f, "Half-edge (BORDER) #{}:\n", half_edge_index)?;
            } else {
                write!(f, "Half-edge #{}:\n", half_edge_index)?;
            }
            write!(f, "\tStarts at: {:?}\n", half_edge.origin_vertex_id)?;
            write!(f, "\tEnds at: {:?}\n", self.get_half_edge(half_edge.pair_id).origin_vertex_id)?;
            write!(f, "\tPair half-edge: {:?}\n", half_edge.pair_id)?;
            write!(f, "\tPrevious half-edge: {:?}\n", half_edge.prev_id)?;
            write!(f, "\tNext half-edge: {:?}\n", half_edge.next_id)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_faces() {
        // 0 -- 3
        // | \  |
        // |  \ |
        // 1 -- 2
        let base_faces = vec![
            [0, 1, 2], // 1st triangle
            [0, 2, 3], // 2nd triangle
        ];

        let base_vertices = vec![
            Vector3::new(-1.0, 1.0, 0.0),  // Vertex #0
            Vector3::new(-1.0, -1.0, 0.0), // Vertex #1
            Vector3::new(1.0, -1.0, 0.0),  // Vertex #2
            Vector3::new(1.0, 1.0, 0.0),   // Vertex #3
        ];

        // There should be 10 half-edges total: 3 for each interior face (of which there are 2) plus
        // 4 for the border (boundary) half-edges
        let half_edge_mesh = HalfEdgeMesh::from_faces(&base_faces, &base_vertices);
        assert_eq!(10, half_edge_mesh.half_edges.len());
        println!("{:?}", half_edge_mesh);

        // Test adjacency query on a particular vertex: note that we sort the array before any
        // assertions, since we don't know what order the indices will be in
        let mut adjacent_0 = half_edge_mesh.get_all_edges_adjacent_to(VertexIndex(0));
        let mut adjacent_1 = half_edge_mesh.get_all_edges_adjacent_to(VertexIndex(1));
        let mut adjacent_2 = half_edge_mesh.get_all_edges_adjacent_to(VertexIndex(2));
        let mut adjacent_3 = half_edge_mesh.get_all_edges_adjacent_to(VertexIndex(3));
        adjacent_0.sort();
        adjacent_1.sort();
        adjacent_2.sort();
        adjacent_3.sort();
        assert_eq!(vec![HalfEdgeIndex(0), HalfEdgeIndex(3), HalfEdgeIndex(9)], adjacent_0);
        assert_eq!(vec![HalfEdgeIndex(1), HalfEdgeIndex(6)], adjacent_1);
        assert_eq!(vec![HalfEdgeIndex(2), HalfEdgeIndex(4), HalfEdgeIndex(7)], adjacent_2);
        assert_eq!(vec![HalfEdgeIndex(5), HalfEdgeIndex(8)], adjacent_3);

        println!("Half-edges adjacent to vertex #0: {:?}", adjacent_0);
        println!("Half-edges adjacent to vertex #1: {:?}", adjacent_1);
        println!("Half-edges adjacent to vertex #2: {:?}", adjacent_2);
        println!("Half-edges adjacent to vertex #3: {:?}", adjacent_3);
    }
}
