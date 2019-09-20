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
pub struct HalfEdgeIndex(pub usize);
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct VertexIndex(pub usize);
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct FaceIndex(pub usize);

/// A simple half-edge data structure.
///
/// Note that face indices are assumed to be in CCW winding
/// order.
///
/// Reference: `http://kaba.hilvi.org/homepage/blog/halfedge/halfedge.htm`
#[derive(Clone, Copy, Debug)]
pub struct HalfEdge {
    // The ID of the previous half-edge
    prev_id: HalfEdgeIndex,

    // The ID of the next half-edge
    next_id: HalfEdgeIndex,

    // The ID of the half-edge that runs parallel to this one but in the opposite direction
    pair_id: HalfEdgeIndex,

    // The ID of the vertex from which this half-edge originates
    origin_vertex_id: VertexIndex,

    // The ID of the face that is adjacent to this half-edge
    face_id: Option<FaceIndex>,
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

    pub fn get_prev(&self) -> HalfEdgeIndex {
        self.prev_id
    }

    pub fn get_next(&self) -> HalfEdgeIndex {
        self.next_id
    }

    pub fn get_pair(&self) -> HalfEdgeIndex {
        self.pair_id
    }

    pub fn get_origin_vertex(&self) -> VertexIndex {
        self.origin_vertex_id
    }

    pub fn get_face(&self) -> Option<FaceIndex> {
        self.face_id
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    // The XYZ-coordinates of this vertex, in 3-space
    coordinates: Vector3<f32>,

    // The ID of the half-edge that this vertex "belongs" to (i.e. is the origin vertex of)
    half_edge_id: HalfEdgeIndex,
}

impl Vertex {
    pub fn new(coordinates: Vector3<f32>, half_edge_id: HalfEdgeIndex) -> Vertex {
        Vertex {
            coordinates,
            half_edge_id,
        }
    }

    pub fn get_coordinates(&self) -> &Vector3<f32> {
        &self.coordinates
    }

    pub fn set_coordinates(&mut self, coords: &Vector3<f32>) {
        self.coordinates = *coords;
    }

    pub fn get_half_edge(&self) -> HalfEdgeIndex {
        self.half_edge_id
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Face {
    // The ID of one of the 3 half-edges that surround this face
    half_edge_id: HalfEdgeIndex,
}

impl Face {
    pub fn new(half_edge_id: HalfEdgeIndex) -> Face {
        Face { half_edge_id }
    }

    pub fn get_half_edge(&self) -> HalfEdgeIndex {
        self.half_edge_id
    }
}

#[derive(Clone)]
pub struct HalfEdgeMesh {
    // The half-edges of this mesh
    half_edges: Vec<HalfEdge>,

    // The vertices of this mesh
    vertices: Vec<Vertex>,

    // The (triangular) faces of this mesh
    faces: Vec<Face>,
}

impl HalfEdgeMesh {
    /// Build a list of half-edges from the given face description.
    ///
    /// Reference: `https://stackoverflow.com/questions/15365471/initializing-half-edge-data-structure-from-vertices`
    pub fn from_faces(
        base_faces: &Vec<[usize; 3]>,
        base_vertices: &Vec<Vector3<f32>>,
    ) -> Result<HalfEdgeMesh, &'static str> {
        println!(
            "Building half-edges from {} faces and {} vertices",
            base_faces.len(),
            base_vertices.len()
        );
        let mut half_edges = vec![];
        let mut vertices = vec![];
        let mut faces = vec![];

        for (face_index, face_indices) in base_faces.iter().enumerate() {
            println!("Examining face {:?}", face_indices);
            for (edge_index, edge_indices) in get_edge_indices_for_face_indices(face_indices)
                .iter()
                .enumerate()
            {
                // Create a new half-edge for each existing edge of this face (there should always be 3)
                println!("\tExamining edge {:?}", edge_indices);
                if edge_index > 2 {
                    return Err("Found face with more than 3 edges: triangulate this mesh before continuing");
                }

                // `face_index` ranges from 0 -> # of faces in the mesh
                // `edge_index` loops between 0, 1, 2 (assuming triangle faces)
                //
                // `current_id` is the index of the new half-edge in the list of half-edges (could
                // also be calculated as: `half_edges.len()`)
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

                // Add a new, tracked face to the mesh (only do this once per set of 3 half-edges)
                if edge_index == 0 {
                    faces.push(Face::new(HalfEdgeIndex(current_id)));
                }

                // Finally, add this half-edge to the mesh
                half_edges.push(half_edge);
            }
        }

        // Create vertices: we do this separately so that each vertex is only added once
        for (base_vertex_index, _base_vertex) in base_vertices.iter().enumerate() {
            for (half_edge_index, half_edge) in half_edges.iter().enumerate() {
                // Is this a half-edge that originates from this vertex?
                if base_vertex_index == half_edge.origin_vertex_id.0 {
                    vertices.push(Vertex::new(
                        base_vertices[half_edge.origin_vertex_id.0],
                        HalfEdgeIndex(half_edge_index),
                    ));
                    break;
                }
            }
        }
        // Some quick sanity checks
        assert_eq!(half_edges.len(), base_faces.len() * 3);
        assert_eq!(vertices.len(), base_vertices.len());
        assert_eq!(faces.len(), base_faces.len());

        // Now, find each half-edge's pair (or "twin" / "opposite"): if one is not found, this means
        // that the half-edge is on the border (i.e. boundary) of the mesh, and a new, "dummy" half-edge
        // will need to be created alongside it
        let mut border_edges = vec![];
        for i in 0..half_edges.len() {
            let mut found_pair = false;

            for j in 0..half_edges.len() {
                if i != j {
                    if half_edges[i].origin_vertex_id
                        == half_edges[half_edges[j].next_id.0].origin_vertex_id
                        && half_edges[half_edges[i].next_id.0].origin_vertex_id
                            == half_edges[j].origin_vertex_id
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
                let mut border_edge = HalfEdge::new();

                border_edge.prev_id = HalfEdgeIndex(0); // This will be set later
                border_edge.next_id = HalfEdgeIndex(0); // This will be set later
                border_edge.pair_id = HalfEdgeIndex(i);
                border_edge.origin_vertex_id = half_edges[half_edges[i].next_id.0].origin_vertex_id;
                border_edge.face_id = None;

                // Set the half-edge's pair to the newly created border half-edge
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
                    if half_edges[border_edges[i].pair_id.0].origin_vertex_id
                        == border_edges[j].origin_vertex_id
                    {
                        // The vertex from which this half-edge's pair originates from is the same as the other one
                        border_edges[i].next_id = HalfEdgeIndex(half_edges.len() + j);
                        found_next = true;
                    } else if border_edges[i].origin_vertex_id
                        == half_edges[border_edges[j].pair_id.0].origin_vertex_id
                    {
                        // The vertex from which this half-edge originates from is the same as the other one's pair
                        border_edges[i].prev_id = HalfEdgeIndex(half_edges.len() + j);
                        found_prev = true;
                    }
                }
            }
            if !found_next || !found_prev {
                return Err("Couldn't find next (or maybe, previous) half-edge corresponding to one or more border half-edges");
            }
        }
        half_edges.extend_from_slice(&border_edges);

        Ok(HalfEdgeMesh {
            half_edges,
            vertices,
            faces,
        })
    }

    /// This function is primarily used to render the half-edge mesh (as in an OpenGL program).
    pub fn gather_triangles(&self) -> Vec<Vector3<f32>> {
        let mut triangles = vec![];

        for (i, &_) in self.faces.iter().enumerate() {
            let indices = self.get_adjacent_vertices_to_face(FaceIndex(i));
            let coordinates: Vec<_> = indices
                .iter()
                .map(|&i| self.get_vertex(i).coordinates)
                .collect();

            triangles.extend_from_slice(&coordinates);
        }

        triangles
    }

    pub fn gather_lines(&self) -> Vec<Vector3<f32>> {
        let mut lines = vec![];

        for i in 0..self.half_edges.len() {
            for j in self
                .get_adjacent_vertices_to_half_edge(HalfEdgeIndex(i))
                .iter()
            {
                lines.push(self.get_vertex(*j).coordinates);
            }
        }

        lines
    }

    //    /// Returns the vertex indices corresponding to each half-edge as a list. This function is primarily
    //    /// used to render the half-edge mesh (as in an OpenGL program).
    //    pub fn gather_edge_indices(&self) -> Vec<[VertexIndex; 2]> {
    //        self.half_edges
    //            .iter()
    //            .enumerate()
    //            .map(|(i, &he)| self.get_adjacent_vertices_to_half_edge(HalfEdgeIndex(i)))
    //            .collect()
    //    }
    //
    //    /// Returns the coordinates of all of the vertices that make up this half-edge mesh as a list. This function
    //    /// is primarily used to render the half-edge mesh (as in an OpenGL program).
    //    pub fn gather_vertex_coordinates(&self) -> Vec<Vector3<f32>> {
    //        self.vertices.iter().map(|&v| v.coordinates).collect()
    //    }

    /// Returns `true` if the half-edge at `index` is along the border (i.e. is a "dummy"
    /// half-edge) and `false` otherwise.
    pub fn is_border_half_edge(&self, index: HalfEdgeIndex) -> bool {
        if let None = self.get_half_edge(index).face_id {
            return true;
        }
        false
    }

    /// Returns an immutable reference to all of the half-edges that make up this mesh.
    pub fn get_half_edges(&self) -> &Vec<HalfEdge> {
        &self.half_edges
    }

    /// Returns an immutable reference to all of the half-edges that make up this mesh.
    pub fn get_half_edges_mut(&mut self) -> &mut Vec<HalfEdge> {
        &mut self.half_edges
    }

    /// Returns an immutable reference to all of the vertices that make up this mesh.
    pub fn get_vertices(&self) -> &Vec<Vertex> {
        &self.vertices
    }

    /// Returns a mutable reference to all of the vertices that make up this mesh.
    pub fn get_vertices_mut(&mut self) -> &mut Vec<Vertex> {
        &mut self.vertices
    }

    /// Returns an immutable reference to all of the faces that make up this mesh.
    pub fn get_faces(&self) -> &Vec<Face> {
        &self.faces
    }

    /// Returns a mutable reference to all of the faces that make up this mesh.
    pub fn get_faces_mut(&mut self) -> &mut Vec<Face> {
        &mut self.faces
    }

    /// Returns an immutable reference to the half-edge at `index`.
    pub fn get_half_edge(&self, index: HalfEdgeIndex) -> &HalfEdge {
        &self.half_edges[index.0]
    }

    /// Returns a mutable reference to the half-edge at `index`.
    pub fn get_half_edge_mut(&mut self, index: HalfEdgeIndex) -> &mut HalfEdge {
        &mut self.half_edges[index.0]
    }

    /// Returns an immutable reference to the vertex at `index`.
    pub fn get_vertex(&self, index: VertexIndex) -> &Vertex {
        &self.vertices[index.0]
    }

    /// Returns a mutable reference to the vertex at `index`.
    pub fn get_vertex_mut(&mut self, index: VertexIndex) -> &mut Vertex {
        &mut self.vertices[index.0]
    }

    /// Returns an immutable reference to the face at `index`.
    pub fn get_face(&self, index: FaceIndex) -> &Face {
        &self.faces[index.0]
    }

    /// Returns a mutable reference to the face at `index`.
    pub fn get_face_mut(&mut self, index: FaceIndex) -> &mut Face {
        &mut self.faces[index.0]
    }

    /// Returns the index of the half-edge that joins the vertices at `index_0` and `index_1` or `None`
    /// if such a half-edge doesn't exist. Note that in many cases, there will be more than one half-edge
    /// that joins the two vertices (since every half-edge also has a pair that runs parallel). In this
    /// case, the index of the first half-edge found is returned.
    pub fn find_half_edge_between_vertices(
        &self,
        index_0: VertexIndex,
        index_1: VertexIndex,
    ) -> Option<HalfEdgeIndex> {
        for i in self.get_adjacent_half_edges_to_vertex(index_0).iter() {
            // Is the second vertex at the other end of this half-edge?
            if self.get_terminating_vertex_along_half_edge(*i) == index_1 {
                return Some(*i);
            }
        }
        None
    }

    /// Returns the index of the vertex opposite this half-edge's origin (i.e. where the half-edge
    /// ends).
    pub fn get_terminating_vertex_along_half_edge(&self, index: HalfEdgeIndex) -> VertexIndex {
        self.get_half_edge(self.get_half_edge(index).pair_id)
            .origin_vertex_id
    }

    /// Returns the indices of the two vertices that are adjacent to this half-edge (
    /// i.e. the two vertices that form this particular edge).
    ///
    /// Reference: `http://www.sccg.sk/~samuelcik/dgs/half_edge.pdf`
    pub fn get_adjacent_vertices_to_half_edge(&self, index: HalfEdgeIndex) -> [VertexIndex; 2] {
        // Returns:
        // [0] edge -> vertex
        // [1] edge -> pair -> vertex
        [
            self.get_half_edge(index).origin_vertex_id,
            self.get_half_edge(self.get_half_edge(index).pair_id)
                .origin_vertex_id,
        ]
    }

    /// Returns the indices of the two faces that are adjacent to this half-edge.
    pub fn get_adjacent_faces_to_half_edge(&self, index: HalfEdgeIndex) -> [Option<FaceIndex>; 2] {
        // Returns:
        // [0] edge -> face
        // [1] edge -> pair -> face
        [
            self.get_half_edge(index).face_id,
            self.get_half_edge(self.get_half_edge(index).pair_id)
                .face_id,
        ]
    }

    /// Returns the indices of all half-edges that originate from the vertex at `index`.
    pub fn get_adjacent_half_edges_to_vertex(&self, index: VertexIndex) -> Vec<HalfEdgeIndex> {
        let vertex = self.get_vertex(index);
        let start = vertex.half_edge_id;
        let mut current = start;

        let mut indices = vec![current];

        loop {
            // Get the current half-edge's pair's next
            current = self
                .get_half_edge(self.get_half_edge(current).pair_id)
                .next_id;
            if current == start {
                break;
            }

            indices.push(current);
        }

        indices
    }

    /// Returns the indices of all half-edges that bound (i.e. surround) the face at `index`.
    /// The half-edge indices returned are guaranteed to be in a counter-clockwise winding order,
    /// but there are no other guarantees about the order of the indices (i.e. they may not be
    /// sorted from lowest to highest index, for example).
    pub fn get_adjacent_half_edges_to_face(&self, index: FaceIndex) -> Vec<HalfEdgeIndex> {
        let face = self.get_face(index);
        let start = face.half_edge_id;
        let mut current = start;

        let mut indices = vec![current];

        loop {
            // Get the current half-edge's next
            current = self.get_half_edge(current).next_id;
            if current == start {
                break;
            }

            indices.push(current);
        }

        indices
    }

    /// Returns the indices of all of the vertices that are 1 edge-length away from the vertex at `index`
    /// (i.e. neighboring vertices).
    pub fn get_adjacent_vertices_to_vertex(&self, index: VertexIndex) -> Vec<VertexIndex> {
        let half_edges = self.get_adjacent_half_edges_to_vertex(index);

        // Below: edge -> pair -> origin
        half_edges
            .iter()
            .map(|&i| self.get_terminating_vertex_along_half_edge(i))
            .collect()
    }

    /// Returns the indices of all of the vertices that bound (i.e. surround) the face at `index`.
    pub fn get_adjacent_vertices_to_face(&self, index: FaceIndex) -> Vec<VertexIndex> {
        let half_edges = self.get_adjacent_half_edges_to_face(index);

        half_edges
            .iter()
            .map(|&i| self.get_half_edge(i).origin_vertex_id)
            .collect()
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
            write!(
                f,
                "\tEnds at: {:?}\n",
                self.get_half_edge(half_edge.pair_id).origin_vertex_id
            )?;
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

        // These don't really matter for testing...
        let base_vertices = vec![
            Vector3::new(0.0, 1.0, 0.0), // Vertex #0
            Vector3::new(0.0, 0.0, 0.0), // Vertex #1
            Vector3::new(1.0, 0.0, 0.0), // Vertex #2
            Vector3::new(1.0, 1.0, 0.0), // Vertex #3
        ];

        // There should be 10 half-edges total: 3 for each interior face (of which there are 2) plus
        // 4 for the border (boundary) half-edges
        let half_edge_mesh = HalfEdgeMesh::from_faces(&base_faces, &base_vertices).unwrap();
        println!("{:?}", half_edge_mesh);

        // First, some basic tests
        assert_eq!(10, half_edge_mesh.half_edges.len());
        assert_eq!(4, half_edge_mesh.vertices.len());
        assert_eq!(2, half_edge_mesh.faces.len());

        // Test adjacency query on a particular vertex: note that we sort the array before any
        // assertions, since we don't know what order the indices will be in
        let mut adjacent_to_vertex_0 =
            half_edge_mesh.get_adjacent_half_edges_to_vertex(VertexIndex(0));
        let mut adjacent_to_vertex_1 =
            half_edge_mesh.get_adjacent_half_edges_to_vertex(VertexIndex(1));
        let mut adjacent_to_vertex_2 =
            half_edge_mesh.get_adjacent_half_edges_to_vertex(VertexIndex(2));
        let mut adjacent_to_vertex_3 =
            half_edge_mesh.get_adjacent_half_edges_to_vertex(VertexIndex(3));
        adjacent_to_vertex_0.sort();
        adjacent_to_vertex_1.sort();
        adjacent_to_vertex_2.sort();
        adjacent_to_vertex_3.sort();
        assert_eq!(
            vec![HalfEdgeIndex(0), HalfEdgeIndex(3), HalfEdgeIndex(9)],
            adjacent_to_vertex_0
        );
        assert_eq!(
            vec![HalfEdgeIndex(1), HalfEdgeIndex(6)],
            adjacent_to_vertex_1
        );
        assert_eq!(
            vec![HalfEdgeIndex(2), HalfEdgeIndex(4), HalfEdgeIndex(7)],
            adjacent_to_vertex_2
        );
        assert_eq!(
            vec![HalfEdgeIndex(5), HalfEdgeIndex(8)],
            adjacent_to_vertex_3
        );
        println!(
            "Half-edges adjacent to vertex #0: {:?}",
            adjacent_to_vertex_0
        );
        println!(
            "Half-edges adjacent to vertex #1: {:?}",
            adjacent_to_vertex_1
        );
        println!(
            "Half-edges adjacent to vertex #2: {:?}",
            adjacent_to_vertex_2
        );
        println!(
            "Half-edges adjacent to vertex #3: {:?}",
            adjacent_to_vertex_3
        );

        // Test adjacency queries on a particular vertex
        let mut neighbors_of_vertex_0 =
            half_edge_mesh.get_adjacent_vertices_to_vertex(VertexIndex(0));
        let mut neighbors_of_vertex_1 =
            half_edge_mesh.get_adjacent_vertices_to_vertex(VertexIndex(1));
        let mut neighbors_of_vertex_2 =
            half_edge_mesh.get_adjacent_vertices_to_vertex(VertexIndex(2));
        let mut neighbors_of_vertex_3 =
            half_edge_mesh.get_adjacent_vertices_to_vertex(VertexIndex(3));
        neighbors_of_vertex_0.sort();
        neighbors_of_vertex_1.sort();
        neighbors_of_vertex_2.sort();
        neighbors_of_vertex_3.sort();
        assert_eq!(
            vec![VertexIndex(1), VertexIndex(2), VertexIndex(3)],
            neighbors_of_vertex_0
        );
        assert_eq!(vec![VertexIndex(0), VertexIndex(2)], neighbors_of_vertex_1);
        assert_eq!(
            vec![VertexIndex(0), VertexIndex(1), VertexIndex(3)],
            neighbors_of_vertex_2
        );
        assert_eq!(vec![VertexIndex(0), VertexIndex(2)], neighbors_of_vertex_3);

        // Test adjacency queries on a particular face
        let mut adjacent_to_face_0 = half_edge_mesh.get_adjacent_half_edges_to_face(FaceIndex(0));
        let mut adjacent_to_face_1 = half_edge_mesh.get_adjacent_half_edges_to_face(FaceIndex(1));
        assert_eq!(
            vec![HalfEdgeIndex(0), HalfEdgeIndex(1), HalfEdgeIndex(2)],
            adjacent_to_face_0
        );
        assert_eq!(
            vec![HalfEdgeIndex(3), HalfEdgeIndex(4), HalfEdgeIndex(5)],
            adjacent_to_face_1
        );

        let mut vertices_around_face_0 = half_edge_mesh.get_adjacent_vertices_to_face(FaceIndex(0));
        let mut vertices_around_face_1 = half_edge_mesh.get_adjacent_vertices_to_face(FaceIndex(1));
        assert_eq!(
            vec![VertexIndex(0), VertexIndex(1), VertexIndex(2)],
            vertices_around_face_0
        );
        assert_eq!(
            vec![VertexIndex(0), VertexIndex(2), VertexIndex(3)],
            vertices_around_face_1
        );

        // Test faces along interior edges
        assert_eq!(
            Some(FaceIndex(0)),
            half_edge_mesh.get_half_edge(HalfEdgeIndex(0)).face_id
        );
        assert_eq!(
            Some(FaceIndex(0)),
            half_edge_mesh.get_half_edge(HalfEdgeIndex(1)).face_id
        );
        assert_eq!(
            Some(FaceIndex(0)),
            half_edge_mesh.get_half_edge(HalfEdgeIndex(2)).face_id
        );
        assert_eq!(
            Some(FaceIndex(1)),
            half_edge_mesh.get_half_edge(HalfEdgeIndex(3)).face_id
        );
        assert_eq!(
            Some(FaceIndex(1)),
            half_edge_mesh.get_half_edge(HalfEdgeIndex(4)).face_id
        );
        assert_eq!(
            Some(FaceIndex(1)),
            half_edge_mesh.get_half_edge(HalfEdgeIndex(5)).face_id
        );

        // Test faces along border edges
        assert_eq!(None, half_edge_mesh.get_half_edge(HalfEdgeIndex(6)).face_id);
        assert_eq!(None, half_edge_mesh.get_half_edge(HalfEdgeIndex(7)).face_id);
        assert_eq!(None, half_edge_mesh.get_half_edge(HalfEdgeIndex(8)).face_id);
        assert_eq!(None, half_edge_mesh.get_half_edge(HalfEdgeIndex(9)).face_id);

        // Test gather methods
        println!("\n{:?}", half_edge_mesh.gather_triangles());
    }
}
