use cgmath::{Vector2, Vector3, Zero};
use gl;
use gl::types::*;
use std::mem;

enum Attribute {
    POSITIONS,
    COLORS,
    NORMALS,
    TEXCOORDS,
}

impl Attribute {
    pub fn get_index(&self) -> u32 {
        match *self {
            Attribute::POSITIONS => 0,
            Attribute::COLORS => 1,
            Attribute::NORMALS => 2,
            Attribute::TEXCOORDS => 3,
        }
    }

    pub fn get_memory_size(&self) -> usize {
        match *self {
            Attribute::TEXCOORDS => mem::size_of::<Vector2<f32>>(),
            _ => mem::size_of::<Vector3<f32>>(),
        }
    }

    pub fn get_relative_offset(&self) -> usize {
        self.get_index() as usize * mem::size_of::<Vector3<f32>>()
    }

    pub fn get_element_count(&self) -> i32 {
        match *self {
            Attribute::TEXCOORDS => 2,
            _ => 3,
        }
    }
}

/// A simple implementation of a GPU-side mesh.
#[derive(Default)]
pub struct Mesh {
    // The vertex array object (VAO)
    vao: u32,

    // The vertex buffer object (VBO)
    vbo: u32,

    // All of the vertex data packed into a single, CPU-side array
    vertex_data: Vec<f32>,

    // The positions of the mesh
    positions: Vec<Vector3<f32>>,

    // The colors of the mesh
    colors: Option<Vec<Vector3<f32>>>,

    // The normals of the mesh
    normals: Option<Vec<Vector3<f32>>>,

    // The UV-coordinates (texture coordinates) of the mesh
    texcoords: Option<Vec<Vector2<f32>>>,
}

impl Mesh {
    pub fn new(
        positions: &Vec<Vector3<f32>>,
        colors: Option<&Vec<Vector3<f32>>>,
        normals: Option<&Vec<Vector3<f32>>>,
        texcoords: Option<&Vec<Vector2<f32>>>,
    ) -> Result<Mesh, &'static str> {
        // TODO: this seems silly...glium just has a series of bools like `has_colors`
        let colors = match colors {
            Some(data) => Some(data.clone()),
            _ => None,
        };

        let normals = match normals {
            Some(data) => Some(data.clone()),
            _ => None,
        };

        let texcoords = match texcoords {
            Some(data) => Some(data.clone()),
            _ => None,
        };

        let mut mesh = Mesh {
            vao: 0,
            vbo: 0,
            vertex_data: vec![],
            positions: positions.clone(),
            colors,
            normals,
            texcoords,
        };

        mesh.allocate()?;
        Ok(mesh)
    }

    /// Returns the bounds of this mesh. In particular, it returns a tuple
    /// with 3 elements:
    ///
    /// 1. The minimum point of the mesh (along each axis)
    /// 2. The maximum point of the mesh (along each axis)
    /// 3. The centroid of the mesh
    pub fn get_bounds(&self) -> (Vector3<f32>, Vector3<f32>, Vector3<f32>){

        let mut min = Vector3::zero();
        let mut max = Vector3::zero();
        let mut centroid = Vector3::zero();

        for position in self.positions.iter() {
            if position.x < min.x {
                min.x = position.x;
            }
            if position.y < min.y {
                min.y = position.y;
            }
            if position.z < min.z {
                min.z = position.z;
            }

            if position.x > max.x {
                max.x = position.x;
            }
            if position.y > max.y {
                max.y = position.y;
            }
            if position.z > max.z {
                max.z = position.z;
            }

            centroid += *position;
        }

        centroid /= self.get_number_of_vertices() as f32;

        (min, max, centroid)
    }

    /// Allocates all OpenGL objects necessary for rendering this mesh.
    fn allocate(&mut self) -> Result<(), &'static str> {
        unsafe {
            // First, initialize the vertex array object
            gl::CreateVertexArrays(1, &mut self.vao);

            // Enable the `0`th attribute (positions), which is required
            self.enable_attribute(Attribute::POSITIONS);

            let mut total_size = mem::size_of::<Vector3<f32>>() * self.positions.len();
            let mut actual_stride = mem::size_of::<Vector3<f32>>();

            // Do the same for the other 3 attributes (if they are enabled), whilst calculating
            // the actual stride in between each vertex
            if let Some(colors) = &self.colors {
                if self.positions.len() != colors.len() {
                    return Err(
                        "The number of colors does not equal the number of vertex positions",
                    );
                }
                total_size += mem::size_of::<Vector3<f32>>() * colors.len();
                actual_stride += mem::size_of::<Vector3<f32>>();
                self.enable_attribute(Attribute::COLORS);
            }
            if let Some(normals) = &self.normals {
                if self.positions.len() != normals.len() {
                    return Err(
                        "The number of normals does not equal the number of vertex positions",
                    );
                }
                total_size += mem::size_of::<Vector3<f32>>() * normals.len();
                actual_stride += mem::size_of::<Vector3<f32>>();
                self.enable_attribute(Attribute::NORMALS);
            }
            if let Some(texcoords) = &self.texcoords {
                if self.positions.len() != texcoords.len() {
                    return Err("The number of texture coordinates does not equal the number of vertex positions");
                }
                total_size += mem::size_of::<Vector2<f32>>() * texcoords.len();
                actual_stride += mem::size_of::<Vector2<f32>>();
                self.enable_attribute(Attribute::TEXCOORDS);
            }

            // Create the vertex buffer that will hold all interleaved vertex attributes
            self.generate_vertex_data();

            gl::CreateBuffers(1, &mut self.vbo);
            gl::NamedBufferData(
                self.vbo,
                total_size as isize,
                self.vertex_data.as_ptr() as *const GLvoid,
                gl::DYNAMIC_DRAW, // TODO: this shouldn't always be set to dynamic
            );

            gl::VertexArrayVertexBuffer(
                self.vao,
                0, // Binding index
                self.vbo,
                0, // Offset
                actual_stride as i32,
            );
        }

        Ok(())
    }

    /// Activates the vertex `attribute` (i.e. colors, normals, etc.).
    fn enable_attribute(&mut self, attribute: Attribute) {
        unsafe {
            gl::EnableVertexArrayAttrib(self.vao, attribute.get_index());
            gl::VertexArrayAttribFormat(
                self.vao,
                attribute.get_index(),
                attribute.get_element_count(),
                gl::FLOAT,
                gl::FALSE,
                attribute.get_relative_offset() as u32,
            );

            // All attributes are bound to index `0` and interleaved in the same VBO
            gl::VertexArrayAttribBinding(self.vao, attribute.get_index(), 0);
        }
    }

    /// Generates a single buffer of floats that will contain all of the interleaved
    /// vertex attributes.
    fn generate_vertex_data(&mut self) {
        self.vertex_data = vec![];

        for index in 0..self.positions.len() {
            self.vertex_data
                .extend_from_slice(&Into::<[f32; 3]>::into(self.positions[index]));

            if let Some(colors) = &self.colors {
                self.vertex_data
                    .extend_from_slice(&Into::<[f32; 3]>::into(colors[index]));
            }
            if let Some(normals) = &self.normals {
                self.vertex_data
                    .extend_from_slice(&Into::<[f32; 3]>::into(normals[index]));
            }
            if let Some(texcoords) = &self.texcoords {
                self.vertex_data
                    .extend_from_slice(&Into::<[f32; 2]>::into(texcoords[index]));
            }
        }
    }

    /// Uploads the interleaved vertex data to the GPU.
    fn upload_vertex_data(&mut self) {
        let size = (self.vertex_data.len() * (mem::size_of::<f32>() as usize)) as GLsizeiptr;

        unsafe {
            gl::NamedBufferSubData(
                self.vbo,
                0,
                size,
                self.vertex_data.as_ptr() as *const GLvoid,
            );
        }
    }

    /// Returns the number of vertices in this mesh.
    pub fn get_number_of_vertices(&self) -> usize {
        self.positions.len()
    }

    /// Draws the mesh using the specified drawing `mode` (i.e. `gl::TRIANGLES`).
    pub fn draw(&self, mode: GLenum) {
        unsafe {
            gl::BindVertexArray(self.vao);
            gl::DrawArrays(mode, 0, self.get_number_of_vertices() as GLsizei);
        }
    }

    /// Sets the positions of the vertices in this mesh to `positions`.
    pub fn set_positions(&mut self, positions: &Vec<Vector3<f32>>) {
        let size_changed = self.get_number_of_vertices() != positions.len();

        // Always copy new positions to CPU-side buffer (small penalty here)
        self.positions = positions.clone();

        // Re-allocate GPU memory, if needed
        if size_changed {
            // The `generate_vertex_data()` function will be called automatically inside of `allocate()`,
            // so no need to do that again here...
            self.allocate().unwrap();
        } else {
            self.generate_vertex_data();
            self.upload_vertex_data();
        }
    }

    /// Sets this meshes vertex colors.
    pub fn set_colors(&mut self, colors: &Vec<Vector3<f32>>) {
        // If this attribute wasn't already enabled, enable it and rebuild OpenGL objects as
        // necessary
        if let None = self.colors {
            self.enable_attribute(Attribute::COLORS);

            self.colors = Some(colors.clone());
            self.allocate().unwrap();
        } else {
            self.colors = Some(colors.clone());
            self.generate_vertex_data();
            self.upload_vertex_data();
        }
    }

    /// Sets this meshes vertex normals.
    pub fn set_normals(&mut self, normals: &Vec<Vector3<f32>>) {
        // If this attribute wasn't already enabled, enable it and rebuild OpenGL objects as
        // necessary
        if let None = self.normals {
            self.enable_attribute(Attribute::NORMALS);

            self.normals = Some(normals.clone());
            self.allocate().unwrap();
        } else {
            self.normals = Some(normals.clone());
            self.generate_vertex_data();
            self.upload_vertex_data();
        }
    }

    /// Sets this meshes vertex texture coordinates.
    pub fn set_texcoords(&mut self, texcoords: &Vec<Vector2<f32>>) {
        // If this attribute wasn't already enabled, enable it and rebuild OpenGL objects as
        // necessary
        if let None = self.texcoords {
            self.enable_attribute(Attribute::TEXCOORDS);

            self.texcoords = Some(texcoords.clone());
            self.allocate().unwrap();
        } else {
            self.texcoords = Some(texcoords.clone());
            self.generate_vertex_data();
            self.upload_vertex_data();
        }
    }
}
