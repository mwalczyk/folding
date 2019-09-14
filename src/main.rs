extern crate gl;
extern crate glutin;
extern crate serde;

mod graphics;
mod half_edge;
mod model;

use crate::graphics::mesh::Mesh;
use crate::graphics::program::Program;
use crate::graphics::utils;
use crate::model::{FoldSpecification, Model};

use cgmath::{EuclideanSpace, InnerSpace, Matrix4, Point3, SquareMatrix, Vector3, Zero};
use gl::types::*;
use glutin::dpi::LogicalSize;
use std::path::Path;

/// Sets the draw state (enables depth testing, etc.)
fn set_draw_state() {
    unsafe {
        // Turn on depth testing
        gl::Enable(gl::DEPTH_TEST);
        gl::DepthFunc(gl::LESS);
    }
}

fn main() {
    let size = LogicalSize::new(720.0, 720.0);
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_title("folding")
        .with_dimensions(size);
    let gl_window = glutin::ContextBuilder::new()
        .build_windowed(window, &events_loop)
        .unwrap();

    let gl_window = unsafe { gl_window.make_current() }.unwrap();
    gl::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _);

    // Load the shader program used for rendering
    let draw_program = Program::from_sources(
        utils::load_file_as_string(Path::new("shaders/draw.vert")),
        utils::load_file_as_string(Path::new("shaders/draw.frag")),
    )
    .unwrap();

    // Set up the model-view-projection (MVP) matrices
    let model = Matrix4::identity();
    let view = Matrix4::look_at(
        Point3::new(0.0, 720.0, 3000.0),
        Point3::origin(),
        Vector3::unit_y(),
    );
    let projection = cgmath::perspective(
        cgmath::Rad(std::f32::consts::FRAC_PI_4),
        720.0 as f32 / 720.0 as f32,
        0.1,
        5000.0,
    );

    // Turn on depth testing, etc. then bind the shader program
    set_draw_state();
    draw_program.bind();
    draw_program.uniform_matrix_4f("u_model", &model);
    draw_program.uniform_matrix_4f("u_view", &view);
    draw_program.uniform_matrix_4f("u_projection", &projection);

    // Load the origami model
    let spec = FoldSpecification::from_file(Path::new("bird_base.fold")).unwrap();
    let mut model = Model::from_specification(&spec, 1000.0);

    // Main rendering loop
    events_loop.run_forever(|event| {
        use glutin::{ControlFlow, Event, WindowEvent};

        if let Event::WindowEvent { event, .. } = event {
            if let WindowEvent::CloseRequested = event {
                return ControlFlow::Break;
            }
        }

        //model.step_simulation();

        unsafe {
            // Clear the screen to black
            gl::ClearColor(0.12, 0.1, 0.1, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);

            model.mesh.draw(gl::LINES);
        }

        gl_window.swap_buffers().unwrap();

        ControlFlow::Continue
    });
}
