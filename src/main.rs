#![allow(dead_code)]
extern crate gl;
extern crate glutin;
extern crate serde;

mod fold_specification;
mod graphics;
mod half_edge;
mod model;

use crate::fold_specification::FoldSpecification;
use crate::graphics::program::Program;
use crate::graphics::utils;
use crate::model::Model;

use cgmath::{EuclideanSpace, Matrix4, Point3, SquareMatrix, Vector3};
use glutin::dpi::LogicalSize;
use glutin::event::{Event, WindowEvent};
use glutin::event_loop::{ControlFlow, EventLoop};
use glutin::window::WindowBuilder;
use glutin::ContextBuilder;
use std::path::Path;

/// Sets the draw state (enables depth testing, etc.)
fn set_draw_state() {
    unsafe {
        // Turn on depth testing
        gl::Enable(gl::DEPTH_TEST);
        gl::DepthFunc(gl::LESS);

        // Set to wireframe rendering
        gl::PolygonMode(gl::FRONT_AND_BACK, gl::LINE);
    }
}

fn clear_screen() {
    unsafe {
        // Clear the screen to (off)black
        gl::ClearColor(0.12, 0.1, 0.1, 1.0);
        gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
    }
}

fn main() {
    let size = LogicalSize::new(720.0, 720.0);
    let el = EventLoop::new();
    let wb = WindowBuilder::new()
        .with_title("folding")
        .with_decorations(false)
        .with_inner_size(size);
    let windowed_context = ContextBuilder::new().build_windowed(wb, &el).unwrap();
    let windowed_context = unsafe { windowed_context.make_current().unwrap() };
    let gl_window = unsafe { windowed_context.make_current() }.unwrap();
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
    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        model.step_simulation();
        clear_screen();
        model.draw_mesh();
        model.draw_normals();
        gl_window.swap_buffers().unwrap();

        match event {
            Event::LoopDestroyed => return,
            Event::WindowEvent { ref event, .. } => match event {
                WindowEvent::Resized(logical_size) => {
                    let dpi_factor = gl_window.window().hidpi_factor();
                    gl_window.resize(logical_size.to_physical(dpi_factor));
                }
                WindowEvent::RedrawRequested => {
                    // TODO: `https://docs.rs/winit/0.20.0-alpha3/winit/window/struct.Window.html#method.request_redraw`
                }
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                _ => (),
            },
            _ => (),
        }
    });
}
