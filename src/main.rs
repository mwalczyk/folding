#![allow(dead_code)]
mod assignment;
mod constants;
mod data;
mod fold_specification;
mod graphics;
mod interaction;
mod model;

use crate::fold_specification::FoldSpecification;
use crate::graphics::program::Program;
use crate::graphics::utils;
use crate::interaction::InteractionState;
use crate::model::Model;

use cgmath::{EuclideanSpace, Matrix4, Point3, SquareMatrix, Vector3};
use glutin::dpi::LogicalSize;
use glutin::event::{ElementState, Event, WindowEvent};
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
    let size = LogicalSize::new(constants::WIDTH as f64, constants::HEIGHT as f64);
    let el = EventLoop::new();
    let wb = WindowBuilder::new()
        .with_title("folding")
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
    let mut model = Matrix4::identity();
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

    // Interaction (mouse clicks, etc.)
    let mut interaction = InteractionState::new();

    // Turn on depth testing, etc. then bind the shader program
    set_draw_state();
    draw_program.bind();
    draw_program.uniform_matrix_4f("u_view", &view);
    draw_program.uniform_matrix_4f("u_projection", &projection);

    // Load the origami model
    let spec = FoldSpecification::from_file(Path::new("folds/bird_base.fold")).unwrap();
    let mut fold = Model::from_specification(&spec, 1000.0).unwrap();

    // Main rendering loop
    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        clear_screen();
        draw_program.uniform_matrix_4f("u_model", &model);
        fold.step_simulation();
        fold.draw_mesh();
        fold.draw_normals();
        gl_window.swap_buffers().unwrap();

        match event {
            Event::LoopDestroyed => return,
            Event::WindowEvent { ref event, .. } => match event {
                WindowEvent::Resized(logical_size) => {
                    let dpi_factor = gl_window.window().hidpi_factor();
                    gl_window.resize(logical_size.to_physical(dpi_factor));
                }
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::CursorMoved { position, .. } => {
                    interaction.cursor_prev = interaction.cursor_curr;
                    interaction.cursor_curr.x = position.x as f32 / constants::WIDTH as f32;
                    interaction.cursor_curr.y = position.y as f32 / constants::HEIGHT as f32;

                    if interaction.lmouse_pressed {
                        let delta = interaction.get_mouse_delta() * constants::MOUSE_SENSITIVITY;

                        let rot_xz = Matrix4::from_angle_y(cgmath::Rad(delta.x));
                        let rot_yz = Matrix4::from_angle_x(cgmath::Rad(delta.y));

                        model = rot_xz * rot_yz * model;
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => match button {
                    glutin::event::MouseButton::Left => {
                        if let ElementState::Pressed = state {
                            interaction.cursor_pressed = interaction.cursor_curr;
                            interaction.lmouse_pressed = true;
                        } else {
                            interaction.lmouse_pressed = false;
                        }
                    }
                    glutin::event::MouseButton::Right => {
                        if let ElementState::Pressed = state {
                            interaction.rmouse_pressed = true;
                        } else {
                            interaction.rmouse_pressed = false;
                        }
                    }
                    _ => (),
                },
                _ => (),
            },
            _ => (),
        }
    });
}
