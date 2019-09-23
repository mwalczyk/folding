#version 460

// These should never change: see the mesh module
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec2 texcoord;

out VS_OUT
{
    vec3 color;
} vs_out;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;

void main()
{
    vs_out.color = color;

    gl_Position = u_projection * u_view * u_model * vec4(position, 1.0);
}
