#version 330

uniform mat4 u_projMatrix;

in vec4 Position;

void main() {
    gl_Position = u_projMatrix * Position;
}