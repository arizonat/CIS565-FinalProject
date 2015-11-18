#version 330

uniform mat4 u_projMatrix;

layout(points) in;
layout(triangle_strip) out;
layout(max_vertices = 200) out;

void main() {

    vec4 pos = vec4(gl_in[0].gl_Position.xyz,1.0);
    vec4 center = u_projMatrix * pos;

    //vec4 pos = vec4(0,0,0,1);  //introduce a single vertex at the origin
    //pos = u_projMatrix * pos;
    vec4 new_pos;
    float radius = 0.1;

    for(float i = 0; i < 6.38 ; i+=0.1)  //generate vertices at positions on the circumference from 0 to 2*pi
    {
	gl_Position = center;
	EmitVertex();
        new_pos = vec4(pos.x+radius*cos(i),pos.y+radius*sin(i),pos.z,1.0);   //circle parametric equation
 	gl_Position = u_projMatrix * new_pos;
	EmitVertex();       
    }
}
