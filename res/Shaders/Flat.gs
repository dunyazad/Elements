#version 330 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 vPos_world[];
in vec4 vColor[];
in vec2 vUV[];

out vec3 gNormal;     // flat normal per triangle
out vec4 gColor;
out vec2 gUV;
out vec3 gFragPos;

void main()
{
    // Calculate flat normal (triangle face normal)
    vec3 U = vPos_world[1] - vPos_world[0];
    vec3 V = vPos_world[2] - vPos_world[0];
    vec3 normal = normalize(cross(U, V));

    for(int i = 0; i < 3; ++i)
    {
        gNormal = normal; // all three vertices get the same normal
        gColor = vColor[i];
        gUV = vUV[i];
        gFragPos = vPos_world[i];
        gl_Position = gl_in[i].gl_Position;
        EmitVertex();
    }
    EndPrimitive();
}
