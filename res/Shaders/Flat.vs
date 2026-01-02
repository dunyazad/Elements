#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec4 aColor;
layout (location = 3) in vec2 aUV;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vPos_world;
out vec4 vColor;
out vec2 vUV;

void main()
{
    vPos_world = vec3(model * vec4(aPos, 1.0));
    vColor = aColor;
    vUV = aUV;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
