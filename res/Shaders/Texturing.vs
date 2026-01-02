#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec4 aColor;
layout (location = 3) in vec2 aUV;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vNormal;
out vec4 vColor;
out vec2 vUV;
out vec3 vFragPos; // New: world-space fragment position

void main() {
    mat4 modelView = view * model;
    
    vFragPos = vec3(model * vec4(aPos, 1.0)); // Transform to world space
    vNormal = mat3(transpose(inverse(model))) * aNormal; // Transform normal correctly
    vColor = aColor;
    vUV = aUV;

    gl_Position = projection * modelView * vec4(aPos, 1.0);
}
