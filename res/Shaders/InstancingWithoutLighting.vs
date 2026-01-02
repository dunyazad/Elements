#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec4 aColor;
layout (location = 3) in vec2 aUV;
layout (location = 4) in vec4 instanceColor; // Instance color
layout (location = 5) in vec3 instanceNormal; // Instance normal
layout (location = 6) in mat4 instanceModel; // Instance transformation matrix

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vNormal;
out vec4 vColor;
out vec2 vUV;
out vec3 vFragPos; // New: world-space fragment position

void main() {
    //mat4 modelView = view * model * instanceModel; // Use instance model matrix
    mat4 modelView = view * instanceModel; // Use instance model matrix

    vFragPos = vec3(instanceModel * vec4(aPos, 1.0)); // Transform to world space
    vNormal = mat3(transpose(inverse(instanceModel))) * aNormal; // Transform normal correctly
    //vNormal = mat3(transpose(inverse(instanceModel))) * instanceNormal; // Transform normal correctly
    //vColor = aColor;
    vColor = instanceColor;
    vUV = aUV;

    gl_Position = projection * modelView * vec4(aPos, 1.0);
}