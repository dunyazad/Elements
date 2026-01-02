#version 330 core

in vec3 vNormal;
in vec4 vColor;
in vec2 vUV;
in vec3 vFragPos; // Fragment position in world space (must be passed from vertex shader)

uniform vec3 cameraPos; // Camera (light) position in world space

uniform int useSolidColor;
uniform vec3 solidColor;

out vec4 FragColor;

void main() {
    vec3 lightDir = normalize(cameraPos - vFragPos); // Light direction from fragment to camera
    float lighting = max(dot(normalize(vNormal), lightDir), 0.2); // Diffuse shading with ambient
    if(0 == useSolidColor)
    {
        FragColor = vec4(vColor.rgb * lighting, vColor.a); // Apply lighting to color
    }
    else
    {
        FragColor = vec4(solidColor, 1.0f);
    }
}
