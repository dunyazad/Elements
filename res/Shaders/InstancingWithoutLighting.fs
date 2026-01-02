#version 330 core

in vec3 vNormal;
in vec4 vColor;
in vec2 vUV;
in vec3 vFragPos;

uniform vec3 cameraPos;

uniform int useSolidColor;
uniform vec3 solidColor;

out vec4 FragColor;

void main()
{
    FragColor = vColor;
}
