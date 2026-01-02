#version 330 core

in vec3 vColor;
in vec3 vNormal;
out vec4 FragColor;

void main()
{
    float l = max(dot(normalize(vNormal), normalize(vec3(1,1,1))), 0.2);
    FragColor = vec4(vColor * l, 1.0);
}
