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
    vec3 lightDir = normalize(cameraPos - vFragPos);
    float lighting = max(dot(normalize(vNormal), lightDir), 0.2);
    if(0 == useSolidColor)
    {
        FragColor = vec4(vColor.rgb * lighting, vColor.a);
    }
    else
    {
        FragColor = vec4(solidColor, 1.0f);
    }
}
