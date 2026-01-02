#version 330 core

in vec3 gNormal;
in vec4 gColor;
in vec2 gUV;
in vec3 gFragPos;

uniform vec3 cameraPos;

uniform int useSolidColor;
uniform vec3 solidColor;

out vec4 FragColor;

void main()
{
    vec3 lightDir = normalize(cameraPos - gFragPos);
    float lighting = max(dot(normalize(gNormal), lightDir), 0.2); // Diffuse + ambient

    if(useSolidColor == 0)
    {
        FragColor = vec4(gColor.rgb * lighting, gColor.a);
    }
    else
    {
        FragColor = vec4(solidColor, 1.0f);
    }
}
