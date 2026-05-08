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
    vec3 normal = normalize(vNormal);

    if (!gl_FrontFacing)
    {
        normal = -normal;
    }

    float lighting = max(dot(normal, lightDir), 0.2);

    if (useSolidColor == 0)
    {
        FragColor = vec4(vColor.rgb * lighting, vColor.a);
    }
    else
    {
        FragColor = vec4(solidColor * lighting, 1.0f);
    }

    //if (!gl_FrontFacing)
    //{
    //    FragColor = vec4(1.0f - FragColor.r * 1.5f, 1.0f - FragColor.g * 1.5f, 1.0f - FragColor.b * 1.5f, FragColor.a);
    //}
}
