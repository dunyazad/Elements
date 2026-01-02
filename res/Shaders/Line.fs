#version 330 core
in vec4 vColor;
out vec4 FragColor;

uniform int useSolidColor;
uniform vec3 solidColor;

void main()
{
    if(0 == useSolidColor)
    {
        FragColor = vColor;
    }
    else
    {
        FragColor = vec4(solidColor, 1.0f);
    }
}
