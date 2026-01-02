#version 450 core

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D screenTexture;

void main()
{
    // 상하 반전 방지 — Feather의 텍스처 좌표계와 일치
    vec2 uv = vec2(TexCoords.x, TexCoords.y);

    // 단순 텍스처 복사
    FragColor = texture(screenTexture, uv);
}
