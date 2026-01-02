#version 330 core

in vec3 vNormal;
in vec4 vColor;
in vec2 vUV;
in vec3 vFragPos;

uniform vec3 cameraPos;
uniform sampler2D texture0; // 새로 추가: 텍스처 유니폼

uniform int useSolidColor;
uniform vec3 solidColor;

out vec4 FragColor;

void main() {
    vec3 lightDir = normalize(cameraPos - vFragPos);
    float lighting = max(dot(normalize(vNormal), lightDir), 0.5);

//    vec4 texColor = texture(texture0, vUV); // UV로부터 텍스처 색 가져오기
//    vec4 finalColor = vColor * texColor;    // vertex color와 texture color 곱하기
//    FragColor = finalColor * lighting;      // 조명 적용

    //FragColor = texture(texture0, vUV); // UV로부터 텍스처 색 가져오기

    if(0 == useSolidColor)
    {
        FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f); // UV로부터 텍스처 색 가져오기
    }
    else
    {
        FragColor = vec4(solidColor, 1.0f);
    }
}
