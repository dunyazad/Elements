#version 330 core

in vec3 vNormal;
in vec4 vColor;
in vec2 vUV;
in vec3 vFragPos;

uniform vec3 cameraPos;
uniform int useSolidColor;
uniform vec3 solidColor;

out vec4 FragColor;

void main() {
    // Billboard는 항상 카메라를 향하므로, Normal과 LightDir가 거의 일치합니다.
    vec3 lightDir = normalize(cameraPos - vFragPos);
    
    // 양면 렌더링을 고려하여 abs(dot)을 쓰거나, 이미 normal을 카메라 쪽으로 맞췄으므로 max(dot) 사용
    // 조명을 약간 강하게 주어(0.5 ~ 1.0) 잘 보이게 합니다.
    float diff = max(dot(normalize(vNormal), lightDir), 0.0);
    
    // Ambient를 높여서 어두운 부분이 없도록 처리 (0.5 + diff * 0.5 등)
    float lighting = 0.4 + (diff * 0.6); 

    if(0 == useSolidColor)
    {
        FragColor = vec4(vColor.rgb * lighting, vColor.a);
    }
    else
    {
        FragColor = vec4(solidColor, 1.0f);
    }
}