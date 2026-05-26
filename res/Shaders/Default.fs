//#version 330 core
//
//in vec3 vNormal;
//in vec4 vColor;
//in vec2 vUV;
//in vec3 vFragPos;
//
//uniform vec3 cameraPos;
//uniform int useSolidColor;
//uniform vec3 solidColor;
//
//out vec4 FragColor;
//
//void main()
//{
//    vec3 lightDir = normalize(cameraPos - vFragPos);
//    float lighting = max(dot(normalize(vNormal), lightDir), 0.2);
//    if(0 == useSolidColor)
//    {
//        FragColor = vec4(vColor.rgb * lighting, vColor.a);
//    }
//    else
//    {
//        FragColor = vec4(solidColor, 1.0f);
//    }
//}
//


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
    // 보간된 법선을 반드시 정규화
    vec3 norm = normalize(vNormal);
    
    // 헤드램프 라이팅 (카메라 위치 = 광원 위치)
    vec3 lightDir = normalize(cameraPos - vFragPos);
    
    // 0~1 사이로 부드럽게 감쇠되는 라이팅 계산
    // dot 결과가 음수로 떨어져도 0.2 ~ 1.0 사이를 유지하도록 보간
    float halfL = dot(norm, lightDir) * 0.5 + 0.5;
    float lighting = max(halfL, 0.2); 
    
    vec4 baseColor;
    if (0 == useSolidColor)
    {
        baseColor = vColor;
    }
    else
    {
        baseColor = vec4(solidColor, 1.0f);
    }
    
    // 감마 보정을 고려하지 않은 단순 결과 출력
    FragColor = vec4(baseColor.rgb * lighting, baseColor.a);
}