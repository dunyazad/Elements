#version 330 core

in vec3 vNormal;
in vec4 vColor;
in vec2 vUV;
in vec3 vFragPos;

uniform vec3 cameraPos;

out vec4 FragColor;

uniform vec3 gridColor  = vec3(0.5, 0.5, 0.5);
uniform vec3 floorColor = vec3(1.0, 1.0, 1.0);
uniform vec3 axisXColor = vec3(1.0, 0.1, 0.1); // X축 (빨강)
uniform vec3 axisZColor = vec3(0.1, 0.4, 1.0); // Z축 (파랑)
uniform float gridSize = 1.0;
uniform float lineThickness = 1.0;
uniform float fadeDistance = 100.0;

void main()
{
    // 카메라 위치 기준으로 좌표 이동
    vec3 worldPos = vFragPos;
    //vec3 relativePos = worldPos - cameraPos;
    //vec2 coord = relativePos.xz / gridSize;

    vec2 coord = worldPos.xz / gridSize;

    vec2 d = fwidth(coord) * lineThickness;

    float gx = abs(fract(coord.x) - 0.5);
    float gz = abs(fract(coord.y) - 0.5);

    float lineX = 1.0 - smoothstep(0.0, d.x, gx);
    float lineZ = 1.0 - smoothstep(0.0, d.y, gz);

    float gridMask = max(lineX, lineZ);
    vec3 baseColor = mix(floorColor, gridColor, gridMask);

    // 중심축 강조 (카메라 기준)
    //float axisMaskX = 1.0 - smoothstep(0.0, d.x * 2.0, abs(relativePos.x));
    //float axisMaskZ = 1.0 - smoothstep(0.0, d.y * 2.0, abs(relativePos.z));
    float axisMaskX = 1.0 - smoothstep(0.0, d.x * 2.0, abs(worldPos.x));
    float axisMaskZ = 1.0 - smoothstep(0.0, d.y * 2.0, abs(worldPos.z));
    baseColor = mix(baseColor, axisXColor, axisMaskX);
    baseColor = mix(baseColor, axisZColor, axisMaskZ);

    // 거리 기반 페이드아웃
    //float dist = length(relativePos.xz);
    //float fade = clamp(1.0 - dist / fadeDistance, 0.0, 1.0);

    float dist = length(worldPos.xz - cameraPos.xz);
    float fade = clamp(1.0 - dist / fadeDistance, 0.0, 1.0);

    FragColor = vec4(baseColor * fade, 1.0);
}
