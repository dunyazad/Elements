#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal; // Billboard에서는 사용하지 않지만 layout 유지를 위해 둠
layout (location = 2) in vec4 aColor;
layout (location = 3) in vec2 aUV;
layout (location = 4) in vec4 instanceColor; 
layout (location = 5) in vec3 instanceNormal; // Billboard에서는 위치 계산엔 쓰지 않음
layout (location = 6) in mat4 instanceModel; 

uniform mat4 view;
uniform mat4 projection;
uniform vec3 cameraPos; // Billboard Normal 계산을 위해 필요

out vec3 vNormal;
out vec4 vColor;
out vec2 vUV;
out vec3 vFragPos;

void main() {
    vec3 centerPos = vec3(instanceModel[3]);
    float sx = length(instanceModel[0]); 
    float sz = length(instanceModel[2]); 
    vec3 cameraRight = vec3(view[0][0], view[1][0], view[2][0]);
    vec3 cameraUp    = vec3(view[0][1], view[1][1], view[2][1]);

    vec3 worldPos =
        centerPos 
        - (cameraRight * aPos.x * sx) // (+)를 (-)로 변경
        + (cameraUp    * aPos.z * sz);

    vFragPos = worldPos;
    
    vNormal = normalize(cameraPos - worldPos);

    vColor = instanceColor;
    vUV = aUV;

    gl_Position = projection * view * vec4(worldPos, 1.0);
}
