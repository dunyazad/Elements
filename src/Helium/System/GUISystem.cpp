#include "pch.h"
#include <Helium/Systems/GUISystem.h>
#include <Helium/HeliumCore.h>
#include <glad/glad.h>
#include <vector>

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb/stb_truetype.h>

// [수정 1] 한글(11,172자) + ASCII(96자)를 담을 충분한 공간 확보
// 11268개 이상의 배열이 필요합니다. 넉넉하게 잡습니다.
stbtt_packedchar charData[12000];

const std::string guiVS = R"(
#version 330 core
layout (location = 0) in vec4 vertex; 
out vec2 TexCoords;

uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
    TexCoords = vertex.zw;
}
)";

const std::string guiFS = R"(
#version 330 core
in vec2 TexCoords;
out vec4 color;

uniform sampler2D text;
uniform vec4 textColor;

void main()
{
    float alpha = texture(text, TexCoords).r;
    if(alpha < 0.1) discard;
    color = vec4(textColor.rgb, textColor.a * alpha);
}
)";

// [수정 2] UTF-8 문자열을 유니코드(uint32_t)로 변환하는 함수 추가
std::vector<uint32_t> DecodeUTF8(const std::string& text)
{
    std::vector<uint32_t> codepoints;
    for (size_t i = 0; i < text.length();)
    {
        unsigned char c = text[i];
        if ((c & 0x80) == 0) // 1 byte (ASCII)
        {
            codepoints.push_back(c);
            i += 1;
        }
        else if ((c & 0xE0) == 0xC0) // 2 bytes
        {
            codepoints.push_back(((c & 0x1F) << 6) | (text[i + 1] & 0x3F));
            i += 2;
        }
        else if ((c & 0xF0) == 0xE0) // 3 bytes (한글)
        {
            codepoints.push_back(((c & 0x0F) << 12) | ((text[i + 1] & 0x3F) << 6) | (text[i + 2] & 0x3F));
            i += 3;
        }
        else // 4 bytes
        {
            codepoints.push_back(((c & 0x07) << 18) | ((text[i + 1] & 0x3F) << 12) | ((text[i + 2] & 0x3F) << 6) | (text[i + 3] & 0x3F));
            i += 4;
        }
    }
    return codepoints;
}

GUISystem::GUISystem(HeliumCore* core)
    : HeliumSystem(core)
{
}

GUISystem::~GUISystem()
{
    Shutdown();
}

void GUISystem::Initialize()
{
}

bool initialized = false;
void GUISystem::Update(float dt)
{
    if (false == initialized)
    {
        // 1. 폰트 파일 로드
        auto fontFile = File("../../res/Fonts/NanumGothic/NanumGothic.ttf", true);
        auto ttf_buffer = fontFile.ReadAllBytes();
        if (ttf_buffer.empty()) return;

        // 2. 비트맵 생성 (4k)
        const int TEX_WIDTH = 4096;
        const int TEX_HEIGHT = 4096;
        unsigned char* bitmap = new unsigned char[TEX_WIDTH * TEX_HEIGHT];

        // 3. 폰트 베이킹 준비
        stbtt_pack_context pc;
        if (!stbtt_PackBegin(&pc, bitmap, TEX_WIDTH, TEX_HEIGHT, 0, 1, NULL))
        {
            delete[] bitmap;
            return;
        }

        // [중요] 멤버 변수 벡터 크기 확보 (ASCII 96 + 한글 11172 = 11268)
        charData.resize(12000);

        // 구조체 초기화 (쓰레기값 방지)
        stbtt_pack_range ranges[2] = { 0 };

        // Range 0: ASCII
        ranges[0].font_size = 24.0f;
        ranges[0].first_unicode_codepoint_in_range = 32;
        ranges[0].num_chars = 96;
        ranges[0].chardata_for_range = charData.data(); // [수정] 벡터의 포인터 전달
        ranges[0].h_oversample = 1;
        ranges[0].v_oversample = 1;

        // Range 1: 한글
        ranges[1].font_size = 24.0f;
        ranges[1].first_unicode_codepoint_in_range = 0xAC00;
        ranges[1].num_chars = 11172;
        // [수정] 벡터 포인터 오프셋 (ASCII 96개 뒤부터 저장)
        ranges[1].chardata_for_range = charData.data() + 96;
        ranges[1].h_oversample = 1;
        ranges[1].v_oversample = 1;

        stbtt_PackFontRanges(&pc, ttf_buffer.data(), 0, ranges, 2);
        stbtt_PackEnd(&pc);

        // 4. OpenGL 텍스처 생성
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, TEX_WIDTH, TEX_HEIGHT, 0, GL_RED, GL_UNSIGNED_BYTE, bitmap);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        delete[] bitmap;

        // 5. VAO/VBO 생성
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);  
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        // 6. 쉐이더 생성
        textShader = Helium.CreateShader("GUIShader", guiVS, guiFS);

        initialized = true;
    }
}

void GUISystem::Render()
{
    RenderText((const char*)u8"Helium Engine! - 한글 테스트", 10.0f, 30.0f, 1.0f, { 1.0f, 1.0f, 1.0f, 1.0f });
}

void GUISystem::RenderText(const std::string& text, float x, float y, float scale, const Eigen::Vector4f& color)
{
    if (textShader == nullptr) return;

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float width = (float)Helium.GetWidth();
    float height = (float)Helium.GetHeight();
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    if (width > 0 && height > 0) {
        projection(0, 0) = 2.0f / width;
        projection(1, 1) = -2.0f / height;
        projection(0, 3) = -1.0f;
        projection(1, 3) = 1.0f;
    }

    textShader->Bind();
    textShader->SetInt("text", 0);
    textShader->SetMatrix4f("projection", projection);
    textShader->SetVector4f("textColor", color);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glBindVertexArray(VAO);

    // [수정 5] UTF-8 디코딩 및 인덱스 계산
    std::vector<uint32_t> codepoints = DecodeUTF8(text);
    float currentX = x;
    float currentY = y;
    const int TEX_WIDTH = 4096; // Initialize와 동일하게
    const int TEX_HEIGHT = 4096;

    for (uint32_t cp : codepoints)
    {
        stbtt_aligned_quad q;
        int charIndex = -1;

        // 인덱스 찾기 (charData 배열 내 위치)
        if (cp >= 32 && cp < 32 + 96) {
            charIndex = cp - 32; // ASCII
        }
        else if (cp >= 0xAC00 && cp <= 0xD7A3) {
            charIndex = 96 + (cp - 0xAC00); // 한글 (ASCII 96개 뒤에 있음)
        }

        if (charIndex == -1) continue; // 지원하지 않는 문자

        stbtt_GetPackedQuad(charData.data(), TEX_WIDTH, TEX_HEIGHT, charIndex, &currentX, &currentY, &q, 1);

        float vertices[6][4] = {
            { q.x0, q.y0, q.s0, q.t0 },
            { q.x1, q.y0, q.s1, q.t0 },
            { q.x1, q.y1, q.s1, q.t1 },

            { q.x0, q.y0, q.s0, q.t0 },
            { q.x1, q.y1, q.s1, q.t1 },
            { q.x0, q.y1, q.s0, q.t1 }
        };

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

void GUISystem::Shutdown()
{
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    if (textureID) glDeleteTextures(1, &textureID);
}

void GUISystem::Resize(int width, int height)
{
}
