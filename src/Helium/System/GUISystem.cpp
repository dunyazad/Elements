#include "pch.h"
#include <Helium/Systems/GUISystem.h>
#include <Helium/HeliumCore.h>
#include <glad/glad.h>
#include <sstream>
#include <vector>

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb/stb_truetype.h>

// [수정] std::vector로 선언 (컴파일러 인식과 일치시킴)
std::vector<stbtt_packedchar> charData;

// 기본 폰트 크기
const float BASE_FONT_SIZE = 32.0f;

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

std::vector<uint32_t> DecodeUTF8(const std::string& text)
{
    std::vector<uint32_t> codepoints;
    for (size_t i = 0; i < text.length();)
    {
        unsigned char c = text[i];
        if ((c & 0x80) == 0) // 1 byte
        {
            codepoints.push_back(c);
            i += 1;
        }
        else if ((c & 0xE0) == 0xC0) // 2 bytes
        {
            codepoints.push_back(((c & 0x1F) << 6) | (text[i + 1] & 0x3F));
            i += 2;
        }
        else if ((c & 0xF0) == 0xE0) // 3 bytes
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
        auto fontFile = File("../../res/Fonts/NanumGothic/NanumGothic.ttf", true);
        auto ttf_buffer = fontFile.ReadAllBytes();
        if (ttf_buffer.empty()) return;

        const int TEX_WIDTH = 4096;
        const int TEX_HEIGHT = 4096;
        unsigned char* bitmap = new unsigned char[TEX_WIDTH * TEX_HEIGHT];

        stbtt_pack_context pc;
        if (!stbtt_PackBegin(&pc, bitmap, TEX_WIDTH, TEX_HEIGHT, 0, 1, NULL))
        {
            delete[] bitmap;
            return;
        }

        // [수정] 벡터 크기 확보 및 0으로 초기화
        charData.resize(12000);
        // [수정] .data()를 사용하여 실제 메모리 주소 전달 (컴파일 에러 해결)
        memset(charData.data(), 0, charData.size() * sizeof(stbtt_packedchar));

        stbtt_pack_range ranges[2] = { 0 };

        // Range 0: ASCII
        ranges[0].font_size = BASE_FONT_SIZE;
        ranges[0].first_unicode_codepoint_in_range = 32;
        ranges[0].num_chars = 96;
        // [수정] .data() 사용
        ranges[0].chardata_for_range = charData.data();
        ranges[0].h_oversample = 1;
        ranges[0].v_oversample = 1;

        // Range 1: 한글
        ranges[1].font_size = BASE_FONT_SIZE;
        ranges[1].first_unicode_codepoint_in_range = 0xAC00;
        ranges[1].num_chars = 11172;
        // [수정] .data() 포인터에 오프셋 더하기 (vector + int는 불가능하므로)
        ranges[1].chardata_for_range = charData.data() + 96;
        ranges[1].h_oversample = 1;
        ranges[1].v_oversample = 1;

        stbtt_PackFontRanges(&pc, ttf_buffer.data(), 0, ranges, 2);
        stbtt_PackEnd(&pc);

        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, TEX_WIDTH, TEX_HEIGHT, 0, GL_RED, GL_UNSIGNED_BYTE, bitmap);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        delete[] bitmap;

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        textShader = Helium.CreateShader("GUIShader", guiVS, guiFS);

        initialized = true;
    }
}

void GUISystem::Render()
{
    static auto lastTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto deltaTime = std::chrono::high_resolution_clock::now() - lastTime;
    lastTime = currentTime;

	// 현재 시간 표시 YYYY-MM-DD HH:MM:SS
	std::stringstream timeSS;
	auto timeNow = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	tm localTime;
	localtime_s(&localTime, &timeNow);
    timeSS << "Time: " << (localTime.tm_year + 1900) << "-"
        << std::setw(2) << std::setfill('0') << (localTime.tm_mon + 1) << "-"
        << std::setw(2) << std::setfill('0') << localTime.tm_mday << " "
        << std::setw(2) << std::setfill('0') << localTime.tm_hour << ":"
        << std::setw(2) << std::setfill('0') << localTime.tm_min << ":"
		<< std::setw(2) << std::setfill('0') << localTime.tm_sec;
    
	RenderText(timeSS.str(), 10.0f, 30.0f, 32.0f, { 1.0f, 1.0f, 1.0f, 1.0f });

    std::stringstream ss;
    ss << "FPS: " << (1.0f / std::chrono::duration<float>(deltaTime).count());
    RenderText(ss.str(), 10.0f, 60.0f, 32.0f, { 1.0f, 1.0f, 1.0f, 1.0f });

    // 테스트 텍스트 (32px)
    // RenderText("Test Code", 10.0f, 80.0f, 32.0f, { 1.0f, 1.0f, 0.0f, 1.0f });
}

void GUISystem::RenderText(const std::string& text, float x, float y, float targetFontSize, const Eigen::Vector4f& color)
{
    if (textShader == nullptr) return;

    float scale = targetFontSize / BASE_FONT_SIZE;

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float width = (float)Helium.GetWidth();
    float height = (float)Helium.GetHeight();
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    if (width > 0 && height > 0)
    {
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

    std::vector<uint32_t> codepoints = DecodeUTF8(text);

    float currentX = x;
    float currentY = y;

    const int TEX_WIDTH = 4096;
    const int TEX_HEIGHT = 4096;

    for (uint32_t cp : codepoints)
    {
        stbtt_aligned_quad q;
        int charIndex = -1;

        if (cp >= 32 && cp < 32 + 96)
        {
            charIndex = cp - 32;
        }
        else if (cp >= 0xAC00 && cp <= 0xD7A3)
        {
            charIndex = 96 + (cp - 0xAC00);
        }

        if (charIndex == -1) continue;

        float x_offset = 0.0f;
        float y_offset = 0.0f;

        stbtt_GetPackedQuad(charData.data(), TEX_WIDTH, TEX_HEIGHT, charIndex, &x_offset, &y_offset, &q, 0);

        float x0 = currentX + (q.x0 * scale);
        float x1 = currentX + (q.x1 * scale);
        float y0 = currentY + (q.y0 * scale);
        float y1 = currentY + (q.y1 * scale);

        float vertices[6][4] = {
            { x0, y0, q.s0, q.t0 },
            { x1, y0, q.s1, q.t0 },
            { x1, y1, q.s1, q.t1 },

            { x0, y0, q.s0, q.t0 },
            { x1, y1, q.s1, q.t1 },
            { x0, y1, q.s0, q.t1 }
        };

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        currentX += (x_offset * scale);
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