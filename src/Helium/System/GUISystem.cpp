#include "pch.h"
#include <Helium/Systems/GUISystem.h>
#include <Helium/HeliumCore.h>

#include <glad/glad.h>
#include <sstream>
#include <vector>
#include <algorithm> 
#include <iomanip>
#include <cmath>

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb/stb_truetype.h>

int GUISystem::zIndexCounter = 0;
const float BASE_FONT_SIZE = 32.0f;
// 한 번에 그릴 수 있는 최대 쿼드 개수 (텍스트 배칭용)
const int MAX_QUAD_COUNT = 1024;
const int MAX_VERTEX_COUNT = MAX_QUAD_COUNT * 6;

// -----------------------------------------------------------------------------
// Shaders
// -----------------------------------------------------------------------------

const std::string textVS = R"(
#version 330 core
layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>
out vec2 TexCoords;
uniform mat4 projection;
void main() {
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
    TexCoords = vertex.zw;
}
)";

const std::string textFS = R"(
#version 330 core
in vec2 TexCoords;
out vec4 color;
uniform sampler2D text;
uniform vec4 textColor;
void main() {
    float alpha = texture(text, TexCoords).r;
    if(alpha < 0.1) discard;
    color = vec4(textColor.rgb, textColor.a * alpha);
}
)";

const std::string geometryVS = R"(
#version 330 core
layout (location = 0) in vec4 vertex;
out vec2 TexCoords;
uniform mat4 projection;
void main() {
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
    TexCoords = vertex.zw;
}
)";

const std::string geometryFS = R"(
#version 330 core
in vec2 TexCoords;
out vec4 FragColor;
uniform vec4 rectColor;
uniform int isCircle;
void main() {
    if (isCircle == 1) {
        vec2 center = vec2(0.5);
        float dist = distance(TexCoords, center);
        float delta = fwidth(dist);
        float alpha = smoothstep(0.5, 0.5 - delta, dist);
        if (alpha < 0.01) discard;
        FragColor = vec4(rectColor.rgb, rectColor.a * alpha);
    } else {
        FragColor = rectColor;
    }
}
)";

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

static std::vector<uint32_t> DecodeUTF8(const std::string& text)
{
    std::vector<uint32_t> codepoints;
    for (size_t i = 0; i < text.length();)
    {
        unsigned char c = text[i];
        if ((c & 0x80) == 0) { codepoints.push_back(c); i += 1; }
        else if ((c & 0xE0) == 0xC0) { codepoints.push_back(((c & 0x1F) << 6) | (text[i + 1] & 0x3F)); i += 2; }
        else if ((c & 0xF0) == 0xE0) { codepoints.push_back(((c & 0x0F) << 12) | ((text[i + 1] & 0x3F) << 6) | (text[i + 2] & 0x3F)); i += 3; }
        else { codepoints.push_back(((c & 0x07) << 18) | ((text[i + 1] & 0x3F) << 12) | ((text[i + 2] & 0x3F) << 6) | (text[i + 3] & 0x3F)); i += 4; }
    }
    return codepoints;
}

// -----------------------------------------------------------------------------
// GUISystem Implementation
// -----------------------------------------------------------------------------

GUISystem::GUISystem(HeliumCore* core) : HeliumSystem(core) {}

GUISystem::~GUISystem() { Shutdown(); }

void GUISystem::Initialize()
{
    // 1. Load Font
    // 경로가 실행 파일 기준 올바른지 확인 필요 (없으면 로딩 실패)
    auto fontFile = File("../../res/Fonts/NanumGothic/NanumGothic.ttf", true);
    auto ttf_buffer = fontFile.ReadAllBytes();

    if (ttf_buffer.empty())
    {
        // 폰트 로드 실패 시 로그 출력 권장
        return;
    }

    stbtt_fontinfo fontInfo;
    if (stbtt_InitFont(&fontInfo, ttf_buffer.data(), stbtt_GetFontOffsetForIndex(ttf_buffer.data(), 0)))
    {
        int ascent, descent, lineGap;
        stbtt_GetFontVMetrics(&fontInfo, &ascent, &descent, &lineGap);
        float scale = stbtt_ScaleForPixelHeight(&fontInfo, BASE_FONT_SIZE);
        this->fontAscent = ascent * scale;
        this->fontDescent = descent * scale;
        this->fontLineGap = lineGap * scale;
    }

    // 2. Pack Font Atlas
    const int TEX_WIDTH = 4096;
    const int TEX_HEIGHT = 4096;
    unsigned char* bitmap = new unsigned char[TEX_WIDTH * TEX_HEIGHT];

    stbtt_pack_context pc;
    if (!stbtt_PackBegin(&pc, bitmap, TEX_WIDTH, TEX_HEIGHT, 0, 1, NULL)) { delete[] bitmap; return; }

    this->charData.resize(12000);
    memset(this->charData.data(), 0, this->charData.size() * sizeof(stbtt_packedchar));

    stbtt_pack_range ranges[2] = { 0 };
    // ASCII (32 ~ 127)
    ranges[0].font_size = BASE_FONT_SIZE;
    ranges[0].first_unicode_codepoint_in_range = 32;
    ranges[0].num_chars = 96;
    ranges[0].chardata_for_range = this->charData.data();
    ranges[0].h_oversample = 1; ranges[0].v_oversample = 1;

    // Hangul (0xAC00 ~ )
    ranges[1].font_size = BASE_FONT_SIZE;
    ranges[1].first_unicode_codepoint_in_range = 0xAC00;
    ranges[1].num_chars = 11172;
    ranges[1].chardata_for_range = this->charData.data() + 96;
    ranges[1].h_oversample = 1; ranges[1].v_oversample = 1;

    stbtt_PackFontRanges(&pc, ttf_buffer.data(), 0, ranges, 2);
    stbtt_PackEnd(&pc);

    // 3. Create Texture
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, TEX_WIDTH, TEX_HEIGHT, 0, GL_RED, GL_UNSIGNED_BYTE, bitmap);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    delete[] bitmap;

    // 4. Create Buffers (Batching을 위해 넉넉한 사이즈 할당)
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // Dynamic Draw: 자주 변경되는 데이터
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 4 * MAX_VERTEX_COUNT, NULL, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // 5. Create Shaders
    textShader = Helium.CreateShader("GUITextShader", textVS, textFS);
    geometryShader = Helium.CreateShader("GUIGeometryShader", geometryVS, geometryFS);
}

void GUISystem::Resize(int width, int height)
{
    // 필요 시 뷰포트나 프로젝션 갱신 로직 추가
}

void GUISystem::Shutdown()
{
    if (VAO) { glDeleteVertexArrays(1, &VAO); VAO = 0; }
    if (VBO) { glDeleteBuffers(1, &VBO); VBO = 0; }
    if (textureID) { glDeleteTextures(1, &textureID); textureID = 0; }
}

void GUISystem::Update(float dt)
{
    // 초기화가 완료되지 않았으면(폰트 로드 실패 등) 리턴
    if (textShader == nullptr || geometryShader == nullptr) return;

    auto& registry = Helium.GetRegistry();

    // Entity 수집 및 Caching
    for (auto& entity : registry.view<GUIRectangle>())
    {
        auto& r = registry.get<GUIRectangle>(entity);
        Submit<GUIRectangle>(r.zIndex, GUICommandType::Rectangle, r);
    }

    for (auto& entity : registry.view<GUICircle>())
    {
        auto& c = registry.get<GUICircle>(entity);
        Submit<GUICircle>(c.zIndex, GUICommandType::Circle, c);
    }

    for (auto& entity : registry.view<GUIText>())
    {
        auto& t = registry.get<GUIText>(entity);

        // 텍스트 내용이나 사이즈가 변경되었을 때만 길이 계산
        if (t.text != t._lastText || t.fontSize != t._lastFontSize)
        {
            float widthCalc = 0.0f;
            float fontScale = t.fontSize / BASE_FONT_SIZE;
            auto codepoints = DecodeUTF8(t.text);

            for (uint32_t cp : codepoints)
            {
                int idx = -1;
                if (cp >= 32 && cp < 128) idx = cp - 32;
                else if (cp >= 0xAC00 && cp <= 0xD7A3) idx = 96 + (cp - 0xAC00);

                if (idx != -1 && idx < (int)this->charData.size())
                {
                    widthCalc += this->charData[idx].xadvance * fontScale;
                }
            }

            t.cachedWidth = widthCalc;
            t._lastText = t.text;
            t._lastFontSize = t.fontSize;
        }

        Submit<GUIText>(t.zIndex, GUICommandType::Text, t);
    }
}

void GUISystem::Render()
{
    if (!renderCommandQueue.empty()) {
        std::sort(renderCommandQueue.begin(), renderCommandQueue.end(), [](const GUIRenderCommand& a, const GUIRenderCommand& b) { return a.zIndex < b.zIndex; });

        for (const auto& cmd : renderCommandQueue)
        {
            switch (cmd.type)
            {
            case GUICommandType::Rectangle:
                if (auto rect = std::any_cast<GUIRectangle>(&cmd.component))
                    RenderRectangle(rect->x, rect->y, rect->width, rect->height, rect->color);
                break;
            case GUICommandType::Circle:
                if (auto circle = std::any_cast<GUICircle>(&cmd.component))
                    RenderCircle(circle->x, circle->y, circle->radius, circle->color);
                break;
            case GUICommandType::Text:
                if (auto text = std::any_cast<GUIText>(&cmd.component))
                {
                    RenderText(text->text, text->x, text->y, text->cachedWidth, text->fontSize, text->color, text->hAlign, text->vAlign);
                }
                break;
            }
        }
        renderCommandQueue.clear();
    }

    static auto lastTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> deltaTime = currentTime - lastTime;
    lastTime = currentTime;

    // 간단한 FPS 표시
    std::stringstream ss;
    ss << "FPS: " << std::fixed << std::setprecision(1) << (1.0f / deltaTime.count());
    RenderText(ss.str(), 10.0f, 60.0f, 0.0f, 32.0f, { 1.0f, 1.0f, 1.0f, 1.0f }, TextHAlign::Left, TextVAlign::Baseline);
}

void GUISystem::RenderRectangle(float x, float y, float width, float height, const Eigen::Vector4f& color)
{
    if (geometryShader == nullptr) return;

    float scrWidth = (float)Helium.GetWidth();
    float scrHeight = (float)Helium.GetHeight();

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    if (scrWidth > 0 && scrHeight > 0) {
        projection(0, 0) = 2.0f / scrWidth;
        projection(1, 1) = -2.0f / scrHeight; // Y축 뒤집기 (Top-Left Origin)
        projection(0, 3) = -1.0f;
        projection(1, 3) = 1.0f;
    }

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    geometryShader->Bind();
    geometryShader->SetVector4f("rectColor", color);
    geometryShader->SetMatrix4f("projection", projection);
    geometryShader->SetInt("isCircle", 0);

    float vertices[6][4] = {
        { x, y, 0.0f, 0.0f },
        { x + width, y, 1.0f, 0.0f },
        { x + width, y + height, 1.0f, 1.0f },
        { x, y, 0.0f, 0.0f },
        { x + width, y + height, 1.0f, 1.0f },
        { x, y + height, 0.0f, 1.0f }
    };

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices); // 단일 Rect는 SubData 유지
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}

void GUISystem::RenderCircle(float x, float y, float radius, const Eigen::Vector4f& color)
{
    if (geometryShader == nullptr) return;

    float scrWidth = (float)Helium.GetWidth();
    float scrHeight = (float)Helium.GetHeight();

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    if (scrWidth > 0 && scrHeight > 0) {
        projection(0, 0) = 2.0f / scrWidth; projection(1, 1) = -2.0f / scrHeight; projection(0, 3) = -1.0f; projection(1, 3) = 1.0f;
    }

    glDisable(GL_CULL_FACE); glDisable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    geometryShader->Bind();
    geometryShader->SetVector4f("rectColor", color);
    geometryShader->SetMatrix4f("projection", projection);
    geometryShader->SetInt("isCircle", 1);

    float d = radius * 2.0f; float sx = x - radius; float sy = y - radius;
    float vertices[6][4] = {
        { sx, sy, 0.0f, 0.0f }, { sx + d, sy, 1.0f, 0.0f }, { sx + d, sy + d, 1.0f, 1.0f },
        { sx, sy, 0.0f, 0.0f }, { sx + d, sy + d, 1.0f, 1.0f }, { sx, sy + d, 0.0f, 1.0f }
    };

    glBindVertexArray(VAO); glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glBindBuffer(GL_ARRAY_BUFFER, 0); glBindVertexArray(0); glEnable(GL_DEPTH_TEST); glEnable(GL_CULL_FACE);
}

void GUISystem::RenderText(const std::string& text, float x, float y, float textWidth, float targetFontSize, const Eigen::Vector4f& color, TextHAlign hAlign, TextVAlign vAlign)
{
    if (textShader == nullptr) return;
    if (this->charData.empty()) return;

    float fontScale = targetFontSize / BASE_FONT_SIZE;

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    float scrWidth = (float)Helium.GetWidth();
    float scrHeight = (float)Helium.GetHeight();

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    if (scrWidth > 0 && scrHeight > 0)
    {
        projection(0, 0) = 2.0f / scrWidth; projection(1, 1) = -2.0f / scrHeight; projection(0, 3) = -1.0f; projection(1, 3) = 1.0f;
    }

    textShader->Bind();
    textShader->SetInt("text", 0);
    textShader->SetMatrix4f("projection", projection);
    textShader->SetVector4f("textColor", color);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    std::vector<uint32_t> codepoints = DecodeUTF8(text);

    float startX = x;
    if (hAlign == TextHAlign::Center) startX -= (textWidth * 0.5f);
    else if (hAlign == TextHAlign::Right) startX -= textWidth;

    float startY = y;
    float currentAscent = this->fontAscent * fontScale;
    float currentDescent = this->fontDescent * fontScale;

    switch (vAlign)
    {
    case TextVAlign::Top:    startY += currentAscent; break;
    case TextVAlign::Middle: startY += (currentAscent + currentDescent) * 0.5f; break;
    case TextVAlign::Bottom: startY += currentDescent; break;
    default: break;
    }

    float currentX = startX;
    float currentY = startY;
    const int TEX_W = 4096, TEX_H = 4096;

    // --- Batching Implementation ---
    std::vector<float> batchVertices;
    batchVertices.reserve(codepoints.size() * 6 * 4); // 6 vertices * 4 floats

    for (uint32_t cp : codepoints)
    {
        int idx = -1;
        if (cp >= 32 && cp < 128) idx = cp - 32;
        else if (cp >= 0xAC00 && cp <= 0xD7A3) idx = 96 + (cp - 0xAC00);

        if (idx == -1 || idx >= (int)this->charData.size()) continue;

        stbtt_aligned_quad q;
        float x_unused = 0, y_unused = 0;
        stbtt_GetPackedQuad(this->charData.data(), TEX_W, TEX_H, idx, &x_unused, &y_unused, &q, 0);

        float x0 = currentX + (q.x0 * fontScale);
        float x1 = currentX + (q.x1 * fontScale);
        float y0 = currentY + (q.y0 * fontScale);
        float y1 = currentY + (q.y1 * fontScale);

        // Quad Vertices (Triangles)
        float quad[24] = {
            x0, y0, q.s0, q.t0,
            x1, y0, q.s1, q.t0,
            x1, y1, q.s1, q.t1,
            x0, y0, q.s0, q.t0,
            x1, y1, q.s1, q.t1,
            x0, y1, q.s0, q.t1
        };

        batchVertices.insert(batchVertices.end(), std::begin(quad), std::end(quad));

        currentX += (this->charData[idx].xadvance * fontScale);
    }

    if (!batchVertices.empty())
    {
        // VBO 업데이트 후 한 번에 Draw
        glBufferSubData(GL_ARRAY_BUFFER, 0, batchVertices.size() * sizeof(float), batchVertices.data());
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(batchVertices.size() / 4));
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
}