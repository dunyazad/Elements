#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS

#define NOMINMAX 
#include <Windows.h>

#include "TextMeshGenerator.h"
#include <Helium/HeliumCore.h>

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb/stb_truetype.h>

// mapbox/earcut.hpp 경로가 올바르게 설정되어 있어야 합니다.
#include <mapbox/earcut.hpp> 

#include <numeric> // for std::abs

namespace mapbox {
    namespace util {
        template <> struct nth<0, std::array<double, 2>> {
            inline static double get(const std::array<double, 2>& t) { return t[0]; };
        };
        template <> struct nth<1, std::array<double, 2>> {
            inline static double get(const std::array<double, 2>& t) { return t[1]; };
        };
    }
}

TextMeshGenerator::TextMeshGenerator() {
    fontInfo = new stbtt_fontinfo();
}

TextMeshGenerator::~TextMeshGenerator() {
    if (fontBuffer) delete[] fontBuffer;
    if (fontInfo) delete fontInfo;
}

bool TextMeshGenerator::LoadFont(const std::string& path) {
    auto file = File(path, true);
    auto data = file.ReadAllBytes();

    // [디버그] 폰트 로드 실패 시 로그
    if (data.empty()) {
        printf("!!! [GEN_ERROR] Font File Empty: %s\n", path.c_str());
        return false;
    }

    if (fontBuffer) delete[] fontBuffer;
    fontBuffer = new unsigned char[data.size()];
    memcpy(fontBuffer, data.data(), data.size());

    if (!stbtt_InitFont(fontInfo, fontBuffer, 0)) {
        printf("!!! [GEN_ERROR] stbtt_InitFont Failed!\n");
        return false;
    }
    return true;
}

manifold::Manifold TextMeshGenerator::Create3DText(const std::string& text, float depth, float scale)
{
    std::vector<manifold::Manifold> charMeshes;
    float xCursor = 0.0f;

    for (char c : text)
    {
        if (c == ' ') {
            int advance, lsb;
            stbtt_GetCodepointHMetrics(fontInfo, c, &advance, &lsb);
            xCursor += advance * scale;
            continue;
        }

        manifold::Manifold charMesh = CreateCharacter(c, depth, scale, xCursor);

        // 생성 성공한 것만 추가
        if (charMesh.NumVert() > 0) {
            charMeshes.push_back(charMesh);
        }
        else {
            // 실패 시 경고 (디버깅용)
            printf("!!! [GEN_WARN] Failed to generate mesh for char '%c' (Verts: 0)\n", c);
        }

        int advance, lsb;
        stbtt_GetCodepointHMetrics(fontInfo, c, &advance, &lsb);
        xCursor += advance * scale;
    }

    if (charMeshes.empty()) return manifold::Manifold();

    manifold::Manifold result = charMeshes[0];
    for (size_t i = 1; i < charMeshes.size(); ++i) {
        result += charMeshes[i];
    }
    return result;
}

// [핵심] 글자 생성 및 방향 보정
manifold::Manifold TextMeshGenerator::CreateCharacter(int charCode, float depth, float scale, float xOffset)
{
    int glyphIndex = stbtt_FindGlyphIndex(fontInfo, charCode);
    if (glyphIndex == 0) return manifold::Manifold();

    // 1. 외곽선 추출
    std::vector<std::vector<std::array<double, 2>>> contours;
    GetGlyphContours(glyphIndex, scale, contours);
    if (contours.empty()) return manifold::Manifold();

    // 2. 삼각형화 (Earcut)
    std::vector<uint32_t> earIndices = mapbox::earcut<uint32_t>(contours);
    if (earIndices.empty()) return manifold::Manifold();

    // 3. Vertices 생성 (좌표)
    std::vector<float> vertProperties;
    std::vector<std::array<double, 2>> flattened;

    // 외곽선 평탄화
    for (const auto& contour : contours) {
        for (const auto& p : contour) {
            flattened.push_back(p);
            // 앞면 (Z+)
            vertProperties.push_back((float)p[0] + xOffset);
            vertProperties.push_back((float)p[1]);
            vertProperties.push_back(depth * 0.5f);
        }
    }
    int numPoints = (int)flattened.size();

    // 뒷면 (Z-)
    for (const auto& p : flattened) {
        vertProperties.push_back((float)p[0] + xOffset);
        vertProperties.push_back((float)p[1]);
        vertProperties.push_back(-depth * 0.5f);
    }

    // [필승 전략] 두 가지 방향(정방향/역방향)을 모두 시도합니다.
    // Manifold는 방향이 맞으면 객체를 반환하고, 틀리면 빈 객체(0 Verts)를 반환합니다.

    // Case 1: 정방향 (CCW 가정) 시도
    {
        std::vector<uint32_t> triVerts;

        // 뚜껑
        for (size_t i = 0; i < earIndices.size(); i += 3) {
            triVerts.push_back(earIndices[i]);
            triVerts.push_back(earIndices[i + 1]);
            triVerts.push_back(earIndices[i + 2]);

            // 뒷면
            triVerts.push_back(numPoints + earIndices[i]);
            triVerts.push_back(numPoints + earIndices[i + 2]);
            triVerts.push_back(numPoints + earIndices[i + 1]);
        }
        // 옆면
        int startIndex = 0;
        for (const auto& contour : contours) {
            int count = (int)contour.size();
            for (int i = 0; i < count; ++i) {
                int curr = startIndex + i;
                int next = startIndex + (i + 1) % count;
                triVerts.push_back(curr); triVerts.push_back(next); triVerts.push_back(next + numPoints);
                triVerts.push_back(curr); triVerts.push_back(next + numPoints); triVerts.push_back(curr + numPoints);
            }
            startIndex += count;
        }

        manifold::MeshGL meshGL;
        meshGL.numProp = 3;
        meshGL.vertProperties = vertProperties; // 복사
        meshGL.triVerts = triVerts;

        manifold::Manifold m(meshGL);

        // 성공했으면 리턴
        if (m.NumVert() > 0) return m;

        // 실패했으면 로그 찍고 역방향 시도
        // printf(">>> [GEN_INFO] CCW failed for char %d. Retrying with CW...\n", charCode);
    }

    // Case 2: 역방향 (CW 가정) 시도 - 인덱스 순서 뒤집기
    {
        std::vector<uint32_t> triVerts;

        // 뚜껑 (순서를 0, 2, 1로 뒤집음)
        for (size_t i = 0; i < earIndices.size(); i += 3) {
            triVerts.push_back(earIndices[i]);
            triVerts.push_back(earIndices[i + 2]); // Swap
            triVerts.push_back(earIndices[i + 1]); // Swap

            // 뒷면 (뒷면도 반대로 뒤집음)
            triVerts.push_back(numPoints + earIndices[i]);
            triVerts.push_back(numPoints + earIndices[i + 1]); // Swap
            triVerts.push_back(numPoints + earIndices[i + 2]); // Swap
        }
        // 옆면 (순서 뒤집기)
        int startIndex = 0;
        for (const auto& contour : contours) {
            int count = (int)contour.size();
            for (int i = 0; i < count; ++i) {
                int curr = startIndex + i;
                int next = startIndex + (i + 1) % count;
                // 이전과 반대 순서로 삼각형 구성
                triVerts.push_back(curr); triVerts.push_back(next + numPoints); triVerts.push_back(next);
                triVerts.push_back(curr); triVerts.push_back(curr + numPoints); triVerts.push_back(next + numPoints);
            }
            startIndex += count;
        }

        manifold::MeshGL meshGL;
        meshGL.numProp = 3;
        meshGL.vertProperties = vertProperties;
        meshGL.triVerts = triVerts;

        // 이것도 실패하면 진짜 답 없는 데이터임
        return manifold::Manifold(meshGL);
    }
}

void TextMeshGenerator::GetGlyphContours(int glyphIndex, float scale, std::vector<std::vector<std::array<double, 2>>>& outContours)
{
    stbtt_vertex* vertices;
    int numVerts = stbtt_GetGlyphShape(fontInfo, glyphIndex, &vertices);
    if (numVerts <= 0) return;

    std::vector<std::array<double, 2>> currentContour;

    for (int i = 0; i < numVerts; ++i)
    {
        stbtt_vertex* v = &vertices[i];

        // [수정] 복잡한 Y반전 제거하고 그냥 원본대로 받습니다.
        // 방향은 위에서 Brute Force로 맞춥니다.
        float x = v->x * scale;
        float y = v->y * scale;
        float cx = v->cx * scale;
        float cy = v->cy * scale;

        if (v->type == STBTT_vmove) {
            if (!currentContour.empty()) {
                // 중복 점 제거 (안정성)
                if (currentContour.size() > 2) {
                    auto& first = currentContour.front();
                    auto& last = currentContour.back();
                    if (std::abs(first[0] - last[0]) < 1e-5 && std::abs(first[1] - last[1]) < 1e-5) {
                        currentContour.pop_back();
                    }
                }
                outContours.push_back(currentContour);
                currentContour.clear();
            }
            currentContour.push_back({ (double)x, (double)y });
        }
        else if (v->type == STBTT_vline) {
            currentContour.push_back({ (double)x, (double)y });
        }
        else if (v->type == STBTT_vcurve) {
            auto lp = currentContour.back();
            float startX = (float)lp[0];
            float startY = (float)lp[1];

            const int segments = 3;
            for (int j = 1; j <= segments; ++j) {
                float t = (float)j / (float)segments;
                float invT = 1.0f - t;
                float px = (invT * invT * startX) + (2 * invT * t * cx) + (t * t * x);
                float py = (invT * invT * startY) + (2 * invT * t * cy) + (t * t * y);
                currentContour.push_back({ (double)px, (double)py });
            }
        }
    }

    if (!currentContour.empty()) {
        if (currentContour.size() > 2) {
            auto& first = currentContour.front();
            auto& last = currentContour.back();
            if (std::abs(first[0] - last[0]) < 1e-5 && std::abs(first[1] - last[1]) < 1e-5) {
                currentContour.pop_back();
            }
        }
        outContours.push_back(currentContour);
    }
    stbtt_FreeShape(fontInfo, vertices);
}
