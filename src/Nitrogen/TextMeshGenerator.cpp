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

namespace mapbox {
    namespace util {
        template <>
        struct nth<0, std::array<double, 2>> {
            inline static double get(const std::array<double, 2>& t) { return t[0]; };
        };
        template <>
        struct nth<1, std::array<double, 2>> {
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
    if (data.empty()) return false;

    fontBuffer = new unsigned char[data.size()];
    memcpy(fontBuffer, data.data(), data.size());

    if (!stbtt_InitFont(fontInfo, fontBuffer, 0)) return false;
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

        if (charMesh.NumVert() > 0) {
            charMeshes.push_back(charMesh);
        }

        int advance, lsb;
        stbtt_GetCodepointHMetrics(fontInfo, c, &advance, &lsb);
        xCursor += advance * scale;
    }

    if (charMeshes.empty()) return manifold::Manifold();

    // 첫 번째 글자부터 순서대로 합칩니다 (Union)
    manifold::Manifold result = charMeshes[0];
    for (size_t i = 1; i < charMeshes.size(); ++i) {
        result += charMeshes[i];
    }

    return result;
}

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

    // 3. Manifold MeshGL 데이터 준비 (Flat Arrays)
    // vertProperties: [x, y, z, x, y, z, ...]
    // triVerts: [i0, i1, i2, i0, i1, i2, ...]
    std::vector<float> vertProperties;
    std::vector<uint32_t> triVerts;

    // (1) Vertices 생성 (앞면 & 뒷면)
    // 외곽선 점들을 일렬로 평탄화
    std::vector<std::array<double, 2>> flattened;
    for (const auto& contour : contours) {
        for (const auto& p : contour) {
            flattened.push_back(p);

            // 앞면 (z = depth/2)
            vertProperties.push_back((float)p[0] + xOffset);
            vertProperties.push_back((float)p[1]);
            vertProperties.push_back(depth * 0.5f);
        }
    }

    int numPoints = (int)flattened.size();

    // 뒷면 Vertices 추가
    for (const auto& p : flattened) {
        // 뒷면 (z = -depth/2)
        vertProperties.push_back((float)p[0] + xOffset);
        vertProperties.push_back((float)p[1]);
        vertProperties.push_back(-depth * 0.5f);
    }

    // (2) Indices 생성 (앞면 & 뒷면 & 옆면)

    // 앞면/뒷면 (Ear Clipping 결과 활용)
    for (size_t i = 0; i < earIndices.size(); i += 3) {
        // 앞면 (CCW)
        triVerts.push_back(earIndices[i]);
        triVerts.push_back(earIndices[i + 1]);
        triVerts.push_back(earIndices[i + 2]);

        // 뒷면 (CW로 뒤집음, 인덱스에 numPoints 오프셋 더함)
        triVerts.push_back(numPoints + earIndices[i]);
        triVerts.push_back(numPoints + earIndices[i + 2]);
        triVerts.push_back(numPoints + earIndices[i + 1]);
    }

    // 옆면 (Stitching)
    int startIndex = 0;
    for (const auto& contour : contours) {
        int count = (int)contour.size();
        for (int i = 0; i < count; ++i) {
            int current = startIndex + i;
            int next = startIndex + (i + 1) % count;

            int currentBack = current + numPoints;
            int nextBack = next + numPoints;

            // Quad를 삼각형 2개로 분할
            // Tri 1
            triVerts.push_back(current);
            triVerts.push_back(next);
            triVerts.push_back(nextBack);
            // Tri 2
            triVerts.push_back(current);
            triVerts.push_back(nextBack);
            triVerts.push_back(currentBack);
        }
        startIndex += count;
    }

    // 4. Manifold 생성
    manifold::MeshGL meshGL;
    meshGL.numProp = 3; // Position(xyz) only
    meshGL.vertProperties = std::move(vertProperties);
    meshGL.triVerts = std::move(triVerts);

    return manifold::Manifold(meshGL);
}

void TextMeshGenerator::GetGlyphContours(int glyphIndex, float scale, std::vector<std::vector<std::array<double, 2>>>& outContours)
{
    stbtt_vertex* vertices;
    int numVerts = stbtt_GetGlyphShape(fontInfo, glyphIndex, &vertices);

    std::vector<std::array<double, 2>> currentContour;

    for (int i = 0; i < numVerts; ++i)
    {
        stbtt_vertex* v = &vertices[i];

        float x = v->x * scale;
        float y = v->y * scale;
        float cx = v->cx * scale;
        float cy = v->cy * scale;

        if (v->type == STBTT_vmove) {
            if (!currentContour.empty()) {
                outContours.push_back(currentContour);
                currentContour.clear();
            }
            currentContour.push_back({ (double)x, (double)y });
        }
        else if (v->type == STBTT_vline) {
            currentContour.push_back({ (double)x, (double)y });
        }
        else if (v->type == STBTT_vcurve) {
            // 2차 베지에
            auto lp = currentContour.back();
            float startX = (float)lp[0];
            float startY = (float)lp[1];

            const int segments = 5;
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
        outContours.push_back(currentContour);
    }

    stbtt_FreeShape(fontInfo, vertices);
}
