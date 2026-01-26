#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define NOMINMAX 
#include <Windows.h>

#include <Helium/HeliumCore.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#define STB_TRUETYPE_IMPLEMENTATION 
#include <stb/stb_truetype.h>
#include <mapbox/earcut.hpp>
#include "TextMeshGenerator.h"
#include <numeric>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>

TextMeshGenerator::TextMeshGenerator() {
    fontInfo = new stbtt_fontinfo();
    fontBuffer = nullptr;
}

TextMeshGenerator::~TextMeshGenerator() {
    if (fontBuffer) delete[] fontBuffer;
    if (fontInfo) delete fontInfo;
}

int GetUtf8Codepoint(const std::string& text, size_t& i) {
    unsigned char c = (unsigned char)text[i];
    int codepoint = 0;

    if (c < 0x80) {
        // 1 byte (ASCII)
        codepoint = c;
    }
    else if ((c & 0xE0) == 0xC0) {
        // 2 bytes
        if (i + 1 >= text.length()) return 0;
        codepoint = ((c & 0x1F) << 6) | ((unsigned char)text[i + 1] & 0x3F);
        i += 1;
    }
    else if ((c & 0xF0) == 0xE0) {
        // 3 bytes (Korean is usually here)
        if (i + 2 >= text.length()) return 0;
        codepoint = ((c & 0x0F) << 12) |
            (((unsigned char)text[i + 1] & 0x3F) << 6) |
            ((unsigned char)text[i + 2] & 0x3F);
        i += 2;
    }
    else if ((c & 0xF8) == 0xF0) {
        // 4 bytes
        if (i + 3 >= text.length()) return 0;
        codepoint = ((c & 0x07) << 18) |
            (((unsigned char)text[i + 1] & 0x3F) << 12) |
            (((unsigned char)text[i + 2] & 0x3F) << 6) |
            ((unsigned char)text[i + 3] & 0x3F);
        i += 3;
    }
    return codepoint;
}

bool TextMeshGenerator::LoadFont(const std::string& path) {
    printf(">>> [DEBUG] Loading Font: %s\n", path.c_str());
    auto file = File(path, true);
    auto data = file.ReadAllBytes();
    if (data.empty()) {
        printf("!!! [ERROR] Failed to read font file.\n");
        return false;
    }
    if (fontBuffer) delete[] fontBuffer;
    fontBuffer = new unsigned char[data.size()];
    memcpy(fontBuffer, data.data(), data.size());
    return stbtt_InitFont(fontInfo, fontBuffer, 0) != 0;
}

// -----------------------------------------------------------------------------
// [수정] UTF-8 디코딩 적용 및 글자 루프 개선
// -----------------------------------------------------------------------------
manifold::Manifold TextMeshGenerator::Create3DText(const std::string& text, float depth, float scale)
{
    printf("\n========================================\n");
    printf(">>> [START] Create 3D Text (Length: %d, Scale: %f)\n", (int)text.length(), scale);
    printf("========================================\n");

    std::vector<manifold::Manifold> charMeshes;
    float xCursor = 0.0f;

    // [수정] 루프에서 직접 i를 증가시키지 않고, GetUtf8Codepoint 내부에서 바이트 수만큼 증가시킴
    for (size_t i = 0; i < text.length(); ++i) {

        // UTF-8 디코딩: 여기서 i가 글자 바이트 수만큼 점프합니다.
        int codepoint = GetUtf8Codepoint(text, i);

        if (codepoint == 0) continue; // 잘못된 문자 무시

        // 공백 처리 (Space)
        if (codepoint == 32) {
            int adv, lsb;
            stbtt_GetCodepointHMetrics(fontInfo, codepoint, &adv, &lsb);
            xCursor += adv * scale;
            continue;
        }

        printf("\n--- Char Code: %d (Unicode) ---\n", codepoint);

        // [수정] char c 대신 int codepoint를 넘깁니다.
        manifold::Manifold m = CreateCharacter(codepoint, depth, scale, xCursor);

        if (m.NumVert() > 0) {
            charMeshes.push_back(m);
            printf("  [SUCCESS] Verts: %d\n", m.NumVert());
        }
        else {
            printf("  [FAIL] Verts: 0 (Check if font supports this char)\n");
        }

        int adv, lsb;
        stbtt_GetCodepointHMetrics(fontInfo, codepoint, &adv, &lsb);
        xCursor += adv * scale;
    }

    if (charMeshes.empty()) {
        printf("!!! [ERROR] No meshes generated.\n");
        return manifold::Manifold();
    }

    printf(">>> [COMPOSE] Merging %d meshes...\n", (int)charMeshes.size());
    return manifold::Manifold::Compose(charMeshes);
}

manifold::Manifold TextMeshGenerator::CreateCharacter(int charCode, float depth, float scale, float xOffset)
{
    // [중요] 한글은 폰트에 없으면 Index가 0이 나옵니다.
    int glyphIndex = stbtt_FindGlyphIndex(fontInfo, charCode);
    if (glyphIndex == 0) {
        printf("    [WARN] Glyph not found for code %d\n", charCode);
        return manifold::Manifold();
    }

    stbtt_vertex* vertices;
    int numVerts = stbtt_GetGlyphShape(fontInfo, glyphIndex, &vertices);

    if (numVerts == 0) return manifold::Manifold();

    std::vector<std::vector<manifold::vec2>> contours;
    std::vector<manifold::vec2> currentContour;

    for (int i = 0; i < numVerts; ++i) {
        stbtt_vertex* v = &vertices[i];

        float x = v->x * scale;
        float y = v->y * scale;
        float cx = v->cx * scale;
        float cy = v->cy * scale;

        if (v->type == STBTT_vmove) {
            if (!currentContour.empty()) {
                contours.push_back(currentContour);
                currentContour.clear();
            }
            currentContour.push_back({ x, y });
        }
        else if (v->type == STBTT_vline) {
            currentContour.push_back({ x, y });
        }
        else if (v->type == STBTT_vcurve) {
            if (currentContour.empty()) continue;

            manifold::vec2 lastP = currentContour.back();
            const int segments = 5;
            for (int j = 1; j <= segments; ++j) {
                float t = j / (float)segments;
                float invT = 1.0f - t;

                float px = (invT * invT * lastP.x) + (2 * invT * t * cx) + (t * t * x);
                float py = (invT * invT * lastP.y) + (2 * invT * t * cy) + (t * t * y);
                currentContour.push_back({ static_cast<float>(px), static_cast<float>(py) });
            }
        }
    }
    if (!currentContour.empty()) contours.push_back(currentContour);
    stbtt_FreeShape(fontInfo, vertices);

    if (contours.empty()) return manifold::Manifold();

    manifold::CrossSection cs(contours, manifold::CrossSection::FillRule::EvenOdd);
    cs = cs.Simplify();

    manifold::Manifold charMesh = manifold::Manifold::Extrude(cs.ToPolygons(), depth);
    charMesh = charMesh.Translate({ xOffset, 0.0f, 0.0f });

    if (charMesh.NumVert() < 3) {
        return manifold::Manifold();
    }

    return charMesh;
}
