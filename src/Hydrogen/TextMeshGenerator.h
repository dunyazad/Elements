#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <manifold/manifold.h>
#include <manifold/cross_section.h>

// 폰트 파일 로딩을 위한 포워드 선언
struct stbtt_fontinfo;

class TextMeshGenerator
{
public:
    TextMeshGenerator();
    ~TextMeshGenerator();

    bool LoadFont(const std::string& path);

    // 텍스트를 3D Manifold 객체로 변환 (Boolean 연산용)
    manifold::Manifold Create3DText(const std::string& text, float depth = 1.0f, float scale = 1.0f);

private:
    // 글자 하나를 3D로 만듦
    manifold::Manifold CreateCharacter(int charCode, float depth, float scale, float xOffset);

    // 베지에 곡선을 점들의 집합으로 변환
    void GetGlyphContours(int glyphIndex, float scale, std::vector<std::vector<std::array<double, 2>>>& outContours);

private:
    unsigned char* fontBuffer = nullptr;
    stbtt_fontinfo* fontInfo = nullptr;
};
