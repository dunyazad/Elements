#pragma once
#include "GraphicsBackend.h"

class OpenGLBackend : public IGraphicsBackend
{
public:
    bool Initialize(HWND hwnd) override;
    void Resize(int width, int height) override;
    void Update(float dt) override;
    void Render() override;
    void Clear(float r, float g, float b, float a) override;
    void DrawScreenQuad() override;
    void Shutdown() override;

private:
    HDC m_hDC = nullptr;
    HGLRC m_hRC = nullptr;

    unsigned int m_QuadVAO = 0;
    unsigned int m_QuadVBO = 0;
    unsigned int m_QuadEBO = 0;

    void InitializeScreenQuad();
};
