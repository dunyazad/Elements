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
    HDC hDC = nullptr;
    HGLRC hRC = nullptr;

    unsigned int quadVAO = 0;
    unsigned int quadVBO = 0;
    unsigned int quadEBO = 0;

    void InitializeScreenQuad();
};
