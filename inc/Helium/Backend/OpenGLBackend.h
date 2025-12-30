#pragma once
#include "GraphicsBackend.h"

class OpenGLBackend : public IGraphicsBackend
{
public:
    bool Initialize(HWND hwnd) override;
    void Resize(int width, int height) override;
    void Update(float dt) override;
    void Render() override;
    void Shutdown() override;

private:
    HDC m_hDC = nullptr;
    HGLRC m_hRC = nullptr;
};
