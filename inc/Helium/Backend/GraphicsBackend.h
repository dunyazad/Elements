#pragma once

#include <Helium/HeliumCommon.h>

enum class BackendType
{
    OpenGL = 0,
    Vulkan = 1
};

class IGraphicsBackend
{
public:
    virtual ~IGraphicsBackend() = default;

    virtual bool Initialize(HWND hwnd) = 0;
    virtual void Resize(int width, int height) = 0;
    virtual void Update(float dt) = 0;
    virtual void Render() = 0;
    virtual void Shutdown() = 0;
};
