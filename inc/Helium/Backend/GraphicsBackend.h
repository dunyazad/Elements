#pragma once

#include <Helium/HeliumCommon.h>

enum class HELIUM_API BackendType
{
    OpenGL = 0,
    Vulkan = 1
};

class HELIUM_API IGraphicsBackend
{
public:
    virtual ~IGraphicsBackend() = default;

    virtual bool Initialize(HWND hwnd) = 0;
    virtual void Resize(int width, int height) = 0;
    virtual void Update(float dt) = 0;
    virtual void Render() = 0;
    virtual void Clear(float r, float g, float b, float a) = 0;
    virtual void DrawScreenQuad() = 0;
    virtual void Shutdown() = 0;

	inline int GetWidth() const { return width; }
	inline int GetHeight() const { return height; }

protected:
	int width = 0;
	int height = 0;
};
