#pragma once
#include "GraphicsBackend.h"

// Vulkan 헤더 필요 (pch.h에 포함하거나 여기서 포함)
// #include <vulkan/vulkan.h> 

class VulkanBackend : public IGraphicsBackend
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
    HWND hWnd = nullptr;
    // VkInstance, VkDevice, VkSwapchainKHR 등이 들어갈 자리
};