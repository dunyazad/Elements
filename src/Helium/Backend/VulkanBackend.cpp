#include "pch.h"
#include <Helium/Backend/VulkanBackend.h>

extern void He_Log(const char* fmt, ...);

bool VulkanBackend::Initialize(HWND hwnd)
{
    hWnd = hwnd;
    He_Log(HE_LOG_INFO, "", "Initializing Vulkan Backend... (Not implemented fully yet)");

    // TODO: 
    // 1. Create Vulkan Instance
    // 2. Create Surface (Win32)
    // 3. Pick Physical Device
    // 4. Create Logical Device & Queue
    // 5. Create Swapchain

    return true;
}

void VulkanBackend::Resize(int width, int height)
{
    this->width = width;
    this->height = height;

    // TODO: Recreate Swapchain
}

void VulkanBackend::Update(float dt)
{
	// TODO: Update Logic
}

void VulkanBackend::Render()
{
    // TODO: Acquire Image -> Submit Command Buffer -> Present
}

void VulkanBackend::Clear(float r, float g, float b, float a)
{

}

void VulkanBackend::DrawScreenQuad()
{

}

void VulkanBackend::Shutdown()
{
    // TODO: Destroy Vulkan Resources
}