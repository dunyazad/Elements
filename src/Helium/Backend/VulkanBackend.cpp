#include "pch.h"
#include <Helium/Backend/VulkanBackend.h>

extern void He_Log(const char* fmt, ...);

bool VulanBackend::Initialize(HWND hwnd)
{
    m_hWnd = hwnd;
    He_Log("Initializing Vulkan Backend... (Not implemented fully yet)");

    // TODO: 
    // 1. Create Vulkan Instance
    // 2. Create Surface (Win32)
    // 3. Pick Physical Device
    // 4. Create Logical Device & Queue
    // 5. Create Swapchain

    return true;
}

void VulanBackend::Resize(int width, int height)
{
    // TODO: Recreate Swapchain
}

void VulanBackend::Update(float dt)
{
	// TODO: Update Logic
}

void VulanBackend::Render()
{
    // TODO: Acquire Image -> Submit Command Buffer -> Present
}

void VulanBackend::Shutdown()
{
    // TODO: Destroy Vulkan Resources
}