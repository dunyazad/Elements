#include "pch.h"

#include <glad/glad.h>
#include <gl/GL.h>
#include <Monitor.h>

#include <Helium/Backend/GraphicsBackend.h>
#include <Helium/Backend/OpenGLBackend.h>
#include <Helium/Backend/VulkanBackend.h>

#include <Helium/HeliumCore.h>
#include <Helium/PointCloud.h>

bool He_Initialize(HWND hwnd, int backendType)
{
	return Helium.Initialize(hwnd, backendType);
}

void He_Resize(int width, int height)
{
	Helium.Resize(width, height);
}

HELIUM_API void He_Update(float dt)
{
	Helium.Update(dt);
}

void He_Render()
{
	Helium.Render();
}

void He_Shutdown()
{
	Helium.Shutdown();
}

void He_CreateConsole()
{
	AllocConsole();
	FILE* fp;
	freopen_s(&fp, "CONOUT$", "w", stdout);
	freopen_s(&fp, "CONOUT$", "w", stderr);
}

HELIUM_API void He_ProcessMessage(UINT message, WPARAM wParam, LPARAM lParam)
{
	Helium.ProcessMessage(message, wParam, lParam);
}

void He_ProcessMouseWheel(float xoffset, float yoffset)
{
	auto inputSystem = Helium.GetInputSystem();
	if (inputSystem)
	{
		inputSystem->OnMouseWheel(xoffset, yoffset);
	}
}

static PointCloudCreatedCallback g_PointCloudCreatedCallback = nullptr;

void He_SetPointCloudCreatedCallback(PointCloudCreatedCallback callback)
{
	g_PointCloudCreatedCallback = callback;
}

void OnPointCloudCreated(int id, const std::string& fileName, const std::string& name)
{
	if (g_PointCloudCreatedCallback)
	{
		g_PointCloudCreatedCallback(id, fileName.c_str(), name.c_str());
	}
}

static PointCloudDeletedCallback g_PointCloudDeletedCallback = nullptr;

void He_SetPointCloudDeletedCallback(PointCloudDeletedCallback callback)
{
	g_PointCloudDeletedCallback = callback;
}

void OnPointCloudDeleted(int id)
{
	if (g_PointCloudDeletedCallback)
	{
		g_PointCloudDeletedCallback(id);
	}
}

bool He_ExecuteCommand(const char* command)
{
	return Helium.ExecuteCommand(command);
}
