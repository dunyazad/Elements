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

void He_ProcessMouseWheel(float xoffset, float yoffset)
{
	auto inputSystem = Helium.GetInputSystem();
	if (inputSystem)
	{
		inputSystem->OnMouseWheel(xoffset, yoffset);
	}
}

int He_LoadPointCloudFromPLY(const char* filename, const char* name)
{
	return Helium.LoadPointCloudFromPLY(filename, name);
}

bool He_PointCloudSelect(int ID)
{
	return Helium.SelectPointCloud(ID);
}

bool He_PointCloudSetVisible(int ID, bool isVisible)
{
	auto pointCloud = Helium.GetPointCloud(ID);
	if (pointCloud)
	{
		pointCloud->SetVisible(isVisible);
		return true;
	}
	return false;
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
