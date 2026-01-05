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

bool He_LoadPointCloudsFromPLY(const char* filename, int ID)
{
	return Helium.LoadPointCloudsFromPLY(filename, ID);
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
