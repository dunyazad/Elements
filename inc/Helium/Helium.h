#pragma once

#ifdef HELIUM_EXPORTS
#define HELIUM_API __declspec(dllexport)
#else
#define HELIUM_API __declspec(dllimport)
#endif

#include <Helium/HeliumLog.h>

extern "C"
{
    HELIUM_API bool He_Initialize(HWND hwnd, int backendType);

    HELIUM_API void He_Resize(int width, int height);
    
    HELIUM_API void He_Update(float dt);
    
    HELIUM_API void He_Render();
    
    HELIUM_API void He_Shutdown();

    HELIUM_API void He_CreateConsole();

    HELIUM_API void He_SetLogCallback(HeliumLogCallback cb);

    HELIUM_API void He_ProcessMouseWheel(float xoffset, float yoffset);

    HELIUM_API void He_TestCreateEntity();
}
