#pragma once

#ifdef HELIUM_EXPORTS
#define HELIUM_API __declspec(dllexport)
#else
#define HELIUM_API __declspec(dllimport)
#endif

extern "C"
{
    HELIUM_API bool Helium_Initialize(HWND hwnd);
    HELIUM_API void Helium_Resize(int width, int height);
    HELIUM_API void Helium_Render();
    HELIUM_API void Helium_Shutdown();

    HELIUM_API void Helium_CreateConsole();
}
