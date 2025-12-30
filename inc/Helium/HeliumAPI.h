#pragma once

#ifdef HELIUM_EXPORTS
#define HELIUM_API __declspec(dllexport)
#else
#define HELIUM_API __declspec(dllimport)
#endif

typedef void(*HeliumLogCallback)(const char* key, const char* message);

extern "C"
{
    HELIUM_API bool Helium_Initialize(HWND hwnd);
    HELIUM_API void Helium_Resize(int width, int height);
    HELIUM_API void Helium_Render();
    HELIUM_API void Helium_Shutdown();

    HELIUM_API void Helium_CreateConsole();

    HELIUM_API void Helium_SetLogCallback(HeliumLogCallback cb);
}

void Helium_Log(const char* fmt, ...);
void Helium_Log(const char* key, const char* fmt, ...);