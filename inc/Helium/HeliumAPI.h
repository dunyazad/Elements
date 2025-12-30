#pragma once

#ifdef HELIUM_EXPORTS
#define HELIUM_API __declspec(dllexport)
#else
#define HELIUM_API __declspec(dllimport)
#endif

enum HeliumLogLevel
{
    HELIUM_LOG_INFO = 0,
    HELIUM_LOG_WARN = 1,
    HELIUM_LOG_ERROR = 2,
    HELIUM_LOG_DEBUG = 3
};

typedef void(*HeliumLogCallback)(
    int level,
    const char* key,
    const char* value);

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
void Helium_Log(HeliumLogLevel level, const char* fmt, ...);
void Helium_Log(HeliumLogLevel level, const char* key, const char* fmt, ...);