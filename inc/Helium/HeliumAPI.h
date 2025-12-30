#pragma once

#ifdef HELIUM_EXPORTS
#define HELIUM_API __declspec(dllexport)
#else
#define HELIUM_API __declspec(dllimport)
#endif

enum HeliumLogLevel
{
    HE_LOG_INFO = 0,
    HE_LOG_WARN = 1,
    HE_LOG_ERROR = 2,
    HE_LOG_DEBUG = 3
};

typedef void(*HeliumLogCallback)(
    int level,
    const char* key,
    const char* value);

extern "C"
{
    HELIUM_API bool He_Initialize(HWND hwnd);
    HELIUM_API void He_Resize(int width, int height);
    HELIUM_API void He_Render();
    HELIUM_API void He_Shutdown();

    HELIUM_API void He_CreateConsole();

    HELIUM_API void He_SetLogCallback(HeliumLogCallback cb);
}

void He_Log(const char* fmt, ...);
void He_Log(const char* key, const char* fmt, ...);
void He_Log(HeliumLogLevel level, const char* fmt, ...);
void He_Log(HeliumLogLevel level, const char* key, const char* fmt, ...);