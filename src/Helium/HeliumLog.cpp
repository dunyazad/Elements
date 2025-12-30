#include "pch.h"

#include <Helium/HeliumLog.h>

static HeliumLogCallback g_LogCallback = nullptr;

void He_SetLogCallback(HeliumLogCallback cb)
{
    g_LogCallback = cb;
}

void He_LogInternal(HeliumLogLevel level, const char* key, char* message)
{
    if (!g_LogCallback) return;

    size_t len = strlen(message);
    while (len > 0 && (message[len - 1] == '\n' || message[len - 1] == '\r'))
    {
        message[len - 1] = '\0';
        len--;
    }

    g_LogCallback(level, key, message);
}

void He_Log(HeliumLogLevel level, const char* key, const char* fmt, ...)
{
    if (!g_LogCallback) return;
    char buffer[4096];
    memset(buffer, 0, sizeof(buffer));

    va_list args;
    va_start(args, fmt);
    _vsnprintf_s(buffer, _countof(buffer), _TRUNCATE, fmt, args);
    va_end(args);

    He_LogInternal(level, key, buffer);
}

void He_Log(const char* fmt, ...)
{
    if (!g_LogCallback) return;
    char buffer[4096];
    memset(buffer, 0, sizeof(buffer));

    va_list args;
    va_start(args, fmt);
    _vsnprintf_s(buffer, _countof(buffer), _TRUNCATE, fmt, args);
    va_end(args);

    He_LogInternal(HE_LOG_INFO, "", buffer);
}

void He_Log(const char* key, const char* fmt, ...)
{
    if (!g_LogCallback) return;
    char buffer[4096];
    memset(buffer, 0, sizeof(buffer));

    va_list args;
    va_start(args, fmt);
    _vsnprintf_s(buffer, _countof(buffer), _TRUNCATE, fmt, args);
    va_end(args);

    He_LogInternal(HE_LOG_INFO, key, buffer);
}

void He_Log(HeliumLogLevel level, const char* fmt, ...)
{
    if (!g_LogCallback) return;
    char buffer[4096];
    memset(buffer, 0, sizeof(buffer));

    va_list args;
    va_start(args, fmt);
    _vsnprintf_s(buffer, _countof(buffer), _TRUNCATE, fmt, args);
    va_end(args);

    He_LogInternal(level, "", buffer);
}
