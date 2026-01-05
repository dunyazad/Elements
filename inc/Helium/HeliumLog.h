#pragma once

enum HeliumLogLevel
{
	HE_LOG_INFO = 0,
	HE_LOG_WARN = 1,
	HE_LOG_ERROR = 2,
	HE_LOG_DEBUG = 3
};

typedef void(*HeliumLogCallback)(int level, const char* key, const char* value);

//void He_Log(const char* fmt, ...);
//void He_Log(const char* key, const char* fmt, ...);
//void He_Log(HeliumLogLevel level, const char* fmt, ...);
void He_Log(HeliumLogLevel level, const char* key, const char* fmt, ...);

#define InfoLog(key, ...)    He_Log(HE_LOG_INFO,  key, __VA_ARGS__)
#define WarningLog(key, ...) He_Log(HE_LOG_WARN,  key, __VA_ARGS__)
#define ErrorLog(key, ...)   He_Log(HE_LOG_ERROR, key, __VA_ARGS__)
#define DebugLog(key, ...)   He_Log(HE_LOG_DEBUG, key, __VA_ARGS__)
