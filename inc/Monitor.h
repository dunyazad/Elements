#pragma once

#ifdef _WINDOWS
#include <windows.h>

// Monitor info structure
struct MonitorInfo
{
    HMONITOR    hMonitor;
    MONITORINFO monitorInfo;
};

// Monitor enumeration callback
BOOL CALLBACK MonitorEnumProc(
    HMONITOR hMonitor,
    HDC hdcMonitor,
    LPRECT lprcMonitor,
    LPARAM dwData
);

// Console window helpers
void MaximizeConsoleWindowOnMonitor(int monitorIndex);
void SetConsoleToHalfOfScreen(int monitorIndex, int halfIndex);
void SetConsoleToOneThirdOfScreen(int monitorIndex, int thirdIndex);

#endif // _WINDOWS
