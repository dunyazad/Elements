#include "Monitor.h"
#include <vector>

#ifdef _WINDOWS

// -----------------------------------------------------------------------------
// Monitor enumeration callback
// -----------------------------------------------------------------------------
BOOL CALLBACK MonitorEnumProc(
    HMONITOR hMonitor,
    HDC,
    LPRECT,
    LPARAM dwData)
{
    auto* monitors =
        reinterpret_cast<std::vector<MonitorInfo>*>(dwData);

    MONITORINFO info;
    info.cbSize = sizeof(MONITORINFO);

    if (GetMonitorInfo(hMonitor, &info))
    {
        monitors->push_back({ hMonitor, info });
    }

    return TRUE;
}

// -----------------------------------------------------------------------------
// Ensure console buffer is large enough
// -----------------------------------------------------------------------------
static void EnsureConsoleBufferIsLargeEnough()
{
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return;

    CONSOLE_SCREEN_BUFFER_INFO info;
    if (!GetConsoleScreenBufferInfo(hOut, &info)) return;

    COORD newSize;
    newSize.X = max(info.dwSize.X, 200);
    newSize.Y = max(info.dwSize.Y, 200);

    if (newSize.X != info.dwSize.X ||
        newSize.Y != info.dwSize.Y)
    {
        SetConsoleScreenBufferSize(hOut, newSize);
    }
}

// -----------------------------------------------------------------------------
// Core: place window into N horizontal segments (WINDOWPLACEMENT)
// -----------------------------------------------------------------------------
static void SetWindowToScreenSegment(
    HWND hwnd,
    int monitorIndex,
    int segmentCount,
    int segmentIndex)
{
    if (!hwnd) return;
    if (segmentCount <= 0) return;
    if (segmentIndex < 0 || segmentIndex >= segmentCount) return;

    std::vector<MonitorInfo> monitors;
    EnumDisplayMonitors(
        NULL,
        NULL,
        MonitorEnumProc,
        reinterpret_cast<LPARAM>(&monitors)
    );

    if (monitorIndex < 0 ||
        monitorIndex >= static_cast<int>(monitors.size()))
        return;

    const RECT& workArea =
        monitors[monitorIndex].monitorInfo.rcWork;

    int screenWidth = workArea.right - workArea.left;
    int screenHeight = workArea.bottom - workArea.top;
    int segmentWidth = screenWidth / segmentCount;

    int x = workArea.left + segmentWidth * segmentIndex;
    int y = workArea.top;

    WINDOWPLACEMENT wp{};
    wp.length = sizeof(WINDOWPLACEMENT);
    wp.showCmd = SW_SHOWNORMAL;
    wp.rcNormalPosition.left = x;
    wp.rcNormalPosition.top = y;
    wp.rcNormalPosition.right = x + segmentWidth;
    wp.rcNormalPosition.bottom = y + screenHeight;

    SetWindowPlacement(hwnd, &wp);
}

// -----------------------------------------------------------------------------
// Console API
// -----------------------------------------------------------------------------
void MaximizeConsoleWindowOnMonitor(int monitorIndex)
{
    HWND hwnd = GetConsoleWindow();
    if (!hwnd) return;

    EnsureConsoleBufferIsLargeEnough();

    std::vector<MonitorInfo> monitors;
    EnumDisplayMonitors(
        NULL,
        NULL,
        MonitorEnumProc,
        reinterpret_cast<LPARAM>(&monitors)
    );

    if (monitorIndex < 0 ||
        monitorIndex >= static_cast<int>(monitors.size()))
        return;

    WINDOWPLACEMENT wp{};
    wp.length = sizeof(WINDOWPLACEMENT);
    wp.showCmd = SW_MAXIMIZE;
    wp.rcNormalPosition =
        monitors[monitorIndex].monitorInfo.rcWork;

    SetWindowPlacement(hwnd, &wp);
}

void SetConsoleToHalfOfScreen(int monitorIndex, int halfIndex)
{
    EnsureConsoleBufferIsLargeEnough();

    SetWindowToScreenSegment(
        GetConsoleWindow(),
        monitorIndex,
        2,
        halfIndex
    );
}

void SetConsoleToOneThirdOfScreen(int monitorIndex, int thirdIndex)
{
    EnsureConsoleBufferIsLargeEnough();

    SetWindowToScreenSegment(
        GetConsoleWindow(),
        monitorIndex,
        3,
        thirdIndex
    );
}

#endif // _WINDOWS
