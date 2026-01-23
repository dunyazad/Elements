#pragma once

#include <vector>

#ifdef _WINDOWS
#include <windows.h>

struct MonitorInfo
{
    HMONITOR    hMonitor;
    MONITORINFO monitorInfo;
};

inline BOOL CALLBACK MonitorEnumProc(
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

static void EnsureConsoleBufferIsLargeEnough()
{
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return;

    CONSOLE_SCREEN_BUFFER_INFO info;
    if (!GetConsoleScreenBufferInfo(hOut, &info)) return;

    COORD newSize;
    newSize.X = info.dwSize.X < 200 ? info.dwSize.X : 200;
    newSize.Y = info.dwSize.Y < 200 ? info.dwSize.Y : 200;

    if (newSize.X != info.dwSize.X ||
        newSize.Y != info.dwSize.Y)
    {
        SetConsoleScreenBufferSize(hOut, newSize);
    }
}

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

inline void MaximizeWindowOnMonitor(HWND hwnd, int monitorIndex)
{
    std::vector<MonitorInfo> monitors;
    EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, reinterpret_cast<LPARAM>(&monitors));

    if (monitorIndex >= 0 && monitorIndex < monitors.size()) {
        const MonitorInfo& monitor = monitors[monitorIndex];
        RECT workArea = monitor.monitorInfo.rcWork;

        // Set the position and size of the VTK window to match the monitor's work area
        //renderWindow->SetPosition(workArea.left, workArea.top);
        //renderWindow->SetSize(workArea.right - workArea.left, workArea.bottom - workArea.top);

        MoveWindow(hwnd, workArea.left, workArea.top,
            workArea.right - workArea.left,
            workArea.bottom - workArea.top, TRUE);

        ShowWindow(hwnd, SW_MAXIMIZE);
    }
}

inline void MaximizeConsoleWindowOnMonitor(int monitorIndex)
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

inline void SetConsoleToHalfOfScreen(int monitorIndex, int halfIndex)
{
    EnsureConsoleBufferIsLargeEnough();

    SetWindowToScreenSegment(
        GetConsoleWindow(),
        monitorIndex,
        2,
        halfIndex
    );
}

inline void SetConsoleToOneThirdOfScreen(int monitorIndex, int thirdIndex)
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
