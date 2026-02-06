#pragma once

#include <vector>

// _WINDOWS 매크로가 없어도 빌드되도록 _WIN32 체크 추가
#if defined(_WINDOWS) || defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cmath> // abs 사용을 위해

// DWM 함수 포인터 및 상수 정의 (dwmapi.lib 의존성 제거)
typedef HRESULT(WINAPI* PDwmGetWindowAttribute)(HWND, DWORD, PVOID, DWORD);

#ifndef DWMWA_EXTENDED_FRAME_BOUNDS
#define DWMWA_EXTENDED_FRAME_BOUNDS 9
#endif

struct MonitorInfo
{
    HMONITOR    hMonitor;
    MONITORINFO monitorInfo;
};

// 헤더에 정의되므로 중복 정의 에러 방지를 위해 모든 함수에 inline 키워드 적용

inline BOOL CALLBACK MonitorEnumProc(HMONITOR hMonitor, HDC, LPRECT, LPARAM dwData)
{
    auto* monitors = reinterpret_cast<std::vector<MonitorInfo>*>(dwData);
    MONITORINFO info;
    info.cbSize = sizeof(MONITORINFO);

    if (GetMonitorInfo(hMonitor, &info))
    {
        monitors->push_back({ hMonitor, info });
    }
    return TRUE;
}

inline void EnsureConsoleBufferIsLargeEnough()
{
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return;

    CONSOLE_SCREEN_BUFFER_INFO info;
    if (!GetConsoleScreenBufferInfo(hOut, &info)) return;

    COORD newSize;
    newSize.X = (info.dwSize.X < 500) ? 500 : info.dwSize.X;
    newSize.Y = (info.dwSize.Y < 3000) ? 3000 : info.dwSize.Y;

    if (newSize.X != info.dwSize.X || newSize.Y != info.dwSize.Y)
    {
        SetConsoleScreenBufferSize(hOut, newSize);
    }
}

// 현재 창이 실제로 눈에 보이는 영역(Visual Rect)을 가져오는 헬퍼 함수
inline bool GetVisualWindowRect(HWND hwnd, RECT* outRect)
{
    HMODULE hDwm = LoadLibrary(TEXT("dwmapi.dll"));
    bool success = false;

    if (hDwm)
    {
        auto pDwmGetWindowAttribute = (PDwmGetWindowAttribute)GetProcAddress(hDwm, "DwmGetWindowAttribute");
        if (pDwmGetWindowAttribute)
        {
            if (SUCCEEDED(pDwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, outRect, sizeof(RECT))))
            {
                success = true;
            }
        }
        FreeLibrary(hDwm);
    }

    if (!success)
    {
        GetWindowRect(hwnd, outRect);
    }
    return success;
}

inline void SetWindowToScreenSegment(HWND hwnd, int monitorIndex, int segmentCount, int segmentIndex)
{
    if (!hwnd) return;
    if (segmentCount <= 0) return;

    std::vector<MonitorInfo> monitors;
    EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, reinterpret_cast<LPARAM>(&monitors));

    if (monitorIndex < 0 || monitorIndex >= static_cast<int>(monitors.size())) return;

    const RECT& workArea = monitors[monitorIndex].monitorInfo.rcWork;
    int screenWidth = workArea.right - workArea.left;
    int screenHeight = workArea.bottom - workArea.top;
    int segmentWidth = screenWidth / segmentCount;

    // 1. 목표 좌표 계산
    RECT targetRect;
    targetRect.left = workArea.left + segmentWidth * segmentIndex;
    targetRect.top = workArea.top;
    targetRect.right = targetRect.left + segmentWidth;
    targetRect.bottom = targetRect.top + screenHeight;

    // 마지막 세그먼트는 남은 공간 채움
    if (segmentIndex == segmentCount - 1)
    {
        targetRect.right = workArea.right;
    }

    int targetW = targetRect.right - targetRect.left;
    int targetH = targetRect.bottom - targetRect.top;

    // 2. 초기 이동
    SetWindowPos(hwnd, NULL, targetRect.left, targetRect.top, targetW, targetH, SWP_NOZORDER | SWP_NOACTIVATE);

    // 3. 피드백 루프 보정 (오차 자동 수정)
    for (int i = 0; i < 3; ++i)
    {
        RECT visualRect;
        GetVisualWindowRect(hwnd, &visualRect);

        int diffLeft = targetRect.left - visualRect.left;
        int diffTop = targetRect.top - visualRect.top;
        int diffW = targetW - (visualRect.right - visualRect.left);
        int diffH = targetH - (visualRect.bottom - visualRect.top);

        if (std::abs(diffLeft) <= 1 && std::abs(diffTop) <= 1 && std::abs(diffW) <= 1 && std::abs(diffH) <= 1)
        {
            break;
        }

        RECT currentRect;
        GetWindowRect(hwnd, &currentRect);
        int currentW = currentRect.right - currentRect.left;
        int currentH = currentRect.bottom - currentRect.top;

        SetWindowPos(hwnd, NULL,
            currentRect.left + diffLeft,
            currentRect.top + diffTop,
            currentW + diffW,
            currentH + diffH,
            SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
    }
}

inline void MaximizeWindowOnMonitor(HWND hwnd, int monitorIndex)
{
    std::vector<MonitorInfo> monitors;
    EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, reinterpret_cast<LPARAM>(&monitors));

    if (monitorIndex >= 0 && monitorIndex < static_cast<int>(monitors.size()))
    {
        const MonitorInfo& monitor = monitors[monitorIndex];
        RECT workArea = monitor.monitorInfo.rcWork;

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
    EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, reinterpret_cast<LPARAM>(&monitors));

    if (monitorIndex < 0 || monitorIndex >= static_cast<int>(monitors.size())) return;

    WINDOWPLACEMENT wp{};
    wp.length = sizeof(WINDOWPLACEMENT);
    wp.showCmd = SW_MAXIMIZE;
    wp.rcNormalPosition = monitors[monitorIndex].monitorInfo.rcWork;

    SetWindowPlacement(hwnd, &wp);
}

inline void SetConsoleToHalfOfScreen(int monitorIndex, int halfIndex)
{
    EnsureConsoleBufferIsLargeEnough();
    SetWindowToScreenSegment(GetConsoleWindow(), monitorIndex, 2, halfIndex);
}

inline void SetConsoleToOneThirdOfScreen(int monitorIndex, int thirdIndex)
{
    EnsureConsoleBufferIsLargeEnough();
    SetWindowToScreenSegment(GetConsoleWindow(), monitorIndex, 3, thirdIndex);
}

#endif // _WINDOWS or _WIN32