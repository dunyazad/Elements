#include "pch.h"
#include "framework.h"
#include <glad/glad.h>
#include <gl/GL.h>
#include <Monitor.h>

static HDC   g_hdc = nullptr;
static HGLRC g_hglrc = nullptr;
static bool  g_ready = false;
static int g_lastW = -1;
static int g_lastH = -1;
static HeliumLogCallback g_LogCallback = nullptr;

bool Helium_Initialize(HWND hwnd)
{
	Helium_Log("Helium_Initialize called\n");

    if (!IsWindow(hwnd))
        return false;

    g_hdc = GetDC(hwnd);

    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE;

    int pf = ChoosePixelFormat(g_hdc, &pfd);
    SetPixelFormat(g_hdc, pf, &pfd);

    g_hglrc = wglCreateContext(g_hdc);
    if (!g_hglrc)
        return false;

    if (!wglMakeCurrent(g_hdc, g_hglrc))
        return false;

    if (!gladLoadGL())
    {
        printf("gladLoadGL failed\n");
        return false;
    }

    {
        const char* vendor = (const char*)glGetString(GL_VENDOR);
        const char* renderer = (const char*)glGetString(GL_RENDERER);
        const char* version = (const char*)glGetString(GL_VERSION);
        const char* extensions = (const char*)glGetString(GL_EXTENSIONS);

        Helium_Log(HELIUM_LOG_INFO, "Helium_Native", vendor);
        Helium_Log(HELIUM_LOG_WARN, "Helium_Native", renderer);
        Helium_Log(HELIUM_LOG_DEBUG, "Helium_Native", version);
        Helium_Log(HELIUM_LOG_ERROR, "Helium_Native", extensions);
    }

    glClearColor(0.3f, 0.5f, 0.7f, 1.0f);

    g_ready = true;
    return true;
}

void Helium_Resize(int width, int height)
{
    if (!g_ready)
        return;

    if (width <= 0 || height <= 0)
        return;

    if (width == g_lastW && height == g_lastH)
        return;

    g_lastW = width;
    g_lastH = height;

    glViewport(0, 0, width, height);
}

void Helium_Render()
{
    if (!g_ready)
        return;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    SwapBuffers(g_hdc);
}

void Helium_Shutdown()
{
    if (g_hglrc)
    {
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(g_hglrc);
        g_hglrc = nullptr;
    }

    if (g_hdc)
    {
        ReleaseDC(WindowFromDC(g_hdc), g_hdc);
        g_hdc = nullptr;
    }

    g_ready = false;
}

void Helium_CreateConsole()
{
	AllocConsole();
    
    FILE* fp;
    freopen_s(&fp, "CONOUT$", "w", stdout);
    setvbuf(stdout, nullptr, _IONBF, 0);
}

void Helium_SetLogCallback(HeliumLogCallback cb)
{
    g_LogCallback = cb;
}

void Helium_Log(const char* fmt, ...)
{
	Helium_Log(HELIUM_LOG_INFO, "", fmt);
}

void Helium_Log(const char* key, const char* fmt, ...)
{
    Helium_Log(HELIUM_LOG_INFO, key, fmt);
}

void Helium_Log(HeliumLogLevel level, const char* fmt, ...)
{
    Helium_Log(level, "", fmt);
}

void Helium_Log(HeliumLogLevel level, const char* key, const char* fmt, ...)
{
    if (!g_LogCallback)
        return;

    char buffer[4096];

    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    g_LogCallback(level, key, buffer);
}
