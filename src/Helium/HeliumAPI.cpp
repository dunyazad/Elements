#include "pch.h"
#include "framework.h"
#include <gl/GL.h>

static HDC   g_hdc = nullptr;
static HGLRC g_hglrc = nullptr;
static bool  g_ready = false;

bool Helium_Initialize(HWND hwnd)
{
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

    glClearColor(0.1f, 0.2f, 0.4f, 1.0f);

    g_ready = true;
    return true;
}

void Helium_Resize(int width, int height)
{
    if (!g_ready)
        return;

    if (width <= 0 || height <= 0)
        return;

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
