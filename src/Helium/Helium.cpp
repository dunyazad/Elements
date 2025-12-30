#include "pch.h"

#include <glad/glad.h>
#include <gl/GL.h>
#include <Monitor.h>

#include <Helium/Backend/GraphicsBackend.h>
#include <Helium/Backend/OpenGLBackend.h>
#include <Helium/Backend/VulkanBackend.h>

static HDC   g_hdc = nullptr;
static HGLRC g_hglrc = nullptr;
static bool  g_ready = false;
static int g_lastW = -1;
static int g_lastH = -1;
static std::unique_ptr<IGraphicsBackend> g_backend = nullptr;

bool He_Initialize(HWND hwnd, int backendType)
{
	He_Log("He_Initialize called\n");

    if (!IsWindow(hwnd))
        return false;

    BackendType type = static_cast<BackendType>(backendType);

    if (type == BackendType::Vulkan)
    {
        g_backend = std::make_unique<VulanBackend>();
        He_Log("Selected Backend: Vulkan");
    }
    else
    {
        g_backend = std::make_unique<OpenGLBackend>();
        He_Log("Selected Backend: OpenGL");
    }

    if (!g_backend->Initialize(hwnd))
    {
        He_Log(HE_LOG_ERROR, "System", "Failed to initialize backend.");
        g_backend.reset();
        return false;
    }

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

        He_Log(HE_LOG_INFO, "He_Native", vendor);
        He_Log(HE_LOG_WARN, "He_Native", renderer);
        He_Log(HE_LOG_DEBUG, "He_Native", version);
        He_Log(HE_LOG_ERROR, "He_Native", extensions);
    }

    glClearColor(0.3f, 0.5f, 0.7f, 1.0f);

    g_ready = true;
    return true;
}

void He_Resize(int width, int height)
{
    if (!g_ready)
        return;

    if (width <= 0 || height <= 0)
        return;

    if (width == g_lastW && height == g_lastH)
        return;

    g_lastW = width;
    g_lastH = height;

    if (g_ready && g_backend)
    {
        g_backend->Resize(width, height);
    }
}

HELIUM_API void He_Update(float dt)
{
    if (g_ready && g_backend)
    {
        g_backend->Update(dt);
    }
}

void He_Render()
{
    if (!g_ready)
        return;

    if (g_ready && g_backend)
    {
        g_backend->Render();
    }
}

void He_Shutdown()
{
    if (g_backend)
    {
        g_backend->Shutdown();
        g_backend.reset();
    }
    g_ready = false;

    g_ready = false;
}

void He_CreateConsole()
{
	AllocConsole();
    
    FILE* fp;
    freopen_s(&fp, "CONOUT$", "w", stdout);
    setvbuf(stdout, nullptr, _IONBF, 0);
}
