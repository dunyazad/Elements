#include "pch.h"
#include <Helium/Backend/OpenGLBackend.h>
#include <glad/glad.h>

extern void He_Log(const char* fmt, ...); // Helium.cpp의 로그 함수 참조

bool OpenGLBackend::Initialize(HWND hwnd)
{
    m_hDC = GetDC(hwnd);

    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE;

    int pf = ChoosePixelFormat(m_hDC, &pfd);
    SetPixelFormat(m_hDC, pf, &pfd);

    m_hRC = wglCreateContext(m_hDC);
    if (!m_hRC) return false;

    if (!wglMakeCurrent(m_hDC, m_hRC)) return false;

    if (!gladLoadGL())
    {
        He_Log("gladLoadGL failed in GLBackend");
        return false;
    }

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // OpenGL 기본 배경색

    // 버전 정보 로그
    const char* version = (const char*)glGetString(GL_VERSION);
    He_Log("OpenGL Initialized: %s", version);

    return true;
}

void OpenGLBackend::Resize(int width, int height)
{
    glViewport(0, 0, width, height);
}

void OpenGLBackend::Update(float dt)
{
    std::stringstream ss;
	ss << "Delta Time: " << dt << " seconds";

    He_Log("dt", "%s\n", ss.str().c_str());
}

void OpenGLBackend::Render()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    SwapBuffers(m_hDC);
}

void OpenGLBackend::Shutdown()
{
    wglMakeCurrent(nullptr, nullptr);
    if (m_hRC)
    {
        wglDeleteContext(m_hRC);
        m_hRC = nullptr;
    }
    if (m_hDC)
    {
        // WindowFromDC 등은 상황에 따라 생략 가능하나 원칙적으로 Release
        // HWND를 멤버로 저장하지 않았으므로 여기선 생략하거나 상위에서 처리
    }
}