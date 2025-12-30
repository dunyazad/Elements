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

    const char* version = (const char*)glGetString(GL_VERSION);
    He_Log("OpenGL Initialized: %s", version);

    glClearColor(0.3f, 0.5f, 0.7f, 1.0f); // OpenGL 기본 배경색

    InitializeScreenQuad();

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

    He_Log("dt", "%s", ss.str().c_str());
}

void OpenGLBackend::Render()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    SwapBuffers(m_hDC);
}

void OpenGLBackend::Clear(float r, float g, float b, float a)
{
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void OpenGLBackend::DrawScreenQuad()
{
    glBindVertexArray(m_QuadVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void OpenGLBackend::Shutdown()
{
    if (m_QuadVAO) glDeleteVertexArrays(1, &m_QuadVAO);
    if (m_QuadVBO) glDeleteBuffers(1, &m_QuadVBO);
    if (m_QuadEBO) glDeleteBuffers(1, &m_QuadEBO);

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

void OpenGLBackend::InitializeScreenQuad()
{
    float vertices[] = {
         1.0f,  1.0f, 0.0f,  1.0f, 1.0f,
         1.0f, -1.0f, 0.0f,  1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,  0.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,  0.0f, 1.0f
    };
    unsigned int indices[] = { 0, 1, 3, 1, 2, 3 };

    glGenVertexArrays(1, &m_QuadVAO);
    glGenBuffers(1, &m_QuadVBO);
    glGenBuffers(1, &m_QuadEBO);

    glBindVertexArray(m_QuadVAO);

    glBindBuffer(GL_ARRAY_BUFFER, m_QuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_QuadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // TexCoord
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}
