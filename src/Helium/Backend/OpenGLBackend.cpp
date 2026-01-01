#include "pch.h"
#include <Helium/Backend/OpenGLBackend.h>
#include <glad/glad.h>

#include <sstream>

extern void He_Log(const char* fmt, ...); // Helium.cpp의 로그 함수 참조

bool OpenGLBackend::Initialize(HWND hwnd)
{
    hDC = GetDC(hwnd);

    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE;

    int pf = ChoosePixelFormat(hDC, &pfd);
    SetPixelFormat(hDC, pf, &pfd);

    hRC = wglCreateContext(hDC);
    if (!hRC) return false;

    if (!wglMakeCurrent(hDC, hRC)) return false;

    if (!gladLoadGL())
    {
        He_Log(HE_LOG_INFO, "", "gladLoadGL failed in GLBackend");
        return false;
    }

    const char* version = (const char*)glGetString(GL_VERSION);
    He_Log(HE_LOG_INFO, "", "OpenGL Initialized: %s", version);

    glClearColor(0.3f, 0.5f, 0.7f, 1.0f); // OpenGL 기본 배경색

    InitializeScreenQuad();

    return true;
}

void OpenGLBackend::Resize(int width, int height)
{
	this->width = width;
	this->height = height;

    glViewport(0, 0, width, height);

	He_Log(HE_LOG_INFO, "", "Resized OpenGL viewport to %d x %d", width, height);
}

void OpenGLBackend::Update(float dt)
{
    std::stringstream ss;
	ss << "Delta Time: " << dt << " seconds";

    He_Log(HE_LOG_INFO, "dt", "%s", ss.str().c_str());
}

void OpenGLBackend::Render()
{
    SwapBuffers(hDC);
}

void OpenGLBackend::Clear(float r, float g, float b, float a)
{
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void OpenGLBackend::DrawScreenQuad()
{
    glBindVertexArray(quadVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void OpenGLBackend::Shutdown()
{
    if (quadVAO) glDeleteVertexArrays(1, &quadVAO);
    if (quadVBO) glDeleteBuffers(1, &quadVBO);
    if (quadEBO) glDeleteBuffers(1, &quadEBO);

    wglMakeCurrent(nullptr, nullptr);

    if (hRC)
    {
        wglDeleteContext(hRC);
        hRC = nullptr;
    }
    if (hDC)
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

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glGenBuffers(1, &quadEBO);

    glBindVertexArray(quadVAO);

    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // TexCoord
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}
