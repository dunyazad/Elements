#include "pch.h"
#include <Helium/HeliumCore.h>
#include <Helium/Backend/OpenGLBackend.h>
#include <Helium/Backend/VulkanBackend.h>
#include <Helium/HeliumEvents.h>

#include <Helium/Systems/EventSystem.h>
#include <Helium/Systems/InputSystem.h>
#include <Helium/Systems/RenderSystem.h>
#include <Helium/Systems/ImmediateModeRenderSystem.h>

#include <Helium/Components/Components.h>

#include <glad/glad.h>

extern void He_LogInternal(HeliumLogLevel level, const char* key, char* message);

HeliumCore::HeliumCore()
    : width(1200), height(800)
{
}

HeliumCore::~HeliumCore()
{
    Shutdown();
}

bool HeliumCore::Initialize(HWND hwnd, int backendType)
{
    if (m_IsInitialized) return true;

    m_hWnd = hwnd;

    eventSystem = std::make_unique<EventSystem>(this);
    eventSystem->Initialize();

    inputSystem = std::make_unique<InputSystem>(this);
    inputSystem->Initialize();

    renderSystem = std::make_unique<RenderSystem>(this);
    renderSystem->Initialize();

    immediateModeRenderSystem = std::make_unique<ImmediateModeRenderSystem>(this);
    immediateModeRenderSystem->Initialize();

    BackendType type = static_cast<BackendType>(backendType);
    if (type == BackendType::Vulkan)
    {
        m_Backend = std::make_unique<VulkanBackend>();
        Log("System", "Backend Selected: Vulkan");
    }
    else
    {
        m_Backend = std::make_unique<OpenGLBackend>();
        Log("System", "Backend Selected: OpenGL");
    }

    if (!m_Backend->Initialize(hwnd))
    {
        Log("System", "Failed to initialize backend.");
        return false;
    }

    m_IsInitialized = true;

    {
        auto cameraEntity = CreateEntity("MainCamera");
        auto camera = CreateComponent<Camera>(cameraEntity);
        camera->SetProjectionMode(Camera::Perspective);

        auto cameraManipulator = CreateComponent<CameraManipulatorTrackball>(cameraEntity);
        cameraManipulator->SetCamera(camera);

        auto eventSystem = GetEventSystem();
        if (eventSystem)
        {
            CreateEventCallback<MousePositionEvent>(cameraEntity, [](Entity entity, const MousePositionEvent& e) {
                auto cmeraEntity = Helium.GetEntityByName("MainCamera");
                auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cmeraEntity);
                if (cameraManipulator) cameraManipulator->OnMousePosition(e);
				He_Log("Input", "Mouse Position Event: x=%f, y=%f", e.xpos, e.ypos);
				});

            CreateEventCallback<MouseButtonEvent>(cameraEntity, [](Entity entity, const MouseButtonEvent& e) {
                auto cmeraEntity = Helium.GetEntityByName("MainCamera");
                auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cmeraEntity);
                if (cameraManipulator) cameraManipulator->OnMouseButton(e);
                });
            CreateEventCallback<MouseWheelEvent>(cameraEntity, [](Entity entity, const MouseWheelEvent& e) {
                auto cmeraEntity = Helium.GetEntityByName("MainCamera");
                auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cmeraEntity);
                if (cameraManipulator) cameraManipulator->OnMouseWheel(e);
                });
            CreateEventCallback<KeyEvent>(cameraEntity, [](Entity entity, const KeyEvent& e) {
                auto cmeraEntity = Helium.GetEntityByName("MainCamera");
                auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cmeraEntity);
                if (cameraManipulator) cameraManipulator->OnKey(e);
				});

 /*           eventSystem->Subscribe<MousePositionEvent>([](const MousePositionEvent& e) {
				auto cmeraEntity = Helium.GetEntityByName("MainCamera");
				auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cmeraEntity);

                if (cameraManipulator) cameraManipulator->OnMousePosition(e);
                });

            eventSystem->Subscribe<MouseButtonEvent>([](const MouseButtonEvent& e) {
                auto cmeraEntity = Helium.GetEntityByName("MainCamera");
                auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cmeraEntity);

                if (cameraManipulator) cameraManipulator->OnMouseButton(e);
                });

            eventSystem->Subscribe<MouseWheelEvent>([](const MouseWheelEvent& e) {
                auto cmeraEntity = Helium.GetEntityByName("MainCamera");
                auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cmeraEntity);

                if (cameraManipulator) cameraManipulator->OnMouseWheel(e);
                });

            eventSystem->Subscribe<KeyEvent>([](const KeyEvent& e) {
                auto cmeraEntity = Helium.GetEntityByName("MainCamera");
                auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(cmeraEntity);

                if (cameraManipulator) cameraManipulator->OnKey(e);
                });*/
        }
    }

    for (auto& callback : m_OnInitializeCallbacks)
    {
        callback();
    }

    return true;
}

void HeliumCore::Update(float dt)
{
    if (!m_IsInitialized) return;

    if (eventSystem) eventSystem->Update(dt);

    if (inputSystem) inputSystem->Update(dt);

    if (renderSystem) renderSystem->Update(dt);

    if (m_Backend) m_Backend->Update(dt);

    for (auto& callback : m_OnUpdateCallbacks)
    {
        callback(dt);
    }
}

void HeliumCore::Render()
{
    if (!m_IsInitialized) return;

    for (auto& callback : m_OnRenderCallbacks)
    {
        callback();
    }

    Shader* shader = GetShader("DefaultQuad");
    if (shader)
    {
        shader->Bind();

        if (m_Backend)
        {
            m_Backend->DrawScreenQuad();
        }

        shader->Unbind();
    }

    if (m_Backend) m_Backend->Clear(0.3f, 0.5f, 0.7f, 1.0f);

    if (renderSystem)
    {
        renderSystem->Render();
    }

    if (immediateModeRenderSystem && immediateModeRenderSystem->IsEnabled())
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glUseProgram(0);

        glBindVertexArray(0);

        glDisable(GL_DEPTH_TEST);

        immediateModeRenderSystem->Update(0.0f);

        glEnable(GL_DEPTH_TEST);
    }

    if (m_Backend) m_Backend->Render();
}

void HeliumCore::Resize(int width, int height)
{
    if (m_hWnd)
    {
        RECT rect;
        if (GetClientRect(m_hWnd, &rect))
        {
            width = rect.right - rect.left;
            height = rect.bottom - rect.top;
        }
    }

    this->width = width;
    this->height = height;

    if (m_Backend) m_Backend->Resize(width, height);

    if (eventSystem) eventSystem->Trigger<WindowResizeEvent>(width, height);

    Log("System", "Resized to %d x %d", width, height);
}

void HeliumCore::Shutdown()
{
    if (!m_IsInitialized) return;

    for (auto& callback : m_OnShutdownCallbacks)
    {
        callback();
    }

    if (inputSystem)
    {
        inputSystem->Shutdown();
        inputSystem.reset();
    }

    if (eventSystem)
    {
        eventSystem->Shutdown();
        eventSystem.reset();
    }

    if (renderSystem)
    {
        renderSystem->Shutdown();
        renderSystem.reset();
    }

    if (immediateModeRenderSystem)
    {
        immediateModeRenderSystem->Shutdown();
        immediateModeRenderSystem.reset();
    }

    if (m_Backend)
    {
        m_Backend->Shutdown();
        m_Backend.reset();
    }

    registry.clear();
    m_IsInitialized = false;
}

Shader* HeliumCore::CreateShader(const std::string& name, const std::string& vsCode, const std::string& fsCode)
{
    Shader* shader = new Shader(name, vsCode, fsCode);
    m_Shaders[name] = shader;
    return shader;
}

Shader* HeliumCore::GetShader(const std::string& name)
{
    if (m_Shaders.find(name) != m_Shaders.end())
        return m_Shaders[name];
    return nullptr;
}

Entity HeliumCore::CreateEntity(const std::string& name)
{
    Entity entity = registry.create();
    m_NameEntityMapping[name] = entity;
    m_EntityNameMapping[entity] = name;
    Log("Entity", "Created Entity: %s", name.c_str());
    return entity;
}

Entity HeliumCore::GetEntityByName(const std::string& name)
{
    if (m_NameEntityMapping.find(name) != m_NameEntityMapping.end())
        return m_NameEntityMapping[name];
    return InvalidEntity;
}

void HeliumCore::RemoveEntity(Entity entity)
{
    if (registry.valid(entity))
    {
        if (m_EntityNameMapping.count(entity))
        {
            std::string name = m_EntityNameMapping[entity];
            m_NameEntityMapping.erase(name);
            m_EntityNameMapping.erase(entity);
        }
        registry.destroy(entity);
    }
}

void HeliumCore::Log(const char* key, const char* fmt, ...)
{
    char buffer[4096] = { 0 };
    va_list args;
    va_start(args, fmt);
    _vsnprintf_s(buffer, _countof(buffer), _TRUNCATE, fmt, args);
    va_end(args);
    He_LogInternal(HE_LOG_INFO, key, buffer);
}

void HeliumCore::OnResize(int newWidth, int newHeight)
{
    width = newWidth;
    height = newHeight;
}
