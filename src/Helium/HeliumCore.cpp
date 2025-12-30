#include "pch.h"
#include <Helium/HeliumCore.h>
#include <Helium/Backend/OpenGLBackend.h>
#include <Helium/Backend/VulkanBackend.h>
#include <Helium/HeliumEvents.h>

extern void He_LogInternal(HeliumLogLevel level, const char* key, char* message);

HeliumCore::HeliumCore() {}
HeliumCore::~HeliumCore() { Shutdown(); }

bool HeliumCore::Initialize(HWND hwnd, int backendType)
{
    if (m_IsInitialized) return true;

    m_hWnd = hwnd;

    m_EventSystem = std::make_unique<EventSystem>(this);
    m_EventSystem->Initialize();

    m_InputSystem = std::make_unique<InputSystem>(this);
    m_InputSystem->Initialize();

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

    for (auto& callback : m_OnInitializeCallbacks)
    {
        callback();
    }

    return true;
}

void HeliumCore::Update(float dt)
{
    if (!m_IsInitialized) return;

    if (m_EventSystem) m_EventSystem->Update(dt);

    if (m_InputSystem) m_InputSystem->Update(dt);

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

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

    if (m_Backend) m_Backend->Render();
}

void HeliumCore::Resize(int width, int height)
{
    if (m_Backend) m_Backend->Resize(width, height);

    // 2. 이벤트 발생! (이제 다른 시스템들이 이 이벤트를 구독해서 반응함)
    if (m_EventSystem)
    {
        m_EventSystem->Trigger<WindowResizeEvent>(width, height);
    }

    Log("System", "Resized to %d x %d", width, height);
}

void HeliumCore::Shutdown()
{
    if (!m_IsInitialized) return;

    for (auto& callback : m_OnShutdownCallbacks)
    {
        callback();
    }

    if (m_InputSystem)
    {
        m_InputSystem->Shutdown();
        m_InputSystem.reset();
    }

    if (m_EventSystem)
    {
        m_EventSystem->Shutdown();
        m_EventSystem.reset();
    }

    if (m_Backend)
    {
        m_Backend->Shutdown();
        m_Backend.reset();
    }

    m_Registry.clear();
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
    Entity entity = m_Registry.create();
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
    if (m_Registry.valid(entity))
    {
        if (m_EntityNameMapping.count(entity))
        {
            std::string name = m_EntityNameMapping[entity];
            m_NameEntityMapping.erase(name);
            m_EntityNameMapping.erase(entity);
        }
        m_Registry.destroy(entity);
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
