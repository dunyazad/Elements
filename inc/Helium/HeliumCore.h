#pragma once

#include <vector>
#include <functional>
#include <string>
#include <unordered_map>
#include <memory>

#include <entt/entt.hpp>
#include <Helium/Backend/GraphicsBackend.h>

#include <Helium/Components/Components.h>

#include <Helium/Systems/Systems.h>

using Entity = entt::entity;
//static const Entity InvalidEntity = (Entity)entt::null;

using Registry = entt::registry;
using Dispatcher = entt::dispatcher;

class HeliumCore
{
public:
    static HeliumCore& GetStaticInstance()
    {
        static HeliumCore instance;
        return instance;
    }

    bool Initialize(HWND hwnd, int backendType);
    void Update(float dt);
    void Render();
    void Resize(int width, int height);
    void Shutdown();

    Shader* CreateShader(const std::string& name, const std::string& vsCode, const std::string& fsCode);
    Shader* GetShader(const std::string& name);

    inline Registry& GetRegistry() { return m_Registry; }
    inline Dispatcher& GetDispatcher() { return m_Dispatcher; }

    inline IGraphicsBackend* GetGraphicsBackend() { return m_Backend.get(); }

    EventSystem* GetEventSystem() { return m_EventSystem.get(); }
    InputSystem* GetInputSystem() { return m_InputSystem.get(); }

    Entity CreateEntity(const std::string& name);
    Entity GetEntityByName(const std::string& name);
    void RemoveEntity(Entity entity);

    inline void AddOnInitializeCallback(std::function<void()> callback) { m_OnInitializeCallbacks.push_back(callback); }
    inline void AddOnUpdateCallback(std::function<void(float)> callback) { m_OnUpdateCallbacks.push_back(callback); }
    inline void AddOnRenderCallback(std::function<void()> callback) { m_OnRenderCallbacks.push_back(callback); }
    inline void AddOnShutdownCallback(std::function<void()> callback) { m_OnShutdownCallbacks.push_back(callback); }

    void Log(const char* key, const char* fmt, ...);

    void OnResize(int width, int height);

private:
    HeliumCore();
    ~HeliumCore();

    HeliumCore(const HeliumCore&) = delete;
    HeliumCore& operator=(const HeliumCore&) = delete;

private:
    bool m_IsInitialized = false;
    HWND m_hWnd = nullptr;

    std::unique_ptr<IGraphicsBackend> m_Backend = nullptr;

    Registry m_Registry;
    Dispatcher m_Dispatcher;

    std::unique_ptr<EventSystem> m_EventSystem;
    std::unique_ptr<InputSystem> m_InputSystem;

    std::unordered_map<std::string, Entity> m_NameEntityMapping;
    std::unordered_map<Entity, std::string> m_EntityNameMapping;

    std::vector<std::function<void()>> m_OnInitializeCallbacks;
    std::vector<std::function<void(float)>> m_OnUpdateCallbacks;
    std::vector<std::function<void()>> m_OnRenderCallbacks;
    std::vector<std::function<void()>> m_OnShutdownCallbacks;

    std::unordered_map<std::string, Shader*> m_Shaders;
};

#define Helium HeliumCore::GetStaticInstance()
