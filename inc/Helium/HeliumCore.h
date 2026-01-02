#pragma once

#include <vector>
#include <functional>
#include <string>
#include <unordered_map>
#include <memory>

#include <entt/entt.hpp>

#include <Helium/File.h>

#include <Helium/Backend/GraphicsBackend.h>

#include <Helium/Components/Components.h>

#include <Helium/Systems/Systems.h>

using Entity = entt::entity;
#define InvalidEntity ((Entity)UINT32_MAX)

using Registry = entt::registry;
using Dispatcher = entt::dispatcher;

template <typename T>
class EventCallback {
public:
    static void OnEvent(const T& event) {
        for (auto* callback : instances) {
            callback->callback(callback->target, event);
        }
    }

    EventCallback(Entity target, std::function<void(Entity, const T&)> callback)
        : target(target), callback(std::move(callback)) {
        HeliumCore::GetStaticInstance().GetDispatcher().sink<T>().connect<&OnEvent>();
        instances.insert(this);
    }

    ~EventCallback() {
        instances.erase(this);
    }

protected:
    Entity target;
    std::function<void(Entity, const T&)> callback;

    static inline std::set<EventCallback<T>*> instances;
};


class HeliumCore
{
public:
    static HeliumCore& GetStaticInstance()
    {
        static HeliumCore instance;
        return instance;
    }

    bool Initialize(HWND hwnd, int backendType);
    void InitializeScene();
    void Update(float dt);
    void Render();
    void Resize(int width, int height);
    void Shutdown();

    inline int GetWidth() const { if (backend) return backend->GetWidth(); else return 0; }
    inline int GetHeight() const { if (backend) return backend->GetHeight(); else return 0; }

    inline Registry& GetRegistry() { return registry; }
    inline Dispatcher& GetDispatcher() { return dispatcher; }

    inline IGraphicsBackend* GetGraphicsBackend() { return backend.get(); }

    EventSystem* GetEventSystem() { return eventSystem.get(); }
    InputSystem* GetInputSystem() { return inputSystem.get(); }
    ImmediateModeRenderSystem* GetImmediateModeRenderSystem() { return immediateModeRenderSystem.get(); }

    Entity CreateEntity(const std::string& name);
    Entity GetEntityByName(const std::string& name);
    template<typename T>
    Entity GetEntityByComponent(T* t)
    {
        auto view = registry.view<T>();

        for (auto entity : view)
        {
            const auto& comp = view.get<T>(entity);
            if (&comp == t)
            {
                return entity;
            }
        }

        return InvalidEntity;
    }

    const std::string& GetEntityName(Entity entity);
    void RemoveEntity(const std::string& name);
    void RemoveEntity(Entity entity);

    template<typename T>
    T* GetComponent(Entity entity)
    {
        if (false == registry.all_of<T>(entity))
            return nullptr;
        else
            return &registry.get<T>(entity);
    }

    template<typename T, typename... Args>
    T* CreateComponent(Entity entity, Args&&... args) {
        if (registry.all_of<T>(entity)) {
            return &registry.get<T>(entity);
        }
        return &(registry.emplace<T>(entity, std::forward<Args>(args)...));
    }

    inline void AddOnInitializeCallback(std::function<void()> callback) { onInitializeCallbacks.push_back(callback); }
    inline void AddOnUpdateCallback(std::function<void(float)> callback) { onUpdateCallbacks.push_back(callback); }
    inline void AddOnRenderCallback(std::function<void()> callback) { onRenderCallbacks.push_back(callback); }
    inline void AddOnShutdownCallback(std::function<void()> callback) { onShutdownCallbacks.push_back(callback); }


    template<typename T>
    EventCallback<T>& GetEventCallback(Entity entity)
    {
        assert(registry.all_of<EventCallback<T>>(entity) && "Entity does not have the requested event callback.");
        return registry.get<EventCallback<T>>(entity);
    }

    template<typename T, typename... Args>
    EventCallback<T>& CreateEventCallback(Entity entity, Args&&... args) {
        if (registry.all_of<EventCallback<T>>(entity)) {
            return registry.get<EventCallback<T>>(entity);
        }
        return registry.emplace<EventCallback<T>>(entity, entity, std::forward<Args>(args)...);
    }

    template<typename T>
    void RemoveEventCallback(Entity entity)
    {
        if (registry.all_of<EventCallback<T>>(entity))
        {
            registry.remove<EventCallback<T>>(entity);
        }
    }


    Shader* CreateShader(const std::string& name, const std::string& vsCode, const std::string& fsCode);
    Shader* CreateShader(const std::string& name, const File& vsFile, const File& fsFile);
    Shader* GetShader(const std::string& name);

	inline HWND GetHWND() const { return hWnd; }

    void Log(const char* key, const char* fmt, ...);

private:
    HeliumCore();
    ~HeliumCore();

    HeliumCore(const HeliumCore&) = delete;
    HeliumCore& operator=(const HeliumCore&) = delete;

private:
    bool isInitialized = false;
    HWND hWnd = nullptr;

    std::unique_ptr<IGraphicsBackend> backend = nullptr;

    Registry registry;
    Dispatcher dispatcher;

    std::unique_ptr<EventSystem> eventSystem;
    std::unique_ptr<InputSystem> inputSystem;
    std::unique_ptr<RenderSystem> renderSystem;
    std::unique_ptr<ImmediateModeRenderSystem> immediateModeRenderSystem;

    std::unordered_map<std::string, Entity> nameEntityMapping;
    std::unordered_map<Entity, std::string> entityNameMapping;

    std::vector<std::function<void()>> onInitializeCallbacks;
    std::vector<std::function<void(float)>> onUpdateCallbacks;
    std::vector<std::function<void()>> onRenderCallbacks;
    std::vector<std::function<void()>> onShutdownCallbacks;

    std::unordered_map<std::string, Shader*> shaders;
};

#define Helium HeliumCore::GetStaticInstance()
