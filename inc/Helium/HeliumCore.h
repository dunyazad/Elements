#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <entt/entt.hpp>

#include <Helium/Color.hpp>
#include <Helium/File.h>
#include <Helium/Backend/GraphicsBackend.h>
#include <Helium/Components/Components.h>
#include <Helium/Systems/Systems.h>
#include <Helium/PointCloudProcessing/PointCloudProcessor.h>

class Scene;
class PointCloud;
class SparseGrid;
class SparseDataBlock;

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

    void ProcessMessage(UINT message, WPARAM wParam, LPARAM lParam);

    inline int GetWidth() const { if (backend) return backend->GetWidth(); else return 0; }
    inline int GetHeight() const { if (backend) return backend->GetHeight(); else return 0; }

    inline Registry& GetRegistry() { return registry; }
    inline Dispatcher& GetDispatcher() { return dispatcher; }

    inline IGraphicsBackend* GetGraphicsBackend() { return backend.get(); }

    EventSystem* GetEventSystem() { return eventSystem.get(); }
	GUISystem* GetGUISystem() { return guiSystem.get(); }
    InputSystem* GetInputSystem() { return inputSystem.get(); }
    ImmediateModeRenderSystem* GetImmediateModeRenderSystem() { return immediateModeRenderSystem.get(); }
	RenderSystem* GetRenderSystem() { return renderSystem.get(); }

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
    T* CreateComponent(Entity entity, Args&&... args)
    {
        if (registry.all_of<T>(entity))
        {
            return &registry.get<T>(entity);
        }
        return &(registry.emplace<T>(entity, std::forward<Args>(args)...));
    }

	template<typename T>
    void RemoveComponent(Entity entity)
    {
        if (registry.all_of<T>(entity))
        {
            registry.remove<T>(entity);
        }
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
    EventCallback<T>& CreateEventCallback(Entity entity, Args&&... args)
    {
        if (registry.all_of<EventCallback<T>>(entity))
        {
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

    template<typename T>
    void EnqueueEvent(const T& event)
    {
        dispatcher.enqueue<T>(event);
    }

    Shader* CreateShader(const std::string& name, const std::string& vsCode, const std::string& fsCode);
    Shader* CreateShader(const std::string& name, const File& vsFile, const File& fsFile);
    Shader* GetShader(const std::string& name);

	inline HWND GetHWND() const { return hWnd; }

    void Log(HeliumLogLevel level, const char* key, const char* fmt, ...);

	int LoadPointCloudFromPLY(const std::string& filename, const std::string& name);
    void BuildSpatialPartitionings(int pointCloudID);
    bool SelectPointCloud(int pointCloudID);
	PointCloud* GetPointCloud(int pointCloudID);
    PointCloud* GetSelectedPointCloud();
    void SetPointCloudVisibility(int pointCloudID, bool visible);
    void ClonePointCloud(int pointCloudID);
    void DeletePointCloud(int pointCloudID);
    void RenamePointCloud(int pointCloudID, const std::string& newName);

	void PerformClustering(int pointCloudID, float searchRadius, float angleThreshold);
    std::vector<uint8_t> PerformSOR(int pointCloudID, int kNeighbors, float stdDevMulThresh, bool deletePoints, PointCloudVisualizationMode visualizationMode);
    std::vector<uint8_t> PerformROR(int pointCloudID, float radius, int minNeighborsInRadius, bool deletePoints, PointCloudVisualizationMode visualizationMode);
    std::vector<uint8_t> PerformCurvatureAnalysis(int pointCloudID, int kNeighbors, float curvatureThreshold, PointCloudVisualizationMode visualizationMode);
    std::vector<uint8_t> PerformNormalDeviationAnalysis(int pointCloudID, float radius, float deviationThreshold, PointCloudVisualizationMode visualizationMode);
    std::vector<uint8_t> PerformPFOR(int pointCloudID, int kNeighbors, float distanceThreshold, PointCloudVisualizationMode visualizationMode);

    void ProcessManagedToNativeEvents();
    void EnqueueManagedToNativeEvent(std::function<void()> event);
    void OnManagedToNative(const char* jsonString);
    void NativeToManaged(const char* jsonString);
    void NotifyMessage(const std::string& message, int durationMS = 3000);

    inline SparseGrid* GetSparseGrid(int pointCloudID)
    {
        if (sparseGrids.find(pointCloudID) != sparseGrids.end())
        {
            return sparseGrids[pointCloudID];
        }
        return nullptr;
	}

    inline SparseDataBlock* GetSparseDataBlock(int pointCloudID)
    {
        if (sparseDataBlocks.find(pointCloudID) != sparseDataBlocks.end())
        {
            return sparseDataBlocks[pointCloudID];
        }
        return nullptr;
	}

private:
    HeliumCore();
    ~HeliumCore();

    HeliumCore(const HeliumCore&) = delete;
    HeliumCore& operator=(const HeliumCore&) = delete;

private:
    bool isInitialized = false;
    HWND hWnd = nullptr;

    std::unique_ptr<IGraphicsBackend> backend = nullptr;

	Eigen::Vector4f clearColor = Eigen::Vector4f(0.298f, 0.337f, 0.416f, 1.0f);

    Registry registry;
    Dispatcher dispatcher;

    std::unique_ptr<EventSystem> eventSystem;
    std::unique_ptr<GUISystem> guiSystem;
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

    std::unordered_map<std::string, Scene*> scenes;

	std::unordered_map<int, PointCloud*> pointClouds;
	PointCloud* selectedPointCloud = nullptr;

    const float voxelSize = 0.3f;
    const float cellSize = 0.3f;
	std::unordered_map<int, SparseGrid*> sparseGrids;
	std::unordered_map<int, SparseDataBlock*> sparseDataBlocks;

    std::mutex managedToNativeEventQueueMutex;
    std::vector<std::function<void()>> managedToNativeEventQueue;
};

#define Helium HeliumCore::GetStaticInstance()
