#include "pch.h"

#include <glad/glad.h>
#include <nlohmann/json.hpp>

#include <Helium/HeliumCore.h>
#include <Helium/Backend/OpenGLBackend.h>
#include <Helium/Backend/VulkanBackend.h>
#include <Helium/HeliumEvents.h>

#include <Helium/Systems/EventSystem.h>
#include <Helium/Systems/InputSystem.h>
#include <Helium/Systems/RenderSystem.h>
#include <Helium/Systems/ImmediateModeRenderSystem.h>

#include <Helium/Components/Components.h>
#include <Helium/GeometryBuilder.h>
#include <Helium/VisualDebugging.h>
#include <Helium/PointCloud.h>

#include <Helium/SpatialPartitionings/SpartialPartitionings.h>

extern void He_LogInternal(HeliumLogLevel level, const char* key, char* message);

HeliumCore::HeliumCore()
{
}

HeliumCore::~HeliumCore()
{
    for (auto& kvp : pointClouds)
    {
        if(nullptr != kvp.second)
        {
            delete kvp.second;
            kvp.second = nullptr;
		}
    }
	pointClouds.clear();

    Shutdown();
}

bool HeliumCore::Initialize(HWND hwnd, int backendType)
{
    if (isInitialized) return true;

    hWnd = hwnd;

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
        backend = std::make_unique<VulkanBackend>();
        //Log(HE_LOG_INFO, "System", "Backend Selected: Vulkan");
    }
    else
    {
        backend = std::make_unique<OpenGLBackend>();
        //Log(HE_LOG_INFO, "System", "Backend Selected: OpenGL");
    }

    if (!backend->Initialize(hwnd))
    {
        Log(HE_LOG_ERROR, "System", "Failed to initialize backend.");
        return false;
    }

    VisualDebugging::Initialize();

    InitializeScene();

    for (auto& callback : onInitializeCallbacks)
    {
        callback();
    }

    isInitialized = true;

    return true;
}

void HeliumCore::Update(float dt)
{
    if (!isInitialized) return;

    for (auto& [ID, pointCloud] : pointClouds)
    {
        if (nullptr != pointCloud)
        {
            pointCloud->UpdateLoading();
        }
    }

    VisualDebugging::DispatchCommands();

    if (eventSystem) eventSystem->Update(dt);

    if (inputSystem) inputSystem->Update(dt);

    if (renderSystem) renderSystem->Update(dt);

    if (backend) backend->Update(dt);

    for (auto& callback : onUpdateCallbacks)
    {
        callback(dt);
    }
}

void HeliumCore::Render()
{
    if (!isInitialized) return;

    for (auto& callback : onRenderCallbacks)
    {
        callback();
    }

    Shader* shader = GetShader("DefaultQuad");
    if (shader)
    {
        shader->Bind();

        if (backend)
        {
            backend->DrawScreenQuad();
        }

        shader->Unbind();
    }

    if (backend) backend->Clear(0.3f, 0.5f, 0.7f, 1.0f);

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

    if (backend) backend->Render();
}

void HeliumCore::Resize(int width, int height)
{
    auto entites = registry.view<Camera>();
    for (auto& entity : entites)
    {
        auto& camera = entites.get<Camera>(entity);
		camera.GetPerspectiveSettings().SetAspectRatio(static_cast<float>(width) / static_cast<float>(height));
    }

    if (backend) backend->Resize(width, height);

    if (eventSystem) eventSystem->Trigger<WindowResizeEvent>(width, height);
}

void HeliumCore::Shutdown()
{
    if (!isInitialized) return;

    for (auto& callback : onShutdownCallbacks)
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

    if (backend)
    {
        backend->Shutdown();
        backend.reset();
    }

    registry.clear();
    isInitialized = false;
}

void HeliumCore::ProcessMessage(UINT message, WPARAM wParam, LPARAM lParam)
{
    if (inputSystem) inputSystem->ProcessMessage(message, wParam, lParam);
}

Shader* HeliumCore::CreateShader(const std::string& name, const std::string& vsCode, const std::string& fsCode)
{
    Shader* shader = new Shader(name, vsCode, fsCode);
    shaders[name] = shader;
    return shader;
}

Shader* HeliumCore::CreateShader(const std::string& name, const File& vsFile, const File& fsFile)
{
	auto vsCode = vsFile.ReadAll();
	auto fsCode = fsFile.ReadAll();
    Shader* shader = new Shader(name, vsCode, fsCode);
    shaders[name] = shader;
	return shader;
}

Shader* HeliumCore::GetShader(const std::string& name)
{
    if (shaders.find(name) != shaders.end())
        return shaders[name];
    return nullptr;
}

Entity HeliumCore::CreateEntity(const std::string& name)
{
    Entity entity = registry.create();
    nameEntityMapping[name] = entity;
    entityNameMapping[entity] = name;
    return entity;
}

Entity HeliumCore::GetEntityByName(const std::string& name)
{
    if (nameEntityMapping.find(name) != nameEntityMapping.end())
        return nameEntityMapping[name];
    return InvalidEntity;
}

void HeliumCore::RemoveEntity(const std::string& name)
{
    if (nameEntityMapping.find(name) != nameEntityMapping.end())
    {
        Entity entity = nameEntityMapping[name];
        nameEntityMapping.erase(name);
        entityNameMapping.erase(entity);
        registry.destroy(entity);
	}
}

void HeliumCore::RemoveEntity(Entity entity)
{
    if (registry.valid(entity))
    {
        if (entityNameMapping.count(entity))
        {
            std::string name = entityNameMapping[entity];
            nameEntityMapping.erase(name);
            entityNameMapping.erase(entity);
        }
        registry.destroy(entity);
    }
}

void HeliumCore::Log(HeliumLogLevel level, const char* key, const char* fmt, ...)
{
    char buffer[4096] = { 0 };
    va_list args;
    va_start(args, fmt);
    _vsnprintf_s(buffer, _countof(buffer), _TRUNCATE, fmt, args);
    va_end(args);
    He_LogInternal(level, key, buffer);
}

int HeliumCore::LoadPointCloudFromPLY(const std::string& fileName, const std::string& name)
{
	PointCloud* pointCloud = new PointCloud();
    if (pointCloud->LoadFromPLY(fileName, name))
    {
        pointClouds[pointCloud->GetID()] = pointCloud;
        selectedPointCloud = pointCloud;
        return pointCloud->GetID();
    }
    else
    {
        delete pointCloud;
        return false;
	}
}

bool HeliumCore::SelectPointCloud(int ID)
{
    if (pointClouds.find(ID) != pointClouds.end())
    {
        selectedPointCloud = pointClouds[ID];
        return true;
    }
    return false;
}

PointCloud* HeliumCore::GetPointCloud(int ID)
{
    if (pointClouds.find(ID) != pointClouds.end())
    {
        return pointClouds[ID];
    }
    return nullptr;
}

PointCloud* HeliumCore::GetSelectedPointCloud()
{
    return selectedPointCloud;
}

void HeliumCore::SetPointCloudVisibility(int ID, bool visible)
{
    auto pointCloud = GetPointCloud(ID);
    if (pointCloud)
    {
        pointCloud->SetVisible(visible);
    }
}

void HeliumCore::ClonePointCloud(int ID)
{
    PointCloud* original = GetPointCloud(ID);
    if (original)
    {
        auto clone = original->Clone();
        pointClouds[clone->GetID()] = clone;
        selectedPointCloud = clone;
    }
}

void HeliumCore::DeletePointCloud(int ID)
{
    if (pointClouds.find(ID) != pointClouds.end())
    {
        PointCloud* pointCloud = pointClouds[ID];
        if (nullptr != pointCloud)
        {
            delete pointCloud;
            pointCloud = nullptr;
        }
        pointClouds.erase(ID);
        if (selectedPointCloud && selectedPointCloud->GetID() == ID)
        {
            selectedPointCloud = nullptr;
        }
	}
}

bool HeliumCore::ExecuteCommand(const char* command)
{
    if (command == nullptr)
    {
        ErrorLog("", "ExecuteCommand: Command string is null");
        return false;
    }

    //InfoLog("", "ExecuteCommand: %s", command);

    try
    {
        auto j = nlohmann::json::parse(command);

        if (j.contains("command"))
        {
            std::string cmd = j["command"];

            if (cmd == "LoadPointCloudFromPLY")
            {
                if (j.contains("fileNames") && j["fileNames"].is_array())
                {
                    for (const auto& fileName : j["fileNames"])
                    {
                        std::string path = fileName.get<std::string>();
						std::filesystem::path fsPath(path);
						auto name = fsPath.filename();
                        LoadPointCloudFromPLY(path, name.string());
                    }
                }
            }
            else if(cmd == "SelectPointCloud")
            {
                if (j.contains("pointCloudID"))
                {
                    int pointCloudID = j["pointCloudID"];
                    SelectPointCloud(pointCloudID);
                }
            }
            else if(cmd == "SetPointCloudVisibility")
            {
                if (j.contains("pointCloudID") && j.contains("isVisible"))
                {
                    int pointCloudID = j["pointCloudID"];
                    bool isVisible = j["isVisible"];
                    SetPointCloudVisibility(pointCloudID, isVisible);
                }
            }
            else if(cmd == "ClonePointCloud")
            {
                if (j.contains("pointCloudID"))
                {
                    int pointCloudID = j["pointCloudID"];
                    ClonePointCloud(pointCloudID);
                }
            }
            else if(cmd == "DeletePointCloud")
            {
                if (j.contains("pointCloudID"))
                {
                    int pointCloudID = j["pointCloudID"];
                    DeletePointCloud(pointCloudID);
                }
			}
            else if (cmd == "BuildSparseGrid")
            {
                if (j.contains("pointCloudID"))
                {
                    int pointCloudID = j["pointCloudID"];
                    auto pointCloud = GetPointCloud(pointCloudID);
                    if (pointCloud)
                    {
                        float voxelSize = j.value("voxelSize", 0.1f);
                        SparseGrid sparseGrid;
                        sparseGrid.Build(pointCloud, voxelSize);
                        sparseGrid.Visualize();
                    }
                }
            }
        }
    }
    catch (const nlohmann::json::parse_error& e)
    {
        ErrorLog("", "ExecuteCommand: JSON Parse Error - %s", e.what());
        return false;
    }
    catch (const std::exception& e)
    {
        ErrorLog("", "ExecuteCommand: Error - %s", e.what());
        return false;
    }

    return true;
}
