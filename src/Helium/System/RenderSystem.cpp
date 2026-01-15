#include "pch.h"
#include <Helium/Systems/RenderSystem.h>
#include <Helium/HeliumCore.h>
#include <Helium/Components/Components.h>

#include <glad/glad.h>
#include <Eigen/Dense>
#include <algorithm>
#include <map>

RenderSystem::RenderSystem(HeliumCore* core)
    : HeliumSystem(core)
{
}

RenderSystem::~RenderSystem()
{
}

void RenderSystem::Initialize()
{
}

template<typename T>
void RenderRenderablesTemplate(
    const Eigen::Matrix4f& viewMatrix,
    const Eigen::Matrix4f& perspectiveMatrix,
    const Eigen::Vector3f& eye,
    const Eigen::Vector4f& lightVector,
    const std::map<Shader*, std::vector<T*>>& shadermap)
{
    for (auto& [shader, renderables] : shadermap)
    {
        if (nullptr == shader) continue;

        shader->Bind();

        shader->SetVector4f("lightPos", lightVector);

        for (auto& renderable : renderables)
        {
            if (false == renderable->IsVisible()) continue;

            // Get Entity
            auto entity = Helium.GetEntityByComponent<T>(renderable);
            if (InvalidEntity == entity) continue;

            auto transform = Helium.GetComponent<Transform>(entity);
            if (nullptr != transform)
            {
                auto& transformMatrix = transform->GetAbsoluteTransformMatrix();
                shader->SetMatrix4f("model", transformMatrix);
            }
            else
            {
                shader->SetMatrix4f("model", Eigen::Matrix4f::Identity());
            }

            shader->SetMatrix4f("view", viewMatrix);
            shader->SetMatrix4f("projection", perspectiveMatrix);

            shader->SetVector3f("cameraPos", eye);

            auto texture = Helium.GetComponent<Texture>(entity);
            if (nullptr != texture)
            {
                texture->Bind();
                shader->SetInt("texture0", 0);
            }

            renderable->Draw();

            if (nullptr != texture)
            {
                texture->Unbind();
            }
        }
    }

    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR)
    {
        std::cerr << "OpenGL Error: " << err << std::endl;
    }
}

void RenderSystem::Update(float dt)
{
    int frameNumber = 0;

    Eigen::Matrix4f viewMatrix;
    Eigen::Matrix4f perspectiveMatrix;
    Eigen::Vector3f eye;

    auto& registry = Helium.GetRegistry();

    // 1. Update Camera and Calculate Matrices
    {
        auto entites = registry.view<Camera>();
        for (auto& entity : entites)
        {
            auto& camera = entites.get<Camera>(entity);

            camera.Update(frameNumber, dt);

            viewMatrix = camera.GetViewMatrix();
            perspectiveMatrix = camera.GetProjectionMatrix();
            eye = camera.GetEye();

            // 마지막 카메라 혹은 특정 로직에 의해 Active Camera 결정
            activeCamera = &camera;
        }
    }

    // 2. Update Transform Hierarchy
    {
        auto entites = registry.view<Transform>();
        for (auto& entity : entites)
        {
            auto& transform = entites.get<Transform>(entity);
            if (nullptr == transform.GetParent())
            {
                transform.UpdateAbsoluteTransformMatrix();
            }
        }
    }

    // 3. Update Logic for Renderables
    {
        for (auto& entity : registry.view<Renderable>())
        {
            registry.get<Renderable>(entity).Update();
        }
        for (auto& entity : registry.view<DebuggingRenderable>())
        {
            registry.get<DebuggingRenderable>(entity).Update();
        }
    }
}

void RenderSystem::Render()
{
    auto& registry = Helium.GetRegistry();

    Eigen::Matrix4f viewMatrix = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f perspectiveMatrix = Eigen::Matrix4f::Identity();
    Eigen::Vector3f eye = Eigen::Vector3f::Zero();

    // 기본 조명값
    auto lightVector = Eigen::Vector4f(0.0f, 100.0f, 0.0f, 1.0f);

    if (activeCamera)
    {
        viewMatrix = activeCamera->GetViewMatrix();
        perspectiveMatrix = activeCamera->GetProjectionMatrix();
        eye = activeCamera->GetEye();

        // 조명 벡터 계산 (Update와 로직 동기화)
        if (activeCamera->GetProjectionMode() == Camera::Orthogonal)
        {
            // Orthogonal Mode -> Directional Light (w = 0.0)
            Eigen::Vector3f camDir = (activeCamera->GetTarget() - activeCamera->GetEye()).normalized();
            lightVector = Eigen::Vector4f(camDir.x(), camDir.y(), camDir.z(), 0.0f);
        }
        else
        {
            // Perspective Mode -> Point Light (w = 1.0)
            lightVector.w() = 1.0f;
        }
    }

    // 1. Opaque Renderables
    {
        std::map<Shader*, std::vector<Renderable*>> shadermap;
        for (auto& entity : registry.view<Renderable>())
        {
            auto& r = registry.get<Renderable>(entity);
            if (r.IsVisible())
            {
                if (!r.IsUsingAlpha())
                {
                    shadermap[r.GetActiveShader()].push_back(&r);
                }
            }
        }

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);

        RenderRenderablesTemplate<Renderable>(
            viewMatrix,
            perspectiveMatrix,
            eye,
            lightVector,
            shadermap);
    }

    // 2. Transparent Renderables (Z-Sorting)
    {
        struct ZRenderable { Renderable* r; float z; };
        std::vector<ZRenderable> transparentObjs;

        for (auto& entity : registry.view<Renderable>())
        {
            auto& r = registry.get<Renderable>(entity);
            if (r.IsUsingAlpha() && r.IsVisible())
            {
                Eigen::Vector3f pos = (0 == r.GetNumberOfVertices()) ? Eigen::Vector3f::Zero() : r.GetVertex(0);
                float z = (eye - pos).norm();
                transparentObjs.push_back({ &r, z });
            }
        }

        std::sort(transparentObjs.begin(), transparentObjs.end(), [](const ZRenderable& a, const ZRenderable& b)
            {
                return a.z > b.z; // Far to Near
            });

        std::map<Shader*, std::vector<Renderable*>> shadermap;
        for (auto& t : transparentObjs)
        {
            shadermap[t.r->GetActiveShader()].push_back(t.r);
        }

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE); // 투명 객체는 Depth Buffer에 쓰지 않음
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        RenderRenderablesTemplate<Renderable>(
            viewMatrix,
            perspectiveMatrix,
            eye,
            lightVector,
            shadermap);

        glDepthMask(GL_TRUE); // 상태 복구
        glDisable(GL_BLEND);
    }

    // 3. Opaque Debugging Renderables
    {
        std::map<Shader*, std::vector<DebuggingRenderable*>> shadermap;
        for (auto& entity : registry.view<DebuggingRenderable>())
        {
            auto& r = registry.get<DebuggingRenderable>(entity);
            if (!r.IsUsingAlpha() && r.IsVisible()) // 가시성 체크 추가
                shadermap[r.GetActiveShader()].push_back(&r);
        }

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);

        RenderRenderablesTemplate<DebuggingRenderable>(
            viewMatrix,
            perspectiveMatrix,
            eye,
            lightVector,
            shadermap);
    }

    // 4. Transparent Debugging Renderables (Z-Sorting)
    {
        struct ZRenderable { DebuggingRenderable* r; float z; };
        std::vector<ZRenderable> transparentObjs;

        for (auto& entity : registry.view<DebuggingRenderable>())
        {
            auto& r = registry.get<DebuggingRenderable>(entity);
            // 중요: IsUsingAlpha()가 true여야만 이 블록으로 들어와서 블렌딩이 적용됨
            if (r.IsUsingAlpha() && r.IsVisible())
            {
                Eigen::Vector3f pos = (0 == r.GetNumberOfVertices()) ? Eigen::Vector3f::Zero() : r.GetVertex(0);
                float z = (eye - pos).norm();
                transparentObjs.push_back({ &r, z });
            }
        }

        std::sort(transparentObjs.begin(), transparentObjs.end(), [](const ZRenderable& a, const ZRenderable& b)
            {
                return a.z > b.z;
            });

        std::map<Shader*, std::vector<DebuggingRenderable*>> shadermap;
        for (auto& t : transparentObjs)
        {
            shadermap[t.r->GetActiveShader()].push_back(t.r);
        }

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        RenderRenderablesTemplate<DebuggingRenderable>(
            viewMatrix,
            perspectiveMatrix,
            eye,
            lightVector,
            shadermap);

        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }
}

void RenderSystem::Shutdown()
{
}