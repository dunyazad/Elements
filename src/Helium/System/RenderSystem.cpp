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

            if (nullptr == activeCamera)
            {
                activeCamera = &camera;
			}
        }
    }

    // 2. Calculate Light Vector
    // Reset to default
    auto lightVector = Eigen::Vector4f(0.0f, 100.0f, 0.0f, 1.0f);

    if (activeCamera)
    {
        if (activeCamera->GetProjectionMode() == Camera::Orthogonal)
        {
            // Orthogonal Mode -> Directional Light (w = 0.0)
            // Camera direction
            Eigen::Vector3f camDir = (activeCamera->GetTarget() - activeCamera->GetEye()).normalized();

            // Set w to 0.0 for directional light
            lightVector = Eigen::Vector4f(camDir.x(), camDir.y(), camDir.z(), 0.0f);
        }
        else
        {
            // Perspective Mode -> Point Light (w = 1.0)
            lightVector.w() = 1.0f;
        }
    }

    // 3. Update Transform Hierarchy
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

    // 4. Update Logic for Renderables
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

    auto lightVector = Eigen::Vector4f(0.0f, 100.0f, 0.0f, 1.0f);

    if(activeCamera)
    {
        viewMatrix = activeCamera->GetViewMatrix();
        perspectiveMatrix = activeCamera->GetProjectionMatrix();
        eye = activeCamera->GetEye();
	}

    // 1. Opaque Renderables
    {
        std::map<Shader*, std::vector<Renderable*>> shadermap;
        for (auto& entity : registry.view<Renderable>())
        {
            auto& r = registry.get<Renderable>(entity);
            if (!r.IsUsingAlpha())
                shadermap[r.GetActiveShader()].push_back(&r);
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
            if (r.IsUsingAlpha())
            {
                Eigen::Vector3f pos = (0 == r.GetNumberOfVertices()) ? Eigen::Vector3f::Zero() : r.GetVertex(0);
                // Eigen uses .norm() for length
                float z = (eye - pos).norm();
                transparentObjs.push_back({ &r, z });
            }
        }

        std::sort(transparentObjs.begin(), transparentObjs.end(), [](const ZRenderable& a, const ZRenderable& b)
            {
                return a.z > b.z;
            });

        std::map<Shader*, std::vector<Renderable*>> shadermap;
        for (auto& t : transparentObjs)
        {
            shadermap[t.r->GetActiveShader()].push_back(t.r);
        }

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        RenderRenderablesTemplate<Renderable>(
            viewMatrix,
            perspectiveMatrix,
            eye,
            lightVector,
            shadermap);

        glDepthMask(GL_TRUE);
    }

    // 3. Opaque Debugging Renderables
    {
        std::map<Shader*, std::vector<DebuggingRenderable*>> shadermap;
        for (auto& entity : registry.view<DebuggingRenderable>())
        {
            auto& r = registry.get<DebuggingRenderable>(entity);
            if (!r.IsUsingAlpha())
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
            if (r.IsUsingAlpha())
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
    }
}

void RenderSystem::Shutdown()
{
}