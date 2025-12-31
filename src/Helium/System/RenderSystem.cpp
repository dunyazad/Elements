#include "pch.h"

#include <Helium/Systems/RenderSystem.h>

#include <Helium/HeliumCore.h>

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
void RenderSystem::Update(float dt)
{
    int frameNumber = 0;

    auto& registry = Helium.GetRegistry();

    {
        auto entites = registry.view<Camera>();
        for (auto& entity : entites)
        {
            auto& camera = entites.get<Camera>(entity);

            camera.Update(frameNumber, dt);

            viewMatrix = camera.GetViewMatrix();
            perspectiveMatrix = camera.GetProjectionMatrix();
            //eye = camera.GetEye();

            //activeCamera = &camera; // 마지막 업데이트된 카메라를 메인으로 간주
        }
    }
}
void RenderSystem::Render()
{
}
void RenderSystem::Shutdown()
{
}