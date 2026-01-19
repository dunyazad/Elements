#include "pch.h"
#include <Helium/Systems/EventSystem.h>
#include <Helium/HeliumCore.h>
#include <algorithm>

EventSystem::EventSystem(HeliumCore* core)
    : HeliumSystem(core)
{
}

void EventSystem::Initialize()
{
    AddLayer("GUI", 100);
    AddLayer("3D", 0);
}

void EventSystem::Update(float timeDelta)
{
    for (auto& layer : layers)
    {
        layer->dispatcher.update();
    }
}

void EventSystem::AddLayer(const std::string& name, int priority)
{
    auto layer = std::make_shared<EventLayer>();
    layer->name = name;
    layer->priority = priority;

    layers.push_back(layer);

    // Sort layers by priority (Higher priority comes first)
    std::sort(layers.begin(), layers.end(), [](const std::shared_ptr<EventLayer>& a, const std::shared_ptr<EventLayer>& b) {
        return a->priority > b->priority;
        });
}

EventSystem::EventLayer* EventSystem::GetLayer(const std::string& name)
{
    for (auto& layer : layers)
    {
        if (layer->name == name)
        {
            return layer.get();
        }
    }
    return nullptr;
}
