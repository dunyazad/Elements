#pragma once

#pragma warning(disable: 4251)

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>

#include <Helium/Systems/HeliumSystem.h>

#include <entt/entt.hpp>

class HeliumCore;

class HELIUM_API EventSystem : public HeliumSystem
{
public:
    template<typename EventType>
    struct EventHandlerWrapper
    {
        std::function<void(const EventType&)> callback;

        void Invoke(const EventType& event)
        {
            if (callback) callback(event);
        }
    };

    struct EventLayer
    {
        std::string name;
        int priority;
        entt::dispatcher dispatcher;

        std::vector<std::shared_ptr<void>> handlers;
    };

    EventSystem(HeliumCore* core);

    virtual void Initialize();
    virtual void Update(float timeDelta);

    EventSystem::EventLayer* AddLayer(const std::string& name, int priority);

    template<typename EventType, typename Handler>
    void Subscribe(const std::string& layerName, Handler&& handler)
    {
        EventLayer* layer = GetLayer(layerName);
        if (nullptr == layer)
        {
            layer = AddLayer(layerName, 0);
        }

        auto wrapper = std::make_shared<EventHandlerWrapper<EventType>>();
        wrapper->callback = std::forward<Handler>(handler);
        layer->handlers.push_back(wrapper);
        layer->dispatcher.sink<EventType>().template connect<&EventHandlerWrapper<EventType>::Invoke>(wrapper.get());
    }

    template<typename EventType>
    bool HasSubscribers(const std::string& layerName)
    {
        EventLayer* layer = GetLayer(layerName);
        if (nullptr == layer) return false;
        auto& handlers = layer->handlers;
        for (auto& h : handlers)
        {
            auto wrapper = std::static_pointer_cast<EventHandlerWrapper<EventType>>(h);
            if (wrapper->callback) return true;
        }
        return false;
	}

    template<typename EventType>
    bool Unsubscribe(const std::string& layerName, std::function<void(const EventType&)> handler)
    {
        EventLayer* layer = GetLayer(layerName);
        if (nullptr == layer) return false;
        auto& handlers = layer->handlers;
        auto it = std::remove_if(handlers.begin(), handlers.end(),
            [&handler](const std::shared_ptr<void>& h) {
                auto wrapper = std::static_pointer_cast<EventHandlerWrapper<EventType>>(h);
                return wrapper->callback.target_type() == handler.target_type() &&
                    wrapper->callback.target<void(const EventType&)>() == handler.target<void(const EventType&)>();
            });
        if (it != handlers.end())
        {
            handlers.erase(it, handlers.end());
            return true;
        }
        return false;
	}

    template<typename EventType>
    void Dispatch(const EventType& event)
    {
        for (auto& layer : layers)
        {
            layer->dispatcher.trigger(event);
        }
    }

    template<typename EventType, typename... Args>
    void Trigger(Args&&... args)
    {
        EventType event{ std::forward<Args>(args)... };
        Dispatch(event);
    }

private:
    EventLayer* GetLayer(const std::string& name);

    std::vector<std::shared_ptr<EventLayer>> layers;
};
