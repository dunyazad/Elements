#pragma once

#include <Helium/HeliumCore.h>

template <typename T>
class EventCallback
{
public:
    static void OnEvent(const T& event)
    {
        for (auto* callback : instances)
        {
            callback->callback(callback->target, event);
        }
    }

    EventCallback(Entity target, std::function<void(Entity, const T&)> callback)
        : target(target), callback(std::move(callback))
    {
        Helium.GetDispatcher().sink<T>().connect<&OnEvent>();
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
