#pragma once

#include <vector>
#include <map>

#include <Helium/Systems/HeliumSystem.h>

#include <entt/entt.hpp>

class HeliumCore;

class EventSystem : public HeliumSystem
{
public:
	EventSystem(HeliumCore* core);

	virtual void Initialize();
	virtual void Update(float timeDelta);

    template<typename EventType, typename Handler>
    void Subscribe(Handler&& handler)
    {
        dispatcher.sink<EventType>().connect(std::forward<Handler>(handler));
    }

    template<typename EventType>
    void Dispatch(const EventType& event)
    {
        dispatcher.trigger(event);
    }

    template<typename EventType, typename... Args>
    void Trigger(Args&&... args)
    {
        //dispatcher.trigger<EventType>(std::forward<Args>(args)...);
    }

	

private:
    entt::dispatcher dispatcher;
};
