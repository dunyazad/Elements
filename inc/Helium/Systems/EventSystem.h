#pragma once

#include <Helium/Systems/HeliumSystem.h>
#include <entt/entt.hpp>
#include <Helium/HeliumCommon.h>

class EventSystem : public HeliumSystem
{
public:
    EventSystem(HeliumCore* core);
    virtual ~EventSystem() = default;

    void Initialize() override;
    void Update(float dt) override;
    void Shutdown() override;

    template<typename T, typename... Args>
    void Trigger(Args&&... args)
    {
        m_Dispatcher.trigger(T(std::forward<Args>(args)...));
    }

    template<typename T, typename... Args>
    void Enqueue(Args&&... args)
    {
        m_Dispatcher.enqueue<T>(std::forward<Args>(args)...);
    }

    template<typename T, typename Receiver>
    void Subscribe(Receiver* instance, void(Receiver::* callback)(const T&))
    {
        m_Dispatcher.sink<T>().template connect<Receiver, callback>(instance);
    }

    template<typename T, typename Receiver>
    void Unsubscribe(Receiver* instance)
    {
        m_Dispatcher.sink<T>().template disconnect<Receiver>(instance);
    }

private:
    entt::dispatcher m_Dispatcher;
};
