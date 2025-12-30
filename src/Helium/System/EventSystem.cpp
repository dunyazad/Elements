#include "pch.h"
#include <Helium/Systems/EventSystem.h>
#include <Helium/HeliumCore.h>

EventSystem::EventSystem(HeliumCore* core)
    : HeliumSystem(core)
{
}

void EventSystem::Initialize()
{
    m_Core->Log("System", "EventSystem Initialized");
}

void EventSystem::Update(float dt)
{
    m_Dispatcher.update();
}

void EventSystem::Shutdown()
{
    m_Dispatcher.clear();
}
