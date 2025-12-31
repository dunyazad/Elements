#include "pch.h"
#include <Helium/Systems/EventSystem.h>
#include <Helium/HeliumCore.h>

EventSystem::EventSystem(HeliumCore* core)
    : HeliumSystem(core)
{
}

void EventSystem::Initialize()
{
}

void EventSystem::Update(float timeDelta)
{
	auto& dispatcher = Helium.GetDispatcher();
	dispatcher.update();
}
