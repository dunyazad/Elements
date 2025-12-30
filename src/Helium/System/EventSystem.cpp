#include "pch.h"
#include <Helium/Systems/EventSystem.h>
#include <Helium/HeliumCore.h>

EventSystem::EventSystem(HeliumCore* core)
    : HeliumSystem(core)
{
}

void EventSystem::Initialize()
{
    // 초기화 로그
    m_Core->Log("System", "EventSystem Initialized");
}

void EventSystem::Update(float dt)
{
    // 큐에 쌓인 이벤트들을 처리
    m_Dispatcher.update();
}

void EventSystem::Shutdown()
{
    // 연결된 모든 리스너 해제
    m_Dispatcher.clear();
}
