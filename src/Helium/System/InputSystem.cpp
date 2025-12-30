#include "pch.h"
#include <Helium/Systems/InputSystem.h>
#include <Helium/HeliumCore.h>

InputSystem::InputSystem(HeliumCore* core)
    : HeliumSystem(core)
    , m_MousePos(0.0f, 0.0f)
{
    // 상태 배열 초기화
    memset(m_KeyStates, 0, sizeof(m_KeyStates));
    memset(m_PrevKeyStates, 0, sizeof(m_PrevKeyStates));
    memset(m_MouseStates, 0, sizeof(m_MouseStates));
    memset(m_PrevMouseStates, 0, sizeof(m_PrevMouseStates));
}

void InputSystem::Initialize()
{
    m_Core->Log("System", "InputSystem Initialized");
}

void InputSystem::Update(float dt)
{
    memcpy(m_PrevKeyStates, m_KeyStates, sizeof(m_KeyStates));
    memcpy(m_PrevMouseStates, m_MouseStates, sizeof(m_MouseStates));

    for (int i = 0; i < 256; ++i)
    {
        m_KeyStates[i] = (GetAsyncKeyState(i) & 0x8000) != 0;
    }

    m_MouseStates[0] = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
    m_MouseStates[1] = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
    m_MouseStates[2] = (GetAsyncKeyState(VK_MBUTTON) & 0x8000) != 0;

    POINT pt;
    if (GetCursorPos(&pt))
    {
        // 스크린 좌표를 클라이언트(렌더링 영역) 좌표로 변환
        // (HeliumCore가 HWND를 가지고 있다고 가정)
        // HWND를 가져오는 방법은 Core에 GetHWND()를 추가하거나 친구 클래스로 접근
        ScreenToClient(GetActiveWindow(), &pt);
        m_MousePos.x() = static_cast<float>(pt.x);
        m_MousePos.y() = static_cast<float>(pt.y);
    }
}

bool InputSystem::IsKeyDown(int key)
{
    return m_KeyStates[key];
}

bool InputSystem::IsKeyPressed(int key)
{
    // 이번엔 눌려있고(Current), 저번엔 안 눌려있었으면(Prev) -> Pressed
    return m_KeyStates[key] && !m_PrevKeyStates[key];
}

bool InputSystem::IsKeyReleased(int key)
{
    // 이번엔 안 눌려있고, 저번엔 눌려있었으면 -> Released
    return !m_KeyStates[key] && m_PrevKeyStates[key];
}

bool InputSystem::IsMouseButtonDown(int button)
{
    if (button < 0 || button > 2) return false;
    return m_MouseStates[button];
}

Eigen::Vector2f InputSystem::GetMousePosition() const
{
    return m_MousePos;
}