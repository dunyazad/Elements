#include "pch.h"
#include <Helium/Systems/InputSystem.h>
#include <Helium/HeliumCore.h>
#include <Helium/HeliumEvents.h>

InputSystem::InputSystem(HeliumCore* core)
    : HeliumSystem(core)
    , m_MousePos(0.0f, 0.0f)
{
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
    EventSystem* eventSystem = m_Core->GetEventSystem();

    memcpy(m_PrevKeyStates, m_KeyStates, sizeof(m_KeyStates));

    for (int i = 0; i < 256; ++i)
    {
        bool isDown = (GetAsyncKeyState(i) & 0x8000) != 0;
        m_KeyStates[i] = isDown;

        if (m_KeyStates[i] != m_PrevKeyStates[i])
        {
            int action = isDown ? 1 : 0; // 1: Press, 0: Release
            int mods = 0; // (필요하면 GetKeyState(VK_CONTROL) 등으로 채움)

            if (eventSystem)
            {
                eventSystem->Enqueue<KeyEvent>(i, action, mods);
            }
        }
    }

    memcpy(m_PrevMouseStates, m_MouseStates, sizeof(m_MouseStates));

    // VK 코드 매핑: Left(0), Right(1), Middle(2)
    int mouseVKs[3] = { VK_LBUTTON, VK_RBUTTON, VK_MBUTTON };

    for (int i = 0; i < 3; ++i)
    {
        bool isDown = (GetAsyncKeyState(mouseVKs[i]) & 0x8000) != 0;
        m_MouseStates[i] = isDown;

        if (m_MouseStates[i] != m_PrevMouseStates[i])
        {
            int action = isDown ? 1 : 0;
            if (eventSystem)
            {
                eventSystem->Enqueue<MouseButtonEvent>(i, action, 0);
            }
        }
    }

    POINT pt;
    if (GetCursorPos(&pt))
    {
        ScreenToClient(m_Core->GetHWND(), &pt);
        float x = static_cast<float>(pt.x);
        float y = static_cast<float>(pt.y);

        if (x != m_MousePos.x() || y != m_MousePos.y())
        {
            m_MousePos.x() = x;
            m_MousePos.y() = y;

            if (eventSystem)
            {
                eventSystem->Enqueue<MousePositionEvent>(x, y);
            }
        }
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