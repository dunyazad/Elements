#include "pch.h"
#include <Helium/Systems/InputSystem.h>
#include <Helium/HeliumCore.h>
#include <Helium/HeliumEvents.h>

InputSystem::InputSystem(HeliumCore* core)
    : HeliumSystem(core)
    , mousePos(0.0f, 0.0f)
{
    memset(keyStates, 0, sizeof(keyStates));
    memset(prevKeyStates, 0, sizeof(prevKeyStates));
    memset(mouseStates, 0, sizeof(mouseStates));
    memset(prevMouseStates, 0, sizeof(prevMouseStates));
}

void InputSystem::Initialize()
{
    //core->Log(HE_LOG_INFO, "System", "InputSystem Initialized");
}

void InputSystem::Update(float dt)
{
    auto& dispatcher = Helium.GetDispatcher();

    memcpy(prevKeyStates, keyStates, sizeof(keyStates));

    for (int i = 0; i < 256; ++i)
    {
        bool isDown = (GetAsyncKeyState(i) & 0x8000) != 0;
        keyStates[i] = isDown;

        if (keyStates[i] != prevKeyStates[i])
        {
            int action = isDown ? 1 : 0; // 1: Press, 0: Release
            int mods = 0; // (필요하면 GetKeyState(VK_CONTROL) 등으로 채움)

            dispatcher.enqueue<KeyEvent>({ i, action, mods });
        }
    }

    memcpy(prevMouseStates, mouseStates, sizeof(mouseStates));

    // VK 코드 매핑: Left(0), Right(1), Middle(2)
    int mouseVKs[3] = { VK_LBUTTON, VK_RBUTTON, VK_MBUTTON };

    for (int i = 0; i < 3; ++i)
    {
        bool isDown = (GetAsyncKeyState(mouseVKs[i]) & 0x8000) != 0;
        mouseStates[i] = isDown;

        if (mouseStates[i] != prevMouseStates[i])
        {
            int action = isDown ? 1 : 0;
            dispatcher.enqueue<MouseButtonEvent>({ i, action, 0, mousePos.x(), mousePos.y() });
        }
    }

    POINT pt;
    if (GetCursorPos(&pt))
    {
        ScreenToClient(core->GetHWND(), &pt);
        float x = static_cast<float>(pt.x);
        float y = static_cast<float>(pt.y);

        if (x != mousePos.x() || y != mousePos.y())
        {
            mousePos.x() = x;
            mousePos.y() = y;

            dispatcher.enqueue<MousePositionEvent>({ x, y });
        }
    }
}

bool InputSystem::IsKeyDown(int key)
{
    return keyStates[key];
}

bool InputSystem::IsKeyPressed(int key)
{
    // 이번엔 눌려있고(Current), 저번엔 안 눌려있었으면(Prev) -> Pressed
    return keyStates[key] && !prevKeyStates[key];
}

bool InputSystem::IsKeyReleased(int key)
{
    // 이번엔 안 눌려있고, 저번엔 눌려있었으면 -> Released
    return !keyStates[key] && prevKeyStates[key];
}

bool InputSystem::IsMouseButtonDown(int button)
{
    if (button < 0 || button > 2) return false;
    return mouseStates[button];
}

Eigen::Vector2f InputSystem::GetMousePosition() const
{
    return mousePos;
}

void InputSystem::OnMouseWheel(float xoffset, float yoffset)
{
    auto& dispatcher = Helium.GetDispatcher();
	dispatcher.enqueue<MouseWheelEvent>({ xoffset, yoffset });
}