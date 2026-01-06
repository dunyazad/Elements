#include "pch.h"
#include <Helium/Systems/InputSystem.h>
#include <Helium/HeliumCore.h>
#include <Helium/HeliumEvents.h>
#include <Helium/HeliumLog.h>

#include <windowsx.h> // GET_X_LPARAM, GET_Y_LPARAM 매크로 사용

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
}

void InputSystem::Update(float dt)
{
    memcpy(prevKeyStates, keyStates, sizeof(keyStates));
    memcpy(prevMouseStates, mouseStates, sizeof(mouseStates));
}

void InputSystem::ProcessMessage(UINT message, WPARAM wParam, LPARAM lParam)
{
    auto& dispatcher = Helium.GetDispatcher();
    int currentMods = 0;

    if (GetKeyState(VK_SHIFT) & 0x8000)   currentMods |= static_cast<int>(KeyModifiers::Shift);
    if (GetKeyState(VK_CONTROL) & 0x8000) currentMods |= static_cast<int>(KeyModifiers::Control);
    if (GetKeyState(VK_MENU) & 0x8000)    currentMods |= static_cast<int>(KeyModifiers::Alt);

    switch (message)
    {
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
    {
        int key = static_cast<int>(wParam);
        if (key < 256)
        {
            keyStates[key] = true;
            dispatcher.enqueue<KeyEvent>({ key, 1, currentMods });
        }
        break;
    }

    case WM_KEYUP:
    case WM_SYSKEYUP:
    {
        int key = static_cast<int>(wParam);
        if (key < 256)
        {
            keyStates[key] = false;
            dispatcher.enqueue<KeyEvent>({ key, 0, currentMods });
        }
        break;
    }

    case WM_MOUSEMOVE:
    {
        int x = GET_X_LPARAM(lParam);
        int y = GET_Y_LPARAM(lParam);

        mousePos.x() = static_cast<float>(x);
        mousePos.y() = static_cast<float>(y);

        dispatcher.enqueue<MousePositionEvent>({ mousePos.x(), mousePos.y() });
        break;
    }

    case WM_LBUTTONDOWN:
    case WM_RBUTTONDOWN:
    case WM_MBUTTONDOWN:
    {
        int button = -1;
        if (message == WM_LBUTTONDOWN) button = 0;
        else if (message == WM_RBUTTONDOWN) button = 1;
        else if (message == WM_MBUTTONDOWN) button = 2;

        if (button != -1)
        {
            mouseStates[button] = true;
            dispatcher.enqueue<MouseButtonEvent>({ button, 1, currentMods, mousePos.x(), mousePos.y() });
        }
        break;
    }

    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_MBUTTONUP:
    {
        int button = -1;
        if (message == WM_LBUTTONUP) button = 0;
        else if (message == WM_RBUTTONUP) button = 1;
        else if (message == WM_MBUTTONUP) button = 2;

        if (button != -1)
        {
            mouseStates[button] = false;
            dispatcher.enqueue<MouseButtonEvent>({ button, 0, currentMods, mousePos.x(), mousePos.y() });
        }
        break;
    }

    case WM_MOUSEWHEEL:
    {
        float wheelDelta = static_cast<float>(GET_WHEEL_DELTA_WPARAM(wParam)) / WHEEL_DELTA;
        OnMouseWheel(0.0f, wheelDelta);
        break;
    }

    case WM_MOUSEHWHEEL:
    {
        float wheelDelta = static_cast<float>(GET_WHEEL_DELTA_WPARAM(wParam)) / WHEEL_DELTA;
        OnMouseWheel(wheelDelta, 0.0f);
        break;
    }
    }
}

bool InputSystem::IsKeyDown(int key)
{
    return keyStates[key];
}

bool InputSystem::IsKeyPressed(int key)
{
    // 현재 프레임엔 눌려있고, 이전 프레임(백업본)엔 안 눌려있었음
    return keyStates[key] && !prevKeyStates[key];
}

bool InputSystem::IsKeyReleased(int key)
{
    // 현재 프레임엔 안 눌려있고, 이전 프레임(백업본)엔 눌려있었음
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
