#pragma once
#include <Helium/Systems/HeliumSystem.h>

class InputSystem : public HeliumSystem
{
public:
    InputSystem(HeliumCore* core);
    virtual ~InputSystem() = default;

    void Initialize() override;
    void Update(float dt) override;

    bool IsKeyDown(int key);        // 누르고 있는 동안 true
    bool IsKeyPressed(int key);     // 누른 순간 한 번만 true
    bool IsKeyReleased(int key);    // 뗀 순간 한 번만 true

    bool IsMouseButtonDown(int button); // 0: Left, 1: Right, 2: Middle
    Eigen::Vector2f GetMousePosition() const;

private:
    bool m_KeyStates[256];
    bool m_PrevKeyStates[256];

    bool m_MouseStates[3];
    bool m_PrevMouseStates[3];

    Eigen::Vector2f m_MousePos;
};
