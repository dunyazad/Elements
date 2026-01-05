#pragma once

#include <map>

#include <Helium/Systems/HeliumSystem.h>

#include <Eigen/Dense>

class InputSystem : public HeliumSystem
{
public:
    InputSystem(HeliumCore* core);
    virtual ~InputSystem() = default;

    void Initialize() override;
    void Update(float dt) override;

    bool IsKeyDown(int key);
    bool IsKeyPressed(int key);
    bool IsKeyReleased(int key);

    bool IsMouseButtonDown(int button); // 0: Left, 1: Right, 2: Middle
    Eigen::Vector2f GetMousePosition() const;

	void OnMouseWheel(float xoffset, float yoffset);

    void ProcessMessage(UINT message, WPARAM wParam, LPARAM lParam);
    
private:
    bool keyStates[256];
    bool prevKeyStates[256];

    bool mouseStates[3];
    bool prevMouseStates[3];

    Eigen::Vector2f mousePos;
    Eigen::Vector2f lastMousePos;

    static std::map<int, bool> s_keyStates;
};
