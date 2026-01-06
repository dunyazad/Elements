#pragma once

struct FrameEvent
{
	unsigned int frameNo;
	float timeDelta;

	FrameEvent(unsigned int fno, float dt) : frameNo(fno), timeDelta(dt) {}
};

struct FrameBufferResizeEvent
{
	int width = 0;
	int height = 0;

	FrameBufferResizeEvent(int w, int h) : width(w), height(h) {}
};

enum class KeyModifiers : int
{
	None = 0,
	Shift = 1 << 0,
	Control = 1 << 1,
	Alt = 1 << 2,       // VK_MENU

	LeftShift = 1 << 3,
	RightShift = 1 << 4,

	LeftControl = 1 << 5,
	RightControl = 1 << 6,

	LeftAlt = 1 << 7,
	RightAlt = 1 << 8
};

inline int operator|(KeyModifiers a, KeyModifiers b) { return static_cast<int>(a) | static_cast<int>(b); }
inline int operator|(int a, KeyModifiers b) { return a | static_cast<int>(b); }
inline bool operator&(int a, KeyModifiers b) { return (a & static_cast<int>(b)) != 0; }

struct KeyEvent
{
	int keyCode;
	int action;
	int modifiers;

	KeyEvent(int key, int action, int modifiers) : keyCode(key), action(action), modifiers(modifiers) {}

	inline bool HasModifier(KeyModifiers modifiers) const { return (this->modifiers & static_cast<int>(modifiers)) != 0; }
	inline bool IsCtrlPressed() const { return HasModifier(KeyModifiers::Control); }
	inline bool IsShiftPressed() const { return HasModifier(KeyModifiers::Shift); }
	inline bool IsAltPressed() const { return HasModifier(KeyModifiers::Alt); }
};

struct MousePositionEvent
{
    float xpos;
    float ypos;

    MousePositionEvent(float xpos, float ypos) : xpos(xpos), ypos(ypos) {}
};

struct MouseButtonEvent
{
    int button; // 0: Left, 1: Right, 2: Middle
    int action; // 0: Release, 1: Press
    int modifiers;
	float xpos = 0.0;
	float ypos = 0.0;

    MouseButtonEvent(int btn, int action, int modifiers, float xpos, float ypos) :
		button(btn), action(action), modifiers(modifiers), xpos(xpos), ypos(ypos) {}

	inline bool HasModifier(KeyModifiers modifiers) const { return (this->modifiers & static_cast<int>(modifiers)) != 0; }
	inline bool IsCtrlPressed() const { return HasModifier(KeyModifiers::Control); }
	inline bool IsShiftPressed() const { return HasModifier(KeyModifiers::Shift); }
	inline bool IsAltPressed() const { return HasModifier(KeyModifiers::Alt); }
};

struct MouseWheelEvent
{
	float xoffset = 0.0;
	float yoffset = 0.0;

	MouseWheelEvent(float xoff, float yoff) : xoffset(xoff), yoffset(yoff) {}
};

struct JoystickEvent
{
	float AxisX = 0.0f;
	float AxisY = 0.0f;
	float AxisZ = 0.0f;
	float RotX = 0.0f;
	float RotY = 0.0f;
	float RotZ = 0.0f;
	bool Buttons[16] = {
		false, false, false, false,
		false, false, false, false,
		false, false, false, false,
		false, false, false, false };

	JoystickEvent(float ax, float ay, float az, float rx, float ry, float rz, const bool* btns)
		: AxisX(ax), AxisY(ay), AxisZ(az), RotX(rx), RotY(ry), RotZ(rz)
	{
		for (int i = 0; i < 16; i++)
		{
			Buttons[i] = btns[i];
		}
	}
};

struct WindowResizeEvent
{
    int Width;
    int Height;
    WindowResizeEvent(int w, int h) : Width(w), Height(h) {}
};

struct CustomEvent
{
	std::string jsonString;
	CustomEvent(const std::string& jsonString) : jsonString(jsonString) {}
};
