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

struct KeyEvent
{
    int keyCode;
    int action;
    int mods;

    KeyEvent(int key, int action, int mods) : keyCode(key), action(action), mods(mods) {}
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
    int mods;
	float xpos = 0.0;
	float ypos = 0.0;

    MouseButtonEvent(int btn, int action, int mods, float xpos, float ypos) :
		button(btn), action(action), mods(mods), xpos(xpos), ypos(ypos) {}
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
