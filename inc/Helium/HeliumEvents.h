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

enum class KeyCode : int
{
	None = 0x00,

	// Control Keys
	Backspace = 0x08,	// VK_BACK
	Tab = 0x09,			// VK_TAB
	Clear = 0x0C,		// VK_CLEAR
	Enter = 0x0D,		// VK_RETURN
	Shift = 0x10,		// VK_SHIFT
	Control = 0x11,		// VK_CONTROL
	Alt = 0x12,			// VK_MENU
	Pause = 0x13,		// VK_PAUSE
	CapsLock = 0x14,	// VK_CAPITAL

	// IME Keys
	Kana = 0x15,		// VK_KANA, VK_HANGUL
	Junja = 0x17,		// VK_JUNJA
	Final = 0x18,		// VK_FINAL
	Hanja = 0x19,		// VK_HANJA, VK_KANJI
	Escape = 0x1B,		// VK_ESCAPE

	Convert = 0x1C,		// VK_CONVERT
	NonConvert = 0x1D,	// VK_NONCONVERT
	Accept = 0x1E,		// VK_ACCEPT
	ModeChange = 0x1F,	// VK_MODECHANGE

	Space = 0x20,		// VK_SPACE
	PageUp = 0x21,		// VK_PRIOR
	PageDown = 0x22,	// VK_NEXT
	End = 0x23,			// VK_END
	Home = 0x24,		// VK_HOME
	Left = 0x25,		// VK_LEFT
	Up = 0x26,			// VK_UP
	Right = 0x27,		// VK_RIGHT
	Down = 0x28,		// VK_DOWN
	Select = 0x29,		// VK_SELECT
	Print = 0x2A,		// VK_PRINT
	Execute = 0x2B,		// VK_EXECUTE
	PrintScreen = 0x2C,	// VK_SNAPSHOT
	Insert = 0x2D,		// VK_INSERT
	Delete = 0x2E,		// VK_DELETE
	Help = 0x2F,		// VK_HELP

	// Numbers
	D0 = 0x30,
	D1 = 0x31,
	D2 = 0x32,
	D3 = 0x33,
	D4 = 0x34,
	D5 = 0x35,
	D6 = 0x36,
	D7 = 0x37,
	D8 = 0x38,
	D9 = 0x39,

	// Letters
	A = 0x41,
	B = 0x42,
	C = 0x43,
	D = 0x44,
	E = 0x45,
	F = 0x46,
	G = 0x47,
	H = 0x48,
	I = 0x49,
	J = 0x4A,
	K = 0x4B,
	L = 0x4C,
	M = 0x4D,
	N = 0x4E,
	O = 0x4F,
	P = 0x50,
	Q = 0x51,
	R = 0x52,
	S = 0x53,
	T = 0x54,
	U = 0x55,
	V = 0x56,
	W = 0x57,
	X = 0x58,
	Y = 0x59,
	Z = 0x5A,

	// Windows Keys
	LeftWin = 0x5B,		// VK_LWIN
	RightWin = 0x5C,	// VK_RWIN
	Apps = 0x5D,		// VK_APPS
	Sleep = 0x5F,		// VK_SLEEP

	// Numpad
	NumPad0 = 0x60,		// VK_NUMPAD0
	NumPad1 = 0x61,		// VK_NUMPAD1
	NumPad2 = 0x62,		// VK_NUMPAD2
	NumPad3 = 0x63,		// VK_NUMPAD3
	NumPad4 = 0x64,		// VK_NUMPAD4
	NumPad5 = 0x65,		// VK_NUMPAD5
	NumPad6 = 0x66,		// VK_NUMPAD6
	NumPad7 = 0x67,		// VK_NUMPAD7
	NumPad8 = 0x68,		// VK_NUMPAD8
	NumPad9 = 0x69,		// VK_NUMPAD9
	Multiply = 0x6A,	// VK_MULTIPLY
	Add = 0x6B,			// VK_ADD
	Separator = 0x6C,	// VK_SEPARATOR
	Subtract = 0x6D,	// VK_SUBTRACT
	Decimal = 0x6E,		// VK_DECIMAL
	Divide = 0x6F,		// VK_DIVIDE

	// Function Keys
	F1 = 0x70,			// VK_F1
	F2 = 0x71,			// VK_F2
	F3 = 0x72,			// VK_F3
	F4 = 0x73,			// VK_F4
	F5 = 0x74,			// VK_F5
	F6 = 0x75,			// VK_F6
	F7 = 0x76,			// VK_F7
	F8 = 0x77,			// VK_F8
	F9 = 0x78,			// VK_F9
	F10 = 0x79,			// VK_F10
	F11 = 0x7A,			// VK_F11
	F12 = 0x7B,			// VK_F12
	F13 = 0x7C,			// VK_F13
	F14 = 0x7D,			// VK_F14
	F15 = 0x7E,			// VK_F15
	F16 = 0x7F,			// VK_F16
	F17 = 0x80,			// VK_F17
	F18 = 0x81,			// VK_F18
	F19 = 0x82,			// VK_F19
	F20 = 0x83,			// VK_F20
	F21 = 0x84,			// VK_F21
	F22 = 0x85,			// VK_F22
	F23 = 0x86,			// VK_F23
	F24 = 0x87,			// VK_F24

	NumLock = 0x90,		// VK_NUMLOCK
	ScrollLock = 0x91,	// VK_SCROLL

	// Specific Left/Right
	LeftShift = 0xA0,	// VK_LSHIFT
	RightShift = 0xA1,	// VK_RSHIFT
	LeftControl = 0xA2,	// VK_LCONTROL
	RightControl = 0xA3,// VK_RCONTROL
	LeftAlt = 0xA4,		// VK_LMENU
	RightAlt = 0xA5,	// VK_RMENU

	// Browser / Media Keys
	BrowserBack = 0xA6,
	BrowserForward = 0xA7,
	BrowserRefresh = 0xA8,
	BrowserStop = 0xA9,
	BrowserSearch = 0xAA,
	BrowserFavorites = 0xAB,
	BrowserHome = 0xAC,
	VolumeMute = 0xAD,
	VolumeDown = 0xAE,
	VolumeUp = 0xAF,
	MediaNextTrack = 0xB0,
	MediaPrevTrack = 0xB1,
	MediaStop = 0xB2,
	MediaPlayPause = 0xB3,

	// OEM Specific & Aliases
	Oem1 = 0xBA,		// VK_OEM_1
	Semicolon = 0xBA,	// ; :

	OemPlus = 0xBB,		// VK_OEM_PLUS
	Plus = 0xBB,		// +

	OemComma = 0xBC,	// VK_OEM_COMMA
	Comma = 0xBC,		// ,

	OemMinus = 0xBD,	// VK_OEM_MINUS
	Minus = 0xBD,		// -

	OemPeriod = 0xBE,	// VK_OEM_PERIOD
	Period = 0xBE,		// .

	Oem2 = 0xBF,		// VK_OEM_2
	Slash = 0xBF,		// / ?
	Question = 0xBF,

	Oem3 = 0xC0,		// VK_OEM_3
	Tilde = 0xC0,		// ` ~
	Backtick = 0xC0,

	Oem4 = 0xDB,		// VK_OEM_4
	LeftBracket = 0xDB,	// [ {

	Oem5 = 0xDC,		// VK_OEM_5
	Backslash = 0xDC,	// \ |
	Pipe = 0xDC,

	Oem6 = 0xDD,		// VK_OEM_6
	RightBracket = 0xDD,// ] }

	Oem7 = 0xDE,		// VK_OEM_7
	Quote = 0xDE,		// ' "

	Oem8 = 0xDF,		// VK_OEM_8
	Oem102 = 0xE2,		// VK_OEM_102

	ProcessKey = 0xE5,
	Packet = 0xE7,
	Attn = 0xF6,
	CrSel = 0xF7,
	ExSel = 0xF8,
	EraseEof = 0xF9,
	Play = 0xFA,
	Zoom = 0xFB,
	NoName = 0xFC,
	Pa1 = 0xFD,
	OemClear = 0xFE
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
	KeyCode keyCode;
	int action;
	int modifiers;

	KeyEvent(KeyCode key, int action, int modifiers) : keyCode(key), action(action), modifiers(modifiers) {}

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

enum class MouseButton : int
{
	None = -1,
	Left = 0,
	Right = 1,
	Middle = 2
};

struct MouseButtonEvent
{
	MouseButton button;
    int action; // 0: Release, 1: Press
    int modifiers;
	float xpos = 0.0;
	float ypos = 0.0;

    MouseButtonEvent(MouseButton button, int action, int modifiers, float xpos, float ypos) :
		button(button), action(action), modifiers(modifiers), xpos(xpos), ypos(ypos) {}

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
