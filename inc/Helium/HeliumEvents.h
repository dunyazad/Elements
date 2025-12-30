#pragma once

struct WindowResizeEvent
{
    int Width;
    int Height;

    WindowResizeEvent(int width, int height)
        : Width(width), Height(height)
    {
    }
};

// struct KeyPressedEvent { int KeyCode; };
// struct MouseMovedEvent { float X, Y; };