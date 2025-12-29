using System;
using System.Runtime.InteropServices;

internal static class HeliumNative
{
    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bool Helium_Initialize(IntPtr hwnd);

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void Helium_Render();

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void Helium_Resize(int width, int height);

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void Helium_Shutdown();
}
