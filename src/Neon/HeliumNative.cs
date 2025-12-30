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

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void HeliumLogDelegate(
        NeonLogger.LogLevel level,
        [MarshalAs(UnmanagedType.LPStr)] string? key,
        [MarshalAs(UnmanagedType.LPStr)] string value
    );

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void Helium_SetLogCallback(HeliumLogDelegate callback);
}
