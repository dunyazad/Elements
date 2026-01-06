using System;
using System.Runtime.InteropServices;

internal static class HeliumNative
{
    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern bool He_Initialize(IntPtr hwnd, int backendType);

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void He_Update(float dt);

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void He_Render();

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void He_Resize(int width, int height);

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void He_Shutdown();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void HeliumLogDelegate(
        NeonLogger.LogLevel level,
        [MarshalAs(UnmanagedType.LPStr)] string? key,
        [MarshalAs(UnmanagedType.LPStr)] string value
    );

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void He_ProcessMessage(uint message, IntPtr wParam, IntPtr lParam);

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void He_SetLogCallback(HeliumLogDelegate callback);

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void He_ProcessMouseWheel(float xoffset, float yoffset);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void PointCloudCreatedDelegate(int ID, [MarshalAs(UnmanagedType.LPStr)] string fileName, [MarshalAs(UnmanagedType.LPStr)] string name);

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void He_SetPointCloudCreatedCallback(PointCloudCreatedDelegate callback);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void PointCloudDeletedDelegate(int ID);

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void He_SetPointCloudDeletedCallback(PointCloudDeletedDelegate callback);

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void He_ExecuteCommand([MarshalAs(UnmanagedType.LPStr)] string command);
}
