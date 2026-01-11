using System;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

internal static class HeliumNative
{
    public class FloatJsonConverter : JsonConverter<float>
    {
        public override float Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType == JsonTokenType.String)
            {
                if (float.TryParse(reader.GetString(), out float value))
                    return value;
            }
            return reader.GetSingle();
        }

        public override void Write(Utf8JsonWriter writer, float value, JsonSerializerOptions options)
        {
            // Write as number, or handle precision if needed
            writer.WriteNumberValue(value);
        }
    }

    public class DoubleJsonConverter : JsonConverter<double>
    {
        public override double Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType == JsonTokenType.String)
            {
                if (double.TryParse(reader.GetString(), out double value))
                    return value;
            }
            return reader.GetDouble();
        }

        public override void Write(Utf8JsonWriter writer, double value, JsonSerializerOptions options)
        {
            writer.WriteNumberValue(value);
        }
    }

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

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void He_ManagedToNative([MarshalAs(UnmanagedType.LPStr)] string command);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void ManagedToNativeDelegate([MarshalAs(UnmanagedType.LPStr)] string jsonString);

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void He_SetManagedToNativeCallback(ManagedToNativeDelegate callback);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void NativeToManagedDelegate([MarshalAs(UnmanagedType.LPStr)] string jsonString);

    [DllImport("Helium.dll", CallingConvention = CallingConvention.Cdecl)]
    public static extern void He_SetNativeToManagedCallback(NativeToManagedDelegate callback);
}
