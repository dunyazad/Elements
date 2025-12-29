using System;
using System.IO;
using System.Runtime.InteropServices;

internal static class ConsoleHelper
{
    [DllImport("kernel32.dll")]
    private static extern bool AllocConsole();

    [DllImport("kernel32.dll")]
    private static extern IntPtr GetStdHandle(int nStdHandle);

    private const int STD_OUTPUT_HANDLE = -11;

    public static void OpenConsole()
    {
        AllocConsole();

        // .NET stdout을 콘솔에 연결
        var stdout = Console.OpenStandardOutput();
        var writer = new StreamWriter(stdout) { AutoFlush = true };
        Console.SetOut(writer);
    }
}
