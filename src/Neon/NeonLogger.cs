using System;
using System.Windows;

public static class NeonLogger
{
    public static Action<string>? OnLog;

    public static void Log(string message)
    {
        Application.Current.Dispatcher.Invoke(() =>
        {
            OnLog?.Invoke(message);
        });
    }
}
