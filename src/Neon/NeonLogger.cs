using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Threading;

public static class NeonLogger
{
    public enum LogLevel
    {
        Info,
        Warn,
        Error,
        Debug
    }

    // key, level, formattedLine
    public static Action<
        IReadOnlyList<(string? key, LogLevel level, string line)>
    >? OnBatch;

    private static readonly ConcurrentQueue<(string? key, LogLevel level, string line)> _queue = new();
    private static readonly Dictionary<(LogLevel level, string key), int> _counts = new();


    private static DispatcherTimer? _timer;
    private static bool _running;

    private const int FlushIntervalMs = 50;
    private const int MaxBatchSize = 500;

    public static void Initialize()
    {
        _timer = new DispatcherTimer
        {
            Interval = TimeSpan.FromMilliseconds(FlushIntervalMs)
        };
        _timer.Tick += Flush;
    }

    public static void Log(string? key, string value)
    {
        Log(LogLevel.Info, key, value);
    }

    public static void Log(LogLevel level, string value)
    {
        Log(level, null, value);
    }

    public static void Log(LogLevel level, string? key, string value)
    {
        if (string.IsNullOrEmpty(key))
        {
            _queue.Enqueue((null, level, value));
        }
        else
        {
            var compositeKey = (level, key);

            if (!_counts.TryGetValue(compositeKey, out int count))
                count = 0;

            count++;
            _counts[compositeKey] = count;

            string line = $"[{level}][{key}][count={count}] {value}";
            _queue.Enqueue((key, level, line));
        }

        if (!_running)
        {
            _running = true;
            Application.Current.Dispatcher.BeginInvoke(() => _timer!.Start());
        }
    }

    private static void Flush(object? sender, EventArgs e)
    {
        var batch = new List<(string? key, LogLevel level, string line)>(MaxBatchSize);

        while (batch.Count < MaxBatchSize && _queue.TryDequeue(out var item))
            batch.Add(item);

        if (batch.Count > 0)
            OnBatch?.Invoke(batch);

        if (_queue.IsEmpty)
        {
            _timer!.Stop();
            _running = false;
        }
    }
}
