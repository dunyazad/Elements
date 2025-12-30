using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Threading;

public static class NeonLogger
{
    // key, formattedLine
    public static Action<IReadOnlyList<(string? key, string line)>>? OnBatch;

    private static readonly ConcurrentQueue<(string? key, string line)> _queue = new();
    private static readonly Dictionary<string, int> _counts = new();

    private static DispatcherTimer? _timer;
    private static bool _running;

    private const int FlushIntervalMs = 50;
    private const int MaxBatchSize = 500;   // 한 번에 UI로 보내는 최대 줄 수

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
        if (string.IsNullOrEmpty(key))
        {
            _queue.Enqueue((null, value));
        }
        else
        {
            if (!_counts.TryGetValue(key, out int count))
                count = 0;

            count++;
            _counts[key] = count;

            string line = $"[{key}][count={count}] {value}";
            _queue.Enqueue((key, line));
        }

        if (!_running)
        {
            _running = true;
            Application.Current.Dispatcher.BeginInvoke(() => _timer!.Start());
        }
    }

    private static void Flush(object? sender, EventArgs e)
    {
        var batch = new List<(string? key, string line)>(MaxBatchSize);

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
