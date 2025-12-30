using Neon.Controls;
using System;
using System.Diagnostics;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using static HeliumNative;

namespace Neon
{
    public partial class MainWindow : Window
    {
        private double _lastLogHeight = 224.0;
        private const double MinLogHeight = 24.0;

        private HeliumLogDelegate _logDelegate;
        private bool _autoScroll = true;

        private readonly Stopwatch _stopwatch = new();
        private long _lastTicks = 0;

        public MainWindow()
        {
            InitializeComponent();

            NeonLogger.Initialize();
            NeonLogger.OnBatch = OnLogBatch;

            _logDelegate = OnHeliumLog;
            HeliumNative.He_SetLogCallback(_logDelegate);

            NeonLogger.Log("system", "Application started");

            _stopwatch.Start();

            CompositionTarget.Rendering += OnRendering;
        }

        private readonly Dictionary<(NeonLogger.LogLevel level, string key), int> _keyToIndex = new();

        private void OnLogBatch(IReadOnlyList<(string? key, NeonLogger.LogLevel level, string line)> batch)
        {
            foreach (var (key, level, line) in batch)
            {
                if (key == null)
                {
                    LogList.Items.Add(MakeItem(level, line));
                    continue;
                }

                var id = (level, key);

                if (_keyToIndex.TryGetValue(id, out int index))
                {
                    LogList.Items[index] = MakeItem(level, line);
                }
                else
                {
                    int newIndex = LogList.Items.Count;
                    LogList.Items.Add(MakeItem(level, line));
                    _keyToIndex[id] = newIndex;
                }
            }
        }

        private object MakeItem(NeonLogger.LogLevel level, string line)
        {
            Brush color = level switch
            {
                NeonLogger.LogLevel.Info => Brushes.LightGray,
                NeonLogger.LogLevel.Warn => Brushes.Yellow,
                NeonLogger.LogLevel.Error => Brushes.OrangeRed,
                NeonLogger.LogLevel.Debug => Brushes.Cyan,
                _ => Brushes.White
            };

            return new LogItem(line, color);
        }

        private sealed class LogItem
        {
            public string Text { get; }
            public Brush Foreground { get; }

            public LogItem(string text, Brush foreground)
            {
                Text = text;
                Foreground = foreground;
            }

            public override string ToString()
            {
                return Text;
            }
        }

        private void OnHeliumLog(NeonLogger.LogLevel level, string key, string value)
        {
            NeonLogger.Log(level, key, value);
        }

        private void LogList_ScrollChanged(object sender, ScrollChangedEventArgs e)
        {
            _autoScroll =
                e.VerticalOffset >= e.ExtentHeight - e.ViewportHeight - 1;
        }

        private void LogList_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.C && Keyboard.Modifiers == ModifierKeys.Control)
            {
                var sb = new StringBuilder();
                foreach (var item in LogList.SelectedItems)
                {
                    // LogItem.ToString()이 호출됨
                    sb.AppendLine(item.ToString());
                }

                if (sb.Length > 0)
                {
                    Clipboard.SetText(sb.ToString());
                }
                e.Handled = true;
            }
        }

        private void OnRendering(object? sender, EventArgs e)
        {
            long currentTicks = _stopwatch.ElapsedTicks;
            float dt = (float)(currentTicks - _lastTicks) / Stopwatch.Frequency;
            _lastTicks = currentTicks;

            if (dt > 0.1f) dt = 0.1f;

            try
            {
                HeliumNative.He_Update(dt);

                HeliumNative.He_Render();
            }
            catch { }
        }

        protected override void OnClosed(EventArgs e)
        {
            CompositionTarget.Rendering -= OnRendering;
            base.OnClosed(e);
        }

        private void LogToggle_Click(object sender, RoutedEventArgs e)
        {
            if (LogToggle.IsChecked == true)
            {
                // [펼치기] 기억해둔 높이로 복구
                MainLogRow.Height = new GridLength(_lastLogHeight);
                LogToggle.Content = "▼ Log";
            }
            else
            {
                // [접기] 현재 높이를 저장해두고, 버튼 높이(24)만큼만 남김
                // 단, 현재 높이가 버튼 높이보다 클 때만 저장 (접힌 상태 저장을 방지)
                if (MainLogRow.Height.Value > MinLogHeight)
                {
                    _lastLogHeight = MainLogRow.Height.Value;
                }

                MainLogRow.Height = new GridLength(MinLogHeight);
                LogToggle.Content = "▲ Log";
            }
        }

        private void TestEntity_Click(object sender, RoutedEventArgs e)
        {
            // C++로 요청 보냄 -> HeliumCore가 엔티티 생성 -> 로그 출력됨
            HeliumNative.He_TestCreateEntity();
        }
    }
}
