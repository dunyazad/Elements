using Neon.Controls;
using System;
using System.Collections.Generic;
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
        private HeliumLogDelegate _logDelegate;
        private bool _autoScroll = true;

        private readonly Stopwatch _stopwatch = new();
        private long _lastTicks = 0;

        private readonly Dictionary<(NeonLogger.LogLevel level, string key), int> _keyToIndex = new();

        private double _lastLogHeight = 224.0;
        private const double MinLogHeight = 30.0;

        
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

        private void Window_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Escape)
            {
                this.Close();
            }
        }

        private void OnRendering(object? sender, EventArgs e)
        {
            long currentTicks = _stopwatch.ElapsedTicks;
            float dt = (float)(currentTicks - _lastTicks) / Stopwatch.Frequency;
            _lastTicks = currentTicks;

            if (dt > 0.1f) dt = 0.1f; // 프레임 튐 방지

            try
            {
                HeliumNative.He_Update(dt);
                HeliumNative.He_Render();
            }
            catch { }
        }

        // --------------------------------------------------------------------
        // 로그 관련
        // --------------------------------------------------------------------
        private void OnHeliumLog(NeonLogger.LogLevel level, string? key, string value)
        {
            NeonLogger.Log(level, key, value);
        }

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

        private void LogList_ScrollChanged(object sender, ScrollChangedEventArgs e)
        {
            _autoScroll = e.VerticalOffset >= e.ExtentHeight - e.ViewportHeight - 1;
        }

        private void LogList_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.C && Keyboard.Modifiers == ModifierKeys.Control)
            {
                var sb = new StringBuilder();
                foreach (var item in LogList.SelectedItems)
                    sb.AppendLine(item.ToString());

                if (sb.Length > 0)
                    Clipboard.SetText(sb.ToString());
                e.Handled = true;
            }
        }

        private void LogToggle_Click(object sender, RoutedEventArgs e)
        {
            if (LogToggle.IsChecked == true)
            {
                // 펼치기
                MainLogRow.Height = new GridLength(_lastLogHeight);
            }
            else
            {
                // 접기 (현재 높이 저장)
                if (MainLogRow.Height.Value > MinLogHeight)
                {
                    _lastLogHeight = MainLogRow.Height.Value;
                }
                MainLogRow.Height = new GridLength(MinLogHeight);
            }
        }

        private void TestEntity_Click(object sender, RoutedEventArgs e)
        {
            HeliumNative.He_TestCreateEntity();
        }

        private void MenuExit_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        // --------------------------------------------------------------------
        // Window TitleBar 버튼 핸들러
        // --------------------------------------------------------------------
        private void Min_Click(object sender, RoutedEventArgs e)
        {
            this.WindowState = WindowState.Minimized;
        }

        private void Max_Click(object sender, RoutedEventArgs e)
        {
            if (this.WindowState == WindowState.Maximized)
            {
                this.WindowState = WindowState.Normal;
                MaxBtn.Content = "1";
            }
            else
            {
                this.WindowState = WindowState.Maximized;
                MaxBtn.Content = "2";
            }
        }

        private void Close_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }

        protected override void OnClosed(EventArgs e)
        {
            CompositionTarget.Rendering -= OnRendering;
            base.OnClosed(e);
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
    }
}