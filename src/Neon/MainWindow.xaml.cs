using Neon.Controls;
using System;
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
        private const double LogHeight = 200;
        private const int MaxLogLines = 100_000;

        private HeliumLogDelegate _logDelegate;
        private bool _autoScroll = true;

        public MainWindow()
        {
            InitializeComponent();

            NeonLogger.Initialize();
            NeonLogger.OnBatch = OnLogBatch;

            _logDelegate = OnHeliumLog;
            HeliumNative.He_SetLogCallback(_logDelegate);

            NeonLogger.Log("system", "Application started");

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
                    sb.AppendLine(item.ToString());

                Clipboard.SetText(sb.ToString());
                e.Handled = true;
            }
        }

        private void OnRendering(object? sender, EventArgs e)
        {
            try { HeliumNative.He_Render(); }
            catch { }
        }

        protected override void OnClosed(EventArgs e)
        {
            CompositionTarget.Rendering -= OnRendering;
            base.OnClosed(e);
        }

        private void LogToggle_Click(object sender, RoutedEventArgs e)
        {
            LogRow.Height = LogToggle.IsChecked == true
                ? new GridLength(LogHeight)
                : new GridLength(0);

            LogToggle.Content = LogToggle.IsChecked == true ? "▼ Log" : "▲ Log";
        }
    }
}
