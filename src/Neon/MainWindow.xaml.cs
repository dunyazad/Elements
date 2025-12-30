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
            HeliumNative.Helium_SetLogCallback(_logDelegate);

            NeonLogger.Log("system", "Application started");

            CompositionTarget.Rendering += OnRendering;
        }

        private readonly Dictionary<string, int> _keyToIndex = new();

        private void OnLogBatch(IReadOnlyList<(string? key, string line)> batch)
        {
            foreach (var (key, line) in batch)
            {
                if (key == null)
                {
                    LogList.Items.Add(line);
                }
                else if (_keyToIndex.TryGetValue(key, out int index))
                {
                    LogList.Items[index] = line;
                }
                else
                {
                    int newIndex = LogList.Items.Count;
                    LogList.Items.Add(line);
                    _keyToIndex[key] = newIndex;
                }

                if (LogList.Items.Count > MaxLogLines)
                    LogList.Items.RemoveAt(0);
            }

            if (_autoScroll && LogList.Items.Count > 0)
                LogList.ScrollIntoView(LogList.Items[^1]);
        }

        private void OnHeliumLog(string key, string value)
        {
            NeonLogger.Log(key, value);
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
            try { HeliumNative.Helium_Render(); }
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
