using Neon.Controls;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using static HeliumNative;

namespace Neon
{
    public partial class MainWindow : Window
    {
        private const int WM_KEYDOWN = 0x0100;
        private const int WM_KEYUP = 0x0101;
        private const int WM_SYSKEYDOWN = 0x0104;
        private const int WM_SYSKEYUP = 0x0105;

        private const int WM_MOUSEMOVE = 0x0200;
        private const int WM_LBUTTONDOWN = 0x0201;
        private const int WM_LBUTTONUP = 0x0202;
        private const int WM_RBUTTONDOWN = 0x0204;
        private const int WM_RBUTTONUP = 0x0205;
        private const int WM_MBUTTONDOWN = 0x0207;
        private const int WM_MBUTTONUP = 0x0208;
        private const int WM_MOUSEWHEEL = 0x020A; // Vertical
        private const int WM_MOUSEHWHEEL = 0x020E; // Horizontal

        public ObservableCollection<SceneNode> SceneItems { get; set; }

        private HeliumLogDelegate _logDelegate;
        private bool _autoScroll = true;

        private HeliumNative.PointCloudCreatedDelegate _pointCloudCreatedDelegate;
        private HeliumNative.PointCloudDeletedDelegate _pointCloudDeletedDelegate;

        private readonly Stopwatch _stopwatch = new();
        private long _lastTicks = 0;

        private readonly Dictionary<(NeonLogger.LogLevel level, string key), int> _keyToIndex = new();

        private double _lastLogHeight = 224.0;
        private const double MinLogHeight = 30.0;
                
        public MainWindow()
        {
            InitializeComponent();

            SceneItems = new ObservableCollection<SceneNode>();
            this.DataContext = this;

            ComponentDispatcher.ThreadFilterMessage += ComponentDispatcher_ThreadFilterMessage;

            NeonLogger.Initialize();
            NeonLogger.OnBatch = OnLogBatch;

            _logDelegate = OnHeliumLog;
            HeliumNative.He_SetLogCallback(_logDelegate);
            //NeonLogger.Log("system", "Application started");

            _stopwatch.Start();
            CompositionTarget.Rendering += OnRendering;

            _pointCloudCreatedDelegate = Treeview_OnPointCloudCreated;
            HeliumNative.He_SetPointCloudCreatedCallback(_pointCloudCreatedDelegate);

            _pointCloudDeletedDelegate = TreeView_OnPointCloudDeleted;
            HeliumNative.He_SetPointCloudDeletedCallback(_pointCloudDeletedDelegate);

            //this.WindowState = WindowState.Maximized;
        }

        #region System Message Processing
        private void Window_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            // Check for Ctrl key combination
            if ((Keyboard.Modifiers & ModifierKeys.Control) == ModifierKeys.Control)
            {
                switch (e.Key)
                {
                    case Key.O:
                        {
                            Menu_File_Open_Click(sender, e);
                            e.Handled = true; // Mark event as handled so it doesn't bubble up
                            break;
                        }
                }
            }
            else
            {
                switch (e.Key)
                {
                    case Key.Escape:
                        {
                            e.Handled = true;
                            this.Close();
                            break;
                        }
                }
            }
        }

        private void ComponentDispatcher_ThreadFilterMessage(ref MSG msg, ref bool handled)
        {
            bool isKeyboard = (msg.message >= WM_KEYDOWN && msg.message <= WM_SYSKEYUP);
            bool isMouse = (msg.message >= WM_MOUSEMOVE && msg.message <= WM_MOUSEHWHEEL);

            if (isKeyboard || isMouse)
            {
                if (HeliumHostControl == null || HeliumHostControl.Handle == IntPtr.Zero)
                {
                    return;
                }

                IntPtr heliumHwnd = HeliumHostControl.Handle;

                if (isKeyboard)
                {
                    if (!HeliumHostControl.IsKeyboardFocusWithin)
                    {
                        return;
                    }
                }

                if (isMouse)
                {
                    if (!IsMouseOverHeliumHost())
                    {
                        return;
                    }
                }

                IntPtr finalLParam = msg.lParam;

                if (isMouse)
                {
                    NativeMethods.POINT cursorPos;
                    NativeMethods.GetCursorPos(out cursorPos);
                    NativeMethods.ScreenToClient(heliumHwnd, ref cursorPos);

                    int x = cursorPos.X;
                    int y = cursorPos.Y;

                    finalLParam = (IntPtr)((y << 16) | (x & 0xFFFF));
                }

                HeliumNative.He_ProcessMessage((uint)msg.message, msg.wParam, finalLParam);
            }
        }

        private bool IsMouseOverHeliumHost()
        {
            if (HeliumHostControl == null || !HeliumHostControl.IsLoaded)
                return false;

            System.Windows.Point pos = Mouse.GetPosition(HeliumHostControl);
            
            return (pos.X >= 0 && pos.X < HeliumHostControl.ActualWidth &&
                    pos.Y >= 0 && pos.Y < HeliumHostControl.ActualHeight);
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
        #endregion

        #region TitleBar
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
            ComponentDispatcher.ThreadFilterMessage -= ComponentDispatcher_ThreadFilterMessage;
            base.OnClosed(e);
        }
        #endregion

        #region MenuBar
        private void Menu_File_Open_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.OpenFileDialog();
            dialog.FileName = "";
            dialog.DefaultExt = ".ply";
            dialog.Multiselect = true;
            dialog.Filter = "Point Cloud Files (*.ply)|*.ply|All files (*.*)|*.*";

            if (dialog.ShowDialog() == true)
            {
                string[] filenames = dialog.FileNames;

                foreach (string filename in filenames)
                {
                    HeliumNative.He_LoadPointCloudFromPLY(filename, System.IO.Path.GetFileName(filename));
                }
            }
        }

        private void Menu_File_Exit_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private void Menu_PointCloud_Clone_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                HeliumNative.He_PointCloudClone(selectedSceneNode.ID);
            }
        }

        private void Menu_PointCloud_Clustering_Click(object sender, RoutedEventArgs e)
        {
        }
        #endregion

        #region TreeView
        public class SceneNode : INotifyPropertyChanged
        {
            public int ID { get; set; }
            public required string Name { get; set; } = string.Empty;
            public required string FullPath { get; set; } = string.Empty;

            // 기본값을 true로 설정하여 생성 시 체크된 상태로 만듦
            private bool _isVisible = true;

            public bool IsVisible
            {
                get => _isVisible;
                set
                {
                    if (_isVisible != value)
                    {
                        _isVisible = value;
                        OnPropertyChanged();
                    }
                }
            }

            private bool _isSelected;
            public bool IsSelected
            {
                get => _isSelected;
                set
                {
                    if (_isSelected != value)
                    {
                        _isSelected = value;
                        OnPropertyChanged();
                    }
                }
            }

            public event PropertyChangedEventHandler? PropertyChanged;

            protected void OnPropertyChanged([CallerMemberName] string? name = null)
            {
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
            }
        }

        SceneNode selectedSceneNode = null!;

        private void Treeview_OnPointCloudCreated(int ID, string fineName, string name)
        {
            // 네이티브 스레드에서 호출될 수 있으므로 Dispatcher 사용
            Application.Current.Dispatcher.Invoke(() =>
            {
                var node = new SceneNode
                {
                    ID = ID,
                    Name = name,
                    FullPath = fineName
                };

                // 중복 방지 (선택 사항)
                bool exists = false;
                foreach (var item in SceneItems)
                {
                    if (item.ID == ID)
                    {
                        exists = true;
                        break;
                    }
                }

                if (!exists)
                {
                    SceneItems.Add(node);
                    //NeonLogger.Log("System", $"PointCloud Added via Callback: {name} (ID: {id})");
                }
                node.IsSelected = true;
            });
        }

        private void TreeView_OnPointCloudDeleted(int ID)
        {
            Application.Current.Dispatcher.Invoke(() =>
            {
                SceneNode? nodeToRemove = null;
                foreach (var item in SceneItems)
                {
                    if (item.ID == ID)
                    {
                        nodeToRemove = item;
                        break;
                    }
                }
                if (nodeToRemove != null)
                {
                    SceneItems.Remove(nodeToRemove);
                    //NeonLogger.Log("System", $"PointCloud Removed via Callback: {nodeToRemove.Name} (ID: {ID})");
                }
            });
        }

        private void SceneTree_SelectedItemChanged(object sender, RoutedPropertyChangedEventArgs<object> e)
        {
            if (e.NewValue is SceneNode sceneNode)
            {
                //NeonLogger.Log("", $"Selected: {sceneNode.Name}");
                HeliumNative.He_PointCloudSelect(sceneNode.ID);

                selectedSceneNode = sceneNode;
            }
        }

        private void SceneItemCheckBox_Click(object sender, RoutedEventArgs e)
        {
            if (sender is CheckBox checkBox && checkBox.DataContext is SceneNode sceneNode)
            {
                // UI의 체크 상태 가져오기 (null일 경우 false 처리)
                bool isVisible = checkBox.IsChecked ?? false;
                sceneNode.IsVisible = isVisible;
                HeliumNative.He_PointCloudSetVisible(sceneNode.ID, isVisible);

                //NeonLogger.Log("", $"PointCloud Visibility Changed: {sceneNode.Name} (ID: {sceneNode.ID}) to {(isVisible ? "Visible" : "Hidden")}");
            }
        }

        private void SceneItemDelete_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.DataContext is SceneNode sceneNode)
            {
                var result = MessageBox.Show(
                    $"Do you want to delete '{sceneNode.Name}'?",
                    "Delete Point Cloud",
                    MessageBoxButton.YesNo,
                    MessageBoxImage.Warning
                );

                if (result == MessageBoxResult.Yes)
                {
                    HeliumNative.He_PointCloudDelete(sceneNode.ID);
                    SceneItems.Remove(sceneNode);

                    if (selectedSceneNode == sceneNode)
                    {
                        selectedSceneNode = null!;
                    }
                    //NeonLogger.Log(NeonLogger.LogLevel.Info, "System", $"Deleted PointCloud: {sceneNode.Name}");
                }

                // 3. 삭제 버튼 클릭 이벤트가 트리뷰 아이템 선택 이벤트로 전파되지 않도록 막음
                e.Handled = true;
            }
        }
        #endregion

        #region Logging
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
                    LogList.Items.Add(MakeLogItem(level, line));
                    continue;
                }

                var id = (level, key);
                if (_keyToIndex.TryGetValue(id, out int index))
                {
                    LogList.Items[index] = MakeLogItem(level, line);
                }
                else
                {
                    int newIndex = LogList.Items.Count;
                    LogList.Items.Add(MakeLogItem(level, line));
                    _keyToIndex[id] = newIndex;
                }
            }

            if (_autoScroll && LogList.Items.Count > 0)
            {
                var lastItem = LogList.Items[LogList.Items.Count - 1];
                LogList.ScrollIntoView(lastItem);
            }
        }

        private object MakeLogItem(NeonLogger.LogLevel level, string line)
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
        #endregion
    }
}
