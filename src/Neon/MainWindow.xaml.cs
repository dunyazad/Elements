using Neon.Controls;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using static HeliumNative;

namespace Neon
{
    public class NotificationItem
    {
        public string Text { get; set; } = string.Empty;
    }

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

    public partial class MainWindow : Window, INotifyPropertyChanged
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

        private readonly Dictionary<(NeonLogger.LogLevel level, string key), int> _keyToIndex = new();

        private double _lastLogHeight = 224.0;
        private const double MinLogHeight = 30.0;

        private SceneNode? selectedSceneNode = null; // null 허용

        public ObservableCollection<NotificationItem> Notifications { get; set; }
        = new ObservableCollection<NotificationItem>();

        private readonly Stopwatch _uiStopwatch = new Stopwatch();
        private long _lastUiTicks = 0;

        private string _heliumStatusText = "Helium Engine Ready";
        public string HeliumStatusText
        {
            get => _heliumStatusText;
            set
            {
                if (_heliumStatusText != value)
                {
                    _heliumStatusText = value;
                    OnPropertyChanged();
                }
            }
        }

        public MainWindow()
        {
            InitializeComponent();
            this.DataContext = this;

            SceneItems = new ObservableCollection<SceneNode>();
            this.DataContext = this;

            ComponentDispatcher.ThreadFilterMessage += ComponentDispatcher_ThreadFilterMessage;

            NeonLogger.Initialize();
            NeonLogger.OnBatch = OnLogBatch;

            _logDelegate = OnHeliumLog;
            HeliumNative.He_SetLogCallback(_logDelegate);
            //NeonLogger.Log("system", "Application started");

            _pointCloudCreatedDelegate = Treeview_OnPointCloudCreated;
            HeliumNative.He_SetPointCloudCreatedCallback(_pointCloudCreatedDelegate);

            _pointCloudDeletedDelegate = TreeView_OnPointCloudDeleted;
            HeliumNative.He_SetPointCloudDeletedCallback(_pointCloudDeletedDelegate);

            _uiStopwatch.Start();
            CompositionTarget.Rendering += OnUpdateUI;

            this.WindowState = WindowState.Maximized;

            Menu_File_Open_Click(this, new RoutedEventArgs());
        }

        private void OnUpdateUI(object? sender, EventArgs e)
        {
            // 1. UI 스레드 델타 타임 및 FPS 계산
            long currentTicks = _uiStopwatch.ElapsedTicks;
            double dt = (double)(currentTicks - _lastUiTicks) / Stopwatch.Frequency;
            _lastUiTicks = currentTicks;

            // 0으로 나누기 방지
            double fps = (dt > 0.0) ? (1.0 / dt) : 60.0;

            // 2. 현재 선택된 클라우드 정보 가져오기
            string selectedName = "None";
            int pointCount = 0;
            int clusterCount = 0;

            if (selectedSceneNode != null)
            {
                selectedName = selectedSceneNode.Name;

                // [TODO] 실제 엔진에서 포인트 개수를 가져오는 함수가 있다면 연결하세요.
                // 예: pointCount = HeliumNative.He_GetPointCount(selectedSceneNode.ID);
                // 현재는 예시로 ID * 1000 등을 넣거나 임시 값을 넣습니다.
                pointCount = 1024000; // 가짜 데이터 (1M Points)
            }

            // 3. 상태 텍스트 갱신 (화면 왼쪽 위)
            // C# 6.0 이상 문자열 보간 ($"...") 사용
            HeliumStatusText = $"FPS: {fps:0}\n" +                    // 소수점 없이 정수로
                               $"Points: {pointCount:N0}\n" +         // 천단위 콤마 (N0)
                               $"Object: {selectedName}\n" +          // 선택된 객체 이름
                               $"Clusters: {clusterCount}\n" +        // 클러스터 개수
                               $"Mode: Edit";                         // 현재 모드 (고정값 예시)
        }

        public async void ShowNotification(string message, int durationMS = 3000)
        {
            // C++ 등 다른 스레드에서 호출될 경우를 대비해 Dispatcher 사용
            await Application.Current.Dispatcher.InvokeAsync(async () =>
            {
                var item = new NotificationItem { Text = message };

                // (1) 리스트에 추가 (화면에 즉시 표시됨)
                Notifications.Add(item);

                // (2) 지정된 시간만큼 대기 (비동기라 UI 안 멈춤)
                await Task.Delay(durationMS);

                // (3) 시간이 지나면 리스트에서 제거 (화면에서 사라짐)
                Notifications.Remove(item);
            });
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
                    case Key.T:
                        {
                            ShowNotification($"Event Triggered at {DateTime.Now:ss.fff}");
                            break;
                        }
                }
            }
        }

        private void ComponentDispatcher_ThreadFilterMessage(ref MSG msg, ref bool handled)
        {
            bool isKeyboard = (msg.message >= 0x0100 && msg.message <= 0x0105);
            bool isMouse = (msg.message >= 0x0200 && msg.message <= 0x020E);

            if (isKeyboard || isMouse)
            {
                if (HeliumHostControl == null || HeliumHostControl.Handle == IntPtr.Zero)
                    return;

                if (!IsMouseOverHeliumHost())
                {
                    return;
                }

                IntPtr heliumHwnd = HeliumHostControl.Handle;
                IntPtr finalLParam = msg.lParam;

                if (isMouse)
                {
                    NativeMethods.POINT cursorPos;
                    NativeMethods.GetCursorPos(out cursorPos);
                    NativeMethods.ScreenToClient(heliumHwnd, ref cursorPos);

                    // C++의 GET_X_LPARAM 매크로 대응을 위한 좌표 패킹
                    finalLParam = (IntPtr)((cursorPos.Y << 16) | (cursorPos.X & 0xFFFF));
                }

                // 5. 엔진 전송
                HeliumNative.He_ProcessMessage((uint)msg.message, msg.wParam, finalLParam);
            }
        }

        private bool IsMouseOverHeliumHost()
        {
            if (HeliumHostControl == null || !HeliumHostControl.IsLoaded) return false;

            // WPF 좌표계 기준 마우스 위치
            System.Windows.Point pos = Mouse.GetPosition(HeliumHostControl);

            // 컨트롤 영역 안에 있는지 확인
            return (pos.X >= 0 && pos.X < HeliumHostControl.ActualWidth &&
                    pos.Y >= 0 && pos.Y < HeliumHostControl.ActualHeight);
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

                var commandData = new
                {
                    command = "LoadPointCloudFromPLY",
                    fileNames = filenames
                };

                string command = System.Text.Json.JsonSerializer.Serialize(commandData);
                HeliumNative.He_ExecuteCommand(command);
            }
        }

        private void Menu_File_Exit_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private void Menu_PointCloud_BuildSparseGrid_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    command = "BuildSparseGrid",
                    pointCloudID = selectedSceneNode.ID
                };

                string command = System.Text.Json.JsonSerializer.Serialize(commandData);

                HeliumNative.He_ExecuteCommand(command);
            }
        }

        private void Menu_PointCloud_BuildSparseDataBlocks_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    command = "BuildSparseDataBlocks",
                    pointCloudID = selectedSceneNode.ID
                };
                string command = System.Text.Json.JsonSerializer.Serialize(commandData);
                HeliumNative.He_ExecuteCommand(command);
            }
        }

        private void Menu_PointCloud_Clone_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    command = "ClonePointCloud",
                    pointCloudID = selectedSceneNode.ID
                };
                string command = System.Text.Json.JsonSerializer.Serialize(commandData);
                HeliumNative.He_ExecuteCommand(command);
            }
        }

        private void Menu_PointCloud_Clustering_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    command = "PerformClustering",
                    pointCloudID = selectedSceneNode.ID,
                    searchRadius = 0.15f,
                    angleThreshold = 0.9f
                };
                string command = System.Text.Json.JsonSerializer.Serialize(commandData);
                HeliumNative.He_ExecuteCommand(command);
            }
        }

        private void Menu_VD_ClearAll_Click(object sender, RoutedEventArgs e)
        {
            var commandData = new
            {
                command = "ClearAllVisualDebugging"
            };
            string command = System.Text.Json.JsonSerializer.Serialize(commandData);
            HeliumNative.He_ExecuteCommand(command);
        }
        private void Menu_View_ToggleGrid_Click(object sender, RoutedEventArgs e)
        {
            var commandData = new
            {
                command = "ToggleGrid"
            };
            string command = System.Text.Json.JsonSerializer.Serialize(commandData);
            HeliumNative.He_ExecuteCommand(command);
        }

        private void Menu_View_ToggleAxisGizmo_Click(object sender, RoutedEventArgs e)
        {
            var commandData = new
            {
                command = "ToggleAxisGizmo"
            };
            string command = System.Text.Json.JsonSerializer.Serialize(commandData);
            HeliumNative.He_ExecuteCommand(command);
        }

        private void Menu_View_ToggleCenterGizmo_Click(object sender, RoutedEventArgs e)
        {
            var commandData = new
            {
                command = "ToggleCenterGizmo"
            };
            string command = System.Text.Json.JsonSerializer.Serialize(commandData);
            HeliumNative.He_ExecuteCommand(command);
        }

        private void Menu_View_ToggleLog_Click(object sender, RoutedEventArgs e)
        {
            LogToggle.IsChecked = !LogToggle.IsChecked;
        }
        #endregion

        #region TreeView
        public event PropertyChangedEventHandler? PropertyChanged;

        protected void OnPropertyChanged([CallerMemberName] string? name = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }

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
                var commandData = new
                {
                    command = "SelectPointCloud",
                    pointCloudID = sceneNode.ID
                };

                string command = System.Text.Json.JsonSerializer.Serialize(commandData);
                HeliumNative.He_ExecuteCommand(command);

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

                var commandData = new
                {
                    command = "SetPointCloudVisibility",
                    pointCloudID = sceneNode.ID,
                    isVisible = isVisible
                };
                var command = System.Text.Json.JsonSerializer.Serialize(commandData);
                HeliumNative.He_ExecuteCommand(command);

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
                    var commandData = new
                    {
                        command = "DeletePointCloud",
                        pointCloudID = sceneNode.ID
                    };
                    string command = System.Text.Json.JsonSerializer.Serialize(commandData);
                    HeliumNative.He_ExecuteCommand(command);

                    SceneItems.Remove(sceneNode);

                    if (selectedSceneNode == sceneNode)
                    {
                        selectedSceneNode = null!;
                    }
                }
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

        private void LogToggle_Checked(object sender, RoutedEventArgs e)
        {
            MainLogRow.Height = new GridLength(_lastLogHeight);
        }

        private void LogToggle_Unchecked(object sender, RoutedEventArgs e)
        {
            if (MainLogRow.Height.Value > MinLogHeight)
            {
                _lastLogHeight = MainLogRow.Height.Value;
            }
            MainLogRow.Height = new GridLength(MinLogHeight);
        }

        private void LogToggle_Click(object sender, RoutedEventArgs e)
        {
            if (LogToggle.IsChecked == true)
            {
                MainLogRow.Height = new GridLength(_lastLogHeight);
            }
            else
            {
                if (MainLogRow.Height.Value > MinLogHeight)
                {
                    _lastLogHeight = MainLogRow.Height.Value;
                }
                MainLogRow.Height = new GridLength(MinLogHeight);
            }
        }
    }
    #endregion
}
