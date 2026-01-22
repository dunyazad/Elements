using Neon.Controls;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;

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

        private HeliumNative.HeliumLogDelegate _logDelegate;
        private bool _autoScroll = true;

        private HeliumNative.NativeToManagedDelegate _nativeToManagedDelegate;
        private HeliumNative.ManagedToNativeDelegate _managedToNativeDelegate;

        private readonly Dictionary<(NeonLogger.LogLevel level, string key), int> _keyToIndex = new();

        private double _lastLogHeight = 224.0;
        private const double MinLogHeight = 30.0;

        private SceneNode? selectedSceneNode = null;
        private int selectedPointCloudID = -1;
        private int selectedPointIndex = -1;

        public ObservableCollection<NotificationItem> Notifications { get; set; } = new ObservableCollection<NotificationItem>();

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

        // Dialogs
        private SorParameterDialog? _sorDialog = null;
        private RorParameterDialog? _rorDialog = null;
        private CurvatureAnalysisParameterDialog? _curvatureDialog = null;
        private NormalDeviationAnalysisParameterDialog? _normalDeviationAnalysisParameterDialog = null;
        private PFORParameterDialog? _pforDialog = null;
        private CompositeFilterDialog? _compositeFilterDialog = null;
        private NodeEditorDialog? _nodeEditorDialog = null;

        // Shared Json Options
        private readonly JsonSerializerOptions _jsonOptions;

        public MainWindow()
        {
            InitializeComponent();
            this.DataContext = this;

            SceneItems = new ObservableCollection<SceneNode>();

            // Initialize JSON Options once
            _jsonOptions = new JsonSerializerOptions();
            _jsonOptions.Converters.Add(new FloatJsonConverter());
            _jsonOptions.Converters.Add(new DoubleJsonConverter());

            ComponentDispatcher.ThreadFilterMessage += ComponentDispatcher_ThreadFilterMessage;

            NeonLogger.Initialize();
            NeonLogger.OnBatch = OnLogBatch;

            _logDelegate = OnHeliumLog;
            HeliumNative.He_SetLogCallback(_logDelegate);

            _nativeToManagedDelegate = OnNativeToManagedMessage;
            HeliumNative.He_SetNativeToManagedCallback(_nativeToManagedDelegate);

            _managedToNativeDelegate = OnManagedToNativeMessage;
            HeliumNative.He_SetManagedToNativeCallback(_managedToNativeDelegate);

            _uiStopwatch.Start();
            CompositionTarget.Rendering += OnUpdateUI;

            this.WindowState = WindowState.Maximized;

            // Auto load for testing
            Menu_File_Open_Click(this, new RoutedEventArgs());
        }

        private void OnUpdateUI(object? sender, EventArgs e)
        {
            // 1. Calculate Delta Time & FPS
            long currentTicks = _uiStopwatch.ElapsedTicks;
            double dt = (double)(currentTicks - _lastUiTicks) / Stopwatch.Frequency;
            _lastUiTicks = currentTicks;

            double fps = (dt > 0.0) ? (1.0 / dt) : 60.0;

            // 2. Get Selected Point Cloud Info
            string selectedName = "None";
            int pointCount = 0;
            int clusterCount = 0;

            if (selectedSceneNode != null)
            {
                selectedName = selectedSceneNode.Name;
                // [TODO] Retrieve actual point count from engine using ID
                pointCount = 1024000;
            }

            // 3. Update Status Text
            HeliumStatusText = $"FPS: {fps:0}\n" +
                               $"Points: {pointCount:N0}\n" +
                               $"Object: {selectedName}\n" +
                               $"Clusters: {clusterCount}\n" +
                               $"Mode: Edit";
        }

        public async void ShowNotification(string message, int durationMS = 3000)
        {
            await Application.Current.Dispatcher.InvokeAsync(async () =>
            {
                var item = new NotificationItem { Text = message };
                Notifications.Add(item);
                await Task.Delay(durationMS);
                Notifications.Remove(item);
            });
        }

        #region System Message Processing
        private void Window_PreviewKeyDown(object sender, KeyEventArgs e)
        {
            if ((Keyboard.Modifiers & ModifierKeys.Control) == ModifierKeys.Control)
            {
                switch (e.Key)
                {
                    case Key.O:
                        Menu_File_Open_Click(sender, e);
                        e.Handled = true;
                        break;
                }
            }
            else
            {
                switch (e.Key)
                {
                    case Key.T:
                        ShowNotification($"Event Triggered at {DateTime.Now:ss.fff}");
                        break;
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

                    finalLParam = (IntPtr)((cursorPos.Y << 16) | (cursorPos.X & 0xFFFF));
                }

                HeliumNative.He_ProcessMessage((uint)msg.message, msg.wParam, finalLParam);
            }
        }

        private bool IsMouseOverHeliumHost()
        {
            if (HeliumHostControl == null || !HeliumHostControl.IsLoaded) return false;

            System.Windows.Point pos = Mouse.GetPosition(HeliumHostControl);

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
                    Command = "LoadPointCloudFromPLY",
                    FileNames = filenames
                };

                string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                HeliumNative.He_ManagedToNative(command);

                for (int i = 0; i < filenames.Length; i++)
                {
                    ShowNotification($"Loading Point Cloud: {filenames[i]}");
                }
            }
        }

        private void Menu_File_Exit_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        private void Menu_Point_ShowPointNormal_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    Command = "ShowPointNormal",
                    PointCloudID = selectedSceneNode.ID,
                    PointIndex = selectedPointIndex
                };
                string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                HeliumNative.He_ManagedToNative(command);
            }
        }

        private void Menu_Point_ShowPointNormalDeviation_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    Command = "ShowPointNormalDeviation",
                    PointCloudID = selectedSceneNode.ID,
                    PointIndex = selectedPointIndex
                };
                string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                HeliumNative.He_ManagedToNative(command);
            }
        }

        private void Menu_Point_ApplyPointPlaneFitting_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    Command = "ApplyPointPlaneFitting",
                    PointCloudID = selectedSceneNode.ID,
                    PointIndex = selectedPointIndex,
                    KNeighbors = 30,
                    DistanceThreshold = 0.07f,
                };
                string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                HeliumNative.He_ManagedToNative(command);
            }
        }

        private void Menu_PointCloud_Clone_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    Command = "ClonePointCloud",
                    PointCloudID = selectedSceneNode.ID
                };
                string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                HeliumNative.He_ManagedToNative(command);
            }
        }

        private void Menu_PointCloud_ShowNormals_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    Command = "ShowPointCloudNormals",
                    PointCloudID = selectedSceneNode.ID
                };
                string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                HeliumNative.He_ManagedToNative(command);
            }
        }

        private void Menu_PointCloud_ShowSparseGrid_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    Command = "ShowSparseGrid",
                    PointCloudID = selectedSceneNode.ID
                };
                string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                HeliumNative.He_ManagedToNative(command);
            }
        }

        private void Menu_PointCloud_ShowSparseDataBlocks_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    Command = "ShowSparseDataBlocks",
                    PointCloudID = selectedSceneNode.ID
                };
                string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                HeliumNative.He_ManagedToNative(command);
            }
        }

        private void Menu_PointCloud_Clustering_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    Command = "PerformClustering",
                    PointCloudID = selectedSceneNode.ID,
                    SearchRadius = 0.15f,
                    AngleThreshold = 0.9f
                };
                string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                HeliumNative.He_ManagedToNative(command);
            }
        }

        private void Menu_PointCloud_SOR_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                if (_sorDialog != null)
                {
                    _sorDialog.Activate();
                    return;
                }

                _sorDialog = new SorParameterDialog();
                _sorDialog.Owner = this;
                _sorDialog.WindowStartupLocation = WindowStartupLocation.Manual;

                double dialogWidth = 380;
                double estimatedHeight = 300;
                _sorDialog.Left = (this.Left + this.ActualWidth) - dialogWidth - 20;
                _sorDialog.Top = (this.Top + (this.ActualHeight / 2)) - (estimatedHeight / 2);

                if (_sorDialog.Parameters is PointCloudProcessorParametersSOR sorParams)
                {
                    sorParams.PointCloudID = selectedSceneNode.ID;
                }

                _sorDialog.AnalyzeAction = (parameters) =>
                {
                    var commandData = new
                    {
                        Command = "PerformSOR",
                        PointCloudID = parameters.PointCloudID,
                        KNeighbors = parameters.KNeighbors,
                        StdDevMulThresh = parameters.StdDevMulThresh,
                        DeletePoints = parameters.DeletePoints,
                        VisualizationMode = parameters.VisualizationMode
                    };

                    string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                    HeliumNative.He_ManagedToNative(command);

                    ShowNotification($"Applied SOR (K={parameters.KNeighbors}, StdDev={parameters.StdDevMulThresh}, Vis={parameters.VisualizationMode})");
                };

                _sorDialog.Closed += (s, args) => { _sorDialog = null; };
                _sorDialog.Show();
            }
            else
            {
                ShowNotification("No Point Cloud Selected.");
            }
        }

        private void Menu_PointCloud_ROR_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                if (_rorDialog != null)
                {
                    _rorDialog.Activate();
                    return;
                }

                _rorDialog = new RorParameterDialog();
                _rorDialog.Owner = this;
                _rorDialog.WindowStartupLocation = WindowStartupLocation.Manual;

                double dialogWidth = 380;
                double estimatedHeight = 300;
                _rorDialog.Left = (this.Left + this.ActualWidth) - dialogWidth - 20;
                _rorDialog.Top = (this.Top + (this.ActualHeight / 2)) - (estimatedHeight / 2);

                if (_rorDialog.Parameters is PointCloudProcessorParametersROR rorParams)
                {
                    rorParams.PointCloudID = selectedSceneNode.ID;
                }

                _rorDialog.AnalyzeAction = (parameters) =>
                {
                    var commandData = new
                    {
                        Command = "PerformROR",
                        PointCloudID = parameters.PointCloudID,
                        Radius = parameters.Radius,
                        MinNeighborsInRadius = parameters.MinNeighborsInRadius,
                        DeletePoints = parameters.DeletePoints,
                        VisualizationMode = parameters.VisualizationMode
                    };

                    string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                    HeliumNative.He_ManagedToNative(command);

                    ShowNotification($"Applied ROR (R={parameters.Radius}, MinN={parameters.MinNeighborsInRadius}, Vis={parameters.VisualizationMode})");
                };

                _rorDialog.Closed += (s, args) => { _rorDialog = null; };
                _rorDialog.Show();
            }
            else
            {
                ShowNotification("No Point Cloud Selected.");
            }
        }

        private void Menu_PointCloud_Curvature_Analysis_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                if (_curvatureDialog != null)
                {
                    _curvatureDialog.Activate();
                    return;
                }

                _curvatureDialog = new CurvatureAnalysisParameterDialog();
                _curvatureDialog.Owner = this;
                _curvatureDialog.WindowStartupLocation = WindowStartupLocation.Manual;

                double dialogWidth = 380;
                double estimatedHeight = 300;
                _curvatureDialog.Left = (this.Left + this.ActualWidth) - dialogWidth - 20;
                _curvatureDialog.Top = (this.Top + (this.ActualHeight / 2)) - (estimatedHeight / 2);

                if (_curvatureDialog.Parameters is PointCloudProcessorParametersCurvatureAnalysis curvParams)
                {
                    curvParams.PointCloudID = selectedSceneNode.ID;
                }

                _curvatureDialog.AnalyzeAction = (parameters) =>
                {
                    var commandData = new
                    {
                        Command = "PerformCurvatureAnalysis",
                        PointCloudID = parameters.PointCloudID,
                        KNeighbors = parameters.KNeighbors,
                        CurvatureThreshold = parameters.CurvatureThreshold,
                        DeletePoints = parameters.DeletePoints,
                        VisualizationMode = parameters.VisualizationMode
                    };

                    string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                    HeliumNative.He_ManagedToNative(command);

                    ShowNotification($"Applied Curvature Analysis (ID: {parameters.PointCloudID}, K={parameters.KNeighbors}, Vis={parameters.VisualizationMode})");
                };

                _curvatureDialog.Closed += (s, args) => { _curvatureDialog = null; };
                _curvatureDialog.Show();
            }
            else
            {
                ShowNotification("No Point Cloud Selected.");
            }
        }

        private void Menu_PointCloud_Normal_Deviation_Analysis_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                if (_normalDeviationAnalysisParameterDialog != null)
                {
                    _normalDeviationAnalysisParameterDialog.Activate();
                    return;
                }

                _normalDeviationAnalysisParameterDialog = new NormalDeviationAnalysisParameterDialog();
                _normalDeviationAnalysisParameterDialog.Owner = this;

                double dialogWidth = 380;
                double estimatedHeight = 300;
                _normalDeviationAnalysisParameterDialog.Left = (this.Left + this.ActualWidth) - dialogWidth - 20;
                _normalDeviationAnalysisParameterDialog.Top = (this.Top + (this.ActualHeight / 2)) - (estimatedHeight / 2);

                if (_normalDeviationAnalysisParameterDialog.Parameters is PointCloudProcessorParametersNormalDeviation normParams)
                {
                    normParams.PointCloudID = selectedSceneNode.ID;
                }

                _normalDeviationAnalysisParameterDialog.AnalyzeAction = (parameters) =>
                {
                    var commandData = new
                    {
                        Command = "PerformNormalDeviationAnalysis",
                        PointCloudID = parameters.PointCloudID,
                        Radius = parameters.Radius,
                        DeviationThreshold = parameters.DeviationThreshold,
                        VisualizationMode = parameters.VisualizationMode
                    };

                    string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                    HeliumNative.He_ManagedToNative(command);

                    ShowNotification($"Applied Normal Deviation Analysis (ID: {parameters.PointCloudID}, R={parameters.Radius}, Vis={parameters.VisualizationMode})");
                };

                _normalDeviationAnalysisParameterDialog.Closed += (s, args) => { _normalDeviationAnalysisParameterDialog = null; };
                _normalDeviationAnalysisParameterDialog.Show();
            }
            else
            {
                ShowNotification("No Point Cloud Selected.");
            }
        }


        private void Menu_PointCloud_PFOR_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                if (_pforDialog != null)
                {
                    _pforDialog.Activate();
                    return;
                }

                _pforDialog = new PFORParameterDialog();
                _pforDialog.Owner = this;
                _pforDialog.WindowStartupLocation = WindowStartupLocation.Manual;

                double dialogWidth = 380;
                double estimatedHeight = 300;
                _pforDialog.Left = (this.Left + this.ActualWidth) - dialogWidth - 20;
                _pforDialog.Top = (this.Top + (this.ActualHeight / 2)) - (estimatedHeight / 2);

                if (_pforDialog.Parameters is PointCloudProcessorParametersPlaneFitOutlierRemoval pforParams)
                {
                    pforParams.PointCloudID = selectedSceneNode.ID;
                }

                _pforDialog.AnalyzeAction = (parameters) =>
                {
                    var commandData = new
                    {
                        Command = "PerformPFOR",
                        PointCloudID = parameters.PointCloudID,
                        KNeighbors = parameters.KNeighbors,
                        DistanceThreshold = parameters.DistanceThreshold,
                        DeletePoints = parameters.DeletePoints,
                        VisualizationMode = parameters.VisualizationMode
                    };
                    string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                    HeliumNative.He_ManagedToNative(command);
                    ShowNotification($"Applied PFOR (K={parameters.KNeighbors}, Dist={parameters.DistanceThreshold}, Vis={parameters.VisualizationMode})");
                };

                _pforDialog.Closed += (s, args) => { _pforDialog = null; };
                _pforDialog.Show();
            }
            else
            {
                ShowNotification("No Point Cloud Selected.");
            }
        }

        private void Menu_PointCloud_Composite_Filter_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                if (_compositeFilterDialog != null)
                {
                    _compositeFilterDialog.Activate();
                    return;
                }

                _compositeFilterDialog = new CompositeFilterDialog();
                _compositeFilterDialog.Owner = this;
                _compositeFilterDialog.WindowStartupLocation = WindowStartupLocation.CenterOwner;

                _compositeFilterDialog.ApplyAction = (vm) =>
                {
                    var steps = vm.Steps;
                    var pipelineList = new List<object>();

                    foreach (var step in steps)
                    {
                        pipelineList.Add(new
                        {
                            Operation = step.SelectedOperation,
                            FilterType = step.SelectedType,
                            // Cast to object to ensure derived class properties are serialized
                            Parameters = (object)step.Parameters
                        });
                    }

                    var commandData = new
                    {
                        Command = "PerformCompositeFilter",
                        PointCloudID = selectedSceneNode.ID,
                        Pipeline = pipelineList
                    };

                    string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                    HeliumNative.He_ManagedToNative(command);

                    ShowNotification($"Applied Composite Filter ({pipelineList.Count} steps)");
                };

                _compositeFilterDialog.Closed += (s, args) => { _compositeFilterDialog = null; };
                _compositeFilterDialog.Show();
            }
            else
            {
                ShowNotification("No Point Cloud Selected.");
            }
        }

        private void Menu_PointCloud_Node_Editor_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                if (_nodeEditorDialog != null)
                {
                    _nodeEditorDialog.Activate();
                    return;
                }

                _nodeEditorDialog = new NodeEditorDialog();
                _nodeEditorDialog.Owner = this;
                _nodeEditorDialog.WindowStartupLocation = WindowStartupLocation.CenterOwner;

                _nodeEditorDialog.ApplyAction = (radius, temp, isGradient) =>
                {
                    var commandData = new
                    {
                        Command = "PerformNodeEditorResult",
                        PointCloudID = selectedSceneNode.ID,
                        Radius = radius,
                        VisualizationMode = isGradient ? "Gradient" : "Binary"
                    };
                    string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                    HeliumNative.He_ManagedToNative(command);

                    ShowNotification($"Node Editor Result (R={radius}, Vis={(isGradient ? "Gradient" : "Binary")})");
                };

                _nodeEditorDialog.Closed += (s, args) => { _nodeEditorDialog = null; };
                _nodeEditorDialog.Show();
            }
        }

        private void Menu_PointCloud_Generate_Mesh_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    Command = "GenerateMesh",
                    PointCloudID = selectedSceneNode.ID
                };
                string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                HeliumNative.He_ManagedToNative(command);
            }
            else
            {
                ShowNotification("No Point Cloud Selected.");
            }
        }

        private void Menu_PointCloud_KDE_Click(object sender, RoutedEventArgs e)
        {
            if (selectedSceneNode != null)
            {
                var commandData = new
                {
                    Command = "PerformKDE",
                    PointCloudID = selectedSceneNode.ID,
                    Bandwidth = 0.1f,
                    SearchRadius = 0.5f,
                    DeletePoints = false,
                    VisualizationMode = PointCloudVisualizationMode.Gradient
                };
                string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                HeliumNative.He_ManagedToNative(command);
            }
            else
            {
                ShowNotification("No Point Cloud Selected.");
            }
        }

        private void Menu_VD_ClearAll_Click(object sender, RoutedEventArgs e)
        {
            var commandData = new { Command = "ClearAllVisualDebugging" };
            string command = JsonSerializer.Serialize(commandData, _jsonOptions);
            HeliumNative.He_ManagedToNative(command);
        }

        private void Menu_View_ToggleGrid_Click(object sender, RoutedEventArgs e)
        {
            var commandData = new { Command = "ToggleGrid" };
            string command = JsonSerializer.Serialize(commandData, _jsonOptions);
            HeliumNative.He_ManagedToNative(command);
        }

        private void Menu_View_ToggleAxisGizmo_Click(object sender, RoutedEventArgs e)
        {
            var commandData = new { Command = "ToggleAxisGizmo" };
            string command = JsonSerializer.Serialize(commandData, _jsonOptions);
            HeliumNative.He_ManagedToNative(command);
        }

        private void Menu_View_ToggleCenterGizmo_Click(object sender, RoutedEventArgs e)
        {
            var commandData = new { Command = "ToggleCenterGizmo" };
            string command = JsonSerializer.Serialize(commandData, _jsonOptions);
            HeliumNative.He_ManagedToNative(command);
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

        private void SceneTree_SelectedItemChanged(object sender, RoutedPropertyChangedEventArgs<object> e)
        {
            if (e.NewValue is SceneNode sceneNode)
            {
                var commandData = new
                {
                    Command = "SelectPointCloud",
                    PointCloudID = sceneNode.ID
                };

                string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                HeliumNative.He_ManagedToNative(command);

                selectedSceneNode = sceneNode;
                selectedPointIndex = sceneNode.ID; // Assuming 1:1 mapping for test
            }
        }

        private void SceneItemCheckBox_Click(object sender, RoutedEventArgs e)
        {
            if (sender is CheckBox checkBox && checkBox.DataContext is SceneNode sceneNode)
            {
                bool isVisible = checkBox.IsChecked ?? false;
                sceneNode.IsVisible = isVisible;

                var commandData = new
                {
                    Command = "SetPointCloudVisibility",
                    PointCloudID = sceneNode.ID,
                    IsVisible = isVisible
                };
                var command = JsonSerializer.Serialize(commandData, _jsonOptions);
                HeliumNative.He_ManagedToNative(command);

                // Auto select on visibility change logic
                selectedSceneNode = sceneNode;
                selectedPointCloudID = sceneNode.ID;

                Application.Current.Dispatcher.InvokeAsync(() =>
                {
                    var selectCmdData = new
                    {
                        Command = "SelectPointCloud",
                        PointCloudID = sceneNode.ID
                    };
                    string selectCmd = JsonSerializer.Serialize(selectCmdData, _jsonOptions);
                    HeliumNative.He_ManagedToNative(selectCmd);
                });
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
                        Command = "DeletePointCloud",
                        PointCloudID = sceneNode.ID
                    };
                    string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                    HeliumNative.He_ManagedToNative(command);
                }
                e.Handled = true;
            }
        }

        private void SceneTree_TogglePointClouds(int order, bool exclusive)
        {
            if (exclusive)
            {
                foreach (var sceneItem in SceneItems)
                {
                    sceneItem.IsVisible = false;
                }
            }

            if (order < SceneItems.Count)
            {
                SceneNode sceneNode = SceneItems[order];
                if (sceneNode != null)
                {
                    sceneNode.IsVisible = !sceneNode.IsVisible;
                    if (exclusive)
                    {
                        selectedSceneNode = sceneNode;
                        selectedPointCloudID = sceneNode.ID;

                        Application.Current.Dispatcher.InvokeAsync(() =>
                        {
                            var commandData = new
                            {
                                Command = "SelectPointCloud",
                                PointCloudID = sceneNode.ID
                            };
                            string command = JsonSerializer.Serialize(commandData, _jsonOptions);
                            HeliumNative.He_ManagedToNative(command);
                        });
                    }
                }
            }

            var pointCloudVisibleInfoList = SceneItems.Select(item => new { pointCloudID = item.ID, visible = item.IsVisible }).ToList();

            var finalCommandData = new
            {
                Command = "TogglePointClouds",
                pointCloudVisibleInfoList = pointCloudVisibleInfoList
            };
            var finalCommand = JsonSerializer.Serialize(finalCommandData, _jsonOptions);
            HeliumNative.He_ManagedToNative(finalCommand);
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
        #endregion

        private void OnNativeToManagedMessage(string jsonString)
        {
            NeonLogger.Log("NativeToManaged", jsonString);

            try
            {
                using (JsonDocument doc = JsonDocument.Parse(jsonString))
                {
                    JsonElement root = doc.RootElement;
                    if (root.TryGetProperty("EventType", out JsonElement eventElement))
                    {
                        string eventType = eventElement.GetString() ?? string.Empty;
                        switch (eventType)
                        {
                            case "Notification":
                                {
                                    if (root.TryGetProperty("Parameters", out JsonElement paramsElement))
                                    {
                                        string message = paramsElement.GetProperty("Message").GetString() ?? string.Empty;
                                        int durationMS = paramsElement.GetProperty("DurationMS").GetInt32();
                                        ShowNotification(message, durationMS);
                                    }
                                    break;
                                }

                            case "TogglePointCloud":
                                {
                                    if (root.TryGetProperty("Parameters", out JsonElement paramsElement))
                                    {
                                        int order = paramsElement.GetProperty("Order").GetInt32();
                                        bool exclusive = paramsElement.GetProperty("Exclusive").GetBoolean();
                                        SceneTree_TogglePointClouds(order, exclusive);
                                    }
                                    break;
                                }

                            case "PointCloudLoaded":
                                {
                                    if (root.TryGetProperty("Parameters", out JsonElement paramsElement))
                                    {
                                        int pointCloudID = paramsElement.GetProperty("PointCloudID").GetInt32();
                                        string fileName = paramsElement.GetProperty("FileName").GetString() ?? string.Empty;
                                        string name = paramsElement.GetProperty("Name").GetString() ?? string.Empty;
                                        OnPointCloudLoaded(pointCloudID, fileName, name);
                                    }
                                    break;
                                }
                            case "PointCloudCloned":
                                {
                                    if (root.TryGetProperty("Parameters", out JsonElement paramsElement))
                                    {
                                        int pointCloudID = paramsElement.GetProperty("PointCloudID").GetInt32();
                                        string fileName = paramsElement.GetProperty("FileName").GetString() ?? string.Empty;
                                        string name = paramsElement.GetProperty("Name").GetString() ?? string.Empty;
                                        OnPointCloudCloned(pointCloudID, fileName, name);
                                    }
                                    break;
                                }
                            case "PointCloudDeleted":
                                {
                                    if (root.TryGetProperty("Parameters", out JsonElement paramsElement))
                                    {
                                        int pointCloudID = paramsElement.GetProperty("PointCloudID").GetInt32();
                                        OnPointCloudDeleted(pointCloudID);
                                    }
                                    break;
                                }
                            case "PointSelected":
                                {
                                    if (root.TryGetProperty("Parameters", out JsonElement paramsElement))
                                    {
                                        int pointCloudID = paramsElement.GetProperty("PointCloudID").GetInt32();
                                        int pointIndex = paramsElement.GetProperty("PointIndex").GetInt32();
                                        OnPointSelected(pointCloudID, pointIndex);
                                    }
                                    break;
                                }
                            default:
                                NeonLogger.Log("NativeToManaged", $"Unknown event type: {eventType}");
                                break;
                        }
                    }
                }
            }
            catch (JsonException ex)
            {
                NeonLogger.Log(NeonLogger.LogLevel.Error, "NativeToManaged", $"JSON Parsing Error: {ex.Message}");
            }
        }

        private void OnManagedToNativeMessage(string jsonString)
        {
            NeonLogger.Log("ManagedToNative", jsonString);
        }

        private void OnPointCloudLoaded(int pointCloudID, string fileName, string name)
        {
            Application.Current.Dispatcher.InvokeAsync(() =>
            {
                var node = new SceneNode
                {
                    ID = pointCloudID,
                    Name = name,
                    FullPath = fileName
                };

                if (!SceneItems.Any(item => item.ID == pointCloudID))
                {
                    SceneItems.Add(node);
                }

                // Select the new node
                node.IsSelected = true;
                selectedPointCloudID = pointCloudID;

                ShowNotification($"Point Cloud Loaded: {name} (ID: {pointCloudID})");
            });
        }

        private void OnPointCloudCloned(int pointCloudID, string fileName, string name)
        {
            Application.Current.Dispatcher.InvokeAsync(() =>
            {
                var node = new SceneNode
                {
                    ID = pointCloudID,
                    Name = name,
                    FullPath = fileName
                };

                if (!SceneItems.Any(item => item.ID == pointCloudID))
                {
                    SceneItems.Add(node);
                }

                node.IsSelected = true;
                selectedPointCloudID = pointCloudID;

                ShowNotification($"Point Cloud Cloned: {name} (ID: {pointCloudID})");
            });
        }

        private void OnPointCloudDeleted(int pointCloudID)
        {
            Application.Current.Dispatcher.InvokeAsync(() =>
            {
                var nodeToRemove = SceneItems.FirstOrDefault(item => item.ID == pointCloudID);
                if (nodeToRemove != null)
                {
                    string name = nodeToRemove.Name;
                    if (selectedSceneNode == nodeToRemove)
                    {
                        selectedSceneNode = null!;
                    }

                    SceneItems.Remove(nodeToRemove);
                    ShowNotification($"Point Cloud Deleted: {name} (ID: {pointCloudID})");
                }
            });
        }

        private void OnPointSelected(int pointCloudID, int pointIndex)
        {
            Application.Current.Dispatcher.InvokeAsync(() =>
            {
                selectedPointCloudID = pointCloudID;
                selectedPointIndex = pointIndex;
            });
        }
    }
}
