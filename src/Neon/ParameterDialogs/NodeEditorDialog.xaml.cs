using Neon.Nodes; // 1단계에서 만든 NeonNodes.cs가 필요합니다.
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;

namespace Neon.Controls
{
    public partial class NodeEditorDialog : Window
    {
        private NetworkViewModel _viewModel;
        private PinViewModel? _draggingSourcePin = null;

        public Action<float, float, bool>? ApplyAction { get; set; }

        public NodeEditorDialog()
        {
            InitializeComponent();
            _viewModel = new NetworkViewModel();
            this.DataContext = _viewModel;

            InitializeSampleData();
        }

        private void InitializeSampleData()
        {
            // SOR Filter 노드
            var nodeSor = new NodeViewModel("SOR Filter", 50, 100);
            nodeSor.Inputs.Add(new PinViewModel { Name = "In Cloud", IsInput = true, ParentNode = nodeSor });
            nodeSor.Outputs.Add(new PinViewModel { Name = "Filtered", IsInput = false, ParentNode = nodeSor });
            nodeSor.Outputs.Add(new PinViewModel { Name = "Outliers", IsInput = false, ParentNode = nodeSor });

            // ROR Filter 노드
            var nodeRor = new NodeViewModel("ROR Filter", 350, 150);
            nodeRor.Inputs.Add(new PinViewModel { Name = "In Cloud", IsInput = true, ParentNode = nodeRor });
            nodeRor.Outputs.Add(new PinViewModel { Name = "Filtered", IsInput = false, ParentNode = nodeRor });

            // Clustering 노드
            var nodeClustering = new NodeViewModel("Euclidean Clustering", 350, 300);
            nodeClustering.Inputs.Add(new PinViewModel { Name = "In Cloud", IsInput = true, ParentNode = nodeClustering });
            nodeClustering.Outputs.Add(new PinViewModel { Name = "Clusters", IsInput = false, ParentNode = nodeClustering });

            _viewModel.Nodes.Add(nodeSor);
            _viewModel.Nodes.Add(nodeRor);
            _viewModel.Nodes.Add(nodeClustering);
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            UpdateAllPinLocations();
        }

        private void Node_DragDelta(object sender, DragDeltaEventArgs e)
        {
            if (sender is Thumb thumb && thumb.Tag is NodeViewModel node)
            {
                node.X += e.HorizontalChange;
                node.Y += e.VerticalChange;

                // 노드가 움직였으므로 핀 위치 업데이트
                UpdatePinLocationsForNode(thumb, node);
            }
        }

        private void Pin_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (sender is Ellipse ellipse && ellipse.DataContext is PinViewModel pin)
            {
                // 출력 핀(Output)에서만 연결 시작
                if (pin.IsInput == false)
                {
                    _draggingSourcePin = pin;
                    _viewModel.PendingConnectionSource = pin.Pos;
                    _viewModel.PendingConnectionTarget = pin.Pos;
                    _viewModel.IsDraggingConnection = true;

                    ellipse.CaptureMouse();
                    e.Handled = true;
                }
            }
        }

        private void Editor_MouseMove(object sender, MouseEventArgs e)
        {
            if (_viewModel.IsDraggingConnection)
            {
                // 마우스 따라다니는 점선 업데이트
                Point mousePos = e.GetPosition(NodesControl);
                _viewModel.PendingConnectionTarget = mousePos;
            }
        }

        private void Pin_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            // 드래그 중 핀 위에서 마우스를 뗐을 때 연결 처리
            if (_viewModel.IsDraggingConnection && sender is Ellipse ellipse && ellipse.DataContext is PinViewModel targetPin)
            {
                // 입력 핀이고, 같은 노드가 아닐 때만 연결
                if (targetPin.IsInput && _draggingSourcePin != null && _draggingSourcePin.ParentNode != targetPin.ParentNode)
                {
                    var newConnection = new ConnectionViewModel(_draggingSourcePin, targetPin);
                    _viewModel.Connections.Add(newConnection);
                }
            }

            FinishConnectionDrag();
            if (sender is UIElement ui) ui.ReleaseMouseCapture();
        }

        private void Editor_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            // 빈 공간에서 마우스를 떼면 연결 취소
            FinishConnectionDrag();
        }

        private void FinishConnectionDrag()
        {
            if (_viewModel.IsDraggingConnection)
            {
                _viewModel.IsDraggingConnection = false;
                _draggingSourcePin = null;
                Mouse.Capture(null);
            }
        }

        // --- 핀 위치 계산 로직 ---

        private void UpdateAllPinLocations()
        {
            for (int i = 0; i < NodesControl.Items.Count; i++)
            {
                var container = NodesControl.ItemContainerGenerator.ContainerFromIndex(i) as UIElement;
                if (container != null)
                {
                    var thumb = FindVisualChild<Thumb>(container);
                    var nodeVM = NodesControl.Items[i] as NodeViewModel;
                    if (thumb != null && nodeVM != null)
                    {
                        UpdatePinLocationsForNode(thumb, nodeVM);
                    }
                }
            }
        }

        private void UpdatePinLocationsForNode(UIElement nodeVisual, NodeViewModel nodeVM)
        {
            var pins = FindVisualChildren<Ellipse>(nodeVisual);
            foreach (var ellipse in pins)
            {
                if (ellipse.DataContext is PinViewModel pinVM)
                {
                    // 핀(Ellipse)의 중심 좌표를 Canvas 기준으로 변환
                    Point center = ellipse.TranslatePoint(new Point(ellipse.ActualWidth / 2, ellipse.ActualHeight / 2), NodesControl);
                    pinVM.Pos = center;
                }
            }
        }

        // WPF VisualTree 헬퍼 메서드
        private static T? FindVisualChild<T>(DependencyObject parent) where T : DependencyObject
        {
            for (int i = 0; i < VisualTreeHelper.GetChildrenCount(parent); i++)
            {
                var child = VisualTreeHelper.GetChild(parent, i);
                if (child is T t) return t;
                var result = FindVisualChild<T>(child);
                if (result != null) return result;
            }
            return null;
        }

        private static List<T> FindVisualChildren<T>(DependencyObject parent) where T : DependencyObject
        {
            List<T> results = new List<T>();
            for (int i = 0; i < VisualTreeHelper.GetChildrenCount(parent); i++)
            {
                var child = VisualTreeHelper.GetChild(parent, i);
                if (child is T t) results.Add(t);
                results.AddRange(FindVisualChildren<T>(child));
            }
            return results;
        }

        private void TitleBar_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Left)
            {
                this.DragMove();
            }
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
    }
}