using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace Neon
{
    // Parameter Data Class
    public class PointCloudProcessorParametersPlaneFitOutlierRemoval : PointCloudProcessorParameters
    {
        public int KNeighbors { get; set; } = 30;
        public float DistanceThreshold { get; set; } = 0.085f;
        public PointCloudVisualizationMode VisualizationMode { get; set; } = PointCloudVisualizationMode.Gradient;
        public bool DeletePoints { get; set; } = false;
    }

    // Code Behind
    public partial class PFORParameterDialog : Window
    {
        public PointCloudProcessorParametersPlaneFitOutlierRemoval Parameters { get; set; } = new PointCloudProcessorParametersPlaneFitOutlierRemoval();

        public Action<PointCloudProcessorParametersPlaneFitOutlierRemoval>? AnalyzeAction { get; set; }

        public PFORParameterDialog()
        {
            InitializeComponent();

            // SOR 예제와 동일하게 Loaded 이벤트에서 UI 초기화
            Loaded += (s, e) => InitializeUI();
        }

        private void InitializeUI()
        {
            // Initialize ComboBox
            CmbColorMode.ItemsSource = Enum.GetValues(typeof(PointCloudVisualizationMode));
            CmbColorMode.SelectedItem = Parameters.VisualizationMode;

            // Initialize Sliders
            if (SliderK != null)
            {
                SliderK.Value = Parameters.KNeighbors;
            }

            if (SliderDistance != null)
            {
                SliderDistance.Value = Parameters.DistanceThreshold;
            }
        }

        private void Apply_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // UI 값을 파라미터로 업데이트
                Parameters.KNeighbors = (int)SliderK.Value;
                Parameters.DistanceThreshold = (float)SliderDistance.Value;

                if (CmbColorMode.SelectedItem is PointCloudVisualizationMode mode)
                {
                    Parameters.VisualizationMode = mode;
                }

                // 체크박스가 있다면 여기서 업데이트
                // Parameters.DeletePoints = ...;

                AnalyzeAction?.Invoke(Parameters);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
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