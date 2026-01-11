using System;
using System.Windows;
using System.Windows.Input;

namespace Neon
{
    public partial class CurvatureAnalysisParameterDialog : Window
    {
        public PointCloudProcessorParametersCurvatureAnalysis Parameters { get; set; } = new PointCloudProcessorParametersCurvatureAnalysis();

        public CurvatureAnalysisParameterDialog()
        {
            InitializeComponent();

            Loaded += (s, e) => InitializeUI();
        }

        private void InitializeUI()
        {
            CmbColorMode.ItemsSource = Enum.GetValues(typeof(PointCloudVisualizationMode));
            CmbColorMode.SelectedItem = Parameters.VisualizationMode;

            if (SliderK != null)
            {
                SliderK.Value = Parameters.KNeighbors;
            }

            if (SliderThreshold != null)
            {
                SliderThreshold.Value = Parameters.CurvatureThreshold;
            }
        }

        public Action<PointCloudProcessorParametersCurvatureAnalysis>? AnalyzeAction { get; set; }

        private void Apply_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                Parameters.KNeighbors = (int)SliderK.Value;
                Parameters.CurvatureThreshold = (float)SliderThreshold.Value;

                if (CmbColorMode.SelectedItem is PointCloudVisualizationMode mode)
                {
                    Parameters.VisualizationMode = mode;
                }

                AnalyzeAction?.Invoke(Parameters);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"설정 적용 중 오류가 발생했습니다: {ex.Message}", "오류", MessageBoxButton.OK, MessageBoxImage.Error);
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
