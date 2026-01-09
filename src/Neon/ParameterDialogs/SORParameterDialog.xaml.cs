using System;
using System.Windows;

namespace Neon
{
    public partial class SorParameterDialog : Window
    {
        public PointCloudProcessorParametersSOR Parameters { get; set; } = new PointCloudProcessorParametersSOR();

        public Action<PointCloudProcessorParametersSOR>? AnalyzeAction { get; set; }

        public SorParameterDialog()
        {
            InitializeComponent();

            Loaded += (s, e) => InitializeUI();
        }

        private void InitializeUI()
        {
            CmbColorMode.ItemsSource = Enum.GetValues(typeof(PointCloudVisualizationMode));
            CmbColorMode.SelectedItem = Parameters.visualizationMode;

            if (SliderK != null)
            {
                SliderK.Value = Parameters.KNeighbors;
            }

            if (SliderStdDev != null)
            {
                SliderStdDev.Value = Parameters.StdDevMulThresh;
            }
        }

        private void Apply_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                Parameters.KNeighbors = (int)SliderK.Value;
                Parameters.StdDevMulThresh = (float)SliderStdDev.Value;

                if (CmbColorMode.SelectedItem is PointCloudVisualizationMode mode)
                {
                    Parameters.visualizationMode = mode;
                }

                AnalyzeAction?.Invoke(Parameters);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"오류가 발생했습니다: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
    }
}