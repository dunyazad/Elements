using System;
using System.Windows;
using System.Windows.Input;

namespace Neon
{
    public partial class RorParameterDialog : Window
    {
        public PointCloudProcessorParametersROR Parameters { get; set; } = new PointCloudProcessorParametersROR();

        public Action<PointCloudProcessorParametersROR>? AnalyzeAction { get; set; }

        public RorParameterDialog()
        {
            InitializeComponent();

            Loaded += (s, e) => InitializeUI();
        }

        private void InitializeUI()
        {
            CmbColorMode.ItemsSource = Enum.GetValues(typeof(PointCloudVisualizationMode));
            CmbColorMode.SelectedItem = Parameters.VisualizationMode;

            if (SliderRadius != null)
            {
                SliderRadius.Value = Parameters.Radius;
            }

            if (SliderMinNeighbors != null)
            {
                SliderMinNeighbors.Value = Parameters.MinNeighborsInRadius;
            }
        }

        private void Apply_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                Parameters.Radius = (float)SliderRadius.Value;
                Parameters.MinNeighborsInRadius = (int)SliderMinNeighbors.Value;

                if (CmbColorMode.SelectedItem is PointCloudVisualizationMode mode)
                {
                    Parameters.VisualizationMode = mode;
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
