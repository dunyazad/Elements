using System;
using System.Globalization;
using System.Windows;

namespace Neon
{
    public partial class NormalDeviationAnalysisParameterDialog : Window
    {
        private static float _lastRadius = 0.1f;
        private static float _lastThreshold = 45.0f;
        private static bool _lastIsGradient = true;

        // Action: (Radius, Threshold, IsGradient)
        public Action<float, float, bool>? ApplyAction { get; set; }

        public NormalDeviationAnalysisParameterDialog()
        {
            InitializeComponent();

            SliderRadius.Value = _lastRadius;
            SliderThreshold.Value = _lastThreshold;

            if (_lastIsGradient)
                RadioGradient.IsChecked = true;
            else
                RadioBinary.IsChecked = true;
        }

        private void Apply_Click(object sender, RoutedEventArgs e)
        {
            if (float.TryParse(TxtRadius.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out float r) &&
                float.TryParse(TxtThreshold.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out float threshold))
            {
                bool isGradient = RadioGradient.IsChecked == true;

                _lastRadius = r;
                _lastThreshold = threshold;
                _lastIsGradient = isGradient;

                ApplyAction?.Invoke(r, threshold, isGradient);
            }
            else
            {
                MessageBox.Show("Invalid input values.", "Error", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}