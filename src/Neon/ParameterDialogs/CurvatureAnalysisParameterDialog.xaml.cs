using System;
using System.Globalization;
using System.Windows;

namespace Neon
{
    public partial class CurvatureAnalysisParameterDialog : Window
    {
        private static int _lastK = 30;
        private static float _lastThreshold = 1.0f; // [변경] Scale -> Threshold
        private static bool _lastIsGradient = true;

        // Action: (K, Threshold, IsGradient)
        public Action<int, float, bool>? ApplyAction { get; set; }

        public CurvatureAnalysisParameterDialog()
        {
            InitializeComponent();

            SliderK.Value = _lastK;
            SliderThreshold.Value = _lastThreshold; // [변경]

            if (_lastIsGradient)
                RadioGradient.IsChecked = true;
            else
                RadioBinary.IsChecked = true;
        }

        private void Apply_Click(object sender, RoutedEventArgs e)
        {
            // [변경] TxtThreshold 참조
            if (int.TryParse(TxtK.Text, out int k) &&
                float.TryParse(TxtThreshold.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out float threshold))
            {
                bool isGradient = RadioGradient.IsChecked == true;

                _lastK = k;
                _lastThreshold = threshold;
                _lastIsGradient = isGradient;

                ApplyAction?.Invoke(k, threshold, isGradient);
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
