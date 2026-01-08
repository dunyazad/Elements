using System;
using System.Globalization;
using System.Windows;

namespace Neon
{
    public partial class SorParameterDialog : Window
    {
        private static int _lastK = 50;
        private static float _lastMul = 3.0f;
        private static bool _lastIsGradient = true;

        // 인자 3개: (K, StdDev, IsGradient)
        public Action<int, float, bool>? ApplyAction { get; set; }

        public SorParameterDialog()
        {
            InitializeComponent();

            SliderK.Value = _lastK;
            SliderStdDev.Value = _lastMul;

            if (_lastIsGradient)
                RadioGradient.IsChecked = true;
            else
                RadioBinary.IsChecked = true;
        }

        private void Apply_Click(object sender, RoutedEventArgs e)
        {
            if (int.TryParse(TxtKNeighbors.Text, out int k) &&
                float.TryParse(TxtStdDevMul.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out float mul))
            {
                bool isGradient = RadioGradient.IsChecked == true;

                _lastK = k;
                _lastMul = mul;
                _lastIsGradient = isGradient;

                // 3개 인자 전달
                ApplyAction?.Invoke(k, mul, isGradient);
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