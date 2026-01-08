using System;
using System.Globalization;
using System.Windows;

namespace Neon
{
    public partial class RorParameterDialog : Window
    {
        private static float _lastRadius = 0.3f;
        private static int _lastMinN = 24;
        private static bool _lastIsGradient = true;

        public Action<float, int, bool>? ApplyAction { get; set; }

        public RorParameterDialog()
        {
            InitializeComponent();

            SliderRadius.Value = _lastRadius;
            SliderMinNeighbors.Value = _lastMinN;

            if (_lastIsGradient)
                RadioGradient.IsChecked = true;
            else
                RadioBinary.IsChecked = true;
        }

        private void Apply_Click(object sender, RoutedEventArgs e)
        {
            if (float.TryParse(TxtRadius.Text, NumberStyles.Any, CultureInfo.InvariantCulture, out float r) &&
                int.TryParse(TxtMinNeighbors.Text, out int n))
            {
                bool isGradient = RadioGradient.IsChecked == true;

                _lastRadius = r;
                _lastMinN = n;
                _lastIsGradient = isGradient;

                ApplyAction?.Invoke(r, n, isGradient);
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