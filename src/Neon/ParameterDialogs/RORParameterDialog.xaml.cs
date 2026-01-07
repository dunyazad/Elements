using System;
using System.Windows;

namespace Neon
{
    public partial class RorParameterDialog : Window
    {
        public Action<float, int>? ApplyAction { get; set; }

        public RorParameterDialog()
        {
            InitializeComponent();
        }

        private void Apply_Click(object sender, RoutedEventArgs e)
        {
            if (float.TryParse(TxtRadius.Text, out float r) && int.TryParse(TxtMinNeighborsInRadius.Text, out int n))
            {
                ApplyAction?.Invoke(r, n);
            }
            else
            {
                MessageBox.Show("Please enter valid numeric values.", "Invalid Input", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }
    }
}