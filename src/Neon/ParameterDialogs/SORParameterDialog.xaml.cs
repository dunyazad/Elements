using System.Windows;

namespace Neon
{
    public partial class SorParameterDialog : Window
    {
        public Action<int, float>? ApplyAction { get; set; }

        public SorParameterDialog()
        {
            InitializeComponent();
        }

        private void Apply_Click(object sender, RoutedEventArgs e)
        {
            if (int.TryParse(TxtKNeighbors.Text, out int k) && float.TryParse(TxtStdDevMul.Text, out float mul))
            {
                ApplyAction?.Invoke(k, mul);
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
