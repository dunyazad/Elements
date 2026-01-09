using Neon.ViewModels;
using System; // Action 사용을 위해 필요
using System.Windows;

namespace Neon.Controls
{
    public partial class CompositeFilterDialog : Window
    {
        public CompositeFilterViewModel ViewModel { get; private set; }

        public Action<CompositeFilterViewModel>? ApplyAction { get; set; }

        public CompositeFilterDialog()
        {
            InitializeComponent();
            ViewModel = new CompositeFilterViewModel();
            this.DataContext = ViewModel;
        }

        private void Apply_Click(object sender, RoutedEventArgs e)
        {
            ApplyAction?.Invoke(this.ViewModel);
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
    }
}
