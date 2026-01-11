using Neon.ParameterDialogs.CompositeFilterDialog;
using System;
using System.Windows;
using System.Windows.Input;

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

        private void TitleBar_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Left)
            {
                this.DragMove();
            }
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                this.DialogResult = false;
            }
            catch (InvalidOperationException)
            {
            }
            this.Close();
        }

        private void Apply_Click(object sender, RoutedEventArgs e)
        {
            ApplyAction?.Invoke(ViewModel);

            try
            {
                this.DialogResult = true;
            }
            catch (InvalidOperationException)
            {
            }
            this.Close();
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                this.DialogResult = false;
            }
            catch (InvalidOperationException)
            {
            }
            this.Close();
        }
    }
}