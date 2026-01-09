using System;
using System.Windows;

namespace Neon
{
    public partial class CompositeOutlierFilterDialog : Window
    {
        // Action: (useSOR, useROR, useCurvature, useNormal, isUnion, deletePoints)
        public Action<bool, bool, bool, bool, bool, bool>? ApplyAction { get; set; }

        public CompositeOutlierFilterDialog()
        {
            InitializeComponent();
        }

        private void Preview_Click(object sender, RoutedEventArgs e)
        {
            InvokeAction(deletePoints: false);
        }

        private void Apply_Click(object sender, RoutedEventArgs e)
        {
            var result = MessageBox.Show("This will permanently delete points matching the criteria. Continue?",
                                         "Confirm Delete", MessageBoxButton.YesNo, MessageBoxImage.Warning);

            if (result == MessageBoxResult.Yes)
            {
                InvokeAction(deletePoints: true);
                Close();
            }
        }

        private void InvokeAction(bool deletePoints)
        {
            bool useSOR = CheckSOR.IsChecked == true;
            bool useROR = CheckROR.IsChecked == true;
            bool useCurv = CheckCurvature.IsChecked == true;
            bool useNormal = CheckNormalDev.IsChecked == true;
            bool isUnion = RadioUnion.IsChecked == true; // True: OR, False: AND

            ApplyAction?.Invoke(useSOR, useROR, useCurv, useNormal, isUnion, deletePoints);
        }
    }
}
