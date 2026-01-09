using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Windows;

namespace Neon.Controls
{
    public class ParameterItem
    {
        public string Key { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;
    }

    public partial class FilterParametersDialog : Window
    {
        public string FilterName { get; set; } = string.Empty; // [수정 1] 초기값 할당
        public ObservableCollection<ParameterItem> ParameterList { get; set; }
        private Dictionary<string, object> _originalParams;

        public FilterParametersDialog(string filterName, Dictionary<string, object> parameters)
        {
            InitializeComponent();
            FilterName = filterName;
            _originalParams = parameters;
            ParameterList = new ObservableCollection<ParameterItem>();

            foreach (var kvp in parameters)
            {
                string safeValue = kvp.Value?.ToString() ?? string.Empty;

                ParameterList.Add(new ParameterItem { Key = kvp.Key, Value = safeValue });
            }

            this.DataContext = this;
        }

        private void OK_Click(object sender, RoutedEventArgs e)
        {
            foreach (var item in ParameterList)
            {
                if (_originalParams.ContainsKey(item.Key))
                {
                    object originalVal = _originalParams[item.Key];

                    string newValue = item.Value;

                    if (originalVal is int)
                    {
                        if (int.TryParse(newValue, out int iVal)) _originalParams[item.Key] = iVal;
                    }
                    else if (originalVal is float)
                    {
                        if (float.TryParse(newValue, out float fVal)) _originalParams[item.Key] = fVal;
                    }
                    else if (originalVal is double) // double도 추가하면 좋음
                    {
                        if (double.TryParse(newValue, out double dVal)) _originalParams[item.Key] = dVal;
                    }
                    else
                    {
                        _originalParams[item.Key] = newValue;
                    }
                }
            }
            this.DialogResult = true;
            this.Close();
        }
    }
}
