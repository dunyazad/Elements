using System;
using System.Collections.ObjectModel;
using System.Linq; // SingleOrDefault 사용을 위해
using System.Windows;
using System.Windows.Input;

namespace Neon.ViewModels
{
    public class FilterStep : BaseViewModel
    {
        public static ObservableCollection<string> FilterTypes { get; } = new ObservableCollection<string>
        {
            "SOR Filter",
            "ROR Filter",
            "Curvature Analysis",
            "Normal Deviation Analysis",
            "Clustering"
        };

        public static ObservableCollection<string> OperationTypes { get; } = new ObservableCollection<string>
        {
            "Union",
            "Subtraction",
            "Intersection"
        };

        private string _selectedType;
        public string SelectedType
        {
            get => _selectedType;
            set
            {
                if (_selectedType != value)
                {
                    _selectedType = value;
                    OnPropertyChanged();

                    UpdateDefaultParameters();
                }
            }
        }

        private string _selectedOperation;
        public string SelectedOperation
        {
            get => _selectedOperation;
            set { _selectedOperation = value; OnPropertyChanged(); }
        }

        private PointCloudProcessorParameters _parameters = new PointCloudProcessorParameters();
        public PointCloudProcessorParameters Parameters
        {
            get => _parameters;
            set { _parameters = value; OnPropertyChanged(); }
        }

        public ICommand ConfigureCommand { get; }

        public FilterStep(string type = "SOR Filter", string operation = "Union")
        {
            _selectedType = type;
            _selectedOperation = operation;

            UpdateDefaultParameters();

            ConfigureCommand = new RelayCommand(_ => OpenParameterDialog());
        }

        public void UpdateDefaultParameters()
        {
            switch (_selectedType)
            {
                case "SOR Filter":
                    Parameters = new PointCloudProcessorParametersSOR();
                    break;
                case "ROR Filter":
                    Parameters = new PointCloudProcessorParametersROR();
                    break;
                case "Curvature Analysis":
                    Parameters = new PointCloudProcessorParametersCurvatureAnalysis();
                    break;
                case "Normal Deviation Analysis":
                    Parameters = new PointCloudProcessorParametersNormalDeviation();
                    break;
                case "Clustering":
                    Parameters = new PointCloudProcessorParametersClustering();
                    break;
                default:
                    Parameters = new PointCloudProcessorParameters();
                    break;
            }

            Parameters.visualizationMode = PointCloudVisualizationMode.None;
        }

        private void OpenParameterDialog()
        {
            Window? dlg = null;

            switch (_selectedType)
            {
                case "SOR Filter":
                    var sorDlg = new SorParameterDialog();
                    if (Parameters is PointCloudProcessorParametersSOR sorParams)
                        sorDlg.Parameters = sorParams;
                    dlg = sorDlg;
                    break;

                case "ROR Filter":
                    var rorDlg = new RorParameterDialog();
                    if (Parameters is PointCloudProcessorParametersROR rorParams)
                        rorDlg.Parameters = rorParams;
                    dlg = rorDlg;
                    break;

                case "Curvature Analysis":
                    var curvDlg = new CurvatureAnalysisParameterDialog();
                    if (Parameters is PointCloudProcessorParametersCurvatureAnalysis curvParams)
                        curvDlg.Parameters = curvParams;
                    dlg = curvDlg;
                    break;

                case "Normal Deviation Analysis":
                    var normDlg = new NormalDeviationAnalysisParameterDialog();
                    if (Parameters is PointCloudProcessorParametersNormalDeviation normParams)
                        normDlg.Parameters = normParams;
                    dlg = normDlg;
                    break;

                case "Clustering":
                    MessageBox.Show("Clustering settings are not implemented yet.", "Info");
                    return;
            }

            if (dlg != null)
            {
                dlg.Owner = Application.Current.Windows.OfType<Window>().SingleOrDefault(x => x.IsActive);

                if (dlg.ShowDialog() == true)
                {
                    OnPropertyChanged(nameof(Parameters));
                }
            }
        }
    }
}
