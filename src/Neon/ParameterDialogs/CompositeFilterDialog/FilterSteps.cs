using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Windows;
using System.Windows.Input;

namespace Neon.ParameterDialogs.CompositeFilterDialog
{
    public class FilterStep : BaseViewModel
    {
        // For XAML CheckBox binding
        private bool _isChecked;
        public bool IsChecked
        {
            get => _isChecked;
            set { _isChecked = value; OnPropertyChanged(); }
        }

        public static ObservableCollection<string> FilterTypes { get; } = new ObservableCollection<string>
        {
            "SOR Filter",
            "ROR Filter",
            "Curvature Analysis",
            "Normal Deviation Analysis",
            "FPOR Filter",
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

        // Assuming PointCloudProcessorParameters and derived classes exist in your project
        private PointCloudProcessorParameters _parameters;
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
            _parameters = new PointCloudProcessorParameters(); // Default init to avoid null

            UpdateDefaultParameters();

            ConfigureCommand = new RelayCommand(_ => OpenParameterDialog());
        }

        public void UpdateDefaultParameters()
        {
            // Logic to instantiate specific parameter classes
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
                case "FPOR Filter":
                    Parameters = new PointCloudProcessorParametersPlaneFitOutlierRemoval();
                    break;
                case "Clustering":
                    Parameters = new PointCloudProcessorParametersClustering();
                    break;
                default:
                    Parameters = new PointCloudProcessorParameters();
                    break;
            }

            if (Parameters != null)
            {
                Parameters.VisualizationMode = PointCloudVisualizationMode.None;
            }
        }

        private void OpenParameterDialog()
        {
            Window? dlg = null;

            // Logic to open specific dialogs based on Type
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

                case "FPOR Filter":
                    var pforDlg = new PFORParameterDialog();
                    if (Parameters is PointCloudProcessorParametersPlaneFitOutlierRemoval pforParams)
                        pforDlg.Parameters = pforParams;
                    dlg = pforDlg;
                    break;

                case "Clustering":
                    MessageBox.Show("Clustering settings are not implemented yet.", "Info");
                    return;
            }

            if (dlg != null)
            {
                // Set owner to the active window to ensure modal behavior
                dlg.Owner = Application.Current.Windows.OfType<Window>().SingleOrDefault(x => x.IsActive);

                if (dlg.ShowDialog() == true)
                {
                    OnPropertyChanged(nameof(Parameters));
                }
            }
        }
    }
}
