using Neon.Controls;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Neon
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            NeonLogger.OnLog = AppendLog;

            NeonLogger.Log("[Neon] Application started");

            CompositionTarget.Rendering += OnRendering;
        }

        private void AppendLog(string msg)
        {
            LogTextBox.AppendText(msg + Environment.NewLine);
            LogTextBox.ScrollToEnd();
        }

        private void OnRendering(object? sender, EventArgs e)
        {
            try
            {
                HeliumNative.Helium_Render();
            }
            catch
            {
                // 절대 throw 금지
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            CompositionTarget.Rendering -= OnRendering;
            base.OnClosed(e);
        }
    }
}
