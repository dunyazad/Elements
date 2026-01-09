using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;

namespace Neon.Nodes
{
    // 1. ±âº» ºä¸ðµ¨ (INotifyPropertyChanged ±¸Çö)
    public class BaseViewModel : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;
        protected void OnPropertyChanged([CallerMemberName] string? name = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }

    // 2. ÀüÃ¼ ³×Æ®¿öÅ© ºä¸ðµ¨
    public class NetworkViewModel : BaseViewModel
    {
        public ObservableCollection<NodeViewModel> Nodes { get; } = new ObservableCollection<NodeViewModel>();
        public ObservableCollection<ConnectionViewModel> Connections { get; } = new ObservableCollection<ConnectionViewModel>();

        // µå·¡±× ÁßÀÎ ÀÓ½Ã ¿¬°á¼± ÁÂÇ¥
        private Point _pendingConnectionSource;
        public Point PendingConnectionSource
        {
            get => _pendingConnectionSource;
            set { _pendingConnectionSource = value; OnPropertyChanged(); }
        }

        private Point _pendingConnectionTarget;
        public Point PendingConnectionTarget
        {
            get => _pendingConnectionTarget;
            set { _pendingConnectionTarget = value; OnPropertyChanged(); }
        }

        private bool _isDraggingConnection;
        public bool IsDraggingConnection
        {
            get => _isDraggingConnection;
            set { _isDraggingConnection = value; OnPropertyChanged(); }
        }
    }

    // 3. ³ëµå ºä¸ðµ¨
    public class NodeViewModel : BaseViewModel
    {
        private double _x;
        private double _y;
        private string _title = string.Empty;

        public NodeViewModel(string title, double x, double y)
        {
            Title = title;
            X = x;
            Y = y;
        }

        public string Title
        {
            get => _title;
            set { _title = value; OnPropertyChanged(); }
        }

        public double X
        {
            get => _x;
            set { _x = value; OnPropertyChanged(); }
        }

        public double Y
        {
            get => _y;
            set { _y = value; OnPropertyChanged(); }
        }

        public ObservableCollection<PinViewModel> Inputs { get; } = new ObservableCollection<PinViewModel>();
        public ObservableCollection<PinViewModel> Outputs { get; } = new ObservableCollection<PinViewModel>();
    }

    // 4. ÇÉ ºä¸ðµ¨
    public class PinViewModel : BaseViewModel
    {
        public string Name { get; set; } = string.Empty;
        public bool IsInput { get; set; }
        public NodeViewModel ParentNode { get; set; } = null!; // ÃÊ±âÈ­ ÇÊ¼ö

        // Canvas »óÀÇ Àý´ë ÁÂÇ¥
        private Point _pos;
        public Point Pos
        {
            get => _pos;
            set
            {
                if (_pos != value)
                {
                    _pos = value;
                    OnPropertyChanged();
                    ConnectionUpdated?.Invoke(this, EventArgs.Empty);
                }
            }
        }

        public event EventHandler? ConnectionUpdated;
    }

    // 5. ¿¬°á¼± ºä¸ðµ¨
    public class ConnectionViewModel : BaseViewModel
    {
        public PinViewModel Source { get; }
        public PinViewModel Target { get; }

        public ConnectionViewModel(PinViewModel source, PinViewModel target)
        {
            Source = source;
            Target = target;

            Source.ConnectionUpdated += OnPinMoved;
            Target.ConnectionUpdated += OnPinMoved;
            UpdateGeometry();
        }

        private void OnPinMoved(object? sender, EventArgs e) => UpdateGeometry();

        public void UpdateGeometry()
        {
            OnPropertyChanged(nameof(P1));
            OnPropertyChanged(nameof(P2));
            OnPropertyChanged(nameof(P3));
            OnPropertyChanged(nameof(P4));
        }

        // º£Áö¾î °î¼± ÁÂÇ¥
        public Point P1 => Source.Pos;
        public Point P4 => Target.Pos;
        public Point P2 => new Point(P1.X + 80, P1.Y);
        public Point P3 => new Point(P4.X - 80, P4.Y);
    }
}