using System;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Input;

namespace Neon.ViewModels
{
    public class BaseViewModel : INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler? PropertyChanged;
        protected void OnPropertyChanged([CallerMemberName] string? name = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        }
    }

    public class RelayCommand : ICommand
    {
        private readonly Action<object?> _execute;
        private readonly Predicate<object?>? _canExecute;

        public RelayCommand(Action<object?> execute, Predicate<object?>? canExecute = null)
        {
            _execute = execute;
            _canExecute = canExecute;
        }

        public bool CanExecute(object? parameter) => _canExecute == null || _canExecute(parameter);
        public void Execute(object? parameter) => _execute(parameter);
        public event EventHandler? CanExecuteChanged
        {
            add { CommandManager.RequerySuggested += value; }
            remove { CommandManager.RequerySuggested -= value; }
        }
    }

    public class CompositeFilterViewModel : BaseViewModel
    {
        public ObservableCollection<FilterStep> Steps { get; } = new ObservableCollection<FilterStep>();

        private FilterStep? _selectedStep;
        public FilterStep? SelectedStep
        {
            get => _selectedStep;
            set
            {
                _selectedStep = value;
                OnPropertyChanged();
                CommandManager.InvalidateRequerySuggested();
            }
        }

        public ICommand AddCommand { get; }
        public ICommand RemoveCommand { get; }
        public ICommand MoveUpCommand { get; }
        public ICommand MoveDownCommand { get; }

        public CompositeFilterViewModel()
        {
            AddCommand = new RelayCommand(_ => AddStep());
            RemoveCommand = new RelayCommand(_ => RemoveStep(), _ => SelectedStep != null);
            MoveUpCommand = new RelayCommand(_ => MoveUp(), _ => CanMoveUp());
            MoveDownCommand = new RelayCommand(_ => MoveDown(), _ => CanMoveDown());

            Steps.Add(new FilterStep("SOR Filter"));
            Steps.Add(new FilterStep("ROR Filter"));
        }

        private void AddStep()
        {
            var newStep = new FilterStep();
            Steps.Add(newStep);
            SelectedStep = newStep;
        }

        private void RemoveStep()
        {
            if (SelectedStep != null)
            {
                Steps.Remove(SelectedStep);
                SelectedStep = null;
            }
        }

        private void MoveUp()
        {
            if (SelectedStep == null) return;
            int index = Steps.IndexOf(SelectedStep);
            if (index > 0)
            {
                Steps.Move(index, index - 1);
            }
        }

        private bool CanMoveUp() => SelectedStep != null && Steps.IndexOf(SelectedStep) > 0;

        private void MoveDown()
        {
            if (SelectedStep == null) return;
            int index = Steps.IndexOf(SelectedStep);
            if (index < Steps.Count - 1)
            {
                Steps.Move(index, index + 1);
            }
        }

        private bool CanMoveDown() => SelectedStep != null && Steps.IndexOf(SelectedStep) < Steps.Count - 1;
    }
}
