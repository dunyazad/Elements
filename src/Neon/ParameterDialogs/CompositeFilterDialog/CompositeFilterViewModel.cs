using Neon.Nodes;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Windows.Input;

namespace Neon.ParameterDialogs.CompositeFilterDialog
{
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
            RemoveCommand = new RelayCommand(_ => RemoveStep(), _ => CanRemoveStep());
            MoveUpCommand = new RelayCommand(_ => MoveUp(), _ => CanMoveUp());
            MoveDownCommand = new RelayCommand(_ => MoveDown(), _ => CanMoveDown());

            // Initial Default Steps
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
            // 1. Remove checked items if any
            var checkedItems = Steps.Where(s => s.IsChecked).ToList();
            if (checkedItems.Any())
            {
                foreach (var item in checkedItems)
                {
                    Steps.Remove(item);
                }
                return;
            }

            // 2. Otherwise remove selected item
            if (SelectedStep != null)
            {
                Steps.Remove(SelectedStep);
                SelectedStep = null;
            }
        }

        private bool CanRemoveStep()
        {
            return SelectedStep != null || Steps.Any(s => s.IsChecked);
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