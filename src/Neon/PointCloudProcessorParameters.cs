using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neon
{
    public enum PointCloudVisualizationMode
    {
        None,
        Gradient,
        Binary,
        OutlierFiltered
    }
    
    public class PointCloudProcessorParameters
    {
        public int PointCloudID { get; set; } = -1;
        public bool DeletePoints { get; set; } = false;
        public PointCloudVisualizationMode VisualizationMode { get; set; } = PointCloudVisualizationMode.Gradient;
    }

    public class PointCloudProcessorParametersSOR : PointCloudProcessorParameters
    {
        public int KNeighbors { get; set; } = 50;
        public float StdDevMulThresh { get; set; } = 3.0f;
    }

    public class PointCloudProcessorParametersROR : PointCloudProcessorParameters
    {
        public float Radius { get; set; } = 0.3f;
        public int MinNeighborsInRadius { get; set; } = 24;
    }

    public class PointCloudProcessorParametersCurvatureAnalysis : PointCloudProcessorParameters
    {
        public int KNeighbors { get; set; } = 30;
        public float CurvatureThreshold { get; set; } = 1.0f;
    }

    public class PointCloudProcessorParametersNormalDeviation : PointCloudProcessorParameters
    {
        public float Radius { get; set; } = 0.1f;
        public float DeviationThreshold { get; set; } = 45.0f;
    }

    public class PointCloudProcessorParametersPlaneFitOutlierRemoval : PointCloudProcessorParameters
    {
        public int KNeighbors { get; set; } = 30;
        public float DistanceThreshold { get; set; } = 0.085f;
    }

    public class PointCloudProcessorParametersClustering : PointCloudProcessorParameters
    {
        public float SearchRadius { get; set; } = 0.15f;
        public float AngleThreshold { get; set; } = 0.9f;
    }
}
