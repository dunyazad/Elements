#include "pch.h"

#include <Helium/PointProcessing/PointCurvatureAnalysis.h>

#include <execution>

#include <Helium/HeliumCore.h>
#include <Helium/PointCloud.h>
#include <Helium/SpatialPartitionings/SparseGrid.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

namespace PointProcessing
{
	CurvatureAnalysis::CurvatureAnalysis()
		: PointProcessor(PointProcessorType::CurvatureAnalysis)
	{
	}

	void CurvatureAnalysis::Process(const PointProcessorParameters& parameters)
	{
	}
}
