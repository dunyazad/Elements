#include "pch.h"

#include <Helium/PointProcessing/PointNormalDeviation.h>

#include <execution>

#include <Helium/HeliumCore.h>
#include <Helium/PointCloud.h>
#include <Helium/SpatialPartitionings/SparseGrid.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

namespace PointProcessing
{
	NormalDeviation::NormalDeviation()
		: PointProcessor(PointProcessorType::NormalDeviation)
	{
	}

	void NormalDeviation::Process(const PointProcessorParameters& parameters)
	{
	}
}
