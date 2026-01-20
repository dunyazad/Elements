#include "pch.h"

#include <Helium/PointProcessing/PointSOR.h>

#include <execution>

#include <Helium/HeliumCore.h>
#include <Helium/PointCloud.h>
#include <Helium/SpatialPartitionings/SparseGrid.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

namespace PointProcessing
{
	SOR::SOR()
		: PointProcessor(PointProcessorType::SOR)
	{
	}

	void SOR::Process(const PointProcessorParameters& parameters)
	{
	}
}
