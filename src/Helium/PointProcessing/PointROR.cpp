#include "pch.h"

#include <Helium/PointProcessing/PointROR.h>

#include <execution>

#include <Helium/HeliumCore.h>
#include <Helium/PointCloud.h>
#include <Helium/SpatialPartitionings/SparseGrid.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

namespace PointProcessing
{
	ROR::ROR()
		: PointProcessor(PointProcessorType::ROR)
	{
	}

	void ROR::Process(const PointProcessorParameters& parameters)
	{
	}
}
