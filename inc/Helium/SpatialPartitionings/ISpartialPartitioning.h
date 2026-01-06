#pragma once

#include <limits>

class ISpatialPartitioning {};

class SpatialPartitioningConfiguration
{
public:
	static constexpr int voxelsPerBlockAxis = 8;
	static constexpr int voxelsPerBlock =
		voxelsPerBlockAxis * voxelsPerBlockAxis * voxelsPerBlockAxis;

	static constexpr float voxelSize = 0.3f;
	static constexpr int sdfOffset = 1;

	inline static Eigen::Vector3f filterMin = Eigen::Vector3f::Constant(-FLT_MAX);
	inline static Eigen::Vector3f filterMax = Eigen::Vector3f::Constant(FLT_MAX);

	static constexpr float pointVisualizationRadius = 0.025f;
};
