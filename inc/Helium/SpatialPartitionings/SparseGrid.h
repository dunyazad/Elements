#pragma once

#include <Eigen/Dense>
#include <robin_hood/robin_hood.h>
#include <Helium/TypeDefinitions.h>

#include <Helium/SpatialPartitionings/ISpartialPartitioning.h>

class PointCloud;

class SparseGrid : public ISpatialPartitioning
{
public:
	robin_hood::unordered_flat_map<uint64_t, int> voxelPointListHead;
	std::vector<int> nextPoint;

	AABB aabb;
	float cellSize = 0.1f;

	uint64_t GetKey(int x, int y, int z) const;

	Eigen::Vector3i GetIndex(const Eigen::Vector3f& position) const;

	void Build(const PointCloud* pc, float cellSize);

	int GetClosestPoint(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3f& queryPos, float& outDist);

	void GetKNearestNeighbors(
		const std::vector<Eigen::Vector3f>& points,
		const Eigen::Vector3f& queryPos,
		int k,
		std::vector<unsigned int>& outIndices,
		std::vector<float>& outDistances) const;

	void GetPointsWithinRadius(
		const std::vector<Eigen::Vector3f>& points,
		const Eigen::Vector3f& queryPos,
		float radius,
		std::vector<unsigned int>& outIndices,
		std::vector<float>& outDistances) const;

	void Visualize();
};
