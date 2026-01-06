#pragma once

#include <mutex>

#include <Eigen/Dense>

#include <Helium/SpatialPartitionings/ISpartialPartitioning.h>

class PointCloud;

typedef uint64_t DataBlockKey;

class SparseDataBlockVoxel
{
public:
	bool valid = false;
	float signedDistance = 0.0f;
	float weight = 0.0f;
	Eigen::Vector3f normal = Eigen::Vector3f::Zero();
	Eigen::Vector3f color = Eigen::Vector3f::Zero();
	int clusterId = -1;
	float divergence = 0.0f;
};

struct DataBlock
{
	Eigen::Vector3f blockMin = Eigen::Vector3f::Zero();
	SparseDataBlockVoxel voxels[SpatialPartitioningConfiguration::voxelsPerBlock];
	std::mutex blockMutex;

	void Initialize();
};

class SparseDataBlock : public ISpatialPartitioning
{
public:
	float voxelSize = SpatialPartitioningConfiguration::voxelSize;
	Eigen::Vector3f gridOrigin = Eigen::Vector3f::Zero();
	std::unordered_map<DataBlockKey, std::unique_ptr<DataBlock>> dataBlocks;

	float blockSizePerAxis = voxelSize * SpatialPartitioningConfiguration::voxelsPerBlockAxis;

	SparseDataBlockVoxel* GetVoxelByIndex(int gx, int gy, int gz);

	void Build(const PointCloud* pc);

	void FromPointsData(const std::vector<Eigen::Vector3f>& points,
		const std::vector<Eigen::Vector3f>& normals,
		const std::vector<Eigen::Vector3f>& colors,
		const std::vector<int>& clusterIds,
		const Eigen::Vector3f& aabbMin);

	void Visualize();
};
