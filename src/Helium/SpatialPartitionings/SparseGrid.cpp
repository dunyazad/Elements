#include "pch.h"

#undef min
#undef max

#include <queue>

#include <Helium/SpatialPartitionings/SparseGrid.h>
#include <Helium/PointCloud.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

uint64_t SparseGrid::GetKey(int x, int y, int z) const
{
	const uint64_t MASK = 0x1FFFFF; // 21 bits
	return (((uint64_t)x & MASK) << 42) | (((uint64_t)y & MASK) << 21) | ((uint64_t)z & MASK);
}

Eigen::Vector3i SparseGrid::GetIndex(const Eigen::Vector3f& position) const
{
	int gx = (int)std::floor((position.x() - aabb.min.x()) / cellSize);
	int gy = (int)std::floor((position.y() - aabb.min.y()) / cellSize);
	int gz = (int)std::floor((position.z() - aabb.min.z()) / cellSize);
	return { gx, gy, gz };
}

void SparseGrid::Build(const PointCloud* pc, float cellSize)
{
	if (pc->Size() == 0)
	{
		ErrorLog("", "PointCloud is empty. Cannot build SparseGrid.\n");
		return;
	}

	this->cellSize = cellSize;

	aabb = pc->GetAABB();

	voxelPointListHead.clear();
	voxelPointListHead.reserve(pc->Size());

	nextPoint.assign(pc->Size(), -1);

	aabb.min -= Eigen::Vector3f::Constant(cellSize * 0.1f);
	aabb.max += Eigen::Vector3f::Constant(cellSize * 0.1f);

	for (int i = 0; i < (int)pc->Size(); ++i)
	{
		int gx = (int)((pc->GetPosition(i).x() - aabb.min.x()) / cellSize);
		int gy = (int)((pc->GetPosition(i).y() - aabb.min.y()) / cellSize);
		int gz = (int)((pc->GetPosition(i).z() - aabb.min.z()) / cellSize);

		uint64_t key = GetKey(gx, gy, gz);

		auto it = voxelPointListHead.find(key);

		if (it != voxelPointListHead.end())
		{
			nextPoint[i] = it->second;
			it->second = i;
		}
		else
		{
			voxelPointListHead[key] = i;
		}
	}
}

int SparseGrid::GetClosestPoint(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3f& queryPos, float& outDist)
{
	outDist = FLT_MAX;
	if (points.empty()) return -1;

	int startGx = (int)std::floor((queryPos.x() - aabb.min.x()) / cellSize);
	int startGy = (int)std::floor((queryPos.y() - aabb.min.y()) / cellSize);
	int startGz = (int)std::floor((queryPos.z() - aabb.min.z()) / cellSize);

	float minDistSq = FLT_MAX;
	int closestIdx = -1;

	int searchRadius = 0;
	const int maxSearchRadius = 100;

	while (searchRadius < maxSearchRadius)
	{
		int minR = -searchRadius;
		int maxR = searchRadius;

		for (int dz = minR; dz <= maxR; ++dz)
		{
			for (int dy = minR; dy <= maxR; ++dy)
			{
				for (int dx = minR; dx <= maxR; ++dx)
				{
					if (searchRadius > 0 && std::abs(dx) != searchRadius && std::abs(dy) != searchRadius && std::abs(dz) != searchRadius)
					{
						continue;
					}

					uint64_t key = GetKey(startGx + dx, startGy + dy, startGz + dz);
					auto it = voxelPointListHead.find(key);

					if (it != voxelPointListHead.end())
					{
						int currIdx = it->second;
						while (currIdx != -1)
						{
							Eigen::Vector3f diff = queryPos - points[currIdx];
							float sqDist = diff.squaredNorm();

							if (sqDist < minDistSq)
							{
								minDistSq = sqDist;
								closestIdx = currIdx;
							}
							currIdx = nextPoint[currIdx];
						}
					}
				}
			}
		}

		if (closestIdx != -1)
		{
			float minX = aabb.min.x() + (startGx - searchRadius) * cellSize;
			float maxX = aabb.min.x() + (startGx + searchRadius + 1) * cellSize;
			float minY = aabb.min.y() + (startGy - searchRadius) * cellSize;
			float maxY = aabb.min.y() + (startGy + searchRadius + 1) * cellSize;
			float minZ = aabb.min.z() + (startGz - searchRadius) * cellSize;
			float maxZ = aabb.min.z() + (startGz + searchRadius + 1) * cellSize;

			float distToX = std::min(std::abs(queryPos.x() - minX), std::abs(queryPos.x() - maxX));
			float distToY = std::min(std::abs(queryPos.y() - minY), std::abs(queryPos.y() - maxY));
			float distToZ = std::min(std::abs(queryPos.z() - minZ), std::abs(queryPos.z() - maxZ));

			float minDistToBoundary = std::min({ distToX, distToY, distToZ });

			if (minDistSq < minDistToBoundary * minDistToBoundary)
			{
				break;
			}
		}

		searchRadius++;
	}

	if (closestIdx != -1)
	{
		outDist = std::sqrt(minDistSq);
	}

	return closestIdx;
}

void SparseGrid::GetKNearestNeighbors(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3f& queryPos, int k, std::vector<unsigned int>& outIndices, std::vector<float>& outDistances) const
{
	outIndices.clear();
	outDistances.clear();
	if (points.empty() || k <= 0) return;

	std::priority_queue<std::pair<float, int>> pq;

	int startGx = (int)std::floor((queryPos.x() - aabb.min.x()) / cellSize);
	int startGy = (int)std::floor((queryPos.y() - aabb.min.y()) / cellSize);
	int startGz = (int)std::floor((queryPos.z() - aabb.min.z()) / cellSize);

	int searchRadius = 0;
	const int maxSearchRadius = 100;

	while (searchRadius < maxSearchRadius)
	{
		int minR = -searchRadius;
		int maxR = searchRadius;

		for (int dz = minR; dz <= maxR; ++dz)
		{
			for (int dy = minR; dy <= maxR; ++dy)
			{
				for (int dx = minR; dx <= maxR; ++dx)
				{
					if (searchRadius > 0 && std::abs(dx) != searchRadius && std::abs(dy) != searchRadius && std::abs(dz) != searchRadius)
					{
						continue;
					}

					uint64_t key = GetKey(startGx + dx, startGy + dy, startGz + dz);
					auto it = voxelPointListHead.find(key);

					if (it != voxelPointListHead.end())
					{
						int currIdx = it->second;
						while (currIdx != -1)
						{
							Eigen::Vector3f diff = queryPos - points[currIdx];
							float sqDist = diff.squaredNorm();

							if (pq.size() < (size_t)k)
							{
								pq.push({ sqDist, currIdx });
							}
							else if (sqDist < pq.top().first)
							{
								pq.pop();
								pq.push({ sqDist, currIdx });
							}
							currIdx = nextPoint[currIdx];
						}
					}
				}
			}
		}

		if (pq.size() == (size_t)k)
		{
			float minX = aabb.min.x() + (startGx - searchRadius) * cellSize;
			float maxX = aabb.min.x() + (startGx + searchRadius + 1) * cellSize;
			float minY = aabb.min.y() + (startGy - searchRadius) * cellSize;
			float maxY = aabb.min.y() + (startGy + searchRadius + 1) * cellSize;
			float minZ = aabb.min.z() + (startGz - searchRadius) * cellSize;
			float maxZ = aabb.min.z() + (startGz + searchRadius + 1) * cellSize;

			float distToX = std::min(std::abs(queryPos.x() - minX), std::abs(queryPos.x() - maxX));
			float distToY = std::min(std::abs(queryPos.y() - minY), std::abs(queryPos.y() - maxY));
			float distToZ = std::min(std::abs(queryPos.z() - minZ), std::abs(queryPos.z() - maxZ));

			float minDistToBoundary = std::min({ distToX, distToY, distToZ });

			if (minDistToBoundary * minDistToBoundary > pq.top().first)
			{
				break;
			}
		}

		searchRadius++;
	}

	size_t count = pq.size();
	outIndices.resize(count);
	outDistances.resize(count);

	for (int i = (int)count - 1; i >= 0; --i)
	{
		outIndices[i] = pq.top().second;
		outDistances[i] = std::sqrt(pq.top().first);
		pq.pop();
	}
}

void SparseGrid::GetPointsWithinRadius(const std::vector<Eigen::Vector3f>& points, const Eigen::Vector3f& queryPos, float radius, std::vector<unsigned int>& outIndices, std::vector<float>& outDistances) const
{
	outIndices.clear();
	outDistances.clear();

	if (points.empty() || radius <= 0.0f) return;

	float radiusSq = radius * radius;

	int range = (int)std::ceil(radius / cellSize);

	int centerGx = (int)std::floor((queryPos.x() - aabb.min.x()) / cellSize);
	int centerGy = (int)std::floor((queryPos.y() - aabb.min.y()) / cellSize);
	int centerGz = (int)std::floor((queryPos.z() - aabb.min.z()) / cellSize);

	for (int dz = -range; dz <= range; ++dz)
	{
		for (int dy = -range; dy <= range; ++dy)
		{
			for (int dx = -range; dx <= range; ++dx)
			{
				uint64_t key = GetKey(centerGx + dx, centerGy + dy, centerGz + dz);
				auto it = voxelPointListHead.find(key);

				if (it != voxelPointListHead.end())
				{
					int currIdx = it->second;
					while (currIdx != -1)
					{
						const auto& pt = points[currIdx];
						float sqDist = (queryPos - pt).squaredNorm();

						if (sqDist <= radiusSq)
						{
							outIndices.push_back((unsigned int)currIdx);
							outDistances.push_back(std::sqrt(sqDist));
						}

						currIdx = nextPoint[currIdx];
					}
				}
			}
		}
	}
}

void SparseGrid::Visualize()
{
	const uint64_t mask = 0x1FFFFF;

	for (const auto& pair : voxelPointListHead)
	{
		uint64_t key = pair.first;
		int headIdx = pair.second;

		uint64_t gz = key & mask;
		uint64_t gy = (key >> 21) & mask;
		uint64_t gx = (key >> 42);

		Eigen::Vector3f cellMin = aabb.min + Eigen::Vector3f((float)gx * cellSize, (float)gy * cellSize, (float)gz * cellSize);
		Eigen::Vector3f cellMax = cellMin + Eigen::Vector3f(cellSize, cellSize, cellSize);

		VD::AddWiredBox("SparseGridCells", { cellMin, cellMax }, Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f));
	}
}
