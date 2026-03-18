#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <unordered_map>

#include <Helium/Serialization.hpp>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;


namespace Eigen {
	template <typename Type, int Size>
	using Vector = Matrix<Type, Size, 1>;
	using Vector3b = Vector<unsigned char, 3>;
	using Vector3ui = Vector<unsigned int, 3>;
}

struct FlatClustering
{
	float voxelSize = 0.1f;

	std::vector<uint64_t> blockKeys;
	std::vector<Eigen::Vector3f> blockMinCorners;
	std::vector<unsigned int> blockOffsets;
	std::vector<unsigned int> blockCounts;
	std::vector<unsigned int> sortedIndices;

	struct KeyIndex {
		uint64_t key;
		unsigned int index;
	};
	std::vector<KeyIndex> proxy;

	inline uint64_t GetBlockKey(const Eigen::Vector3f& point, float invVoxelSize) const
	{
		int64_t xi = static_cast<int64_t>(std::floor(point.x() * invVoxelSize));
		int64_t yi = static_cast<int64_t>(std::floor(point.y() * invVoxelSize));
		int64_t zi = static_cast<int64_t>(std::floor(point.z() * invVoxelSize));

		return ((uint64_t)(xi & 0x1FFFFF) << 42) |
			((uint64_t)(yi & 0x1FFFFF) << 21) |
			((uint64_t)(zi & 0x1FFFFF));
	}

	inline uint64_t GetKeyFromIndices(int64_t xi, int64_t yi, int64_t zi) const
	{
		return ((uint64_t)(xi & 0x1FFFFF) << 42) |
			((uint64_t)(yi & 0x1FFFFF) << 21) |
			((uint64_t)(zi & 0x1FFFFF));
	}

	void Build(const std::vector<Eigen::Vector3f>& points, float blockSize = 0.3f)
	{
		this->voxelSize = blockSize;
		const size_t pointCount = points.size();
		if (pointCount == 0)
		{
			blockKeys.clear();
			return;
		}

		const float invVoxelSize = 1.0f / blockSize;

		if (proxy.size() != pointCount) proxy.resize(pointCount);
		if (sortedIndices.size() != pointCount) sortedIndices.resize(pointCount);

		const Eigen::Vector3f* pPoints = points.data();
		KeyIndex* pProxy = proxy.data();

		std::for_each(std::execution::par, pProxy, pProxy + pointCount, [this, pPoints, pProxy, invVoxelSize](KeyIndex& item) {
			size_t i = &item - pProxy;
			item.key = GetBlockKey(pPoints[i], invVoxelSize);
			item.index = static_cast<unsigned int>(i);
			});

		std::sort(std::execution::par, proxy.begin(), proxy.end(), [](const KeyIndex& a, const KeyIndex& b) {
			return a.key < b.key;
			});

		blockKeys.clear();
		blockMinCorners.clear();
		blockOffsets.clear();
		blockCounts.clear();

		if (pointCount > 0)
		{
			uint64_t currentKey = proxy[0].key;
			unsigned int currentOffset = 0;

			for (size_t i = 0; i < pointCount; ++i)
			{
				sortedIndices[i] = proxy[i].index;

				if (proxy[i].key != currentKey)
				{
					unsigned int count = static_cast<unsigned int>(i) - currentOffset;

					blockKeys.push_back(currentKey);
					blockOffsets.push_back(currentOffset);
					blockCounts.push_back(count);

					int64_t xi = (int64_t)((currentKey >> 42) & 0x1FFFFF); if (xi & 0x100000) xi |= ~0x1FFFFF;
					int64_t yi = (int64_t)((currentKey >> 21) & 0x1FFFFF); if (yi & 0x100000) yi |= ~0x1FFFFF;
					int64_t zi = (int64_t)(currentKey & 0x1FFFFF); if (zi & 0x100000) zi |= ~0x1FFFFF;

					blockMinCorners.emplace_back(
						static_cast<float>(xi) * blockSize,
						static_cast<float>(yi) * blockSize,
						static_cast<float>(zi) * blockSize);

					currentKey = proxy[i].key;
					currentOffset = static_cast<unsigned int>(i);
				}
			}
			blockKeys.push_back(currentKey);
			blockOffsets.push_back(currentOffset);
			blockCounts.push_back(static_cast<unsigned int>(pointCount) - currentOffset);
		}
	}

	std::vector<std::vector<unsigned int>> ExtractClusters(
		const std::vector<Eigen::Vector3f>& points,
		float distThreshold = 0.1f)
	{
		std::vector<std::vector<unsigned int>> clusters;
		if (blockKeys.empty())
		{
			return clusters;
		}

		const float sqDistThreshold = distThreshold * distThreshold;
		std::unordered_map<uint64_t, size_t> keyToBlockIdx;
		for (size_t i = 0; i < blockKeys.size(); ++i) keyToBlockIdx[blockKeys[i]] = i;

		std::vector<bool> blockVisited(blockKeys.size(), false);

		for (size_t i = 0; i < blockKeys.size(); ++i)
		{
			if (blockVisited[i]) continue;

			std::vector<unsigned int> currentCluster;
			std::vector<size_t> blockQueue;

			blockQueue.push_back(i);
			blockVisited[i] = true;

			size_t head = 0;
			while (head < blockQueue.size())
			{
				size_t currBlockIdx = blockQueue[head++];
				uint64_t key = blockKeys[currBlockIdx];
				unsigned int curOffset = blockOffsets[currBlockIdx];
				unsigned int curCount = blockCounts[currBlockIdx];

				for (unsigned int p = 0; p < curCount; ++p) {
					unsigned int pIdx = sortedIndices[curOffset + p];
					currentCluster.push_back(pIdx);
				}

				int64_t xi = (int64_t)((key >> 42) & 0x1FFFFF); if (xi & 0x100000) xi |= ~0x1FFFFF;
				int64_t yi = (int64_t)((key >> 21) & 0x1FFFFF); if (yi & 0x100000) yi |= ~0x1FFFFF;
				int64_t zi = (int64_t)(key & 0x1FFFFF); if (zi & 0x100000) zi |= ~0x1FFFFF;

				for (int64_t dz = -1; dz <= 1; ++dz) {
					for (int64_t dy = -1; dy <= 1; ++dy) {
						for (int64_t dx = -1; dx <= 1; ++dx) {
							if (dx == 0 && dy == 0 && dz == 0) continue;

							uint64_t neighborKey = GetKeyFromIndices(xi + dx, yi + dy, zi + dz);
							auto it = keyToBlockIdx.find(neighborKey);

							if (it != keyToBlockIdx.end() && !blockVisited[it->second])
							{
								size_t neighborBlockIdx = it->second;
								unsigned int nOffset = blockOffsets[neighborBlockIdx];
								unsigned int nCount = blockCounts[neighborBlockIdx];
								bool isConnected = false;

								for (unsigned int p1 = 0; p1 < curCount; ++p1) {
									unsigned int idx1 = sortedIndices[curOffset + p1];
									const Eigen::Vector3f& pt1 = points[idx1];

									for (unsigned int p2 = 0; p2 < nCount; ++p2) {
										unsigned int idx2 = sortedIndices[nOffset + p2];
										if ((pt1 - points[idx2]).squaredNorm() <= sqDistThreshold) {
											isConnected = true;
											break;
										}
									}
									if (isConnected) break;
								}

								if (isConnected) {
									blockVisited[neighborBlockIdx] = true;
									blockQueue.push_back(neighborBlockIdx);
								}
							}
						}
					}
				}
			}

			if (!currentCluster.empty()) {
				clusters.push_back(std::move(currentCluster));
			}
		}
		return clusters;
	}

	std::vector<std::vector<unsigned int>> ExtractClusters(
		const std::vector<Eigen::Vector3f>& points,
		const std::vector<Eigen::Vector3f>& normals,
		float distThreshold = 0.1f,
		float normalThreshold = 0.9f)
	{
		std::vector<std::vector<unsigned int>> clusters;
		if (blockKeys.empty() || points.size() != normals.size())
		{
			return clusters;
		}

		const float sqDistThreshold = distThreshold * distThreshold;
		std::unordered_map<uint64_t, size_t> keyToBlockIdx;
		for (size_t i = 0; i < blockKeys.size(); ++i) keyToBlockIdx[blockKeys[i]] = i;

		std::vector<bool> blockVisited(blockKeys.size(), false);

		for (size_t i = 0; i < blockKeys.size(); ++i)
		{
			if (blockVisited[i]) continue;

			std::vector<unsigned int> currentCluster;
			std::vector<size_t> blockQueue;

			blockQueue.push_back(i);
			blockVisited[i] = true;

			size_t head = 0;
			while (head < blockQueue.size())
			{
				size_t currBlockIdx = blockQueue[head++];
				uint64_t key = blockKeys[currBlockIdx];
				unsigned int curOffset = blockOffsets[currBlockIdx];
				unsigned int curCount = blockCounts[currBlockIdx];

				for (unsigned int p = 0; p < curCount; ++p) {
					unsigned int pIdx = sortedIndices[curOffset + p];
					currentCluster.push_back(pIdx);
				}

				int64_t xi = (int64_t)((key >> 42) & 0x1FFFFF); if (xi & 0x100000) xi |= ~0x1FFFFF;
				int64_t yi = (int64_t)((key >> 21) & 0x1FFFFF); if (yi & 0x100000) yi |= ~0x1FFFFF;
				int64_t zi = (int64_t)(key & 0x1FFFFF); if (zi & 0x100000) zi |= ~0x1FFFFF;

				for (int64_t dz = -1; dz <= 1; ++dz) {
					for (int64_t dy = -1; dy <= 1; ++dy) {
						for (int64_t dx = -1; dx <= 1; ++dx) {
							if (dx == 0 && dy == 0 && dz == 0) continue;

							uint64_t neighborKey = GetKeyFromIndices(xi + dx, yi + dy, zi + dz);
							auto it = keyToBlockIdx.find(neighborKey);

							if (it != keyToBlockIdx.end() && !blockVisited[it->second])
							{
								size_t neighborBlockIdx = it->second;
								unsigned int nOffset = blockOffsets[neighborBlockIdx];
								unsigned int nCount = blockCounts[neighborBlockIdx];
								bool isConnected = false;

								for (unsigned int p1 = 0; p1 < curCount; ++p1) {
									unsigned int idx1 = sortedIndices[curOffset + p1];
									const Eigen::Vector3f& pt1 = points[idx1];
									const Eigen::Vector3f& nm1 = normals[idx1];

									for (unsigned int p2 = 0; p2 < nCount; ++p2) {
										unsigned int idx2 = sortedIndices[nOffset + p2];
										if ((pt1 - points[idx2]).squaredNorm() <= sqDistThreshold) {
											if (std::abs(nm1.dot(normals[idx2])) >= normalThreshold) {
												isConnected = true;
												break;
											}
										}
									}
									if (isConnected) break;
								}

								if (isConnected) {
									blockVisited[neighborBlockIdx] = true;
									blockQueue.push_back(neighborBlockIdx);
								}
							}
						}
					}
				}
			}

			if (!currentCluster.empty()) {
				clusters.push_back(std::move(currentCluster));
			}
		}
		return clusters;
	}
};

class AppClustering : public App
{
public:
	virtual void Execute() override
	{
		PLYFormat ply;
		ply.Deserialize("D:\\Debug\\original_points.ply");
		if (ply.GetPoints().empty())
		{
			printf("Failed to load point cloud.\n");
			return;
		}

		TS(Total);

		static FlatClustering clustering;

		TS(Build);
		clustering.Build(ply.GetPoints(), 0.1f);
		TE(Build);

		TS(Extract);
		//auto clusters = clustering.ExtractClusters(
		//	ply.GetPoints(),
		//	ply.GetNormals(),
		//	0.15f,
		//	0.9f);

		auto clusters = clustering.ExtractClusters(
			ply.GetPoints(),
			0.175f);
		TE(Extract);

		TE(Total);

		auto colors = Color::GetContrastingColorsWithoutBWRGB(128);

		for (size_t i = 0; i < clusters.size(); ++i)
		{
			if (clusters[i].size() < 10) continue;

			Eigen::Vector4f randomColor = colors[i % colors.size()];
			std::vector<Eigen::Vector3f> clusterPoints;
			clusterPoints.reserve(clusters[i].size());

			for (auto idx : clusters[i]) {
				clusterPoints.push_back(ply.GetPoints()[idx]);
			}

			VD::AddSphereBatch("Cluster_" + std::to_string(i), clusterPoints, 0.05f, randomColor);
		}
	}
};

REGISTER_APP(AppClustering, "AppClustering");