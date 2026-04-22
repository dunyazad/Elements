#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <unordered_map>

#include <robin_hood/robin_hood.h>

#include <Helium/Serialization.hpp>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;


namespace Eigen {
	template <typename Type, int Size>
	using Vector = Matrix<Type, Size, 1>;
	using Vector3b = Vector<unsigned char, 3>;
	using Vector3ui = Vector<unsigned int, 3>;
}

class FlatClustering
{
public:
	struct KeyIndex {
		uint64_t key;
		unsigned int index;
	};

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
		if (blockKeys.empty())
		{
			return {};
		}

		const float sqDistThreshold = distThreshold * distThreshold;
		size_t numBlocks = blockKeys.size();

		robin_hood::unordered_flat_map<uint64_t, size_t> keyToBlockIdx;
		for (size_t i = 0; i < numBlocks; ++i)
		{
			keyToBlockIdx[blockKeys[i]] = i;
		}

		std::vector<std::vector<size_t>> adj(numBlocks);
		std::vector<size_t> blockIndices(numBlocks);
		std::iota(blockIndices.begin(), blockIndices.end(), 0);

		std::for_each(std::execution::par, blockIndices.begin(), blockIndices.end(), [&](size_t i)
			{
				uint64_t key = blockKeys[i];
				unsigned int curOffset = blockOffsets[i];
				unsigned int curCount = blockCounts[i];

				int64_t xi = (int64_t)((key >> 42) & 0x1FFFFF); if (xi & 0x100000) xi |= ~0x1FFFFF;
				int64_t yi = (int64_t)((key >> 21) & 0x1FFFFF); if (yi & 0x100000) yi |= ~0x1FFFFF;
				int64_t zi = (int64_t)(key & 0x1FFFFF); if (zi & 0x100000) zi |= ~0x1FFFFF;

				for (int64_t dz = -1; dz <= 1; ++dz)
				{
					for (int64_t dy = -1; dy <= 1; ++dy)
					{
						for (int64_t dx = -1; dx <= 1; ++dx)
						{
							if (dx == 0 && dy == 0 && dz == 0) continue;

							uint64_t neighborKey = GetKeyFromIndices(xi + dx, yi + dy, zi + dz);
							auto it = keyToBlockIdx.find(neighborKey);

							if (it != keyToBlockIdx.end())
							{
								size_t ni = it->second;
								unsigned int nOffset = blockOffsets[ni];
								unsigned int nCount = blockCounts[ni];
								bool isConnected = false;

								for (unsigned int p1 = 0; p1 < curCount; ++p1)
								{
									const Eigen::Vector3f& pt1 = points[sortedIndices[curOffset + p1]];
									for (unsigned int p2 = 0; p2 < nCount; ++p2)
									{
										const Eigen::Vector3f& pt2 = points[sortedIndices[nOffset + p2]];
										if ((pt1 - pt2).squaredNorm() <= sqDistThreshold)
										{
											isConnected = true;
											break;
										}
									}
									if (isConnected) break;
								}

								if (isConnected)
								{
									adj[i].push_back(ni);
								}
							}
						}
					}
				}
			});

		std::vector<std::vector<unsigned int>> clusters;
		std::vector<bool> blockVisited(numBlocks, false);

		for (size_t i = 0; i < numBlocks; ++i)
		{
			if (blockVisited[i]) continue;

			std::vector<unsigned int> currentCluster;
			std::deque<size_t> q;

			q.push_back(i);
			blockVisited[i] = true;

			while (!q.empty())
			{
				size_t curr = q.front();
				q.pop_front();

				unsigned int offset = blockOffsets[curr];
				unsigned int count = blockCounts[curr];
				for (unsigned int p = 0; p < count; ++p)
				{
					currentCluster.push_back(sortedIndices[offset + p]);
				}

				// 인접 블록 탐색 (무향 그래프처럼 처리하기 위해 양방향 확인 필요 없음, 이미 adj에 상대 인덱스가 포함됨)
				for (size_t neighbor : adj[curr])
				{
					if (!blockVisited[neighbor])
					{
						blockVisited[neighbor] = true;
						q.push_back(neighbor);
					}
				}

				// 단, adj[i].push_back(ni)만 했으므로 역방향 체크 로직이 필요할 수 있으나, 
				// 모든 i에 대해 주변 26개를 다 뒤졌으므로 adj[i]에는 모든 연결된 ni가 이미 들어가 있습니다.
			}

			if (!currentCluster.empty())
			{
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
		if (blockKeys.empty() || points.size() != normals.size())
		{
			return {};
		}

		const float sqDistThreshold = distThreshold * distThreshold;
		size_t numBlocks = blockKeys.size();

		robin_hood::unordered_flat_map<uint64_t, size_t> keyToBlockIdx;
		for (size_t i = 0; i < numBlocks; ++i)
		{
			keyToBlockIdx[blockKeys[i]] = i;
		}

		std::vector<std::vector<size_t>> adj(numBlocks);
		std::vector<size_t> blockIndices(numBlocks);
		std::iota(blockIndices.begin(), blockIndices.end(), 0);

		std::for_each(std::execution::par, blockIndices.begin(), blockIndices.end(), [&](size_t i)
			{
				uint64_t key = blockKeys[i];
				unsigned int curOffset = blockOffsets[i];
				unsigned int curCount = blockCounts[i];

				int64_t xi = (int64_t)((key >> 42) & 0x1FFFFF); if (xi & 0x100000) xi |= ~0x1FFFFF;
				int64_t yi = (int64_t)((key >> 21) & 0x1FFFFF); if (yi & 0x100000) yi |= ~0x1FFFFF;
				int64_t zi = (int64_t)(key & 0x1FFFFF); if (zi & 0x100000) zi |= ~0x1FFFFF;

				for (int64_t dz = -1; dz <= 1; ++dz)
				{
					for (int64_t dy = -1; dy <= 1; ++dy)
					{
						for (int64_t dx = -1; dx <= 1; ++dx)
						{
							if (dx == 0 && dy == 0 && dz == 0) continue;

							uint64_t neighborKey = GetKeyFromIndices(xi + dx, yi + dy, zi + dz);
							auto it = keyToBlockIdx.find(neighborKey);

							if (it != keyToBlockIdx.end())
							{
								size_t ni = it->second;
								unsigned int nOffset = blockOffsets[ni];
								unsigned int nCount = blockCounts[ni];
								bool isConnected = false;

								for (unsigned int p1 = 0; p1 < curCount; ++p1)
								{
									unsigned int idx1 = sortedIndices[curOffset + p1];
									const Eigen::Vector3f& pt1 = points[idx1];
									const Eigen::Vector3f& nm1 = normals[idx1];

									for (unsigned int p2 = 0; p2 < nCount; ++p2)
									{
										unsigned int idx2 = sortedIndices[nOffset + p2];
										if ((pt1 - points[idx2]).squaredNorm() <= sqDistThreshold)
										{
											if (std::abs(nm1.dot(normals[idx2])) >= normalThreshold)
											{
												isConnected = true;
												break;
											}
										}
									}
									if (isConnected) break;
								}

								if (isConnected)
								{
									adj[i].push_back(ni);
								}
							}
						}
					}
				}
			});

		std::vector<std::vector<unsigned int>> clusters;
		std::vector<bool> blockVisited(numBlocks, false);

		for (size_t i = 0; i < numBlocks; ++i)
		{
			if (blockVisited[i]) continue;

			std::vector<unsigned int> currentCluster;
			std::deque<size_t> q;

			q.push_back(i);
			blockVisited[i] = true;

			while (!q.empty())
			{
				size_t curr = q.front();
				q.pop_front();

				unsigned int offset = blockOffsets[curr];
				unsigned int count = blockCounts[curr];
				for (unsigned int p = 0; p < count; ++p)
				{
					currentCluster.push_back(sortedIndices[offset + p]);
				}

				for (size_t neighbor : adj[curr])
				{
					if (!blockVisited[neighbor])
					{
						blockVisited[neighbor] = true;
						q.push_back(neighbor);
					}
				}
			}

			if (!currentCluster.empty())
			{
				clusters.push_back(std::move(currentCluster));
			}
		}

		return clusters;
	}

private:
	float voxelSize = 0.1f;

	std::vector<uint64_t> blockKeys;
	std::vector<Eigen::Vector3f> blockMinCorners;
	std::vector<unsigned int> blockOffsets;
	std::vector<unsigned int> blockCounts;
	std::vector<unsigned int> sortedIndices;

	std::vector<KeyIndex> proxy;
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
