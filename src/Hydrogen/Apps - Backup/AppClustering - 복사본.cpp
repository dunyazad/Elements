#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <vector>

#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

namespace Eigen {
	template <typename Type, int Size>
	using Vector = Matrix<Type, Size, 1>;
	using Vector3b = Vector<unsigned char, 3>;
	using Vector3ui = Vector<unsigned int, 3>;
}

// Data-Oriented Design: Flat Layout Structure
struct FlatClustering
{
	float voxelSize = 0.1f;

	// Structure of Arrays (SoA) layout
	std::vector<uint64_t> blockKeys;        // Unique Voxel Keys
	std::vector<Eigen::Vector3f> blockMinCorners; // Min Corner per block
	std::vector<unsigned int> blockOffsets; // Start index in 'sortedIndices'
	std::vector<unsigned int> blockCounts;  // Number of points in the block

	std::vector<unsigned int> sortedIndices; // All point indices packed contiguously

	// Optimized key generation
	inline uint64_t GetBlockKey(const Eigen::Vector3f& point, float invVoxelSize) const
	{
		int xi = static_cast<int>(std::floor(point.x() * invVoxelSize));
		int yi = static_cast<int>(std::floor(point.y() * invVoxelSize));
		int zi = static_cast<int>(std::floor(point.z() * invVoxelSize));

		uint64_t hash = 0;
		hash ^= std::hash<int>{}(xi)+0x9e3779b9 + (hash << 6) + (hash >> 2);
		hash ^= std::hash<int>{}(yi)+0x9e3779b9 + (hash << 6) + (hash >> 2);
		hash ^= std::hash<int>{}(zi)+0x9e3779b9 + (hash << 6) + (hash >> 2);
		return hash;
	}

	void Build(const std::vector<Eigen::Vector3f>& points, float blockSize = 0.3f)
	{
		this->voxelSize = blockSize;

		// Clear previous data
		blockKeys.clear();
		blockMinCorners.clear();
		blockOffsets.clear();
		blockCounts.clear();
		sortedIndices.clear();

		if (points.empty())
		{
			return;
		}

		const size_t pointCount = points.size();
		const float invVoxelSize = 1.0f / blockSize;

		// 1. Pre-allocate memory
		struct KeyIndex {
			uint64_t key;
			unsigned int index;
		};

		std::vector<KeyIndex> proxy(pointCount);
		sortedIndices.resize(pointCount);

		// 2. Compute keys in parallel
		auto* proxyPtr = proxy.data();
		const auto* pointsPtr = points.data();

		std::vector<unsigned int> iter(pointCount);
		std::iota(iter.begin(), iter.end(), 0);

		std::for_each(std::execution::par, iter.begin(), iter.end(), [=](unsigned int i) {
			proxyPtr[i] = { GetBlockKey(pointsPtr[i], invVoxelSize), i };
			});

		// 3. Sort by key in parallel (O(N log N))
		std::sort(std::execution::par, proxy.begin(), proxy.end(),
			[](const KeyIndex& a, const KeyIndex& b) {
				return a.key < b.key; // Fixed: .first -> .key
			});

		// 4. Linear Scan to build Flat Structure
		size_t estimatedBlocks = pointCount / 10 + 1;
		blockKeys.reserve(estimatedBlocks);
		blockMinCorners.reserve(estimatedBlocks);
		blockOffsets.reserve(estimatedBlocks);
		blockCounts.reserve(estimatedBlocks);

		if (pointCount > 0)
		{
			// First block initialization
			uint64_t currentKey = proxy[0].key; // Fixed: .first -> .key
			unsigned int currentOffset = 0;

			blockKeys.push_back(currentKey);
			blockOffsets.push_back(currentOffset);

			// Calculate minCorner for the first block
			{
				const Eigen::Vector3f& p = points[proxy[0].index]; // Fixed: .second -> .index
				int xi = static_cast<int>(std::floor(p.x() * invVoxelSize));
				int yi = static_cast<int>(std::floor(p.y() * invVoxelSize));
				int zi = static_cast<int>(std::floor(p.z() * invVoxelSize));
				blockMinCorners.push_back(Eigen::Vector3f(xi * blockSize, yi * blockSize, zi * blockSize));
			}

			// Flatten copy of indices
			for (size_t i = 0; i < pointCount; ++i)
			{
				sortedIndices[i] = proxy[i].index; // Fixed: .second -> .index

				// Detect block change
				if (proxy[i].key != currentKey) // Fixed: .first -> .key
				{
					// Close previous block
					unsigned int count = static_cast<unsigned int>(i) - currentOffset;
					blockCounts.push_back(count);

					// Start new block
					currentKey = proxy[i].key; // Fixed: .first -> .key
					currentOffset = static_cast<unsigned int>(i);

					blockKeys.push_back(currentKey);
					blockOffsets.push_back(currentOffset);

					// Calculate minCorner for the new block
					const Eigen::Vector3f& p = points[proxy[i].index]; // Fixed: .second -> .index
					int xi = static_cast<int>(std::floor(p.x() * invVoxelSize));
					int yi = static_cast<int>(std::floor(p.y() * invVoxelSize));
					int zi = static_cast<int>(std::floor(p.z() * invVoxelSize));
					blockMinCorners.push_back(Eigen::Vector3f(xi * blockSize, yi * blockSize, zi * blockSize));
				}
			}

			// Close last block
			blockCounts.push_back(static_cast<unsigned int>(pointCount) - currentOffset);
		}
	}
};

class AppClustering : public App
{
public:
	virtual void Execute() override
	{
		PLYFormat ply;
		ply.Deserialize("D:\\Resources\\Debug\\3D\\BasePoints.ply");

		VD::AddSphereBatch(
			"PointCloud",
			ply.GetPoints(),
			ply.GetNormals(),
			0.02f,
			ply.GetColors());

		FlatClustering clustering;
		TS(Build);
		clustering.Build(ply.GetPoints(), 0.3f);
		TE(Build);
	}
};

REGISTER_APP(AppClustering, "AppClustering");
