#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

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

	// 멤버 변수로 유지하여 메모리 재사용 (Zero Allocation)
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

		// 1. 메모리 재확보 (크기가 늘어날 때만 재할당)
		if (proxy.size() != pointCount) proxy.resize(pointCount);
		if (sortedIndices.size() != pointCount) sortedIndices.resize(pointCount);

		const Eigen::Vector3f* pPoints = points.data();
		KeyIndex* pProxy = proxy.data();

		// 2. 키 생성 및 인덱스 할당 (Parallel)
		std::for_each(std::execution::par, pProxy, pProxy + pointCount, [this, pPoints, pProxy, invVoxelSize](KeyIndex& item) {
			size_t i = &item - pProxy;
			item.key = GetBlockKey(pPoints[i], invVoxelSize);
			item.index = static_cast<unsigned int>(i);
			});

		// 3. 병렬 정렬 (타입 호환성 해결을 위해 직접 par 전달)
		// std::sort 내에서 KeyIndex(12~16 bytes) 정렬은 매우 효율적임
		std::sort(std::execution::par, proxy.begin(), proxy.end(), [](const KeyIndex& a, const KeyIndex& b) {
			return a.key < b.key;
			});

		// 4. 결과 컨테이너 초기화 (Capacity 유지)
		blockKeys.clear();
		blockMinCorners.clear();
		blockOffsets.clear();
		blockCounts.clear();

		// 5. 선형 스캔 (순차 접근 최적화)
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

				// 비트 연산을 통한 MinCorner 복원
				int64_t xi = (int64_t)((currentKey >> 42) & 0x1FFFFF);
				if (xi & 0x100000) xi |= ~0x1FFFFF;
				int64_t yi = (int64_t)((currentKey >> 21) & 0x1FFFFF);
				if (yi & 0x100000) yi |= ~0x1FFFFF;
				int64_t zi = (int64_t)(currentKey & 0x1FFFFF);
				if (zi & 0x100000) zi |= ~0x1FFFFF;

				blockMinCorners.emplace_back(
					static_cast<float>(xi) * blockSize,
					static_cast<float>(yi) * blockSize,
					static_cast<float>(zi) * blockSize);

				currentKey = proxy[i].key;
				currentOffset = static_cast<unsigned int>(i);
			}
		}

		// 마지막 블록 닫기
		blockKeys.push_back(currentKey);
		blockOffsets.push_back(currentOffset);
		blockCounts.push_back(static_cast<unsigned int>(pointCount) - currentOffset);

		int64_t xi = (int64_t)((currentKey >> 42) & 0x1FFFFF);
		if (xi & 0x100000) xi |= ~0x1FFFFF;
		int64_t yi = (int64_t)((currentKey >> 21) & 0x1FFFFF);
		if (yi & 0x100000) yi |= ~0x1FFFFF;
		int64_t zi = (int64_t)(currentKey & 0x1FFFFF);
		if (zi & 0x100000) zi |= ~0x1FFFFF;

		blockMinCorners.emplace_back(
			static_cast<float>(xi) * blockSize,
			static_cast<float>(yi) * blockSize,
			static_cast<float>(zi) * blockSize);
	}
};

class AppClustering : public App
{
public:
	virtual void Execute() override
	{
		PLYFormat ply;
		ply.Deserialize("D:\\Resources\\Debug\\3D\\BasePoints.ply");

		if (ply.GetPoints().empty()) return;

		VD::AddSphereBatch(
			"PointCloud",
			ply.GetPoints(),
			ply.GetNormals(),
			0.02f,
			ply.GetColors());

		// static을 사용하여 메모리 풀 효과 적용
		static FlatClustering clustering;

		TS(Build);
		clustering.Build(ply.GetPoints(), 0.3f);
		TE(Build);
	}
};

REGISTER_APP(AppClustering, "AppClustering");