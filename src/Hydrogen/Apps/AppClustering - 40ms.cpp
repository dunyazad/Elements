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

	// 최적화된 비트 패킹 키 생성 (std::hash보다 훨씬 빠름)
	inline uint64_t GetBlockKey(const Eigen::Vector3f& point, float invVoxelSize) const
	{
		// 음수 영역 대응을 위해 floor 후 int64_t 변환
		int64_t xi = static_cast<int64_t>(std::floor(point.x() * invVoxelSize));
		int64_t yi = static_cast<int64_t>(std::floor(point.y() * invVoxelSize));
		int64_t zi = static_cast<int64_t>(std::floor(point.z() * invVoxelSize));

		// 각 축당 21비트 할당 (약 +-100만 보셀 커버 가능)
		// bitwise 연산은 정렬 시 정수 비교 효율을 극대화함
		return ((uint64_t)(xi & 0x1FFFFF) << 42) |
			((uint64_t)(yi & 0x1FFFFF) << 21) |
			((uint64_t)(zi & 0x1FFFFF));
	}

	void Build(const std::vector<Eigen::Vector3f>& points, float blockSize = 0.3f)
	{
		this->voxelSize = blockSize;

		if (points.empty())
		{
			blockKeys.clear();
			blockMinCorners.clear();
			blockOffsets.clear();
			blockCounts.clear();
			sortedIndices.clear();
			return;
		}

		const size_t pointCount = points.size();
		const float invVoxelSize = 1.0f / blockSize;

		// 1. 임시 데이터 구조 (정렬용)
		struct KeyIndex {
			uint64_t key;
			unsigned int index;
		};
		std::vector<KeyIndex> proxy(pointCount);

		// 2. 키 생성 및 인덱스 맵핑 (Parallel)
		// std::iota를 사용하지 않고 바로 인덱스를 주입하여 overhead 감소
		const Eigen::Vector3f* pointsPtr = points.data();
		KeyIndex* proxyPtr = proxy.data();

		std::for_each(std::execution::par, proxy.begin(), proxy.end(), [this, pointsPtr, proxyPtr, invVoxelSize](KeyIndex& item) {
			size_t i = &item - proxyPtr;
			item.key = GetBlockKey(pointsPtr[i], invVoxelSize);
			item.index = static_cast<unsigned int>(i);
			});

		// 3. 고속 병렬 정렬
		// 64비트 정수 키 비교는 매우 빠르며 캐시 친화적임
		std::sort(std::execution::par, proxy.begin(), proxy.end(), [](const KeyIndex& a, const KeyIndex& b) {
			return a.key < b.key;
			});

		// 4. 결과 컨테이너 초기화
		blockKeys.clear();
		blockMinCorners.clear();
		blockOffsets.clear();
		blockCounts.clear();
		sortedIndices.resize(pointCount);

		// 블록 수 예측을 통한 사전 예약 (realloc 방지)
		size_t estimatedBlocks = std::min(pointCount, (size_t)1024);
		blockKeys.reserve(estimatedBlocks);
		blockMinCorners.reserve(estimatedBlocks);
		blockOffsets.reserve(estimatedBlocks);
		blockCounts.reserve(estimatedBlocks);

		// 5. 선형 스캔으로 플랫 구조 빌드
		uint64_t currentKey = proxy[0].key;
		unsigned int currentOffset = 0;

		auto AddBlockInfo = [&](uint64_t key, unsigned int offset, unsigned int count) {
			blockKeys.push_back(key);
			blockOffsets.push_back(offset);
			blockCounts.push_back(count);

			// 키로부터 좌표 복원 (MinCorner)
			int64_t xi = (int64_t)((key >> 42) & 0x1FFFFF);
			if (xi & 0x100000) xi |= ~0x1FFFFF; // 21비트 부호 확장
			int64_t yi = (int64_t)((key >> 21) & 0x1FFFFF);
			if (yi & 0x100000) yi |= ~0x1FFFFF;
			int64_t zi = (int64_t)(key & 0x1FFFFF);
			if (zi & 0x100000) zi |= ~0x1FFFFF;

			blockMinCorners.push_back(Eigen::Vector3f(
				static_cast<float>(xi) * blockSize,
				static_cast<float>(yi) * blockSize,
				static_cast<float>(zi) * blockSize));
			};

		for (size_t i = 0; i < pointCount; ++i)
		{
			sortedIndices[i] = proxy[i].index;

			if (proxy[i].key != currentKey)
			{
				unsigned int count = static_cast<unsigned int>(i) - currentOffset;
				AddBlockInfo(currentKey, currentOffset, count);

				currentKey = proxy[i].key;
				currentOffset = static_cast<unsigned int>(i);
			}
		}

		// 마지막 블록 닫기
		AddBlockInfo(currentKey, currentOffset, static_cast<unsigned int>(pointCount) - currentOffset);
	}
};

class AppClustering : public App
{
public:
	virtual void Execute() override
	{
		PLYFormat ply;
		// 경로 및 파일 존재 여부 확인 필요
		ply.Deserialize("D:\\Resources\\Debug\\3D\\BasePoints.ply");

		if (ply.GetPoints().empty()) return;

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

		// 성능 확인을 위한 로그 출력 (선택 사항)
		// printf("Blocks Created: %zu\n", clustering.blockKeys.size());
	}
};

REGISTER_APP(AppClustering, "AppClustering");