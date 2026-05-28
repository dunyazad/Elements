#include "Apps.h"
#include <vector>
#include <unordered_set>
#include <queue>
#include <iostream>
#include <algorithm>
#include <cmath>

#include <Helium/Serialization.hpp>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

namespace Eigen {
	template <typename Type, int Size>
	using Vector = Matrix<Type, Size, 1>;
	using Vector3i = Vector<int, 3>;
}

// -------------------------------------------------------------
// Morphological Filtering Engine
// -------------------------------------------------------------
class MorphEngine
{
public:
	// 복셀 해싱을 위한 구조체
	struct VoxelKey {
		int x, y, z;

		bool operator==(const VoxelKey& other) const {
			return x == other.x && y == other.y && z == other.z;
		}
	};

	struct KeyHasher {
		std::size_t operator()(const VoxelKey& k) const {
			// 간단한 해시 함수
			return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^ (std::hash<int>()(k.z) << 1);
		}
	};

	using VoxelSet = std::unordered_set<VoxelKey, KeyHasher>;

	// 1. 포인트 클라우드를 복셀 그리드(Set)로 변환
	static VoxelSet Voxelize(const std::vector<Eigen::Vector3f>& points, float voxelSize)
	{
		VoxelSet voxels;
		float invSize = 1.0f / voxelSize;

		for (const auto& p : points)
		{
			int x = static_cast<int>(std::floor(p.x() * invSize));
			int y = static_cast<int>(std::floor(p.y() * invSize));
			int z = static_cast<int>(std::floor(p.z() * invSize));
			voxels.insert({ x, y, z });
		}
		return voxels;
	}

	// 2. 침식 연산 (Erosion): 표면을 한 겹 깎아냄
	static VoxelSet Erode(const VoxelSet& inputVoxels, int iterations = 1)
	{
		VoxelSet currentVoxels = inputVoxels;

		for (int i = 0; i < iterations; ++i)
		{
			VoxelSet nextVoxels;
			// 모든 복셀을 순회하며 '내부'에 있는 것만 남김
			for (const auto& key : currentVoxels)
			{
				bool isInterior = true;

				// 6-connectivity (상하좌우전후) 이웃 확인
				// 이웃 중 하나라도 비어있으면(표면이면) 제거 대상
				const int dx[] = { -1, 1, 0, 0, 0, 0 };
				const int dy[] = { 0, 0, -1, 1, 0, 0 };
				const int dz[] = { 0, 0, 0, 0, -1, 1 };

				for (int k = 0; k < 6; ++k)
				{
					VoxelKey neighbor = { key.x + dx[k], key.y + dy[k], key.z + dz[k] };
					if (currentVoxels.find(neighbor) == currentVoxels.end())
					{
						isInterior = false;
						break;
					}
				}

				if (isInterior) {
					nextVoxels.insert(key);
				}
			}
			currentVoxels = nextVoxels;
			if (currentVoxels.empty()) break;
		}
		return currentVoxels;
	}

	// 3. 가장 큰 덩어리만 추출 (Connected Component Analysis)
	static VoxelSet KeepLargestComponent(const VoxelSet& voxels)
	{
		if (voxels.empty()) return {};

		VoxelSet visited;
		VoxelSet largestCluster;
		size_t maxCount = 0;

		for (const auto& startVoxel : voxels)
		{
			if (visited.find(startVoxel) != visited.end()) continue;

			// BFS 탐색
			VoxelSet currentCluster;
			std::queue<VoxelKey> q;
			q.push(startVoxel);
			visited.insert(startVoxel);
			currentCluster.insert(startVoxel);

			while (!q.empty())
			{
				VoxelKey curr = q.front();
				q.pop();

				const int dx[] = { -1, 1, 0, 0, 0, 0 };
				const int dy[] = { 0, 0, -1, 1, 0, 0 };
				const int dz[] = { 0, 0, 0, 0, -1, 1 };

				for (int k = 0; k < 6; ++k)
				{
					VoxelKey neighbor = { curr.x + dx[k], curr.y + dy[k], curr.z + dz[k] };

					// 존재하는 복셀이고, 아직 방문 안 했으면
					if (voxels.find(neighbor) != voxels.end() && visited.find(neighbor) == visited.end())
					{
						visited.insert(neighbor);
						currentCluster.insert(neighbor);
						q.push(neighbor);
					}
				}
			}

			if (currentCluster.size() > maxCount)
			{
				maxCount = currentCluster.size();
				largestCluster = currentCluster;
			}
		}

		return largestCluster;
	}

	// 4. 복셀 마스크를 이용해 원본 포인트 필터링
	// (살아남은 복셀 근처에 있는 포인트만 살림)
	static std::vector<Eigen::Vector3f> FilterPointsByVoxels(
		const std::vector<Eigen::Vector3f>& points,
		const VoxelSet& validVoxels,
		float voxelSize,
		int expansion = 1) // 침식으로 줄어든 부피를 보정하기 위한 확장 범위
	{
		std::vector<Eigen::Vector3f> result;
		float invSize = 1.0f / voxelSize;

		for (const auto& p : points)
		{
			int x = static_cast<int>(std::floor(p.x() * invSize));
			int y = static_cast<int>(std::floor(p.y() * invSize));
			int z = static_cast<int>(std::floor(p.z() * invSize));

			// 현재 포인트가 유효한 복셀(혹은 그 주변)에 속하는지 확인
			bool keep = false;

			// expansion 범위 내에 유효 복셀이 하나라도 있으면 유지
			for (int dz = -expansion; dz <= expansion; ++dz) {
				for (int dy = -expansion; dy <= expansion; ++dy) {
					for (int dx = -expansion; dx <= expansion; ++dx) {
						if (validVoxels.find({ x + dx, y + dy, z + dz }) != validVoxels.end()) {
							keep = true;
							goto FOUND;
						}
					}
				}
			}

		FOUND:
			if (keep) {
				result.push_back(p);
			}
		}
		return result;
	}
};

// -------------------------------------------------------------
// App Class
// -------------------------------------------------------------
class AppMorphology : public App
{
public:
	virtual void Execute() override
	{
		// 1. 데이터 로드
		PLYFormat ply;
		ply.Deserialize("D:\\Resources\\Default\\BasePoints.ply"); // 파일 경로
		if (ply.GetPoints().empty()) return;

		// 원본 시각화 (빨간색, 반투명)
		VD::AddSphereBatch("Original", ply.GetPoints(), 0.02f, { 1.0f, 0.0f, 0.0f, 0.2f });

		TS(Total_Morphology);

		// [설정 파라미터]
		float voxelSize = 0.3f; // 1. 복셀 크기 (너무 작으면 구멍이 생김, 적당히 크게)
		int erodeIter = 2;      // 2. 깎아낼 횟수 (이 값을 늘리면 더 깊게 파고들어 연결을 끊음)

		// Step 1. Voxelization
		auto voxels = MorphEngine::Voxelize(ply.GetPoints(), voxelSize);

		// Step 2. Erosion (침식) - 여기서 연결 부위가 끊어짐!
		// 튀어나온 부분이 본체와 가늘게 연결되어 있다면 여기서 분리됩니다.
		auto erodedVoxels = MorphEngine::Erode(voxels, erodeIter);

		// Step 3. Largest Component (가장 큰 덩어리 찾기)
		// 연결이 끊어진 작은 조각(튀어나온 부분)들은 여기서 탈락합니다.
		auto coreVoxels = MorphEngine::KeepLargestComponent(erodedVoxels);

		// Step 4. Restore Points (포인트 복원)
		// 남은 '핵심 복셀' 주변의 원본 포인트들을 다시 가져옵니다.
		// expansion을 erodeIter와 비슷하게 주면 원래 크기로 복구됩니다.
		auto finalPoints = MorphEngine::FilterPointsByVoxels(ply.GetPoints(), coreVoxels, voxelSize, erodeIter);

		TE(Total_Morphology);

		// 결과 시각화 (녹색, 불투명)
		VD::AddSphereBatch("Result_Cleaned", finalPoints, 0.03f, { 0.0f, 1.0f, 0.0f, 1.0f });

		// 정보 출력
		//VD::AddText("Original: " + std::to_string(ply.GetPoints().size()), { 10, 10 }, { 1,0,0,1 });
		//VD::AddText("Filtered: " + std::to_string(finalPoints.size()), { 10, 30 }, { 0,1,0,1 });

		std::cout << "Original Points: " << ply.GetPoints().size() << std::endl;
		std::cout << "Filtered Points: " << finalPoints.size() << std::endl;
	}
};

REGISTER_APP(AppMorphology, "AppMorphology");
