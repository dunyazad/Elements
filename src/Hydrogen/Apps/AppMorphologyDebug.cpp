#include "Apps.h"
#include <vector>
#include <unordered_set>
#include <queue>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdio>

#include <Helium/Serialization.hpp>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

namespace Eigen {
	template <typename Type, int Size>
	using Vector = Matrix<Type, Size, 1>;
}

class MorphEngine
{
public:
	struct VoxelKey {
		int x, y, z;
		bool operator==(const VoxelKey& other) const {
			return x == other.x && y == other.y && z == other.z;
		}
	};

	struct KeyHasher {
		std::size_t operator()(const VoxelKey& k) const {
			return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^ (std::hash<int>()(k.z) << 1);
		}
	};

	using VoxelSet = std::unordered_set<VoxelKey, KeyHasher>;

	static VoxelSet Voxelize(const std::vector<Eigen::Vector3f>& points, float voxelSize)
	{
		VoxelSet voxels;
		if (points.empty()) return voxels;

		float invSize = 1.0f / voxelSize;
		for (const auto& p : points) {
			int x = static_cast<int>(std::floor(p.x() * invSize));
			int y = static_cast<int>(std::floor(p.y() * invSize));
			int z = static_cast<int>(std::floor(p.z() * invSize));
			voxels.insert({ x, y, z });
		}
		return voxels;
	}

	// 팽창 (Dilation)
	static VoxelSet Dilate(const VoxelSet& inputVoxels, int iterations)
	{
		VoxelSet current = inputVoxels;
		for (int i = 0; i < iterations; ++i) {
			VoxelSet next = current;
			for (const auto& key : current) {
				const int dx[] = { -1, 1, 0, 0, 0, 0 };
				const int dy[] = { 0, 0, -1, 1, 0, 0 };
				const int dz[] = { 0, 0, 0, 0, -1, 1 };
				for (int k = 0; k < 6; ++k) {
					next.insert({ key.x + dx[k], key.y + dy[k], key.z + dz[k] });
				}
			}
			current = next;
		}
		return current;
	}

	// 침식 (Erosion)
	static VoxelSet Erode(const VoxelSet& inputVoxels, int iterations)
	{
		VoxelSet current = inputVoxels;
		for (int i = 0; i < iterations; ++i) {
			VoxelSet next;
			for (const auto& key : current) {
				bool isInterior = true;
				const int dx[] = { -1, 1, 0, 0, 0, 0 };
				const int dy[] = { 0, 0, -1, 1, 0, 0 };
				const int dz[] = { 0, 0, 0, 0, -1, 1 };

				for (int k = 0; k < 6; ++k) {
					VoxelKey n = { key.x + dx[k], key.y + dy[k], key.z + dz[k] };
					if (current.find(n) == current.end()) {
						isInterior = false; break;
					}
				}
				// 6면이 모두 막혀있어야(내부여야) 살아남음
				if (isInterior) next.insert(key);
			}
			current = next;
			if (current.empty()) break;
		}
		return current;
	}

	static VoxelSet KeepLargestComponent(const VoxelSet& voxels)
	{
		if (voxels.empty()) return {};
		VoxelSet visited;
		VoxelSet largestCluster;

		for (const auto& startVoxel : voxels) {
			if (visited.find(startVoxel) != visited.end()) continue;

			VoxelSet currentCluster;
			std::queue<VoxelKey> q;
			q.push(startVoxel);
			visited.insert(startVoxel);
			currentCluster.insert(startVoxel);

			while (!q.empty()) {
				VoxelKey curr = q.front(); q.pop();
				const int dx[] = { -1, 1, 0, 0, 0, 0 };
				const int dy[] = { 0, 0, -1, 1, 0, 0 };
				const int dz[] = { 0, 0, 0, 0, -1, 1 };

				for (int k = 0; k < 6; ++k) {
					VoxelKey n = { curr.x + dx[k], curr.y + dy[k], curr.z + dz[k] };
					if (voxels.find(n) != voxels.end() && visited.find(n) == visited.end()) {
						visited.insert(n);
						currentCluster.insert(n);
						q.push(n);
					}
				}
			}
			if (currentCluster.size() > largestCluster.size()) {
				largestCluster = currentCluster;
			}
		}
		return largestCluster;
	}

	static std::vector<Eigen::Vector3f> GetPointsInVoxels(
		const std::vector<Eigen::Vector3f>& points,
		const VoxelSet& validVoxels,
		float voxelSize)
	{
		std::vector<Eigen::Vector3f> result;
		float invSize = 1.0f / voxelSize;
		for (const auto& p : points) {
			int x = static_cast<int>(std::floor(p.x() * invSize));
			int y = static_cast<int>(std::floor(p.y() * invSize));
			int z = static_cast<int>(std::floor(p.z() * invSize));

			// 주변 1칸까지 여유 있게 검색
			bool found = false;
			for (int dz = -1; dz <= 1; ++dz) {
				for (int dy = -1; dy <= 1; ++dy) {
					for (int dx = -1; dx <= 1; ++dx) {
						if (validVoxels.find({ x + dx, y + dy, z + dz }) != validVoxels.end()) {
							found = true; goto END_SEARCH;
						}
					}
				}
			}
		END_SEARCH:
			if (found) result.push_back(p);
		}
		return result;
	}
};

class AppMorphologyDebug : public App
{
public:
	virtual void Initialize() override
	{
	}

	void DrawVoxels(const MorphEngine::VoxelSet& voxels, float voxelSize, const Eigen::Vector4f& color, const std::string& name)
	{
		std::vector<Eigen::Vector3f> centers;
		// 너무 많으면 렌더링 느려지므로 샘플링
		int skip = voxels.size() > 50000 ? 10 : 1;
		int count = 0;
		for (const auto& v : voxels) {
			if (count++ % skip == 0) {
				centers.push_back(Eigen::Vector3f(
					(v.x + 0.5f) * voxelSize,
					(v.y + 0.5f) * voxelSize,
					(v.z + 0.5f) * voxelSize
				));
			}
		}
		if (!centers.empty())
			VD::AddSphereBatch(name, centers, voxelSize * 0.4f, color);
	}

	virtual void Execute() override
	{
		PLYFormat ply;
		ply.Deserialize("D:\\Resources\\Default\\BasePoints.ply");
		if (ply.GetPoints().empty()) {
			printf("Error: No Points.\n");
			return;
		}

		TS(Total_Morphology);

		// ==========================================
		// [핵심 전략: 선-팽창 후-침식]
		// 1. preDilate로 몸집을 불려서 '속이 찬 상태'로 만듭니다.
		// 2. erodeIter로 다시 깎아내면, 본체는 남고 얇은 목만 끊어집니다.
		// ==========================================

		float voxelSize = 0.05f;  // 5cm (정밀도 유지)

		int preDilate = 2;        // [중요] 먼저 2겹 살찌우기 (빈 껍데기 방지)
		int erodeIter = 3;        // 살찌운 것보다 1번 더 깎아서 연결 끊기 (Net: -1 효과)
		int postDilate = 2;       // 끊어진 본체를 다시 원래 크기로 복구

		printf("\n=== Morphology (Solidify Strategy) ===\n");
		printf("Settings: Voxel=%.3f, PreDilate=%d, Erode=%d, PostDilate=%d\n", voxelSize, preDilate, erodeIter, postDilate);

		// 1. 복셀화
		auto voxels = MorphEngine::Voxelize(ply.GetPoints(), voxelSize);
		printf("[1] Init Voxels: %zu\n", voxels.size());

		// 2. 선-팽창 (Solidify) - 껍데기를 두껍게 만듦
		TS(PreDilate);
		auto solidVoxels = MorphEngine::Dilate(voxels, preDilate);
		TE(PreDilate);
		printf("[2] Solidified Voxels: %zu (Thickened)\n", solidVoxels.size());

		// 3. 침식 (Cut) - 두꺼워진 상태에서 깎음
		TS(Erode);
		auto erodedVoxels = MorphEngine::Erode(solidVoxels, erodeIter);
		TE(Erode);
		printf("[3] Eroded Voxels: %zu\n", erodedVoxels.size());

		// 만약 0이 나온다면 voxelSize를 0.05 -> 0.08로 조금 키워보세요.
		if (erodedVoxels.empty()) {
			printf("!!! Failed: All voxels eroded. Try increasing VoxelSize slightly.\n");
			return;
		}

		// 빨간색: 절단된 핵심 뼈대 (여기서 연결이 끊어져 있어야 성공)
		DrawVoxels(erodedVoxels, voxelSize, { 1.0f, 0.0f, 0.0f, 0.5f }, "Debug_Core");

		// 4. 가장 큰 덩어리만 선택
		auto coreVoxels = MorphEngine::KeepLargestComponent(erodedVoxels);
		printf("[4] Core Voxels: %zu\n", coreVoxels.size());

		// 5. 후-팽창 (Restore) - 본체 복구
		auto finalVoxels = MorphEngine::Dilate(coreVoxels, postDilate);
		printf("[5] Final Voxels: %zu\n", finalVoxels.size());

		// 파란색: 최종 복원된 본체
		DrawVoxels(finalVoxels, voxelSize, { 0.0f, 0.0f, 1.0f, 0.5f }, "Debug_Final");

		// 6. 포인트 필터링
		auto finalPoints = MorphEngine::GetPointsInVoxels(ply.GetPoints(), finalVoxels, voxelSize);
		printf("[6] Result Points: %zu\n", finalPoints.size());

		TE(Total_Morphology);

		VD::AddSphereBatch("FinalResult", finalPoints, 0.03f, { 0.0f, 1.0f, 0.0f, 1.0f });
		printf("=== Done ===\n\n");
	}
};

REGISTER_APP(AppMorphologyDebug, "AppMorphologyDebug");
