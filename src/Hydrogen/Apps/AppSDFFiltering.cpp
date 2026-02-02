#include "Apps.h"

#include <execution>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

namespace Eigen {
	template <typename Type, int Size>
	using Vector = Matrix<Type, Size, 1>;
	using Vector3i = Vector<int, 3>;
}

// =============================================================
// SDF 필터링 엔진
// =============================================================
class SDFEngine
{
public:
	struct Grid
	{
		Eigen::Vector3i dim;       // 그리드 해상도 (x, y, z)
		Eigen::Vector3f minBound;  // 그리드 시작 위치
		float voxelSize;           // 복셀 크기
		std::vector<float> data;   // 거리 데이터 (Distance Field)

		size_t GetIdx(int x, int y, int z) const {
			return (size_t)z * dim.y() * dim.x() + (size_t)y * dim.x() + x;
		}

		bool IsValid(int x, int y, int z) const {
			return x >= 0 && x < dim.x() && y >= 0 && y < dim.y() && z >= 0 && z < dim.z();
		}

		Eigen::Vector3f GridToWorld(int x, int y, int z) const {
			return minBound + Eigen::Vector3f((float)x, (float)y, (float)z) * voxelSize;
		}

		Eigen::Vector3i WorldToGrid(const Eigen::Vector3f& pos) const {
			return ((pos - minBound) / voxelSize).cast<int>();
		}
	};

	// 1. 그리드 생성 및 거리장 계산 (Splatting)
	static Grid BuildField(const std::vector<Eigen::Vector3f>& points, float voxelSize, float padding = 5.0f)
	{
		Grid grid;
		if (points.empty()) return grid;

		// AABB 계산
		Eigen::Vector3f minB = points[0];
		Eigen::Vector3f maxB = points[0];
		for (const auto& p : points) {
			minB = minB.cwiseMin(p);
			maxB = maxB.cwiseMax(p);
		}

		// 여유 공간 추가
		float padDist = voxelSize * padding;
		grid.minBound = minB - Eigen::Vector3f::Constant(padDist);
		maxB += Eigen::Vector3f::Constant(padDist);
		grid.voxelSize = voxelSize;
		grid.dim = ((maxB - grid.minBound) / voxelSize).cast<int>() + Eigen::Vector3i::Constant(1);

		size_t totalSize = (size_t)grid.dim.x() * grid.dim.y() * grid.dim.z();

		// 초기화: 아주 큰 값으로 설정
		grid.data.assign(totalSize, 100.0f);

		// Distance Field 구축 (Splatting)
		// 각 포인트 주변의 복셀에 거리를 기록합니다.
		int range = 2; // 영향 범위 (복셀 단위)

		for (const auto& p : points)
		{
			Eigen::Vector3i center = grid.WorldToGrid(p);

			for (int z = -range; z <= range; ++z) {
				for (int y = -range; y <= range; ++y) {
					for (int x = -range; x <= range; ++x) {
						Eigen::Vector3i cur = center + Eigen::Vector3i(x, y, z);

						if (grid.IsValid(cur.x(), cur.y(), cur.z()))
						{
							float dist = (grid.GridToWorld(cur.x(), cur.y(), cur.z()) - p).norm();
							size_t idx = grid.GetIdx(cur.x(), cur.y(), cur.z());

							// 최소 거리 유지 (UDF 특징)
							if (dist < grid.data[idx]) {
								grid.data[idx] = dist;
							}
						}
					}
				}
			}
		}

		return grid;
	}

	// 2. 그리드 스무딩 (노이즈 및 스파이크 제거의 핵심)
	static void SmoothField(Grid& grid, int iterations = 1)
	{
		std::vector<float> tempBuffer = grid.data;

		for (int iter = 0; iter < iterations; ++iter)
		{
			// 병렬 처리로 Box Blur 적용
			// Z축 기준으로 루프를 나누어 처리
#pragma omp parallel for
			for (int z = 1; z < grid.dim.z() - 1; ++z)
			{
				for (int y = 1; y < grid.dim.y() - 1; ++y)
				{
					for (int x = 1; x < grid.dim.x() - 1; ++x)
					{
						size_t idx = grid.GetIdx(x, y, z);

						// 표면 근처가 아니면 연산 스킵 (최적화)
						if (grid.data[idx] > grid.voxelSize * 3.0f) continue;

						float sum = 0.0f;
						int count = 0;

						// 3x3x3 이웃 평균
						for (int kz = -1; kz <= 1; ++kz) {
							for (int ky = -1; ky <= 1; ++ky) {
								for (int kx = -1; kx <= 1; ++kx) {
									size_t nIdx = grid.GetIdx(x + kx, y + ky, z + kz);
									// 유효한 거리 값이 있는 경우만 평균에 포함
									if (grid.data[nIdx] < 50.0f) {
										sum += grid.data[nIdx];
										count++;
									}
								}
							}
						}

						if (count > 0) {
							tempBuffer[idx] = sum / (float)count;
						}
					}
				}
			}
			grid.data = tempBuffer;
		}
	}

	// 3. 포인트 및 법선 복원 (Isosurface Sampling)
	static void ExtractSurface(
		const Grid& grid,
		std::vector<Eigen::Vector3f>& outPoints,
		std::vector<Eigen::Vector3f>& outNormals,
		float isoLevelRatio = 0.5f)
	{
		outPoints.clear();
		outNormals.clear();

		float isoLevel = grid.voxelSize * isoLevelRatio; // 표면으로 간주할 거리 임계값

		for (int z = 1; z < grid.dim.z() - 1; ++z) {
			for (int y = 1; y < grid.dim.y() - 1; ++y) {
				for (int x = 1; x < grid.dim.x() - 1; ++x) {

					size_t idx = grid.GetIdx(x, y, z);
					float val = grid.data[idx];

					// 표면 근처의 복셀만 선택
					if (val < isoLevel)
					{
						// 위치 복원: 복셀 중심
						Eigen::Vector3f pos = grid.GridToWorld(x, y, z);
						outPoints.push_back(pos);

						// 법선 복원: SDF의 기울기(Gradient) 계산
						// Gradient = ( dF/dx, dF/dy, dF/dz )
						float dx = grid.data[grid.GetIdx(x + 1, y, z)] - grid.data[grid.GetIdx(x - 1, y, z)];
						float dy = grid.data[grid.GetIdx(x, y + 1, z)] - grid.data[grid.GetIdx(x, y - 1, z)];
						float dz = grid.data[grid.GetIdx(x, y, z + 1)] - grid.data[grid.GetIdx(x, y, z - 1)];

						Eigen::Vector3f normal(dx, dy, dz);

						// UDF에서는 Gradient가 표면 밖으로 향하므로 정규화만 하면 됨
						if (normal.squaredNorm() > 1e-6f) {
							normal.normalize();
						}
						else {
							normal = Eigen::Vector3f::UnitY();
						}
						outNormals.push_back(normal);
					}
				}
			}
		}
	}
};

// =============================================================
// AppSDFFiltering 클래스
// =============================================================
class AppSDFFiltering : public App
{
public:
	virtual void Execute() override
	{
		// 1. 데이터 로드
		PLYFormat ply;
		ply.Deserialize("D:\\Resources\\Debug\\3D\\BasePoints.ply");

		std::vector<Eigen::Vector3f> rawPoints = ply.GetPoints();
		if (rawPoints.empty()) {
			//VD::AddText("No points loaded.", { 10, 10 }, { 1, 0, 0, 1 });
			printf("No points loaded.\n");
			return;
		}

		// 원본 시각화 (빨간색, 투명하게)
		VD::AddSphereBatch("Original", rawPoints, 0.05f, { 1.0f, 1.0f, 1.0f, 0.3f });

		TS(Total_SDF_Process);

		// ---------------------------------------------------------
		// [Step 1] SDF 파라미터 설정
		// ---------------------------------------------------------
		// voxelSize: 작을수록 정교하지만 메모리를 많이 씀. (0.1f ~ 0.3f 추천)
		// smoothIter: 반복 횟수가 많을수록 뾰족한게 많이 사라지고 둥글어짐.
		float voxelSize = 0.1f;
		int smoothIter = 5;

		// ---------------------------------------------------------
		// [Step 2] 그리드 생성 (Distance Field 구축)
		// ---------------------------------------------------------
		TS(SDF_Build);
		SDFEngine::Grid grid = SDFEngine::BuildField(rawPoints, voxelSize);
		TE(SDF_Build);

		// ---------------------------------------------------------
		// [Step 3] 스무딩 (스파이크 제거 단계)
		// ---------------------------------------------------------
		TS(SDF_Smooth);
		// 이 과정에서 얇은 스파이크는 주변 값과 섞이면서 사라짐
		SDFEngine::SmoothField(grid, smoothIter);
		TE(SDF_Smooth);

		// ---------------------------------------------------------
		// [Step 4] 표면 복원 (Points Reconstruction)
		// ---------------------------------------------------------
		TS(SDF_Extract);
		std::vector<Eigen::Vector3f> refinedPoints;
		std::vector<Eigen::Vector3f> refinedNormals;

		// isoLevelRatio: 0.5 ~ 0.8 추천. 
		// 값이 클수록 표면이 두꺼워져 점이 많아짐.
		SDFEngine::ExtractSurface(grid, refinedPoints, refinedNormals, 0.7f);
		TE(SDF_Extract);

		TE(Total_SDF_Process);

		// 2. 결과 시각화
		// 정제된 포인트 (초록색, 불투명)
		// 법선 정보도 같이 넣어주면 조명을 받아 입체감이 살아남
		VD::AddSphereBatch(
			//"SDF_Filtered",
			"PointCloud",
			refinedPoints,
			refinedNormals,
			voxelSize * 0.4f, // 점 크기는 복셀보다 약간 작게
			{ 0.0f, 1.0f, 0.2f, 1.0f }
		);

		// (선택 사항) 결과 정보 출력
		//VD::AddText("Original Count: " + std::to_string(rawPoints.size()), { 10, 30 }, { 1,1,1,1 });
		//VD::AddText("Refined Count: " + std::to_string(refinedPoints.size()), { 10, 50 }, { 0,1,0,1 });

		printf("Original Points: %zu\n", rawPoints.size());
		printf("Refined Points: %zu\n", refinedPoints.size());
	}
};

REGISTER_APP(AppSDFFiltering, "AppSDFFiltering");
