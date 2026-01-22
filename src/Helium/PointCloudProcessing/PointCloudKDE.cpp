#include "pch.h"
#include <Helium/PointCloudProcessing/PointCloudKDE.h>

#include <execution>

#include <Helium/HeliumCore.h>
#include <Helium/PointCloud.h>
#include <Helium/SpatialPartitionings/SparseGrid.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

namespace PointCloudProcessing
{
	KDE::KDE()
		: PointCloudProcessor(PointCloudProcessorType::KDE)
	{
	}

	std::vector<uint8_t> KDE::Process(const PointCloudProcessorParameters& parameters)
	{
		int pointCloudID = -1;

		// KDE 파라미터
		float bandwidth = 0.5f;     // 커널의 대역폭 (h), 보통 반경과 연동하거나 따로 설정
		float searchRadius = 1.5f;  // 이웃을 찾을 실제 반경 (보통 bandwidth의 3배)
		float stdDevMulThresh = 1.0f; // 밀도 하위 제거를 위한 임계값 계수

		bool deletePoints = false;
		int visualizationMode = 0;

		pointCloudID = parameters.GetParameter<int>("PointCloudID", pointCloudID);
		bandwidth = parameters.GetParameter<float>("Bandwidth", bandwidth); // 혹은 Radius 파라미터를 사용
		stdDevMulThresh = parameters.GetParameter<float>("StdDevMulThresh", stdDevMulThresh);
		deletePoints = parameters.GetParameter<bool>("DeletePoints", deletePoints);
		visualizationMode = parameters.GetParameter<int>("VisualizationMode", visualizationMode);

		// bandwidth 기반으로 검색 반경 설정 (Gaussian 분포의 3-sigma 규칙 적용 등)
		// 사용자가 Radius만 입력했다면 그것을 bandwidth로 쓰고 searchRadius를 3배로 잡는 등의 로직 필요
		// 여기서는 편의상 입력받은 Bandwidth(혹은 Radius)를 기준으로 설정합니다.
		float kernelH = bandwidth;
		if (searchRadius < kernelH * 3.0f) searchRadius = kernelH * 3.0f;

		InfoLog("", "Starting KDE Filter (Bandwidth=%.3f, SearchRadius=%.3f, delete=%s)",
			kernelH, searchRadius, deletePoints ? "true" : "false");

		std::vector<uint8_t> outlierMarking;

		auto currentPointCloud = Helium.GetPointCloud(pointCloudID);
		if (nullptr == currentPointCloud)
		{
			ErrorLog("", "PointCloud with ID %d not found.", pointCloudID);
			return outlierMarking;
		}

		auto sparseGrid = Helium.GetSparseGrid(pointCloudID);
		if (nullptr == sparseGrid)
		{
			Helium.BuildSpatialPartitionings(pointCloudID);
			sparseGrid = Helium.GetSparseGrid(pointCloudID);
		}

		TS(KDE_Filter);

		size_t numberOfPoints = currentPointCloud->Size();
		if (numberOfPoints == 0) return outlierMarking;

		outlierMarking.resize(numberOfPoints, 0);
		std::vector<float> pointDensities(numberOfPoints); // 거리 대신 밀도를 저장
		std::vector<int> indices(numberOfPoints);
		std::iota(indices.begin(), indices.end(), 0);

		// Gaussian Kernel 계산을 위한 상수 미리 계산
		// density += exp( - dist^2 / h^2 )
		float invHSq = 1.0f / (kernelH * kernelH);

		std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i)
			{
				const Eigen::Vector3f& p = currentPointCloud->GetPosition(i);

				std::vector<unsigned int> neighbors;
				std::vector<float> distances;

				// [주의] KDE는 KNN이 아니라 Radius Search를 해야 합니다.
				// SparseGrid에 GetNeighborsWithinRadius 구현이 있다고 가정합니다.
				sparseGrid->GetPointsWithinRadius(
					currentPointCloud->GetPositions(),
					p,
					searchRadius,
					neighbors,
					distances
				);

				float density = 0.0f;
				for (float d : distances)
				{
					// Gaussian Kernel 적용
					// 거리가 0이면(자기자신) 1.0, 멀어질수록 0에 수렴
					float w = std::expf(-(d * d) * invHSq);
					density += w;
				}

				pointDensities[i] = density;
			});

		// 통계 계산 (평균 밀도 및 표준편차)
		double totalSum = 0.0;
		double totalSqSum = 0.0;
		for (float d : pointDensities)
		{
			totalSum += d;
			totalSqSum += d * d;
		}

		float globalMeanDensity = (float)(totalSum / numberOfPoints);
		double variance = (totalSqSum / numberOfPoints) - (globalMeanDensity * globalMeanDensity);
		float globalStdDev = std::sqrtf(std::max(0.0f, (float)variance));

		// KDE는 밀도가 "낮은" 것이 이상치(Outlier)입니다.
		// Threshold = 평균 - (N * 표준편차)
		float densityThreshold = globalMeanDensity - (stdDevMulThresh * globalStdDev);
		if (densityThreshold < 0.0f) densityThreshold = 0.0f;

		InfoLog("", "[KDE] Mean Density: %.4f, StdDev: %.4f, Cutoff Threshold: %.4f",
			globalMeanDensity, globalStdDev, densityThreshold);

		int outlierCount = 0;
		for (size_t i = 0; i < numberOfPoints; ++i)
		{
			// 밀도가 임계값보다 낮으면 이상치로 판단
			if (pointDensities[i] < densityThreshold)
			{
				outlierMarking[i] = 1;
				outlierCount++;
			}
		}

		TE(KDE_Filter);

		// 시각화 (Visual Debugging)
		{
			VD::Clear("KDE");

			if (visualizationMode != (int)PointCloudVisualizationMode::None)
			{
				const auto& positions = currentPointCloud->GetPositions();
				const auto& colors = currentPointCloud->GetColors();

				// 시각화를 위한 Max Density 설정 (최댓값 혹은 평균+표준편차 등)
				float visMaxDensity = globalMeanDensity + 2.0f * globalStdDev;
				float visMinDensity = 0.0f; // KDE 최소값은 0

				for (size_t i = 0; i < numberOfPoints; ++i)
				{
					const bool isOutlier = (outlierMarking[i] == 1);

					// 이상치만 보기 모드 등 처리
					if (visualizationMode == (int)PointCloudVisualizationMode::OutlierFiltered && isOutlier)
					{
						continue;
					}

					Eigen::Vector4f colorRGBA = colors[i];
					float visRadius = 0.05f;

					if (isOutlier)
					{
						// 이상치는 빨간색 (혹은 눈에 띄는 색)
						colorRGBA = Color::red();
						visRadius = 0.03f; // 조금 작게
					}
					else
					{
						if (visualizationMode == (int)PointCloudVisualizationMode::Binary)
						{
							colorRGBA = Color::green(1.0f);
						}
						else if (visualizationMode == (int)PointCloudVisualizationMode::Gradient)
						{
							// 밀도에 따른 히트맵 (Jet/Rainbow)
							// 밀도가 높을수록 붉은색, 낮을수록 파란색 계열이 일반적
							colorRGBA = Color::GetHeatMapColor(pointDensities[i], visMinDensity, visMaxDensity, 1.0f);
						}
					}

					VD::AddSphere("KDE", positions[i], colorRGBA.head<3>(), visRadius, colorRGBA);
				}
			}
		}

		// 포인트 삭제 처리
		if (deletePoints)
		{
			if (outlierCount > 0)
			{
				size_t newSize = numberOfPoints - outlierCount;

				std::vector<Eigen::Vector3f> newPositions;
				std::vector<Eigen::Vector3f> newNormals;
				std::vector<Eigen::Vector4f> newColors;

				newPositions.reserve(newSize);
				newNormals.reserve(newSize);
				newColors.reserve(newSize);

				for (size_t i = 0; i < numberOfPoints; ++i)
				{
					if (outlierMarking[i] == 0) // 이상치가 아닌 것만 유지
					{
						newPositions.push_back(currentPointCloud->GetPosition(i));
						newNormals.push_back(currentPointCloud->GetNormal(i));
						newColors.push_back(currentPointCloud->GetColor(i));
					}
				}

				currentPointCloud->SetPositions(newPositions);
				currentPointCloud->SetNormals(newNormals);
				currentPointCloud->SetColors(newColors);

				// 파티션 재구축 (포인트 개수가 변했으므로)
				Helium.BuildSpatialPartitionings(pointCloudID);

				InfoLog("", "[KDE] Removed %d outliers (Low Density). Remaining: %zu", outlierCount, newSize);
			}
			else
			{
				InfoLog("", "[KDE] Found 0 outliers (Clean).");
			}
		}

		return outlierMarking;
	}
}
