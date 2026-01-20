#include "pch.h"

#include <Helium/PointProcessing/PointPFOR.h>

#include <execution>
#include <cmath>
#include <atomic>
#include <limits>
#include <numeric>

#include <Helium/HeliumCore.h>
#include <Helium/PointCloud.h>
#include <Helium/SpatialPartitionings/SparseGrid.h>
#include <Helium/VisualDebugging.h>

using VD = VisualDebugging;

namespace PointProcessing
{
	PFOR::PFOR()
		: PointProcessor(PointProcessorType::PFOR)
	{
	}

	void PFOR::Process(const PointProcessorParameters& parameters)
	{
		int pointCloudID = -1;
		int pointIndex = -1;
		int kNeighbors = 30;
		float distanceThreshold = 0.085f;
		bool deletePoints = false;
		int visualizationMode = 0;

		pointCloudID = parameters.GetParameter<int>("PointCloudID", pointCloudID);
		pointIndex = parameters.GetParameter<int>("PointIndex", pointIndex);
		kNeighbors = parameters.GetParameter<int>("KNeighbors", kNeighbors);
		distanceThreshold = parameters.GetParameter<float>("DistanceThreshold", distanceThreshold);
		deletePoints = parameters.GetParameter<bool>("DeletePoints", deletePoints);
		visualizationMode = parameters.GetParameter<int>("VisualizationMode", visualizationMode);

		InfoLog("", "Point Index for PFOR: %d", pointIndex);

		auto currentPointCloud = Helium.GetPointCloud(pointCloudID);
		if (nullptr == currentPointCloud)
		{
			ErrorLog("", "PointCloud with ID %d not found.", pointCloudID);
			return;
		}

		if (-1 == pointIndex || pointIndex < 0 || pointIndex >= (int)currentPointCloud->Size())
		{
			ErrorLog("", "Invalid Point Index %d for PointCloud ID %d.", pointIndex, pointCloudID);
			return;
		}

		auto sparseGrid = Helium.GetSparseGrid(pointCloudID);
		if (nullptr == sparseGrid)
		{
			Helium.BuildSpatialPartitionings(pointCloudID);
			sparseGrid = Helium.GetSparseGrid(pointCloudID);
		}

		TS(PlaneFit_Outlier_Removal);

		size_t numberOfPoints = currentPointCloud->Size();
		if (numberOfPoints == 0)
		{
			ErrorLog("", "[PFOR] PointCloud is empty.");
			return;
		}

		const auto& positions = currentPointCloud->GetPositions();
		const auto& colors = currentPointCloud->GetColors();

		std::vector<uint8_t> outlierMarking(numberOfPoints, 0);
		size_t outlierCount = 0;

		const Eigen::Vector3f& p = positions[pointIndex];
		std::vector<unsigned int> neighbors;
		std::vector<float> distances;

		sparseGrid->GetKNearestNeighbors(positions, p, kNeighbors, neighbors, distances);

		std::vector<float> distToPlane(neighbors.size(), FLT_MAX);

		if (neighbors.size() < 3)
		{
			WarningLog("", "[PFOR] Not enough neighbors (%zu) for Point Index %d", neighbors.size(), pointIndex);
			return;
		}

		Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
		for (unsigned int idx : neighbors)
		{
			centroid += positions[idx];
		}
		centroid /= (float)neighbors.size();

		Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
		for (unsigned int idx : neighbors)
		{
			Eigen::Vector3f d = positions[idx] - centroid;
			covariance += d * d.transpose();
		}

		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
		Eigen::Vector3f planeNormal = solver.eigenvectors().col(0);

		for (size_t i = 0; i < neighbors.size(); i++)
		{
			auto& idx = neighbors[i];
			const Eigen::Vector3f& neighborPos = positions[idx];
			float dist = std::abs(planeNormal.dot(neighborPos - centroid));
			distToPlane[i] = dist;
		}

		float centerPointDist = std::abs(planeNormal.dot(p - centroid));
		if (centerPointDist > distanceThreshold)
		{
			outlierMarking[pointIndex] = 1;
			outlierCount = 1;
		}

		TE(PlaneFit_Outlier_Removal);

		{
			VD::Clear("PFOR");

			if (visualizationMode != (int)PointVisualizationMode::None)
			{
				float visMaxDist = distanceThreshold;

				// 1. 선택된 포인트(Target) 그리기
				{
					bool isTargetOutlier = (outlierMarking[pointIndex] == 1);
					Eigen::Vector4f targetColor;
					float targetRadius = 0.08f;

					if (isTargetOutlier)
					{
						targetColor = Color::red();
					}
					else
					{
						targetColor = Color::cyan();
					}

					VD::AddSphere("PFOR", positions[pointIndex], targetColor.head<3>(), targetRadius, targetColor);
				}

				// 2. 이웃 포인트들(Neighbors) 그리기
				for (size_t i = 0; i < neighbors.size(); i++)
				{
					unsigned int idx = neighbors[i];

					// 타겟 포인트와 겹칠 경우 이웃 루프에서는 스킵 (위에서 강조해서 그렸으므로)
					if ((int)idx == pointIndex)
						continue;

					Eigen::Vector4f neighborColor;
					float neighborRadius = 0.05f;

					if (visualizationMode == (int)PointVisualizationMode::Binary)
					{
						neighborColor = Color::green(1.0f);
					}
					else if (visualizationMode == (int)PointVisualizationMode::Gradient)
					{
						float d = distToPlane[i];
						neighborColor = Color::GetHeatMapColor(d, 0.0f, visMaxDist, 1.0f);
					}
					else
					{
						neighborColor = colors[idx];
					}

					VD::AddSphere("PFOR", positions[idx], neighborColor.head<3>(), neighborRadius, neighborColor);
				}

				// 평면과 법선 벡터 시각화
				VD::AddDisk("PFOR", centroid, planeNormal, distanceThreshold * 5.0f, Color::yellow(0.3f));
				VD::AddLine("PFOR", centroid, centroid + planeNormal * 0.2f, Color::yellow());
			}
		}

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
					if (outlierMarking[i] == 0)
					{
						newPositions.push_back(currentPointCloud->GetPosition(i));
						newNormals.push_back(currentPointCloud->GetNormal(i));
						newColors.push_back(currentPointCloud->GetColor(i));
					}
				}

				currentPointCloud->SetPositions(newPositions);
				currentPointCloud->SetNormals(newNormals);
				currentPointCloud->SetColors(newColors);

				Helium.BuildSpatialPartitionings(pointCloudID);

				InfoLog("", "[PFOR] Removed %zu outliers (Target Point). Remaining: %zu", outlierCount, newSize);
			}
			else
			{
				InfoLog("", "[PFOR] Found 0 outliers (Clean).");
			}
		}
		else
		{
			InfoLog("", "[PFOR] Analysis Done for Point %d. Is Outlier: %s", pointIndex, (outlierCount > 0 ? "Yes" : "No"));
		}
	}
}
