#include "pch.h"

#include <Helium/PointProcessing/PointPFOR.h>

#include <execution>
#include <cmath>
#include <atomic>
#include <limits>

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

	std::vector<uint8_t> PFOR::Process(const PointProcessorParameters& parameters)
	{
		int pointCloudID = -1;
		int kNeighbors = 30;
		float distanceThreshold = 0.085f;
		bool deletePoints = false;
		int visualizationMode = 0;

		pointCloudID = parameters.GetParameter<int>("PointCloudID", pointCloudID);
		kNeighbors = parameters.GetParameter<int>("KNeighbors", kNeighbors);
		distanceThreshold = parameters.GetParameter<float>("DistanceThreshold", distanceThreshold);
		deletePoints = parameters.GetParameter<bool>("DeletePoints", deletePoints);
		visualizationMode = parameters.GetParameter<int>("VisualizationMode", visualizationMode);

		InfoLog("", "Starting PFOR Filter (k=%d, distThresh=%.4f, delete=%s)",
			kNeighbors, distanceThreshold, deletePoints ? "true" : "false");

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

		TS(PlaneFit_Outlier_Removal);

		size_t numberOfPoints = currentPointCloud->Size();
		if (numberOfPoints == 0) return outlierMarking;

		outlierMarking.resize(numberOfPoints, 0);

		const auto& positions = currentPointCloud->GetPositions();
		std::vector<float> distToPlane(numberOfPoints, 0.0f);
		std::vector<int> indices(numberOfPoints);
		std::iota(indices.begin(), indices.end(), 0);

		std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i)
			{
				const Eigen::Vector3f& p = positions[i];
				std::vector<unsigned int> neighbors;
				std::vector<float> distances;

				sparseGrid->GetKNearestNeighbors(positions, p, kNeighbors, neighbors, distances);

				if (neighbors.size() < 3)
				{
					distToPlane[i] = FLT_MAX;
					return;
				}

				// Calculate Centroid
				Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
				for (unsigned int idx : neighbors)
				{
					centroid += positions[idx];
				}
				centroid /= (float)neighbors.size();

				// Calculate Covariance Matrix
				Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
				for (unsigned int idx : neighbors)
				{
					Eigen::Vector3f d = positions[idx] - centroid;
					covariance += d * d.transpose();
				}

				// PCA for Normal Estimation
				Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
				Eigen::Vector3f planeNormal = solver.eigenvectors().col(0);

				// Calculate Distance to Plane
				float dist = std::abs(planeNormal.dot(p - centroid));
				distToPlane[i] = dist;
			});

		int outlierCount = 0;
		for (size_t i = 0; i < numberOfPoints; ++i)
		{
			if (distToPlane[i] > distanceThreshold)
			{
				outlierMarking[i] = 1;
				outlierCount++;
			}
		}

		TE(PlaneFit_Outlier_Removal);

		{
			VD::Clear("PFOR");

			if (visualizationMode != (int)PointVisualizationMode::None)
			{
				const auto& positions = currentPointCloud->GetPositions();
				const auto& colors = currentPointCloud->GetColors();
				float visMaxDist = distanceThreshold;

				for (size_t i = 0; i < numberOfPoints; ++i)
				{
					const bool isOutlier = (outlierMarking[i] == 1);

					if (visualizationMode == (int)PointVisualizationMode::OutlierFiltered && isOutlier)
					{
						continue;
					}

					Eigen::Vector4f colorRGBA = colors[i];
					float radius;

					if (isOutlier)
					{
						colorRGBA = Color::red();
						radius = 0.06f;
					}
					else
					{
						radius = 0.05f;

						if (visualizationMode == (int)PointVisualizationMode::Binary)
						{
							colorRGBA = Color::green(1.0f);
						}
						else if (visualizationMode == (int)PointVisualizationMode::Gradient)
						{
							colorRGBA = Color::GetHeatMapColor(distToPlane[i], 0.0f, visMaxDist, 1.0f);
						}
						else
						{
							colorRGBA = Color::white();
						}
					}

					VD::AddSphere("PFOR", positions[i], colorRGBA.head<3>(), radius, colorRGBA);
				}
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

				InfoLog("", "[PFOR] Removed %d outliers. Remaining: %zu", outlierCount, newSize);
			}
			else
			{
				InfoLog("", "[PFOR] Found 0 outliers (Clean).");
			}
		}
		else
		{
			InfoLog("", "[PFOR] Analysis Done. Outliers Marked: %d", outlierCount);
		}

		return outlierMarking;
	}
}