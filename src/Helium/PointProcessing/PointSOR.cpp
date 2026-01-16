#include "pch.h"

#include <Helium/PointProcessing/PointSOR.h>

#include <execution>

#include <Helium/HeliumCore.h>
#include <Helium/PointCloud.h>
#include <Helium/SpatialPartitionings/SparseGrid.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

namespace PointProcessing
{
	SOR::SOR()
		: PointProcessor(PointProcessorType::SOR)
	{
	}

	std::vector<uint8_t> SOR::Process(const PointProcessorParameters& parameters)
	{
		int pointCloudID = -1;
		int kNeighbors = 50;
		float stdDevMulThresh = 3.0f;
		bool deletePoints = false;
		int visualizationMode = 0;

		pointCloudID = parameters.GetParameter<int>("PointCloudID", pointCloudID);
		kNeighbors = parameters.GetParameter<int>("KNeighbors", kNeighbors);
		stdDevMulThresh = parameters.GetParameter<float>("StdDevMulThresh", stdDevMulThresh);
		deletePoints = parameters.GetParameter<bool>("DeletePoints", deletePoints);
		visualizationMode = parameters.GetParameter<int>("VisualizationMode", visualizationMode);

		InfoLog("", "Starting SOR Filter (k=%d, mul=%.1f, delete=%s)",
			kNeighbors, stdDevMulThresh, deletePoints ? "true" : "false");

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

		TS(SOR_Filter);

		size_t numberOfPoints = currentPointCloud->Size();
		if (numberOfPoints == 0) return outlierMarking;

		outlierMarking.resize(numberOfPoints, 0);

		std::vector<float> pointMeanDistances(numberOfPoints);
		std::vector<int> indices(numberOfPoints);
		std::iota(indices.begin(), indices.end(), 0);

		std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i)
			{
				const Eigen::Vector3f& p = currentPointCloud->GetPosition(i);

				std::vector<unsigned int> neighbors;
				std::vector<float> distances;
				neighbors.reserve(kNeighbors);
				distances.reserve(kNeighbors);

				sparseGrid->GetKNearestNeighbors(
					currentPointCloud->GetPositions(),
					p,
					kNeighbors,
					neighbors,
					distances
				);

				double sumDist = 0.0;
				int validCount = 0;
				for (float d : distances)
				{
					if (d > 1e-6f)
					{
						sumDist += d;
						validCount++;
					}
				}

				if (validCount > 0)
					pointMeanDistances[i] = (float)(sumDist / validCount);
				else
					pointMeanDistances[i] = 0.0f;
			});

		double totalSum = 0.0;
		double totalSqSum = 0.0;
		for (float d : pointMeanDistances)
		{
			totalSum += d;
			totalSqSum += d * d;
		}

		float globalMean = (float)(totalSum / numberOfPoints);
		double variance = (totalSqSum / numberOfPoints) - (globalMean * globalMean);
		float globalStdDev = std::sqrtf(std::max(0.0f, (float)variance));

		float distanceThreshold = globalMean + stdDevMulThresh * globalStdDev;

		InfoLog("", "[SOR] Mean: %.4f, StdDev: %.4f, Threshold: %.4f (k=%d, mul=%.1f)",
			globalMean, globalStdDev, distanceThreshold, kNeighbors, stdDevMulThresh);

		int outlierCount = 0;
		for (size_t i = 0; i < numberOfPoints; ++i)
		{
			if (pointMeanDistances[i] > distanceThreshold)
			{
				outlierMarking[i] = 1;
				outlierCount++;
			}
		}

		TE(SOR_Filter);

		{
			VD::Clear("SOR");

			if (visualizationMode != (int)PointProcessing::PointVisualizationMode::None)
			{
				const auto& positions = currentPointCloud->GetPositions();
				const auto& colors = currentPointCloud->GetColors();

				float visMaxDist = distanceThreshold;
				if (visMaxDist < 1e-6f) visMaxDist = 1.0f;

				for (size_t i = 0; i < numberOfPoints; ++i)
				{
					const bool isOutlier = (outlierMarking[i] == 1);

					if (visualizationMode == (int)PointProcessing::PointVisualizationMode::OutlierFiltered && isOutlier)
					{
						continue;
					}

					Eigen::Vector4f colorRGBA = colors[i];
					float radius = 0.05f;

					if (isOutlier)
					{
						colorRGBA = Color::red();
						radius = 0.06f;
					}
					else
					{
						if (visualizationMode == (int)PointProcessing::PointVisualizationMode::Binary)
						{
							colorRGBA = Color::green(1.0f);
						}
						else if (visualizationMode == (int)PointProcessing::PointVisualizationMode::Gradient)
						{
							colorRGBA = Color::GetHeatMapColor(pointMeanDistances[i], 0.0f, visMaxDist, 1.0f);
						}
					}

					VD::AddSphere("SOR", positions[i], colorRGBA.head<3>(), radius, colorRGBA);
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

				InfoLog("", "[SOR] Removed %d outliers. Remaining: %zu", outlierCount, newSize);
			}
			else
			{
				InfoLog("", "[SOR] Found 0 outliers (Clean).");
			}
		}

		return outlierMarking;
	}
}
