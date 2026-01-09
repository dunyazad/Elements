#include "pch.h"
#include <Helium/PointCloudProcessing/ROR.h>

#include <execution>

#include <Helium/HeliumCore.h>
#include <Helium/PointCloud.h>
#include <Helium/SpatialPartitionings/SparseGrid.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

ROR::ROR()
	: PointCloudProcessor(PointCloudProcessorType::ROR)
{
}

std::vector<uint8_t> ROR::Process(const PointCloudProcessorParameters& parameters)
{
	int pointCloudID = -1;
	float radius = 0.3f;
	int minNeighborsInRadius = 24;
	bool deletePoints = false;
	int visualizationMode = 0;
	
	pointCloudID = parameters.GetParameter<int>("PointCloudID", pointCloudID);
	radius = parameters.GetParameter<float>("Radius", radius);
	minNeighborsInRadius = parameters.GetParameter<int>("MinNeighborsInRadius", minNeighborsInRadius);
	deletePoints = parameters.GetParameter<bool>("DeletePoints", deletePoints);
	visualizationMode = parameters.GetParameter<int>("VisualizationMode", visualizationMode);

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
	
	TS(ROR_Filter);

	size_t numberOfPoints = currentPointCloud->Size();
	if (numberOfPoints == 0) return outlierMarking;

	outlierMarking.resize(numberOfPoints, 0);

	std::vector<int> neighborCounts(numberOfPoints, 0);
	std::vector<int> indices(numberOfPoints);
	std::iota(indices.begin(), indices.end(), 0);

	std::atomic<int> totalOutlierCount = 0;

	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i)
		{
			const Eigen::Vector3f& p = currentPointCloud->GetPosition(i);
			std::vector<unsigned int> neighbors;

			sparseGrid->GetPointsWithinRadius(
				currentPointCloud->GetPositions(),
				p,
				radius,
				neighbors
			);

			int count = (int)neighbors.size();
			neighborCounts[i] = count;

			if (count < minNeighborsInRadius)
			{
				outlierMarking[i] = 1;
				totalOutlierCount++;
			}
		});

	InfoLog("", "[ROR] Radius: %.4f, MinNeighbors: %d, Found Outliers: %d", radius, minNeighborsInRadius, totalOutlierCount.load());

	TE(ROR_Filter);

	{
		VD::Clear("ROR");

		if (visualizationMode != (int)PointCloudVisualizationMode::None)
		{
			const auto& positions = currentPointCloud->GetPositions();
			size_t displayCount = currentPointCloud->Size();

			float maxSafeCount = (float)minNeighborsInRadius * 3.0f;

			for (size_t i = 0; i < displayCount; ++i)
			{
				Eigen::Vector3f colorRGB;
				Eigen::Vector4f colorRGBA;
				float radiusVal = 0.05f;
				int count = neighborCounts[i];

				if (outlierMarking[i] == 1)
				{
					colorRGB = { 1.0f, 0.0f, 0.0f }; // Red
					colorRGBA = { 1.0f, 0.0f, 0.0f, 1.0f };
					radiusVal = 0.08f;
				}
				else
				{
					if ((int)PointCloudVisualizationMode::Binary == visualizationMode)
					{
						colorRGB = { 0.0f, 1.0f, 0.0f };
						colorRGBA = { 0.0f, 1.0f, 0.0f, 0.2f };
					}
					else if ((int)PointCloudVisualizationMode::Gradient == visualizationMode)
					{
						colorRGBA = Color::GetHeatMapColor((float)count, (float)minNeighborsInRadius, maxSafeCount);
						colorRGB = colorRGBA.head<3>();

						colorRGBA[3] = 0.6f;
					}
				}
				
				VD::AddSphere("ROR", positions[i], colorRGB, radiusVal, colorRGBA);
			}
		}
	}

	if (deletePoints)
	{
		int outlierCount = totalOutlierCount.load();
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

			InfoLog("", "[ROR] Removed %d outliers. Remaining: %zu", outlierCount, newSize);
		}
		else
		{
			InfoLog("", "[ROR] Found 0 outliers (Clean).");
		}
	}

	return outlierMarking;
}
