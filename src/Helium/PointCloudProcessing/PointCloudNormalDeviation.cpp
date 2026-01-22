#include "pch.h"

#include <Helium/PointCloudProcessing/PointCloudNormalDeviation.h>

#include <execution>

#include <Helium/HeliumCore.h>
#include <Helium/PointCloud.h>
#include <Helium/SpatialPartitionings/SparseGrid.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

namespace PointCloudProcessing
{
	NormalDeviation::NormalDeviation()
		: PointCloudProcessor(PointCloudProcessorType::NormalDeviation)
	{
	}

	std::vector<uint8_t> NormalDeviation::Process(const PointCloudProcessorParameters& parameters)
	{
		int pointCloudID = -1;
		float radius = 0.1f;
		float deviationThreshold = 45.0f;
		int visualizationMode = 0;

		pointCloudID = parameters.GetParameter<int>("PointCloudID", pointCloudID);
		radius = parameters.GetParameter<float>("Radius", radius);
		deviationThreshold = parameters.GetParameter<float>("DeviationThreshold", deviationThreshold);
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

		TS(Normal_Deviation);

		size_t numberOfPoints = currentPointCloud->Size();
		std::vector<float> deviationValues(numberOfPoints, 0.0f);
		std::vector<int> indices(numberOfPoints);
		std::iota(indices.begin(), indices.end(), 0);

		outlierMarking.resize(numberOfPoints, 0);

		const auto& normals = currentPointCloud->GetNormals();

		std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i)
			{
				const Eigen::Vector3f& p = currentPointCloud->GetPosition(i);
				const Eigen::Vector3f& n = normals[i];

				std::vector<unsigned int> neighbors;
				std::vector<float> distances;

				sparseGrid->GetPointsWithinRadius(
					currentPointCloud->GetPositions(),
					p,
					radius,
					neighbors,
					distances
				);

				if (neighbors.empty())
				{
					deviationValues[i] = 0.0f;
					return;
				}

				double sumAngle = 0.0;
				int validCount = 0;

				for (unsigned int idx : neighbors)
				{
					if (i == idx) continue;

					float dot = n.dot(normals[idx]);
					dot = std::clamp(dot, -1.0f, 1.0f);

					float angleRad = std::acos(dot);
					sumAngle += angleRad;
					validCount++;
				}

				if (validCount > 0)
				{
					deviationValues[i] = (float)((sumAngle / validCount) * (180.0 / 3.14159265359));
				}
				else
				{
					deviationValues[i] = 0.0f;
				}
			});

		float maxAngle = deviationThreshold;

		int outlierCount = 0;
		for (size_t i = 0; i < numberOfPoints; ++i)
		{
			if (deviationValues[i] > maxAngle)
			{
				outlierMarking[i] = 1;
				outlierCount++;
			}
		}

		TE(Normal_Deviation);

		{
			VD::Clear("NormalDeviation");

			if (visualizationMode != (int)PointCloudVisualizationMode::None)
			{
				const auto& positions = currentPointCloud->GetPositions();
				const auto& colors = currentPointCloud->GetColors();

				for (size_t i = 0; i < numberOfPoints; ++i)
				{
					const bool isOutlier = (outlierMarking[i] == 1);

					if (visualizationMode == (int)PointCloudVisualizationMode::OutlierFiltered && isOutlier)
					{
						continue;
					}

					Eigen::Vector4f colorRGBA = colors[i];

					if (isOutlier)
					{
						colorRGBA = Color::red();
					}
					else
					{
						if (visualizationMode == (int)PointCloudVisualizationMode::Binary)
						{
							colorRGBA = Color::green(0.2f);
						}
						else if (visualizationMode == (int)PointCloudVisualizationMode::Gradient)
						{
							float val = deviationValues[i];
							colorRGBA = Color::GetHeatMapColor(val, 0.0f, maxAngle, 1.0f);
						}
					}

					VD::AddSphere("NormalDeviation", positions[i], colorRGBA.head<3>(), 0.05f, colorRGBA);
				}
			}

			InfoLog("", "[NormalDeviation] Analysis Done. Threshold: %.1f deg, Outliers Marked: %d", maxAngle, outlierCount);
		}

		return outlierMarking;
	}
}
