#include "pch.h"
#include <Helium/PointCloudProcessing/SOR.h>

#include <execution>

#include <Helium/HeliumCore.h>
#include <Helium/PointCloud.h>
#include <Helium/SpatialPartitionings/SparseGrid.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

SOR::SOR()
	: PointCloudProcessor(PointCloudProcessorType::SOR)
{
}

std::vector<uint8_t> SOR::Process(const PointCloudProcessorParameters& parameters)
{
	int pointCloudID = -1;
	int kNeighbors = 50;
	float stdDevMulThresh = 3.0f;
	bool deletePoints = false;
	bool binaryVisualizationMode = false;

	pointCloudID = parameters.GetParameter<int>("PointCloudID", pointCloudID);
	kNeighbors = parameters.GetParameter<int>("KNeighbors", kNeighbors);
	stdDevMulThresh = parameters.GetParameter<float>("StdDevMulThresh", stdDevMulThresh);
	deletePoints = parameters.GetParameter<bool>("DeletePoints", deletePoints);
	binaryVisualizationMode = parameters.GetParameter<bool>("BinaryVisualizationMode", binaryVisualizationMode);

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

	{
		VD::Clear("SOR");

		const auto& positions = currentPointCloud->GetPositions();
		size_t displayCount = currentPointCloud->Size();

		float visMaxDist = distanceThreshold;

		for (size_t i = 0; i < displayCount; ++i)
		{
			Eigen::Vector3f colorRGB;
			Eigen::Vector4f colorRGBA;
			float radius = 0.05f;

			if (outlierMarking[i] == 1)
			{
				colorRGB = { 1.0f, 0.0f, 0.0f };
				colorRGBA = { 1.0f, 0.0f, 0.0f, 1.0f };
				radius = 0.08f;
			}
			else
			{
				if (binaryVisualizationMode)
				{
					colorRGB = { 0.0f, 1.0f, 0.0f };
					colorRGBA = { 0.0f, 1.0f, 0.0f, 0.2f };
				}
				else
				{
					float t = std::clamp(pointMeanDistances[i] / visMaxDist, 0.0f, 1.0f);
					Eigen::Vector3f c;
					if (t < 0.5f) {
						c = Eigen::Vector3f(0.0f, t * 2.0f, 1.0f);
					}
					else {
						c = Eigen::Vector3f(0.0f, 1.0f, 1.0f - (t - 0.5f) * 2.0f);
					}

					colorRGB = c;
					colorRGBA = { c.x(), c.y(), c.z(), 0.5f };
				}
			}

			VD::AddSphere("SOR", positions[i], colorRGB, radius, colorRGBA);
		}
	}

	return outlierMarking;
}
