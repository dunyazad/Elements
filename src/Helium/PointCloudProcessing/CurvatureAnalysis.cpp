#include "pch.h"

#include <Helium/PointCloudProcessing/CurvatureAnalysis.h>

#include <execution>

#include <Helium/HeliumCore.h>
#include <Helium/PointCloud.h>
#include <Helium/SpatialPartitionings/SparseGrid.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

CurvatureAnalysis::CurvatureAnalysis()
	: PointCloudProcessor(PointCloudProcessorType::CurvatureAnalysis)
{
}

std::vector<uint8_t> CurvatureAnalysis::Process(const PointCloudProcessorParameters& parameters)
{
	int pointCloudID = -1;
	int kNeighbors = 30;
	float curvatureThreshold = 1.0f;
	bool binaryVisualizationMode = false;

	pointCloudID = parameters.GetParameter<int>("PointCloudID", pointCloudID);
	kNeighbors = parameters.GetParameter<int>("KNeighbors", kNeighbors);
	curvatureThreshold = parameters.GetParameter<float>("CurvatureThreshold", curvatureThreshold);
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

	TS(Curvature_Analysis);

	size_t numberOfPoints = currentPointCloud->Size();
	if (numberOfPoints == 0) return outlierMarking;

	outlierMarking.resize(numberOfPoints, 0);

	std::vector<float> curvatureValues(numberOfPoints, 0.0f);
	std::vector<int> indices(numberOfPoints);
	std::iota(indices.begin(), indices.end(), 0);

	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](int i)
		{
			const Eigen::Vector3f& p = currentPointCloud->GetPosition(i);
			std::vector<unsigned int> neighbors;
			std::vector<float> distances;

			sparseGrid->GetKNearestNeighbors(
				currentPointCloud->GetPositions(),
				p,
				kNeighbors,
				neighbors,
				distances
			);

			if (neighbors.size() < 3)
			{
				curvatureValues[i] = 0.0f;
				return;
			}

			Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
			for (unsigned int idx : neighbors)
			{
				centroid += currentPointCloud->GetPosition(idx);
			}
			centroid /= (float)neighbors.size();

			Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
			for (unsigned int idx : neighbors)
			{
				Eigen::Vector3f d = currentPointCloud->GetPosition(idx) - centroid;
				covariance += d * d.transpose();
			}
			covariance /= (float)neighbors.size();

			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
			Eigen::Vector3f eigenValues = solver.eigenvalues();

			float sumEigen = eigenValues[0] + eigenValues[1] + eigenValues[2];
			if (sumEigen > 1e-9f)
			{
				curvatureValues[i] = eigenValues[0] / sumEigen;
			}
			else
			{
				curvatureValues[i] = 0.0f;
			}
		});

	float maxVal = 0.1f * curvatureThreshold;

	int outlierCount = 0;
	for (size_t i = 0; i < numberOfPoints; ++i)
	{
		if (curvatureValues[i] > maxVal)
		{
			outlierMarking[i] = 1;
			outlierCount++;
		}
	}

	TE(Curvature_Analysis);

	{
		VD::Clear("Curvature");
		const auto& positions = currentPointCloud->GetPositions();

		for (size_t i = 0; i < numberOfPoints; ++i)
		{
			float val = curvatureValues[i];
			Eigen::Vector3f colorRGB;
			Eigen::Vector4f colorRGBA;

			if (outlierMarking[i] == 1)
			{
				colorRGB = { 1.0f, 0.0f, 0.0f }; // Red
				colorRGBA = { 1.0f, 0.0f, 0.0f, 1.0f };
			}
			else
			{
				if (binaryVisualizationMode)
				{
					colorRGB = { 0.0f, 1.0f, 0.0f }; // Green
					colorRGBA = { 0.0f, 1.0f, 0.0f, 0.2f };
				}
				else
				{
					// Gradient Mode
					colorRGBA = Color::GetHeatMapColor(val, 0.0f, maxVal);
					colorRGBA.w() = 0.8f;
					colorRGB = colorRGBA.head<3>();
				}
			}

			VD::AddSphere("Curvature", positions[i], colorRGB, 0.05f, colorRGBA);
		}
		InfoLog("", "[Curvature] Analysis Done. Threshold: %.3f, Outliers Marked: %d", maxVal, outlierCount);
	}

	return outlierMarking;
}
