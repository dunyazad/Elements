#include "pch.h"

#include <execution>
#include <sstream>

#include <glad/glad.h>
#include <nlohmann/json.hpp>
#include <robin_hood/robin_hood.h>

#include <Helium/Color.hpp>

#include <Helium/HeliumCore.h>
#include <Helium/Backend/OpenGLBackend.h>
#include <Helium/Backend/VulkanBackend.h>
#include <Helium/HeliumEvents.h>

#include <Helium/Systems/EventSystem.h>
#include <Helium/Systems/InputSystem.h>
#include <Helium/Systems/RenderSystem.h>
#include <Helium/Systems/ImmediateModeRenderSystem.h>

#include <Helium/Components/Components.h>
#include <Helium/GeometryBuilder.h>
#include <Helium/VisualDebugging.h>
#include <Helium/PointCloud.h>

#include <Helium/SpatialPartitionings/SpartialPartitionings.h>
#include <Helium/PointCloudProcessing/PointCloudProcessing.h>
#include <Helium/PointCloudProcessing/AtomicDisjointSet.h>

using VD = VisualDebugging;

extern void He_LogInternal(HeliumLogLevel level, const char* key, char* message);

int HeliumCore::LoadPointCloudFromPLY(const std::string& fileName, const std::string& name)
{
	PointCloud* pointCloud = new PointCloud();
	if (pointCloud->LoadFromPLY(fileName, name, [this](PointCloud* pc)
		{
			this->BuildSpatialPartitionings(pc->GetID());
		}))
	{
		pointClouds[pointCloud->GetID()] = pointCloud;
		selectedPointCloud = pointCloud;
		return pointCloud->GetID();
	}
	else
	{
		delete pointCloud;
		return false;
	}
}

void HeliumCore::BuildSpatialPartitionings(int pointCloudID)
{
	auto pointCloud = Helium.GetPointCloud(pointCloudID);
	if (pointCloud)
	{
		if (sparseGrids.find(pointCloudID) != sparseGrids.end())
		{
			delete sparseGrids[pointCloudID];
			sparseGrids.erase(pointCloudID);
		}

		auto sparseGrid = new SparseGrid();
		sparseGrid->Build(pointCloud, 0.3f);
		sparseGrids[pointCloudID] = sparseGrid;

		if (sparseDataBlocks.find(pointCloudID) != sparseDataBlocks.end())
		{
			delete sparseDataBlocks[pointCloudID];
			sparseDataBlocks.erase(pointCloudID);
		}

		auto sparseDataBlock = new SparseDataBlock();
		sparseDataBlock->Build(pointCloud);
		sparseDataBlocks[pointCloudID] = sparseDataBlock;
	}
}

bool HeliumCore::SelectPointCloud(int pointCloudID)
{
	if (pointClouds.find(pointCloudID) != pointClouds.end())
	{
		selectedPointCloud = pointClouds[pointCloudID];
		return true;
	}
	return false;
}

PointCloud* HeliumCore::GetPointCloud(int pointCloudID)
{
	if (pointClouds.find(pointCloudID) != pointClouds.end())
	{
		return pointClouds[pointCloudID];
	}
	return nullptr;
}

PointCloud* HeliumCore::GetSelectedPointCloud()
{
	return selectedPointCloud;
}

void HeliumCore::SetPointCloudVisibility(int pointCloudID, bool visible)
{
	auto pointCloud = GetPointCloud(pointCloudID);
	if (pointCloud)
	{
		pointCloud->SetVisible(visible);
	}
}

void HeliumCore::ClonePointCloud(int pointCloudID)
{
	PointCloud* original = GetPointCloud(pointCloudID);
	if (original)
	{
		auto clone = original->Clone([this](PointCloud* pc)
			{
				this->BuildSpatialPartitionings(pc->GetID());
			});
		pointClouds[clone->GetID()] = clone;
		selectedPointCloud = clone;
	}
}

void HeliumCore::DeletePointCloud(int pointCloudID)
{
	auto pointCloud = GetPointCloud(pointCloudID);
	if (pointCloud == nullptr)
	{
		ErrorLog("", "PointCloud with ID %d not found.", pointCloudID);
		return;
	}

	{
		auto it = sparseGrids.find(pointCloudID);
		if (it != sparseGrids.end())
		{
			SparseGrid* sparseGrid = it->second;
			if (sparseGrid)
			{
				delete sparseGrid;
				sparseGrid = nullptr;
			}
			sparseGrids.erase(it);
		}
	}
	{
		auto it = sparseDataBlocks.find(pointCloudID);
		if (it != sparseDataBlocks.end())
		{
			SparseDataBlock* sparseDataBlock = it->second;
			if (sparseDataBlock)
			{
				delete sparseDataBlock;
				sparseDataBlock = nullptr;
			}
			sparseDataBlocks.erase(it);
		}
	}

	std::string name = "";

	name = pointCloud->GetName();

	delete pointCloud;
	pointCloud = nullptr;

	pointClouds.erase(pointCloudID);
	if (selectedPointCloud && selectedPointCloud->GetID() == pointCloudID)
	{
		selectedPointCloud = nullptr;
	}

	json j;
	j["EventType"] = "PointCloudDeleted";
	j["Parameters"]["PointCloudID"] = pointCloudID;
	j["Parameters"]["Name"] = name;
	Helium.NativeToManaged(j.dump().c_str());
}

void HeliumCore::RenamePointCloud(int pointCloudID, const std::string& newName)
{
	auto pointCloud = GetPointCloud(pointCloudID);
	if (pointCloud)
	{
		pointCloud->SetName(newName);
	}
}

void HeliumCore::PerformClustering(int pointCloudID, float searchRadius, float angleThreshold)
{
	TS(Clustering_Parallel);

	auto currentPointCloud = GetPointCloud(pointCloudID);
	if (nullptr == currentPointCloud)
	{
		ErrorLog("", "PointCloud with ID %d not found.", pointCloudID);
		return;
	}

	const float voxelSize = 0.3f;

	auto sparseGrid = GetSparseGrid(pointCloudID);
	if (nullptr == sparseGrid)
	{
		BuildSpatialPartitionings(pointCloudID);
		sparseGrid = GetSparseGrid(pointCloudID);
	}

	TS(InitializeADS);
	AtomicDisjointSet ads;
	size_t numberOfPoints = currentPointCloud->Size();
	ads.Initialize(numberOfPoints);
	TE(InitializeADS);

	const float searchRadiusSq = searchRadius * searchRadius;
	const float strictAngleThreshold = angleThreshold;
	const float planeDistThreshold = voxelSize * 0.2f; // 0.3 * 0.2 = 0.06f

	struct CellData { uint64_t key; int headIdx; };
	std::vector<CellData> flatCells;
	flatCells.reserve(sparseGrid->voxelPointListHead.size());

	for (const auto& pair : sparseGrid->voxelPointListHead)
	{
		flatCells.push_back({ pair.first, pair.second });
	}

	const uint64_t mask = 0x1FFFFF;

	const auto& positions = currentPointCloud->GetPositions();
	const auto& normals = currentPointCloud->GetNormals();

	const Eigen::Vector3f* pPos = positions.data();
	const Eigen::Vector3f* pNor = normals.data();

	std::for_each(std::execution::par, flatCells.begin(), flatCells.end(), [&](const CellData& cell)
		{
			uint64_t key = cell.key;
			int headIdx = cell.headIdx;
			int gx = (int)(key >> 42);
			int gy = (int)((key >> 21) & mask);
			int gz = (int)(key & mask);

			for (int i = headIdx; i != -1; i = sparseGrid->nextPoint[i])
			{
				const Eigen::Vector3f& pA = pPos[i];
				const Eigen::Vector3f& nA = pNor[i];

				for (int j = sparseGrid->nextPoint[i]; j != -1; j = sparseGrid->nextPoint[j])
				{
					const Eigen::Vector3f& pB = pPos[j];
					if ((pA - pB).squaredNorm() > searchRadiusSq) continue;

					const Eigen::Vector3f& nB = pNor[j];
					if (nA.dot(nB) < strictAngleThreshold) continue;

					float planeDist = std::abs(nA.dot(pB - pA));
					if (planeDist > planeDistThreshold) continue;

					ads.Union(i, j);
				}

				for (int dz = -1; dz <= 1; ++dz)
				{
					for (int dy = -1; dy <= 1; ++dy)
					{
						for (int dx = -1; dx <= 1; ++dx)
						{
							if (dx == 0 && dy == 0 && dz == 0) continue;

							uint64_t neighborKey = sparseGrid->GetKey(gx + dx, gy + dy, gz + dz);
							if (neighborKey < key) continue;

							auto it = sparseGrid->voxelPointListHead.find(neighborKey);
							if (it == sparseGrid->voxelPointListHead.end()) continue;

							int neighborHead = it->second;
							for (int j = neighborHead; j != -1; j = sparseGrid->nextPoint[j])
							{
								const Eigen::Vector3f& pB = pPos[j];
								if ((pA - pB).squaredNorm() > searchRadiusSq) continue;

								const Eigen::Vector3f& nB = pNor[j];
								if (nA.dot(nB) < strictAngleThreshold) continue;

								float planeDist = std::abs(nA.dot(pB - pA));
								if (planeDist > planeDistThreshold) continue;

								ads.Union(i, j);
							}
						}
					}
				}
			}
		});

	std::unordered_map<int, int> rootSizeMap;
	std::vector<std::pair<int, int>> sortedClusters;

	for (size_t i = 0; i < numberOfPoints; ++i)
	{
		int root = ads.Find((int)i);
		rootSizeMap[root]++;
	}

	for (auto const& [root, size] : rootSizeMap)
	{
		sortedClusters.emplace_back(root, size);
	}

	std::sort(sortedClusters.begin(), sortedClusters.end(),
		[](const std::pair<int, int>& a, const std::pair<int, int>& b) {
			return a.second > b.second;
		});

	std::unordered_map<int, int> rootToSortedId;
	rootToSortedId.reserve(rootSizeMap.size());

	int currentClusterCount = 0;
	for (const auto& pair : sortedClusters)
	{
		rootToSortedId[pair.first] = currentClusterCount++;
	}

	std::vector<int> clusterIDs(numberOfPoints);
	for (size_t i = 0; i < numberOfPoints; ++i)
	{
		int root = ads.Find((int)i);
		clusterIDs[i] = rootToSortedId[root];
	}
	currentPointCloud->SetAttribute<std::vector<int>>("ClusterIDs", clusterIDs);
	currentPointCloud->SetAttribute<std::vector<std::pair<int, int>>>("SortedClusters", sortedClusters);

	TE(Clustering_Parallel);

	InfoLog("", "Clustering Done. Found %d clusters.\n", currentClusterCount);
	if (sortedClusters.size() > 0) InfoLog("", " - Biggest(ID 0): %d points\n", sortedClusters[0].second);
	if (sortedClusters.size() > 1) InfoLog("", " - 2nd(ID 1): %d points\n", sortedClusters[1].second);
	if (sortedClusters.size() > 2) InfoLog("", " - 3rd(ID 2): %d points\n", sortedClusters[2].second);

	{
		VD::Clear("Clustering_Parallel");
		const auto& positions = currentPointCloud->GetPositions();
		const auto& clusterIDs = currentPointCloud->GetAttribute<std::vector<int>>("ClusterIDs");
		std::vector<Eigen::Vector4f> clusterColors = Color::GetContrastingColorsWithoutBWRGB(32);
		for (size_t i = 0; i < numberOfPoints; ++i)
		{
			int clusterID = clusterIDs[i];
			Eigen::Vector4f color = clusterColors[clusterID % clusterColors.size()];
			VD::AddSphere("Clustering_Parallel", positions[i], Eigen::Vector3f(0, 1, 0), 0.05f, color);
		}
	}
}

std::vector<uint8_t> HeliumCore::PerformSOR(int pointCloudID, int kNeighbors, float stdDevMulThresh, bool deletePoints, PointCloudProcessing::PointCloudVisualizationMode visualizationMode)
{
	PointCloudProcessing::PointCloudProcessorParameters parameters;
	parameters.SetParameter<int>("PointCloudID", pointCloudID);
	parameters.SetParameter<int>("KNeighbors", kNeighbors);
	parameters.SetParameter<float>("StdDevMulThresh", stdDevMulThresh);
	parameters.SetParameter<bool>("DeletePoints", deletePoints);
	parameters.SetParameter<int>("VisualizationMode", (int)visualizationMode);

	PointCloudProcessing::SOR processor;

	return processor.Process(parameters);
}

std::vector<uint8_t> HeliumCore::PerformROR(int pointCloudID, float radius, int minNeighborsInRadius, bool deletePoints, PointCloudProcessing::PointCloudVisualizationMode visualizationMode)
{
	PointCloudProcessing::PointCloudProcessorParameters parameters;
	parameters.SetParameter<int>("PointCloudID", pointCloudID);
	parameters.SetParameter<float>("Radius", radius);
	parameters.SetParameter<int>("MinNeighborsInRadius", minNeighborsInRadius);
	parameters.SetParameter<bool>("DeletePoints", deletePoints);
	parameters.SetParameter<int>("VisualizationMode", (int)visualizationMode);
	
	PointCloudProcessing::ROR processor;
	
	return processor.Process(parameters);
}

std::vector<uint8_t> HeliumCore::PerformCurvatureAnalysis(int pointCloudID, int kNeighbors, float curvatureThreshold, PointCloudProcessing::PointCloudVisualizationMode visualizationMode)
{
	PointCloudProcessing::PointCloudProcessorParameters parameters;
	parameters.SetParameter<int>("PointCloudID", pointCloudID);
	parameters.SetParameter<int>("KNeighbors", kNeighbors);
	parameters.SetParameter<float>("CurvatureThreshold", curvatureThreshold);
	parameters.SetParameter<int>("VisualizationMode", (int)visualizationMode);

	PointCloudProcessing::CurvatureAnalysis processor;

	return processor.Process(parameters);
}

std::vector<uint8_t> HeliumCore::PerformNormalDeviationAnalysis(int pointCloudID, float radius, float deviationThreshold, PointCloudProcessing::PointCloudVisualizationMode visualizationMode)
{
	PointCloudProcessing::PointCloudProcessorParameters parameters;
	parameters.SetParameter<int>("PointCloudID", pointCloudID);
	parameters.SetParameter<float>("Radius", radius);
	parameters.SetParameter<float>("DeviationThreshold", deviationThreshold);
	parameters.SetParameter<int>("VisualizationMode", (int)visualizationMode);

	PointCloudProcessing::NormalDeviation processor;
	return processor.Process(parameters);
}

std::vector<uint8_t> HeliumCore::PerformPFOR(int pointCloudID, int kNeighbors, float distanceThreshold, PointCloudProcessing::PointCloudVisualizationMode visualizationMode)
{
	PointCloudProcessing::PointCloudProcessorParameters parameters;
	parameters.SetParameter<int>("PointCloudID", pointCloudID);
	parameters.SetParameter<int>("KNeighbors", kNeighbors);
	parameters.SetParameter<float>("DistanceThreshold", distanceThreshold);
	parameters.SetParameter<int>("VisualizationMode", (int)visualizationMode);

	PointCloudProcessing::PFOR processor;
	return processor.Process(parameters);
}

std::vector<uint8_t> HeliumCore::PerformGenerateMesh(int pointCloudID)
{
	PointCloudProcessing::PointCloudProcessorParameters parameters;
	parameters.SetParameter<int>("PointCloudID", pointCloudID);

	PointCloudProcessing::PointCloudGenerateMesh processor;
	return processor.Process(parameters);
}

void HeliumCore::ProcessManagedToNativeEvents()
{
	std::lock_guard<std::mutex> lock(managedToNativeEventQueueMutex);
	for (auto& event : managedToNativeEventQueue)
	{
		event();
	}
	managedToNativeEventQueue.clear();
}

void HeliumCore::EnqueueManagedToNativeEvent(std::function<void()> event)
{
	try
	{
		std::lock_guard<std::mutex> lock(managedToNativeEventQueueMutex);
		managedToNativeEventQueue.push_back(event);
	}
	catch (const std::system_error& e)
	{
		ErrorLog("", "EnqueueCommand System Error: %s", e.what());
	}
	catch (const std::exception& e)
	{
		ErrorLog("", "EnqueueCommand Exception: %s", e.what());
	}
	catch (...)
	{
		ErrorLog("", "EnqueueCommand: Unknown Exception");
	}
}

void HeliumCore::OnManagedToNative(const char* jsonString)
{
	if (jsonString == nullptr)
	{
		ErrorLog("", "ExecuteCommand: jsonString is null");
		return;
	}

	std::string jsonStr(jsonString);

	EnqueueManagedToNativeEvent([this, jsonStr]()
		{
			try
			{
				auto j = nlohmann::json::parse(jsonStr);

				if (j.contains("Command"))
				{
					std::string cmd = j["Command"];

					if (cmd == "LoadPointCloudFromPLY")
					{
						if (j.contains("FileNames") && j["FileNames"].is_array())
						{
							for (const auto& fileName : j["FileNames"])
							{
								std::string path = fileName.get<std::string>();
								std::filesystem::path fsPath(path);
								auto name = fsPath.filename();
								LoadPointCloudFromPLY(path, name.string());
							}
						}
					}
					else if (cmd == "SelectPointCloud")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							SelectPointCloud(pointCloudID);
						}
					}
					else if (cmd == "SetPointCloudVisibility")
					{
						if (j.contains("PointCloudID") && j.contains("IsVisible"))
						{
							int pointCloudID = j["PointCloudID"];
							bool isVisible = j["IsVisible"];
							SetPointCloudVisibility(pointCloudID, isVisible);
						}
					}
					else if (cmd == "TogglePointClouds")
					{
						if (j.contains("PointCloudVisibleInfoList"))
						{
							const auto& visibleList = j["PointCloudVisibleInfoList"];

							for (const auto& item : visibleList)
							{
								if (item.contains("PointCloudID") && item.contains("IsVisible"))
								{
									int pointCloudID = item["PointCloudID"].get<int>();
									bool isVisible = item["IsVisible"].get<bool>();
									auto pointCloud = GetPointCloud(pointCloudID);
									if (nullptr != pointCloud)
									{
										pointCloud->SetVisible(isVisible);
									}
								}
							}
						}
					}
					else if (cmd == "ClonePointCloud")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							ClonePointCloud(pointCloudID);
						}
					}
					else if (cmd == "DeletePointCloud")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							DeletePointCloud(pointCloudID);
						}
					}
					else if (cmd == "RenamePointCloud")
					{
						if (j.contains("PointCloudID") && j.contains("NewName"))
						{
							int pointCloudID = j["PointCloudID"];
							std::string newName = j["NewName"];
							RenamePointCloud(pointCloudID, newName);
						}
					}
					else if (cmd == "ShowSparseGrid")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							auto sparseGrid = GetSparseGrid(pointCloudID);
							if (sparseGrid)
							{
								sparseGrid->Visualize();
							}
						}
					}
					else if (cmd == "ShowSparseDataBlocks")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							auto sparseDataBlock = GetSparseDataBlock(pointCloudID);
							if (sparseDataBlock)
							{
								sparseDataBlock->Visualize();
							}
						}
					}
					else if (cmd == "PerformClustering")
					{
						if (nullptr != selectedPointCloud)
						{
							if (j.contains("PointCloudID"))
							{
								int pointCloudID = j["PointCloudID"];
								if (pointCloudID == selectedPointCloud->GetID())
								{
									float searchRadius = j.value("SearchRadius", 0.15f);
									float angleThreshold = j.value("AngleThreshold", 0.9f);

									PerformClustering(pointCloudID, searchRadius, angleThreshold);
								}
							}
						}
					}
					else if (cmd == "PerformSOR")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							auto pointCloud = GetPointCloud(pointCloudID);
							if (nullptr != pointCloud)
							{
								int kNeighbors = j.value("KNeighbors", 50);
								float stdDevMulThresh = j.value("StdDevMulThresh", 1.0f);
								bool deletePoints = j.value("DeletePoints", false);
								int visualizationMode = j.value("VisualizationMode", 0);

								PerformSOR(pointCloudID, kNeighbors, stdDevMulThresh, deletePoints, (PointCloudProcessing::PointCloudVisualizationMode)visualizationMode);
							}
						}
						}
					else if (cmd == "PerformROR")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							auto pointCloud = GetPointCloud(pointCloudID);
							if (nullptr != pointCloud)
							{
								float radius = j.value("Radius", 0.5f);
								int minNeighborsInRadius = j.value("MinNeighborsInRadius", 5);
								bool deletePoints = j.value("DeletePoints", false);
								int visualizationMode = j.value("VisualizationMode", 0);

								PerformROR(pointCloudID, radius, minNeighborsInRadius, deletePoints, (PointCloudProcessing::PointCloudVisualizationMode)visualizationMode);
							}
						}
					}
					else if (cmd == "PerformCurvatureAnalysis")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							int kNeighbors = j.value("KNeighbors", 30);
							float curvatureThreshold = j.value("CurvatureThreshold", 1.0f);
							int visualizationMode = j.value("VisualizationMode", 0);

							PerformCurvatureAnalysis(pointCloudID, kNeighbors, curvatureThreshold, (PointCloudProcessing::PointCloudVisualizationMode)visualizationMode);
						}
					}
					else if (cmd == "PerformNormalDeviationAnalysis")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							float radius = j.value("Radius", 0.1f);
							float deviationThreshold = j.value("DeviationThreshold", 45.0f);
							int visualizationMode = j.value("VisualizationMode", 0);

							PerformNormalDeviationAnalysis(pointCloudID, radius, deviationThreshold, (PointCloudProcessing::PointCloudVisualizationMode)visualizationMode);
						}
					}
					else if (cmd == "PerformPFOR")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							int kNeighbors = j.value("KNeighbors", 30);
							float distanceThreshold = j.value("DistanceThreshold", 0.1f);
							int visualizationMode = j.value("VisualizationMode", 0);
					
							PerformPFOR(pointCloudID, kNeighbors, distanceThreshold, (PointCloudProcessing::PointCloudVisualizationMode)visualizationMode);
						}
					}
					else if (cmd == "PerformCompositeFilter")
					{
						if (j.contains("PointCloudID") && j.contains("Pipeline"))
						{
							int pointCloudID = j["PointCloudID"];
							auto pointCloud = GetPointCloud(pointCloudID);

							if (pointCloud != nullptr)
							{
								size_t numPoints = pointCloud->Size();

								std::vector<bool> finalMask(numPoints, false);
								bool isFirstStep = true;

								const auto& pipeline = j["Pipeline"];

								for (const auto& step : pipeline)
								{
									std::string op = step.value("Operation", "Union"); // Union, Intersection, Difference
									std::string type = step.value("FilterType", "");

									if(false == step.contains("Parameters"))
										continue;

									const auto& parameters = step["Parameters"];

									std::vector<uint8_t> outlierMarking;
									bool validStep = false;

									PointCloudProcessing::PointCloudProcessorParameters params;
									params.SetParameter<int>("PointCloudID", pointCloudID);

									if (type == "SOR Filter")
									{
										int kNeighbors = parameters.value("KNeighbors", 50);
										float stdDevMulThresh = parameters.value("StdDevMulThresh", 1.0f);
										int visualizationMode = parameters.value("VisualizationMode", 0);

										params.SetParameter<int>("KNeighbors", kNeighbors);
										params.SetParameter<float>("StdDevMulThresh", stdDevMulThresh);
										params.SetParameter<int>("VisualizationMode", visualizationMode);

										PointCloudProcessing::SOR processor;
										outlierMarking = processor.Process(params);
										validStep = true;
									}
									else if (type == "ROR Filter")
									{
										float radius = parameters.value("Radius", 0.5f);
										int minNeighbors = parameters.value("MinNeighborsInRadius", 5);

										params.SetParameter<float>("Radius", radius);
										params.SetParameter<int>("MinNeighborsInRadius", minNeighbors);

										PointCloudProcessing::ROR processor;
										outlierMarking = processor.Process(params);
										validStep = true;
									}
									else if (type == "Curvature Analysis")
									{
										int kNeighbors = parameters.value("KNeighbors", 30);
										float curvatureThreshold = parameters.value("CurvatureThreshold", 0.1f);
										params.SetParameter<int>("KNeighbors", kNeighbors);
										params.SetParameter<float>("CurvatureThreshold", curvatureThreshold);

										PointCloudProcessing::CurvatureAnalysis processor;
										outlierMarking = processor.Process(params);
										validStep = true;
									}
									else if (type == "Normal Deviation Analysis")
									{
										float radius = parameters.value("Radius", 0.1f);
										float deviationThreshold = parameters.value("DeviationThreshold", 30.0f);

										params.SetParameter<float>("Radius", radius);
										params.SetParameter<float>("DeviationThreshold", deviationThreshold);

										PointCloudProcessing::NormalDeviation processor;
										outlierMarking = processor.Process(params);
										validStep = true;
									}

									if (validStep)
									{
										std::vector<bool> stepMask(numPoints, false);

#pragma omp parallel for
										for (int i = 0; i < (int)numPoints; ++i)
										{
											if (i < (int)outlierMarking.size() && outlierMarking[i] == 1)
											{
												stepMask[i] = true;
											}
										}

										if (isFirstStep)
										{
											finalMask = stepMask;
											isFirstStep = false;
										}
										else
										{
											if (op == "Union")
											{
#pragma omp parallel for
												for (int i = 0; i < (int)numPoints; ++i)
													finalMask[i] = finalMask[i] || stepMask[i];
											}
											else if (op == "Intersection")
											{
#pragma omp parallel for
												for (int i = 0; i < (int)numPoints; ++i)
													finalMask[i] = finalMask[i] && stepMask[i];
											}
											else if (op == "Subtraction")
											{
#pragma omp parallel for
												for (int i = 0; i < (int)numPoints; ++i)
													finalMask[i] = finalMask[i] && !stepMask[i];
											}
										}
									}
								}

								VD::Clear("CompositeResult");
								const auto& positions = pointCloud->GetPositions();
								const auto& normals = pointCloud->GetNormals();

								int selectedCount = 0;
								for (size_t i = 0; i < numPoints; ++i)
								{
									if (finalMask[i])
									{
										VD::AddSphere("CompositeResult", positions[i], normals[i], 0.126f, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
										selectedCount++;
									}
								}

								InfoLog("", "Composite Filter applied. Selected %d points.", selectedCount);
							}
						}
					}
					else if (cmd == "GenerateMesh")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							auto pointCloud = GetPointCloud(pointCloudID);

							if (nullptr != pointCloud)
							{
								PerformGenerateMesh(pointCloudID);

								std::string message = "Generating Mesh from PointCloud ID: " + std::to_string(pointCloudID) + "\n";
								NotifyMessage(message, 5000);
							}
						}
					}
					else if (cmd == "ToggleGrid")
					{
						auto entity = GetEntityByName("Grid");
						auto renderable = registry.try_get<Renderable>(entity);
						if (renderable)
						{
							renderable->SetVisible(!renderable->IsVisible());
						}
					}
					else if (cmd == "ToggleAxisGizmo")
					{
						if (immediateModeRenderSystem)
						{
							immediateModeRenderSystem->ToggleAxisGizmo();
						}
					}
					else if (cmd == "ToggleCenterGizmo")
					{
						if (immediateModeRenderSystem)
						{
							immediateModeRenderSystem->ToggleCenterGizmo();
						}
					}
					else if (cmd == "ClearAllVisualDebugging")
					{
						VD::ClearAll();
					}
					else if (cmd == "ShowPointNormal")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							int pointIndex = j.value("pointIndex", -1);
							if (-1 != pointIndex)
							{
								auto pointCloud = GetPointCloud(pointCloudID);
								if (nullptr != pointCloud)
								{
									const auto& positions = pointCloud->GetPositions();
									const auto& normals = pointCloud->GetNormals();
									VD::Clear("PointNormal");
									VD::AddArrow("PointNormal", positions[pointIndex], normals[pointIndex], 1.0f, Eigen::Vector4f(1, 0, 0, 1));

									/*size_t numberOfPoints = pointCloud->Size();
									if (pointIndex >= 0 && pointIndex < (int)numberOfPoints)
									{
										VD::Clear("PointNormal_SelectedPoint");
										Eigen::Vector3f p0 = positions[pointIndex];
										Eigen::Vector3f p1 = positions[pointIndex] + normals[pointIndex] * 0.1f;
										VD::AddLine("PointNormal_SelectedPoint", p0, p1, Eigen::Vector4f(1, 0, 0, 1));
									}*/
								}
							}
						}
					}
					else if (cmd == "ShowPointCloudNormals")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							auto pointCloud = GetPointCloud(pointCloudID);
							if (nullptr != pointCloud)
							{
								VD::Clear("PointNormals");

								size_t numberOfPoints = pointCloud->Size();
								for (size_t i = 0; i < numberOfPoints; i++)
								{
									Eigen::Vector3f p0 = pointCloud->GetPosition(i);
									Eigen::Vector3f p1 = p0 + pointCloud->GetNormal(i) * 0.1f;
									VD::AddLine("PointNormals", p0, p1, Eigen::Vector4f(1, 0, 0, 1));
								}
							}
						}
					}
					else if (cmd == "ApplyPointPlaneFitting")
					{
						if (j.contains("PointCloudID"))
						{
							int pointCloudID = j["PointCloudID"];
							auto pointCloud = GetPointCloud(pointCloudID);
							if (nullptr != pointCloud)
							{
								int kNeighbors = j.value("KNeighbors", 30);
								float distanceThreshold = j.value("DistanceThreshold", 0.07f);

								//float searchRadius = j.value("SearchRadius", 0.2f);
								//PointPlaneFitting(pointCloud, searchRadius);
							}
						}
					}
				}
			}
			catch (const nlohmann::json::parse_error& e)
			{
				ErrorLog("", "ExecuteCommand: JSON Parse Error - %s", e.what());
				return;
			}
			catch (const std::exception& e)
			{
				ErrorLog("", "ExecuteCommand: Error - %s", e.what());
				return;
			}
		});
}

extern void OnNativeToManaged(const char* jsonString);
void HeliumCore::NativeToManaged(const char* jsonString)
{
	OnNativeToManaged(jsonString);
}

void HeliumCore::NotifyMessage(const std::string& message, int durationMS)
{
	json j;
	j["EventType"] = "Notification";
	j["Parameters"]["Message"] = message;
	j["Parameters"]["DurationMS"] = durationMS;
	Helium.NativeToManaged(j.dump().c_str());
}
