#include "pch.h"

#include <execution>

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

void HeliumCore::PerformSOR(int pointCloudID, int kNeighbors, float stdDevMulThresh, bool deletePoints, bool binaryVisualizationMode)
{
	auto currentPointCloud = GetPointCloud(pointCloudID);
	if (nullptr == currentPointCloud)
	{
		ErrorLog("", "PointCloud with ID %d not found.", pointCloudID);
		return;
	}

	auto sparseGrid = GetSparseGrid(pointCloudID);
	if (nullptr == sparseGrid)
	{
		BuildSpatialPartitionings(pointCloudID);
		sparseGrid = GetSparseGrid(pointCloudID);
	}

	TS(SOR_Filter);

	size_t numberOfPoints = currentPointCloud->Size();
	if (numberOfPoints == 0) return;

	std::vector<float> pointMeanDistances(numberOfPoints);
	std::vector<uint8_t> outlierMarking(numberOfPoints, 0);
	std::vector<int> indices(numberOfPoints);
	std::iota(indices.begin(), indices.end(), 0);

	// 1. 평균 거리 계산
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

	// 2. 통계 계산
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

	// ★ 임계값 계산 (이 값이 시각화의 기준이 됨)
	float distanceThreshold = globalMean + stdDevMulThresh * globalStdDev;

	InfoLog("", "[SOR] Mean: %.4f, StdDev: %.4f, Threshold: %.4f (k=%d, mul=%.1f)",
		globalMean, globalStdDev, distanceThreshold, kNeighbors, stdDevMulThresh);

	// 3. Outlier 판별
	int outlierCount = 0;
	for (size_t i = 0; i < numberOfPoints; ++i)
	{
		if (pointMeanDistances[i] > distanceThreshold)
		{
			outlierMarking[i] = 1;
			outlierCount++;
		}
	}

	currentPointCloud->SetAttribute<std::vector<uint8_t>>("SOR_OutlierMarking", outlierMarking);

	TE(SOR_Filter);

	// 4. 삭제 처리
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

			BuildSpatialPartitionings(pointCloudID);

			InfoLog("", "[SOR] Removed %d outliers. Remaining: %zu", outlierCount, newSize);
		}
		else
		{
			InfoLog("", "[SOR] Found 0 outliers (Clean).");
		}
	}

	// 5. 시각화 (Visual Debugging)
	{
		VD::Clear("SOR");

		const auto& positions = currentPointCloud->GetPositions();
		size_t displayCount = currentPointCloud->Size();

		// 히트맵의 최대값은 Threshold와 동일하게 설정 (그 이상은 Outlier이므로)
		float visMaxDist = distanceThreshold;

		for (size_t i = 0; i < displayCount; ++i)
		{
			Eigen::Vector3f colorRGB;
			Eigen::Vector4f colorRGBA;
			float radius = 0.05f;

			// Outlier (Threshold 초과) -> 빨간색 강조
			if (pointMeanDistances[i] > distanceThreshold)
			{
				colorRGB = { 1.0f, 0.0f, 0.0f }; // Red
				colorRGBA = { 1.0f, 0.0f, 0.0f, 1.0f }; // 불투명
				radius = 0.08f;
			}
			// Inlier -> Gradient or Green
			else
			{
				if (binaryVisualizationMode)
				{
					colorRGB = { 0.0f, 1.0f, 0.0f }; // Green
					colorRGBA = { 0.0f, 1.0f, 0.0f, 0.2f }; // 투명하게
				}
				else
				{
					// Gradient: 0.0(Blue) ~ Threshold(Green/Yellow)
					// Color::GetHeatMapColor는 0(Blue)->0.5(Green)->1(Red) 이므로
					// 여기서는 0.0 ~ 0.5 사이의 값으로 매핑해서 Blue->Green 느낌을 냄
					// 혹은 그냥 전체 레인지 사용하되, 1.0(Red)는 위에서 이미 걸러졌으므로 Green까지만 감.

					// 정규화 (0 ~ 1)
					float t = std::clamp(pointMeanDistances[i] / visMaxDist, 0.0f, 1.0f);

					// t가 1에 가까울수록(Threshold에 가까울수록) Green, 0이면 Blue
					Eigen::Vector3f c;
					if (t < 0.5f) {
						// Blue -> Cyan
						c = Eigen::Vector3f(0.0f, t * 2.0f, 1.0f);
					}
					else {
						// Cyan -> Green
						c = Eigen::Vector3f(0.0f, 1.0f, 1.0f - (t - 0.5f) * 2.0f);
					}

					colorRGB = c;
					colorRGBA = { c.x(), c.y(), c.z(), 0.5f };
				}
			}

			VD::AddSphere("SOR", positions[i], colorRGB, radius, colorRGBA);
		}
	}
}

// PerformROR: minNeighborsInRadius가 곧 Outlier의 기준선이 됩니다.
void HeliumCore::PerformROR(int pointCloudID, float radius, int minNeighborsInRadius, bool deletePoints, bool binaryVisualizationMode)
{
	auto currentPointCloud = GetPointCloud(pointCloudID);
	if (nullptr == currentPointCloud)
	{
		ErrorLog("", "PointCloud with ID %d not found.", pointCloudID);
		return;
	}

	auto sparseGrid = GetSparseGrid(pointCloudID);
	if (nullptr == sparseGrid)
	{
		BuildSpatialPartitionings(pointCloudID);
		sparseGrid = GetSparseGrid(pointCloudID);
	}

	TS(ROR_Filter);

	size_t numberOfPoints = currentPointCloud->Size();
	if (numberOfPoints == 0) return;

	std::vector<uint8_t> outlierMarking(numberOfPoints, 0);
	std::vector<int> neighborCounts(numberOfPoints, 0);
	std::vector<int> indices(numberOfPoints);
	std::iota(indices.begin(), indices.end(), 0);

	std::atomic<int> totalOutlierCount = 0;

	// 1. 반경 검색
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

	currentPointCloud->SetAttribute<std::vector<uint8_t>>("ROR_OutlierMarking", outlierMarking);

	InfoLog("", "[ROR] Radius: %.4f, MinNeighbors: %d, Found Outliers: %d", radius, minNeighborsInRadius, totalOutlierCount.load());

	TE(ROR_Filter);

	// 2. 삭제 처리
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

			BuildSpatialPartitionings(pointCloudID);

			InfoLog("", "[ROR] Removed %d outliers. Remaining: %zu", outlierCount, newSize);
		}
		else
		{
			InfoLog("", "[ROR] Found 0 outliers (Clean).");
		}
	}

	// 3. 시각화 (Visual Debugging)
	{
		VD::Clear("ROR");

		const auto& positions = currentPointCloud->GetPositions();
		size_t displayCount = currentPointCloud->Size();

		// Inlier들의 시각화를 위해 "충분히 안전한 개수"를 설정 (Min의 3배 정도)
		float maxSafeCount = (float)minNeighborsInRadius * 3.0f;

		for (size_t i = 0; i < displayCount; ++i)
		{
			Eigen::Vector3f colorRGB;
			Eigen::Vector4f colorRGBA;
			float radiusVal = 0.05f;
			int count = neighborCounts[i];

			// Outlier (기준 미달) -> 빨간색 강조
			if (count < minNeighborsInRadius)
			{
				colorRGB = { 1.0f, 0.0f, 0.0f }; // Red
				colorRGBA = { 1.0f, 0.0f, 0.0f, 1.0f };
				radiusVal = 0.08f;
			}
			// Inlier (기준 충족) -> Blue/Green
			else
			{
				if (binaryVisualizationMode)
				{
					colorRGB = { 0.0f, 1.0f, 0.0f };
					colorRGBA = { 0.0f, 1.0f, 0.0f, 0.2f };
				}
				else
				{
					// Gradient: Min(턱걸이, Green) -> MaxSafe(안전, Blue)
					float t = std::clamp((float)(count - minNeighborsInRadius) / (maxSafeCount - minNeighborsInRadius), 0.0f, 1.0f);

					// t=0 (Green) -> t=1 (Blue)
					colorRGB = Eigen::Vector3f(0.0f, 1.0f - t, t);
					colorRGBA = { colorRGB.x(), colorRGB.y(), colorRGB.z(), 0.6f };
				}
			}

			VD::AddSphere("ROR", positions[i], colorRGB, radiusVal, colorRGBA);
		}
	}
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

				if (j.contains("command"))
				{
					std::string cmd = j["command"];

					if (cmd == "LoadPointCloudFromPLY")
					{
						if (j.contains("fileNames") && j["fileNames"].is_array())
						{
							for (const auto& fileName : j["fileNames"])
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
						if (j.contains("pointCloudID"))
						{
							int pointCloudID = j["pointCloudID"];
							SelectPointCloud(pointCloudID);
						}
					}
					else if (cmd == "SetPointCloudVisibility")
					{
						if (j.contains("pointCloudID") && j.contains("isVisible"))
						{
							int pointCloudID = j["pointCloudID"];
							bool isVisible = j["isVisible"];
							SetPointCloudVisibility(pointCloudID, isVisible);
						}
					}
					else if (cmd == "TogglePointClouds")
					{
						if (j.contains("pointCloudVisibleInfoList"))
						{
							const auto& visibleList = j["pointCloudVisibleInfoList"];

							for (const auto& item : visibleList)
							{
								if (item.contains("pointCloudID") && item.contains("visible"))
								{
									int pointCloudID = item["pointCloudID"].get<int>();
									bool isVisible = item["visible"].get<bool>();
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
						if (j.contains("pointCloudID"))
						{
							int pointCloudID = j["pointCloudID"];
							ClonePointCloud(pointCloudID);
						}
					}
					else if (cmd == "DeletePointCloud")
					{
						if (j.contains("pointCloudID"))
						{
							int pointCloudID = j["pointCloudID"];
							DeletePointCloud(pointCloudID);
						}
					}
					else if (cmd == "ShowSparseGrid")
					{
						if (j.contains("pointCloudID"))
						{
							int pointCloudID = j["pointCloudID"];
							auto sparseGrid = GetSparseGrid(pointCloudID);
							if (sparseGrid)
							{
								sparseGrid->Visualize();
							}
						}
					}
					else if (cmd == "ShowSparseDataBlocks")
					{
						if (j.contains("pointCloudID"))
						{
							int pointCloudID = j["pointCloudID"];
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
							if (j.contains("pointCloudID"))
							{
								int pointCloudID = j["pointCloudID"];
								if (pointCloudID == selectedPointCloud->GetID())
								{
									float searchRadius = j.value("searchRadius", 0.15f);
									float angleThreshold = j.value("angleThreshold", 0.9f);

									PerformClustering(pointCloudID, searchRadius, angleThreshold);
								}
							}
						}
					}
					else if (cmd == "PerformSOR")
					{
						if (j.contains("pointCloudID"))
						{
							int pointCloudID = j["pointCloudID"];
							auto pointCloud = GetPointCloud(pointCloudID);
							if (nullptr != pointCloud)
							{
								int kNeighbors = j.value("kNeighbors", 50);
								float stdDevMulThresh = j.value("stdDevMulThresh", 1.0f);
								bool deletePoints = j.value("deletePoints", false);
								std::string visualizationMode = j.value("visualizationMode", "Binary");
								bool binaryVisualizationMode = (visualizationMode == "Binary");

								PerformSOR(pointCloudID, kNeighbors, stdDevMulThresh, deletePoints, binaryVisualizationMode);
							}
						}
						}
					else if (cmd == "PerformROR")
					{
						if (j.contains("pointCloudID"))
						{
							int pointCloudID = j["pointCloudID"];
							auto pointCloud = GetPointCloud(pointCloudID);
							if (nullptr != pointCloud)
							{
								float radius = j.value("radius", 0.5f);
								int minNeighborsInRadius = j.value("minNeighborsInRadius", 5);
								bool deletePoints = j.value("deletePoints", false);
								std::string visualizationMode = j.value("visualizationMode", "Binary");
								bool binaryVisualizationMode = (visualizationMode == "Binary");

								PerformROR(pointCloudID, radius, minNeighborsInRadius, deletePoints, binaryVisualizationMode);
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
						if (j.contains("pointCloudID"))
						{
							int pointCloudID = j["pointCloudID"];
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
						if (j.contains("pointCloudID"))
						{
							int pointCloudID = j["pointCloudID"];
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
