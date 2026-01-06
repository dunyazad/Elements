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
    if (pointCloud->LoadFromPLY(fileName, name))
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

bool HeliumCore::SelectPointCloud(int ID)
{
    if (pointClouds.find(ID) != pointClouds.end())
    {
        selectedPointCloud = pointClouds[ID];
        return true;
    }
    return false;
}

PointCloud* HeliumCore::GetPointCloud(int ID)
{
    if (pointClouds.find(ID) != pointClouds.end())
    {
        return pointClouds[ID];
    }
    return nullptr;
}

PointCloud* HeliumCore::GetSelectedPointCloud()
{
    return selectedPointCloud;
}

void HeliumCore::SetPointCloudVisibility(int ID, bool visible)
{
    auto pointCloud = GetPointCloud(ID);
    if (pointCloud)
    {
        pointCloud->SetVisible(visible);
    }
}

void HeliumCore::ClonePointCloud(int ID)
{
    PointCloud* original = GetPointCloud(ID);
    if (original)
    {
        auto clone = original->Clone();
        pointClouds[clone->GetID()] = clone;
        selectedPointCloud = clone;
    }
}

void HeliumCore::DeletePointCloud(int ID)
{
    if (pointClouds.find(ID) != pointClouds.end())
    {
        PointCloud* pointCloud = pointClouds[ID];
        if (nullptr != pointCloud)
        {
            delete pointCloud;
            pointCloud = nullptr;
        }
        pointClouds.erase(ID);
        if (selectedPointCloud && selectedPointCloud->GetID() == ID)
        {
            selectedPointCloud = nullptr;
        }
	}
}

void HeliumCore::PerformClustering(int ID, float searchRadius, float angleThreshold)
{
    TS(Clustering_Parallel);

	AtomicDisjointSet ads;
	auto currentPointCloud = GetPointCloud(ID);
    if (nullptr == currentPointCloud)
    {
		ErrorLog("", "PointCloud with ID %d not found.", ID);
    }

    auto spatialPartitioning = SparseGrid();
    spatialPartitioning.Build(currentPointCloud, 0.3f);

	ads.Initialize(currentPointCloud->Size());

    float searchRadiusSq = searchRadius * searchRadius;
    float strictAngleThreshold = 0.9f;
    float planeDistThreshold = 0.1f * 0.2f;

    struct CellData { uint64_t key; int headIdx; };
    std::vector<CellData> flatCells;
    flatCells.reserve(spatialPartitioning.voxelPointListHead.size());

    for (const auto& pair : spatialPartitioning.voxelPointListHead)
    {
        flatCells.push_back({ pair.first, pair.second });
    }

    const uint64_t mask = 0x1FFFFF;

    auto& flags = currentPointCloud->GetPointFlags();

    std::for_each(std::execution::par, flatCells.begin(), flatCells.end(), [&](const CellData& cell)
        {
            uint64_t key = cell.key;
            int headIdx = cell.headIdx;

            int gz = (int)(key & mask);
            int gy = (int)((key >> 21) & mask);
            int gx = (int)(key >> 42);

            for (int i = headIdx; i != -1; i = spatialPartitioning.nextPoint[i])
            {
                const Eigen::Vector3f& pA = currentPointCloud->GetPosition(i);
                const Eigen::Vector3f& nA = currentPointCloud->GetNormal(i);

                for (int j = spatialPartitioning.nextPoint[i]; j != -1; j = spatialPartitioning.nextPoint[j])
                {
                    const Eigen::Vector3f& pB = currentPointCloud->GetPosition(j);

                    if ((pA - pB).squaredNorm() > searchRadiusSq) continue;

                    //const Eigen::Vector3f& nB = currentPointCloud->GetNormal(j);

                    //if (nA.dot(nB) < strictAngleThreshold) continue;

                    //float planeDist = std::abs(nA.dot(pB - pA));
                    //if (planeDist > planeDistThreshold) continue;

                    ads.Union(i, j);
                }

                for (int dz = -1; dz <= 1; ++dz)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            if (dx == 0 && dy == 0 && dz == 0) continue;

                            uint64_t neighborKey = spatialPartitioning.GetKey(gx + dx, gy + dy, gz + dz);
                            if (neighborKey < key) continue;

                            auto it = spatialPartitioning.voxelPointListHead.find(neighborKey);
                            if (it == spatialPartitioning.voxelPointListHead.end()) continue;

                            int neighborHead = it->second;
                            for (int j = neighborHead; j != -1; j = spatialPartitioning.nextPoint[j])
                            {
                                const Eigen::Vector3f& pB = currentPointCloud->GetPosition(j);

                                if ((pA - pB).squaredNorm() > searchRadiusSq) continue;

                                //const Eigen::Vector3f& nB = currentPointCloud->GetNormal(j);

                                //if (nA.dot(nB) < strictAngleThreshold) continue;

                                //float planeDist = std::abs(nA.dot(pB - pA));
                                //if (planeDist > planeDistThreshold) continue;

                                ads.Union(i, j);
                            }
                        }
                    }
                }
            }
        });

    std::map<int, int> rootToClusterID;
    int currentClusterCount = 0;
	size_t numberOfPoints = currentPointCloud->Size();

    for (size_t i = 0; i < numberOfPoints; ++i)
    {
        int root = ads.Find((int)i);
        if (rootToClusterID.find(root) == rootToClusterID.end())
        {
            rootToClusterID[root] = currentClusterCount++;
        }
        currentPointCloud->SetClusterID(i, rootToClusterID[root]);
    }

    {
        std::unordered_map<int, int> rootSizeMap;
        for (size_t i = 0; i < numberOfPoints; ++i)
        {
            int root = ads.Find((int)i);
            rootSizeMap[root]++;
        }

        currentPointCloud->GetSortedClusters().clear();
        currentPointCloud->GetSortedClusters().reserve(rootSizeMap.size());
        for (auto const& [root, size] : rootSizeMap)
        {
            currentPointCloud->GetSortedClusters().emplace_back(root, size);
        }

        std::sort(currentPointCloud->GetSortedClusters().begin(), currentPointCloud->GetSortedClusters().end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.second > b.second;
            });

        std::unordered_map<int, int> rootToSortedId;
        int currentClusterCount = 0;
        for (const auto& pair : currentPointCloud->GetSortedClusters())
        {
            rootToSortedId[pair.first] = currentClusterCount++;
        }

        for (size_t i = 0; i < numberOfPoints; ++i)
        {
            int root = ads.Find((int)i);
            currentPointCloud->SetClusterID(i, rootToSortedId[root]);
        }

        if (currentPointCloud->GetSortedClusters().size() > 0) alog(" - Biggest(ID 0): %d points\n", currentPointCloud->GetSortedClusters()[0].second);
        if (currentPointCloud->GetSortedClusters().size() > 1) alog(" - 2nd(ID 1): %d points\n", currentPointCloud->GetSortedClusters()[1].second);
        if (currentPointCloud->GetSortedClusters().size() > 2) alog(" - 3rd(ID 2): %d points\n", currentPointCloud->GetSortedClusters()[2].second);
    }

    // Visualize Clustering Result using VD::AddSphere
    {
        VD::Clear("Clustering_Parallel");
        const auto& positions = currentPointCloud->GetPositions();
        const auto& clusterIDs = currentPointCloud->GetClusterIDs();
        std::vector<Eigen::Vector4f> clusterColors = Color::GetContrastingColorsWithoutBWRGB(32);
        for (size_t i = 0; i < numberOfPoints; ++i)
        {
            int clusterID = clusterIDs[i];
            Eigen::Vector4f color = clusterColors[clusterID % clusterColors.size()];
            VD::AddSphere(
                "Clustering_Parallel",
                positions[i],
                Eigen::Vector3f(0, 1, 0),
                0.05f,
                color);
        }
	}

    TE(Clustering_Parallel);
}

bool HeliumCore::ExecuteCommand(const char* command)
{
    if (command == nullptr)
    {
        ErrorLog("", "ExecuteCommand: Command string is null");
        return false;
    }

    //InfoLog("", "ExecuteCommand: %s", command);

    try
    {
        auto j = nlohmann::json::parse(command);

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
            else if(cmd == "SelectPointCloud")
            {
                if (j.contains("pointCloudID"))
                {
                    int pointCloudID = j["pointCloudID"];
                    SelectPointCloud(pointCloudID);
                }
            }
            else if(cmd == "SetPointCloudVisibility")
            {
                if (j.contains("pointCloudID") && j.contains("isVisible"))
                {
                    int pointCloudID = j["pointCloudID"];
                    bool isVisible = j["isVisible"];
                    SetPointCloudVisibility(pointCloudID, isVisible);
                }
            }
            else if(cmd == "ClonePointCloud")
            {
                if (j.contains("pointCloudID"))
                {
                    int pointCloudID = j["pointCloudID"];
                    ClonePointCloud(pointCloudID);
                }
            }
            else if(cmd == "DeletePointCloud")
            {
                if (j.contains("pointCloudID"))
                {
                    int pointCloudID = j["pointCloudID"];
                    DeletePointCloud(pointCloudID);
                }
			}
            else if (cmd == "BuildSparseGrid")
            {
                if (j.contains("pointCloudID"))
                {
                    int pointCloudID = j["pointCloudID"];
                    auto pointCloud = GetPointCloud(pointCloudID);
                    if (pointCloud)
                    {
                        float voxelSize = j.value("voxelSize", 0.1f);
                        SparseGrid sparseGrid;
                        sparseGrid.Build(pointCloud, voxelSize);
                        sparseGrid.Visualize();
                    }
                }
            }
            else if (cmd == "BuildSparseDataBlocks")
            {
                if (j.contains("pointCloudID"))
                {
                    int pointCloudID = j["pointCloudID"];
                    auto pointCloud = GetPointCloud(pointCloudID);
                    if (pointCloud)
                    {
                        float voxelSize = j.value("voxelSize", 0.1f);
                        SparseDataBlock sparseDataBlock;
                        sparseDataBlock.Build(pointCloud);
                        sparseDataBlock.Visualize();
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
							float searchRadius = j.value("searchRadius", 0.05f);
                            float angleThreshold = j.value("angleThreshold", 0.9f);

                            PerformClustering(pointCloudID, searchRadius, angleThreshold);
                        }
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
                if(immediateModeRenderSystem)
                {
                    immediateModeRenderSystem->ToggleCenterGizmo();
				}
            }
            else if (cmd == "ClearAllVisualDebugging")
            {
                VD::ClearAll();
			}
        }
    }
    catch (const nlohmann::json::parse_error& e)
    {
        ErrorLog("", "ExecuteCommand: JSON Parse Error - %s", e.what());
        return false;
    }
    catch (const std::exception& e)
    {
        ErrorLog("", "ExecuteCommand: Error - %s", e.what());
        return false;
    }

    return true;
}
