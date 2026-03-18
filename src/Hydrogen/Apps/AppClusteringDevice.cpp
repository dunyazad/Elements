#include "Apps.h"
#include <map>
#include <vector>
#include <Eigen/Core>
#include <iostream>

#include <Copper/Copper.h>
#include <Copper/CuPointCloud.h>
#include <Copper/CuSparseCells.h>
#include <Copper/OperatorCollection/CuOperatorCollection.h>
#include <Copper/CuVoxelStreaming.h>

#include <Helium/HeliumCommon.h>
#include <Helium/Serialization.hpp>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

class AppClusteringDevice : public App
{
public:
    virtual void Execute_NotSorted() 
    {
        cudaFree(0);

        CuSparseCells cellGrid;
        cellGrid.cellSize = 0.3f;
        CuPointCloud cloud;

        PLYFormat ply;
        if (!ply.Deserialize("D:\\Resources\\Default\\ExtractedSurfacePoints.ply"))
        {
			printf("[Error] Failed to load PLY file.\n");
            return;
        }

        size_t rawCount = ply.GetPoints().size();
        cloud.FromHostPointers(
            (float3*)ply.GetPoints().data(),
            (float3*)ply.GetNormals().data(),
            (float4*)ply.GetColors().data(),
            rawCount,
            { ply.GetAABBMin().x(), ply.GetAABBMin().y(), ply.GetAABBMin().z() },
            { ply.GetAABBMax().x(), ply.GetAABBMax().y(), ply.GetAABBMax().z() });

        //auto activeBounds = cellGrid.GetActiveCellBounds();
        //for (const auto& bound : activeBounds)
        //{
        //    Eigen::Vector3f minP(bound.first.x, bound.first.y, bound.first.z);
        //    Eigen::Vector3f maxP(bound.second.x, bound.second.y, bound.second.z);
        //    VD::AddWiredBox("GridWire", (minP + maxP) * 0.5f, maxP - minP, Eigen::Vector4f(0, 1, 0, 1));
        //}

        TS(ClusteringTotal);
        
        cellGrid.Build(&cloud, cellGrid.cellSize);

        unsigned int* d_labels = nullptr;
        cudaMalloc(&d_labels, rawCount * sizeof(unsigned int));
        cellGrid.ApplyClustering(&cloud, d_labels, 0.15f);

        TE(ClusteringTotal);

        std::vector<unsigned int> h_labels(rawCount);
        std::vector<float3> h_points(rawCount);
        std::vector<float3> h_normals(rawCount);

        cudaMemcpy(h_labels.data(), d_labels, rawCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_points.data(), (const float3*)thrust::raw_pointer_cast(cloud.points.data()), rawCount * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_normals.data(), (const float3*)thrust::raw_pointer_cast(cloud.normals.data()), rawCount * sizeof(float3), cudaMemcpyDeviceToHost);

        std::vector<Eigen::Vector3f> batchPos;
        std::vector<Eigen::Vector3f> batchNorm;
        std::vector<Eigen::Vector4f> batchCol;

        batchPos.reserve(rawCount);
        batchNorm.reserve(rawCount);
        batchCol.reserve(rawCount);

        auto getClusterColor = [](unsigned int label) -> Eigen::Vector4f {
            unsigned int h = label * 0x45d9f3b;
            h = ((h >> 16) ^ h) * 0x45d9f3b;
            h = ((h >> 16) ^ h);
            return Eigen::Vector4f(
                (float)(h & 0xFF) / 255.0f,
                (float)((h >> 8) & 0xFF) / 255.0f,
                (float)((h >> 16) & 0xFF) / 255.0f,
                1.0f
            );
            };

        for (size_t i = 0; i < rawCount; ++i)
        {
            batchPos.emplace_back(h_points[i].x, h_points[i].y, h_points[i].z);

            Eigen::Vector3f n(h_normals[i].x, h_normals[i].y, h_normals[i].z);
            if (n.squaredNorm() < 0.001f) n = Eigen::Vector3f::UnitY();
            batchNorm.push_back(n);

            batchCol.push_back(getClusterColor(h_labels[i]));
        }

        if (!batchPos.empty())
        {
            float pointSize = 0.05f;
            VD::AddSphereBatch("AllPointsClustered", batchPos, batchNorm, pointSize, batchCol);
        }

        std::cout << "[DONE] 렌더링 요청 완료: " << batchPos.size() << " points." << std::endl;

        cudaFree(d_labels);
    }

    virtual void Execute() override
    {
        cudaFree(0);

        CuSparseCells cellGrid;
        cellGrid.cellSize = 0.3f;
        CuPointCloud cloud;

        PLYFormat ply;
        if (!ply.Deserialize("D:\\Resources\\Default\\ExtractedSurfacePoints.ply"))
        {
            printf("[Error] Failed to load PLY file.\n");
            return;
        }

        size_t rawCount = ply.GetPoints().size();
        cloud.FromHostPointers(
            (float3*)ply.GetPoints().data(),
            (float3*)ply.GetNormals().data(),
            (float4*)ply.GetColors().data(),
            rawCount,
            {ply.GetAABBMin().x(), ply.GetAABBMin().y(), ply.GetAABBMin().z()},
            {ply.GetAABBMax().x(), ply.GetAABBMax().y(), ply.GetAABBMax().z()});

        TS(ClusteringTotal);

        cellGrid.Build(&cloud, cellGrid.cellSize);

        unsigned int* d_labels = nullptr;
        cudaMalloc(&d_labels, rawCount * sizeof(unsigned int));
        cellGrid.ApplyClustering(&cloud, d_labels, 0.15f);

        TE(ClusteringTotal);

        std::vector<unsigned int> h_labels(rawCount);
        std::vector<float3> h_points(rawCount);
        std::vector<float3> h_normals(rawCount);
        std::vector<uchar3> h_colors(rawCount);

        cudaMemcpy(h_labels.data(), d_labels, rawCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_points.data(), (const float3*)thrust::raw_pointer_cast(cloud.points.data()), rawCount * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_normals.data(), (const float3*)thrust::raw_pointer_cast(cloud.normals.data()), rawCount * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_colors.data(), (const uchar3*)thrust::raw_pointer_cast(cloud.colors.data()), rawCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

        struct ClusterStats
        {
            int count = 0;
            unsigned int originalLabel = 0;
        };
        std::map<unsigned int, ClusterStats> clusterMap;

        for (size_t i = 0; i < rawCount; ++i)
        {
            auto& stats = clusterMap[h_labels[i]];
            stats.count++;
            stats.originalLabel = h_labels[i];
        }

        std::vector<ClusterStats> sortedClusters;
        sortedClusters.reserve(clusterMap.size());
        for (auto const& [label, stats] : clusterMap)
        {
            if (stats.count >= 3)
            {
                sortedClusters.push_back(stats);
            }
        }

        std::sort(sortedClusters.begin(), sortedClusters.end(),
            [](const ClusterStats& a, const ClusterStats& b) {
                return a.count > b.count;
            });

        auto getRankColor = [](int rank) -> Eigen::Vector4f {
            if (rank < 0) return Eigen::Vector4f(0.2f, 0.2f, 0.2f, 1.0f);

            unsigned int h = (unsigned int)rank * 0x45d9f3b;
            h = ((h >> 16) ^ h) * 0x45d9f3b;
            h = ((h >> 16) ^ h);

            return Eigen::Vector4f(
                (float)(h & 0xFF) / 255.0f,
                (float)((h >> 8) & 0xFF) / 255.0f,
                (float)((h >> 16) & 0xFF) / 255.0f,
                1.0f
            );
            };

        std::map<unsigned int, int> labelToRank;
        for (int rank = 0; rank < (int)sortedClusters.size(); ++rank)
        {
            labelToRank[sortedClusters[rank].originalLabel] = rank;
        }

        std::vector<Eigen::Vector3f> mainPos, otherPos;
        std::vector<Eigen::Vector3f> mainNorm, otherNorm;
        std::vector<Eigen::Vector4f> mainCol, otherCol;

        for (size_t i = 0; i < rawCount; ++i)
        {
            Eigen::Vector3f p(h_points[i].x, h_points[i].y, h_points[i].z);
            Eigen::Vector3f n(h_normals[i].x, h_normals[i].y, h_normals[i].z);
            Eigen::Vector4f c(
                (float)h_colors[i].x / 255.0f,
                (float)h_colors[i].y / 255.0f,
                (float)h_colors[i].z / 255.0f,
				1.0f);
            if (n.squaredNorm() < 0.001f) n = Eigen::Vector3f::UnitY();

            auto it = labelToRank.find(h_labels[i]);
            int rank = (it != labelToRank.end()) ? it->second : -1;
            Eigen::Vector4f color = getRankColor(rank);

            if (rank == 0)
            {
                mainPos.push_back(p);
                mainNorm.push_back(n);
                mainCol.push_back(c);
            }
            else
            {
                otherPos.push_back(p);
                otherNorm.push_back(n);
                otherCol.push_back(color);
            }
        }

        if (!mainPos.empty())
        {
            //VD::AddSphereBatch("PointCloud", mainPos, mainNorm, 0.025f, mainCol);
            VD::AddSphereBatch("PointCloud", mainPos, mainNorm, 0.025f, Color::white());
        }
        if (!otherPos.empty())
        {
            VD::AddSphereBatch("OtherClusters", otherPos, otherNorm, 0.05f, otherCol);
        }

        {
            PLYFormat ply;
            for (size_t i = 0; i < mainPos.size(); i++)
            {
                ply.AddPoint(mainPos[i]);
                ply.AddNormal(mainNorm[i].x(), mainNorm[i].y(), mainNorm[i].z());
                ply.AddColor(mainCol[i].x(), mainCol[i].y(), mainCol[i].z(), mainCol[i].w());
            }
            ply.Serialize("D:\\Resources\\Default\\Clustered.ply");
        }

        cudaFree(d_labels);
    }
};

REGISTER_APP(AppClusteringDevice, "AppClusteringDevice");
