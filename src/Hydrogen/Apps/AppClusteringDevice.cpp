#include "Apps.h"
#include <map>
#include <vector>
#include <Eigen/Core>
#include <iostream>

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
        if (!ply.Deserialize("D:\\Resources\\Debug\\3D\\BasePoints.ply")) return;

        size_t rawCount = ply.GetPoints().size();
        cloud.FromHostPointers(
            (float3*)ply.GetPoints().data(),
            (float3*)ply.GetNormals().data(),
            (float4*)ply.GetColors().data(),
            rawCount);

        //auto activeBounds = cellGrid.GetActiveCellBounds();
        //for (const auto& bound : activeBounds)
        //{
        //    Eigen::Vector3f minP(bound.first.x, bound.first.y, bound.first.z);
        //    Eigen::Vector3f maxP(bound.second.x, bound.second.y, bound.second.z);
        //    VD::AddWiredBox("GridWire", (minP + maxP) * 0.5f, maxP - minP, Eigen::Vector4f(0, 1, 0, 1));
        //}

        TS(ClusteringTotal);
        
        // 1. Grid Build
        cellGrid.Build(&cloud, cellGrid.cellSize);

        // 2. 클러스터링 실행 (거리 임계값 설정)
        unsigned int* d_labels = nullptr;
        cudaMalloc(&d_labels, rawCount * sizeof(unsigned int));
        cellGrid.ApplyClustering(&cloud, d_labels, 0.15f);

        TE(ClusteringTotal);

        // 3. 정렬된 최신 데이터 다운로드
        std::vector<unsigned int> h_labels(rawCount);
        std::vector<float3> h_points(rawCount);
        std::vector<float3> h_normals(rawCount);

        cudaMemcpy(h_labels.data(), d_labels, rawCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_points.data(), (const float3*)thrust::raw_pointer_cast(cloud.points.data()), rawCount * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_normals.data(), (const float3*)thrust::raw_pointer_cast(cloud.normals.data()), rawCount * sizeof(float3), cudaMemcpyDeviceToHost);

        // 4. 모든 포인트를 Batch 데이터로 구성 (집계 안 함)
        std::vector<Eigen::Vector3f> batchPos;
        std::vector<Eigen::Vector3f> batchNorm;
        std::vector<Eigen::Vector4f> batchCol;

        batchPos.reserve(rawCount);
        batchNorm.reserve(rawCount);
        batchCol.reserve(rawCount);

        // 레이블별 고정 색상을 위한 테이블 (미리 생성)
        auto getClusterColor = [](unsigned int label) -> Eigen::Vector4f {
            // 해시 함수를 이용해 레이블별로 고유한 색상 생성
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
            // 모든 좌표와 법선 그대로 추가
            batchPos.emplace_back(h_points[i].x, h_points[i].y, h_points[i].z);

            Eigen::Vector3f n(h_normals[i].x, h_normals[i].y, h_normals[i].z);
            if (n.squaredNorm() < 0.001f) n = Eigen::Vector3f::UnitY();
            batchNorm.push_back(n);

            // 색상은 클러스터 레이블에 따라 지정
            batchCol.push_back(getClusterColor(h_labels[i]));
        }

        // 5. 전체 포인트 시각화 (인스턴싱 개수가 많으므로 메모리 주의)
        if (!batchPos.empty())
        {
            // 포인트를 아주 작은 구(0.01f)로 그려서 원래 형태 유지
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
        if (!ply.Deserialize("D:\\Resources\\Debug\\3D\\BasePoints.ply")) return;

        size_t rawCount = ply.GetPoints().size();
        cloud.FromHostPointers(
            (float3*)ply.GetPoints().data(),
            (float3*)ply.GetNormals().data(),
            (float4*)ply.GetColors().data(),
            rawCount);

        TS(ClusteringTotal);

        // 1. Grid Build
        cellGrid.Build(&cloud, cellGrid.cellSize);

        // 2. 클러스터링 실행 (거리 임계값 설정)
        unsigned int* d_labels = nullptr;
        cudaMalloc(&d_labels, rawCount * sizeof(unsigned int));
        cellGrid.ApplyClustering(&cloud, d_labels, 0.15f);

        TE(ClusteringTotal);

        // 3. 데이터 다운로드
        std::vector<unsigned int> h_labels(rawCount);
        std::vector<float3> h_points(rawCount);
        std::vector<float3> h_normals(rawCount);
        std::vector<uchar3> h_colors(rawCount);

        cudaMemcpy(h_labels.data(), d_labels, rawCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_points.data(), (const float3*)thrust::raw_pointer_cast(cloud.points.data()), rawCount * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_normals.data(), (const float3*)thrust::raw_pointer_cast(cloud.normals.data()), rawCount * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_colors.data(), (const uchar3*)thrust::raw_pointer_cast(cloud.colors.data()), rawCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

        // 4. 클러스터별 집계 (CPU)
        struct ClusterStats
        {
            Eigen::Vector3f sumPos = { 0, 0, 0 };
            Eigen::Vector3f sumNormal = { 0, 0, 0 };
            int count = 0;
            unsigned int originalLabel = 0;
        };
        std::map<unsigned int, ClusterStats> clusterMap;

        for (size_t i = 0; i < rawCount; ++i)
        {
            auto& stats = clusterMap[h_labels[i]];
            stats.sumPos += Eigen::Vector3f(h_points[i].x, h_points[i].y, h_points[i].z);
            stats.sumNormal += Eigen::Vector3f(h_normals[i].x, h_normals[i].y, h_normals[i].z);
            stats.count++;
            stats.originalLabel = h_labels[i];
        }

        // 5. 포인트 개수(count) 순서로 정렬하기 위한 벡터 변환
        std::vector<ClusterStats> sortedClusters;
        sortedClusters.reserve(clusterMap.size());
        for (auto const& [label, stats] : clusterMap)
        {
            if (stats.count >= 3) // 최소 포인트 조건 필터링
            {
                sortedClusters.push_back(stats);
            }
        }

        // 내림차순 정렬 (포인트 많은 순)
        std::sort(sortedClusters.begin(), sortedClusters.end(),
            [](const ClusterStats& a, const ClusterStats& b) {
                return a.count > b.count;
            });

        auto getRankColor = [](int rank) -> Eigen::Vector4f {
            if (rank < 0) return Eigen::Vector4f(0.2f, 0.2f, 0.2f, 1.0f); // 노이즈 색상

            // 골든 레이블(0번)은 눈에 띄는 색상으로 고정하거나 해시 활용
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

        // 2. 매핑 테이블 준비
        std::map<unsigned int, int> labelToRank;
        for (int rank = 0; rank < (int)sortedClusters.size(); ++rank)
        {
            labelToRank[sortedClusters[rank].originalLabel] = rank;
        }

        // 3. 전체 포인트 순회 및 데이터 구성
        std::vector<Eigen::Vector3f> batchPos;
        std::vector<Eigen::Vector3f> batchNorm;
        std::vector<Eigen::Vector4f> batchCol;

        batchPos.reserve(rawCount);
        batchNorm.reserve(rawCount);
        batchCol.reserve(rawCount);

        for (size_t i = 0; i < rawCount; ++i)
        {
            batchPos.emplace_back(h_points[i].x, h_points[i].y, h_points[i].z);

            Eigen::Vector3f n(h_normals[i].x, h_normals[i].y, h_normals[i].z);
            if (n.squaredNorm() < 0.001f) n = Eigen::Vector3f::UnitY();
            batchNorm.push_back(n);

            // 여기서 getRankColor 호출
            auto it = labelToRank.find(h_labels[i]);
            if (it != labelToRank.end())
            {
                batchCol.push_back(getRankColor(it->second));
            }
            else
            {
                batchCol.push_back(getRankColor(-1)); // 순위권 외 노이즈
            }
        }

        // 4. 시각화 호출
        if (!batchPos.empty())
        {
            VD::AddSphereBatch("AllPointsRankedColor", batchPos, batchNorm, 0.02f, batchCol);
        }

        cudaFree(d_labels);
    }
};

REGISTER_APP(AppClusteringDevice, "AppClusteringDevice");
