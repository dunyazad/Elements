//#include "Apps.h"
//#include <map>
//#include <vector>
//#include <Eigen/Core>
//
//class AppClusteringDevice : public App
//{
//public:
//    virtual void Execute() override
//    {
//        cudaFree(0);
//
//        CuSparseCells cellGrid;
//        cellGrid.cellSize = 0.3f;
//        CuPointCloud cloud;
//
//        PLYFormat ply;
//        ply.Deserialize("D:\\Resources\\Debug\\3D\\BasePoints.ply");
//
//        cloud.FromHostPointers(
//            (float3*)ply.GetPoints().data(),
//            (float3*)ply.GetNormals().data(),
//            (float4*)ply.GetColors().data(),
//            ply.GetPoints().size());
//
//        cellGrid.Build(&cloud, cellGrid.cellSize);
//
//        unsigned int* d_labels = nullptr;
//        size_t numPoints = cloud.size();
//        cudaMalloc(&d_labels, numPoints * sizeof(unsigned int));
//
//        cellGrid.ApplyClustering(&cloud, d_labels, 0.1f);
//
//        std::vector<unsigned int> h_labels(numPoints);
//        std::vector<float3> h_points(numPoints);
//        std::vector<float3> h_normals(numPoints);
//        std::vector<uchar3> h_colors(numPoints);
//
//        cudaMemcpy(h_labels.data(), d_labels, numPoints * sizeof(unsigned int), cudaMemcpyDeviceToHost);
//        cudaMemcpy(h_points.data(), (float3*)thrust::raw_pointer_cast(cloud.points.data()), numPoints * sizeof(float3), cudaMemcpyDeviceToHost);
//        cudaMemcpy(h_normals.data(), (float3*)thrust::raw_pointer_cast(cloud.normals.data()), numPoints * sizeof(float3), cudaMemcpyDeviceToHost);
//        cudaMemcpy(h_colors.data(), (uchar3*)thrust::raw_pointer_cast(cloud.colors.data()), numPoints * sizeof(uchar3), cudaMemcpyDeviceToHost);
//
//        struct ClusterStats
//        {
//            double3 sumPos = { 0, 0, 0 };
//            double3 sumNormal = { 0, 0, 0 };
//            double3 sumColor = { 0, 0, 0 };
//            int count = 0;
//        };
//        std::map<unsigned int, ClusterStats> clusterMap;
//
//        for (size_t i = 0; i < numPoints; ++i)
//        {
//            unsigned int label = h_labels[i];
//            auto& stats = clusterMap[label];
//            stats.sumPos.x += h_points[i].x;
//            stats.sumPos.y += h_points[i].y;
//            stats.sumPos.z += h_points[i].z;
//            stats.sumNormal.x += h_normals[i].x;
//            stats.sumNormal.y += h_normals[i].y;
//            stats.sumNormal.z += h_normals[i].z;
//            stats.sumColor.x += h_colors[i].x;
//            stats.sumColor.y += h_colors[i].y;
//            stats.sumColor.z += h_colors[i].z;
//            stats.count++;
//        }
//
//        // --- Batch 처리를 위한 데이터 컨테이너 준비 ---
//        std::vector<Eigen::Vector3f> positions;
//        std::vector<Eigen::Vector3f> normals;
//        std::vector<float> radii;
//        std::vector<Eigen::Vector4f> colors;
//
//        positions.reserve(clusterMap.size());
//        normals.reserve(clusterMap.size());
//        radii.reserve(clusterMap.size());
//        colors.reserve(clusterMap.size());
//
//        for (auto const& [label, stats] : clusterMap)
//        {
//            if (stats.count < 3) continue;
//
//            float invCount = 1.0f / (float)stats.count;
//
//            // 1. Position
//            positions.emplace_back(
//                (float)(stats.sumPos.x * invCount),
//                (float)(stats.sumPos.y * invCount),
//                (float)(stats.sumPos.z * invCount)
//            );
//
//            // 2. Normal
//            Eigen::Vector3f eigNorm(
//                (float)(stats.sumNormal.x * invCount),
//                (float)(stats.sumNormal.y * invCount),
//                (float)(stats.sumNormal.z * invCount)
//            );
//            if (eigNorm.squaredNorm() > 1e-6f) eigNorm.normalize();
//            else eigNorm = Eigen::Vector3f::UnitY();
//            normals.push_back(eigNorm);
//
//            // 3. Color
//            colors.emplace_back(
//                (float)(stats.sumColor.x * invCount) / 255.0f,
//                (float)(stats.sumColor.y * invCount) / 255.0f,
//                (float)(stats.sumColor.z * invCount) / 255.0f,
//                1.0f
//            );
//
//            // 4. Radius
//            radii.push_back(fminf(0.5f, 0.05f + (stats.count * 0.0005f)));
//        }
//
//        // --- Batch 호출 ---
//        if (!positions.empty())
//        {
//            VD::AddSphereBatch("Clusters", positions, normals, 0.05f, colors);
//        }
//
//        cudaFree(d_labels);
//    }
//};
//
//REGISTER_APP(AppClusteringDevice, "AppClusteringDevice");













































#include "Apps.h"
#include <map>
#include <vector>
#include <Eigen/Core>
#include <iostream>

class AppClusteringDevice : public App
{
public:
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
};

REGISTER_APP(AppClusteringDevice, "AppClusteringDevice");