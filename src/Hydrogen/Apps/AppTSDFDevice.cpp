#include "Apps.h"

#include <robin_hood/robin_hood.h>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <limits>

#include <VVV/VVV.h>
#pragma comment(lib, "VVV.lib")

#include <robin_hood/robin_hood.h>

#include <Copper/Copper.h>
#include <Copper/CuPointCloud.h>
#include <Copper/CuSparseCells.h>
#include <Copper/OperatorCollection/CuOperatorCollection.h>
#include <Copper/CuVoxelStreaming.h>

#include <Helium/HeliumCommon.h>
#include <Helium/Serialization.hpp>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include <VVV/VVV.h>
#pragma comment(lib, "VVV.lib")

typedef struct CamInfo_
{
    float cfx;
    float cfy;
    float ccx;
    float ccy;
    int cx;
    int cy;
    int img_width;
    int img_height;
    double R[9];
    double T[3];
    VVV::Vector3f dlpPos;
    VVV::Vector3f camPos;
    VVV::Matrix3f invMatTilt;
    VVV::Matrix3f matTilt;

    VVV::Matrix4f GetViewMatrix(const CamInfo_& info)
    {
        VVV::Matrix4f view = VVV::Matrix4f::Identity();
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                view(i, j) = (float)info.R[i * 3 + j];
            }
            view(i, 3) = (float)info.T[i];
        }
        return view;
    }

    VVV::Matrix4f GetProjectionMatrix(const CamInfo_& info, float n, float f)
    {
        VVV::Matrix4f proj = VVV::Matrix4f::Zero();
        proj(0, 0) = 2.0f * info.cfx / info.img_width;
        proj(0, 2) = 1.0f - (2.0f * info.ccx / info.img_width);
        proj(1, 1) = 2.0f * info.cfy / info.img_height;
        proj(1, 2) = (2.0f * info.ccy / info.img_height) - 1.0f;
        proj(2, 2) = -(f + n) / (f - n);
        proj(2, 3) = -(2.0f * f * n) / (f - n);
        proj(3, 2) = -1.0f;
        return proj;
    }
} CamInfo_;


class AppTSDFDevice : public App
{
public:
    virtual void Execute() override
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        auto [initialUsed, totalGpu] = CheckDeviceMemory("초기 상태");

        VVV::VoxelDataBase voxelDb;
        uint32_t maxBlocks = 160000;

        voxelDb.Allocate(maxBlocks);

        std::ifstream ifs("D:\\Resources\\Default\\Patches.bin", std::ios::binary);
        if (!ifs.is_open())
        {
            std::cout << "[Error] Failed to open Patches.bin" << std::endl;
            return;
        }

        CamInfo_ cam;
        Eigen::Matrix4f camRT;
        ifs.read(reinterpret_cast<char*>(&cam), sizeof(CamInfo_));
        ifs.read(reinterpret_cast<char*>(camRT.data()), sizeof(float) * 16);

        size_t numberOfPatches = 0;
        ifs.read(reinterpret_cast<char*>(&numberOfPatches), sizeof(size_t));

        robin_hood::unordered_map<size_t, std::tuple<int, Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector4f>> donwnSampling;

        VVV::Vector3f* d_points0 = nullptr;
        VVV::Vector3f* d_normals0 = nullptr;
        VVV::Vector3b* d_colors0 = nullptr;
        unsigned int numberOfPoints0 = 0;

        VVV::Vector3f* d_points45 = nullptr;
        VVV::Vector3f* d_normals45 = nullptr;
        VVV::Vector3b* d_colors45 = nullptr;
        unsigned int numberOfPoints45 = 0;

        TS(PatchTotal);
        for (size_t i = 0; i < numberOfPatches; i++)
        {
            TS(patch);

            size_t patchIndex = 0;
            ifs.read(reinterpret_cast<char*>(&patchIndex), sizeof(size_t));
            Eigen::Matrix4f rt0;
            ifs.read(reinterpret_cast<char*>(rt0.data()), sizeof(float) * 16);
            Eigen::Vector3f aabbMin0, aabbMax0;
            ifs.read(reinterpret_cast<char*>(aabbMin0.data()), sizeof(float) * 3);
            ifs.read(reinterpret_cast<char*>(aabbMax0.data()), sizeof(float) * 3);
            Eigen::Matrix4f rt45;
            ifs.read(reinterpret_cast<char*>(rt45.data()), sizeof(float) * 16);
            Eigen::Vector3f aabbMin45, aabbMax45;
            ifs.read(reinterpret_cast<char*>(aabbMin45.data()), sizeof(float) * 3);
            ifs.read(reinterpret_cast<char*>(aabbMax45.data()), sizeof(float) * 3);

            size_t numPts0 = 0;
            ifs.read(reinterpret_cast<char*>(&numPts0), sizeof(size_t));

            std::vector<Eigen::Vector3f> pts0(numPts0), normals0(numPts0);
            std::vector<Eigen::Vector3b> colors0(numPts0);
            ifs.read(reinterpret_cast<char*>(pts0.data()), sizeof(Eigen::Vector3f) * numPts0);
            ifs.read(reinterpret_cast<char*>(normals0.data()), sizeof(Eigen::Vector3f) * numPts0);
            ifs.read(reinterpret_cast<char*>(colors0.data()), sizeof(Eigen::Vector3b) * numPts0);

            size_t numPts45 = 0;
            ifs.read(reinterpret_cast<char*>(&numPts45), sizeof(size_t));
            std::vector<Eigen::Vector3f> pts45(numPts45), normals45(numPts45);
            std::vector<Eigen::Vector3b> colors45(numPts45);
            ifs.read(reinterpret_cast<char*>(pts45.data()), sizeof(Eigen::Vector3f) * numPts45);
            ifs.read(reinterpret_cast<char*>(normals45.data()), sizeof(Eigen::Vector3f) * numPts45);
            //ifs.read(reinterpret_cast<char*>(colors45.data()), sizeof(Eigen::Vector3b) * numPts45);

            Eigen::Vector3f sensorPos = rt0.block<3, 1>(0, 3);
            Eigen::Matrix3f rot = rt0.block<3, 3>(0, 0);

            VVV::Matrix4f gpuMatrix0;
            std::copy(rt0.transpose().data(), rt0.data() + 16, gpuMatrix0.data);

            if (numberOfPoints0 < numPts0)
            {
                if (d_points0) cudaFree(d_points0);
                if (d_normals0) cudaFree(d_normals0);
                if (d_colors0) cudaFree(d_colors0);
                numberOfPoints0 = (unsigned int)numPts0 * 2;
                cudaMalloc(&d_points0, sizeof(VVV::Vector3f) * numberOfPoints0);
                cudaMalloc(&d_normals0, sizeof(VVV::Vector3f) * numberOfPoints0);
                cudaMalloc(&d_colors0, sizeof(VVV::Vector3b) * numberOfPoints0);

                printf("Allocated GPU 0 buffers for %u points.\n", numberOfPoints0);
            }

            cudaMemcpy(d_points0, pts0.data(), sizeof(VVV::Vector3f) * numPts0, cudaMemcpyHostToDevice);
            cudaMemcpy(d_normals0, normals0.data(), sizeof(VVV::Vector3f) * numPts0, cudaMemcpyHostToDevice);
            cudaMemcpy(d_colors0, colors0.data(), sizeof(VVV::Vector3b) * numPts0, cudaMemcpyHostToDevice);

            voxelDb.IntegrateESDF(
                gpuMatrix0,
                d_points0,
                d_normals0,
                d_colors0,
                (uint32_t)numPts0,
                0.8f,
                (unsigned int)i);

            //        voxelDb.OccupyVoxelFromPoints(
            //            gpuMatrix0,
            //            d_points0,
            //            d_colors0,
            //            (uint32_t)numPts0,
            //            0.8f,
                        //(unsigned int)i);

            
            VVV::Matrix4f gpuMatrix45;
            std::copy(rt45.transpose().data(), rt45.data() + 16, gpuMatrix45.data);

            if (numberOfPoints45 < numPts45)
            {
                if (d_points45) cudaFree(d_points45);
                if (d_normals45) cudaFree(d_normals45);
                if (d_colors45) cudaFree(d_colors45);
                numberOfPoints45 = (unsigned int)numPts45 * 2;
                cudaMalloc(&d_points45, sizeof(VVV::Vector3f) * numberOfPoints45);
                cudaMalloc(&d_normals45, sizeof(VVV::Vector3f) * numberOfPoints45);
                cudaMalloc(&d_colors45, sizeof(VVV::Vector3b) * numberOfPoints45);

                printf("Allocated GPU 45 buffers for %u points.\n", numberOfPoints45);
            }

            cudaMemcpy(d_points45, pts45.data(), sizeof(VVV::Vector3f) * numPts45, cudaMemcpyHostToDevice);
            cudaMemcpy(d_normals45, normals45.data(), sizeof(VVV::Vector3f) * numPts45, cudaMemcpyHostToDevice);
            cudaMemcpy(d_colors45, colors45.data(), sizeof(VVV::Vector3b) * numPts45, cudaMemcpyHostToDevice);

            voxelDb.IntegrateESDF(
                gpuMatrix45,
                d_points45,
                d_normals45,
                d_colors45,
                (uint32_t)numPts45,
                0.8f,
                (unsigned int)i);

            //        voxelDb.OccupyVoxelFromPoints(
            //            gpuMatrix45,
            //            d_points45,
            //            d_colors45,
            //            (uint32_t)numPts45,
            //            0.8f,
                        //(unsigned int)i);

            printf("[%5zd] =-=-= ", i);
            TE(patch);
        }

        printf("numberOfPatches: %zu\n", numberOfPatches);

        TE(PatchTotal);

        if (d_points0) cudaFree(d_points0);
        if (d_normals0) cudaFree(d_normals0);
        if (d_colors0) cudaFree(d_colors0);

        if (d_points45) cudaFree(d_points45);
        if (d_normals45) cudaFree(d_normals45);
        if (d_colors45) cudaFree(d_colors45);

        uint32_t activeBlocksCount = 0;
        cudaMemcpy(&activeBlocksCount, voxelDb.d_blockCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        VisualizeVoxelsBatch_Surface_Clustering_Filtered(voxelDb, maxBlocks, 0.8f, activeBlocksCount);

        voxelDb.Free();
    }

private:
    void VisualizeVoxelsBatch(VVV::VoxelDataBase& voxelDb, uint32_t maxBlocks, float blockSize, uint32_t activeBlocksCount)
    {
        uint32_t maxOut = 50000000;
        std::vector<VVV::ExtractedVoxel> hostOut(maxOut);
        uint32_t finalCnt = voxelDb.ExtractActiveVoxelsToHost(blockSize, hostOut.data(), maxOut);

        if (finalCnt > 0)
        {
            uint32_t limit = (finalCnt > maxOut) ? maxOut : finalCnt;
            float voxelDrawSize = (blockSize / 8.0f) * 0.9f;

            std::vector<Eigen::Vector3f> voxelCenters;
            std::vector<Eigen::Vector3f> voxelDimensions;
            std::vector<Eigen::Vector4f> voxelColors;
            voxelCenters.reserve(limit);
            voxelDimensions.reserve(limit);
            voxelColors.reserve(limit);

            for (uint32_t i = 0; i < limit; i++)
            {
                voxelCenters.emplace_back(hostOut[i].position.x, hostOut[i].position.y, hostOut[i].position.z);
                voxelDimensions.emplace_back(voxelDrawSize, voxelDrawSize, voxelDrawSize);
                voxelColors.emplace_back(hostOut[i].color[0] / 255.f, hostOut[i].color[1] / 255.f, hostOut[i].color[2] / 255.f, 1.f);
            }
            VD::AddWiredBoxBatch("Voxels", voxelCenters, voxelDimensions, voxelColors);

            std::vector<uint64_t> hostHashTable(maxBlocks);
            cudaMemcpy(hostHashTable.data(), voxelDb.d_hashTable, sizeof(uint64_t) * maxBlocks, cudaMemcpyDeviceToHost);

            std::vector<Eigen::Vector3f> blockCenters;
            blockCenters.reserve(activeBlocksCount); // 실제 사용된 수만큼 예약

            //for (uint32_t i = 0; i < maxBlocks; ++i)
            //{
            //    uint64_t mKey = hostHashTable[i];
            //    if (mKey != 0 && mKey != 0xFFFFFFFFFFFFFFFFULL)
            //    {
            //        VVV::Morton64 morton(mKey);
            //        VVV::Vector3f bPos = morton.ToPosition(blockSize);
            //        blockCenters.emplace_back(bPos.x, bPos.y, bPos.z);
            //    }
            //}
            //VD::AddWiredBoxBatch("LDE_SparseDataBlocks", blockCenters, Eigen::Vector3f(blockSize, blockSize, blockSize), Eigen::Vector4f(0, 1, 0, 0.2f));
        }

        printf("\n>>> [최종 리포트]\n");
        printf("    - 추출된 복셀 수     : %u 개\n", finalCnt);
        printf("    - 해시 적재율        : %.2f%% (%u / %u)\n",
            (double)activeBlocksCount / maxBlocks * 100.0, activeBlocksCount, maxBlocks);
    }

    void VisualizeVoxelsBatch_Surface(VVV::VoxelDataBase& voxelDb, uint32_t maxBlocks, float blockSize, uint32_t activeBlocksCount)
    {
        // 1. Zero-crossing 정점 추출을 위한 메모리 준비
        // Marching Cubes 에지 기반이므로 복셀 수보다 넉넉하게 할당합니다.
        uint32_t maxOut = 10000000;
        std::vector<VVV::ExtractedVoxel> hostOut(maxOut);

        // 2. 선형 보간 기반 Zero-crossing 추출 함수 호출
        uint32_t finalCnt = voxelDb.ExtractZeroCrossingVoxelsToHost(blockSize, hostOut.data(), maxOut);

        if (finalCnt > 0)
        {
            uint32_t limit = (finalCnt > maxOut) ? maxOut : finalCnt;

            std::vector<Eigen::Vector3f> surfacePoints;
            std::vector<Eigen::Vector3f> surfaceNormals;
            std::vector<Eigen::Vector4f> surfaceColors;
            surfacePoints.reserve(limit);
            surfaceNormals.reserve(limit);
            surfaceColors.reserve(limit);

            float3 aabbMin = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
            float3 aabbMax = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };

            for (uint32_t i = 0; i < limit; i++)
            {
                surfacePoints.emplace_back(hostOut[i].position.x, hostOut[i].position.y, hostOut[i].position.z);

                surfaceNormals.emplace_back(hostOut[i].normal.x, hostOut[i].normal.y, hostOut[i].normal.z);

                surfaceColors.emplace_back(
                    hostOut[i].color[0] / 255.f,
                    hostOut[i].color[1] / 255.f,
                    hostOut[i].color[2] / 255.f,
                    1.f
                );

                aabbMin.x = std::min(aabbMin.x, hostOut[i].position.x);
                aabbMin.y = std::min(aabbMin.y, hostOut[i].position.y);
                aabbMin.z = std::min(aabbMin.z, hostOut[i].position.z);

                aabbMax.x = std::max(aabbMax.x, hostOut[i].position.x);
                aabbMax.y = std::max(aabbMax.y, hostOut[i].position.y);
                aabbMax.z = std::max(aabbMax.z, hostOut[i].position.z);
            }

            VD::AddDiskBatch("ZeroCrossingSurface", surfacePoints, surfaceNormals, 0.05f, 16, surfaceColors, true);

            //std::vector<uint64_t> hostHashTable(maxBlocks);
            //cudaMemcpy(hostHashTable.data(), voxelDb.d_hashTable, sizeof(uint64_t) * maxBlocks, cudaMemcpyDeviceToHost);

            //std::vector<Eigen::Vector3f> blockCenters;
            //blockCenters.reserve(activeBlocksCount);

            //for (uint32_t i = 0; i < maxBlocks; ++i)
            //{
            //    uint64_t key = hostHashTable[i];
            //    if (key != 0 && key != 0xFFFFFFFFFFFFFFFFULL)
            //    {
            //        VVV::Morton64 morton(key);
            //        VVV::Vector3f blockPos = morton.ToPosition(blockSize);
            //        blockCenters.emplace_back(blockPos.x, blockPos.y, blockPos.z);
            //    }
            //}
            //VD::AddWiredBoxBatch("SparseDataBlocks", blockCenters, Eigen::Vector3f(blockSize, blockSize, blockSize), Eigen::Vector4f(0, 1, 0, 0.1f));
        }

        printf("\n>>> [Iso-surface 리포트]\n");
        printf("    - 추출된 Zero-crossing 정점 : %u 개\n", finalCnt);
        printf("    - 해시 적재율             : %.2f%% (%u / %u)\n",
            (double)activeBlocksCount / maxBlocks * 100.0, activeBlocksCount, maxBlocks);
    }

    void VisualizeVoxelsBatch_Surface_Clustering(VVV::VoxelDataBase& voxelDb, uint32_t maxBlocks, float blockSize, uint32_t activeBlocksCount)
    {
        // 1. Zero-crossing 정점 추출을 위한 메모리 준비
        // Marching Cubes 에지 기반이므로 복셀 수보다 넉넉하게 할당합니다.
        uint32_t maxOut = 10000000;
        std::vector<VVV::ExtractedVoxel> hostOut(maxOut);

        // 2. 선형 보간 기반 Zero-crossing 추출 함수 호출
        uint32_t finalCnt = voxelDb.ExtractZeroCrossingVoxelsToHost(blockSize, hostOut.data(), maxOut);

        if (finalCnt > 0)
        {
            uint32_t limit = (finalCnt > maxOut) ? maxOut : finalCnt;

            std::vector<Eigen::Vector3f> surfacePoints;
            std::vector<Eigen::Vector3f> surfaceNormals;
            std::vector<Eigen::Vector4f> surfaceColors;
            surfacePoints.reserve(limit);
            surfaceNormals.reserve(limit);
            surfaceColors.reserve(limit);

            float3 aabbMin = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
            float3 aabbMax = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };

            for (uint32_t i = 0; i < limit; i++)
            {
                surfacePoints.emplace_back(hostOut[i].position.x, hostOut[i].position.y, hostOut[i].position.z);

                surfaceNormals.emplace_back(hostOut[i].normal.x, hostOut[i].normal.y, hostOut[i].normal.z);

                surfaceColors.emplace_back(
                    hostOut[i].color[0] / 255.f,
                    hostOut[i].color[1] / 255.f,
                    hostOut[i].color[2] / 255.f,
                    1.f
                );

                aabbMin.x = std::min(aabbMin.x, hostOut[i].position.x);
                aabbMin.y = std::min(aabbMin.y, hostOut[i].position.y);
                aabbMin.z = std::min(aabbMin.z, hostOut[i].position.z);

                aabbMax.x = std::max(aabbMax.x, hostOut[i].position.x);
                aabbMax.y = std::max(aabbMax.y, hostOut[i].position.y);
                aabbMax.z = std::max(aabbMax.z, hostOut[i].position.z);
            }

            size_t rawCount = surfacePoints.size();

            CuPointCloud cloud;
            cloud.FromHostPointers(
                (float3*)surfacePoints.data(),
                (float3*)surfaceNormals.data(),
                (float4*)surfaceColors.data(),
                (uint32_t)surfacePoints.size(),
                aabbMin,
                aabbMax);

            TS(ClusteringTotal);

            CuSparseCells cellGrid;
            cellGrid.cellSize = 0.3f;

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

            cudaFree(d_labels);
        }
    }

    void VisualizeVoxelsBatch_Surface_Clustering_Filtered(VVV::VoxelDataBase& voxelDb, uint32_t maxBlocks, float blockSize, uint32_t activeBlocksCount)
    {
        uint32_t maxOut = 70000000;
        std::vector<VVV::ExtractedVoxel> hostOut(maxOut);

        // 1. Zero-crossing 포인트 추출
        uint32_t finalCnt = voxelDb.ExtractZeroCrossingVoxelsToHost(blockSize, hostOut.data(), maxOut);

        if (finalCnt > 0)
        {
            uint32_t limit = (finalCnt > maxOut) ? maxOut : finalCnt;

            std::vector<Eigen::Vector3f> surfacePoints;
            std::vector<Eigen::Vector3f> surfaceNormals;
            std::vector<Eigen::Vector4f> surfaceColors;
            surfacePoints.reserve(limit);
            surfaceNormals.reserve(limit);
            surfaceColors.reserve(limit);

            float3 aabbMin = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
            float3 aabbMax = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };

            for (uint32_t i = 0; i < limit; i++)
            {
                surfacePoints.emplace_back(hostOut[i].position.x, hostOut[i].position.y, hostOut[i].position.z);
                surfaceNormals.emplace_back(hostOut[i].normal.x, hostOut[i].normal.y, hostOut[i].normal.z);
                surfaceColors.emplace_back(
                    hostOut[i].color[0] / 255.0f,
                    hostOut[i].color[1] / 255.0f,
                    hostOut[i].color[2] / 255.0f,
                    1.0f
                );

                aabbMin.x = std::min(aabbMin.x, hostOut[i].position.x);
                aabbMin.y = std::min(aabbMin.y, hostOut[i].position.y);
                aabbMin.z = std::min(aabbMin.z, hostOut[i].position.z);
                aabbMax.x = std::max(aabbMax.x, hostOut[i].position.x);
                aabbMax.y = std::max(aabbMax.y, hostOut[i].position.y);
                aabbMax.z = std::max(aabbMax.z, hostOut[i].position.z);
            }

            size_t rawCount = surfacePoints.size();

            // 2. GPU PointCloud 생성 및 클러스터링
            CuPointCloud cloud;
            cloud.FromHostPointers(
                (float3*)surfacePoints.data(),
                (float3*)surfaceNormals.data(),
                (float4*)surfaceColors.data(),
                (uint32_t)rawCount,
                aabbMin,
                aabbMax);

            CuSparseCells cellGrid;
            cellGrid.cellSize = 0.3f;
            cellGrid.Build(&cloud, cellGrid.cellSize);

            unsigned int* labelsDevice = nullptr;
            cudaMalloc(&labelsDevice, rawCount * sizeof(unsigned int));

            // Execute 함수와 동일한 임계값 적용
            cellGrid.ApplyClustering(&cloud, labelsDevice, 0.125f);

            // 3. 결과 다운로드 및 통계 처리 (Execute 로직 반영)
            std::vector<unsigned int> labelsHost(rawCount);
            std::vector<float3> pointsHost(rawCount);
            std::vector<float3> normalsHost(rawCount);
            std::vector<uchar3> colorsHost(rawCount);

            cudaMemcpy(labelsHost.data(), labelsDevice, rawCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaMemcpy(pointsHost.data(), (const float3*)thrust::raw_pointer_cast(cloud.points.data()), rawCount * sizeof(float3), cudaMemcpyDeviceToHost);
            cudaMemcpy(normalsHost.data(), (const float3*)thrust::raw_pointer_cast(cloud.normals.data()), rawCount * sizeof(float3), cudaMemcpyDeviceToHost);
            cudaMemcpy(colorsHost.data(), (const uchar3*)thrust::raw_pointer_cast(cloud.colors.data()), rawCount * sizeof(uchar3), cudaMemcpyDeviceToHost);

            struct ClusterStats
            {
                int count = 0;
                unsigned int originalLabel = 0;
            };
            std::map<unsigned int, ClusterStats> clusterMap;

            for (size_t i = 0; i < rawCount; ++i)
            {
                auto& stats = clusterMap[labelsHost[i]];
                stats.count++;
                stats.originalLabel = labelsHost[i];
            }

            // 4. 크기순 정렬 및 랭킹 부여
            std::vector<ClusterStats> sortedClusters;
            sortedClusters.reserve(clusterMap.size());
            for (auto const& [label, stats] : clusterMap)
            {
                if (stats.count >= 3) sortedClusters.push_back(stats);
            }

            std::sort(sortedClusters.begin(), sortedClusters.end(),
                [](const ClusterStats& a, const ClusterStats& b) { return a.count > b.count; });

            unsigned int maxClusterId = sortedClusters.empty() ? 0xFFFFFFFF : sortedClusters[0].originalLabel;

            // 5. 가장 큰 클러스터(Rank 0)만 선별하여 시각화
            std::vector<Eigen::Vector3f> mainPos;
            std::vector<Eigen::Vector3f> mainNorm;
            std::vector<Eigen::Vector4f> mainCol;

            aabbMin = { std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
            aabbMax = { std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest() };

            for (size_t i = 0; i < rawCount; ++i)
            {
                if (labelsHost[i] == maxClusterId && maxClusterId != 0xFFFFFFFF)
                {
                    mainPos.emplace_back(pointsHost[i].x, pointsHost[i].y, pointsHost[i].z);

                    Eigen::Vector3f n(normalsHost[i].x, normalsHost[i].y, normalsHost[i].z);
                    if (n.squaredNorm() < 0.001f) n = Eigen::Vector3f::UnitY();
                    mainNorm.push_back(n);

                    mainCol.emplace_back(
                        (float)colorsHost[i].x / 255.0f,
                        (float)colorsHost[i].y / 255.0f,
                        (float)colorsHost[i].z / 255.0f,
                        1.0f
                    );

					aabbMin.x = std::min(aabbMin.x, pointsHost[i].x);
					aabbMin.y = std::min(aabbMin.y, pointsHost[i].y);
					aabbMin.z = std::min(aabbMin.z, pointsHost[i].z);
					aabbMax.x = std::max(aabbMax.x, pointsHost[i].x);
					aabbMax.y = std::max(aabbMax.y, pointsHost[i].y);
					aabbMax.z = std::max(aabbMax.z, pointsHost[i].z);
                }
            }

            if (!mainPos.empty())
            {
                //VD::AddSphereBatch("PointCloud", mainPos, mainNorm, 0.05f, mainCol);
            }

            cudaFree(labelsDevice);

            CuPointCloud filterd;
            filterd.FromHostPointers(
                (float3*)mainPos.data(),
                (float3*)mainNorm.data(),
                (float4*)mainCol.data(),
                (uint32_t)mainPos.size(),
                aabbMin,
				aabbMax);

			CuSparseCells filterCellGrid;
			filterCellGrid.cellSize = 0.3f;
			filterCellGrid.Build(&filterd, filterCellGrid.cellSize);

    //        filterCellGrid.ApplyEdgePreservingSmoothing(
    //            &filterd,
    //            0.5f,   // radius: 주변 이웃 탐색 반경
    //            0.7f,   // factor: 스무딩 강도 (0.0 ~ 1.0)
    //            0.15f,  // edgeThreshold: 엣지 보존 임계값 (0.0 ~ 1.0)
	//			  30);    // iterations: 반복 횟수 

            filterCellGrid.ApplyEnergySmoothing(
                &filterd,
                0.5f,   // radius: 주변 이웃 탐색 반경
                0.1f,   // dataWeight: 데이터 적합도 가중치
                0.9f,   // smoothWeight: 스무딩 가중치
				30);    // iterations: 반복 횟수


            std::vector<float3> smoothPoints(mainPos.size());
            std::vector<float3> smoothNormals(mainPos.size());
            std::vector<uchar3> smoothColors(mainPos.size());
            cudaMemcpy(smoothPoints.data(), (const float3*)thrust::raw_pointer_cast(filterd.points.data()), mainPos.size() * sizeof(float3), cudaMemcpyDeviceToHost);
            cudaMemcpy(smoothNormals.data(), (const float3*)thrust::raw_pointer_cast(filterd.normals.data()), mainPos.size() * sizeof(float3), cudaMemcpyDeviceToHost);
            cudaMemcpy(smoothColors.data(), (const uchar3*)thrust::raw_pointer_cast(filterd.colors.data()), mainPos.size() * sizeof(uchar3), cudaMemcpyDeviceToHost);
            std::vector<Eigen::Vector3f> smoothPos;
            std::vector<Eigen::Vector3f> smoothNorm;
            std::vector<Eigen::Vector4f> smoothCol;
            smoothPos.reserve(mainPos.size());
            smoothNorm.reserve(mainPos.size());
            smoothCol.reserve(mainPos.size());
            for (size_t i = 0; i < mainPos.size(); ++i)
            {
                smoothPos.emplace_back(smoothPoints[i].x, smoothPoints[i].y, smoothPoints[i].z);
                Eigen::Vector3f n(smoothNormals[i].x, smoothNormals[i].y, smoothNormals[i].z);
                if (n.squaredNorm() < 0.001f) n = Eigen::Vector3f::UnitY();
                smoothNorm.push_back(n);
                smoothCol.emplace_back(
                    (float)smoothColors[i].x / 255.0f,
                    (float)smoothColors[i].y / 255.0f,
                    (float)smoothColors[i].z / 255.0f,
                    1.0f
                );
            }
			VD::AddSphereBatch("SmoothedPointCloud", smoothPos, smoothNorm, 0.05f, smoothCol);
        }
    }
};

REGISTER_APP(AppTSDFDevice, "AppTSDFDevice");
