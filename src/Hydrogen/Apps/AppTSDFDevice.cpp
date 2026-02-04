#include "Apps.h"

#include <robin_hood/robin_hood.h>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <limits>

class AppTSDFDevice : public App
{
public:
    virtual void Execute() override
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        auto [initialUsed, totalGpu] = CheckDeviceMemory("초기 상태");

        VVV::VoxelDataBase voxelDb;
        uint32_t maxBlocks = 80000;

        VVV_Allocate(voxelDb, maxBlocks);

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

            VVV_IntegrateTSDF(
                voxelDb,
                gpuMatrix0,
                d_points0,
                d_normals0,
                d_colors0,
                (uint32_t)numPts0,
                0.8f,
                i);


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

            VVV_IntegrateTSDF(
                voxelDb,
                gpuMatrix45,
                d_points45,
                d_normals45,
                d_colors45,
                (uint32_t)numPts45,
                0.8f,
                i);

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

        VisualizeVoxelsBatch(voxelDb, maxBlocks, 0.8f, activeBlocksCount);

        VVV_Free(voxelDb);
    }

private:
    void VisualizeVoxelsBatch(VVV::VoxelDataBase& voxelDb, uint32_t maxBlocks, float blockSize, uint32_t activeBlocksCount)
    {
        uint32_t maxOut = 50000000;
        std::vector<VVV::ExtractedVoxel> hostOut(maxOut);
        uint32_t finalCnt = VVV_ExtractActiveVoxelsToHost(voxelDb, blockSize, hostOut.data(), maxOut);

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
        uint32_t finalCnt = VVV_ExtractZeroCrossingVoxelsToHost(voxelDb, blockSize, hostOut.data(), maxOut);

        if (finalCnt > 0)
        {
            uint32_t limit = (finalCnt > maxOut) ? maxOut : finalCnt;

            std::vector<Eigen::Vector3f> surfacePoints;
            std::vector<Eigen::Vector3f> surfaceNormals;
            std::vector<Eigen::Vector4f> surfaceColors;
            surfacePoints.reserve(limit);
            surfaceNormals.reserve(limit);
            surfaceColors.reserve(limit);

            for (uint32_t i = 0; i < limit; i++)
            {
                // 보간된 정밀 좌표 사용
                surfacePoints.emplace_back(hostOut[i].position.x, hostOut[i].position.y, hostOut[i].position.z);

				// 보간된 법선 사용
				surfaceNormals.emplace_back(hostOut[i].normal.x, hostOut[i].normal.y, hostOut[i].normal.z);

                // 보간된 색상 사용
                surfaceColors.emplace_back(
                    hostOut[i].color[0] / 255.f,
                    hostOut[i].color[1] / 255.f,
                    hostOut[i].color[2] / 255.f,
                    1.f
                );
            }

            // 3. 점(Point) 형태로 시각화 (보간 덕분에 점만 찍어도 면처럼 보입니다)
            VD::AddDiskBatch("ZeroCrossingSurface", surfacePoints, surfaceNormals, 0.05f, 16, surfaceColors, true);

            //// 디버깅을 위한 스파스 블록 가이드는 유지
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
};

REGISTER_APP(AppTSDFDevice, "AppTSDFDevice");
