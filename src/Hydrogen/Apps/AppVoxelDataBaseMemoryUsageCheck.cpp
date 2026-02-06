#include "Apps.h"

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

class AppVoxelDataBaseMemoryUsageCheck : public App
{
public:
    virtual void Execute() override
    {
        auto startTime = std::chrono::high_resolution_clock::now();

        // 1. 초기 메모리 상태 기록
        auto [initialUsed, totalGpu] = CheckDeviceMemory("초기 상태");

        VVV::VoxelDataBase voxelDb;
        uint32_t maxBlocks = 80000;

        PrintAllocationPrediction(maxBlocks);

        // 2. 할당 및 증분량 체크
        voxelDb.Allocate(maxBlocks);
        auto [afterAllocUsed, ignore1] = CheckDeviceMemory("할당 완료");
        PrintMemoryDelta("할당 후 증분", (size_t)initialUsed, (size_t)afterAllocUsed);

        // 3. 데이터 로드 및 업데이트
        if (!ProcessVoxelData(voxelDb, maxBlocks, (size_t)initialUsed))
        {
            voxelDb.Free();
            return;
        }

        // 4. 해제 및 최종 누수 점검
        voxelDb.Free();
        auto [finalUsed, ignore2] = CheckDeviceMemory("해제 완료");
        PrintMemoryDelta("최종 잔류량", (size_t)initialUsed, (size_t)finalUsed);

        PrintFinalSummary((size_t)initialUsed, (size_t)finalUsed, startTime);
    }

private:
    void PrintMemoryDelta(const std::string& label, size_t initial, size_t current)
    {
        double deltaGb = (double)(current - initial) / (1024.0 * 1024.0 * 1024.0);
        printf("    [Delta] %-15s : %+.4f GB\n", label.c_str(), deltaGb);
    }

    void PrintAllocationPrediction(uint32_t maxBlocks)
    {
        size_t blockBytes = sizeof(VVV::VoxelBlock) * (size_t)maxBlocks;
        size_t hashBytes = sizeof(uint64_t) * (size_t)maxBlocks;
        size_t theoryTotal = blockBytes + hashBytes + sizeof(uint32_t);

        printf("\n>>> [메모리 분석: 할당 예측]\n");
        printf("    - 설정 블록 수       : %u 개\n", maxBlocks);
        printf("    - 예상 메모리 점유   : %.4f GB\n", (double)theoryTotal / (1024.0 * 1024.0 * 1024.0));
    }

    bool ProcessVoxelData(VVV::VoxelDataBase& voxelDb, uint32_t maxBlocks, size_t initialUsed)
    {
        PLYFormat ply;
        if (!ply.Deserialize("D:\\Resources\\Default\\VoxelValues_Unlock.ply"))
        {
            printf("!!! PLY 파일 로드 실패\n");
            return false;
        }

        size_t nPoints = ply.GetPoints().size();
        std::vector<VVV::Vector3f> points(nPoints);
        std::vector<VVV::Vector3b> colors(nPoints);

        for (size_t i = 0; i < nPoints; i++)
        {
            auto& p = ply.GetPoints()[i];
            points[i] = { p.x(), p.y(), p.z() };

            if (!ply.GetColors().empty())
            {
                auto& c = ply.GetColors()[i];
                colors[i] = {
                    static_cast<uint8_t>(c.x() * 255.0f),
                    static_cast<uint8_t>(c.y() * 255.0f),
                    static_cast<uint8_t>(c.z() * 255.0f)
                };
            }
            else
            {
                colors[i] = { 255, 255, 255 };
            }
        }

        float blockSize = 0.8f;

        printf("\n>>> [GPU 연산] 복셀 데이터 생성 중...\n");
        TS(VVV_UpdateVoxelFromPoints);
        VVV::Matrix4f identity = VVV::Matrix4f::Identity();
        voxelDb.OccupyVoxelFromPoints(identity, points.data(), colors.data(), (uint32_t)nPoints, blockSize, 1);
        cudaDeviceSynchronize();
        TE(VVV_UpdateVoxelFromPoints);

        // 사용된 블록 수 확인
        uint32_t activeBlocksCount = 0;
        cudaMemcpy(&activeBlocksCount, voxelDb.d_blockCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        auto [afterUpdateUsed, ignore] = CheckDeviceMemory("업데이트 완료");
        printf("    - 사용 중인 블록 수  : %u / %u\n", activeBlocksCount, maxBlocks);
        PrintMemoryDelta("업데이트 후 증분", (size_t)initialUsed, (size_t)afterUpdateUsed);

        VisualizeVoxelsBatch(voxelDb, maxBlocks, blockSize, activeBlocksCount);

        return true;
    }

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

            for (uint32_t i = 0; i < maxBlocks; ++i)
            {
                uint64_t mKey = hostHashTable[i];
                if (mKey != 0 && mKey != 0xFFFFFFFFFFFFFFFFULL)
                {
                    VVV::Morton64 morton(mKey);
                    VVV::Vector3f bPos = morton.ToPosition(blockSize);
                    blockCenters.emplace_back(bPos.x, bPos.y, bPos.z);
                }
            }
            VD::AddWiredBoxBatch("LDE_SparseDataBlocks", blockCenters, Eigen::Vector3f(blockSize, blockSize, blockSize), Eigen::Vector4f(0, 1, 0, 0.2f));
        }

        printf("\n>>> [최종 리포트]\n");
        printf("    - 추출된 복셀 수     : %u 개\n", finalCnt);
        printf("    - 해시 적재율        : %.2f%% (%u / %u)\n",
            (double)activeBlocksCount / maxBlocks * 100.0, activeBlocksCount, maxBlocks);
    }

    void PrintFinalSummary(size_t initialUsed, size_t finalUsed, std::chrono::steady_clock::time_point startTime)
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        printf("\n>>> [최종 메모리 점검]\n");
        printf("    - 초기 대비 잔류량   : %+.4f MB\n", (double)(finalUsed - initialUsed) / (1024.0 * 1024.0));
        printf("    - 총 소요 시간       : %.4fs\n", std::chrono::duration<double>(endTime - startTime).count());
        printf("==========================================================\n");
    }
};

REGISTER_APP(AppVoxelDataBaseMemoryUsageCheck, "AppVoxelDataBaseMemoryUsageCheck");