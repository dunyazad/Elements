//#include "Apps.h"
//
//#include <VVV/VVV.h>
//#pragma comment(lib, "VVV.lib")
//
//#include <robin_hood/robin_hood.h>
//
//#include <Copper/Copper.h>
//#include <Copper/CuPointCloud.h>
//#include <Copper/CuSparseCells.h>
//#include <Copper/OperatorCollection/CuOperatorCollection.h>
//#include <Copper/CuVoxelStreaming.h>
//
//#include <Helium/HeliumCommon.h>
//#include <Helium/Serialization.hpp>
//#include <Helium/VisualDebugging.h>
//using VD = VisualDebugging;
//
//#include <VVV/VVV.h>
//#pragma comment(lib, "VVV.lib")
//
//typedef struct CamInfo_
//{
//    float cfx;
//    float cfy;
//    float ccx;
//    float ccy;
//    int cx;
//    int cy;
//    int img_width;
//    int img_height;
//    double R[9];
//    double T[3];
//    VVV::Vector3f dlpPos;
//    VVV::Vector3f camPos;
//    VVV::Matrix3f invMatTilt;
//    VVV::Matrix3f matTilt;
//
//    VVV::Matrix4f GetViewMatrix(const CamInfo_& info)
//    {
//        VVV::Matrix4f view = VVV::Matrix4f::Identity();
//        for (int i = 0; i < 3; ++i)
//        {
//            for (int j = 0; j < 3; ++j)
//            {
//                view(i, j) = (float)info.R[i * 3 + j];
//            }
//            view(i, 3) = (float)info.T[i];
//        }
//        return view;
//    }
//
//    VVV::Matrix4f GetProjectionMatrix(const CamInfo_& info, float n, float f)
//    {
//        VVV::Matrix4f proj = VVV::Matrix4f::Zero();
//        proj(0, 0) = 2.0f * info.cfx / info.img_width;
//        proj(0, 2) = 1.0f - (2.0f * info.ccx / info.img_width);
//        proj(1, 1) = 2.0f * info.cfy / info.img_height;
//        proj(1, 2) = (2.0f * info.ccy / info.img_height) - 1.0f;
//        proj(2, 2) = -(f + n) / (f - n);
//        proj(2, 3) = -(2.0f * f * n) / (f - n);
//        proj(3, 2) = -1.0f;
//        return proj;
//    }
//} CamInfo_;
//
//class AppICP : public App
//{
//public:
//	  virtual void Initialize() override
//	  {
//	  }
//
//    virtual void Execute() override
//    {
//		PLYFormat ply;
//		if (!ply.Deserialize("D:\\Resources\\Default\\Compound.ply")) return;
//
//		VVV::VoxelDataBase voxelDb;
//        uint32_t maxBlocks = 80000;
//
//        voxelDb.Allocate(maxBlocks);
//
//        size_t nPoints = ply.GetPoints().size();
//        std::vector<VVV::Vector3f> points(nPoints);
//        std::vector<VVV::Vector3b> colors(nPoints);
//
//        for (size_t i = 0; i < nPoints; i++)
//        {
//            auto& p = ply.GetPoints()[i];
//            points[i] = { p.x(), p.y(), p.z() };
//
//            if (!ply.GetColors().empty())
//            {
//                auto& c = ply.GetColors()[i];
//                colors[i] = {
//                    static_cast<uint8_t>(c.x() * 255.0f),
//                    static_cast<uint8_t>(c.y() * 255.0f),
//                    static_cast<uint8_t>(c.z() * 255.0f)
//                };
//            }
//            else
//            {
//                colors[i] = { 255, 255, 255 };
//            }
//        }
//
//        float blockSize = 0.8f;
//
//        TS(VVV_UpdateVoxelFromPoints);
//        VVV::Matrix4f identity = VVV::Matrix4f::Identity();
//        voxelDb.OccupyVoxelFromPoints(identity, points.data(), colors.data(), (uint32_t)nPoints, blockSize, 1);
//        cudaDeviceSynchronize();
//        TE(VVV_UpdateVoxelFromPoints);
//
//        // 사용된 블록 수 확인
//        uint32_t activeBlocksCount = 0;
//        cudaMemcpy(&activeBlocksCount, voxelDb.d_blockCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
//
//        uint32_t maxOut = 50000000;
//        std::vector<VVV::ExtractedVoxel> hostOut(maxOut);
//        uint32_t finalCnt = voxelDb.ExtractActiveVoxelsToHost(blockSize, hostOut.data(), maxOut);
//
//        if (finalCnt > 0)
//        {
//            uint32_t limit = (finalCnt > maxOut) ? maxOut : finalCnt;
//            float voxelDrawSize = (blockSize / 8.0f) * 0.9f;
//
//            //std::vector<Eigen::Vector3f> voxelCenters;
//            //std::vector<Eigen::Vector3f> voxelDimensions;
//            //std::vector<Eigen::Vector4f> voxelColors;
//            //voxelCenters.reserve(limit);
//            //voxelDimensions.reserve(limit);
//            //voxelColors.reserve(limit);
//
//            //for (uint32_t i = 0; i < limit; i++)
//            //{
//            //    voxelCenters.emplace_back(hostOut[i].position.x, hostOut[i].position.y, hostOut[i].position.z);
//            //    voxelDimensions.emplace_back(voxelDrawSize, voxelDrawSize, voxelDrawSize);
//            //    voxelColors.emplace_back(hostOut[i].color[0] / 255.f, hostOut[i].color[1] / 255.f, hostOut[i].color[2] / 255.f, 1.f);
//            //}
//            //VD::AddWiredBoxBatch("Voxels", voxelCenters, voxelDimensions, voxelColors);
//
//            std::vector<uint64_t> hostHashTable(maxBlocks);
//            cudaMemcpy(hostHashTable.data(), voxelDb.d_hashTable, sizeof(uint64_t) * maxBlocks, cudaMemcpyDeviceToHost);
//
//            std::vector<Eigen::Vector3f> blockCenters;
//            std::vector<Eigen::Vector3f> blockNormals;
//            blockCenters.reserve(activeBlocksCount);
//            blockNormals.reserve(activeBlocksCount);
//
//            for (uint32_t i = 0; i < maxBlocks; ++i)
//            {
//                uint64_t mKey = hostHashTable[i];
//                if (mKey != 0 && mKey != 0xFFFFFFFFFFFFFFFFULL)
//                {
//                    VVV::Morton64 morton(mKey);
//                    VVV::Vector3f bPos = morton.ToPosition(blockSize);
//                    blockCenters.emplace_back(bPos.x, bPos.y, bPos.z);
//					blockNormals.emplace_back(0.0f, 1.0f, 0.0f);
//                }
//            }
//            //VD::AddWiredBoxBatch("SparseDataBlocks", blockCenters, Eigen::Vector3f(blockSize, blockSize, blockSize), Eigen::Vector4f(0, 1, 0, 0.2f));
//
//			VD::AddDiskBatch("BlockCenters", blockCenters, blockNormals, voxelDrawSize, 16, Eigen::Vector4f(1, 0, 0, 1), true);
//        }
//
//        printf("\n>>> [최종 리포트]\n");
//        printf("    - 추출된 복셀 수     : %u 개\n", finalCnt);
//        printf("    - 해시 적재율        : %.2f%% (%u / %u)\n",
//            (double)activeBlocksCount / maxBlocks * 100.0, activeBlocksCount, maxBlocks);
//
//        voxelDb.Free();
//        return;
//
//
//
//        std::ifstream ifs("D:\\Resources\\Default\\Patches.bin", std::ios::binary);
//        if (!ifs.is_open())
//        {
//			std::cout << "[Error] Failed to open Patches.bin" << std::endl;
//            return;
//        }
//
//        CamInfo_ cam;
//        Eigen::Matrix4f camRT;
//        ifs.read(reinterpret_cast<char*>(&cam), sizeof(CamInfo_));
//        ifs.read(reinterpret_cast<char*>(camRT.data()), sizeof(float) * 16);
//
//        size_t numberOfPatches = 0;
//        ifs.read(reinterpret_cast<char*>(&numberOfPatches), sizeof(size_t));
//
//		robin_hood::unordered_map<size_t, std::tuple<int, Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector4f>> donwnSampling;
//
//        for (size_t i = 0; i < numberOfPatches; i++)
//        {
//            TS(patch);
//            size_t patchIndex = 0;
//            ifs.read(reinterpret_cast<char*>(&patchIndex), sizeof(size_t));
//            Eigen::Matrix4f rt0;
//            ifs.read(reinterpret_cast<char*>(rt0.data()), sizeof(float) * 16);
//			Eigen::Vector3f aabbMin0, aabbMax0;
//			ifs.read(reinterpret_cast<char*>(aabbMin0.data()), sizeof(float) * 3);
//			ifs.read(reinterpret_cast<char*>(aabbMax0.data()), sizeof(float) * 3);
//            Eigen::Matrix4f rt45;
//            ifs.read(reinterpret_cast<char*>(rt45.data()), sizeof(float) * 16);
//			Eigen::Vector3f aabbMin45, aabbMax45;
//			ifs.read(reinterpret_cast<char*>(aabbMin45.data()), sizeof(float) * 3);
//			ifs.read(reinterpret_cast<char*>(aabbMax45.data()), sizeof(float) * 3);
//
//            size_t numPts = 0;
//            ifs.read(reinterpret_cast<char*>(&numPts), sizeof(size_t));
//
//            std::vector<Eigen::Vector3f> pts(numPts), normals(numPts);
//            std::vector<Eigen::Vector3b> colors(numPts);
//            ifs.read(reinterpret_cast<char*>(pts.data()), sizeof(Eigen::Vector3f) * numPts);
//            ifs.read(reinterpret_cast<char*>(normals.data()), sizeof(Eigen::Vector3f) * numPts);
//            ifs.read(reinterpret_cast<char*>(colors.data()), sizeof(Eigen::Vector3b) * numPts);
//
//            size_t numPts45 = 0;
//            ifs.read(reinterpret_cast<char*>(&numPts45), sizeof(size_t));
//            ifs.seekg(numPts45 * (sizeof(Eigen::Vector3f) * 2), std::ios::cur);
//
//            Eigen::Vector3f sensorPos = rt0.block<3, 1>(0, 3);
//            Eigen::Matrix3f rot = rt0.block<3, 3>(0, 0);
//
//			CuPointCloud sourcePointCloud;
//			sourcePointCloud.FromHostPointers(
//				(float3*)pts.data(),
//				(float3*)normals.data(),
//				(uchar3*)colors.data(),
//				numPts,
//                {aabbMin0.x(), aabbMin0.y(), aabbMin0.z()},
//				{aabbMax0.x(), aabbMax0.y(), aabbMax0.z()});
//
//			CuSparseCells sourceCells;
//			sourceCells.Build(&sourcePointCloud, 0.3f);
//
//			auto activeSourceCells = sourceCells.GetActiveCellStats(&sourcePointCloud);
//
//
//			//sourcePointCloud.GlobalRegistration(targetPointCloud, rt0, 0.1f, 100);
//
//    //        for (size_t j = 0; j < numPts; ++j)
//    //        {
//    //            Eigen::Vector3f pW = rot * pts[j] + sensorPos;
//    //            Eigen::Vector3f nW = (rot * normals[j]).normalized();
//    //            //VD::AddSphere("PointCloud", pW, nW, 0.05f, { (float)colors[j].x() / 255.0f, (float)colors[j].y() / 255.0f , (float)colors[j].z() / 255.0f , 1.0f });
//				////source_points.push_back(pW);
//				////source_normals.push_back(nW);
//				////source_colors.push_back(Eigen::Vector4f{ (float)colors[j].x() / 255.0f, (float)colors[j].y() / 255.0f , (float)colors[j].z() / 255.0f , 1.0f });
//
//				//auto x = static_cast<size_t>(std::floor(pW.x() / 0.1f));
//    //            auto y = static_cast<size_t>(std::floor(pW.y() / 0.1f));
//    //            auto z = static_cast<size_t>(std::floor(pW.z() / 0.1f));
//				//auto hash = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
//    //            if (donwnSampling.find(hash) == donwnSampling.end())
//    //            {
//    //                donwnSampling[hash] = std::make_tuple(1, pW, nW, Eigen::Vector4f{ (float)colors[j].x() / 255.0f, (float)colors[j].y() / 255.0f, (float)colors[j].z() / 255.0f, 1.0f });
//    //            }
//    //            else
//    //            {
//    //                auto& tup = donwnSampling[hash];
//    //                std::get<0>(tup) += 1;
//    //                std::get<1>(tup) += pW;
//    //                std::get<2>(tup) += nW;
//				//	std::get<3>(tup) += Eigen::Vector4f{ (float)colors[j].x() / 255.0f, (float)colors[j].y() / 255.0f, (float)colors[j].z() / 255.0f, 1.0f };
//    //            }
//    //        }
//
//            {
//				//std::string patchTag = "SourcePatch_" + std::to_string(i);
//                //VD::AddSphereBatch("patchTag", source_points, source_normals, 0.05f, source_colors);
//                //source_points.clear();
//				//source_normals.clear();
//                //source_colors.clear();
//            }
//
//            TE(patch);
//
//            break;
//        }
//
// /*       for (auto& kvp : donwnSampling)
//        {
//			auto& [count, pSum, nSum, cSum] = kvp.second;
//            Eigen::Vector3f pAvg = pSum / static_cast<float>(count);
//			Eigen::Vector3f nAvg = nSum.normalized();
//			Eigen::Vector4f cAvg = cSum / static_cast<float>(count);
//
//			VD::AddSphere("DownSampled", pAvg, nAvg, 0.05f, { cAvg.x(), cAvg.y(), cAvg.z(), cAvg.w() });
//        }*/
//    }
//};
//
//REGISTER_APP(AppICP, "AppICP");
