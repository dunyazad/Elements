#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdio.h>
#include <vector>

#include <robin_hood/robin_hood.h>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <Helium/IVisualDebugging.h>
using VD = IVisualDebugging;

#include <Helium/Serialization.hpp>
#include <Helium/Color.hpp>

#include <Core/Core.h>
#include <Core/DataStructures/SparseCells.h>
#include <Core/DataStructures/VoxelDataBase.h>
using namespace Huvitz;

#ifndef CUDA_TS
#define CUDA_TS(name) \
    cudaEvent_t time_##name##_start;\
    cudaEvent_t time_##name##_stop;\
    cudaEventCreate(&time_##name##_start);\
    cudaEventCreate(&time_##name##_stop);\
    cudaEventRecord(time_##name##_start);
#endif

#ifndef CUDA_TE
#define CUDA_TE(name) \
    cudaEventRecord(time_##name##_stop);\
    cudaEventSynchronize(time_##name##_stop);\
    float time_##name##_miliseconds = 0.0f;\
    cudaEventElapsedTime(&time_##name##_miliseconds, time_##name##_start, time_##name##_stop);\
    printf("[<[%s]>] %f ms\n", #name, time_##name##_miliseconds);\
    cudaEventDestroy(time_##name##_start);\
    cudaEventDestroy(time_##name##_stop);
#endif

//typedef unsigned char uchar;
//typedef float voxel_value_t;
//
//namespace Huvitz
//{
//	struct VoxelExtraAttrib {
//		static const VoxelExtraAttrib Zero;
//
//		uchar deepLearningClass; // enum DL_Class_Names
//		uchar materialID; // 0: other 255: tooth
//		unsigned short startPatchID;	// 복셀의 조합에 영향을 미친 첫 패치 번호
//		unsigned int flags : 2; // 복셀의 잠금상태 등의 상태를 저장.VOXEL_FLAG_**** _BIT 로 비교
//		unsigned int label : 30;	// 클러스터링 된 레이블 번호
//
//		uint8_t colorMap[4]; // colors[3], alpha (신뢰도)
//	};
//
//	struct Voxel {
//		voxel_value_t		value;
//		unsigned short		valueCount;
//		Eigen::Vector3f		normal;
//		Eigen::Vector3b		color;
//
//		Eigen::Vector3b		color_list[3];
//		uint8_t				color_score[3];
//
//		char				segmentation;
//		VoxelExtraAttrib	extraAttrib;
//	};
//}

class CUDAManager
{
public:
	struct RegModule
	{
		CUDAManager* manager;
		cudaStream_t stream;
		cudaStream_t computeStream;
		CUDAManager* GetManager() { return manager; }
		cudaStream_t GetStream() { return stream; }
		cudaStream_t GetComputeStream() { return computeStream; }
		cudaStream_t CreateStream()
		{
			cudaStream_t newStream;
			cudaStreamCreate(&newStream);
			return newStream;
		}
		void DestroyStream(cudaStream_t stream)
		{
			cudaStreamDestroy(stream);
		}
	};
	std::unique_ptr<RegModule> regModule;
	CUDAManager()
	{
		regModule = std::make_unique<RegModule>();
		regModule->manager = this;
		cudaStreamCreate(&regModule->stream);
		cudaStreamCreate(&regModule->computeStream);
	}
	~CUDAManager()
	{
		cudaStreamDestroy(regModule->stream);
		cudaStreamDestroy(regModule->computeStream);
	}
};




#define DTSDF 0
#define CLUSTERING 1

#include "Apps.h"
class AppVoxelDataBase : public App
{
public:
	/*
	virtual void Execute() override
	{
		nvDriverSetting.forceGPUPerformance();

		cudaStream_t stream = nullptr;
		cached_allocator* alloc = nullptr;

		cudaFree(0); // warm up

		VoxelDataBase<DirectionalVoxel<DummyVoxel>> voxelDataBase;
		if (voxelDataBase.Initialize(80000))
		{
			CheckDeviceMemory("After VoxelDataBase Initialization");

			bool usingSerializedData = false;
			if (false == usingSerializedData)
			{
				DataFrameReader reader("D:\\Debug\\voxel_database_integration.df");

				CUDA_TS(Total);

				VoxelDataBaseIntegrationParameters integrationParams;
				integrationParams.d_depthMap = nullptr;
				integrationParams.d_normalMap = nullptr;
				integrationParams.d_colorMap = nullptr;

				while (auto frame = reader.next())
				{
					CUDA_TS(Patch);

					size_t offset = 0;
					uint8_t* basePtr = frame->data.data();

					unsigned int mapWidth = *reinterpret_cast<unsigned int*>(basePtr + offset); offset += sizeof(unsigned int);
					unsigned int mapHeight = *reinterpret_cast<unsigned int*>(basePtr + offset); offset += sizeof(unsigned int);
					size_t mapSize = (size_t)mapWidth * mapHeight;

					std::vector<Eigen::Vector3f> h_depthMap(mapSize);
					memcpy(h_depthMap.data(), basePtr + offset, sizeof(Eigen::Vector3f) * mapSize);
					offset += sizeof(Eigen::Vector3f) * mapSize;

					std::vector<Eigen::Vector3f> h_normalMap(mapSize);
					memcpy(h_normalMap.data(), basePtr + offset, sizeof(Eigen::Vector3f) * mapSize);
					offset += sizeof(Eigen::Vector3f) * mapSize;

					unsigned int* h_colorMap = reinterpret_cast<unsigned int*>(basePtr + offset);
					offset += sizeof(unsigned int) * mapSize * 3;

					float m[16];
					memcpy(m, basePtr + offset, sizeof(float) * 16);
					offset += sizeof(float) * 16;
					Eigen::Matrix4f transform;
					transform.data()[0] = m[0]; transform.data()[1] = m[1]; transform.data()[2] = m[2]; transform.data()[3] = m[3];
					transform.data()[4] = m[4]; transform.data()[5] = m[5]; transform.data()[6] = m[6]; transform.data()[7] = m[7];
					transform.data()[8] = m[8]; transform.data()[9] = m[9]; transform.data()[10] = m[10]; transform.data()[11] = m[11];
					transform.data()[12] = m[12]; transform.data()[13] = m[13]; transform.data()[14] = m[14]; transform.data()[15] = m[15];

					float voxelSize = *reinterpret_cast<float*>(basePtr + offset); offset += sizeof(float);

					integrationParams.mapWidth = mapWidth;
					integrationParams.mapHeight = mapHeight;
					integrationParams.transform = transform;
					integrationParams.voxelSize = voxelSize;

					printf("Integrating frame %u, map size: %u x %u, voxel size: %f\n", frame->frameIndex, mapWidth, mapHeight, voxelSize);

					if (nullptr == integrationParams.d_depthMap)
					{
						cudaMallocAsync(&integrationParams.d_depthMap, sizeof(Eigen::Vector3f) * mapSize, nullptr);
						cudaMallocAsync(&integrationParams.d_normalMap, sizeof(Eigen::Vector3f) * mapSize, nullptr);
						cudaMallocAsync(&integrationParams.d_colorMap, sizeof(unsigned int) * mapSize * 3, nullptr);
					}

					cudaMemcpyAsync(integrationParams.d_depthMap, h_depthMap.data(), sizeof(Eigen::Vector3f) * mapSize, cudaMemcpyHostToDevice, nullptr);
					cudaMemcpyAsync(integrationParams.d_normalMap, h_normalMap.data(), sizeof(Eigen::Vector3f) * mapSize, cudaMemcpyHostToDevice, nullptr);
					cudaMemcpyAsync(integrationParams.d_colorMap, h_colorMap, sizeof(unsigned int) * mapSize * 3, cudaMemcpyHostToDevice, nullptr);

					voxelDataBase.IntegrateDirectional(&integrationParams, nullptr, nullptr);
					voxelDataBase.PerFrameFilter(&integrationParams, nullptr, nullptr);

					//voxelDataBase.ApplyIntraRegionNoiseFilter(&integrationParams, &CUDA_MANAGER->regModule->alloc, CUDA_MANAGER->regModule->GetStream());

					{
						auto error = cudaGetLastError();
						if (error != cudaSuccess) {
							printf("CUDA error : %s\n", cudaGetErrorString(error));
						}
					}

					//PLYFormat ply;
					//ply.SetDataType(PLYFormat::PLYFormatDataType::ASCII);
					//for (size_t i = 0; i < mapWidth * mapHeight; i++)
					//{
					//	auto& d = h_depthMap[i];
					//	if (false == VECTOR3F_VALID_(d)) continue;

					//	Eigen::Vector3f p = (transform * Eigen::Vector4f(d.x(), d.y(), d.z(), 1.0f)).head<3>();

					//	ply.AddPoint(p.x(), p.y(), p.z());
					//	auto& n = h_normalMap[i];
					//	ply.AddNormal(n.x(), n.y(), n.z());
					//	auto& r = h_colorMap[i * 3];
					//	auto& g = h_colorMap[i * 3 + 1];
					//	auto& b = h_colorMap[i * 3 + 2];
					//	ply.AddColor((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);
					//}
					//ply.Serialize(L"D:\\Debug\\InputFrames\\InputFrame_" + std::to_wstring(frame->frameIndex) + L".ply");

					CUDA_TE(Patch);

					//static int count = 0;
					//count++;

					//if(16 == count)
					//{
					//	break;
					//}
				}
				CUDA_TE(Total);

				cudaFree(integrationParams.d_depthMap);
				cudaFree(integrationParams.d_normalMap);
				cudaFree(integrationParams.d_colorMap);

				//voxelDataBase.Serialize(L"D:\\Debug\\VoxelDataBase.VDB");
			}
			else
			{
				voxelDataBase.Deserialize(L"D:\\Debug\\VoxelDataBase.VDB");
			}

			uint32_t maxVoxelOut = voxelDataBase.GetMaxBlockCount() * 512;

			printf("maxVoxelOut: %u\n", maxVoxelOut);

			ExtractedVoxel* d_out = nullptr;
			uint32_t* d_count = nullptr;
			cudaMalloc(&d_out, sizeof(ExtractedVoxel) * maxVoxelOut);
			cudaMalloc(&d_count, sizeof(uint32_t));

			VoxelDataBaseExtractionParameters extractParams;
			extractParams.mode = VoxelDataBaseExtractionParameters::Mode::ZeroCrossing;
			//extractParams.mode = VoxelDataBaseExtractionParameters::Mode::AllOccupied;
			extractParams.d_out = d_out;
			extractParams.d_count = d_count;
			extractParams.maxOut = maxVoxelOut;

			voxelDataBase.ExtractDirectional(&extractParams, nullptr, nullptr);
			//cudaStreamSynchronize(CUDA_MANAGER->regModule->GetStream());

			uint32_t resultCount = 0;
			cudaMemcpy(&resultCount, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

			printf("Extracted voxel count: %u\n", resultCount);

			ExtractedVoxel* h_result = new ExtractedVoxel[resultCount];
			cudaMemcpy(h_result, d_out, sizeof(ExtractedVoxel) * resultCount, cudaMemcpyDeviceToHost);

			//std::vector<ExtractedVoxel> h_result(resultCount);
			//cudaMemcpy(h_result.data(), d_out, sizeof(ExtractedVoxel)* resultCount, cudaMemcpyDeviceToHost);

			cudaDeviceSynchronize();

			PLYFormat ply;
			for (size_t i = 0; i < resultCount; i++)
			{
				ExtractedVoxel& voxel = h_result[i];

				//printf("%f, %f, %f\n", voxel.position.x(), voxel.position.y(), voxel.position.z());

				ply.AddPoint(voxel.position.x(), voxel.position.y(), voxel.position.z());
				ply.AddNormal(voxel.normal.x(), voxel.normal.y(), voxel.normal.z());
				ply.AddColor(voxel.color[0], voxel.color[1], voxel.color[2]);

				VD::AddSphere("points",
					{ voxel.position.x(), voxel.position.y(), voxel.position.z() },
					{ voxel.normal.x(), voxel.normal.y(), voxel.normal.z() },
					0.05f, { (float)voxel.color[0] / 255.0f, (float)voxel.color[1] / 255.0f, (float)voxel.color[2] / 255.0f, 1.0f });
			}
			ply.Serialize("D:\\Debug\\ExtractedVoxels.ply");

			delete[] h_result;

			cudaFree(d_out);
			cudaFree(d_count);

			voxelDataBase.Terminate();
		}
		else
		{
			CheckDeviceMemory("After VoxelDataBase Initialization");
			printf("Failed to initialize VoxelDataBase\n");
		}
	}
	*/

#if DTSDF
virtual void Execute() override
{
	nvDriverSetting.forceGPUPerformance();

	{
		thrust::device_vector<int> d(1 << 20);
		thrust::sequence(d.begin(), d.end());
		thrust::sort(d.begin(), d.end(), thrust::greater<int>());
		cudaDeviceSynchronize();
	}

	cudaStream_t stream = nullptr;
	cached_allocator* alloc = nullptr;

	cudaFree(0); // warm up

	VoxelDataBase<DirectionalVoxel<DummyVoxel>> voxelDataBase;
	if (voxelDataBase.Initialize(80000))
	{
		CheckDeviceMemory("After VoxelDataBase Initialization");

		bool usingSerializedData = false;
		if (false == usingSerializedData)
		{
			DataFrameReader reader("D:\\Debug\\voxel_database_integration.df");

			CUDA_TS(Total);

			VoxelDataBaseIntegrationParameters integrationParams;
			integrationParams.d_depthMap = nullptr;
			integrationParams.d_normalMap = nullptr;
			integrationParams.d_colorMap = nullptr;

			SparseCells cells;
			unsigned int* d_labels = nullptr;
			cudaMalloc(&d_labels, sizeof(unsigned int) * 226 * 200);

			while (auto frame = reader.next())
			{
				//CUDA_TS(Patch);

				size_t offset = 0;
				uint8_t* basePtr = frame->data.data();

				unsigned int mapWidth = *reinterpret_cast<unsigned int*>(basePtr + offset); offset += sizeof(unsigned int);
				unsigned int mapHeight = *reinterpret_cast<unsigned int*>(basePtr + offset); offset += sizeof(unsigned int);
				size_t mapSize = (size_t)mapWidth * mapHeight;

				std::vector<Eigen::Vector3f> h_depthMap(mapSize);
				memcpy(h_depthMap.data(), basePtr + offset, sizeof(Eigen::Vector3f) * mapSize);
				offset += sizeof(Eigen::Vector3f) * mapSize;

				std::vector<Eigen::Vector3f> h_normalMap(mapSize);
				memcpy(h_normalMap.data(), basePtr + offset, sizeof(Eigen::Vector3f) * mapSize);
				offset += sizeof(Eigen::Vector3f) * mapSize;

				unsigned int* h_colorMap = reinterpret_cast<unsigned int*>(basePtr + offset);
				offset += sizeof(unsigned int) * mapSize * 3;

				float m[16];
				memcpy(m, basePtr + offset, sizeof(float) * 16);
				offset += sizeof(float) * 16;
				Eigen::Matrix4f transform;
				transform.data()[0] = m[0]; transform.data()[1] = m[1]; transform.data()[2] = m[2]; transform.data()[3] = m[3];
				transform.data()[4] = m[4]; transform.data()[5] = m[5]; transform.data()[6] = m[6]; transform.data()[7] = m[7];
				transform.data()[8] = m[8]; transform.data()[9] = m[9]; transform.data()[10] = m[10]; transform.data()[11] = m[11];
				transform.data()[12] = m[12]; transform.data()[13] = m[13]; transform.data()[14] = m[14]; transform.data()[15] = m[15];

				float voxelSize = *reinterpret_cast<float*>(basePtr + offset); offset += sizeof(float);

				integrationParams.mapWidth = mapWidth;
				integrationParams.mapHeight = mapHeight;
				integrationParams.transform = transform;
				integrationParams.voxelSize = voxelSize;

				printf("Integrating frame %u, map size: %u x %u, voxel size: %f\n", frame->frameIndex, mapWidth, mapHeight, voxelSize);

				if (nullptr == integrationParams.d_depthMap)
				{
					cudaMallocAsync(&integrationParams.d_depthMap, sizeof(Eigen::Vector3f) * mapSize, nullptr);
					cudaMallocAsync(&integrationParams.d_normalMap, sizeof(Eigen::Vector3f) * mapSize, nullptr);
					cudaMallocAsync(&integrationParams.d_colorMap, sizeof(unsigned int) * mapSize * 3, nullptr);
				}

				cudaMemcpyAsync(integrationParams.d_depthMap, h_depthMap.data(), sizeof(Eigen::Vector3f) * mapSize, cudaMemcpyHostToDevice, nullptr);
				cudaMemcpyAsync(integrationParams.d_normalMap, h_normalMap.data(), sizeof(Eigen::Vector3f) * mapSize, cudaMemcpyHostToDevice, nullptr);
				cudaMemcpyAsync(integrationParams.d_colorMap, h_colorMap, sizeof(unsigned int) * mapSize * 3, cudaMemcpyHostToDevice, nullptr);

				voxelDataBase.IntegrateDirectional(&integrationParams, nullptr, nullptr);
				voxelDataBase.PerFrameFilter(&integrationParams, nullptr, nullptr);

				//voxelDataBase.ApplyIntraRegionNoiseFilter(&integrationParams, &CUDA_MANAGER->regModule->alloc, CUDA_MANAGER->regModule->GetStream());

				{
					auto error = cudaGetLastError();
					if (error != cudaSuccess) {
						printf("CUDA error : %s\n", cudaGetErrorString(error));
					}
				}

				//PLYFormat ply;
				//ply.SetDataType(PLYFormat::PLYFormatDataType::ASCII);
				//for (size_t i = 0; i < mapWidth * mapHeight; i++)
				//{
				//	auto& d = h_depthMap[i];
				//	if (false == VECTOR3F_VALID_(d)) continue;

				//	Eigen::Vector3f p = (transform * Eigen::Vector4f(d.x(), d.y(), d.z(), 1.0f)).head<3>();

				//	ply.AddPoint(p.x(), p.y(), p.z());
				//	auto& n = h_normalMap[i];
				//	ply.AddNormal(n.x(), n.y(), n.z());
				//	auto& r = h_colorMap[i * 3];
				//	auto& g = h_colorMap[i * 3 + 1];
				//	auto& b = h_colorMap[i * 3 + 2];
				//	ply.AddColor((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);
				//}
				//ply.Serialize(L"D:\\Debug\\InputFrames\\InputFrame_" + std::to_wstring(frame->frameIndex) + L".ply");

				//CUDA_TE(Patch);

				//static int count = 0;
				//count++;

				//if (5 == count)
				//{
				//	break;
				//}
			}
			CUDA_TE(Total);

			cudaFree(integrationParams.d_depthMap);
			cudaFree(integrationParams.d_normalMap);
			cudaFree(integrationParams.d_colorMap);

			//voxelDataBase.Serialize(L"D:\\Debug\\VoxelDataBase.VDB");
		}
		else
		{
			voxelDataBase.Deserialize(L"D:\\Debug\\VoxelDataBase.VDB");
		}

		uint32_t maxVoxelOut = voxelDataBase.GetMaxBlockCount() * 512;

		printf("maxVoxelOut: %u\n", maxVoxelOut);

		ExtractedVoxel* d_out = nullptr;
		uint32_t* d_count = nullptr;
		cudaMalloc(&d_out, sizeof(ExtractedVoxel) * maxVoxelOut);
		cudaMalloc(&d_count, sizeof(uint32_t));

		VoxelDataBaseExtractionParameters extractParams;
		extractParams.mode = VoxelDataBaseExtractionParameters::Mode::ZeroCrossing;
		//extractParams.mode = VoxelDataBaseExtractionParameters::Mode::AllOccupied;
		extractParams.d_out = d_out;
		extractParams.d_count = d_count;
		extractParams.maxOut = maxVoxelOut;

		voxelDataBase.ExtractDirectional(&extractParams, nullptr, nullptr);
		//cudaStreamSynchronize(CUDA_MANAGER->regModule->GetStream());

		uint32_t resultCount = 0;
		cudaMemcpy(&resultCount, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

		printf("Extracted voxel count: %u\n", resultCount);

		ExtractedVoxel* h_result = new ExtractedVoxel[resultCount];
		cudaMemcpy(h_result, d_out, sizeof(ExtractedVoxel) * resultCount, cudaMemcpyDeviceToHost);

		//std::vector<ExtractedVoxel> h_result(resultCount);
		//cudaMemcpy(h_result.data(), d_out, sizeof(ExtractedVoxel)* resultCount, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		PLYFormat ply;
		for (size_t i = 0; i < resultCount; i++)
		{
			ExtractedVoxel& voxel = h_result[i];

			//printf("%f, %f, %f\n", voxel.position.x(), voxel.position.y(), voxel.position.z());

			ply.AddPoint(voxel.position.x(), voxel.position.y(), voxel.position.z());
			ply.AddNormal(voxel.normal.x(), voxel.normal.y(), voxel.normal.z());
			ply.AddColor((float)voxel.color[0] / 255.0f, (float)voxel.color[1] / 255.0f, (float)voxel.color[2] / 255.0f);

			VD::AddSphere("points",
				{ voxel.position.x(), voxel.position.y(), voxel.position.z() },
				{ voxel.normal.x(), voxel.normal.y(), voxel.normal.z() },
				0.05f, { (float)voxel.color[0] / 255.0f, (float)voxel.color[1] / 255.0f, (float)voxel.color[2] / 255.0f, 1.0f });
		}
		ply.Serialize("D:\\Debug\\ExtractedVoxels.ply");

		delete[] h_result;

		cudaFree(d_out);
		cudaFree(d_count);

		voxelDataBase.Terminate();
	}
	else
	{
		CheckDeviceMemory("After VoxelDataBase Initialization");
		printf("Failed to initialize VoxelDataBase\n");
	}
}
#endif


#if CLUSTERING
virtual void Execute() override
{
	nvDriverSetting.forceGPUPerformance();

	{
		thrust::device_vector<int> d(1 << 20);
		thrust::sequence(d.begin(), d.end());
		thrust::sort(d.begin(), d.end(), thrust::greater<int>());
		cudaDeviceSynchronize();
	}

	cudaStream_t stream = nullptr;
	cached_allocator* alloc = nullptr;

	cudaFree(0); // warm up

	PLYFormat ply;
	ply.Deserialize("D:\\Debug\\Compound.ply");
	//ply.Deserialize("D:\\Debug\\BasePoints.ply");

	std::vector<float3> h_positions(ply.GetPoints().size());
	std::vector<float3> h_normals(ply.GetPoints().size());
	std::vector<uchar3> h_colors(ply.GetPoints().size());

	for (size_t i = 0; i < ply.GetPoints().size(); i++)
	{
		auto& p = ply.GetPoints()[i];
		auto& n = ply.GetNormals()[i];
		auto& c = ply.GetColors()[i];

		h_positions[i] = make_float3(p.x(), p.y(), p.z());
		h_normals[i] = make_float3(n.x(), n.y(), n.z());
		h_colors[i] = make_uchar3((unsigned char)(c.x() * 255.0f), (unsigned char)(c.y() * 255.0f), (unsigned char)(c.z() * 255.0f));
	}

	PCD pcd;
	pcd.FromHostVectors(h_positions, h_normals, h_colors);

	SparseCells cells;

	CUDA_TS(BuildSparseCells);
	cells.Build(&pcd, 0.1f, nullptr);
	CUDA_TE(BuildSparseCells);

	unsigned int* d_labels = nullptr;
	CUDA_MALLOC(&d_labels, sizeof(unsigned int) * pcd.GetNumberOfPositions());

	CUDA_TS(ApplyClustering);
	cells.ApplyClustering(&pcd, d_labels, 0.125f, 15.0f * D2R, nullptr);
	CUDA_TE(ApplyClustering);

	std::vector<unsigned int> h_labels(pcd.GetNumberOfPositions());
	CUDA_COPY_D2H(h_labels.data(), d_labels, sizeof(unsigned int) * pcd.GetNumberOfPositions());

	CUDA_SAFE_FREE(d_labels);

	std::vector<float3> rps;
	std::vector<float3> rns;
	std::vector<uchar3> rcs;

	pcd.ToHostVectors(rps, rns, rcs);

	std::map<unsigned int, std::vector<size_t>> clusters;
	for (size_t i = 0; i < pcd.GetNumberOfPositions(); i++)
	{
		unsigned int label = h_labels[i];
		clusters[label].push_back(i);
	}

	auto colors = Color::GetContrastingColors(32);

	std::vector<std::pair<unsigned int, std::vector<size_t>>> sortedClusters(clusters.begin(), clusters.end());
	std::sort(sortedClusters.begin(), sortedClusters.end(),
		[](const std::pair<unsigned int, std::vector<size_t>>& a, const std::pair<unsigned int, std::vector<size_t>>& b) {
			return a.second.size() > b.second.size();
		});
	for (auto& lcPair : sortedClusters)
	{
		//if (100 < lcPair.second.size())
		{
			printf("Label: %u, Cluster Size: %zu\n", lcPair.first, lcPair.second.size());
		}
		break;
	}

	for (size_t i = 0; i < pcd.GetNumberOfPositions(); i++)
	{
		auto& p = rps[i];
		auto& n = rns[i];
		auto& c = rcs[i];
		auto& l = h_labels[i];

		auto color = colors[l % colors.size()];

		VD::AddSphere("PointCloud",
			{ p.x, p.y, p.z },
			{ n.x, n.y, n.z },
			0.05f,
			{ color.x(), color.y(), color.z(), 1.0f });
		//{ (float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f, 1.0f });
	}
}
#endif // CLUSTERING

};

REGISTER_APP(AppVoxelDataBase, "AppVoxelDataBase");
