#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
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



class Pipeline
{
public:
	PCD compound;
	
	PCD localCompound;
	unsigned int* d_localCompoundLabels = nullptr;
	unsigned int* d_localCompoundSubLabels = nullptr;

	PCD patch;
	Eigen::Matrix4f patchRT;
	Huvitz::cuAABB patchAABB;

	VoxelDataBase<DirectionalVoxel<DummyVoxel>> voxelDataBase;
};


#define DTSDF 1
#define CLUSTERING 0
#define FILTERING 0

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

	CheckDeviceMemory("Before VoxelDataBase Initialization");

	VoxelDataBase<DirectionalVoxel<DummyVoxel>> voxelDataBase;
	if (voxelDataBase.Initialize(160000))
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

				CUDA_TS(IntegrateDirectional);
				voxelDataBase.IntegrateDirectional(&integrationParams, nullptr, nullptr);
				//voxelDataBase.PerFrameFilter(&integrationParams, nullptr, nullptr);

				//voxelDataBase.ApplyIntraRegionNoiseFilter(&integrationParams, &CUDA_MANAGER->regModule->alloc, CUDA_MANAGER->regModule->GetStream());
				CUDA_TE(IntegrateDirectional);

				{
					auto error = cudaGetLastError();
					if (error != cudaSuccess) {
						printf("CUDA error : %s\n", cudaGetErrorString(error));
					}
				}

				PLYFormat ply;
				//ply.SetDataType(PLYFormat::PLYFormatDataType::ASCII);
				for (size_t i = 0; i < mapWidth * mapHeight; i++)
				{
					auto& d = h_depthMap[i];
					if (false == VECTOR3F_VALID_(d)) continue;

					Eigen::Vector3f p = (transform * Eigen::Vector4f(d.x(), d.y(), d.z(), 1.0f)).head<3>();

					ply.AddPoint(p.x(), p.y(), p.z());
					auto& n = h_normalMap[i];
					ply.AddNormal(n.x(), n.y(), n.z());
					auto& r = h_colorMap[i * 3];
					auto& g = h_colorMap[i * 3 + 1];
					auto& b = h_colorMap[i * 3 + 2];
					ply.AddColor((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);
				}
				ply.Serialize("D:\\Debug\\InputFrames\\InputFrame_" + std::to_string(frame->frameIndex) + ".ply");

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

			voxelDataBase.Serialize(L"D:\\Debug\\VoxelDataBase.VDB");
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

		CUDA_TS(Extract);
		voxelDataBase.ExtractDirectional(&extractParams, nullptr, nullptr);
		//cudaStreamSynchronize(CUDA_MANAGER->regModule->GetStream());
		CUDA_TE(Extract);

		uint32_t resultCount = 0;
		cudaMemcpy(&resultCount, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

		printf("Extracted voxel count: %u\n", resultCount);

		ExtractedVoxel* h_result = new ExtractedVoxel[resultCount];
		cudaMemcpy(h_result, d_out, sizeof(ExtractedVoxel) * resultCount, cudaMemcpyDeviceToHost);

		//std::vector<ExtractedVoxel> h_result(resultCount);
		//cudaMemcpy(h_result.data(), d_out, sizeof(ExtractedVoxel)* resultCount, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

		PCD pcd(resultCount);
		auto d_pcd_positions = pcd.GetPositions();
		auto d_pcd_normals = pcd.GetNormals();
		auto d_pcd_colors = pcd.GetColors();

		thrust::for_each_n(
			thrust::device,
			thrust::make_counting_iterator<size_t>(0), resultCount, [=]__device__(size_t i) {
			ExtractedVoxel& voxel = d_out[i];
			d_pcd_positions[i] = { voxel.position.x(), voxel.position.y(), voxel.position.z() };
			d_pcd_normals[i] = { voxel.normal.x(), voxel.normal.y(), voxel.normal.z() };
			d_pcd_colors[i] = { (unsigned char)voxel.color[0], (unsigned char)voxel.color[1], (unsigned char)voxel.color[2] };
			});

		pcd.SetNumberOfPositions(resultCount);
		pcd.RebuildAABB();
		pcd.FillIsAlive(true);

		pcd.Clustering(0.1f, 0.125f/*, 20.0f * D2R*/, nullptr, nullptr);
		pcd.BuildLabelCountTable();

		std::vector<unsigned int> h_labels(pcd.GetNumberOfPositions());
		CUDA_COPY_D2H(h_labels.data(), pcd.GetLabels(), sizeof(unsigned int)* pcd.GetNumberOfPositions());

		std::vector<unsigned int> h_labelCounts(pcd.GetLabelCountTableSize());
		CUDA_COPY_D2H(h_labelCounts.data(), pcd.GetLabelCountTable(), sizeof(unsigned int)* pcd.GetLabelCountTableSize());

		std::vector<float3> rps;
		std::vector<float3> rns;
		std::vector<uchar3> rcs;

		pcd.ToHostVectors(rps, rns, rcs);

		unsigned int dominantLabel = (unsigned int)(
			std::max_element(h_labelCounts.begin(), h_labelCounts.end()) - h_labelCounts.begin()
			);

		PLYFormat ply;
		for (size_t i = 0; i < pcd.GetNumberOfPositions(); i++)
		{
			auto& p = rps[i];
			auto& n = rns[i];
			auto& c = rcs[i];
			auto& l = h_labels[i];
			auto& lc = h_labelCounts[l];
			if (l != dominantLabel) continue;

			ply.AddPoint(p.x, p.y, p.z);
			ply.AddNormal(n.x, n.y, n.z);
			ply.AddColor((float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f);

			VD::AddSphere("PointCloud",
				{ p.x, p.y, p.z },
				{ n.x, n.y, n.z },
				0.05f,
				//{ color.x(), color.y(), color.z(), 1.0f });
			    { (float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f, 1.0f });
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

	////CUDA_TS(Downsample);
	//pcd = pcd.Downsample(0.1f);
	////CUDA_TE(Downsample);

	SparseCells cells;

	CUDA_TS(BuildSparseCells);
	cells.Build(&pcd, 0.1f, nullptr);
	CUDA_TE(BuildSparseCells);

	unsigned int* d_labels = nullptr;
	CUDA_MALLOC(&d_labels, sizeof(unsigned int) * pcd.GetNumberOfPositions());

	CUDA_TS(ApplyClustering);
	//cells.ApplyClustering(&pcd, d_labels, 1.0f * 0.125f, 15.0f * D2R, nullptr);
	//cells.ApplyClustering(&pcd, d_labels, 0.125f, 15.0f * D2R, nullptr);
	//cells.ApplyClustering(&pcd, d_labels, 0.125f, nullptr);
	cells.ApplyClustering(&pcd, d_labels, 0.175f, nullptr);
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
			//{ color.x(), color.y(), color.z(), 1.0f });
		{ (float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f, 1.0f });
	}
}
#endif // CLUSTERING

#if FILTERING
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

	Eigen::Matrix4f transform;
	Huvitz::cuAABB localAABB;
	std::vector<float> occupancyMap(400 * 400, FLT_MAX);
	float voxelSize = 0.0f;

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

		//while (auto frame = reader.next())
		if (auto frame = reader.read_at(50))
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
			//Eigen::Matrix4f transform;
			transform.data()[0] = m[0]; transform.data()[1] = m[1]; transform.data()[2] = m[2]; transform.data()[3] = m[3];
			transform.data()[4] = m[4]; transform.data()[5] = m[5]; transform.data()[6] = m[6]; transform.data()[7] = m[7];
			transform.data()[8] = m[8]; transform.data()[9] = m[9]; transform.data()[10] = m[10]; transform.data()[11] = m[11];
			transform.data()[12] = m[12]; transform.data()[13] = m[13]; transform.data()[14] = m[14]; transform.data()[15] = m[15];

			voxelSize = *reinterpret_cast<float*>(basePtr + offset); offset += sizeof(float);

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

			{
				auto error = cudaGetLastError();
				if (error != cudaSuccess) {
					printf("CUDA error : %s\n", cudaGetErrorString(error));
				}
			}

			PLYFormat ply;
			//ply.SetDataType(PLYFormat::PLYFormatDataType::ASCII);
			for (size_t i = 0; i < mapWidth * mapHeight; i++)
			{
				auto& d = h_depthMap[i];
				if (false == VECTOR3F_VALID_(d)) continue;

				localAABB.expand(make_float3(d.x(), d.y(), d.z()));

				Eigen::Vector3f p = (transform * Eigen::Vector4f(d.x(), d.y(), d.z(), 1.0f)).head<3>();

				ply.AddPoint(p.x(), p.y(), p.z());
				auto& n = h_normalMap[i];
				ply.AddNormal(n.x(), n.y(), n.z());
				auto& r = h_colorMap[i * 3];
				auto& g = h_colorMap[i * 3 + 1];
				auto& b = h_colorMap[i * 3 + 2];
				ply.AddColor((float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f);

				VD::AddSphere("Patch",
					{ d.x(), d.y(), d.z() + 0.5f },
					{ n.x(), n.y(), n.z() },
					0.1f,
					//{ (float)r / 255.0f, (float)g / 255.0f, (float)b / 255.0f, 1.0f });
					{ 1.0f, 0.0f, 0.0f, 1.0f });
			}
			//ply.Serialize("D:\\Debug\\InputFrames\\InputFrame_" + std::to_string(frame->frameIndex) + ".ply");

			//CUDA_TE(Patch);

			//static int count = 0;
			//count++;

			//if (5 == count)
			//{
			//	break;
			//}

			VD::AddWiredBox("LocalAABB", { 0.0f, 0.0f, 0.0f }, { 20.0f, 20.0f, 20.0f }, { 0.0f, 1.0f, 0.0f, 1.0f });

			auto aabbCenter = (localAABB.min + localAABB.max) * 0.5f;
			//float occupancyMap[400 * 400];
			//std::fill(std::begin(occupancyMap), std::end(occupancyMap), FLT_MAX);
			for (size_t i = 0; i < mapWidth * mapHeight; i++)
			{
				auto& d = h_depthMap[i];
				if (false == VECTOR3F_VALID_(d)) continue;

				auto mapLocalX = (d.x() - aabbCenter.x) / voxelSize + 200.0f;
				auto mapLocalY = (d.y() - aabbCenter.y) / voxelSize + 200.0f;
				occupancyMap[(int)mapLocalY * 400 + (int)mapLocalX] = d.z() + 0.5f;
			}

			auto copiedOccupancyMap = occupancyMap;

			for (size_t y = 1; y < 400 - 1; y++)
			{
				for (size_t x = 1; x < 400 - 1; x++)
				{
					auto l = copiedOccupancyMap[y * 400 + (x - 1)];
					auto r = copiedOccupancyMap[y * 400 + (x + 1)];
					auto u = copiedOccupancyMap[(y - 1) * 400 + x];
					auto d = copiedOccupancyMap[(y + 1) * 400 + x];

					if (FLT_MAX == l || FLT_MAX == r || FLT_MAX == u || FLT_MAX == d)
					{
						occupancyMap[y * 400 + x] = FLT_MAX;
					}
				}
			}

			for (size_t y = 0; y < 400; y++)
			{
				for (size_t x = 0; x < 400; x++)
				{
					float val = occupancyMap[y * 400 + x];
					if (val != FLT_MAX)
					{
						VD::AddBox("OccupancyMap",
							//{ aabbCenter.x + (x - 200.0f) * voxelSize, aabbCenter.y + (y - 200.0f) * voxelSize, val },
							{ aabbCenter.x + (x - 200.0f) * voxelSize, aabbCenter.y + (y - 200.0f) * voxelSize, -10.0f },
							{ voxelSize, voxelSize, voxelSize },
							{ 0.0f, 0.0f, 1.0f, 1.0f });
					}
				}
			}
		}
		CUDA_TE(Total);

		cudaFree(integrationParams.d_depthMap);
		cudaFree(integrationParams.d_normalMap);
		cudaFree(integrationParams.d_colorMap);
	}

	{
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

			//VD::AddSphere("PointCloud",
			//	{ p.x(), p.y(), p.z() },
			//	{ n.x(), n.y(), n.z() },
			//	0.05f,
			//	{ c.x(), c.y(), c.z(), 1.0f });
		}

		float3* d_positions = nullptr;
		float3* d_normals = nullptr;
		uchar3* d_colors = nullptr;

		CUDA_MALLOC(&d_positions, sizeof(float3)* h_positions.size());
		CUDA_MALLOC(&d_normals, sizeof(float3)* h_normals.size());
		CUDA_MALLOC(&d_colors, sizeof(uchar3)* h_colors.size());

		CUDA_COPY_H2D(d_positions, h_positions.data(), sizeof(float3)* h_positions.size());
		CUDA_COPY_H2D(d_normals, h_normals.data(), sizeof(float3)* h_normals.size());
		CUDA_COPY_H2D(d_colors, h_colors.data(), sizeof(uchar3)* h_colors.size());

		PCD pcd;
		Eigen::Matrix4f inverseTransfrom = transform.inverse();
		localAABB.min.z = -FLT_MAX;
		localAABB.max.z = FLT_MAX;
		pcd.FromDevicePointers(d_positions, d_normals, d_colors, h_positions.size(), localAABB, inverseTransfrom.data(), nullptr, nullptr);
		//pcd.FromDevicePointers(d_positions, d_normals, d_colors, h_positions.size(), nullptr, nullptr);

		//SparseCells cells;
		//cells.Build(&pcd, 0.1f, nullptr);
		//unsigned int* d_labels = nullptr;
		//CUDA_MALLOC(&d_labels, sizeof(unsigned int) * pcd.GetNumberOfPositions());
		//cells.ApplyClustering(&pcd, d_labels, 0.125f, nullptr);
		//std::vector<unsigned int> h_labels(pcd.GetNumberOfPositions());
		//CUDA_COPY_D2H(h_labels.data(), d_labels, sizeof(unsigned int) * pcd.GetNumberOfPositions());
		//CUDA_SAFE_FREE(d_labels);

		CUDA_SAFE_FREE(d_positions);
		CUDA_SAFE_FREE(d_normals);
		CUDA_SAFE_FREE(d_colors);

		std::vector<float3> rps;
		std::vector<float3> rns;
		std::vector<uchar3> rcs;

		PCD croppedPcd(pcd.GetNumberOfPositions());
		pcd.CropUsingLocalRTandAABB(inverseTransfrom.data(), localAABB, croppedPcd, nullptr, nullptr);

		CUDA_TS(Clustering);
		croppedPcd.Clustering(0.1f, 0.125f, nullptr, nullptr);
		CUDA_TE(Clustering);

		CUDA_TS(BuildLabelCountTable);
		croppedPcd.BuildLabelCountTable();
		CUDA_TE(BuildLabelCountTable);

		CUDA_TS(BuildPointCountTable);
		croppedPcd.BuildPointCountTable();
		CUDA_TE(BuildPointCountTable);

		croppedPcd.ToHostVectors(rps, rns, rcs);

		auto colors = Color::GetContrastingColors(32);

		std::vector<unsigned int> h_labels(croppedPcd.GetNumberOfPositions());
		CUDA_COPY_D2H(h_labels.data(), croppedPcd.GetLabels(), sizeof(unsigned int)* croppedPcd.GetNumberOfPositions());

		std::vector<unsigned int> h_pointCounts(croppedPcd.GetNumberOfPositions());
		CUDA_COPY_D2H(h_pointCounts.data(), croppedPcd.GetPointCountTable(), sizeof(unsigned int)* croppedPcd.GetNumberOfPositions());

		std::set<unsigned int> toDelete;

		auto aabbCenter = (localAABB.min + localAABB.max) * 0.5f;
		for (size_t i = 0; i < croppedPcd.GetNumberOfPositions(); i++)
		{
			auto& p = rps[i];
			auto& n = rns[i];
			auto& c = rcs[i];
			auto& l = h_labels[i];
			auto& count = h_pointCounts[i];
			auto& color = colors[l % colors.size()];

			if (500 > count)
			{
				toDelete.insert(l);
				continue;
			}

			Eigen::Vector4f itp = inverseTransfrom * Eigen::Vector4f(p.x, p.y, p.z, 1.0f);

			auto mapLocalX = (itp.x() - aabbCenter.x) / voxelSize + 200.0f;
			auto mapLocalY = (itp.y() - aabbCenter.y) / voxelSize + 200.0f;
			auto depth = occupancyMap[(int)mapLocalY * 400 + (int)mapLocalX];
			if (itp.z() > depth)
			{
				VD::AddWiredBox("ToFilter",
					{ itp.x(), itp.y(), itp.z() },
					{ 0.0f, 0.0f, 1.0f },
					{ voxelSize, voxelSize, voxelSize },
					{ 0.0f, 1.0f, 0.0f, 1.0f });

				toDelete.insert(l);
			}
		}

		for (size_t i = 0; i < croppedPcd.GetNumberOfPositions(); i++)
		{
			auto& p = rps[i];
			auto& n = rns[i];
			auto& c = rcs[i];
			auto& l = h_labels[i];

			if(toDelete.find(l) != toDelete.end()) continue;

			Eigen::Vector4f itp = inverseTransfrom * Eigen::Vector4f(p.x, p.y, p.z, 1.0f);

			VD::AddSphere("PointCloud",
				{ itp.x(), itp.y(), itp.z() },
				{ n.x, n.y, n.z },
				0.05f,
				{ (float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f, 1.0f });
		}
	}

#if 0
	//{
	//	PLYFormat ply;
	//	ply.Deserialize("D:\\Debug\\Compound.ply");
	//	std::vector<float3> h_positions(ply.GetPoints().size());
	//	std::vector<float3> h_normals(ply.GetPoints().size());
	//	std::vector<uchar3> h_colors(ply.GetPoints().size());
	//	for (size_t i = 0; i < ply.GetPoints().size(); i++)
	//	{
	//		auto& p = ply.GetPoints()[i];
	//		auto& n = ply.GetNormals()[i];
	//		auto& c = ply.GetColors()[i];
	//		h_positions[i] = make_float3(p.x(), p.y(), p.z());
	//		h_normals[i] = make_float3(n.x(), n.y(), n.z());
	//		h_colors[i] = make_uchar3((unsigned char)(c.x() * 255.0f), (unsigned char)(c.y() * 255.0f), (unsigned char)(c.z() * 255.0f));
	//	}
	//	PCD pcd;
	//	pcd.FromHostVectors(h_positions, h_normals, h_colors);
	//	cudaStream_t stream = nullptr;
	//	SparseCells cells;
	//	cells.Build(&pcd, 0.1f, nullptr);
	//	unsigned int* d_labels = nullptr;
	//	CUDA_MALLOC(&d_labels, sizeof(unsigned int) * pcd.GetNumberOfPositions());
	//	cells.ApplyClustering(&pcd, d_labels, 0.125f, nullptr);
	//	std::vector<unsigned int> h_labels(pcd.GetNumberOfPositions());
	//	CUDA_COPY_D2H(h_labels.data(), d_labels, sizeof(unsigned int) * pcd.GetNumberOfPositions());
	//	CUDA_SAFE_FREE(d_labels);
	//}

	//{
	//	PLYFormat ply;
	//	ply.Deserialize("D:\\Debug\\InputFrames\\InputFrame_100.ply");
	//	//ply.Deserialize("D:\\Debug\\BasePoints.ply");

	//	std::vector<float3> h_positions(ply.GetPoints().size());
	//	std::vector<float3> h_normals(ply.GetPoints().size());
	//	std::vector<uchar3> h_colors(ply.GetPoints().size());

	//	for (size_t i = 0; i < ply.GetPoints().size(); i++)
	//	{
	//		auto& p = ply.GetPoints()[i];
	//		auto& n = ply.GetNormals()[i];
	//		auto& c = ply.GetColors()[i];

	//		h_positions[i] = make_float3(p.x(), p.y(), p.z());
	//		h_normals[i] = make_float3(n.x(), n.y(), n.z());
	//		h_colors[i] = make_uchar3((unsigned char)(c.x() * 255.0f), (unsigned char)(c.y() * 255.0f), (unsigned char)(c.z() * 255.0f));

	//		VD::AddSphere("PointCloud",
	//			{ p.x(), p.y(), p.z() },
	//			{ n.x(), n.y(), n.z() },
	//			0.10f,
	//			{ 1.0f, 0.0f, 0.0f, 1.0f });
	//	}
	//}

	//PCD pcd;
	//pcd.FromHostVectors(h_positions, h_normals, h_colors);

	//////CUDA_TS(Downsample);
	////pcd = pcd.Downsample(0.1f);
	//////CUDA_TE(Downsample);

	//SparseCells cells;

	//CUDA_TS(BuildSparseCells);
	//cells.Build(&pcd, 0.1f, nullptr);
	//CUDA_TE(BuildSparseCells);

	//unsigned int* d_labels = nullptr;
	//CUDA_MALLOC(&d_labels, sizeof(unsigned int) * pcd.GetNumberOfPositions());

	//CUDA_TS(ApplyClustering);
	////cells.ApplyClustering(&pcd, d_labels, 1.0f * 0.125f, 15.0f * D2R, nullptr);
	////cells.ApplyClustering(&pcd, d_labels, 0.125f, 15.0f * D2R, nullptr);
	//cells.ApplyClustering(&pcd, d_labels, 0.125f, nullptr);
	//CUDA_TE(ApplyClustering);
#endif // 0
}
#endif // FILTERING

};

REGISTER_APP(AppVoxelDataBase, "AppVoxelDataBase");
