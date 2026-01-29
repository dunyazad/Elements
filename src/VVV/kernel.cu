#include <VVV/VVV.h>
#include <device_launch_parameters.h>
#include <cstdio>

namespace VVV
{
	CUDA_HOST_DEVICE static inline uint32_t CompactBits(uint64_t v)
	{
		v &= 0x1249249249249249ull;
		v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3ull;
		v = (v ^ (v >> 4)) & 0x100f00f00f00f00full;
		v = (v ^ (v >> 8)) & 0x1f0000ff0000ffull;
		v = (v ^ (v >> 16)) & 0x1f00000000ffffull;
		v = (v ^ (v >> 32)) & 0x001FFFFFull;
		return static_cast<uint32_t>(v);
	}

	CUDA_HOST_DEVICE Vector3f Morton64::ToPosition(float blockSize) const
	{
		uint32_t ux = CompactBits(code >> 0);
		uint32_t uy = CompactBits(code >> 1);
		uint32_t uz = CompactBits(code >> 2);
		int32_t vx = static_cast<int32_t>(ux) - AXIS_BIAS;
		int32_t vy = static_cast<int32_t>(uy) - AXIS_BIAS;
		int32_t vz = static_cast<int32_t>(uz) - AXIS_BIAS;
		return Vector3f{ (vx + 0.5f) * blockSize, (vy + 0.5f) * blockSize, (vz + 0.5f) * blockSize };
	}

	void VoxelDataBase::InternalAllocate(uint32_t maxBlocks)
	{
		maxBlockCount = maxBlocks;
		cudaMalloc(&d_blocks, sizeof(VoxelBlock) * maxBlockCount);
		cudaMemset(d_blocks, 0, sizeof(VoxelBlock) * maxBlockCount);
		cudaMalloc(&d_hashTable, sizeof(uint64_t) * maxBlockCount);
		cudaMemset(d_hashTable, 0, sizeof(uint64_t) * maxBlockCount);
		cudaMalloc(&d_blockCount, sizeof(uint32_t));
		cudaMemset(d_blockCount, 0, sizeof(uint32_t));
	}

	void VoxelDataBase::InternalFree()
	{
		if (d_blocks) cudaFree(d_blocks);
		if (d_hashTable) cudaFree(d_hashTable);
		if (d_blockCount) cudaFree(d_blockCount);
		d_blocks = nullptr; d_hashTable = nullptr; d_blockCount = nullptr; maxBlockCount = 0;
	}

#ifdef __CUDACC__
	__device__ uint32_t StrongHash(uint64_t key, uint32_t maxBlocks)
	{
		key ^= key >> 33;
		key *= 0xff51afd7ed558ccdULL;
		key ^= key >> 33;
		key *= 0xc4ceb9fe1a85ec53ULL;
		key ^= key >> 33;
		return static_cast<uint32_t>(key % maxBlocks);
	}

	__global__ void InsertKernel(VoxelDataBase db, const Vector3f* points, const Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
	{
		uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= count) return;

		Vector3f p = points[idx];
		Vector3b c = colors[idx];
		Morton64 blockKey = Morton64::FromPosition(p, blockSize);
		uint64_t mKey = blockKey.code; if (mKey == 0) mKey = 0xFFFFFFFFFFFFFFFFULL;

		uint32_t slot = StrongHash(mKey, db.maxBlockCount);
		uint32_t start = slot;
		BlockID bid = INVALID_BLOCK;
		while (true)
		{
			unsigned long long* slotPtr = (unsigned long long*) & db.d_hashTable[slot];
			unsigned long long prev = atomicCAS(slotPtr, 0ULL, (unsigned long long)mKey);
			if (prev == 0) { atomicAdd(db.d_blockCount, 1); bid = slot; break; }
			if (prev == mKey) { bid = slot; break; }
			slot = (slot + 1) % db.maxBlockCount;
			if (slot == start) break;
		}

		if (bid != INVALID_BLOCK)
		{
			float vSize = blockSize / 8.0f;
			Vector3f bc = blockKey.ToPosition(blockSize);
			int lx = static_cast<int>(floorf((p.x - (bc.x - blockSize * 0.5f)) / vSize + 1e-5f));
			int ly = static_cast<int>(floorf((p.y - (bc.y - blockSize * 0.5f)) / vSize + 1e-5f));
			int lz = static_cast<int>(floorf((p.z - (bc.z - blockSize * 0.5f)) / vSize + 1e-5f));
			lx = (lx < 0) ? 0 : (lx > 7 ? 7 : lx); ly = (ly < 0) ? 0 : (ly > 7 ? 7 : ly); lz = (lz < 0) ? 0 : (lz > 7 ? 7 : lz);

			Voxel& v = db.d_blocks[bid].voxels[(lz << 6) | (ly << 3) | lx];
			atomicAdd(&v.value, 1.0f);
			v.color = c;
		}
	}

	__global__ void ExtractKernel(VoxelDataBase db, float blockSize, ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
	{
		uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
		if (slot >= db.maxBlockCount) return;
		uint64_t key = db.d_hashTable[slot];
		if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL) return;

		Morton64 bKey(key); Vector3f bc = bKey.ToPosition(blockSize);
		float vSize = blockSize / 8.0f;
		for (int i = 0; i < 512; ++i)
		{
			Voxel& v = db.d_blocks[slot].voxels[i];
			if (v.value > 0.0f)
			{
				uint32_t idx = atomicAdd(count, 1);
				if (idx < maxOut)
				{
					int lz = i / 64; int ly = (i % 64) / 8; int lx = i % 8;
					out[idx].position = { (bc.x - blockSize * 0.5f) + (lx + 0.5f) * vSize, (bc.y - blockSize * 0.5f) + (ly + 0.5f) * vSize, (bc.z - blockSize * 0.5f) + (lz + 0.5f) * vSize };
					out[idx].weight = v.value;
					out[idx].color[0] = v.color.x; out[idx].color[1] = v.color.y; out[idx].color[2] = v.color.z;
				}
			}
		}
	}
#endif
}

extern "C"
{
	void VVV_Allocate(VVV::VoxelDataBase& db, uint32_t maxBlocks) { db.InternalAllocate(maxBlocks); }
	void VVV_Free(VVV::VoxelDataBase& db) { db.InternalFree(); }
	void VVV_UpdateVoxelFromPoints(VVV::VoxelDataBase& db, const VVV::Vector3f* points, const VVV::Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
	{
		VVV::Vector3f* d_p; VVV::Vector3b* d_c;
		cudaMalloc(&d_p, sizeof(VVV::Vector3f) * count); cudaMalloc(&d_c, sizeof(VVV::Vector3b) * count);
		cudaMemcpy(d_p, points, sizeof(VVV::Vector3f) * count, cudaMemcpyHostToDevice);
		cudaMemcpy(d_c, colors, sizeof(VVV::Vector3b) * count, cudaMemcpyHostToDevice);
#ifdef __CUDACC__
		VVV::InsertKernel << <(count + 255) / 256, 256 >> > (db, d_p, d_c, count, blockSize, frameId);
		cudaDeviceSynchronize();
#endif
		cudaFree(d_p); cudaFree(d_c);
	}
	uint32_t VVV_ExtractActiveVoxelsToHost(VVV::VoxelDataBase& db, float blockSize, VVV::ExtractedVoxel* hostBuffer, uint32_t maxOut)
	{
		VVV::ExtractedVoxel* d_out; uint32_t* d_cnt;
		cudaMalloc(&d_out, sizeof(VVV::ExtractedVoxel) * maxOut);
		cudaMalloc(&d_cnt, sizeof(uint32_t)); cudaMemset(d_cnt, 0, sizeof(uint32_t));
#ifdef __CUDACC__
		VVV::ExtractKernel << <(db.maxBlockCount + 255) / 256, 256 >> > (db, blockSize, d_out, d_cnt, maxOut);
		cudaDeviceSynchronize();
#endif
		uint32_t res; cudaMemcpy(&res, d_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
		uint32_t copyAmt = (res > maxOut) ? maxOut : res;
		cudaMemcpy(hostBuffer, d_out, sizeof(VVV::ExtractedVoxel) * copyAmt, cudaMemcpyDeviceToHost);
		cudaFree(d_out); cudaFree(d_cnt);
		return res;
	}
}