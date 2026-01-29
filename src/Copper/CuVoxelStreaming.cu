#include <Copper/CuVoxelStreaming.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cstdio>

#include <thrust/device_vector.h>

#include <sstream>
#include <string>
#include <iostream>
#include <iomanip>

#ifdef __INTELLISENSE__
extern "C" unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val);
extern "C" int atomicAdd(int* address, int val);
extern "C" void __threadfence();
extern "C" int atomicExch(int* address, int val);
#endif

namespace Copper
{
	__constant__ float c_depth_scale;
	__constant__ float c_depth_min;
	__constant__ float c_depth_max;
	__constant__ float c_fx, c_fy, c_cx, c_cy;
	__constant__ int c_width, c_height;

	__device__ inline void atomicAddByte(unsigned char* address, unsigned char val)
	{
		unsigned int* base_address = (unsigned int*)((size_t)address & ~3);
		unsigned int selectors[] = { 0, 8, 16, 24 };
		unsigned int shift = selectors[(size_t)address & 3];
		unsigned int old, assumed, updated;

		old = *base_address;
		do
		{
			assumed = old;
			unsigned int byte_val = (assumed >> shift) & 0xFF;
			unsigned int new_byte = (byte_val + val > 255) ? 255 : byte_val + val;
			updated = (assumed & ~(0xFF << shift)) | (new_byte << shift);
			old = atomicCAS(base_address, assumed, updated);
		} while (assumed != old);
	}

	__device__ inline int compute_hash(int3 pos)
	{
		int res = ((pos.x * HASH_P1) ^ (pos.y * HASH_P2) ^ (pos.z * HASH_P3)) % HASH_TABLE_SIZE;
		if (res < 0)
		{
			res += HASH_TABLE_SIZE;
		}
		return res;
	}

	__device__ inline int3 world_to_block_coord(float3 p)
	{
		const float inv_block_size = 1.25f;
		return make_int3(
			static_cast<int>(floorf(p.x * inv_block_size + 1e-5f)),
			static_cast<int>(floorf(p.y * inv_block_size + 1e-5f)),
			static_cast<int>(floorf(p.z * inv_block_size + 1e-5f))
		);
	}

	__device__ inline float2 project_point(float3 p)
	{
		return make_float2(c_fx * p.x / p.z + c_cx, c_fy * p.y / p.z + c_cy);
	}

	__device__ inline int find_block_ptr(HashEntry* hash_table, int3 b_pos)
	{
		int hash = compute_hash(b_pos);
		for (int step = 0; step < 32; ++step)
		{
			HashEntry entry = hash_table[hash];
			if (entry.voxelBlockIndex > -1 &&
				entry.position.x == b_pos.x &&
				entry.position.y == b_pos.y &&
				entry.position.z == b_pos.z)
			{
				return entry.voxelBlockIndex;
			}
			if (entry.voxelBlockIndex == FREE_ENTRY)
			{
				break;
			}
			hash = (hash + 1) % HASH_TABLE_SIZE;
		}
		return -1;
	}

	__global__ void init_hash_table_kernel(HashEntry* hash_table)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx < HASH_TABLE_SIZE)
		{
			hash_table[idx].position = make_int3(0, 0, 0);
			hash_table[idx].voxelBlockIndex = FREE_ENTRY;
		}
	}

	__global__ void alloc_points_kernel(HashEntry* hash_table, HeapCounter* heap_counter, const float3* points, int point_count)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= point_count)
		{
			return;
		}

		int3 base_b_pos = world_to_block_coord(points[idx]);
		// 할당 범위를 3x3x3으로 확장하여 누락 방지
		for (int i = 0; i < 27; ++i)
		{
			int3 b_pos = make_int3(base_b_pos.x + (i % 3) - 1, base_b_pos.y + ((i / 3) % 3) - 1, base_b_pos.z + (i / 9) - 1);
			int hash = compute_hash(b_pos);
			for (int step = 0; step < 32; ++step)
			{
				int* ptr_addr = (int*)&(hash_table[hash].voxelBlockIndex);
				if (*ptr_addr != FREE_ENTRY)
				{
					if (hash_table[hash].position.x == b_pos.x && hash_table[hash].position.y == b_pos.y && hash_table[hash].position.z == b_pos.z)
					{
						break;
					}
				}
				else
				{
					int old = atomicCAS((unsigned int*)ptr_addr, (unsigned int)FREE_ENTRY, (unsigned int)-2);
					if (old == FREE_ENTRY)
					{
						int new_ptr = atomicAdd(&(heap_counter->count), 1);
						if (new_ptr < MAX_BLOCKS)
						{
							hash_table[hash].position = b_pos;
							__threadfence();
							atomicExch(ptr_addr, new_ptr);
						}
						else
						{
							atomicExch(ptr_addr, FREE_ENTRY);
						}
						break;
					}
				}
				hash = (hash + 1) % HASH_TABLE_SIZE;
			}
		}
	}

	__global__ void insert_points_kernel(HashEntry* hash_table, VoxelBlock* voxel_blocks, const float3* points, const float3* normals, const uchar3* colors, int point_count)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= point_count)
		{
			return;
		}

		const float3 p = points[idx];
		const float3 n = normals[idx];
		const uchar3 c = colors[idx];

		const float inv_voxel_size = 10.0f;
		const float radius = SAMPLING_RADIUS;

		// 8회 루프 대신 3x3x3 주변 복셀을 모두 탐색하여 빈틈을 완전히 메움
		for (int i = 0; i < 27; ++i)
		{
			float3 offset = make_float3(
				((i % 3) - 1) * radius,
				(((i / 3) % 3) - 1) * radius,
				((i / 9) - 1) * radius
			);

			float3 sp = make_float3(p.x + offset.x, p.y + offset.y, p.z + offset.z);
			int3 b_pos = world_to_block_coord(sp);
			int ptr = find_block_ptr(hash_table, b_pos);

			if (ptr > -1)
			{
				int vx = (int)floorf((sp.x - b_pos.x * 0.8f) * inv_voxel_size + 1e-4f) & 7;
				int vy = (int)floorf((sp.y - b_pos.y * 0.8f) * inv_voxel_size + 1e-4f) & 7;
				int vz = (int)floorf((sp.z - b_pos.z * 0.8f) * inv_voxel_size + 1e-4f) & 7;
				int v_idx = (vz << 6) | (vy << 3) | vx;

				VoxelBlock* block = &voxel_blocks[ptr];

				// 법선 벡터 내적을 통해 거리와 방향(Inside/Outside) 결정
				float dist = offset.x * n.x + offset.y * n.y + offset.z * n.z;
				short tsdf_short = static_cast<short>(fminf(1.0f, fmaxf(-1.0f, dist * inv_voxel_size)) * SHORT_MAX_FLOAT);

				atomicAddByte(&(block->weight[v_idx]), 1);

				// 데이터 덮어쓰기 (가중치가 낮을 때만 쓰거나 단순 누적)
				block->tsdf[v_idx] = tsdf_short;
				block->normal[v_idx] = n;
				block->color[v_idx] = c;
			}
		}
	}

	__global__ void integrate_blocks_kernel(HashEntry* hash_table, VoxelBlock* voxel_blocks, int3* queue, int queue_count, const float* depth_map, const float* pose_inv)
	{
		int idx = blockIdx.x;
		if (idx >= queue_count)
		{
			return;
		}
		int3 b_pos = queue[idx];
		int ptr = find_block_ptr(hash_table, b_pos);
		if (ptr < 0)
		{
			return;
		}

		int tid = threadIdx.x;
		int vx = tid & 7, vy = (tid >> 3) & 7, vz = (tid >> 6);
		float3 p_w = make_float3((b_pos.x * 8 + vx) * VOXEL_SIZE, (b_pos.y * 8 + vy) * VOXEL_SIZE, (b_pos.z * 8 + vz) * VOXEL_SIZE);
		float3 p_c = make_float3(
			pose_inv[0] * p_w.x + pose_inv[1] * p_w.y + pose_inv[2] * p_w.z + pose_inv[3],
			pose_inv[4] * p_w.x + pose_inv[5] * p_w.y + pose_inv[6] * p_w.z + pose_inv[7],
			pose_inv[8] * p_w.x + pose_inv[9] * p_w.y + pose_inv[10] * p_w.z + pose_inv[11]
		);
		if (p_c.z <= 0.1f)
		{
			return;
		}
		float2 uv = project_point(p_c);
		int u = (int)(uv.x + 0.5f);
		int v = (int)(uv.y + 0.5f);
		if (u < 0 || u >= c_width || v < 0 || v >= c_height)
		{
			return;
		}
		float d = depth_map[v * c_width + u] * c_depth_scale;
		if (d < c_depth_min || d > c_depth_max)
		{
			return;
		}
		float sdf = d - p_c.z;
		if (sdf > -TRUNCATION_DIST)
		{
			float tsdf = fminf(TRUNCATION_DIST, sdf);
			VoxelBlock* block = &voxel_blocks[ptr];
			float old_tsdf = (float)block->tsdf[tid] / SHORT_MAX_FLOAT;
			float old_w = (float)block->weight[tid];
			block->tsdf[tid] = (short)(((old_tsdf * old_w + tsdf) / (old_w + 1.0f)) * SHORT_MAX_FLOAT);
			block->weight[tid] = (unsigned char)fminf(255.0f, old_w + 1.0f);
		}
	}

	__global__ void find_visible_blocks_kernel(const float* depth_map, const float* pose, int3* queue, int* queue_count)
	{
		int u = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
		int v = (blockIdx.y * blockDim.y + threadIdx.y) * 4;
		if (u >= c_width || v >= c_height)
		{
			return;
		}
		float depth = depth_map[v * c_width + u] * c_depth_scale;
		if (depth < c_depth_min || depth > c_depth_max)
		{
			return;
		}

		float x = (u - c_cx) * depth / c_fx;
		float y = (v - c_cy) * depth / c_fy;
		float3 p_w = make_float3(
			pose[0] * x + pose[1] * y + pose[2] * depth + pose[3],
			pose[4] * x + pose[5] * y + pose[6] * depth + pose[7],
			pose[8] * x + pose[9] * y + pose[10] * depth + pose[11]
		);
		int3 b_surf = make_int3(
			static_cast<int>(floorf(p_w.x * 1.25f)),
			static_cast<int>(floorf(p_w.y * 1.25f)),
			static_cast<int>(floorf(p_w.z * 1.25f))
		);

		int idx = atomicAdd(queue_count, 1);
		if (idx < MAX_BLOCKS)
		{
			queue[idx] = b_surf;
		}
	}

	__global__ void alloc_blocks_kernel(HashEntry* hash_table, HeapCounter* heap_counter, int3* queue, int queue_count)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= queue_count)
		{
			return;
		}
		int3 pos = queue[idx];
		int hash = compute_hash(pos);
		for (int step = 0; step < 64; ++step)
		{
			int* ptr_addr = (int*)&(hash_table[hash].voxelBlockIndex);
			if (*ptr_addr != FREE_ENTRY &&
				hash_table[hash].position.x == pos.x &&
				hash_table[hash].position.y == pos.y &&
				hash_table[hash].position.z == pos.z)
			{
				return;
			}
			if (*ptr_addr == FREE_ENTRY)
			{
				int old = atomicCAS((unsigned int*)ptr_addr, (unsigned int)FREE_ENTRY, (unsigned int)-2);
				if (old == FREE_ENTRY)
				{
					int new_ptr = atomicAdd(&(heap_counter->count), 1);
					if (new_ptr < MAX_BLOCKS)
					{
						hash_table[hash].position = pos;
						atomicExch(ptr_addr, new_ptr);
					}
					else
					{
						atomicExch(ptr_addr, FREE_ENTRY);
					}
					return;
				}
			}
			hash = (hash + 1) % HASH_TABLE_SIZE;
		}
	}

	CuVoxelStreaming::CuVoxelStreaming(int w, int h, float fx, float fy, float cx, float cy, float ds, float dmin, float dmax, float k1, float k2, float p1, float p2)
	{
		cam_params.width = w;
		cam_params.height = h;
		cam_params.fx = fx;
		cam_params.fy = fy;
		cam_params.cx = cx;
		cam_params.cy = cy;
		cam_params.depth_scale = ds;
		cam_params.depth_min = dmin;
		cam_params.depth_max = dmax;
		cam_params.k1 = k1;
		cam_params.k2 = k2;
		cam_params.p1 = p1;
		cam_params.p2 = p2;

		cudaMalloc((void**)&d_hash_table, HASH_TABLE_SIZE * sizeof(HashEntry));
		cudaMalloc((void**)&d_voxel_blocks, MAX_BLOCKS * sizeof(VoxelBlock));
		cudaMalloc((void**)&d_heap_counter, sizeof(HeapCounter));
		cudaMalloc((void**)&d_visible_block_queue, MAX_BLOCKS * sizeof(int3));
		cudaMalloc((void**)&d_visible_block_count, sizeof(int));
		Reset();
	}

	CuVoxelStreaming::~CuVoxelStreaming()
	{
		cudaFree(d_hash_table);
		cudaFree(d_voxel_blocks);
		cudaFree(d_heap_counter);
		cudaFree(d_visible_block_queue);
		cudaFree(d_visible_block_count);
	}

	void CuVoxelStreaming::Reset()
	{
		init_hash_table_kernel << <(HASH_TABLE_SIZE + 255) / 256, 256 >> > (d_hash_table);
		cudaMemset(d_voxel_blocks, 0, MAX_BLOCKS * sizeof(VoxelBlock));
		cudaMemset(d_heap_counter, 0, sizeof(HeapCounter));
		cudaDeviceSynchronize();
	}

	void CuVoxelStreaming::ProcessPointCloud(const float3* d_p, const float3* d_n, const uchar3* d_c, int count)
	{
		if (count <= 0)
		{
			return;
		}

		CUDA_TS(ProcessPointCloud);

		CUDA_TS(alloc_points_kernel);
		alloc_points_kernel << <(count + 255) / 256, 256 >> > (d_hash_table, d_heap_counter, d_p, count);
		CUDA_TE(alloc_points_kernel);

		CUDA_TS(insert_points_kernel);
		insert_points_kernel << <(count + 255) / 256, 256 >> > (d_hash_table, d_voxel_blocks, d_p, d_n, d_c, count);
		CUDA_TE(insert_points_kernel);

		cudaDeviceSynchronize();

		CUDA_TE(ProcessPointCloud);
	}

	void CuVoxelStreaming::ProcessFrame(const float* d_dm, const float* d_pm)
	{
		cudaMemcpyToSymbol(c_depth_scale, &cam_params.depth_scale, sizeof(float));
		cudaMemcpyToSymbol(c_depth_min, &cam_params.depth_min, sizeof(float));
		cudaMemcpyToSymbol(c_depth_max, &cam_params.depth_max, sizeof(float));
		cudaMemcpyToSymbol(c_fx, &cam_params.fx, sizeof(float));
		cudaMemcpyToSymbol(c_fy, &cam_params.fy, sizeof(float));
		cudaMemcpyToSymbol(c_cx, &cam_params.cx, sizeof(float));
		cudaMemcpyToSymbol(c_cy, &cam_params.cy, sizeof(float));
		cudaMemcpyToSymbol(c_width, &cam_params.width, sizeof(int));
		cudaMemcpyToSymbol(c_height, &cam_params.height, sizeof(int));

		cudaMemset(d_visible_block_count, 0, sizeof(int));
		dim3 b(16, 16);
		dim3 g((cam_params.width / 4 + 15) / 16, (cam_params.height / 4 + 15) / 16);
		find_visible_blocks_kernel << <g, b >> > (d_dm, d_pm, d_visible_block_queue, d_visible_block_count);

		int count;
		cudaMemcpy(&count, d_visible_block_count, sizeof(int), cudaMemcpyDeviceToHost);
		if (count > 0)
		{
			alloc_blocks_kernel << <(count + 255) / 256, 256 >> > (d_hash_table, d_heap_counter, d_visible_block_queue, count);

			float h_pm[16];
			cudaMemcpy(h_pm, d_pm, 64, cudaMemcpyDeviceToHost);
			float h_inv[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
			float* d_inv;
			cudaMalloc(&d_inv, 48);
			cudaMemcpy(d_inv, h_inv, 48, cudaMemcpyHostToDevice);

			integrate_blocks_kernel << <count, 512 >> > (d_hash_table, d_voxel_blocks, d_visible_block_queue, count, d_dm, d_inv);
			cudaFree(d_inv);
		}
	}

	int CuVoxelStreaming::GetActiveBlockCount()
	{
		int count;
		cudaMemcpy(&count, d_heap_counter, sizeof(int), cudaMemcpyDeviceToHost);
		return count;
	}

	std::vector<float3> CuVoxelStreaming::GetActiveBlockCenters()
	{
		std::vector<float3> res;
		HashEntry* h = new HashEntry[HASH_TABLE_SIZE];
		cudaMemcpy(h, d_hash_table, HASH_TABLE_SIZE * sizeof(HashEntry), cudaMemcpyDeviceToHost);
		for (int i = 0; i < HASH_TABLE_SIZE; ++i)
		{
			if (h[i].voxelBlockIndex != FREE_ENTRY)
			{
				float3 p;
				p.x = h[i].position.x * 0.8f + 0.4f;
				p.y = h[i].position.y * 0.8f + 0.4f;
				p.z = h[i].position.z * 0.8f + 0.4f;
				res.push_back(p);
			}
		}
		delete[] h;
		return res;
	}

	std::vector<std::tuple<float3, uchar3, float3>> CuVoxelStreaming::GetActiveZeroCrossingPoints()
	{
		std::vector<std::tuple<float3, uchar3, float3>> res;
		HashEntry* h = new HashEntry[HASH_TABLE_SIZE];
		VoxelBlock* vb = new VoxelBlock[MAX_BLOCKS];
		cudaMemcpy(h, d_hash_table, HASH_TABLE_SIZE * sizeof(HashEntry), cudaMemcpyDeviceToHost);
		cudaMemcpy(vb, d_voxel_blocks, MAX_BLOCKS * sizeof(VoxelBlock), cudaMemcpyDeviceToHost);

		for (int i = 0; i < HASH_TABLE_SIZE; ++i)
		{
			int ptr = h[i].voxelBlockIndex;
			if (ptr < 0)
			{
				continue;
			}

			for (int t = 0; t < 512; ++t)
			{
				int vx = t & 7, vy = (t >> 3) & 7, vz = (t >> 6);
				if (vx >= 7 || vy >= 7 || vz >= 7)
				{
					continue;
				}

				if (vb[ptr].weight[t] == 0)
				{
					continue;
				}

				float v0 = (float)vb[ptr].tsdf[t] / SHORT_MAX_FLOAT;
				int neighbors[3] = { t + 1, t + 8, t + 64 };
				for (int j = 0; j < 3; ++j)
				{
					int nt = neighbors[j];
					if (vb[ptr].weight[nt] > 0)
					{
						float v1 = (float)vb[ptr].tsdf[nt] / SHORT_MAX_FLOAT;
						if (v0 * v1 <= 0.0f && v0 != v1)
						{
							float alpha = fabsf(v0) / (fabsf(v0) + fabsf(v1));
							float3 p0 = make_float3((h[i].position.x * 8 + vx) * VOXEL_SIZE, (h[i].position.y * 8 + vy) * VOXEL_SIZE, (h[i].position.z * 8 + vz) * VOXEL_SIZE);
							float3 p_interp = p0;
							if (j == 0) p_interp.x += alpha * VOXEL_SIZE;
							else if (j == 1) p_interp.y += alpha * VOXEL_SIZE;
							else p_interp.z += alpha * VOXEL_SIZE;

							uchar3 c0 = vb[ptr].color[t], c1 = vb[ptr].color[nt];
							uchar3 c_interp = make_uchar3(
								(unsigned char)((1.0f - alpha) * c0.x + alpha * c1.x + 0.5f),
								(unsigned char)((1.0f - alpha) * c0.y + alpha * c1.y + 0.5f),
								(unsigned char)((1.0f - alpha) * c0.z + alpha * c1.z + 0.5f)
							);

							float3 n0 = vb[ptr].normal[t], n1 = vb[ptr].normal[nt];
							float3 n_interp = make_float3(
								(1.0f - alpha) * n0.x + alpha * n1.x,
								(1.0f - alpha) * n0.y + alpha * n1.y,
								(1.0f - alpha) * n0.z + alpha * n1.z
							);
							float len = sqrtf(n_interp.x * n_interp.x + n_interp.y * n_interp.y + n_interp.z * n_interp.z);
							if (len > 0.0f)
							{
								n_interp.x /= len; n_interp.y /= len; n_interp.z /= len;
							}

							res.push_back({ p_interp, c_interp, n_interp });
						}
					}
				}
			}
		}
		delete[] h; delete[] vb;
		return res;
	}

	std::vector<std::pair<float3, uchar3>> CuVoxelStreaming::GetActiveVoxelData()
	{
		std::vector<std::pair<float3, uchar3>> res;
		HashEntry* h = new HashEntry[HASH_TABLE_SIZE];
		VoxelBlock* b = new VoxelBlock[MAX_BLOCKS];
		cudaMemcpy(h, d_hash_table, HASH_TABLE_SIZE * sizeof(HashEntry), cudaMemcpyDeviceToHost);
		cudaMemcpy(b, d_voxel_blocks, MAX_BLOCKS * sizeof(VoxelBlock), cudaMemcpyDeviceToHost);
		for (int i = 0; i < HASH_TABLE_SIZE; ++i)
		{
			int ptr = h[i].voxelBlockIndex;
			if (ptr == FREE_ENTRY)
			{
				continue;
			}
			for (int t = 0; t < 512; ++t)
			{
				if (b[ptr].weight[t] > 0)
				{
					float3 p;
					p.x = (h[i].position.x * 8 + (t & 7)) * VOXEL_SIZE + 0.05f;
					p.y = (h[i].position.y * 8 + ((t >> 3) & 7)) * VOXEL_SIZE + 0.05f;
					p.z = (h[i].position.z * 8 + (t >> 6)) * VOXEL_SIZE + 0.05f;
					res.push_back({ p, b[ptr].color[t] });
				}
			}
		}
		delete[] h; delete[] b;
		return res;
	}

	std::vector<float3> CuVoxelStreaming::GetActiveVoxelPositions()
	{
		std::vector<float3> res;
		auto data = GetActiveVoxelData();
		for (auto& d : data)
		{
			res.push_back(d.first);
		}
		return res;
	}
}
