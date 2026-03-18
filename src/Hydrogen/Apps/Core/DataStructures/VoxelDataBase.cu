#include <Core/DataStructures/VoxelDataBase.h>
#include <Core/Common/DeviceCommon.h>

typedef unsigned char uchar;
typedef float voxel_value_t;

namespace Huvitz
{
	struct VoxelExtraAttrib {
		static const VoxelExtraAttrib Zero;

		uchar deepLearningClass; // enum DL_Class_Names
		uchar materialID; // 0: other 255: tooth
		unsigned short startPatchID;	// 복셀의 조합에 영향을 미친 첫 패치 번호
		unsigned int flags : 2; // 복셀의 잠금상태 등의 상태를 저장.VOXEL_FLAG_**** _BIT 로 비교
		unsigned int label : 30;	// 클러스터링 된 레이블 번호

		uint8_t colorMap[4]; // colors[3], alpha (신뢰도)
	};

	struct Voxel {
		voxel_value_t		value;
		unsigned short		valueCount;
		Eigen::Vector3f		normal;
		Eigen::Vector3b		color;

		Eigen::Vector3b		color_list[3];
		uint8_t				color_score[3];

		char				segmentation;
		VoxelExtraAttrib	extraAttrib;
	};

	template <typename T>
	__device__ inline float GetVoxelValue(VoxelDataBase<T>& db, const Eigen::Vector3f& pos)
	{
		T* v = db.GetVoxel(pos);
		if (v != nullptr && v->valueCount > 0)
		{
			return v->value;
		}
		return FLT_MAX;
	}

	template <typename T>
	__global__ void Kernel_Clear(VoxelDataBase<T> db)
	{
		uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
		if (slot >= db.GetMaxBlockCount())
		{
			return;
		}

		uint64_t* hashTable = db.GetHashTable();
		uint64_t key = hashTable[slot];

		if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL)
		{
			return;
		}

		hashTable[slot] = 0;

		VoxelBlock<T>* blockPtr = &db.GetBlocks()[slot];
		*blockPtr = {};
	}

	template <typename T>
	__global__ void Kernel_Integrate(VoxelDataBase<T> db, VoxelDataBaseIntegrationParameters parameters)
	{
		uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid >= parameters.mapWidth * parameters.mapHeight)
		{
			return;
		}

		uint32_t u = threadid % parameters.mapWidth;
		uint32_t v = threadid / parameters.mapWidth;

		Eigen::Vector3f p_cam = parameters.d_depthMap[v * parameters.mapWidth + u];
		if (false == VECTOR3F_VALID_(p_cam))
		{
			return;
		}

		if (p_cam.z() <= 0)
		{
			return;
		}

		Eigen::Vector3f n_cam = parameters.d_normalMap[v * parameters.mapWidth + u];
		auto r = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 0];
		auto g = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 1];
		auto b = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 2];
		Eigen::Vector3b color(r, g, b);

		Eigen::Vector4f p_world4 = parameters.transform * Eigen::Vector4f(p_cam.x(), p_cam.y(), p_cam.z(), 1.0f);
		Eigen::Vector4f n_world4 = parameters.transform * Eigen::Vector4f(n_cam.x(), n_cam.y(), n_cam.z(), 0.0f);

		Eigen::Vector3f p_world = { p_world4.x(), p_world4.y(), p_world4.z() };
		Eigen::Vector3f n_world = { n_world4.x(), n_world4.y(), n_world4.z() };

		float voxelSize = db.GetBlockSize() / 8.0f;
		float truncDist = voxelSize * 10.0f;
		float invVoxelSize = 1.0f / voxelSize;

		Eigen::Vector3f camPos = { parameters.transform(0, 3), parameters.transform(1, 3), parameters.transform(2, 3) };
		Eigen::Vector3f rayDir = p_world - camPos;
		float rayLen = sqrtf(rayDir.x() * rayDir.x() + rayDir.y() * rayDir.y() + rayDir.z() * rayDir.z());

		if (rayLen < 1e-6f)
		{
			return;
		}

		rayDir.x() /= rayLen;
		rayDir.y() /= rayLen;
		rayDir.z() /= rayLen;

		float tStart = fmaxf(0.0f, rayLen - truncDist);
		float tEnd = rayLen + truncDist;
		float step = voxelSize * 0.8f;

		VoxelBlock<T>* cachedBlock = nullptr;
		uint64_t       cachedBlockKey = 0;
		Eigen::Vector3f       cachedBlockCenter = { 0.0f, 0.0f, 0.0f };
		float          blockSize = db.GetBlockSize();
		float          halfBlock = blockSize * 0.5f;

		for (float t = tStart; t <= tEnd; t += step)
		{
			Eigen::Vector3f samplePos;
			samplePos.x() = camPos.x() + rayDir.x() * t;
			samplePos.y() = camPos.y() + rayDir.y() * t;
			samplePos.z() = camPos.z() + rayDir.z() * t;

			Morton64 blockKey = Morton64::FromPosition(samplePos, blockSize);
			uint64_t key = blockKey.code;
			if (key == 0) key = 0xFFFFFFFFFFFFFFFFULL;

			if (key != cachedBlockKey)
			{
				cachedBlock = db.GetOrCreateVoxelBlock(samplePos);
				cachedBlockKey = key;
				cachedBlockCenter = blockKey.ToPosition(blockSize);
			}

			if (cachedBlock == nullptr)
			{
				continue;
			}

			int lx = static_cast<int>(floorf((samplePos.x() - (cachedBlockCenter.x() - halfBlock)) * invVoxelSize + 1e-4f));
			int ly = static_cast<int>(floorf((samplePos.y() - (cachedBlockCenter.y() - halfBlock)) * invVoxelSize + 1e-4f));
			int lz = static_cast<int>(floorf((samplePos.z() - (cachedBlockCenter.z() - halfBlock)) * invVoxelSize + 1e-4f));
			lx = (lx < 0) ? 0 : (lx > 7 ? 7 : lx);
			ly = (ly < 0) ? 0 : (ly > 7 ? 7 : ly);
			lz = (lz < 0) ? 0 : (lz > 7 ? 7 : lz);

			T* voxelPtr = &(cachedBlock->voxels[(lz << 6) | (ly << 3) | lx]);

			float sdf = rayLen - t;

			unsigned short oldWeightUS = atomicAddUShort(&(voxelPtr->valueCount), 1);
			float oldWeight = fminf((float)oldWeightUS, MAX_WEIGHT);
			float newWeight = oldWeight + 1.0f;

			float weightFactor = oldWeight / newWeight;
			float invNewWeight = 1.0f / newWeight;

			float tsdfValue = fmaxf(-1.0f, fminf(1.0f, sdf / truncDist));
			voxelPtr->value = (voxelPtr->value * weightFactor) + (tsdfValue * invNewWeight);

			voxelPtr->color.x() = (uint8_t)((float)voxelPtr->color.x() * weightFactor + (float)color.x() * invNewWeight + 0.5f);
			voxelPtr->color.y() = (uint8_t)((float)voxelPtr->color.y() * weightFactor + (float)color.y() * invNewWeight + 0.5f);
			voxelPtr->color.z() = (uint8_t)((float)voxelPtr->color.z() * weightFactor + (float)color.z() * invNewWeight + 0.5f);

			float bNx = voxelPtr->normal.x() * weightFactor + n_world.x() * invNewWeight;
			float bNy = voxelPtr->normal.y() * weightFactor + n_world.y() * invNewWeight;
			float bNz = voxelPtr->normal.z() * weightFactor + n_world.z() * invNewWeight;

			float nInvLen = rsqrtf(bNx * bNx + bNy * bNy + bNz * bNz + 1e-10f);

			voxelPtr->normal.x() = bNx * nInvLen;
			voxelPtr->normal.y() = bNy * nInvLen;
			voxelPtr->normal.z() = bNz * nInvLen;
		}
	}

	struct NoiseFilter2DCache
	{
		float* d_data;
		uint32_t dimX;
		uint32_t dimY;
		float    originX;
		float    originY;
		float    cellSize;
	};

	// Step 1: depth map -> 2D XY 깊이 캐시 구축
	template <typename T>
	__global__ void Kernel_BuildNoiseFilter2DCache(NoiseFilterParameters nf)
	{
		uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadid >= nf.mapWidth * nf.mapHeight)
			return;

		Eigen::Vector3f p_cam = nf.d_depthMap[threadid];
		if (false == VECTOR3F_VALID_(p_cam) || p_cam.z() <= 0.0f)
			return;

		Eigen::Vector4f p_world4 = nf.transform *
			Eigen::Vector4f(p_cam.x(), p_cam.y(), p_cam.z(), 1.0f);

		int ix = (int)floorf((p_world4.x() - nf.originX) / nf.cellSize);
		int iy = (int)floorf((p_world4.y() - nf.originY) / nf.cellSize);

		if (ix < 0 || iy < 0 || ix >= (int)nf.dimX || iy >= (int)nf.dimY)
			return;

		// 동일 컬럼에 여러 픽셀이 동시에 쓰므로 atomic min (가장 가까운 표면 유지)
		atomicMinFloat(&nf.d_depthCache[(uint32_t)iy * nf.dimX + (uint32_t)ix], p_world4.z());
	}

	template <typename T>
	__global__ void Kernel_IntraRegionNoiseFilter(VoxelDataBase<T> db, NoiseFilterParameters nf)
	{
		uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
		if (slot >= db.GetMaxBlockCount())
			return;

		uint64_t key = db.GetHashTable()[slot];
		if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL)
			return;

		Morton64 bKey(key);
		float    bSize = db.GetBlockSize();
		float    vSize = bSize / 8.0f;
		Eigen::Vector3f bc = bKey.ToPosition(bSize);
		Eigen::Vector3f origin = { bc.x() - bSize * 0.5f, bc.y() - bSize * 0.5f, bc.z() - bSize * 0.5f };

		const uint16_t MIN_WEIGHT = 3;
		const uint16_t WEAK_WEIGHT = 10;
		const int      MIN_NEIGHBORS = 2;
		const float    nearTolerance = vSize * 2.0f;
		const float    farCutoff = vSize * 500.0f;

		// constexpr 배열 대신 static const 로 선언 (CUDA 커널 내부 호환)
		static const int DX[6] = { 1, -1,  0,  0,  0,  0 };
		static const int DY[6] = { 0,  0,  1, -1,  0,  0 };
		static const int DZ[6] = { 0,  0,  0,  0,  1, -1 };

		VoxelBlock<T>& block = db.GetBlocks()[slot];

		for (int lz = 0; lz < 8; ++lz)
			for (int ly = 0; ly < 8; ++ly)
				for (int lx = 0; lx < 8; ++lx)
				{
					T& voxel = block.voxels[(lz << 6) | (ly << 3) | lx];
					if (voxel.valueCount < MIN_WEIGHT)
						continue;

					float wx = origin.x() + (lx + 0.5f) * vSize;
					float wy = origin.y() + (ly + 0.5f) * vSize;
					float wz = origin.z() + (lz + 0.5f) * vSize;

					// Phase 1: 2D 깊이 캐시 조회
					int   ix = (int)floorf((wx - nf.originX) / nf.cellSize);
					int   iy = (int)floorf((wy - nf.originY) / nf.cellSize);
					bool  inCache = (ix >= 0 && iy >= 0 && ix < (int)nf.dimX && iy < (int)nf.dimY);
					float surfaceZ = inCache ? nf.d_depthCache[(uint32_t)iy * nf.dimX + (uint32_t)ix] : FLT_MAX;
					bool  hasSurface = (surfaceZ < FLT_MAX * 0.5f);

					if (false == hasSurface)
					{
						if (voxel.valueCount < WEAK_WEIGHT)
							voxel = {};
						continue;
					}

					if (wz > surfaceZ + nearTolerance && wz <= surfaceZ + farCutoff)
					{
						voxel = {};
						continue;
					}

					// Phase 2: 약한 복셀 6-connectivity 이웃 검사
					if (voxel.valueCount >= WEAK_WEIGHT)
						continue;

					int occupiedNeighbors = 0;
					for (int d = 0; d < 6; ++d)
					{
						int nx = lx + DX[d];
						int ny = ly + DY[d];
						int nz = lz + DZ[d];

						T* neighbor = nullptr;
						bool isInternal = (nx >= 0 && nx < 8 && ny >= 0 && ny < 8 && nz >= 0 && nz < 8);

						if (isInternal)
						{
							neighbor = &block.voxels[(nz << 6) | (ny << 3) | nx];
						}
						else
						{
							Eigen::Vector3f neighborPos = {
								origin.x() + (nx + 0.5f) * vSize,
								origin.y() + (ny + 0.5f) * vSize,
								origin.z() + (nz + 0.5f) * vSize
							};
							neighbor = db.GetVoxel(neighborPos);
						}

						if (neighbor != nullptr && neighbor->valueCount >= MIN_WEIGHT)
							++occupiedNeighbors;
					}

					if (occupiedNeighbors < MIN_NEIGHBORS)
						voxel = {};
				}
	}

	template <typename T>
	__global__ void InsertKernel(VoxelDataBase<T> db, Eigen::Matrix4f rt, const Eigen::Vector3f* points, const Eigen::Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
	{
		uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= count)
		{
			return;
		}

		Eigen::Vector3f p = (rt * Eigen::Vector4f(points[idx], 1.0f)).head<3>();

		T* v = db.GetOrCreateVoxel(p);

		if (v != nullptr)
		{
			atomicAdd(&(v->value), 1.0f);

			if (v->valueCount < 65535)
			{
				atomicAddUShort(&(v->valueCount), 1);
			}

			v->color = colors[idx];

			auto block = db.GetVoxelBlock(p);
			if (nullptr != block)
			{
				block->lastTouchedFrameId = frameId;
			}
		}
	}

	template <typename T>
	__global__ void TSDFIntegrateKernel(VoxelDataBase<T> db, Eigen::Matrix4f rt, const Eigen::Vector3f* points, const Eigen::Vector3f* normals, const Eigen::Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
	{
		uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index >= count)
		{
			return;
		}

		Eigen::Vector3f p_local = points[index];
		Eigen::Vector3f n_local = normals[index];
		Eigen::Vector3b color = colors[index];

		Eigen::Vector3f p_world = (rt * Eigen::Vector4f(p_local, 1.0f)).head<3>();
		Eigen::Vector3f n_world = (rt * Eigen::Vector4f(n_local, 0.0f)).head<3>();

		float n_len = rsqrtf(n_world.x() * n_world.x() + n_world.y() * n_world.y() + n_world.z() * n_world.z() + 1e-10f);
		n_world.x() *= n_len; n_world.y() *= n_len; n_world.z() *= n_len;

		float voxel_size = blockSize / 8.0f;
		float inv_voxel_size = 1.0f / voxel_size;
		float trunc_dist = voxel_size * 5.0f;

		Eigen::Vector3f start_pos = {
			p_world.x() - n_world.x() * trunc_dist,
			p_world.y() - n_world.y() * trunc_dist,
			p_world.z() - n_world.z() * trunc_dist
		};

		int curr_x = __float2int_rd(start_pos.x() * inv_voxel_size);
		int curr_y = __float2int_rd(start_pos.y() * inv_voxel_size);
		int curr_z = __float2int_rd(start_pos.z() * inv_voxel_size);

		int step_x = (n_world.x() > 0) ? 1 : -1;
		int step_y = (n_world.y() > 0) ? 1 : -1;
		int step_z = (n_world.z() > 0) ? 1 : -1;
		auto calc_t_max = [&](float pos, float dir, int step, int curr) {
			if (fabsf(dir) < 1e-7f) return 1e30f;
			float border = (float)(curr + (step > 0 ? 1 : 0)) * voxel_size;
			return (border - pos) / dir;
			};

		float t_max_x = calc_t_max(start_pos.x(), n_world.x(), step_x, curr_x);
		float t_max_y = calc_t_max(start_pos.y(), n_world.y(), step_y, curr_y);
		float t_max_z = calc_t_max(start_pos.z(), n_world.z(), step_z, curr_z);

		float t_delta_x = (fabsf(n_world.x()) > 1e-7f) ? fabsf(voxel_size / n_world.x()) : 1e30f;
		float t_delta_y = (fabsf(n_world.y()) > 1e-7f) ? fabsf(voxel_size / n_world.y()) : 1e30f;
		float t_delta_z = (fabsf(n_world.z()) > 1e-7f) ? fabsf(voxel_size / n_world.z()) : 1e30f;

		float max_t = 2.0f * trunc_dist;
		float t = 0.0f;

		while (t <= max_t)
		{
			Eigen::Vector3f voxel_center = { (curr_x + 0.5f) * voxel_size, (curr_y + 0.5f) * voxel_size, (curr_z + 0.5f) * voxel_size };
			T* voxel_ptr = db.GetOrCreateVoxel(voxel_center);

			if (voxel_ptr != nullptr)
			{
				float dist = (voxel_center.x() - p_world.x()) * n_world.x() +
					(voxel_center.y() - p_world.y()) * n_world.y() +
					(voxel_center.z() - p_world.z()) * n_world.z();

				unsigned short old_w_us = atomicAddUShort(&(voxel_ptr->valueCount), 1);

				float old_w = fminf((float)old_w_us, MAX_WEIGHT);
				float new_w = old_w + 1.0f;

				float weight_factor = old_w / new_w;
				float new_factor = 1.0f / new_w;

				// 1. TSDF 값 통합
				voxel_ptr->value = (voxel_ptr->value * weight_factor) + (dist * new_factor);

				// 2. Color 통합
				voxel_ptr->color.x() = (uint8_t)((float)voxel_ptr->color.x() * weight_factor + (float)color.x() * new_factor + 0.5f);
				voxel_ptr->color.y() = (uint8_t)((float)voxel_ptr->color.y() * weight_factor + (float)color.y() * new_factor + 0.5f);
				voxel_ptr->color.z() = (uint8_t)((float)voxel_ptr->color.z() * weight_factor + (float)color.z() * new_factor + 0.5f);

				// 3. Normal 통합
				Eigen::Vector3f blended_n = {
					voxel_ptr->normal.x() * weight_factor + n_world.x() * new_factor,
					voxel_ptr->normal.y() * weight_factor + n_world.y() * new_factor,
					voxel_ptr->normal.z() * weight_factor + n_world.z() * new_factor
				};
				float bn_len = rsqrtf(blended_n.x() * blended_n.x() + blended_n.y() * blended_n.y() + blended_n.z() * blended_n.z() + 1e-10f);
				voxel_ptr->normal = { blended_n.x() * bn_len, blended_n.y() * bn_len, blended_n.z() * bn_len };
				VoxelBlock<T>* block = db.GetVoxelBlock(voxel_center);
				if (block)
				{
					block->lastTouchedFrameId = frameId;
				}
			}

			if (t_max_x < t_max_y)
			{
				if (t_max_x < t_max_z) { t = t_max_x; t_max_x += t_delta_x; curr_x += step_x; }
				else { t = t_max_z; t_max_z += t_delta_z; curr_z += step_z; }
			}
			else
			{
				if (t_max_y < t_max_z) { t = t_max_y; t_max_y += t_delta_y; curr_y += step_y; }
				else { t = t_max_z; t_max_z += t_delta_z; curr_z += step_z; }
			}
		}
	}

	template <typename T>
	__global__ void Kernel_ExtractAllOccupied(
		VoxelDataBase<T> db, float blockSize,
		ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
	{
		uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
		if (slot >= db.GetMaxBlockCount()) return;

		uint64_t key = db.GetHashTable()[slot];
		if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL) return;

		Morton64 bKey(key);
		Eigen::Vector3f bc = bKey.ToPosition(blockSize);
		float    vSize = blockSize / 8.0f;

		for (int i = 0; i < 512; ++i)
		{
			T& v = db.GetBlocks()[slot].voxels[i];
			if (v.valueCount == 0) continue;

			uint32_t idx = atomicAdd(count, 1);
			if (idx >= maxOut) return;

			int lz = i / 64;
			int ly = (i % 64) / 8;
			int lx = i % 8;

			out[idx].position.x() = (bc.x() - blockSize * 0.5f) + (lx + 0.5f) * vSize;
			out[idx].position.y() = (bc.y() - blockSize * 0.5f) + (ly + 0.5f) * vSize;
			out[idx].position.z() = (bc.z() - blockSize * 0.5f) + (lz + 0.5f) * vSize;
			out[idx].normal = v.normal;
			out[idx].weight = (float)v.valueCount;
			out[idx].color[0] = v.color.x();
			out[idx].color[1] = v.color.y();
			out[idx].color[2] = v.color.z();
		}
	}

	template <typename T>
	__global__ void Kernel_ExtractZeroCrossing(
		VoxelDataBase<T> db, float blockSize,
		ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
	{
		uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
		if (slot >= db.GetMaxBlockCount()) return;

		uint64_t key = db.GetHashTable()[slot];
		if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL) return;

		Morton64 bKey(key);
		Eigen::Vector3f bc = bKey.ToPosition(blockSize);
		Eigen::Vector3f origin = { bc.x() - blockSize * 0.5f, bc.y() - blockSize * 0.5f, bc.z() - blockSize * 0.5f};
		float voxelSize = blockSize / 8.0f;
		float eps = voxelSize;

		for (int lz = 0; lz < 8; ++lz)
			for (int ly = 0; ly < 8; ++ly)
				for (int lx = 0; lx < 8; ++lx)
				{
					int idx0 = (lz << 6) | (ly << 3) | lx;
					T& v0 = db.GetBlocks()[slot].voxels[idx0];

					if (v0.valueCount < 5) continue;

					Eigen::Vector3f p0 = {
						origin.x() + (lx + 0.5f) * voxelSize,
						origin.y() + (ly + 0.5f) * voxelSize,
						origin.z() + (lz + 0.5f) * voxelSize
					};

					int neighborStride[3] = { 1, 8, 64 };

					for (int axis = 0; axis < 3; ++axis)
					{
						bool isInternal = false;
						if (axis == 0 && lx < 7) isInternal = true;
						else if (axis == 1 && ly < 7) isInternal = true;
						else if (axis == 2 && lz < 7) isInternal = true;

						Eigen::Vector3f p1 = p0;
						if (axis == 0) p1.x() += voxelSize;
						else if (axis == 1) p1.y() += voxelSize;
						else                p1.z() += voxelSize;

						T* v1 = isInternal
							? &db.GetBlocks()[slot].voxels[idx0 + neighborStride[axis]]
							: db.GetVoxel(p1);

						if (v1 == nullptr || v1->valueCount < 5) continue;

						float f0 = v0.value;
						float f1 = v1->value;
						if (f0 * f1 >= 0.0f) continue;

						uint32_t outIdx = atomicAdd(count, 1);
						if (outIdx >= maxOut) return;

						float mu = fminf(fmaxf(-f0 / (f1 - f0), 0.0f), 1.0f);

						Eigen::Vector3f interpPos = {
							p0.x() + mu * (p1.x() - p0.x()),
							p0.y() + mu * (p1.y() - p0.y()),
							p0.z() + mu * (p1.z() - p0.z())
						};

						out[outIdx].position = interpPos;
						out[outIdx].weight = (float)v0.valueCount;

						// gradient 기반 normal
						float vxp = GetVoxelValue<T>(db, { interpPos.x() + eps, interpPos.y(),       interpPos.z()});
						float vxm = GetVoxelValue<T>(db, { interpPos.x() - eps, interpPos.y(),       interpPos.z() });
						float vyp = GetVoxelValue<T>(db, { interpPos.x(),       interpPos.y() + eps, interpPos.z() });
						float vym = GetVoxelValue<T>(db, { interpPos.x(),       interpPos.y() - eps, interpPos.z() });
						float vzp = GetVoxelValue<T>(db, { interpPos.x(),       interpPos.y(),       interpPos.z() + eps });
						float vzm = GetVoxelValue<T>(db, { interpPos.x(),       interpPos.y(),       interpPos.z() - eps });

						bool hasGrad = (vxp < FLT_MAX * 0.5f) && (vxm < FLT_MAX * 0.5f) &&
							(vyp < FLT_MAX * 0.5f) && (vym < FLT_MAX * 0.5f) &&
							(vzp < FLT_MAX * 0.5f) && (vzm < FLT_MAX * 0.5f);

						if (hasGrad)
						{
							Eigen::Vector3f grad = { vxp - vxm, vyp - vym, vzp - vzm };
							float len = sqrtf(grad.x() * grad.x() + grad.y() * grad.y() + grad.z() * grad.z());
							out[outIdx].normal = (len > 1e-6f)
								? Eigen::Vector3f(grad.x() / len, grad.y() / len, grad.z() / len)
							: Eigen::Vector3f(0.0f, 1.0f, 0.0f);
						}
						else
						{
							Eigen::Vector3f blended = {
								v0.normal.x() + mu * (v1->normal.x() - v0.normal.x()),
								v0.normal.y() + mu * (v1->normal.y() - v0.normal.y()),
								v0.normal.z() + mu * (v1->normal.z() - v0.normal.z())
							};
							float len = sqrtf(blended.x() * blended.x() + blended.y() * blended.y() + blended.z() * blended.z());
							out[outIdx].normal = (len > 1e-6f)
								? Eigen::Vector3f(blended.x() / len, blended.y() / len, blended.z() / len)
							: v0.normal;
						}

						// color 보간
						out[outIdx].color[0] = (uint8_t)(v0.color.x() + mu * (float(v1->color.x()) - v0.color.x()));
						out[outIdx].color[1] = (uint8_t)(v0.color.y() + mu * (float(v1->color.y()) - v0.color.y()));
						out[outIdx].color[2] = (uint8_t)(v0.color.z() + mu * (float(v1->color.z()) - v0.color.z()));
					}
				}
	}

	template <typename T>
	__global__ void Kernel_ExtractIntraRegion(
		VoxelDataBase<T> db, float blockSize,
		VoxelDataBaseExtractionParameters::Mode mode,
		Eigen::Vector3f aabbMin, Eigen::Vector3f aabbMax,
		ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
	{
		uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
		if (slot >= db.GetMaxBlockCount()) return;

		uint64_t key = db.GetHashTable()[slot];
		if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL) return;

		Morton64 bKey(key);
		Eigen::Vector3f bc = bKey.ToPosition(blockSize);

		Eigen::Vector3f bMin = { bc.x() - blockSize * 0.5f, bc.y() - blockSize * 0.5f, bc.z() - blockSize * 0.5f };
		Eigen::Vector3f bMax = { bc.x() + blockSize * 0.5f, bc.y() + blockSize * 0.5f, bc.z() + blockSize * 0.5f };
		if (bMax.x() < aabbMin.x() || bMin.x() > aabbMax.x()) return;
		if (bMax.y() < aabbMin.y() || bMin.y() > aabbMax.y()) return;
		if (bMax.z() < aabbMin.z() || bMin.z() > aabbMax.z()) return;

		Eigen::Vector3f origin = bMin;
		float voxelSize = blockSize / 8.0f;

		auto sampleTSDF = [&](float sx, float sy, float sz) -> float
			{
				if (sx >= bMin.x() && sx < bMax.x() &&
					sy >= bMin.y() && sy < bMax.y() &&
					sz >= bMin.z() && sz < bMax.z())
				{
					int lx_ = (int)floorf((sx - bMin.x()) / voxelSize);
					int ly_ = (int)floorf((sy - bMin.y()) / voxelSize);
					int lz_ = (int)floorf((sz - bMin.z()) / voxelSize);
					lx_ = lx_ < 0 ? 0 : (lx_ > 7 ? 7 : lx_);
					ly_ = ly_ < 0 ? 0 : (ly_ > 7 ? 7 : ly_);
					lz_ = lz_ < 0 ? 0 : (lz_ > 7 ? 7 : lz_);
					T& vn = db.GetBlocks()[slot].voxels[(lz_ << 6) | (ly_ << 3) | lx_];
					return (vn.valueCount > 0) ? vn.value : FLT_MAX;
				}
				Eigen::Vector3f p = { sx, sy, sz };
				return GetVoxelValue<T>(db, p);
			};

		if (mode == VoxelDataBaseExtractionParameters::Mode::AllOccupied)
		{
			for (int i = 0; i < 512; ++i)
			{
				T& v = db.GetBlocks()[slot].voxels[i];
				if (v.valueCount == 0) continue;

				int lz = i / 64;
				int ly = (i % 64) / 8;
				int lx = i % 8;

				Eigen::Vector3f pos = {
					origin.x() + (lx + 0.5f) * voxelSize,
					origin.y() + (ly + 0.5f) * voxelSize,
					origin.z() + (lz + 0.5f) * voxelSize
				};

				if (pos.x() < aabbMin.x() || pos.x() > aabbMax.x()) continue;
				if (pos.y() < aabbMin.y() || pos.y() > aabbMax.y()) continue;
				if (pos.z() < aabbMin.z() || pos.z() > aabbMax.z()) continue;

				uint32_t idx = atomicAdd(count, 1);
				if (idx >= maxOut) return;

				out[idx].position = pos;
				out[idx].normal = v.normal;
				out[idx].weight = (float)v.valueCount;
				out[idx].color[0] = v.color.x();
				out[idx].color[1] = v.color.y();
				out[idx].color[2] = v.color.z();
			}
		}
		else if (mode == VoxelDataBaseExtractionParameters::Mode::ZeroCrossing)
		{
			float eps = voxelSize;
			int neighborStride[3] = { 1, 8, 64 };

			for (int lz = 0; lz < 8; ++lz)
				for (int ly = 0; ly < 8; ++ly)
					for (int lx = 0; lx < 8; ++lx)
					{
						int idx0 = (lz << 6) | (ly << 3) | lx;
						T& v0 = db.GetBlocks()[slot].voxels[idx0];
						if (v0.valueCount < 5) continue;

						Eigen::Vector3f p0 = {
							origin.x() + (lx + 0.5f) * voxelSize,
							origin.y() + (ly + 0.5f) * voxelSize,
							origin.z() + (lz + 0.5f) * voxelSize
						};

						for (int axis = 0; axis < 3; ++axis)
						{
							bool isInternal = false;
							if (axis == 0 && lx < 7) isInternal = true;
							else if (axis == 1 && ly < 7) isInternal = true;
							else if (axis == 2 && lz < 7) isInternal = true;

							Eigen::Vector3f p1 = p0;
							if (axis == 0) p1.x() += voxelSize;
							else if (axis == 1) p1.y() += voxelSize;
							else                p1.z() += voxelSize;

							T* v1 = isInternal
								? &db.GetBlocks()[slot].voxels[idx0 + neighborStride[axis]]
								: db.GetVoxel(p1);

							if (v1 == nullptr || v1->valueCount < 5) continue;

							float f0 = v0.value;
							float f1 = v1->value;
							if (f0 * f1 >= 0.0f) continue;

							float mu = fminf(fmaxf(-f0 / (f1 - f0), 0.0f), 1.0f);

							Eigen::Vector3f interpPos = {
								p0.x() + mu * (p1.x() - p0.x()),
								p0.y() + mu * (p1.y() - p0.y()),
								p0.z() + mu * (p1.z() - p0.z())
							};

							if (interpPos.x() < aabbMin.x() || interpPos.x() > aabbMax.x()) continue;
							if (interpPos.y() < aabbMin.y() || interpPos.y() > aabbMax.y()) continue;
							if (interpPos.z() < aabbMin.z() || interpPos.z() > aabbMax.z()) continue;

							uint32_t outIdx = atomicAdd(count, 1);
							if (outIdx >= maxOut) return;

							out[outIdx].position = interpPos;
							out[outIdx].weight = (float)v0.valueCount;

							float vxp = sampleTSDF(interpPos.x() + eps, interpPos.y(), interpPos.z());
							float vxm = sampleTSDF(interpPos.x() - eps, interpPos.y(), interpPos.z());
							float vyp = sampleTSDF(interpPos.x(), interpPos.y() + eps, interpPos.z());
							float vym = sampleTSDF(interpPos.x(), interpPos.y() - eps, interpPos.z());
							float vzp = sampleTSDF(interpPos.x(), interpPos.y(), interpPos.z() + eps);
							float vzm = sampleTSDF(interpPos.x(), interpPos.y(), interpPos.z() - eps);

							bool hasGrad = (vxp < FLT_MAX * 0.5f) && (vxm < FLT_MAX * 0.5f) &&
								(vyp < FLT_MAX * 0.5f) && (vym < FLT_MAX * 0.5f) &&
								(vzp < FLT_MAX * 0.5f) && (vzm < FLT_MAX * 0.5f);

							if (hasGrad)
							{
								Eigen::Vector3f grad = { vxp - vxm, vyp - vym, vzp - vzm };
								float len = sqrtf(grad.x() * grad.x() + grad.y() * grad.y() + grad.z() * grad.z());
								out[outIdx].normal = (len > 1e-6f)
									? Eigen::Vector3f{ grad.x() / len, grad.y() / len, grad.z() / len }
								: Eigen::Vector3f{ 0.0f, 1.0f, 0.0f };
							}
							else
							{
								Eigen::Vector3f blended = {
									v0.normal.x() + mu * (v1->normal.x() - v0.normal.x()),
									v0.normal.y() + mu * (v1->normal.y() - v0.normal.y()),
									v0.normal.z() + mu * (v1->normal.z() - v0.normal.z())
								};
								float len = sqrtf(blended.x() * blended.x() + blended.y() * blended.y() + blended.z() * blended.z());
								out[outIdx].normal = (len > 1e-6f)
									? Eigen::Vector3f{ blended.x() / len, blended.y() / len, blended.z() / len }
								: v0.normal;
							}

							out[outIdx].color[0] = (uint8_t)(v0.color.x() + mu * (float(v1->color.x()) - v0.color.x()));
							out[outIdx].color[1] = (uint8_t)(v0.color.y() + mu * (float(v1->color.y()) - v0.color.y()));
							out[outIdx].color[2] = (uint8_t)(v0.color.z() + mu * (float(v1->color.z()) - v0.color.z()));
						}
					}
		}
	}

	template <typename T>
	__global__ void Kernel_Serialize(
		VoxelDataBase<T> db,
		float3* positions,
		float3* normals,
		uchar3* colors,
		uint32_t* numberOfVoxels,
		uint32_t maxOut)
	{
		uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
		if (slot >= db.GetMaxBlockCount())
		{
			return;
		}

		uint64_t key = db.GetHashTable()[slot];
		if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL)
		{
			return;
		}

		auto& block = db.GetBlocks()[slot];

		Morton64 bKey(key);
		float bSize = db.GetBlockSize();
		float vSize = bSize / 8.0f;
		Eigen::Vector3f blockOrigin = {
			bKey.ToPosition(bSize).x() - bSize * 0.5f,
			bKey.ToPosition(bSize).y() - bSize * 0.5f,
			bKey.ToPosition(bSize).z() - bSize * 0.5f
		};

		for (int voxelIdx = 0; voxelIdx < 512; ++voxelIdx)
		{
			auto& voxel = block.voxels[voxelIdx];
			if (0 == voxel.valueCount) continue;

			uint32_t outIdx = atomicAdd(numberOfVoxels, 1);
			if (outIdx >= maxOut) return;

			int lz = voxelIdx / 64;
			int ly = (voxelIdx % 64) / 8;
			int lx = voxelIdx % 8;

			positions[outIdx] = {
				blockOrigin.x() + (lx + 0.5f) * vSize,
				blockOrigin.y() + (ly + 0.5f) * vSize,
				blockOrigin.z() + (lz + 0.5f) * vSize
			};

			normals[outIdx] = {
				voxel.normal.x(),
				voxel.normal.y(),
				voxel.normal.z()
			};

			colors[outIdx] = {
				(unsigned char)voxel.color.x(),
				(unsigned char)voxel.color.y(),
				(unsigned char)voxel.color.z()
			};
		}
	}

	template <typename T>
	void VoxelDataBase<T>::Initialize(uint32_t maxBlocks, float blockSize = 0.8f)
	{
		maxBlockCount = maxBlocks;
		this->blockSize = blockSize;

		cudaMalloc(&d_blocks, sizeof(VoxelBlock<T>) * maxBlocks);
		cudaMalloc(&d_hashTable, sizeof(uint64_t) * maxBlocks);
		cudaMalloc(&d_blockCount, sizeof(uint32_t));
		cudaMemset(d_blocks, 0, sizeof(VoxelBlock<T>) * maxBlocks);
		cudaMemset(d_hashTable, 0, sizeof(uint64_t) * maxBlocks);
		cudaMemset(d_blockCount, 0, sizeof(uint32_t));
	}

	template <typename T>
	void VoxelDataBase<T>::Terminate()
	{
		if (d_blocks) cudaFree(d_blocks);
		if (d_hashTable) cudaFree(d_hashTable);
		if (d_blockCount) cudaFree(d_blockCount);
		d_blocks = nullptr;
		d_hashTable = nullptr;
		d_blockCount = nullptr;
	}

	template <typename T>
	void VoxelDataBase<T>::IntegrateTSDF(const Eigen::Matrix4f& rt, const Eigen::Vector3f* d_points, const Eigen::Vector3f* d_normals, const Eigen::Vector3b* d_colors, uint32_t count, float blockSize, uint32_t frameId)
	{
#ifdef __CUDACC__
		int threads = 256;
		int blocks = (count + threads - 1) / threads;
		//TSDFIntegrateKernel << <blocks, threads >> > (*this, rt, d_points, d_normals, d_colors, count, blockSize, frameId);
		cudaDeviceSynchronize();
#endif
	}

	template <typename T>
	__device__ VoxelBlock<T>* VoxelDataBase<T>::GetVoxelBlock(const Eigen::Vector3f& position)
	{
		uint32_t slot = FindBlockSlot(position);
		return (slot != INVALID_BLOCK) ? &d_blocks[slot] : nullptr;
	}

	template <typename T>
	__device__ VoxelBlock<T>* VoxelDataBase<T>::GetOrCreateVoxelBlock(const Eigen::Vector3f& position)
	{
		uint32_t slot = GetOrCreateBlockSlot(position);
		return (slot != INVALID_BLOCK) ? &d_blocks[slot] : nullptr;
	}

	template <typename T>
	__device__ T* VoxelDataBase<T>::GetVoxel(const Eigen::Vector3f& position)
	{
		uint32_t slot = FindBlockSlot(position);
		if (slot == INVALID_BLOCK) return nullptr;

		Morton64 blockKey = Morton64::FromPosition(position, blockSize);
		Eigen::Vector3f bc = blockKey.ToPosition(blockSize);

		const float vSize = blockSize / 8.0f;

		int lx = static_cast<int>(floorf((position.x() - (bc.x() - blockSize * 0.5f)) / vSize + 1e-5f));
		int ly = static_cast<int>(floorf((position.y() - (bc.y() - blockSize * 0.5f)) / vSize + 1e-5f));
		int lz = static_cast<int>(floorf((position.z() - (bc.z() - blockSize * 0.5f)) / vSize + 1e-5f));

		lx = (lx < 0) ? 0 : (lx > 7 ? 7 : lx);
		ly = (ly < 0) ? 0 : (ly > 7 ? 7 : ly);
		lz = (lz < 0) ? 0 : (lz > 7 ? 7 : lz);

		return &d_blocks[slot].voxels[(lz << 6) | (ly << 3) | lx];
	}

	template <typename T>
	__device__ T* VoxelDataBase<T>::GetOrCreateVoxel(const Eigen::Vector3f& position)
	{
		VoxelBlock<T>* block = GetOrCreateVoxelBlock(position);
		if (block == nullptr) return nullptr;

		const float vSize = blockSize / 8.0f;

		Morton64 blockKey = Morton64::FromPosition(position, blockSize);
		Eigen::Vector3f bc = blockKey.ToPosition(blockSize);

		int lx = static_cast<int>(floorf((position.x() - (bc.x() - blockSize * 0.5f)) / vSize + 1e-4f));
		int ly = static_cast<int>(floorf((position.y() - (bc.y() - blockSize * 0.5f)) / vSize + 1e-4f));
		int lz = static_cast<int>(floorf((position.z() - (bc.z() - blockSize * 0.5f)) / vSize + 1e-4f));
		lx = (lx < 0) ? 0 : (lx > 7 ? 7 : lx);
		ly = (ly < 0) ? 0 : (ly > 7 ? 7 : ly);
		lz = (lz < 0) ? 0 : (lz > 7 ? 7 : lz);

		return &(block->voxels[(lz << 6) | (ly << 3) | lx]);
	}

	template <typename T>
	__device__ inline uint32_t VoxelDataBase<T>::FindBlockSlot(const Eigen::Vector3f& position)
	{
		Morton64 blockKey = Morton64::FromPosition(position, blockSize);
		uint64_t key = blockKey.code;
		if (key == 0) key = 0xFFFFFFFFFFFFFFFFULL;

		uint32_t slot = StrongHash(key, maxBlockCount);
		uint32_t start = slot;

		while (true)
		{
			uint64_t hKey = d_hashTable[slot];
			if (hKey == key) return slot;
			if (hKey == 0) return INVALID_BLOCK;
			slot = (slot + 1) % maxBlockCount;
			if (slot == start) break;
		}
		return INVALID_BLOCK;
	}

	template <typename T>
	__device__ inline uint32_t VoxelDataBase<T>::GetOrCreateBlockSlot(const Eigen::Vector3f& position)
	{
		Morton64 blockKey = Morton64::FromPosition(position, blockSize);
		uint64_t key = blockKey.code;
		if (key == 0) key = 0xFFFFFFFFFFFFFFFFULL;

		uint32_t slot = StrongHash(key, maxBlockCount);
		uint32_t start = slot;

		while (true)
		{
			unsigned long long* slotPtr = (unsigned long long*) & d_hashTable[slot];
			unsigned long long prev = atomicCAS(slotPtr, 0ULL, (unsigned long long)key);

			if (prev == 0)
			{
				atomicAdd(d_blockCount, 1);
				return slot;
			}
			if (prev == key) return slot;

			slot = (slot + 1) % maxBlockCount;
			if (slot == start) break;
		}
		return INVALID_BLOCK;
	}

	struct VoxelDBHeader
	{
		uint32_t magic;          // 0x564F5842 ("VOXB")
		uint32_t version;        // 1
		uint32_t maxBlockCount;
		uint32_t blockCount;     // 실제 점유된 블록 수
		float    blockSize;
		uint32_t voxelStride;    // sizeof(T) 검증용
		uint32_t reserved[2];
	};

	static constexpr uint32_t VOXEL_DB_MAGIC = 0x564F5842u;
	static constexpr uint32_t VOXEL_DB_VERSION = 1u;


	template <typename T>
	void VoxelDataBase<T>::Serialize(const std::wstring& filename)
	{
		// 실제 점유 블록 수 확인
		uint32_t blockCount = 0;
		cudaMemcpy(&blockCount, d_blockCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

		// GPU -> CPU 복사
		std::vector<uint64_t>     h_hashTable(maxBlockCount);
		std::vector<VoxelBlock<T>> h_blocks(maxBlockCount);

		cudaMemcpy(h_hashTable.data(), d_hashTable,
			sizeof(uint64_t) * maxBlockCount, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_blocks.data(), d_blocks,
			sizeof(VoxelBlock<T>) * maxBlockCount, cudaMemcpyDeviceToHost);

		// 파일 쓰기
		FILE* fp = _wfopen(filename.c_str(), L"wb");
		if (nullptr == fp)
		{
			printf("[VoxelDataBase::Serialize] Failed to open file\n");
			return;
		}

		VoxelDBHeader header = {};
		header.magic = VOXEL_DB_MAGIC;
		header.version = VOXEL_DB_VERSION;
		header.maxBlockCount = maxBlockCount;
		header.blockCount = blockCount;
		header.blockSize = blockSize;
		header.voxelStride = (uint32_t)sizeof(T);

		fwrite(&header, sizeof(VoxelDBHeader), 1, fp);
		fwrite(h_hashTable.data(), sizeof(uint64_t), maxBlockCount, fp);
		fwrite(h_blocks.data(), sizeof(VoxelBlock<T>), maxBlockCount, fp);

		fclose(fp);

		printf("[VoxelDataBase::Serialize] Saved %u blocks to file\n", blockCount);
	}

	template <typename T>
	void VoxelDataBase<T>::Deserialize(const std::wstring& filename)
	{
		FILE* fp = _wfopen(filename.c_str(), L"rb");
		if (nullptr == fp)
		{
			printf("[VoxelDataBase::Deserialize] Failed to open file\n");
			return;
		}

		// 헤더 읽기 및 검증
		VoxelDBHeader header = {};
		fread(&header, sizeof(VoxelDBHeader), 1, fp);

		if (header.magic != VOXEL_DB_MAGIC)
		{
			printf("[VoxelDataBase::Deserialize] Invalid magic number\n");
			fclose(fp);
			return;
		}

		if (header.version != VOXEL_DB_VERSION)
		{
			printf("[VoxelDataBase::Deserialize] Version mismatch: file=%u, expected=%u\n",
				header.version, VOXEL_DB_VERSION);
			fclose(fp);
			return;
		}

		if (header.voxelStride != (uint32_t)sizeof(T))
		{
			printf("[VoxelDataBase::Deserialize] Voxel stride mismatch: file=%u, sizeof(T)=%u\n",
				header.voxelStride, (uint32_t)sizeof(T));
			fclose(fp);
			return;
		}

		// 현재 할당 크기가 다르면 재초기화
		if (header.maxBlockCount != maxBlockCount || header.blockSize != blockSize)
		{
			printf("[VoxelDataBase::Deserialize] Re-initializing: maxBlocks %u->%u, blockSize %.4f->%.4f\n",
				maxBlockCount, header.maxBlockCount, blockSize, header.blockSize);

			Terminate();
			Initialize(header.maxBlockCount, header.blockSize);
		}

		// CPU 버퍼에 읽기
		std::vector<uint64_t>      h_hashTable(maxBlockCount);
		std::vector<VoxelBlock<T>> h_blocks(maxBlockCount);

		fread(h_hashTable.data(), sizeof(uint64_t), maxBlockCount, fp);
		fread(h_blocks.data(), sizeof(VoxelBlock<T>), maxBlockCount, fp);

		fclose(fp);

		// CPU -> GPU 업로드
		cudaMemcpy(d_hashTable, h_hashTable.data(),
			sizeof(uint64_t) * maxBlockCount, cudaMemcpyHostToDevice);
		cudaMemcpy(d_blocks, h_blocks.data(),
			sizeof(VoxelBlock<T>) * maxBlockCount, cudaMemcpyHostToDevice);
		cudaMemcpy(d_blockCount, &header.blockCount,
			sizeof(uint32_t), cudaMemcpyHostToDevice);

		printf("[VoxelDataBase::Deserialize] Loaded %u blocks from file\n", header.blockCount);
	}

	template class VoxelDataBase<Voxel>;
}
