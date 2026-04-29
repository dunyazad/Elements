#include <Core/DataStructures/VoxelDataBase.h>
#include <Core/Common/DeviceCommon.h>

namespace Huvitz
{
	namespace Core
	{
		template <typename T>
		__global__ void Kernel_Clear(VoxelDataBase<T> db)
		{
			uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
			if (slot >= db.GetMaxBlockCount())
				return;

			uint64_t* hashTable = db.GetHashTable();
			uint64_t key = hashTable[slot];

			if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL)
				return;

			hashTable[slot] = 0;

			VoxelBlock<T>* blockPtr = &db.GetBlocks()[slot];
			*blockPtr = {};
		}

		template <typename T>
		__global__ void Kernel_Integrate(VoxelDataBase<T> db, VoxelDataBaseIntegrationParameters parameters)
		{
			uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadid >= parameters.mapWidth * parameters.mapHeight)
				return;

			uint32_t u = threadid % parameters.mapWidth;
			uint32_t v = threadid / parameters.mapWidth;

			Vector3f p_cam = parameters.d_depthMap[v * parameters.mapWidth + u];
			if (false == VECTOR3F_VALID_(p_cam))
				return;

			Vector3f n_cam = parameters.d_normalMap[v * parameters.mapWidth + u];
			unsigned int r = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 0];
			unsigned int g = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 1];
			unsigned int b = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 2];
			Vector3b color(r, g, b);

			Vector3f p_world = parameters.transform.Transform(p_cam);
			Vector3f n_world = parameters.transform.TransformNormal(n_cam);

			float voxelSize = db.GetBlockSize() / 8.0f;
			float truncDist = voxelSize * 10.0f;
			float invVoxelSize = 1.0f / voxelSize;

			Vector3f camPos = { parameters.transform(0, 3), parameters.transform(1, 3), parameters.transform(2, 3) };
			Vector3f rayDir = p_world - camPos;
			float rayLen = sqrtf(rayDir.x() * rayDir.x() + rayDir.y() * rayDir.y() + rayDir.z() * rayDir.z());

			if (rayLen < 1e-6f)
				return;

			rayDir.x() /= rayLen;
			rayDir.y() /= rayLen;
			rayDir.z() /= rayLen;

			float tStart = fmaxf(0.0f, rayLen - truncDist);
			float tEnd = rayLen + truncDist;
			float step = voxelSize * 0.8f;

			VoxelBlock<T>* cachedBlock = nullptr;
			uint64_t cachedBlockKey = 0xFFFFFFFFFFFFFFFFULL;
			Vector3f cachedBlockCenter = { 0.0f, 0.0f, 0.0f };
			float blockSize = db.GetBlockSize();
			float halfBlock = blockSize * 0.5f;

			for (float t = tStart; t <= tEnd; t += step)
			{
				Vector3f samplePos;
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
					continue;

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

		template <typename T>
		__global__ void Kernel_IntegrateSurfaceNormal(VoxelDataBase<T> db, VoxelDataBaseIntegrationParameters parameters)
		{
			uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadid >= parameters.mapWidth * parameters.mapHeight)
				return;

			uint32_t u = threadid % parameters.mapWidth;
			uint32_t v = threadid / parameters.mapWidth;

			Vector3f p_cam = parameters.d_depthMap[v * parameters.mapWidth + u];
			if (false == VECTOR3F_VALID_(p_cam))
				return;

			Vector3f n_cam = parameters.d_normalMap[v * parameters.mapWidth + u];
			if (false == VECTOR3F_VALID_(n_cam))
				return;

			Vector3f p_world = parameters.transform.Transform(p_cam);
			Vector3f n_world = parameters.transform.TransformNormal(n_cam);

			float nLen = sqrtf(n_world.x() * n_world.x() + n_world.y() * n_world.y() + n_world.z() * n_world.z());
			if (nLen < 1e-6f)
				return;
			float invNLen = 1.0f / nLen;
			n_world.x() *= invNLen;
			n_world.y() *= invNLen;
			n_world.z() *= invNLen;

			Vector3f camPos = {
				parameters.transform(0, 3),
				parameters.transform(1, 3),
				parameters.transform(2, 3)
			};

			float vvx = p_world.x() - camPos.x();
			float vvy = p_world.y() - camPos.y();
			float vvz = p_world.z() - camPos.z();
			float viewDist = sqrtf(vvx * vvx + vvy * vvy + vvz * vvz);
			if (viewDist < 1e-6f)
				return;

			float invViewDist = 1.0f / viewDist;
			float vdx = vvx * invViewDist;
			float vdy = vvy * invViewDist;
			float vdz = vvz * invViewDist;

			float cosAngle = fabsf(vdx * n_world.x() + vdy * n_world.y() + vdz * n_world.z());
			if (cosAngle < 0.1f)
				return;

			float w_depth = fminf(1.0f / fmaxf(viewDist, 1e-4f), 1.0f);

			float voxelSize = db.GetBlockSize() / 8.0f;
			float truncDist = voxelSize * 10.0f;
			float step = voxelSize * 0.8f;
			float invVoxelSize = 1.0f / voxelSize;
			float blockSize = db.GetBlockSize();
			float halfBlock = blockSize * 0.5f;

			unsigned int colorR = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 0];
			unsigned int colorG = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 1];
			unsigned int colorB = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 2];

			VoxelBlock<T>* cachedBlock = nullptr;
			uint64_t cachedBlockKey = 0xFFFFFFFFFFFFFFFFULL;
			Vector3f cachedBlockCenter = { 0.0f, 0.0f, 0.0f };

			for (float t = -truncDist; t <= truncDist; t += step)
			{
				Vector3f samplePos = {
					p_world.x() + n_world.x() * t,
					p_world.y() + n_world.y() * t,
					p_world.z() + n_world.z() * t
				};

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
					continue;

				int lx = static_cast<int>(floorf((samplePos.x() - (cachedBlockCenter.x() - halfBlock)) * invVoxelSize + 1e-4f));
				int ly = static_cast<int>(floorf((samplePos.y() - (cachedBlockCenter.y() - halfBlock)) * invVoxelSize + 1e-4f));
				int lz = static_cast<int>(floorf((samplePos.z() - (cachedBlockCenter.z() - halfBlock)) * invVoxelSize + 1e-4f));
				lx = lx < 0 ? 0 : (lx > 7 ? 7 : lx);
				ly = ly < 0 ? 0 : (ly > 7 ? 7 : ly);
				lz = lz < 0 ? 0 : (lz > 7 ? 7 : lz);

				T* voxelPtr = &(cachedBlock->voxels[(lz << 6) | (ly << 3) | lx]);

				float vcx = cachedBlockCenter.x() - halfBlock + (lx + 0.5f) * voxelSize;
				float vcy = cachedBlockCenter.y() - halfBlock + (ly + 0.5f) * voxelSize;
				float vcz = cachedBlockCenter.z() - halfBlock + (lz + 0.5f) * voxelSize;

				float sdf_raw = (p_world.x() - vcx) * n_world.x()
					+ (p_world.y() - vcy) * n_world.y()
					+ (p_world.z() - vcz) * n_world.z();

				float tsdfValue = fmaxf(-1.0f, fminf(1.0f, sdf_raw / truncDist));

				unsigned short oldWeightUS = atomicAddUShort(&(voxelPtr->valueCount), 1);
				float oldWeight = fminf((float)oldWeightUS, MAX_WEIGHT);
				float newWeight = oldWeight + 1.0f;
				float weightFactor = oldWeight / newWeight;
				float invNewWeight = 1.0f / newWeight;

				float effectiveFactor = cosAngle * w_depth;
				float blendedInvNew = invNewWeight * effectiveFactor;
				float blendedFactor = 1.0f - blendedInvNew;

				voxelPtr->value = voxelPtr->value * blendedFactor + tsdfValue * blendedInvNew;

				float bNx = voxelPtr->normal.x() * weightFactor + n_world.x() * invNewWeight;
				float bNy = voxelPtr->normal.y() * weightFactor + n_world.y() * invNewWeight;
				float bNz = voxelPtr->normal.z() * weightFactor + n_world.z() * invNewWeight;
				float nInvLen = rsqrtf(bNx * bNx + bNy * bNy + bNz * bNz + 1e-10f);
				voxelPtr->normal.x() = bNx * nInvLen;
				voxelPtr->normal.y() = bNy * nInvLen;
				voxelPtr->normal.z() = bNz * nInvLen;

				voxelPtr->color.x() = (uint8_t)((float)voxelPtr->color.x() * weightFactor + (float)colorR * invNewWeight + 0.5f);
				voxelPtr->color.y() = (uint8_t)((float)voxelPtr->color.y() * weightFactor + (float)colorG * invNewWeight + 0.5f);
				voxelPtr->color.z() = (uint8_t)((float)voxelPtr->color.z() * weightFactor + (float)colorB * invNewWeight + 0.5f);
			}
		}

		__global__ void Kernel_IntegrateDirectional_Phase1(
			VoxelDataBase<DirectionalVoxel<Voxel>> db,
			VoxelDataBaseIntegrationParameters parameters)
		{
			uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadid >= parameters.mapWidth * parameters.mapHeight)
				return;

			uint32_t u = threadid % parameters.mapWidth;
			uint32_t v = threadid / parameters.mapWidth;

			Vector3f p_cam = parameters.d_depthMap[v * parameters.mapWidth + u];
			if (false == VECTOR3F_VALID_(p_cam))
				return;

			Vector3f n_cam = parameters.d_normalMap[v * parameters.mapWidth + u];
			if (false == VECTOR3F_VALID_(n_cam))
				return;

			Vector3f p_world = parameters.transform.Transform(p_cam);
			Vector3f n_world = parameters.transform.TransformNormal(n_cam);

			float nLen = sqrtf(n_world.x() * n_world.x() + n_world.y() * n_world.y() + n_world.z() * n_world.z());
			if (nLen < 1e-6f)
				return;
			float invNLen = 1.f / nLen;
			n_world.x() *= invNLen;
			n_world.y() *= invNLen;
			n_world.z() *= invNLen;

			Vector3f camPos = {
				parameters.transform(0, 3),
				parameters.transform(1, 3),
				parameters.transform(2, 3)
			};

			float vvx = p_world.x() - camPos.x();
			float vvy = p_world.y() - camPos.y();
			float vvz = p_world.z() - camPos.z();
			float viewDist = sqrtf(vvx * vvx + vvy * vvy + vvz * vvz);
			if (viewDist < 1e-6f)
				return;
			float invVD = 1.f / viewDist;

			float cosAngle = fabsf((vvx * n_world.x() + vvy * n_world.y() + vvz * n_world.z()) * invVD);
			if (cosAngle < 0.1f)
				return;

			float w_depth = fminf(1.f / fmaxf(viewDist, 1e-4f), 1.f);
			float viewDotN = (vvx * n_world.x() + vvy * n_world.y() + vvz * n_world.z()) * invVD;

			int   applicableDirs[3];
			float dirWeights[3];
			int   numDirs = 0;

			for (int d = 0; d < DIR_COUNT; ++d)
			{
				Vector3f vD = GetDirectionVector(d);
				float dot = n_world.x() * vD.x() + n_world.y() * vD.y() + n_world.z() * vD.z();
				if (dot > kDirThreshold)
				{
					applicableDirs[numDirs] = d;
					dirWeights[numDirs] = dot;
					++numDirs;
				}
			}

			if (numDirs == 0)
				return;

			float voxelSize = db.GetBlockSize() / 8.f;
			float truncDist = voxelSize * 10.f;
			float step = voxelSize * 0.8f;
			float invVoxSize = 1.f / voxelSize;
			float blockSize = db.GetBlockSize();
			float halfBlock = blockSize * 0.5f;

			unsigned int colorR = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 0];
			unsigned int colorG = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 1];
			unsigned int colorB = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 2];

			VoxelBlock<DirectionalVoxel<Voxel>>* cachedBlock = nullptr;
			uint64_t       cachedBlockKey = 0xFFFFFFFFFFFFFFFFULL;
			Vector3f cachedCenter = { 0.f, 0.f, 0.f };

			for (float t = -truncDist; t <= truncDist; t += step)
			{
				Vector3f samplePos = {
					p_world.x() + n_world.x() * t,
					p_world.y() + n_world.y() * t,
					p_world.z() + n_world.z() * t
				};

				Morton64 bKey = Morton64::FromPosition(samplePos, blockSize);
				uint64_t key = bKey.code;
				if (key == 0) key = 0xFFFFFFFFFFFFFFFFULL;

				if (key != cachedBlockKey)
				{
					cachedBlock = db.GetOrCreateVoxelBlock(samplePos);
					cachedBlockKey = key;
					cachedCenter = bKey.ToPosition(blockSize);
				}
				if (cachedBlock == nullptr)
					continue;

				int lx = (int)floorf((samplePos.x() - (cachedCenter.x() - halfBlock)) * invVoxSize + 1e-4f);
				int ly = (int)floorf((samplePos.y() - (cachedCenter.y() - halfBlock)) * invVoxSize + 1e-4f);
				int lz = (int)floorf((samplePos.z() - (cachedCenter.z() - halfBlock)) * invVoxSize + 1e-4f);
				lx = lx < 0 ? 0 : (lx > 7 ? 7 : lx);
				ly = ly < 0 ? 0 : (ly > 7 ? 7 : ly);
				lz = lz < 0 ? 0 : (lz > 7 ? 7 : lz);

				DirectionalVoxel<Voxel>* voxelPtr = &(cachedBlock->voxels[(lz << 6) | (ly << 3) | lx]);

				float vcx = cachedCenter.x() - halfBlock + (lx + 0.5f) * voxelSize;
				float vcy = cachedCenter.y() - halfBlock + (ly + 0.5f) * voxelSize;
				float vcz = cachedCenter.z() - halfBlock + (lz + 0.5f) * voxelSize;

				float sdf_raw = (p_world.x() - vcx) * n_world.x()
					+ (p_world.y() - vcy) * n_world.y()
					+ (p_world.z() - vcz) * n_world.z();
				float sdf_norm = fmaxf(-1.f, fminf(1.f, sdf_raw / truncDist));

				if (sdf_norm < -0.5f && viewDotN > 0.f)
					continue;

				for (int i = 0; i < numDirs; ++i)
				{
					int   d = applicableDirs[i];
					float w = dirWeights[i] * cosAngle * w_depth;
					atomicAdd(&voxelPtr->accSd[d], w * sdf_norm);
					atomicAdd(&voxelPtr->accSw[d], w);
				}

				unsigned short oldCnt = atomicAddUShort(&voxelPtr->valueCount, 1);
				float oldW = fminf((float)oldCnt, 100.f);
				float newW = oldW + 1.f;
				float invNew = 1.f / newW;
				float oldFactor = oldW * invNew;

				voxelPtr->color.x() = (uint8_t)(voxelPtr->color.x() * oldFactor + colorR * invNew + 0.5f);
				voxelPtr->color.y() = (uint8_t)(voxelPtr->color.y() * oldFactor + colorG * invNew + 0.5f);
				voxelPtr->color.z() = (uint8_t)(voxelPtr->color.z() * oldFactor + colorB * invNew + 0.5f);

				voxelPtr->normal.x() = n_world.x();
				voxelPtr->normal.y() = n_world.y();
				voxelPtr->normal.z() = n_world.z();
			}
		}

		__global__ void Kernel_IntegrateDirectional_Phase2_Voxel(
			VoxelDataBase<DirectionalVoxel<Voxel>> db,
			uint32_t* occupiedSlots,
			uint32_t occupiedCount)
		{
			uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

			// 1 thread = 1 voxel
			uint32_t slotIdx = tid / VoxelBlock<DirectionalVoxel<Voxel>>::VOXELS_PER_BLOCK;
			uint32_t voxelIdx = tid % VoxelBlock<DirectionalVoxel<Voxel>>::VOXELS_PER_BLOCK;

			if (slotIdx >= occupiedCount)
				return;

			uint32_t slot = occupiedSlots[slotIdx];
			DirectionalVoxel<Voxel>& v = db.GetBlocks()[slot].voxels[voxelIdx];

			bool anyUpdated = false;

			for (int d = 0; d < DIR_COUNT; ++d)
			{
				float sw = v.accSw[d];
				if (sw < 1e-9f) continue;

				float sd = v.accSd[d];
				float W_old = v.weight[d];
				float D_old = v.HasDirection(d) ? v.dirValue[d] : 0.f;

				v.dirValue[d] = (W_old * D_old + sd) / (W_old + sw);
				v.weight[d] = W_old + sw;
				v.validMask |= (1u << d);
				v.accSd[d] = 0.f;
				v.accSw[d] = 0.f;
				anyUpdated = true;
			}

			for (int d = 0; d < DIR_COUNT; d += 2)
			{
				int dOpp = d + 1;
				if (v.HasDirection(d) && v.HasDirection(dOpp))
				{
					if (v.dirValue[d] < 0.f && v.dirValue[dOpp] < 0.f)
					{
						v.validMask &= ~(1u << d);
						v.validMask &= ~(1u << dOpp);
						v.weight[d] = 0.f;
						v.weight[dOpp] = 0.f;
					}
				}
			}

			if (anyUpdated && v.valueCount == 0)
				atomicAddUShort(&v.valueCount, 1);
		}

		__global__ void Kernel_IntegrateDirectional_Phase1(
			VoxelDataBase<DirectionalVoxel<DummyVoxel>> db,
			VoxelDataBaseIntegrationParameters parameters)
		{
			uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadid >= parameters.mapWidth * parameters.mapHeight)
				return;

			uint32_t u = threadid % parameters.mapWidth;
			uint32_t v = threadid / parameters.mapWidth;

			Vector3f p_cam = parameters.d_depthMap[v * parameters.mapWidth + u];
			if (false == VECTOR3F_VALID_(p_cam))
				return;

			Vector3f n_cam = parameters.d_normalMap[v * parameters.mapWidth + u];
			if (false == VECTOR3F_VALID_(n_cam))
				return;

			Vector3f p_world = parameters.transform.Transform(p_cam);
			Vector3f n_world = parameters.transform.TransformNormal(n_cam);

			float nLen = sqrtf(n_world.x() * n_world.x() + n_world.y() * n_world.y() + n_world.z() * n_world.z());
			if (nLen < 1e-6f)
				return;
			float invNLen = 1.f / nLen;
			n_world.x() *= invNLen;
			n_world.y() *= invNLen;
			n_world.z() *= invNLen;

			Vector3f camPos = {
				parameters.transform(0, 3),
				parameters.transform(1, 3),
				parameters.transform(2, 3)
			};

			float vvx = p_world.x() - camPos.x();
			float vvy = p_world.y() - camPos.y();
			float vvz = p_world.z() - camPos.z();
			float viewDist = sqrtf(vvx * vvx + vvy * vvy + vvz * vvz);
			if (viewDist < 1e-6f)
				return;
			float invVD = 1.f / viewDist;

			float cosAngle = fabsf((vvx * n_world.x() + vvy * n_world.y() + vvz * n_world.z()) * invVD);
			if (cosAngle < 0.1f)
				return;

			float w_depth = fminf(1.f / fmaxf(viewDist, 1e-4f), 1.f);
			float viewDotN = (vvx * n_world.x() + vvy * n_world.y() + vvz * n_world.z()) * invVD;

			int   applicableDirs[3];
			float dirWeights[3];
			int   numDirs = 0;

			for (int d = 0; d < DIR_COUNT; ++d)
			{
				Vector3f vD = GetDirectionVector(d);
				float dot = n_world.x() * vD.x() + n_world.y() * vD.y() + n_world.z() * vD.z();
				if (dot > kDirThreshold)
				{
					applicableDirs[numDirs] = d;
					dirWeights[numDirs] = dot;
					++numDirs;
				}
			}

			if (numDirs == 0)
				return;

			float voxelSize = db.GetBlockSize() / 8.f;
			float truncDist = voxelSize * 5.0f;
			float step = voxelSize * 0.8f;
			float invVoxSize = 1.f / voxelSize;
			float blockSize = db.GetBlockSize();
			float halfBlock = blockSize * 0.5f;

			float maxLateralDist2 = (voxelSize * 0.8f) * (voxelSize * 0.8f);

			unsigned int colorR = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 0];
			unsigned int colorG = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 1];
			unsigned int colorB = parameters.d_colorMap[(v * parameters.mapWidth + u) * 3 + 2];

			VoxelBlock<DirectionalVoxel<DummyVoxel>>* cachedBlock = nullptr;
			uint64_t cachedBlockKey = 0xFFFFFFFFFFFFFFFFULL;
			Vector3f cachedCenter = { 0.f, 0.f, 0.f };

			for (float t = -truncDist; t <= truncDist; t += step)
			{
				Vector3f samplePos = {
					p_world.x() + n_world.x() * t,
					p_world.y() + n_world.y() * t,
					p_world.z() + n_world.z() * t
				};

				Morton64 bKey = Morton64::FromPosition(samplePos, blockSize);
				uint64_t key = bKey.code;
				if (key == 0) key = 0xFFFFFFFFFFFFFFFFULL;

				if (key != cachedBlockKey)
				{
					cachedBlock = db.GetOrCreateVoxelBlock(samplePos);
					cachedBlockKey = key;
					cachedCenter = bKey.ToPosition(blockSize);
				}
				if (cachedBlock == nullptr)
					continue;

				int lx = (int)floorf((samplePos.x() - (cachedCenter.x() - halfBlock)) * invVoxSize + 1e-4f);
				int ly = (int)floorf((samplePos.y() - (cachedCenter.y() - halfBlock)) * invVoxSize + 1e-4f);
				int lz = (int)floorf((samplePos.z() - (cachedCenter.z() - halfBlock)) * invVoxSize + 1e-4f);
				lx = lx < 0 ? 0 : (lx > 7 ? 7 : lx);
				ly = ly < 0 ? 0 : (ly > 7 ? 7 : ly);
				lz = lz < 0 ? 0 : (lz > 7 ? 7 : lz);

				float vcx = cachedCenter.x() - halfBlock + (lx + 0.5f) * voxelSize;
				float vcy = cachedCenter.y() - halfBlock + (ly + 0.5f) * voxelSize;
				float vcz = cachedCenter.z() - halfBlock + (lz + 0.5f) * voxelSize;

				float dx = vcx - p_world.x();
				float dy = vcy - p_world.y();
				float dz = vcz - p_world.z();

				float dotN = dx * n_world.x() + dy * n_world.y() + dz * n_world.z();
				float latX = dx - dotN * n_world.x();
				float latY = dy - dotN * n_world.y();
				float latZ = dz - dotN * n_world.z();
				float lateralDist2 = latX * latX + latY * latY + latZ * latZ;

				if (lateralDist2 > maxLateralDist2)
					continue;

				DirectionalVoxel<DummyVoxel>* voxelPtr = &(cachedBlock->voxels[(lz << 6) | (ly << 3) | lx]);

				float sdf_raw = (p_world.x() - vcx) * n_world.x()
					+ (p_world.y() - vcy) * n_world.y()
					+ (p_world.z() - vcz) * n_world.z();
				float sdf_norm = fmaxf(-1.f, fminf(1.f, sdf_raw / truncDist));

				if (sdf_norm < -0.5f && viewDotN > 0.f)
					continue;

				for (int i = 0; i < numDirs; ++i)
				{
					int   d = applicableDirs[i];
					float w = dirWeights[i] * cosAngle * w_depth;
					atomicAdd(&voxelPtr->accSd[d], w * sdf_norm);
					atomicAdd(&voxelPtr->accSw[d], w);
				}

				unsigned short oldCnt = atomicAddUShort(&voxelPtr->valueCount, 1);
				float oldW = fminf((float)oldCnt, 100.f);
				float newW = oldW + 1.f;
				float invNew = 1.f / newW;
				float oldFactor = oldW * invNew;

				voxelPtr->color.x() = (uint8_t)(voxelPtr->color.x() * oldFactor + colorR * invNew + 0.5f);
				voxelPtr->color.y() = (uint8_t)(voxelPtr->color.y() * oldFactor + colorG * invNew + 0.5f);
				voxelPtr->color.z() = (uint8_t)(voxelPtr->color.z() * oldFactor + colorB * invNew + 0.5f);

				voxelPtr->normal.x() = n_world.x();
				voxelPtr->normal.y() = n_world.y();
				voxelPtr->normal.z() = n_world.z();
			}
		}

		__global__ void Kernel_IntegrateDirectional_Phase2_DummyVoxel(
			VoxelDataBase<DirectionalVoxel<DummyVoxel>> db,
			uint32_t* occupiedSlots,
			uint32_t occupiedCount)
		{
			uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

			uint32_t slotIdx = tid / VoxelBlock<DirectionalVoxel<DummyVoxel>>::VOXELS_PER_BLOCK;
			uint32_t voxelIdx = tid % VoxelBlock<DirectionalVoxel<DummyVoxel>>::VOXELS_PER_BLOCK;

			if (slotIdx >= occupiedCount)
				return;

			uint32_t slot = occupiedSlots[slotIdx];
			DirectionalVoxel<DummyVoxel>& v = db.GetBlocks()[slot].voxels[voxelIdx];

			bool anyUpdated = false;

			for (int d = 0; d < DIR_COUNT; ++d)
			{
				float sw = v.accSw[d];
				if (sw < 1e-9f) continue;

				float sd = v.accSd[d];
				float W_old = v.weight[d];
				float D_old = v.HasDirection(d) ? v.dirValue[d] : 0.f;

				v.dirValue[d] = (W_old * D_old + sd) / (W_old + sw);
				v.weight[d] = W_old + sw;
				v.validMask |= (1u << d);
				v.accSd[d] = 0.f;
				v.accSw[d] = 0.f;
				anyUpdated = true;
			}

			for (int d = 0; d < DIR_COUNT; d += 2)
			{
				int dOpp = d + 1;
				if (v.HasDirection(d) && v.HasDirection(dOpp))
				{
					if (v.dirValue[d] < 0.f && v.dirValue[dOpp] < 0.f)
					{
						v.validMask &= ~(1u << d);
						v.validMask &= ~(1u << dOpp);
						v.weight[d] = 0.f;
						v.weight[dOpp] = 0.f;
					}
				}
			}

			if (anyUpdated && v.valueCount == 0)
				atomicAddUShort(&v.valueCount, 1);
		}

		template <typename T>
		__global__ void Kernel_BuildNoiseFilter2DCache(NoiseFilterParameters nf)
		{
			uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadid >= nf.mapWidth * nf.mapHeight)
				return;

			Vector3f p_cam = nf.d_depthMap[threadid];
			if (false == VECTOR3F_VALID_(p_cam) || p_cam.z() <= 0.0f)
				return;

			Vector3f p_world = nf.transform.Transform(p_cam);

			int ix = (int)floorf((p_world.x() - nf.originX) / nf.cellSize);
			int iy = (int)floorf((p_world.y() - nf.originY) / nf.cellSize);

			if (ix < 0 || iy < 0 || ix >= (int)nf.dimX || iy >= (int)nf.dimY)
				return;

			atomicMinFloat(&nf.d_depthCache[(uint32_t)iy * nf.dimX + (uint32_t)ix], p_world.z());
		}

		template <typename T>
		__global__ void InsertKernel(VoxelDataBase<T> db, Matrix4f rt, const Vector3f* points, const Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
		{
			uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
			if (idx >= count)
				return;

			Vector3f p = rt.Transform(points[idx]);
			T* v = db.GetOrCreateVoxel(p);

			if (v != nullptr)
			{
				atomicAdd(&(v->value), 1.0f);

				if (v->valueCount < 65535)
					atomicAddUShort(&(v->valueCount), 1);

				v->color = colors[idx];

				VoxelBlock<T>* block = db.GetVoxelBlock(p);
				if (nullptr != block)
					block->lastTouchedFrameId = frameId;
			}
		}

		template <typename T>
		__global__ void Kernel_ExtractAllOccupied(
			VoxelDataBase<T> db, float blockSize,
			ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
		{
			uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
			if (slot >= db.GetMaxBlockCount())
				return;

			uint64_t key = db.GetHashTable()[slot];
			if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL)
				return;

			Morton64 bKey(key);
			Vector3f bc = bKey.ToPosition(blockSize);
			float vSize = blockSize / 8.0f;
			float halfBlock = blockSize * 0.5f;

			VoxelBlock<T>& block = db.GetBlocks()[slot];

			for (int i = 0; i < 512; ++i)
			{
				T& v = block.voxels[i];
				if (v.valueCount == 0)
					continue;

				uint32_t idx = atomicAdd(count, 1);
				if (idx >= maxOut)
					return;

				int lz = i / 64;
				int ly = (i % 64) / 8;
				int lx = i % 8;

				out[idx].position.x() = (bc.x() - halfBlock) + (lx + 0.5f) * vSize;
				out[idx].position.y() = (bc.y() - halfBlock) + (ly + 0.5f) * vSize;
				out[idx].position.z() = (bc.z() - halfBlock) + (lz + 0.5f) * vSize;
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

			VoxelBlock<T>* block = &db.GetBlocks()[slot];
			Vector3f       bc = db.GetBlockCenter(block);
			Vector3f       origin = {
				bc.x() - blockSize * 0.5f,
				bc.y() - blockSize * 0.5f,
				bc.z() - blockSize * 0.5f
			};
			float voxelSize = blockSize / 8.0f;
			float eps = voxelSize;
			int   neighborStride[3] = { 1, 8, 64 };

			VoxelBlock<T>* cachedNeighborBlock = nullptr;
			uint64_t       cachedNeighborBlockKey = 0xFFFFFFFFFFFFFFFFULL;

			// Block-cached voxel value sampler
			auto sampleValue = [&](const Vector3f& p) -> float
				{
					Morton64 nKey = Morton64::FromPosition(p, blockSize);
					uint64_t nKeyCode = nKey.code;
					if (nKeyCode == 0) nKeyCode = 0xFFFFFFFFFFFFFFFFULL;

					VoxelBlock<T>* targetBlock = nullptr;
					if (nKeyCode == Morton64::FromPosition(bc, blockSize).code)
					{
						targetBlock = block;
					}
					else
					{
						if (nKeyCode != cachedNeighborBlockKey)
						{
							cachedNeighborBlock = db.GetVoxelBlock(p);
							cachedNeighborBlockKey = nKeyCode;
						}
						targetBlock = cachedNeighborBlock;
					}

					if (targetBlock == nullptr) return FLT_MAX;
					T* v = db.GetVoxel(targetBlock, p);
					return (v != nullptr && v->valueCount > 0) ? v->value : FLT_MAX;
				};

			for (int lz = 0; lz < 8; ++lz)
			{
				for (int ly = 0; ly < 8; ++ly)
				{
					for (int lx = 0; lx < 8; ++lx)
					{
						int idx0 = (lz << 6) | (ly << 3) | lx;
						T& v0 = block->voxels[idx0];
						if (v0.valueCount < 1) continue;

						Vector3f p0 = {
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

							Vector3f p1 = p0;
							if (axis == 0) p1.x() += voxelSize;
							else if (axis == 1) p1.y() += voxelSize;
							else                p1.z() += voxelSize;

							T* v1 = nullptr;
							if (isInternal)
							{
								v1 = &block->voxels[idx0 + neighborStride[axis]];
							}
							else
							{
								Morton64 nKey = Morton64::FromPosition(p1, blockSize);
								uint64_t nKeyCode = nKey.code;
								if (nKeyCode == 0) nKeyCode = 0xFFFFFFFFFFFFFFFFULL;

								if (nKeyCode != cachedNeighborBlockKey)
								{
									cachedNeighborBlock = db.GetVoxelBlock(p1);
									cachedNeighborBlockKey = nKeyCode;
								}

								if (cachedNeighborBlock != nullptr)
									v1 = db.GetVoxel(cachedNeighborBlock, p1);
							}

							if (v1 == nullptr || v1->valueCount < 1) continue;

							float f0 = v0.value;
							float f1 = v1->value;
							if (f0 * f1 >= 0.0f) continue;

							uint32_t outIdx = atomicAdd(count, 1);
							if (outIdx >= maxOut) return;

							float    mu = fminf(fmaxf(-f0 / (f1 - f0), 0.0f), 1.0f);
							Vector3f interpPos = {
								p0.x() + mu * (p1.x() - p0.x()),
								p0.y() + mu * (p1.y() - p0.y()),
								p0.z() + mu * (p1.z() - p0.z())
							};
							out[outIdx].position = interpPos;
							out[outIdx].weight = (float)v0.valueCount;

							float vxp = sampleValue(Vector3f{ interpPos.x() + eps, interpPos.y(),       interpPos.z() });
							float vxm = sampleValue(Vector3f{ interpPos.x() - eps, interpPos.y(),       interpPos.z() });
							float vyp = sampleValue(Vector3f{ interpPos.x(),       interpPos.y() + eps, interpPos.z() });
							float vym = sampleValue(Vector3f{ interpPos.x(),       interpPos.y() - eps, interpPos.z() });
							float vzp = sampleValue(Vector3f{ interpPos.x(),       interpPos.y(),       interpPos.z() + eps });
							float vzm = sampleValue(Vector3f{ interpPos.x(),       interpPos.y(),       interpPos.z() - eps });

							bool hasGrad = (vxp < FLT_MAX * 0.5f) && (vxm < FLT_MAX * 0.5f) &&
								(vyp < FLT_MAX * 0.5f) && (vym < FLT_MAX * 0.5f) &&
								(vzp < FLT_MAX * 0.5f) && (vzm < FLT_MAX * 0.5f);

							if (hasGrad)
							{
								Vector3f grad = { vxp - vxm, vyp - vym, vzp - vzm };
								float    len = sqrtf(grad.x() * grad.x() + grad.y() * grad.y() + grad.z() * grad.z());
								out[outIdx].normal = (len > 1e-6f)
									? Vector3f(-grad.x() / len, -grad.y() / len, -grad.z() / len)
									: Vector3f(0.0f, 1.0f, 0.0f);
							}
							else
							{
								Vector3f blended = {
									v0.normal.x() + mu * (v1->normal.x() - v0.normal.x()),
									v0.normal.y() + mu * (v1->normal.y() - v0.normal.y()),
									v0.normal.z() + mu * (v1->normal.z() - v0.normal.z())
								};
								float len = sqrtf(blended.x() * blended.x() + blended.y() * blended.y() + blended.z() * blended.z());
								out[outIdx].normal = (len > 1e-6f)
									? Vector3f(blended.x() / len, blended.y() / len, blended.z() / len)
									: v0.normal;
							}

							out[outIdx].color[0] = (uint8_t)(v0.color.x() + mu * (float(v1->color.x()) - v0.color.x()));
							out[outIdx].color[1] = (uint8_t)(v0.color.y() + mu * (float(v1->color.y()) - v0.color.y()));
							out[outIdx].color[2] = (uint8_t)(v0.color.z() + mu * (float(v1->color.z()) - v0.color.z()));
						}
					}
				}
			}
		}

		template <typename T>
		__global__ void Kernel_ExtractIntraRegion(
			VoxelDataBase<T> db, float blockSize,
			VoxelDataBaseExtractionParameters::Mode mode,
			Vector3f aabbMin, Vector3f aabbMax,
			ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
		{
			uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
			if (slot >= db.GetMaxBlockCount()) return;

			uint64_t key = db.GetHashTable()[slot];
			if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL) return;

			VoxelBlock<T>* block = &db.GetBlocks()[slot];
			Vector3f       bc = db.GetBlockCenter(block);

			Vector3f bMin = { bc.x() - blockSize * 0.5f, bc.y() - blockSize * 0.5f, bc.z() - blockSize * 0.5f };
			Vector3f bMax = { bc.x() + blockSize * 0.5f, bc.y() + blockSize * 0.5f, bc.z() + blockSize * 0.5f };
			if (bMax.x() < aabbMin.x() || bMin.x() > aabbMax.x()) return;
			if (bMax.y() < aabbMin.y() || bMin.y() > aabbMax.y()) return;
			if (bMax.z() < aabbMin.z() || bMin.z() > aabbMax.z()) return;

			Vector3f origin = bMin;
			float    voxelSize = blockSize / 8.0f;

			VoxelBlock<T>* cachedNeighborBlock = nullptr;
			uint64_t       cachedNeighborBlockKey = 0xFFFFFFFFFFFFFFFFULL;

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
						T& vn = block->voxels[(lz_ << 6) | (ly_ << 3) | lx_];
						return (vn.valueCount > 0) ? vn.value : FLT_MAX;
					}

					Vector3f p = { sx, sy, sz };
					Morton64 nKey = Morton64::FromPosition(p, blockSize);
					uint64_t nKeyCode = nKey.code;
					if (nKeyCode == 0) nKeyCode = 0xFFFFFFFFFFFFFFFFULL;

					if (nKeyCode != cachedNeighborBlockKey)
					{
						cachedNeighborBlock = db.GetVoxelBlock(p);
						cachedNeighborBlockKey = nKeyCode;
					}

					if (cachedNeighborBlock == nullptr) return FLT_MAX;
					T* vn = db.GetVoxel(cachedNeighborBlock, p);
					return (vn != nullptr && vn->valueCount > 0) ? vn->value : FLT_MAX;
				};

			if (mode == VoxelDataBaseExtractionParameters::Mode::AllOccupied)
			{
				for (int i = 0; i < 512; ++i)
				{
					T& v = block->voxels[i];
					if (v.valueCount == 0) continue;

					int lz = i / 64, ly = (i % 64) / 8, lx = i % 8;
					Vector3f pos = {
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
				int   neighborStride[3] = { 1, 8, 64 };

				for (int lz = 0; lz < 8; ++lz)
				{
					for (int ly = 0; ly < 8; ++ly)
					{
						for (int lx = 0; lx < 8; ++lx)
						{
							int idx0 = (lz << 6) | (ly << 3) | lx;
							T& v0 = block->voxels[idx0];
							if (v0.valueCount < 5) continue;

							Vector3f p0 = {
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

								Vector3f p1 = p0;
								if (axis == 0) p1.x() += voxelSize;
								else if (axis == 1) p1.y() += voxelSize;
								else                p1.z() += voxelSize;

								T* v1 = nullptr;
								if (isInternal)
								{
									v1 = &block->voxels[idx0 + neighborStride[axis]];
								}
								else
								{
									Morton64 nKey = Morton64::FromPosition(p1, blockSize);
									uint64_t nKeyCode = nKey.code;
									if (nKeyCode == 0) nKeyCode = 0xFFFFFFFFFFFFFFFFULL;

									if (nKeyCode != cachedNeighborBlockKey)
									{
										cachedNeighborBlock = db.GetVoxelBlock(p1);
										cachedNeighborBlockKey = nKeyCode;
									}

									if (cachedNeighborBlock != nullptr)
										v1 = db.GetVoxel(cachedNeighborBlock, p1);
								}

								if (v1 == nullptr || v1->valueCount < 5) continue;

								float f0 = v0.value, f1 = v1->value;
								if (f0 * f1 >= 0.0f) continue;

								float    mu = fminf(fmaxf(-f0 / (f1 - f0), 0.0f), 1.0f);
								Vector3f interpPos = {
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
									Vector3f grad = { vxp - vxm, vyp - vym, vzp - vzm };
									float    len = sqrtf(grad.x() * grad.x() + grad.y() * grad.y() + grad.z() * grad.z());
									out[outIdx].normal = (len > 1e-6f)
										? Vector3f(-grad.x() / len, -grad.y() / len, -grad.z() / len)
										: Vector3f(0.0f, 1.0f, 0.0f);
								}
								else
								{
									Vector3f blended = {
										v0.normal.x() + mu * (v1->normal.x() - v0.normal.x()),
										v0.normal.y() + mu * (v1->normal.y() - v0.normal.y()),
										v0.normal.z() + mu * (v1->normal.z() - v0.normal.z())
									};
									float len = sqrtf(blended.x() * blended.x() + blended.y() * blended.y() + blended.z() * blended.z());
									out[outIdx].normal = (len > 1e-6f)
										? Vector3f(blended.x() / len, blended.y() / len, blended.z() / len)
										: v0.normal;
								}

								out[outIdx].color[0] = (uint8_t)(v0.color.x() + mu * (float(v1->color.x()) - v0.color.x()));
								out[outIdx].color[1] = (uint8_t)(v0.color.y() + mu * (float(v1->color.y()) - v0.color.y()));
								out[outIdx].color[2] = (uint8_t)(v0.color.z() + mu * (float(v1->color.z()) - v0.color.z()));
							}
						}
					}
				}
			}
		}

		__global__ void Kernel_ExtractZeroCrossing_Directional(
			VoxelDataBase<DirectionalVoxel<DummyVoxel>> db, float blockSize,
			ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
		{
			uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
			if (slot >= db.GetMaxBlockCount()) return;

			uint64_t key = db.GetHashTable()[slot];
			if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL) return;

			VoxelBlock<DirectionalVoxel<DummyVoxel>>* block = &db.GetBlocks()[slot];
			Vector3f bc = db.GetBlockCenter(block);
			Vector3f origin = {
				bc.x() - blockSize * 0.5f,
				bc.y() - blockSize * 0.5f,
				bc.z() - blockSize * 0.5f
			};
			float voxelSize = blockSize / 8.0f;
			int neighborStride[3] = { 1, 8, 64 };

			// Neighbor block cache
			VoxelBlock<DirectionalVoxel<DummyVoxel>>* cachedNeighborBlock = nullptr;
			uint64_t                                  cachedNeighborBlockKey = 0xFFFFFFFFFFFFFFFFULL;

			auto getScalarSDF = [](const DirectionalVoxel<DummyVoxel>& v) -> float {
				float totalW = 0.f, totalVal = 0.f;
				for (int d = 0; d < DIR_COUNT; ++d)
				{
					if (v.HasDirection(d) && v.weight[d] > 0.f)
					{
						totalVal += v.weight[d] * v.dirValue[d];
						totalW += v.weight[d];
					}
				}
				return (totalW > 1e-9f) ? (totalVal / totalW) : FLT_MAX;
				};

			auto getMaxWeight = [](const DirectionalVoxel<DummyVoxel>& v) -> float {
				float maxW = 0.f;
				for (int d = 0; d < DIR_COUNT; ++d)
					if (v.HasDirection(d) && v.weight[d] > maxW)
						maxW = v.weight[d];
				return maxW;
				};

			const float MIN_WEIGHT = 0.0f;

			for (int lz = 0; lz < 8; ++lz)
			{
				for (int ly = 0; ly < 8; ++ly)
				{
					for (int lx = 0; lx < 8; ++lx)
					{
						int idx0 = (lz << 6) | (ly << 3) | lx;
						DirectionalVoxel<DummyVoxel>& v0 = block->voxels[idx0];
						if (v0.validMask == 0) continue;
						if (getMaxWeight(v0) < MIN_WEIGHT) continue;

						float f0 = getScalarSDF(v0);
						if (f0 == FLT_MAX) continue;

						Vector3f p0 = {
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

							Vector3f p1 = p0;
							if (axis == 0) p1.x() += voxelSize;
							else if (axis == 1) p1.y() += voxelSize;
							else                p1.z() += voxelSize;

							DirectionalVoxel<DummyVoxel>* v1 = nullptr;
							if (isInternal)
							{
								v1 = &block->voxels[idx0 + neighborStride[axis]];
							}
							else
							{
								Morton64 nKey = Morton64::FromPosition(p1, blockSize);
								uint64_t nKeyCode = nKey.code;
								if (nKeyCode == 0) nKeyCode = 0xFFFFFFFFFFFFFFFFULL;

								if (nKeyCode != cachedNeighborBlockKey)
								{
									cachedNeighborBlock = db.GetVoxelBlock(p1);
									cachedNeighborBlockKey = nKeyCode;
								}

								if (cachedNeighborBlock != nullptr)
									v1 = db.GetVoxel(cachedNeighborBlock, p1);
							}

							if (v1 == nullptr || v1->validMask == 0) continue;
							if (getMaxWeight(*v1) < MIN_WEIGHT) continue;

							float f1 = getScalarSDF(*v1);
							if (f1 == FLT_MAX) continue;
							if (f0 * f1 >= 0.0f) continue;

							uint32_t outIdx = atomicAdd(count, 1);
							if (outIdx >= maxOut) return;

							float mu = fminf(fmaxf(-f0 / (f1 - f0), 0.0f), 1.0f);
							out[outIdx].position = {
								p0.x() + mu * (p1.x() - p0.x()),
								p0.y() + mu * (p1.y() - p0.y()),
								p0.z() + mu * (p1.z() - p0.z())
							};
							out[outIdx].weight = getMaxWeight(v0);
							out[outIdx].normal = v0.normal;
							out[outIdx].color[0] = (uint8_t)(v0.color.x() + mu * (float(v1->color.x()) - v0.color.x()));
							out[outIdx].color[1] = (uint8_t)(v0.color.y() + mu * (float(v1->color.y()) - v0.color.y()));
							out[outIdx].color[2] = (uint8_t)(v0.color.z() + mu * (float(v1->color.z()) - v0.color.z()));
						}
					}
				}
			}
		}

		__global__ void Kernel_ExtractSurface_Directional(
			VoxelDataBase<DirectionalVoxel<DummyVoxel>> db, float blockSize,
			ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
		{
			uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
			if (slot >= db.GetMaxBlockCount()) return;

			uint64_t key = db.GetHashTable()[slot];
			if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL) return;

			Morton64 bKey(key);
			Vector3f bc = bKey.ToPosition(blockSize);
			float vSize = blockSize / 8.0f;
			float halfBlock = blockSize * 0.5f;
			float surfaceThresh = 0.1f;
			float minWeight = 5.0f;

			VoxelBlock<DirectionalVoxel<DummyVoxel>>& block = db.GetBlocks()[slot];

			for (int i = 0; i < 512; ++i)
			{
				DirectionalVoxel<DummyVoxel>& v = block.voxels[i];
				if (v.validMask == 0) continue;

				bool  isSurface = false;
				float bestWeight = 0.f;
				for (int d = 0; d < DIR_COUNT; ++d)
				{
					if (!v.HasDirection(d)) continue;
					if (v.weight[d] < minWeight) continue;
					if (fabsf(v.dirValue[d]) < surfaceThresh)
					{
						isSurface = true;
						if (v.weight[d] > bestWeight)
							bestWeight = v.weight[d];
					}
				}
				if (!isSurface) continue;

				uint32_t idx = atomicAdd(count, 1);
				if (idx >= maxOut) return;

				int lz = i / 64, ly = (i % 64) / 8, lx = i % 8;
				out[idx].position.x() = (bc.x() - halfBlock) + (lx + 0.5f) * vSize;
				out[idx].position.y() = (bc.y() - halfBlock) + (ly + 0.5f) * vSize;
				out[idx].position.z() = (bc.z() - halfBlock) + (lz + 0.5f) * vSize;
				out[idx].normal = v.normal;
				out[idx].weight = bestWeight;
				out[idx].color[0] = v.color.x();
				out[idx].color[1] = v.color.y();
				out[idx].color[2] = v.color.z();
			}
		}

		template <typename T>
		__global__ void Kernel_PerFrameFilter(
			VoxelDataBase<T> db,
			VoxelDataBaseIntegrationParameters parameters)
		{
			uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;
			if (threadid >= parameters.mapWidth * parameters.mapHeight)
				return;

			uint32_t u = threadid % parameters.mapWidth;
			uint32_t v = threadid / parameters.mapWidth;

			Vector3f p_cam = parameters.d_depthMap[v * parameters.mapWidth + u];
			if (false == VECTOR3F_VALID_(p_cam))
				return;

			Vector3f p_world = parameters.transform.Transform(p_cam);

			Vector3f camPos = {
				parameters.transform(0, 3),
				parameters.transform(1, 3),
				parameters.transform(2, 3)
			};

			float dx = p_world.x() - camPos.x();
			float dy = p_world.y() - camPos.y();
			float dz = p_world.z() - camPos.z();
			float rayLen = sqrtf(dx * dx + dy * dy + dz * dz);
			if (rayLen < 1e-6f)
				return;

			float invRayLen = 1.f / rayLen;
			float rdx = dx * invRayLen;
			float rdy = dy * invRayLen;
			float rdz = dz * invRayLen;

			float voxelSize = db.GetBlockSize() / 8.f;
			// Tighter guard zone: only protect voxels very close to the actual surface
			float truncDist = voxelSize * 1.5f;
			float step = voxelSize * 0.8f;
			float blockSize = db.GetBlockSize();
			float halfBlock = blockSize * 0.5f;
			float invVoxSize = 1.f / voxelSize;

			float tEnd = rayLen - truncDist;
			if (tEnd <= step)
				return;

			// Weak voxels in free-space are likely noise: remove if below this count
			// Strong voxels are confirmed surface from multiple frames: preserve
			static constexpr uint16_t CARVE_THRESHOLD = 4;

			VoxelBlock<T>* cachedBlock = nullptr;
			uint64_t       cachedBlockKey = 0xFFFFFFFFFFFFFFFFULL;
			Vector3f       cachedCenter = { 0.f, 0.f, 0.f };

			for (float t = step; t < tEnd; t += step)
			{
				Vector3f samplePos = {
					camPos.x() + rdx * t,
					camPos.y() + rdy * t,
					camPos.z() + rdz * t
				};

				Morton64 bKey = Morton64::FromPosition(samplePos, blockSize);
				uint64_t key = bKey.code;
				if (key == 0) key = 0xFFFFFFFFFFFFFFFFULL;

				if (key != cachedBlockKey)
				{
					cachedBlock = db.GetVoxelBlock(samplePos); // read-only: no new block creation
					cachedBlockKey = key;
					cachedCenter = bKey.ToPosition(blockSize);
				}

				if (cachedBlock == nullptr)
					continue;

				int lx = (int)floorf((samplePos.x() - (cachedCenter.x() - halfBlock)) * invVoxSize + 1e-4f);
				int ly = (int)floorf((samplePos.y() - (cachedCenter.y() - halfBlock)) * invVoxSize + 1e-4f);
				int lz = (int)floorf((samplePos.z() - (cachedCenter.z() - halfBlock)) * invVoxSize + 1e-4f);
				lx = lx < 0 ? 0 : (lx > 7 ? 7 : lx);
				ly = ly < 0 ? 0 : (ly > 7 ? 7 : ly);
				lz = lz < 0 ? 0 : (lz > 7 ? 7 : lz);

				T* voxelPtr = &(cachedBlock->voxels[(lz << 6) | (ly << 3) | lx]);
				if (voxelPtr->valueCount == 0)
					continue;

				// Only carve weak voxels: strong voxels were confirmed from many frames
				if (voxelPtr->valueCount < CARVE_THRESHOLD)
					*voxelPtr = {};
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
				return;

			uint64_t key = db.GetHashTable()[slot];
			if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL)
				return;

			VoxelBlock<T>& block = db.GetBlocks()[slot];

			Morton64 bKey(key);
			float bSize = db.GetBlockSize();
			float vSize = bSize / 8.0f;
			Vector3f blockOrigin = {
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

				int lz = voxelIdx / 64, ly = (voxelIdx % 64) / 8, lx = voxelIdx % 8;
				positions[outIdx] = {
					blockOrigin.x() + (lx + 0.5f) * vSize,
					blockOrigin.y() + (ly + 0.5f) * vSize,
					blockOrigin.z() + (lz + 0.5f) * vSize
				};
				normals[outIdx] = { voxel.normal.x(), voxel.normal.y(), voxel.normal.z() };
				colors[outIdx] = {
					(unsigned char)voxel.color.x(),
					(unsigned char)voxel.color.y(),
					(unsigned char)voxel.color.z()
				};
			}
		}

		__global__ void Kernel_ComputeAABB(
			const Vector3f* d_depthMap,
			uint32_t        pixelCount,
			Matrix4f        transform,
			cuAABB* d_aabb)
		{
			__shared__ float s_min[3];
			__shared__ float s_max[3];

			if (threadIdx.x == 0)
			{
				s_min[0] = s_min[1] = s_min[2] = FLT_MAX;
				s_max[0] = s_max[1] = s_max[2] = -FLT_MAX;
			}
			__syncthreads();

			uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid < pixelCount)
			{
				Vector3f p_cam = d_depthMap[tid];
				if (VECTOR3F_VALID_(p_cam))
				{
					Vector3f p = transform.Transform(p_cam);

					// Intra-block reduction via atomicCAS on shared memory
					// shared memory atomicCAS is cheaper than global
					atomicMinFloatCAS(&s_min[0], p.x());
					atomicMinFloatCAS(&s_min[1], p.y());
					atomicMinFloatCAS(&s_min[2], p.z());
					atomicMaxFloatCAS(&s_max[0], p.x());
					atomicMaxFloatCAS(&s_max[1], p.y());
					atomicMaxFloatCAS(&s_max[2], p.z());
				}
			}
			__syncthreads();

			// One thread per block writes block result to global
			// -> global atomic contention = num_blocks not num_pixels
			if (threadIdx.x == 0)
			{
				atomicMinFloatCAS(&d_aabb->min.x, s_min[0]);
				atomicMinFloatCAS(&d_aabb->min.y, s_min[1]);
				atomicMinFloatCAS(&d_aabb->min.z, s_min[2]);
				atomicMaxFloatCAS(&d_aabb->max.x, s_max[0]);
				atomicMaxFloatCAS(&d_aabb->max.y, s_max[1]);
				atomicMaxFloatCAS(&d_aabb->max.z, s_max[2]);
			}
		}

		template <typename T>
		bool VoxelDataBase<T>::Initialize(uint32_t maxBlocks, float blockSize)
		{
			maxBlockCount = maxBlocks;
			this->blockSize = blockSize;

			cudaMalloc(&d_blocks, sizeof(VoxelBlock<T>) * maxBlocks);
			cudaMalloc(&d_hashTable, sizeof(uint64_t) * maxBlocks);
			cudaMalloc(&d_blockCount, sizeof(uint32_t));
			cudaMalloc(&d_occupiedSlots, sizeof(uint32_t) * maxBlocks);
			cudaMalloc(&d_dirtySlots, sizeof(uint32_t) * maxBlocks);
			cudaMalloc(&d_dirtyCount, sizeof(uint32_t));

			cudaMemset(d_blocks, 0, sizeof(VoxelBlock<T>) * maxBlocks);
			cudaMemset(d_hashTable, 0, sizeof(uint64_t) * maxBlocks);
			cudaMemset(d_blockCount, 0, sizeof(uint32_t));
			cudaMemset(d_occupiedSlots, 0, sizeof(uint32_t) * maxBlocks);
			cudaMemset(d_dirtySlots, 0, sizeof(uint32_t) * maxBlocks);
			cudaMemset(d_dirtyCount, 0, sizeof(uint32_t));

			return true;
		}

		template <typename T>
		void VoxelDataBase<T>::Terminate()
		{
			if (d_blocks)        cudaFree(d_blocks);
			if (d_hashTable)     cudaFree(d_hashTable);
			if (d_blockCount)    cudaFree(d_blockCount);
			if (d_occupiedSlots) cudaFree(d_occupiedSlots);
			if (d_dirtySlots)    cudaFree(d_dirtySlots);
			if (d_dirtyCount)    cudaFree(d_dirtyCount);

			d_blocks = nullptr;
			d_hashTable = nullptr;
			d_blockCount = nullptr;
			d_occupiedSlots = nullptr;
			d_dirtySlots = nullptr;
			d_dirtyCount = nullptr;
		}

		template <typename T>
		void VoxelDataBase<T>::Integrate(IntegrationParameters* parameters, cached_allocator* allocator, CUstream_st* stream)
		{
#ifdef __CUDACC__
			VoxelDataBaseIntegrationParameters* params = dynamic_cast<VoxelDataBaseIntegrationParameters*>(parameters);
			if (nullptr == params)
				return;

			int threads = 256;
			int blocks = (params->mapWidth * params->mapHeight + threads - 1) / threads;

			nvtxRangePushA("!!!Integrate");
			Kernel_Integrate << <blocks, threads, 0, stream >> > (*this, *params);
			cudaStreamSynchronize(stream);
			nvtxRangePop();
#endif
		}

		template <typename T>
		void VoxelDataBase<T>::IntegrateSurfaceNormal(IntegrationParameters* parameters, cached_allocator* allocator, CUstream_st* stream)
		{
#ifdef __CUDACC__
			VoxelDataBaseIntegrationParameters* params = dynamic_cast<VoxelDataBaseIntegrationParameters*>(parameters);
			if (nullptr == params)
				return;

			int threads = 256;
			int blocks = (params->mapWidth * params->mapHeight + threads - 1) / threads;

			nvtxRangePushA("IntegrateSurfaceNormal");
			Kernel_IntegrateSurfaceNormal << <blocks, threads, 0, stream >> > (*this, *params);
			cudaStreamSynchronize(stream);
			nvtxRangePop();
#endif
		}

		template <typename T>
		void VoxelDataBase<T>::IntegrateDirectional(
			IntegrationParameters* parameters,
			cached_allocator* allocator,
			CUstream_st* stream)
		{
#ifdef __CUDACC__
			if constexpr (std::is_same_v<T, DirectionalVoxel<DummyVoxel>>)
			{
				VoxelDataBaseIntegrationParameters* params =
					dynamic_cast<VoxelDataBaseIntegrationParameters*>(parameters);
				if (nullptr == params) return;

				auto& self = reinterpret_cast<VoxelDataBase<DirectionalVoxel<DummyVoxel>>&>(*this);

				// --- 프레임 시작: dirty 카운터 리셋 ---
				cudaMemsetAsync(d_dirtyCount, 0, sizeof(uint32_t), stream);

				const int threads = 256;
				const int pixBlocks =
					(params->mapWidth * params->mapHeight + threads - 1) / threads;

				// Phase 1
				nvtxRangePushA("IntegrateDirectional_Phase1");
				Kernel_IntegrateDirectional_Phase1 << <pixBlocks, threads, 0, stream >> > (self, *params);
				cudaStreamSynchronize(stream);
				nvtxRangePop();

				// Phase 2: 이번 프레임에 실제로 터치된 슬롯만 처리
				uint32_t dirtyCount = 0;
				cudaMemcpy(&dirtyCount, d_dirtyCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

				if (dirtyCount > 0)
				{
					constexpr uint32_t VPB =
						VoxelBlock<DirectionalVoxel<DummyVoxel>>::VOXELS_PER_BLOCK;

					// dirty 슬롯에 중복이 있어도 총 스레드 수는 프레임당 터치 블록 수에 비례
					uint32_t totalThreads = dirtyCount * VPB;
					int volBlocks = (totalThreads + threads - 1) / threads;

					nvtxRangePushA("IntegrateDirectional_Phase2");
					Kernel_IntegrateDirectional_Phase2_DummyVoxel << <volBlocks, threads, 0, stream >> > (
						self, d_dirtySlots, dirtyCount);
					cudaStreamSynchronize(stream);
					nvtxRangePop();
				}
			}
			else if constexpr (std::is_same_v<T, DirectionalVoxel<Voxel>>)
			{
				// Voxel 버전도 동일 패턴으로 수정
				VoxelDataBaseIntegrationParameters* params =
					dynamic_cast<VoxelDataBaseIntegrationParameters*>(parameters);
				if (nullptr == params) return;

				auto& self = reinterpret_cast<VoxelDataBase<DirectionalVoxel<Voxel>>&>(*this);

				cudaMemsetAsync(d_dirtyCount, 0, sizeof(uint32_t), stream);

				const int threads = 256;
				const int pixBlocks =
					(params->mapWidth * params->mapHeight + threads - 1) / threads;

				nvtxRangePushA("IntegrateDirectional_Phase1");
				Kernel_IntegrateDirectional_Phase1 << <pixBlocks, threads, 0, stream >> > (self, *params);
				cudaStreamSynchronize(stream);
				nvtxRangePop();

				uint32_t dirtyCount = 0;
				cudaMemcpy(&dirtyCount, d_dirtyCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

				if (dirtyCount > 0)
				{
					constexpr uint32_t VPB =
						VoxelBlock<DirectionalVoxel<Voxel>>::VOXELS_PER_BLOCK;
					uint32_t totalThreads = dirtyCount * VPB;
					int volBlocks = (totalThreads + threads - 1) / threads;

					nvtxRangePushA("IntegrateDirectional_Phase2");
					Kernel_IntegrateDirectional_Phase2_Voxel << <volBlocks, threads, 0, stream >> > (
						self, d_dirtySlots, dirtyCount);
					cudaStreamSynchronize(stream);
					nvtxRangePop();
				}
			}
#endif
		}

		template <typename T>
		void VoxelDataBase<T>::Extract(ExtractionParameters* parameters, cached_allocator* allocator, CUstream_st* stream)
		{
#ifdef __CUDACC__
			VoxelDataBaseExtractionParameters* params = dynamic_cast<VoxelDataBaseExtractionParameters*>(parameters);
			if (nullptr == params)
				return;
			if (nullptr == params->d_out || nullptr == params->d_count || params->maxOut == 0)
				return;

			nvtxRangePushA("ExtetractVoxelDataBase");
			cudaMemsetAsync(params->d_count, 0, sizeof(uint32_t), stream);
			cudaStreamSynchronize(stream);

			int threads = 256;
			int numBlocks = (maxBlockCount + threads - 1) / threads;

			switch (params->mode)
			{
			case VoxelDataBaseExtractionParameters::Mode::AllOccupied:
				Kernel_ExtractAllOccupied << <numBlocks, threads, 0, stream >> > (
					*this, blockSize, params->d_out, params->d_count, params->maxOut);
				break;
			case VoxelDataBaseExtractionParameters::Mode::ZeroCrossing:
				Kernel_ExtractZeroCrossing << <numBlocks, threads, 0, stream >> > (
					*this, blockSize, params->d_out, params->d_count, params->maxOut);
				break;
			default:
				break;
			}

			cudaStreamSynchronize(stream);
			nvtxRangePop();
#endif
		}

		template <typename T>
		void VoxelDataBase<T>::ExtractIntraRegion(ExtractionParameters* parameters, cached_allocator* allocator, CUstream_st* stream)
		{
#ifdef __CUDACC__
			VoxelDataBaseExtractionParameters* params = dynamic_cast<VoxelDataBaseExtractionParameters*>(parameters);
			if (nullptr == params)
				return;
			if (nullptr == params->d_out || nullptr == params->d_count || params->maxOut == 0)
				return;

			nvtxRangePushA("ExtractIntraRegion");
			cudaMemsetAsync(params->d_count, 0, sizeof(uint32_t), stream);
			cudaStreamSynchronize(stream);

			int threads = 256;
			int numBlocks = (maxBlockCount + threads - 1) / threads;

			Kernel_ExtractIntraRegion << <numBlocks, threads, 0, stream >> > (
				*this, blockSize, params->mode,
				params->cacheMin, params->cacheMax,
				params->d_out, params->d_count, params->maxOut);

			cudaStreamSynchronize(stream);
			nvtxRangePop();
#endif
		}

		template <typename T>
		void VoxelDataBase<T>::ExtractDirectional(ExtractionParameters* parameters, cached_allocator* allocator, CUstream_st* stream)
		{
#ifdef __CUDACC__
			VoxelDataBaseExtractionParameters* params =
				dynamic_cast<VoxelDataBaseExtractionParameters*>(parameters);
			if (nullptr == params) return;
			if (nullptr == params->d_out || nullptr == params->d_count || params->maxOut == 0) return;

			cudaMemsetAsync(params->d_count, 0, sizeof(uint32_t), stream);
			cudaStreamSynchronize(stream);

			int threads = 256;
			int numBlocks = (maxBlockCount + threads - 1) / threads;

			if constexpr (std::is_same_v<T, DirectionalVoxel<Voxel>> ||
				std::is_same_v<T, DirectionalVoxel<DummyVoxel>>)
			{
				auto& self = reinterpret_cast<VoxelDataBase<DirectionalVoxel<DummyVoxel>>&>(*this);
				Kernel_ExtractZeroCrossing_Directional << <numBlocks, threads, 0, stream >> > (
					self, blockSize, params->d_out, params->d_count, params->maxOut);
			}
			else
			{
				switch (params->mode)
				{
				case VoxelDataBaseExtractionParameters::Mode::AllOccupied:
					Kernel_ExtractAllOccupied << <numBlocks, threads, 0, stream >> > (
						*this, blockSize, params->d_out, params->d_count, params->maxOut);
					break;
				case VoxelDataBaseExtractionParameters::Mode::ZeroCrossing:
					Kernel_ExtractZeroCrossing << <numBlocks, threads, 0, stream >> > (
						*this, blockSize, params->d_out, params->d_count, params->maxOut);
					break;
				default: break;
				}
			}

			cudaStreamSynchronize(stream);
#endif
		}

		template <typename T>
		void VoxelDataBase<T>::PerFrameFilter(
			IntegrationParameters* parameters,
			cached_allocator* allocator,
			CUstream_st* stream)
		{
#ifdef __CUDACC__
			VoxelDataBaseIntegrationParameters* params =
				dynamic_cast<VoxelDataBaseIntegrationParameters*>(parameters);
			if (nullptr == params)
				return;

			int threads = 256;
			int numBlocks = (params->mapWidth * params->mapHeight + threads - 1) / threads;

			nvtxRangePushA("PerFrameFilter");
			Kernel_PerFrameFilter << <numBlocks, threads, 0, stream >> > (*this, *params);
			cudaStreamSynchronize(stream);
			nvtxRangePop();
#endif
		}

		struct VoxelDBHeader
		{
			uint32_t magic;
			uint32_t version;
			uint32_t maxBlockCount;
			uint32_t blockCount;
			float    blockSize;
			uint32_t voxelStride;
			uint32_t reserved[2];
		};

		static constexpr uint32_t VOXEL_DB_MAGIC = 0x564F5842u;
		static constexpr uint32_t VOXEL_DB_VERSION = 1u;

		template <typename T>
		void VoxelDataBase<T>::Serialize(const std::wstring& filename)
		{
			uint32_t blockCount = 0;
			cudaMemcpy(&blockCount, d_blockCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

			std::vector<uint64_t>      h_hashTable(maxBlockCount);
			std::vector<VoxelBlock<T>> h_blocks(maxBlockCount);

			cudaMemcpy(h_hashTable.data(), d_hashTable, sizeof(uint64_t) * maxBlockCount, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_blocks.data(), d_blocks, sizeof(VoxelBlock<T>) * maxBlockCount, cudaMemcpyDeviceToHost);

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

			VoxelDBHeader header = {};
			fread(&header, sizeof(VoxelDBHeader), 1, fp);

			if (header.magic != VOXEL_DB_MAGIC)
			{
				printf("[VoxelDataBase::Deserialize] Invalid magic number\n");
				fclose(fp); return;
			}
			if (header.version != VOXEL_DB_VERSION)
			{
				printf("[VoxelDataBase::Deserialize] Version mismatch: file=%u, expected=%u\n",
					header.version, VOXEL_DB_VERSION);
				fclose(fp); return;
			}
			if (header.voxelStride != (uint32_t)sizeof(T))
			{
				printf("[VoxelDataBase::Deserialize] Voxel stride mismatch: file=%u, sizeof(T)=%u\n",
					header.voxelStride, (uint32_t)sizeof(T));
				fclose(fp); return;
			}

			if (header.maxBlockCount != maxBlockCount || header.blockSize != blockSize)
			{
				printf("[VoxelDataBase::Deserialize] Re-initializing: maxBlocks %u->%u, blockSize %.4f->%.4f\n",
					maxBlockCount, header.maxBlockCount, blockSize, header.blockSize);
				Terminate();
				Initialize(header.maxBlockCount, header.blockSize);
			}

			std::vector<uint64_t>      h_hashTable(maxBlockCount);
			std::vector<VoxelBlock<T>> h_blocks(maxBlockCount);

			fread(h_hashTable.data(), sizeof(uint64_t), maxBlockCount, fp);
			fread(h_blocks.data(), sizeof(VoxelBlock<T>), maxBlockCount, fp);
			fclose(fp);

			cudaMemcpy(d_hashTable, h_hashTable.data(), sizeof(uint64_t) * maxBlockCount, cudaMemcpyHostToDevice);
			cudaMemcpy(d_blocks, h_blocks.data(), sizeof(VoxelBlock<T>) * maxBlockCount, cudaMemcpyHostToDevice);
			cudaMemcpy(d_blockCount, &header.blockCount, sizeof(uint32_t), cudaMemcpyHostToDevice);

			printf("[VoxelDataBase::Deserialize] Loaded %u blocks from file\n", header.blockCount);
		}

		template <typename T>
		void VoxelDataBase<T>::SaveToPLY(const std::wstring& filename)
		{
			uint32_t blockCount = 0;
			cudaMemcpy(&blockCount, d_blockCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

			if (blockCount == 0)
			{
				printf("[VoxelDataBase::SaveToPLY] Database is empty. Nothing to save.\n");
				return;
			}

			uint32_t maxVoxelsOut = blockCount * 512;

			float3* d_pos = nullptr;
			float3* d_nor = nullptr;
			uchar3* d_col = nullptr;
			uint32_t* d_extractedCount = nullptr;

			cudaMalloc(&d_pos, sizeof(float3) * maxVoxelsOut);
			cudaMalloc(&d_nor, sizeof(float3) * maxVoxelsOut);
			cudaMalloc(&d_col, sizeof(uchar3) * maxVoxelsOut);
			cudaMalloc(&d_extractedCount, sizeof(uint32_t));

			cudaMemset(d_extractedCount, 0, sizeof(uint32_t));

			int threads = 256;
			int blocks = (maxBlockCount + threads - 1) / threads;

			Kernel_Serialize << <blocks, threads >> > (*this, d_pos, d_nor, d_col, d_extractedCount, maxVoxelsOut);
			cudaDeviceSynchronize();

			uint32_t voxelCount = 0;
			cudaMemcpy(&voxelCount, d_extractedCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

			if (voxelCount > 0)
			{
				std::vector<float3> h_pos(voxelCount);
				std::vector<float3> h_nor(voxelCount);
				std::vector<uchar3> h_col(voxelCount);

				cudaMemcpy(h_pos.data(), d_pos, sizeof(float3) * voxelCount, cudaMemcpyDeviceToHost);
				cudaMemcpy(h_nor.data(), d_nor, sizeof(float3) * voxelCount, cudaMemcpyDeviceToHost);
				cudaMemcpy(h_col.data(), d_col, sizeof(uchar3) * voxelCount, cudaMemcpyDeviceToHost);

				PLYFormat ply;
				for (size_t i = 0; i < voxelCount; i++)
				{
					auto& p = h_pos[i];
					auto& n = h_nor[i];
					auto& c = h_col[i];

					ply.AddPoint(p.x, p.y, p.z);
					ply.AddNormal(n.x, n.y, n.z);
					ply.AddColor(c.x, c.y, c.z);
				}
				ply.Serialize(filename);

				printf("[VoxelDataBase::SaveToPLY] Successfully saved %u voxels to PLY.\n", voxelCount);
			}

			cudaFree(d_pos);
			cudaFree(d_nor);
			cudaFree(d_col);
			cudaFree(d_extractedCount);
		}

		template class VoxelDataBase<Voxel>;
		template<> std::unique_ptr<DataFrameRecorder<VoxelDataBaseIntegrationParameters>> VoxelDataBase<Voxel>::recorder = nullptr;

		template class VoxelDataBase<DirectionalVoxel<Voxel>>;
		template<> std::unique_ptr<DataFrameRecorder<VoxelDataBaseIntegrationParameters>> VoxelDataBase<DirectionalVoxel<Voxel>>::recorder = nullptr;

		template class VoxelDataBase<DirectionalVoxel<DummyVoxel>>;
		template<> std::unique_ptr<DataFrameRecorder<VoxelDataBaseIntegrationParameters>> VoxelDataBase<DirectionalVoxel<DummyVoxel>>::recorder = nullptr;
	}
}
