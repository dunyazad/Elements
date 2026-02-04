#include <VVV/VVV.h>
#include <device_launch_parameters.h>
#include <cstdio>

namespace VVV
{
    static constexpr float TSDF_TRUNC_DIST = 1.0f;
    static constexpr float MAX_WEIGHT = 100.0f;

    CUDA_HOST_DEVICE inline float Clamp(float val, float min, float max)
    {
        return fminf(max, fmaxf(min, val));
    }

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
        if (d_blocks)
        {
            cudaFree(d_blocks);
        }
        if (d_hashTable)
        {
            cudaFree(d_hashTable);
        }
        if (d_blockCount)
        {
            cudaFree(d_blockCount);
        }
        d_blocks = nullptr;
        d_hashTable = nullptr;
        d_blockCount = nullptr;
        maxBlockCount = 0;
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
        if (idx >= count)
        {
            return;
        }

        Vector3f p = points[idx];
        Vector3b c = colors[idx];
        Morton64 blockKey = Morton64::FromPosition(p, blockSize);
        uint64_t mKey = blockKey.code;
        if (mKey == 0)
        {
            mKey = 0xFFFFFFFFFFFFFFFFULL;
        }

        uint32_t slot = StrongHash(mKey, db.maxBlockCount);
        uint32_t start = slot;
        BlockID bid = INVALID_BLOCK;

        while (true)
        {
            unsigned long long* slotPtr = (unsigned long long*) & db.d_hashTable[slot];
            unsigned long long prev = atomicCAS(slotPtr, 0ULL, (unsigned long long)mKey);

            if (prev == 0)
            {
                atomicAdd(db.d_blockCount, 1);
                bid = slot;
                break;
            }
            if (prev == mKey)
            {
                bid = slot;
                break;
            }

            slot = (slot + 1) % db.maxBlockCount;
            if (slot == start)
            {
                break;
            }
        }

        if (bid != INVALID_BLOCK)
        {
            float vSize = blockSize / 8.0f;
            Vector3f bc = blockKey.ToPosition(blockSize);
            int lx = static_cast<int>(floorf((p.x - (bc.x - blockSize * 0.5f)) / vSize + 1e-5f));
            int ly = static_cast<int>(floorf((p.y - (bc.y - blockSize * 0.5f)) / vSize + 1e-5f));
            int lz = static_cast<int>(floorf((p.z - (bc.z - blockSize * 0.5f)) / vSize + 1e-5f));

            lx = (lx < 0) ? 0 : (lx > 7 ? 7 : lx);
            ly = (ly < 0) ? 0 : (ly > 7 ? 7 : ly);
            lz = (lz < 0) ? 0 : (lz > 7 ? 7 : lz);

            Voxel& v = db.d_blocks[bid].voxels[(lz << 6) | (ly << 3) | lx];
            atomicAdd(&v.value, 1.0f);
            v.color = c; // Note: Race condition point
            db.d_blocks[bid].lastTouchedFrameId = frameId;
        }
    }

    __global__ void TSDFIntegrateKernel(VoxelDataBase db, VVV::Matrix4f invRT, const Vector3f* points, const Vector3f* normals, const Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
    {
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count)
        {
            return;
        }

        // 1. 입력 로컬 데이터 로드 및 월드 변환
        Vector3f p_local = points[index];
        Vector3f n_local = normals[index];
        Vector3b color = colors[index];

        // Camera to World 변환 (invRT = rt0)
        Vector3f p_world = invRT.Transform(p_local);

        // 법선 벡터 월드 회전 적용 (회전만 적용)
        Vector3f n_world = {
            invRT.data[0] * n_local.x + invRT.data[4] * n_local.y + invRT.data[8] * n_local.z,
            invRT.data[1] * n_local.x + invRT.data[5] * n_local.y + invRT.data[9] * n_local.z,
            invRT.data[2] * n_local.x + invRT.data[6] * n_local.y + invRT.data[10] * n_local.z
        };

        float voxelSize = blockSize / 8.0f;
        float truncDist = voxelSize * 2.0f;

        // 2. [검토 핵심] 주변 8개 복셀의 정수 그리드 좌표(Global Grid Index)를 직접 구함
        // floorf(p_world / voxelSize)가 해당 포인트가 속한 복셀의 시작점입니다.
        int gx = static_cast<int>(floorf(p_world.x / voxelSize));
        int gy = static_cast<int>(floorf(p_world.y / voxelSize));
        int gz = static_cast<int>(floorf(p_world.z / voxelSize));

        // 3. 2x2x2 주변 복셀 순회 (각 복셀의 블록 소속을 매번 확인)
        for (int lz = gz; lz <= gz + 1; ++lz)
        {
            for (int ly = gy; ly <= gy + 1; ++ly)
            {
                for (int lx = gx; lx <= gx + 1; ++lx)
                {
                    // 타겟 복셀의 정밀한 월드 위치 (중심점 잡지 말고 인덱스로만 계산)
                    Vector3f v_pos = { lx * voxelSize, ly * voxelSize, lz * voxelSize };

                    // [핵심] 타겟 복셀 위치로 블록 키를 생성하여 인접 블록을 찾아감
                    Morton64 blockKey = Morton64::FromPosition(v_pos, blockSize);
                    uint64_t key = blockKey.code;
                    if (key == 0) key = 0xFFFFFFFFFFFFFFFFULL;

                    uint32_t slot = StrongHash(key, db.maxBlockCount);
                    uint32_t start = slot;
                    BlockID bid = INVALID_BLOCK;

                    while (true)
                    {
                        unsigned long long* slotPtr = (unsigned long long*) & db.d_hashTable[slot];
                        unsigned long long prev = atomicCAS(slotPtr, 0ULL, (unsigned long long)key);

                        if (prev == 0 || prev == key)
                        {
                            if (prev == 0) atomicAdd(db.d_blockCount, 1);
                            bid = slot;
                            break;
                        }
                        slot = (slot + 1) % db.maxBlockCount;
                        if (slot == start) break;
                    }

                    if (bid != INVALID_BLOCK)
                    {
                        // 블록의 월드 시작점(Origin)을 다시 구함
                        Vector3f bc = blockKey.ToPosition(blockSize);
                        Vector3f blockOrigin = { bc.x - blockSize * 0.5f, bc.y - blockSize * 0.5f, bc.z - blockSize * 0.5f };

                        // [핵심] 블록 내 로컬 인덱스 (0~7) 계산
                        // gx, gy, gz와 블록 시작점 사이의 거리 차이를 voxelSize로 나눈 정수값
                        int local_lx = lx - static_cast<int>(floorf(blockOrigin.x / voxelSize + 0.1f));
                        int local_ly = ly - static_cast<int>(floorf(blockOrigin.y / voxelSize + 0.1f));
                        int local_lz = lz - static_cast<int>(floorf(blockOrigin.z / voxelSize + 0.1f));

                        if (local_lx >= 0 && local_lx < 8 && local_ly >= 0 && local_ly < 8 && local_lz >= 0 && local_lz < 8)
                        {
                            // 법선 방향 SDF 계산 ($dist$)
                            Vector3f diff = { v_pos.x - p_world.x, v_pos.y - p_world.y, v_pos.z - p_world.z };
                            float dist = diff.x * n_world.x + diff.y * n_world.y + diff.z * n_world.z;

                            if (fabs(dist) < truncDist)
                            {
                                Voxel& voxel = db.d_blocks[bid].voxels[(local_lz << 6) | (local_ly << 3) | local_lx];

                                float oldW = (float)voxel.valueCount;
                                float newW = 1.0f;
                                float combinedW = fminf(oldW + newW, MAX_WEIGHT);

                                // 가중치 평균으로 매끄러운 SDF 형성
                                voxel.value = (voxel.value * oldW + dist) / combinedW;
                                voxel.valueCount = (unsigned short)combinedW;

                                voxel.color.x = (uint8_t)((oldW * voxel.color.x + newW * color.x) / combinedW);
                                voxel.color.y = (uint8_t)((oldW * voxel.color.y + newW * color.y) / combinedW);
                                voxel.color.z = (uint8_t)((oldW * voxel.color.z + newW * color.z) / combinedW);

                                voxel.normal = n_world;
                            }
                        }
                        db.d_blocks[bid].lastTouchedFrameId = frameId;
                    }
                }
            }
        }
    }

    __global__ void ExtractKernel(VoxelDataBase db, float blockSize, ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
    {
        uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (slot >= db.maxBlockCount)
        {
            return;
        }

        uint64_t key = db.d_hashTable[slot];
        if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL)
        {
            return;
        }

        Morton64 bKey(key);
        Vector3f bc = bKey.ToPosition(blockSize);
        float vSize = blockSize / 8.0f;

        for (int i = 0; i < 512; ++i)
        {
            Voxel& v = db.d_blocks[slot].voxels[i];
            if (v.valueCount > 0)
            {
                uint32_t idx = atomicAdd(count, 1);
                if (idx < maxOut)
                {
                    int lz = i / 64;
                    int ly = (i % 64) / 8;
                    int lx = i % 8;

                    out[idx].position = {
                        (bc.x - blockSize * 0.5f) + (lx + 0.5f) * vSize,
                        (bc.y - blockSize * 0.5f) + (ly + 0.5f) * vSize,
                        (bc.z - blockSize * 0.5f) + (lz + 0.5f) * vSize
                    };
                    out[idx].weight = (float)v.valueCount;
                    out[idx].color[0] = v.color.x;
                    out[idx].color[1] = v.color.y;
                    out[idx].color[2] = v.color.z;
                }
            }
        }
    }

    //__global__ void ExtractZeroCrossingKernel(VoxelDataBase db, float blockSize, ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
    //{
    //    uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    //    if (slot >= db.maxBlockCount)
    //    {
    //        return;
    //    }

    //    uint64_t key = db.d_hashTable[slot];
    //    if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL)
    //    {
    //        return;
    //    }

    //    Morton64 blockKey(key);
    //    Vector3f bc = blockKey.ToPosition(blockSize);
    //    Vector3f blockOrigin = { bc.x - blockSize * 0.5f, bc.y - blockSize * 0.5f, bc.z - blockSize * 0.5f };
    //    float vSize = blockSize / 8.0f;

    //    // 블록 내부 8x8x8 복셀 중 7x7x7 큐브를 검사
    //    for (int lz = 0; lz < 7; ++lz)
    //    {
    //        for (int ly = 0; ly < 7; ++ly)
    //        {
    //            for (int lx = 0; lx < 7; ++lx)
    //            {
    //                // 현재 복셀과 축 방향 인접 복셀들 가져오기 (7x7x7 범위 내라 블록 내 인덱싱 가능)
    //                Voxel& v0 = db.d_blocks[slot].voxels[(lz << 6) | (ly << 3) | lx];
    //                Voxel& vx = db.d_blocks[slot].voxels[(lz << 6) | (ly << 3) | (lx + 1)];
    //                Voxel& vy = db.d_blocks[slot].voxels[(lz << 6) | ((ly + 1) << 3) | lx];
    //                Voxel& vz = db.d_blocks[slot].voxels[((lz + 1) << 6) | (ly << 3) | lx];

    //                // 가중치가 있고 부호가 바뀌는 지점(Zero-crossing)인지 확인
    //                // X, Y, Z 각 축 방향으로 인접한 복셀 사이에서 부호 변화가 있는지 체크
    //                bool hasCrossing = false;

    //                // X-axis crossing
    //                if (v0.valueCount > 0 && vx.valueCount > 0 && (v0.value * vx.value <= 0.0f)) hasCrossing = true;
    //                // Y-axis crossing
    //                else if (v0.valueCount > 0 && vy.valueCount > 0 && (v0.value * vy.value <= 0.0f)) hasCrossing = true;
    //                // Z-axis crossing
    //                else if (v0.valueCount > 0 && vz.valueCount > 0 && (v0.value * vz.value <= 0.0f)) hasCrossing = true;

    //                if (hasCrossing)
    //                {
    //                    uint32_t idx = atomicAdd(count, 1);
    //                    if (idx < maxOut)
    //                    {
    //                        // 선형 보간(Linear Interpolation)을 통해 더 정확한 Zero-point 계산 가능하나, 
    //                        // 여기서는 간단히 Crossing이 발생한 복셀의 위치를 저장
    //                        out[idx].position = {
    //                            blockOrigin.x + (lx + 0.5f) * vSize,
    //                            blockOrigin.y + (ly + 0.5f) * vSize,
    //                            blockOrigin.z + (lz + 0.5f) * vSize
    //                        };
    //                        out[idx].weight = v0.value;
    //                        out[idx].color[0] = v0.color.x;
    //                        out[idx].color[1] = v0.color.y;
    //                        out[idx].color[2] = v0.color.z;
    //                    }
    //                }
    //            }
    //        }
    //    }
    //}

    __device__ inline Vector3f GetInterpolatedPos(Vector3f p1, Vector3f p2, float v1, float v2)
    {
        // 선형 보간 공식: P = P1 + (-V1 / (V2 - V1)) * (P2 - P1)
        float mu = -v1 / (v2 - v1);
        return {
            p1.x + mu * (p2.x - p1.x),
            p1.y + mu * (p2.y - p1.y),
            p1.z + mu * (p2.z - p1.z)
        };
    }

    __global__ void ExtractZeroCrossingKernel(VoxelDataBase db, float blockSize, ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
    {
        uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (slot >= db.maxBlockCount) return;

        uint64_t key = db.d_hashTable[slot];
        if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL) return;

        Morton64 bKey(key);
        Vector3f bc = bKey.ToPosition(blockSize);
        Vector3f origin = { bc.x - blockSize * 0.5f, bc.y - blockSize * 0.5f, bc.z - blockSize * 0.5f };
        float voxelSize = blockSize / 8.0f;

        for (int lz = 0; lz < 7; ++lz)
            for (int ly = 0; ly < 7; ++ly)
                for (int lx = 0; lx < 7; ++lx)
                {
                    int idx0 = (lz << 6) | (ly << 3) | lx;
                    Voxel& v0 = db.d_blocks[slot].voxels[idx0];

                    // 최소 가중치 조건을 두어 노이즈에 의한 구멍 방지
                    if (v0.valueCount < 2) continue;

                    Vector3f p0 = { origin.x + (lx + 0.5f) * voxelSize, origin.y + (ly + 0.5f) * voxelSize, origin.z + (lz + 0.5f) * voxelSize };

                    int neighbors[3] = { 1, 8, 64 };
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        Voxel& v1 = db.d_blocks[slot].voxels[idx0 + neighbors[axis]];

                        // 두 복셀 사이에서 SDF 값이 0을 통과하는지 확인
                        if (v1.valueCount >= 2 && (v0.value * v1.value < 0.0f))
                        {
                            uint32_t outIdx = atomicAdd(count, 1);
                            if (outIdx < maxOut)
                            {
                                Vector3f p1 = p0;
                                if (axis == 0) p1.x += voxelSize;
                                else if (axis == 1) p1.y += voxelSize;
                                else p1.z += voxelSize;

                                out[outIdx].position = GetInterpolatedPos(p0, p1, v0.value, v1.value);
                                out[outIdx].normal = v0.normal;
                                out[outIdx].color[0] = (uint8_t)((v0.color.x + v1.color.x) / 2);
                                out[outIdx].color[1] = (uint8_t)((v0.color.y + v1.color.y) / 2);
                                out[outIdx].color[2] = (uint8_t)((v0.color.z + v1.color.z) / 2);
                            }
                        }
                    }
                }
    }

#endif
}

extern "C"
{
    void VVV_Allocate(VVV::VoxelDataBase& db, uint32_t maxBlocks)
    {
        db.InternalAllocate(maxBlocks);
    }

    void VVV_Free(VVV::VoxelDataBase& db)
    {
        db.InternalFree();
    }

    void VVV_UpdateVoxelFromPoints(VVV::VoxelDataBase& db, const VVV::Vector3f* points, const VVV::Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
    {
        VVV::Vector3f* d_p;
        VVV::Vector3b* d_c;
        cudaMalloc(&d_p, sizeof(VVV::Vector3f) * count);
        cudaMalloc(&d_c, sizeof(VVV::Vector3b) * count);

        cudaMemcpy(d_p, points, sizeof(VVV::Vector3f) * count, cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, colors, sizeof(VVV::Vector3b) * count, cudaMemcpyHostToDevice);

        CUDA_TS(VVV_InsertKernel);
#ifdef __CUDACC__
        int threadsPerBlock = 256;
        int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
        VVV::InsertKernel << <blocksPerGrid, threadsPerBlock >> > (db, d_p, d_c, count, blockSize, frameId);
        cudaDeviceSynchronize();
#endif
        CUDA_TE(VVV_InsertKernel);

        cudaFree(d_p);
        cudaFree(d_c);
    }

    VVV_API void VVV_IntegrateTSDF(VVV::VoxelDataBase& db, const VVV::Matrix4f& rt, const VVV::Vector3f* d_points, const VVV::Vector3f* d_normals, const VVV::Vector3b* d_colors, uint32_t count, float blockSize, uint32_t frameId)
    {
        //VVV::Vector3f* d_p, * d_n;
        //VVV::Vector3b* d_c;
        //cudaMalloc(&d_p, sizeof(VVV::Vector3f) * count);
        //cudaMalloc(&d_n, sizeof(VVV::Vector3f) * count);
        //cudaMalloc(&d_c, sizeof(VVV::Vector3b) * count);

        //cudaMemcpy(d_p, points, sizeof(VVV::Vector3f) * count, cudaMemcpyHostToDevice);
        //cudaMemcpy(d_n, normals, sizeof(VVV::Vector3f) * count, cudaMemcpyHostToDevice);
        //cudaMemcpy(d_c, colors, sizeof(VVV::Vector3b) * count, cudaMemcpyHostToDevice);

#ifdef __CUDACC__
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        VVV::TSDFIntegrateKernel << <blocks, threads >> > (db, rt, d_points, d_normals, d_colors, count, blockSize, frameId);
        cudaDeviceSynchronize();
#endif

        //cudaFree(d_p); cudaFree(d_n); cudaFree(d_c);
    }

    uint32_t VVV_ExtractActiveVoxelsToHost(VVV::VoxelDataBase& db, float blockSize, VVV::ExtractedVoxel* hostBuffer, uint32_t maxOut)
    {
        VVV::ExtractedVoxel* d_out;
        uint32_t* d_cnt;
        cudaMalloc(&d_out, sizeof(VVV::ExtractedVoxel) * maxOut);
        cudaMalloc(&d_cnt, sizeof(uint32_t));
        cudaMemset(d_cnt, 0, sizeof(uint32_t));

#ifdef __CUDACC__
        int threadsPerBlock = 256;
        int blocksPerGrid = (db.maxBlockCount + threadsPerBlock - 1) / threadsPerBlock;
        VVV::ExtractKernel << <blocksPerGrid, threadsPerBlock >> > (db, blockSize, d_out, d_cnt, maxOut);
        cudaDeviceSynchronize();
#endif
        uint32_t res;
        cudaMemcpy(&res, d_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        uint32_t copyAmt = (res > maxOut) ? maxOut : res;
        cudaMemcpy(hostBuffer, d_out, sizeof(VVV::ExtractedVoxel) * copyAmt, cudaMemcpyDeviceToHost);

        cudaFree(d_out);
        cudaFree(d_cnt);
        return res;
    }

    uint32_t VVV_ExtractZeroCrossingVoxelsToHost(VVV::VoxelDataBase& db, float blockSize, VVV::ExtractedVoxel* hostBuffer, uint32_t maxOut)
    {
        VVV::ExtractedVoxel* d_out;
        uint32_t* d_cnt;
        cudaMalloc(&d_out, sizeof(VVV::ExtractedVoxel) * maxOut);
        cudaMalloc(&d_cnt, sizeof(uint32_t));
        cudaMemset(d_cnt, 0, sizeof(uint32_t));

#ifdef __CUDACC__
        int threadsPerBlock = 256;
        int blocksPerGrid = (db.maxBlockCount + threadsPerBlock - 1) / threadsPerBlock;
        VVV::ExtractZeroCrossingKernel << <blocksPerGrid, threadsPerBlock >> > (db, blockSize, d_out, d_cnt, maxOut);
        cudaDeviceSynchronize();
#endif
        uint32_t res;
        cudaMemcpy(&res, d_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        uint32_t copyAmt = (res > maxOut) ? maxOut : res;
        cudaMemcpy(hostBuffer, d_out, sizeof(VVV::ExtractedVoxel) * copyAmt, cudaMemcpyDeviceToHost);

        cudaFree(d_out);
        cudaFree(d_cnt);
        return res;
    }
}
