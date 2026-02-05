#include <VVV/VVV.h>
#include <device_launch_parameters.h>
#include <cstdio>

namespace VVV
{
    static constexpr float TSDF_TRUNC_DIST = 1.0f;
    static constexpr float MAX_WEIGHT = 100.0f;

    __host__ __device__ inline float Clamp(float val, float min, float max)
    {
        return fminf(max, fmaxf(min, val));
    }

    __global__ void InsertKernel(VoxelDataBase db, VVV::Matrix4f rt, const Vector3f* points, const Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count)
        {
            return;
        }

        Vector3f p = rt.Transform(points[idx]);

        // 해시 슬롯 할당 (이 단계에서 블록 카운트 증가)
        Voxel* v = db.GetOrCreateVoxel(p);

        if (v != nullptr)
        {
            // 1. 값 업데이트 (포인트 클라우드 적립 방식)
            atomicAdd(&(v->value), 1.0f);

            // 2. 중요: 가중치(valueCount) 업데이트
            // ExtractActiveVoxelsToHost 커널은 보통 이 값이 0보다 큰 복셀만 추출합니다.
            // valueCount가 unsigned short 혹은 정수형인 경우 아래와 같이 처리합니다.
            if (v->valueCount < 65535)
            {
                atomicAdd((unsigned int*)&(v->valueCount), 1);
                // 주의: 구조체 메모리 레이아웃에 따라 정수형 캐스팅 혹은 
                // 전용 atomic 함수를 사용해야 합니다.
            }

            // 3. 색상 기록
            v->color = colors[idx];

            // 4. 블록 활성화 프레임 업데이트
            BlockID bid = db.GetOrCreateBlockSlot(p);
            if (bid != INVALID_BLOCK)
            {
                db.d_blocks[bid].lastTouchedFrameId = frameId;
            }
        }
    }

#if 0
    __global__ void TSDFIntegrateKernel(VoxelDataBase db, VVV::Matrix4f rt, const Vector3f* points, const Vector3f* normals, const Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
    {
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count) return;

        Vector3f p_local = points[index];
        Vector3f n_local = normals[index];
        Vector3b color = colors[index];

        // 월드 좌표계 변환 및 노멀 정규화
        Vector3f p_world = rt.Transform(p_local);
        Vector3f n_world = {
            rt.data[0] * n_local.x + rt.data[4] * n_local.y + rt.data[8] * n_local.z,
            rt.data[1] * n_local.x + rt.data[5] * n_local.y + rt.data[9] * n_local.z,
            rt.data[2] * n_local.x + rt.data[6] * n_local.y + rt.data[10] * n_local.z
        };

        float n_len = rsqrtf(n_world.x * n_world.x + n_world.y * n_world.y + n_world.z * n_world.z + 1e-10f);
        n_world.x *= n_len; n_world.y *= n_len; n_world.z *= n_len;

        float voxel_size = blockSize / 8.0f;
        float inv_voxel_size = 1.0f / voxel_size;
        float trunc_dist = voxel_size * 5.0f;

        // DDA 시작점 (표면 뒤에서 앞으로 관통)
        Vector3f start_pos = { p_world.x - n_world.x * trunc_dist, p_world.y - n_world.y * trunc_dist, p_world.z - n_world.z * trunc_dist };

        // 정수 좌표 계산 (floorf 대용으로 정밀도 확보)
        int curr_x = __float2int_rd(start_pos.x * inv_voxel_size);
        int curr_y = __float2int_rd(start_pos.y * inv_voxel_size);
        int curr_z = __float2int_rd(start_pos.z * inv_voxel_size);

        int step_x = (n_world.x > 0) ? 1 : -1;
        int step_y = (n_world.y > 0) ? 1 : -1;
        int step_z = (n_world.z > 0) ? 1 : -1;

        // t_max 초기값 (부동소수점 오차 방지를 위해 아주 작은 epsilon 적용)
        auto calc_t_max = [&](float pos, float dir, int step, int curr) {
            if (fabsf(dir) < 1e-7f) return 1e30f;
            float border = (float)(curr + (step > 0 ? 1 : 0)) * voxel_size;
            return (border - pos) / dir;
            };

        float t_max_x = calc_t_max(start_pos.x, n_world.x, step_x, curr_x);
        float t_max_y = calc_t_max(start_pos.y, n_world.y, step_y, curr_y);
        float t_max_z = calc_t_max(start_pos.z, n_world.z, step_z, curr_z);

        float t_delta_x = (fabsf(n_world.x) > 1e-7f) ? fabsf(voxel_size / n_world.x) : 1e30f;
        float t_delta_y = (fabsf(n_world.y) > 1e-7f) ? fabsf(voxel_size / n_world.y) : 1e30f;
        float t_delta_z = (fabsf(n_world.z) > 1e-7f) ? fabsf(voxel_size / n_world.z) : 1e30f;

        float max_t = 2.0f * trunc_dist;
        float t = 0.0f;

        while (t <= max_t)
        {
            Vector3f voxel_center = { (curr_x + 0.5f) * voxel_size, (curr_y + 0.5f) * voxel_size, (curr_z + 0.5f) * voxel_size };
            Voxel* voxel_ptr = db.GetOrCreateVoxel(voxel_center);

            if (voxel_ptr != nullptr)
            {
                // 투영 거리(Dot product) 기반 SDF
                float dist = (voxel_center.x - p_world.x) * n_world.x +
                    (voxel_center.y - p_world.y) * n_world.y +
                    (voxel_center.z - p_world.z) * n_world.z;

                // 가중치 업데이트 (Atomic 기반 안전 확보)
                unsigned int* weight_ptr = (unsigned int*)&(voxel_ptr->valueCount);
                unsigned int old_w_int = atomicAdd(weight_ptr, 1);
                float old_w = (float)old_w_int;

                // 가중치 제한 및 통합
                float limited_w = fminf(old_w + 1.0f, MAX_WEIGHT);
                float weight_factor = old_w / (old_w + 1.0f);
                float new_factor = 1.0f / (old_w + 1.0f);

                voxel_ptr->value = (voxel_ptr->value * weight_factor) + (dist * new_factor);

                voxel_ptr->color.x = (uint8_t)(voxel_ptr->color.x * weight_factor + color.x * new_factor);
                voxel_ptr->color.y = (uint8_t)(voxel_ptr->color.y * weight_factor + color.y * new_factor);
                voxel_ptr->color.z = (uint8_t)(voxel_ptr->color.z * weight_factor + color.z * new_factor);

                voxel_ptr->normal = n_world;

                VoxelBlock* block = db.GetVoxelBlock(voxel_center);
                if (block) block->lastTouchedFrameId = frameId;
            }

            // DDA 이동: 수치적 안정성을 위해 작은 epsilon 보정 고려 가능
            if (t_max_x < t_max_y) {
                if (t_max_x < t_max_z) { t = t_max_x; t_max_x += t_delta_x; curr_x += step_x; }
                else { t = t_max_z; t_max_z += t_delta_z; curr_z += step_z; }
            }
            else {
                if (t_max_y < t_max_z) { t = t_max_y; t_max_y += t_delta_y; curr_y += step_y; }
                else { t = t_max_z; t_max_z += t_delta_z; curr_z += step_z; }
            }
        }
    }
#endif // 0

    __global__ void TSDFIntegrateKernel(VoxelDataBase db, VVV::Matrix4f rt, const Vector3f* points, const Vector3f* normals, const Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
    {
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count)
        {
            return;
        }

        Vector3f p_local = points[index];
        Vector3f n_local = normals[index];
        Vector3b color = colors[index];

        Vector3f p_world = rt.Transform(p_local);
        Vector3f n_world = {
            rt.data[0] * n_local.x + rt.data[4] * n_local.y + rt.data[8] * n_local.z,
            rt.data[1] * n_local.x + rt.data[5] * n_local.y + rt.data[9] * n_local.z,
            rt.data[2] * n_local.x + rt.data[6] * n_local.y + rt.data[10] * n_local.z
        };

        float n_len = rsqrtf(n_world.x * n_world.x + n_world.y * n_world.y + n_world.z * n_world.z + 1e-10f);
        n_world.x *= n_len; n_world.y *= n_len; n_world.z *= n_len;

        float voxel_size = blockSize / 8.0f;
        float inv_voxel_size = 1.0f / voxel_size;
        float trunc_dist = voxel_size * 5.0f;

        Vector3f start_pos = {
            p_world.x - n_world.x * trunc_dist,
            p_world.y - n_world.y * trunc_dist,
            p_world.z - n_world.z * trunc_dist
        };

        int curr_x = __float2int_rd(start_pos.x * inv_voxel_size);
        int curr_y = __float2int_rd(start_pos.y * inv_voxel_size);
        int curr_z = __float2int_rd(start_pos.z * inv_voxel_size);

        int step_x = (n_world.x > 0) ? 1 : -1;
        int step_y = (n_world.y > 0) ? 1 : -1;
        int step_z = (n_world.z > 0) ? 1 : -1;

        auto calc_t_max = [&](float pos, float dir, int step, int curr) {
            if (fabsf(dir) < 1e-7f) return 1e30f;
            float border = (float)(curr + (step > 0 ? 1 : 0)) * voxel_size;
            return (border - pos) / dir;
            };

        float t_max_x = calc_t_max(start_pos.x, n_world.x, step_x, curr_x);
        float t_max_y = calc_t_max(start_pos.y, n_world.y, step_y, curr_y);
        float t_max_z = calc_t_max(start_pos.z, n_world.z, step_z, curr_z);

        float t_delta_x = (fabsf(n_world.x) > 1e-7f) ? fabsf(voxel_size / n_world.x) : 1e30f;
        float t_delta_y = (fabsf(n_world.y) > 1e-7f) ? fabsf(voxel_size / n_world.y) : 1e30f;
        float t_delta_z = (fabsf(n_world.z) > 1e-7f) ? fabsf(voxel_size / n_world.z) : 1e30f;

        float max_t = 2.0f * trunc_dist;
        float t = 0.0f;

        while (t <= max_t)
        {
            Vector3f voxel_center = { (curr_x + 0.5f) * voxel_size, (curr_y + 0.5f) * voxel_size, (curr_z + 0.5f) * voxel_size };
            Voxel* voxel_ptr = db.GetOrCreateVoxel(voxel_center);

            if (voxel_ptr != nullptr)
            {
                float dist = (voxel_center.x - p_world.x) * n_world.x +
                    (voxel_center.y - p_world.y) * n_world.y +
                    (voxel_center.z - p_world.z) * n_world.z;

                // 가중치 업데이트 (MAX_WEIGHT 제한 적용)
                unsigned int* weight_ptr = (unsigned int*)&(voxel_ptr->valueCount);
                unsigned int old_w_int = atomicAdd(weight_ptr, 1);

                float old_w = fminf((float)old_w_int, MAX_WEIGHT);
                float new_w = old_w + 1.0f;

                float weight_factor = old_w / new_w;
                float new_factor = 1.0f / new_w;

                // 1. TSDF 값 통합
                voxel_ptr->value = (voxel_ptr->value * weight_factor) + (dist * new_factor);

                // 2. Color 통합 (반올림 보정을 위해 +0.5f 적용하여 탁해짐 방지)
                voxel_ptr->color.x = (uint8_t)((float)voxel_ptr->color.x * weight_factor + (float)color.x * new_factor + 0.5f);
                voxel_ptr->color.y = (uint8_t)((float)voxel_ptr->color.y * weight_factor + (float)color.y * new_factor + 0.5f);
                voxel_ptr->color.z = (uint8_t)((float)voxel_ptr->color.z * weight_factor + (float)color.z * new_factor + 0.5f);

                // 3. Normal 통합 (단순 덮어쓰기가 아니라 가중 평균 후 정규화)
                Vector3f blended_n = {
                    voxel_ptr->normal.x * weight_factor + n_world.x * new_factor,
                    voxel_ptr->normal.y * weight_factor + n_world.y * new_factor,
                    voxel_ptr->normal.z * weight_factor + n_world.z * new_factor
                };
                float bn_len = rsqrtf(blended_n.x * blended_n.x + blended_n.y * blended_n.y + blended_n.z * blended_n.z + 1e-10f);
                voxel_ptr->normal = { blended_n.x * bn_len, blended_n.y * bn_len, blended_n.z * bn_len };

                VoxelBlock* block = db.GetVoxelBlock(voxel_center);
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

    __global__ void ExtractKernel(VoxelDataBase db, float blockSize, ExtractedVoxel* out, uint32_t* count, uint32_t maxOut)
    {
        uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (slot >= db.maxBlockCount) return;

        uint64_t key = db.d_hashTable[slot];
        if (key == 0 || key == 0xFFFFFFFFFFFFFFFFULL) return;

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

    __device__ inline Vector3f GetInterpolatedPos(Vector3f p1, Vector3f p2, float v1, float v2)
    {
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
        Vector3f origin = { bc.x - blockSize * 0.5f, bc.y - blockSize * 0.5f, bc.z - blockSize * 0.5f };
        float voxelSize = blockSize / 8.0f;

        for (int lz = 0; lz < 8; ++lz)
        {
            for (int ly = 0; ly < 8; ++ly)
            {
                for (int lx = 0; lx < 8; ++lx)
                {
                    int idx0 = (lz << 6) | (ly << 3) | lx;
                    Voxel& v0 = db.d_blocks[slot].voxels[idx0];

                    // 노이즈 제거를 위해 가중치 임계값 상향 조정 고려
                    if (v0.valueCount < 2)
                    {
                        continue;
                    }

                    Vector3f p0 = {
                        origin.x + (lx + 0.5f) * voxelSize,
                        origin.y + (ly + 0.5f) * voxelSize,
                        origin.z + (lz + 0.5f) * voxelSize
                    };

                    int neighbors[3] = { 1, 8, 64 };
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        bool isInternal = false;
                        if (axis == 0 && lx < 7) isInternal = true;
                        else if (axis == 1 && ly < 7) isInternal = true;
                        else if (axis == 2 && lz < 7) isInternal = true;

                        Voxel* v1_ptr = nullptr;
                        Vector3f p1 = p0;

                        if (axis == 0) p1.x += voxelSize;
                        else if (axis == 1) p1.y += voxelSize;
                        else p1.z += voxelSize;

                        if (isInternal)
                        {
                            v1_ptr = &db.d_blocks[slot].voxels[idx0 + neighbors[axis]];
                        }
                        else
                        {
                            v1_ptr = db.GetVoxel(p1);
                        }

                        if (v1_ptr != nullptr && v1_ptr->valueCount >= 2)
                        {
                            float v0v = v0.value;
                            float v1v = v1_ptr->value;

                            // Zero-crossing 체크
                            if (v0v * v1v < 0.0f)
                            {
                                uint32_t outIdx = atomicAdd(count, 1);
                                if (outIdx < maxOut)
                                {
                                    // 선형 보간 계수 계산
                                    float mu = -v0v / (v1v - v0v);
                                    mu = fminf(fmaxf(mu, 0.0f), 1.0f);

                                    // 1. 위치 보간
                                    out[outIdx].position = {
                                        p0.x + mu * (p1.x - p0.x),
                                        p0.y + mu * (p1.y - p0.y),
                                        p0.z + mu * (p1.z - p0.z)
                                    };

                                    // 2. 법선 보간 및 정규화 (표면 부드러움 향상)
                                    Vector3f n0 = v0.normal;
                                    Vector3f n1 = v1_ptr->normal;
                                    Vector3f blendedNormal = {
                                        n0.x + mu * (n1.x - n0.x),
                                        n0.y + mu * (n1.y - n0.y),
                                        n0.z + mu * (n1.z - n0.z)
                                    };
                                    float invLen = rsqrtf(blendedNormal.x * blendedNormal.x +
                                        blendedNormal.y * blendedNormal.y +
                                        blendedNormal.z * blendedNormal.z + 1e-8f);
                                    out[outIdx].normal = {
                                        blendedNormal.x * invLen,
                                        blendedNormal.y * invLen,
                                        blendedNormal.z * invLen
                                    };

                                    // 3. 색상 보간 (정수 오버플로우 방지 및 부드러운 전이)
                                    out[outIdx].color[0] = (uint8_t)(v0.color.x + mu * (static_cast<float>(v1_ptr->color.x) - v0.color.x));
                                    out[outIdx].color[1] = (uint8_t)(v0.color.y + mu * (static_cast<float>(v1_ptr->color.y) - v0.color.y));
                                    out[outIdx].color[2] = (uint8_t)(v0.color.z + mu * (static_cast<float>(v1_ptr->color.z) - v0.color.z));

                                    out[outIdx].weight = (float)v0.valueCount;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    void VoxelDataBase::Allocate(uint32_t maxBlocks)
    {
        maxBlockCount = maxBlocks;
        cudaMalloc(&d_blocks, sizeof(VoxelBlock) * maxBlockCount);
        cudaMemset(d_blocks, 0, sizeof(VoxelBlock) * maxBlockCount);
        cudaMalloc(&d_hashTable, sizeof(uint64_t) * maxBlockCount);
        cudaMemset(d_hashTable, 0, sizeof(uint64_t) * maxBlockCount);
        cudaMalloc(&d_blockCount, sizeof(uint32_t));
        cudaMemset(d_blockCount, 0, sizeof(uint32_t));
    }

    void VoxelDataBase::Free()
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

    void VoxelDataBase::OccupyVoxelFromPoints(const VVV::Matrix4f& rt, const VVV::Vector3f* points, const VVV::Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
    {
        VVV::Vector3f* d_p;
        VVV::Vector3b* d_c;
        cudaMalloc(&d_p, sizeof(VVV::Vector3f) * count);
        cudaMalloc(&d_c, sizeof(VVV::Vector3b) * count);

        cudaMemcpy(d_p, points, sizeof(VVV::Vector3f) * count, cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, colors, sizeof(VVV::Vector3b) * count, cudaMemcpyHostToDevice);

        CUDA_TS(VVV_InsertKernel);

        int threadsPerBlock = 256;
        int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
        InsertKernel<<<blocksPerGrid, threadsPerBlock>>>(*this, rt, d_p, d_c, count, blockSize, frameId);
        cudaDeviceSynchronize();

        CUDA_TE(VVV_InsertKernel);

        cudaFree(d_p);
        cudaFree(d_c);
    }

    void VoxelDataBase::IntegrateTSDF(const VVV::Matrix4f& rt, const VVV::Vector3f* d_points, const VVV::Vector3f* d_normals, const VVV::Vector3b* d_colors, uint32_t count, float blockSize, uint32_t frameId)
    {
        //VVV::Vector3f* d_p, * d_n;
        //VVV::Vector3b* d_c;
        //cudaMalloc(&d_p, sizeof(VVV::Vector3f) * count);
        //cudaMalloc(&d_n, sizeof(VVV::Vector3f) * count);
        //cudaMalloc(&d_c, sizeof(VVV::Vector3b) * count);

        //cudaMemcpy(d_p, points, sizeof(VVV::Vector3f) * count, cudaMemcpyHostToDevice);
        //cudaMemcpy(d_n, normals, sizeof(VVV::Vector3f) * count, cudaMemcpyHostToDevice);
        //cudaMemcpy(d_c, colors, sizeof(VVV::Vector3b) * count, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        TSDFIntegrateKernel<<<blocks, threads>>> (*this, rt, d_points, d_normals, d_colors, count, blockSize, frameId);
        cudaDeviceSynchronize();

        //cudaFree(d_p); cudaFree(d_n); cudaFree(d_c);
    }

    uint32_t VoxelDataBase::ExtractActiveVoxelsToHost(float blockSize, VVV::ExtractedVoxel* hostBuffer, uint32_t maxOut)
    {
        VVV::ExtractedVoxel* d_out;
        uint32_t* d_cnt;
        cudaMalloc(&d_out, sizeof(VVV::ExtractedVoxel) * maxOut);
        cudaMalloc(&d_cnt, sizeof(uint32_t));
        cudaMemset(d_cnt, 0, sizeof(uint32_t));

        int threadsPerBlock = 256;
        int blocksPerGrid = (maxBlockCount + threadsPerBlock - 1) / threadsPerBlock;
        ExtractKernel << <blocksPerGrid, threadsPerBlock >> > (*this, blockSize, d_out, d_cnt, maxOut);
        cudaDeviceSynchronize();

        uint32_t res;
        cudaMemcpy(&res, d_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        uint32_t copyAmt = (res > maxOut) ? maxOut : res;
        cudaMemcpy(hostBuffer, d_out, sizeof(VVV::ExtractedVoxel) * copyAmt, cudaMemcpyDeviceToHost);

        cudaFree(d_out);
        cudaFree(d_cnt);
        return res;
    }

    uint32_t VoxelDataBase::ExtractZeroCrossingVoxelsToHost(float blockSize, VVV::ExtractedVoxel* hostBuffer, uint32_t maxOut)
    {
        ExtractedVoxel* d_out;
        uint32_t* d_cnt;
        cudaMalloc(&d_out, sizeof(VVV::ExtractedVoxel) * maxOut);
        cudaMalloc(&d_cnt, sizeof(uint32_t));
        cudaMemset(d_cnt, 0, sizeof(uint32_t));

        int threadsPerBlock = 256;
        int blocksPerGrid = (maxBlockCount + threadsPerBlock - 1) / threadsPerBlock;
        ExtractZeroCrossingKernel << <blocksPerGrid, threadsPerBlock >> > (*this, blockSize, d_out, d_cnt, maxOut);
        cudaDeviceSynchronize();

        uint32_t res;
        cudaMemcpy(&res, d_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        uint32_t copyAmt = (res > maxOut) ? maxOut : res;
        cudaMemcpy(hostBuffer, d_out, sizeof(VVV::ExtractedVoxel) * copyAmt, cudaMemcpyDeviceToHost);

        cudaFree(d_out);
        cudaFree(d_cnt);
        return res;
    }
}
