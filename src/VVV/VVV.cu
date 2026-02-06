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

    __device__ inline unsigned short atomicAddUShort(unsigned short* address, unsigned short val)
    {
        unsigned int* base_address = (unsigned int*)((size_t)address & ~3);
        unsigned int selectors[] = { 0, 16 };
        unsigned int sel = selectors[((size_t)address & 2) >> 1];
        unsigned int old, assumed, sum, new_val;

        old = *base_address;

        do
        {
            assumed = old;
            unsigned short old_us = (unsigned short)((old >> sel) & 0xFFFF);
            unsigned short sum_us = old_us + val;

            if (old_us + val > 0xFFFF) sum_us = 0xFFFF;

            new_val = (old & ~(0xFFFF << sel)) | (sum_us << sel);
            old = atomicCAS(base_address, assumed, new_val);
        } while (assumed != old);

        return (unsigned short)((old >> sel) & 0xFFFF);
    }

    __device__ inline float GetVoxelValueSafe(VoxelDataBase& db, const Vector3f& pos)
    {
        Voxel* v = db.GetVoxel(pos);
        if (v != nullptr && v->valueCount > 0)
        {
            return v->value;
        }
        return NAN;
    }

    __device__ inline float atomicMinFloat(float* addr, float val)
    {
        int* addr_as_int = (int*)addr;
        int old = *addr_as_int, assumed;

        while (val < __int_as_float(old))
        {
            assumed = old;
            old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
            if (assumed == old)
            {
                break;
            }
        }
        return __int_as_float(old);
    }

    // ---------------------------------------------------------
    // Kernels
    // ---------------------------------------------------------

    __global__ void InsertKernel(VoxelDataBase db, VVV::Matrix4f rt, const Vector3f* points, const Vector3b* colors, uint32_t count, float blockSize, uint32_t frameId)
    {
        uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count)
        {
            return;
        }

        Vector3f p = rt.Transform(points[idx]);

        Voxel* v = db.GetOrCreateVoxel(p);

        if (v != nullptr)
        {
            atomicAdd(&(v->value), 1.0f);

            if (v->valueCount < 65535)
            {
                atomicAddUShort(&(v->valueCount), 1);
            }

            v->color = colors[idx];

            BlockID bid = db.GetOrCreateBlockSlot(p);
            if (bid != INVALID_BLOCK)
            {
                db.d_blocks[bid].lastTouchedFrameId = frameId;
            }
        }
    }

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

                // [수정됨] valueCount atomicAddUShort로 안전하게 증가
                unsigned short old_w_us = atomicAddUShort(&(voxel_ptr->valueCount), 1);

                float old_w = fminf((float)old_w_us, MAX_WEIGHT);
                float new_w = old_w + 1.0f;

                float weight_factor = old_w / new_w;
                float new_factor = 1.0f / new_w;

                // 1. TSDF 값 통합
                voxel_ptr->value = (voxel_ptr->value * weight_factor) + (dist * new_factor);

                // 2. Color 통합
                voxel_ptr->color.x = (uint8_t)((float)voxel_ptr->color.x * weight_factor + (float)color.x * new_factor + 0.5f);
                voxel_ptr->color.y = (uint8_t)((float)voxel_ptr->color.y * weight_factor + (float)color.y * new_factor + 0.5f);
                voxel_ptr->color.z = (uint8_t)((float)voxel_ptr->color.z * weight_factor + (float)color.z * new_factor + 0.5f);

                // 3. Normal 통합
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

    __global__ void ESDFIntegrateKernel(
        VoxelDataBase db,
        Matrix4f rt,
        const Vector3f* points,
        const Vector3f* normals,
        const Vector3b* colors,
        uint32_t count,
        float blockSize,
        uint32_t frameId)
    {
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= count)
        {
            return;
        }

        Vector3f p_world = rt.Transform(points[index]);
        Vector3f n_world = rt.TransformNormal(normals[index]);
        Vector3b color = colors[index];

        Vector3f cam_pos = { rt.data[12], rt.data[13], rt.data[14] };
        Vector3f ray_vec = { p_world.x - cam_pos.x, p_world.y - cam_pos.y, p_world.z - cam_pos.z };
        float ray_len = sqrtf(ray_vec.x * ray_vec.x + ray_vec.y * ray_vec.y + ray_vec.z * ray_vec.z);

        if (ray_len < 1e-6f)
        {
            return;
        }
        Vector3f ray_dir = { ray_vec.x / ray_len, ray_vec.y / ray_len, ray_vec.z / ray_len };

        float voxel_size = blockSize / 8.0f;
        float trunc_dist = voxel_size * 4.0f;

        float t_start = fmaxf(0.0f, ray_len - trunc_dist);
        float t_end = ray_len + trunc_dist;
        float step = voxel_size * 0.5f;

        for (float t = t_start; t <= t_end; t += step)
        {
            Vector3f sample_pos = {
                cam_pos.x + ray_dir.x * t,
                cam_pos.y + ray_dir.y * t,
                cam_pos.z + ray_dir.z * t
            };

            Voxel* voxel_ptr = db.GetOrCreateVoxel(sample_pos);
            if (voxel_ptr != nullptr)
            {
                float current_sdf = ray_len - t;
                float esdf_val = current_sdf;
                unsigned short current_w = voxel_ptr->valueCount;

                if (current_w == 0)
                {
                    if (atomicAddUShort(&(voxel_ptr->valueCount), 1) == 0)
                    {
                        voxel_ptr->value = esdf_val;
                        voxel_ptr->color = color;
                        voxel_ptr->normal = n_world;
                    }
                    else
                    {
                        goto UPDATE_VOXEL;
                    }
                }
                else if (current_sdf >= -trunc_dist)
                {
                UPDATE_VOXEL:

                    if (current_w > 5)
                    {
                        if (fabsf(voxel_ptr->value - esdf_val) > trunc_dist * 0.8f)
                        {
                            continue;
                        }
                    }

                    unsigned short old_w_us = atomicAddUShort(&(voxel_ptr->valueCount), 1);

                    float old_w = fminf((float)old_w_us, 15.0f);
                    float alpha = 1.0f / (old_w + 1.0f);

                    voxel_ptr->value = voxel_ptr->value * (1.0f - alpha) + esdf_val * alpha;

                    voxel_ptr->value = fmaxf(-trunc_dist, fminf(trunc_dist, voxel_ptr->value));

                    atomicMinFloat(&(voxel_ptr->value), esdf_val);

                    voxel_ptr->color.x = (uint8_t)((float)voxel_ptr->color.x * (1.0f - alpha) + (float)color.x * alpha + 0.5f);
                    voxel_ptr->color.y = (uint8_t)((float)voxel_ptr->color.y * (1.0f - alpha) + (float)color.y * alpha + 0.5f);
                    voxel_ptr->color.z = (uint8_t)((float)voxel_ptr->color.z * (1.0f - alpha) + (float)color.z * alpha + 0.5f);

                    voxel_ptr->normal.x = voxel_ptr->normal.x * (1.0f - alpha) + n_world.x * alpha;
                    voxel_ptr->normal.y = voxel_ptr->normal.y * (1.0f - alpha) + n_world.y * alpha;
                    voxel_ptr->normal.z = voxel_ptr->normal.z * (1.0f - alpha) + n_world.z * alpha;
                }

                VoxelBlock* block = db.GetVoxelBlock(sample_pos);
                if (block)
                {
                    block->lastTouchedFrameId = frameId;
                }
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
        float eps = voxelSize;

        for (int lz = 0; lz < 8; ++lz)
        {
            for (int ly = 0; ly < 8; ++ly)
            {
                for (int lx = 0; lx < 8; ++lx)
                {
                    int idx0 = (lz << 6) | (ly << 3) | lx;
                    Voxel& v0 = db.d_blocks[slot].voxels[idx0];

                    if (v0.valueCount < 5)
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

                        if (v1_ptr != nullptr && v1_ptr->valueCount >= 5)
                        {
                            float v0v = v0.value;
                            float v1v = v1_ptr->value;

                            if (v0v * v1v < 0.0f)
                            {
                                uint32_t outIdx = atomicAdd(count, 1);
                                if (outIdx < maxOut)
                                {
                                    float mu = -v0v / (v1v - v0v);
                                    mu = fminf(fmaxf(mu, 0.0f), 1.0f);

                                    Vector3f interpPos = {
                                        p0.x + mu * (p1.x - p0.x),
                                        p0.y + mu * (p1.y - p0.y),
                                        p0.z + mu * (p1.z - p0.z)
                                    };

                                    out[outIdx].position = interpPos;

                                    float val_x_p = GetVoxelValueSafe(db, { interpPos.x + eps, interpPos.y, interpPos.z });
                                    float val_x_m = GetVoxelValueSafe(db, { interpPos.x - eps, interpPos.y, interpPos.z });
                                    float val_y_p = GetVoxelValueSafe(db, { interpPos.x, interpPos.y + eps, interpPos.z });
                                    float val_y_m = GetVoxelValueSafe(db, { interpPos.x, interpPos.y - eps, interpPos.z });
                                    float val_z_p = GetVoxelValueSafe(db, { interpPos.x, interpPos.y, interpPos.z + eps });
                                    float val_z_m = GetVoxelValueSafe(db, { interpPos.x, interpPos.y, interpPos.z - eps });

                                    Vector3f finalNormal;
                                    bool hasGradient = !isnan(val_x_p) && !isnan(val_x_m) &&
                                        !isnan(val_y_p) && !isnan(val_y_m) &&
                                        !isnan(val_z_p) && !isnan(val_z_m);

                                    if (hasGradient)
                                    {
                                        Vector3f grad = { val_x_p - val_x_m, val_y_p - val_y_m, val_z_p - val_z_m };
                                        float len = sqrtf(grad.x * grad.x + grad.y * grad.y + grad.z * grad.z);
                                        if (len > 1e-6f)
                                        {
                                            finalNormal = { grad.x / len, grad.y / len, grad.z / len };
                                        }
                                        else
                                        {
                                            finalNormal = { 0, 1, 0 };
                                        }
                                    }
                                    else
                                    {
                                        Vector3f n0 = v0.normal;
                                        Vector3f n1 = v1_ptr->normal;
                                        Vector3f blended = {
                                            n0.x + mu * (n1.x - n0.x),
                                            n0.y + mu * (n1.y - n0.y),
                                            n0.z + mu * (n1.z - n0.z)
                                        };
                                        float len = sqrtf(blended.x * blended.x + blended.y * blended.y + blended.z * blended.z);
                                        if (len > 1e-6f)
                                            finalNormal = { blended.x / len, blended.y / len, blended.z / len };
                                        else
                                            finalNormal = n0;
                                    }

                                    out[outIdx].normal = finalNormal;

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

    // ---------------------------------------------------------
    // Host Functions
    // ---------------------------------------------------------

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
        if (d_blocks) cudaFree(d_blocks);
        if (d_hashTable) cudaFree(d_hashTable);
        if (d_blockCount) cudaFree(d_blockCount);
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
        InsertKernel << <blocksPerGrid, threadsPerBlock >> > (*this, rt, d_p, d_c, count, blockSize, frameId);
        cudaDeviceSynchronize();

        CUDA_TE(VVV_InsertKernel);

        cudaFree(d_p);
        cudaFree(d_c);
    }

    void VoxelDataBase::IntegrateTSDF(const VVV::Matrix4f& rt, const VVV::Vector3f* d_points, const VVV::Vector3f* d_normals, const VVV::Vector3b* d_colors, uint32_t count, float blockSize, uint32_t frameId)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        TSDFIntegrateKernel << <blocks, threads >> > (*this, rt, d_points, d_normals, d_colors, count, blockSize, frameId);
        cudaDeviceSynchronize();
    }

    void VoxelDataBase::IntegrateESDF(
        const Matrix4f& rt,
        const Vector3f* d_points,
        const Vector3f* d_normals,
        const Vector3b* d_colors,
        uint32_t count,
        float blockSize,
        uint32_t frameId)
    {
        int threads = 256;
        int blocks = (count + threads - 1) / threads;

        ESDFIntegrateKernel << <blocks, threads >> > (*this, rt, d_points, d_normals, d_colors, count, blockSize, frameId);
        cudaDeviceSynchronize();
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