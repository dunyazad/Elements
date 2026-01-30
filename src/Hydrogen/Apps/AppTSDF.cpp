#include "Apps.h"

#include <robin_hood/robin_hood.h>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <limits>

using VDBKey = uint64_t;
struct VDBBlockIndex { int x = 0; int y = 0; int z = 0; };

typedef struct CamInfo_
{
    float cfx;
    float cfy;
    float ccx;
    float ccy;
    int cx;
    int cy;
    int img_width;
    int img_height;
    double R[9];
    double T[3];
    Eigen::Vector3f dlpPos;
    Eigen::Vector3f camPos;
    Eigen::Matrix3f invMatTilt;
    Eigen::Matrix3f matTilt;

    Eigen::Matrix4f GetViewMatrix(const CamInfo_& info)
    {
        Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                view(i, j) = (float)info.R[i * 3 + j];
            }
            view(i, 3) = (float)info.T[i];
        }
        return view;
    }

    Eigen::Matrix4f GetProjectionMatrix(const CamInfo_& info, float n, float f)
    {
        Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
        proj(0, 0) = 2.0f * info.cfx / info.img_width;
        proj(0, 2) = 1.0f - (2.0f * info.ccx / info.img_width);
        proj(1, 1) = 2.0f * info.cfy / info.img_height;
        proj(1, 2) = (2.0f * info.ccy / info.img_height) - 1.0f;
        proj(2, 2) = -(f + n) / (f - n);
        proj(2, 3) = -(2.0f * f * n) / (f - n);
        proj(3, 2) = -1.0f;
        return proj;
    }
} CamInfo_;

struct VDBVoxel
{
    float tsdf = 0.0f;
    uint8_t weight = 0;
    Eigen::Vector3f normal = Eigen::Vector3f::Zero();
    Eigen::Vector3b color = Eigen::Vector3b::Zero();
};

template<int VoxelsPerBlockAxis = 8>
struct VDBVoxelBlock
{
    static constexpr int AXIS_COUNT = VoxelsPerBlockAxis;
    static constexpr int TOTAL_COUNT = AXIS_COUNT * AXIS_COUNT * AXIS_COUNT;

    VDBVoxel voxels[TOTAL_COUNT];

    static int ToLocalIndex(int x, int y, int z)
    {
        return x + (y * AXIS_COUNT) + (z * AXIS_COUNT * AXIS_COUNT);
    }
};

template<int VoxelsPerBlockAxis = 8>
struct VDB
{
    float voxelSize = 0.1f;
    float truncationDistance = 0.3f;
    uint8_t maxWeight = 255;

    robin_hood::unordered_map<VDBKey, size_t> blockMapping;
    std::vector<VDBVoxelBlock<VoxelsPerBlockAxis>> blocks;

    float GetBlockAxisSize() const
    {
        return voxelSize * VoxelsPerBlockAxis;
    }

    VDBKey ToKey(VDBBlockIndex index)
    {
        uint64_t ux = static_cast<uint64_t>(index.x) & 0x1FFFFF;
        uint64_t uy = static_cast<uint64_t>(index.y) & 0x1FFFFF;
        uint64_t uz = static_cast<uint64_t>(index.z) & 0x1FFFFF;

        return (ux << 42) | (uy << 21) | uz;
    }

    VDBBlockIndex FromKey(VDBKey key)
    {
        VDBBlockIndex index;
        int64_t ix = static_cast<int64_t>((key >> 42) & 0x1FFFFF);
        int64_t iy = static_cast<int64_t>((key >> 21) & 0x1FFFFF);
        int64_t iz = static_cast<int64_t>((key >> 0) & 0x1FFFFF);

        if (ix & 0x100000) ix |= ~0x1FFFFF;
        if (iy & 0x100000) iy |= ~0x1FFFFF;
        if (iz & 0x100000) iz |= ~0x1FFFFF;

        index.x = static_cast<int>(ix);
        index.y = static_cast<int>(iy);
        index.z = static_cast<int>(iz);

        return index;
    }

    void Integrate(const Eigen::Vector3f& sensorPosWorld, const Eigen::Vector3f& measuredPointWorld, const Eigen::Vector3f& normalWorld, const Eigen::Vector3b& colorRGB)
    {
        Eigen::Vector3f rayDir = measuredPointWorld - sensorPosWorld;
        float measuredDepth = rayDir.norm();

        if (measuredDepth < 1e-4f)
        {
            return;
        }

        rayDir /= measuredDepth;

        const float startDist = std::max(0.0f, measuredDepth - truncationDistance);
        const float endDist = measuredDepth + truncationDistance;
        const float invTruncation = 1.0f / truncationDistance;
        const float invBlockSize = 1.0f / GetBlockAxisSize();
        const float invVoxelSize = 1.0f / voxelSize;

        VDBKey lastKey = 0xFFFFFFFFFFFFFFFF;
        VDBVoxelBlock<VoxelsPerBlockAxis>* currentBlock = nullptr;
        VDBBlockIndex lastBIdx = { -999999, -999999, -999999 };

        const float stepSize = voxelSize * 0.4f;

        for (float dist = startDist; dist <= endDist; dist += stepSize)
        {
            Eigen::Vector3f currentPos = sensorPosWorld + rayDir * dist;

            VDBBlockIndex bIdx;
            bIdx.x = static_cast<int>(std::floor(currentPos.x() * invBlockSize));
            bIdx.y = static_cast<int>(std::floor(currentPos.y() * invBlockSize));
            bIdx.z = static_cast<int>(std::floor(currentPos.z() * invBlockSize));

            VDBKey key = ToKey(bIdx);
            if (key != lastKey)
            {
                auto it = blockMapping.find(key);
                if (it == blockMapping.end())
                {
                    size_t newIdx = blocks.size();
                    blockMapping[key] = newIdx;
                    blocks.emplace_back();
                    currentBlock = &blocks.back();
                }
                else
                {
                    currentBlock = &blocks[it->second];
                }
                lastKey = key;
                lastBIdx = bIdx;
            }

            int lx = static_cast<int>(std::floor(currentPos.x() * invVoxelSize)) - (lastBIdx.x * VoxelsPerBlockAxis);
            int ly = static_cast<int>(std::floor(currentPos.y() * invVoxelSize)) - (lastBIdx.y * VoxelsPerBlockAxis);
            int lz = static_cast<int>(std::floor(currentPos.z() * invVoxelSize)) - (lastBIdx.z * VoxelsPerBlockAxis);

            if (lx >= 0 && lx < VoxelsPerBlockAxis && ly >= 0 && ly < VoxelsPerBlockAxis && lz >= 0 && lz < VoxelsPerBlockAxis)
            {
                VDBVoxel& voxel = currentBlock->voxels[VDBVoxelBlock<VoxelsPerBlockAxis>::ToLocalIndex(lx, ly, lz)];

                float sdf = measuredDepth - dist;
                float tsdf = std::clamp(sdf * invTruncation, -1.0f, 1.0f);

                float oldWeight = (float)voxel.weight;
                float absTsdf = std::abs(tsdf);
                float newWeight = std::exp(-2.0f * absTsdf);
                float totalWeight = oldWeight + newWeight;

                voxel.tsdf = (voxel.tsdf * oldWeight + tsdf * newWeight) / totalWeight;
                if (absTsdf < 0.5f)
                {
                    voxel.normal = (voxel.normal * oldWeight + normalWorld * newWeight).normalized();
                    Eigen::Vector3f fusedColor = voxel.color.cast<float>() * oldWeight + colorRGB.cast<float>() * newWeight;
                    voxel.color = (fusedColor / totalWeight).cast<uint8_t>();
                }
                voxel.weight = static_cast<uint8_t>(std::min((float)maxWeight, totalWeight));
            }
        }
    }

    void Visualize(const std::string& blockTag = "VDB_Blocks", const std::string& voxelTag = "VDB_Voxels")
    {
        float bSize = GetBlockAxisSize();
        Eigen::Vector3f bExtent(bSize, bSize, bSize);
        Eigen::Vector3f vExtent(voxelSize, voxelSize, voxelSize);

        std::vector<Eigen::Vector3f> blockCenters;
        std::vector<Eigen::Vector3f> voxelCenters;
        std::vector<Eigen::Vector4f> voxelColors;

        for (auto const& [key, index] : blockMapping)
        {
            VDBBlockIndex bIdx = FromKey(key);
            Eigen::Vector3f bMin(bIdx.x * bSize, bIdx.y * bSize, bIdx.z * bSize);
            blockCenters.push_back(bMin + bExtent * 0.5f);

            const VDBVoxelBlock<VoxelsPerBlockAxis>& block = blocks[index];

            for (int z = 0; z < VoxelsPerBlockAxis; ++z)
            {
                for (int y = 0; y < VoxelsPerBlockAxis; ++y)
                {
                    for (int x = 0; x < VoxelsPerBlockAxis; ++x)
                    {
                        const VDBVoxel& voxel = block.voxels[VDBVoxelBlock<VoxelsPerBlockAxis>::ToLocalIndex(x, y, z)];

                        if (voxel.weight > 0)
                        {
                            Eigen::Vector3f vMin = bMin + Eigen::Vector3f(x * voxelSize, y * voxelSize, z * voxelSize);
                            voxelCenters.push_back(vMin + vExtent * 0.5f);

                            Eigen::Vector4f color;
                            if (std::abs(voxel.tsdf) < 0.1f)
                            {
                                color = Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f);
                            }
                            else
                            {
                                color = Eigen::Vector4f(voxel.color.x() / 255.0f, voxel.color.y() / 255.0f, voxel.color.z() / 255.0f, 1.0f);
                            }
                            voxelColors.push_back(color);
                        }
                    }
                }
            }
        }

        VisualDebugging::AddWiredBoxBatch(blockTag, blockCenters, bExtent, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
        VisualDebugging::AddWiredBoxBatch(voxelTag, voxelCenters, vExtent, voxelColors);
    }

    void VisualizeZeroCrossing(const std::string& tag = "Zero_Crossings", float threshold = 0.05f, float minWeight = 10.0f)
    {
        float bSize = GetBlockAxisSize();
        std::vector<Eigen::Vector3f> positions, normals;
        std::vector<Eigen::Vector4f> colors;

        for (auto const& [key, index] : blockMapping)
        {
            VDBBlockIndex bIdx = FromKey(key);
            Eigen::Vector3f bMin(bIdx.x * bSize, bIdx.y * bSize, bIdx.z * bSize);
            const VDBVoxelBlock<VoxelsPerBlockAxis>& block = blocks[index];
            for (int z = 0; z < VoxelsPerBlockAxis; ++z)
            {
                for (int y = 0; y < VoxelsPerBlockAxis; ++y)
                {
                    for (int x = 0; x < VoxelsPerBlockAxis; ++x)
                    {
                        const VDBVoxel& voxel = block.voxels[VDBVoxelBlock<VoxelsPerBlockAxis>::ToLocalIndex(x, y, z)];

                        if (voxel.weight >= minWeight && std::abs(voxel.tsdf) < threshold)
                        {
                            positions.push_back(bMin + Eigen::Vector3f((x + 0.5f) * voxelSize, (y + 0.5f) * voxelSize, (z + 0.5f) * voxelSize));
                            normals.push_back(voxel.normal);
                            colors.push_back(Eigen::Vector4f(voxel.color.x() / 255.f, voxel.color.y() / 255.f, voxel.color.z() / 255.f, 1.f));
                        }
                    }
                }
            }
        }
        //VisualDebugging::AddDiskBatch(tag, positions, normals, voxelSize * 0.7f, 8, colors, false);
        VisualDebugging::AddSphereBatch(tag, positions, voxelSize * 0.7f, colors);
    }
};

class AppTSDF : public App
{
public:
    virtual void Execute() override
    {
        VDB<8> vdb;
        vdb.voxelSize = 0.05f;
        vdb.truncationDistance = 0.1f;

        std::ifstream ifs("D:\\Resources\\Debug\\3D\\Patches.bin", std::ios::binary);
        if (!ifs.is_open())
        {
            return;
        }

        CamInfo_ cam;
        Eigen::Matrix4f camRT;
        ifs.read(reinterpret_cast<char*>(&cam), sizeof(CamInfo_));
        ifs.read(reinterpret_cast<char*>(camRT.data()), sizeof(float) * 16);

        size_t numberOfPatches = 0;
        ifs.read(reinterpret_cast<char*>(&numberOfPatches), sizeof(size_t));

        for (size_t i = 0; i < numberOfPatches; i++)
        {
            TS(patch);
            size_t patchIndex = 0;
            ifs.read(reinterpret_cast<char*>(&patchIndex), sizeof(size_t));
            Eigen::Matrix4f rt0;
            ifs.read(reinterpret_cast<char*>(rt0.data()), sizeof(float) * 16);
            ifs.seekg(sizeof(float) * 16, std::ios::cur);

            size_t numPts = 0;
            ifs.read(reinterpret_cast<char*>(&numPts), sizeof(size_t));
            if (numPts > 10000000)
            {
                continue;
            }

            std::vector<Eigen::Vector3f> pts(numPts), normals(numPts);
            std::vector<Eigen::Vector3b> colors(numPts);
            ifs.read(reinterpret_cast<char*>(pts.data()), sizeof(Eigen::Vector3f) * numPts);
            ifs.read(reinterpret_cast<char*>(normals.data()), sizeof(Eigen::Vector3f) * numPts);
            ifs.read(reinterpret_cast<char*>(colors.data()), sizeof(Eigen::Vector3b) * numPts);

            size_t numPts45 = 0;
            ifs.read(reinterpret_cast<char*>(&numPts45), sizeof(size_t));
            ifs.seekg(numPts45 * (sizeof(Eigen::Vector3f) * 2), std::ios::cur);

            Eigen::Vector3f sensorPos = rt0.block<3, 1>(0, 3);
            Eigen::Matrix3f rot = rt0.block<3, 3>(0, 0);

            for (size_t j = 0; j < numPts; ++j)
            {
                Eigen::Vector3f pW = rot * pts[j] + sensorPos;
                Eigen::Vector3f nW = (rot * normals[j]).normalized();
                vdb.Integrate(sensorPos, pW, nW, colors[j]);
            }
            TE(patch);

            break;

        }
        //VisualDebugging::ClearAll();
        //vdb.Visualize();
        vdb.VisualizeZeroCrossing("Zero_Crossings", 0.005f, 0.005f);
    }
};

REGISTER_APP(AppTSDF, "AppTSDF");