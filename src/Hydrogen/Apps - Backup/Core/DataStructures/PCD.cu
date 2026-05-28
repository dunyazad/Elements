#include <Core/DataStructures/PCD.h>
#include <Core/DataStructures/SparseCells.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/partition.h>
#include <thrust/scan.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

namespace Huvitz
{
    struct DuplicateFloat3
    {
        __host__ __device__
            thrust::tuple<float3, float3> operator()(const float3& p) const
        {
            return thrust::make_tuple(p, p);
        }
    };

    struct MinMaxReducer
    {
        __host__ __device__
            thrust::tuple<float3, float3> operator()(
                const thrust::tuple<float3, float3>& a,
                const thrust::tuple<float3, float3>& b) const
        {
            return thrust::make_tuple(
                make_float3(
                    fminf(thrust::get<0>(a).x, thrust::get<0>(b).x),
                    fminf(thrust::get<0>(a).y, thrust::get<0>(b).y),
                    fminf(thrust::get<0>(a).z, thrust::get<0>(b).z)),
                make_float3(
                    fmaxf(thrust::get<1>(a).x, thrust::get<1>(b).x),
                    fmaxf(thrust::get<1>(a).y, thrust::get<1>(b).y),
                    fmaxf(thrust::get<1>(a).z, thrust::get<1>(b).z)));
        }
    };

    struct ValidPositionPred4
    {
        __host__ __device__
            bool operator()(
                const thrust::tuple<float3, float3, uchar3, bool>& t) const
        {
            const float3& p = thrust::get<0>(t);
            return p.x != FLT_MAX && p.y != FLT_MAX && p.z != FLT_MAX;
        }
    };

    struct InsideAABBMask
    {
        float3 bmin, bmax;

        __host__ __device__
            bool operator()(const thrust::tuple<float3, bool>& t) const
        {
            if (!thrust::get<1>(t)) return false;
            const float3& p = thrust::get<0>(t);
            return p.x >= bmin.x && p.x <= bmax.x
                && p.y >= bmin.y && p.y <= bmax.y
                && p.z >= bmin.z && p.z <= bmax.z;
        }
    };

    struct TransformAABBMask
    {
        float  m[16];
        cuAABB aabb;

        __host__ __device__
            bool operator()(const thrust::tuple<float3, bool>& t) const
        {
            if (!thrust::get<1>(t)) return false;

            const float3& p = thrust::get<0>(t);
            if (p.x == FLT_MAX && p.y == FLT_MAX && p.z == FLT_MAX) return false;

            const float lx = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
            const float ly = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
            const float lz = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];

            return lx >= aabb.min.x && lx <= aabb.max.x
                && ly >= aabb.min.y && ly <= aabb.max.y
                && lz >= aabb.min.z && lz <= aabb.max.z;
        }
    };

    struct TransformAndAABBFilter
    {
        float  m[16];
        cuAABB aabb;

        __host__ __device__
            float3 operator()(const float3& p) const
        {
            if (p.x == FLT_MAX && p.y == FLT_MAX && p.z == FLT_MAX)
                return p;

            const float localX = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
            const float localY = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
            const float localZ = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];

            if (localX >= aabb.min.x && localX <= aabb.max.x &&
                localY >= aabb.min.y && localY <= aabb.max.y &&
                localZ >= aabb.min.z && localZ <= aabb.max.z)
            {
                return p;
            }

            return make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        }
    };

    struct PointCountLookup
    {
        const unsigned int* lut;
        __host__ __device__
            unsigned int operator()(unsigned int label) const { return lut[label]; }
    };

    void ComputeAABBSinglePass(
        const thrust::device_vector<float3>& pos,
        size_t count,
        float3& outMin,
        float3& outMax)
    {
        auto t_begin = thrust::make_transform_iterator(pos.begin(), DuplicateFloat3{});
        const auto init = thrust::make_tuple(
            make_float3(FLT_MAX, FLT_MAX, FLT_MAX),
            make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX));
        auto result = thrust::reduce(t_begin, t_begin + count, init, MinMaxReducer{});
        outMin = thrust::get<0>(result);
        outMax = thrust::get<1>(result);
    }

    template<typename Policy>
    void ComputeAABBSinglePassOnStream(
        Policy policy,
        const thrust::device_vector<float3>& pos,
        size_t count,
        float3& outMin,
        float3& outMax)
    {
        auto t_begin = thrust::make_transform_iterator(pos.begin(), DuplicateFloat3{});
        const auto init = thrust::make_tuple(
            make_float3(FLT_MAX, FLT_MAX, FLT_MAX),
            make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX));
        auto result = thrust::reduce(policy, t_begin, t_begin + count, init, MinMaxReducer{});
        outMin = thrust::get<0>(result);
        outMax = thrust::get<1>(result);
    }

    static void BuildCountTableImpl(
        const thrust::device_vector<unsigned int>& src,
        size_t count,
        thrust::device_vector<unsigned int>& outCountTable)
    {
        const unsigned int maxLabel = thrust::reduce(
            src.begin(), src.begin() + count,
            0u, thrust::maximum<unsigned int>());

        outCountTable.assign(maxLabel + 1, 0u);

        thrust::device_vector<unsigned int> d_sorted(src.begin(), src.begin() + count);
        thrust::sort(d_sorted.begin(), d_sorted.end());

        thrust::device_vector<unsigned int> d_keys(count);
        thrust::device_vector<unsigned int> d_cnts(count);

        auto end_pair = thrust::reduce_by_key(
            d_sorted.begin(), d_sorted.end(),
            thrust::make_constant_iterator(1u),
            d_keys.begin(),
            d_cnts.begin());

        const size_t numLabels = static_cast<size_t>(end_pair.first - d_keys.begin());

        thrust::scatter(
            d_cnts.begin(), d_cnts.begin() + numLabels,
            d_keys.begin(),
            outCountTable.begin());
    }

    template<typename Policy>
    static void BuildCountTableOnStreamImpl(
        Policy policy,
        const thrust::device_vector<unsigned int>& src,
        size_t count,
        thrust::device_vector<unsigned int>& outCountTable)
    {
        const unsigned int maxLabel = thrust::reduce(
            policy,
            src.begin(), src.begin() + count,
            0u, thrust::maximum<unsigned int>());

        outCountTable.assign(maxLabel + 1, 0u);

        thrust::device_vector<unsigned int> d_sorted(src.begin(), src.begin() + count);
        thrust::sort(policy, d_sorted.begin(), d_sorted.end());

        thrust::device_vector<unsigned int> d_keys(count);
        thrust::device_vector<unsigned int> d_cnts(count);

        auto end_pair = thrust::reduce_by_key(
            policy,
            d_sorted.begin(), d_sorted.end(),
            thrust::make_constant_iterator(1u),
            d_keys.begin(),
            d_cnts.begin());

        const size_t numLabels = static_cast<size_t>(end_pair.first - d_keys.begin());

        thrust::scatter(
            policy,
            d_cnts.begin(), d_cnts.begin() + numLabels,
            d_keys.begin(),
            outCountTable.begin());
    }

    static void BuildPointCountTableImpl(
        const thrust::device_vector<unsigned int>& src,
        size_t count,
        thrust::device_vector<unsigned int>& outPointCountTable)
    {
        const unsigned int maxLabel = thrust::reduce(
            src.begin(), src.begin() + count,
            0u, thrust::maximum<unsigned int>());

        thrust::device_vector<unsigned int> d_lut(maxLabel + 1, 0u);

        thrust::device_vector<unsigned int> d_sorted(src.begin(), src.begin() + count);
        thrust::sort(d_sorted.begin(), d_sorted.end());

        thrust::device_vector<unsigned int> d_keys(count);
        thrust::device_vector<unsigned int> d_cnts(count);

        auto end_pair = thrust::reduce_by_key(
            d_sorted.begin(), d_sorted.end(),
            thrust::make_constant_iterator(1u),
            d_keys.begin(),
            d_cnts.begin());

        const size_t numLabels = static_cast<size_t>(end_pair.first - d_keys.begin());
        thrust::scatter(
            d_cnts.begin(), d_cnts.begin() + numLabels,
            d_keys.begin(),
            d_lut.begin());

        outPointCountTable.resize(count);
        thrust::transform(
            src.begin(), src.begin() + count,
            outPointCountTable.begin(),
            PointCountLookup{ thrust::raw_pointer_cast(d_lut.data()) });
    }

    template<typename Policy>
    static void BuildPointCountTableOnStreamImpl(
        Policy policy,
        const thrust::device_vector<unsigned int>& src,
        size_t count,
        thrust::device_vector<unsigned int>& outPointCountTable)
    {
        const unsigned int maxLabel = thrust::reduce(
            policy,
            src.begin(), src.begin() + count,
            0u, thrust::maximum<unsigned int>());

        thrust::device_vector<unsigned int> d_lut(maxLabel + 1, 0u);

        thrust::device_vector<unsigned int> d_sorted(src.begin(), src.begin() + count);
        thrust::sort(policy, d_sorted.begin(), d_sorted.end());

        thrust::device_vector<unsigned int> d_keys(count);
        thrust::device_vector<unsigned int> d_cnts(count);

        auto end_pair = thrust::reduce_by_key(
            policy,
            d_sorted.begin(), d_sorted.end(),
            thrust::make_constant_iterator(1u),
            d_keys.begin(),
            d_cnts.begin());

        const size_t numLabels = static_cast<size_t>(end_pair.first - d_keys.begin());
        thrust::scatter(
            policy,
            d_cnts.begin(), d_cnts.begin() + numLabels,
            d_keys.begin(),
            d_lut.begin());

        outPointCountTable.resize(count);
        thrust::transform(
            policy,
            src.begin(), src.begin() + count,
            outPointCountTable.begin(),
            PointCountLookup{ thrust::raw_pointer_cast(d_lut.data()) });
    }

#pragma region PCD

    PCD::PCD() {}

    PCD::PCD(size_t n) { resize(n); }

    PCD::~PCD() {}

    size_t PCD::size()     const { return numberOfPositions; }
    size_t PCD::capacity() const { return allocated; }

    void PCD::resize(size_t n)
    {
        if (allocated >= n) return;

        try
        {
            positions.resize(n);
            normals.resize(n);
            colors.resize(n);
            isAlive.resize(n);
            labels.resize(n);
            subLabels.resize(n);
            allocated = n;
        }
        catch (const std::exception& e)
        {
            printf("[PCD::resize] Exception caught! Requested size: %zu, Error: %s\n",
                n, e.what());
            throw;
        }
    }

    void PCD::clear()
    {
        thrust::device_vector<float3>().swap(positions);
        thrust::device_vector<float3>().swap(normals);
        thrust::device_vector<uchar3>().swap(colors);
        thrust::device_vector<bool>().swap(isAlive);
        thrust::device_vector<unsigned int>().swap(labels);
        thrust::device_vector<unsigned int>().swap(subLabels);
        thrust::device_vector<unsigned int>().swap(labelCountTable);
        thrust::device_vector<unsigned int>().swap(pointCountTable);
        thrust::device_vector<unsigned int>().swap(subLabelCountTable);
        thrust::device_vector<unsigned int>().swap(subLabelPointCountTable);
        allocated = 0;
        numberOfPositions = 0;
    }

    void PCD::FromHostVectors(
        const std::vector<float3>& h_positions,
        const std::vector<float3>& h_normals,
        const std::vector<uchar3>& h_colors)
    {
        const size_t n = h_positions.size();
        resize(n);

        if (!h_positions.empty())
        {
            thrust::copy(h_positions.begin(), h_positions.end(), positions.begin());
            ComputeAABBSinglePass(positions, n, aabb.min, aabb.max);
        }

        if (!h_normals.empty())
            thrust::copy(h_normals.begin(), h_normals.end(), normals.begin());
        else
            thrust::fill(normals.begin(), normals.begin() + n, make_float3(0.0f, 0.0f, 1.0f));

        if (!h_colors.empty())
            thrust::copy(h_colors.begin(), h_colors.end(), colors.begin());
        else
            thrust::fill(colors.begin(), colors.begin() + n, make_uchar3(255, 255, 255));

        thrust::fill(isAlive.begin(), isAlive.begin() + n, true);
        numberOfPositions = n;
        FilterInvalidPositions();
    }

    void PCD::FromHostVectors(
        const std::vector<float3>& h_positions,
        const std::vector<float3>& h_normals,
        const std::vector<uchar3>& h_colors,
        const float3& h_aabbMin,
        const float3& h_aabbMax)
    {
        const size_t n = h_positions.size();
        resize(n);

        if (!h_positions.empty())
            thrust::copy(h_positions.begin(), h_positions.end(), positions.begin());

        if (!h_normals.empty())
            thrust::copy(h_normals.begin(), h_normals.end(), normals.begin());
        else
            thrust::fill(normals.begin(), normals.begin() + n, make_float3(0.0f, 0.0f, 1.0f));

        if (!h_colors.empty())
            thrust::copy(h_colors.begin(), h_colors.end(), colors.begin());
        else
            thrust::fill(colors.begin(), colors.begin() + n, make_uchar3(255, 255, 255));

        thrust::fill(isAlive.begin(), isAlive.begin() + n, true);
        aabb.min = h_aabbMin;
        aabb.max = h_aabbMax;
        numberOfPositions = n;
        FilterInvalidPositions();
    }

    void PCD::FromHostVectors(
        const std::vector<float3>& h_positions,
        const std::vector<float3>& h_normals,
        const std::vector<float4>& h_colors)
    {
        const size_t n = h_positions.size();
        resize(n);

        if (!h_positions.empty())
        {
            thrust::copy(h_positions.begin(), h_positions.end(), positions.begin());
            ComputeAABBSinglePass(positions, n, aabb.min, aabb.max);
        }

        if (!h_normals.empty())
            thrust::copy(h_normals.begin(), h_normals.end(), normals.begin());
        else
            thrust::fill(normals.begin(), normals.begin() + n, make_float3(0.0f, 0.0f, 1.0f));

        if (!h_colors.empty())
        {
            thrust::device_vector<float4> d_temp(h_colors.begin(), h_colors.end());
            thrust::transform(d_temp.begin(), d_temp.end(), colors.begin(), Float4ToUChar3());
        }
        else
        {
            thrust::fill(colors.begin(), colors.begin() + n, make_uchar3(255, 255, 255));
        }

        thrust::fill(isAlive.begin(), isAlive.begin() + n, true);
        numberOfPositions = n;
        FilterInvalidPositions();
    }

    void PCD::FromHostVectors(
        const std::vector<float3>& h_positions,
        const std::vector<float3>& h_normals,
        const std::vector<float4>& h_colors,
        const float3& h_aabbMin,
        const float3& h_aabbMax)
    {
        const size_t n = h_positions.size();
        resize(n);

        if (!h_positions.empty())
            thrust::copy(h_positions.begin(), h_positions.end(), positions.begin());

        if (!h_normals.empty())
            thrust::copy(h_normals.begin(), h_normals.end(), normals.begin());
        else
            thrust::fill(normals.begin(), normals.begin() + n, make_float3(0.0f, 0.0f, 1.0f));

        if (!h_colors.empty())
        {
            thrust::device_vector<float4> d_temp(h_colors.begin(), h_colors.end());
            thrust::transform(d_temp.begin(), d_temp.end(), colors.begin(), Float4ToUChar3());
        }
        else
        {
            thrust::fill(colors.begin(), colors.begin() + n, make_uchar3(255, 255, 255));
        }

        thrust::fill(isAlive.begin(), isAlive.begin() + n, true);
        aabb.min = h_aabbMin;
        aabb.max = h_aabbMax;
        numberOfPositions = n;
        FilterInvalidPositions();
    }

    void PCD::FromHostPointers(
        const float3* h_positions,
        const float3* h_normals,
        const uchar3* h_colors,
        size_t numberOfPositions)
    {
        resize(numberOfPositions);

        if (h_positions)
        {
            thrust::copy(h_positions, h_positions + numberOfPositions, positions.begin());
            ComputeAABBSinglePass(positions, numberOfPositions, aabb.min, aabb.max);
        }

        if (h_normals)
            thrust::copy(h_normals, h_normals + numberOfPositions, normals.begin());
        else
            thrust::fill(normals.begin(), normals.begin() + numberOfPositions, make_float3(0.0f, 0.0f, 1.0f));

        if (h_colors)
            thrust::copy(h_colors, h_colors + numberOfPositions, colors.begin());
        else
            thrust::fill(colors.begin(), colors.begin() + numberOfPositions, make_uchar3(255, 255, 255));

        thrust::fill(isAlive.begin(), isAlive.begin() + numberOfPositions, true);
        this->numberOfPositions = numberOfPositions;
        FilterInvalidPositions();
    }

    void PCD::FromHostPointers(
        const float3* h_positions,
        const float3* h_normals,
        const uchar3* h_colors,
        size_t numberOfPositions,
        const float3& h_aabbMin,
        const float3& h_aabbMax)
    {
        resize(numberOfPositions);

        if (h_positions)
            thrust::copy(h_positions, h_positions + numberOfPositions, positions.begin());

        if (h_normals)
            thrust::copy(h_normals, h_normals + numberOfPositions, normals.begin());
        else
            thrust::fill(normals.begin(), normals.begin() + numberOfPositions, make_float3(0.0f, 0.0f, 1.0f));

        if (h_colors)
            thrust::copy(h_colors, h_colors + numberOfPositions, colors.begin());
        else
            thrust::fill(colors.begin(), colors.begin() + numberOfPositions, make_uchar3(255, 255, 255));

        thrust::fill(isAlive.begin(), isAlive.begin() + numberOfPositions, true);
        aabb.min = h_aabbMin;
        aabb.max = h_aabbMax;
        this->numberOfPositions = numberOfPositions;
        FilterInvalidPositions();
    }

    void PCD::FromHostPointers(
        const float3* h_positions,
        const float3* h_normals,
        const float4* h_colors,
        size_t numberOfPositions)
    {
        resize(numberOfPositions);

        if (h_positions)
        {
            thrust::copy(h_positions, h_positions + numberOfPositions, positions.begin());
            ComputeAABBSinglePass(positions, numberOfPositions, aabb.min, aabb.max);
        }

        if (h_normals)
            thrust::copy(h_normals, h_normals + numberOfPositions, normals.begin());
        else
            thrust::fill(normals.begin(), normals.begin() + numberOfPositions, make_float3(0.0f, 0.0f, 1.0f));

        if (h_colors)
        {
            thrust::device_vector<float4> d_temp(h_colors, h_colors + numberOfPositions);
            thrust::transform(d_temp.begin(), d_temp.end(), colors.begin(), Float4ToUChar3());
        }
        else
        {
            thrust::fill(colors.begin(), colors.begin() + numberOfPositions, make_uchar3(255, 255, 255));
        }

        thrust::fill(isAlive.begin(), isAlive.begin() + numberOfPositions, true);
        this->numberOfPositions = numberOfPositions;
        FilterInvalidPositions();
    }

    void PCD::FromHostPointers(
        const float3* h_positions,
        const float3* h_normals,
        const float4* h_colors,
        size_t numberOfPositions,
        const float3& h_aabbMin,
        const float3& h_aabbMax)
    {
        resize(numberOfPositions);

        if (h_positions)
            thrust::copy(h_positions, h_positions + numberOfPositions, positions.begin());

        if (h_normals)
            thrust::copy(h_normals, h_normals + numberOfPositions, normals.begin());
        else
            thrust::fill(normals.begin(), normals.begin() + numberOfPositions, make_float3(0.0f, 0.0f, 1.0f));

        if (h_colors)
        {
            thrust::device_vector<float4> d_temp(h_colors, h_colors + numberOfPositions);
            thrust::transform(d_temp.begin(), d_temp.end(), colors.begin(), Float4ToUChar3());
        }
        else
        {
            thrust::fill(colors.begin(), colors.begin() + numberOfPositions, make_uchar3(255, 255, 255));
        }

        thrust::fill(isAlive.begin(), isAlive.begin() + numberOfPositions, true);
        aabb.min = h_aabbMin;
        aabb.max = h_aabbMax;
        this->numberOfPositions = numberOfPositions;
        FilterInvalidPositions();
    }

    void PCD::FromDevicePointers(
        const float3* d_positions,
        const float3* d_normals,
        const uchar3* d_colors,
        size_t numberOfPositions,
        CUstream_st* stream,
        [[maybe_unused]] cached_allocator* alloc)
    {
        resize(numberOfPositions);
        auto policy = thrust::cuda::par.on(stream);

        if (d_positions)
        {
            thrust::copy(policy, d_positions, d_positions + numberOfPositions, positions.begin());
            if (numberOfPositions > 0)
                ComputeAABBSinglePassOnStream(policy, positions, numberOfPositions, aabb.min, aabb.max);
        }

        if (d_normals)
            thrust::copy(policy, d_normals, d_normals + numberOfPositions, normals.begin());
        else
            thrust::fill(policy, normals.begin(), normals.begin() + numberOfPositions, make_float3(0.0f, 0.0f, 1.0f));

        if (d_colors)
            thrust::copy(policy, d_colors, d_colors + numberOfPositions, colors.begin());
        else
            thrust::fill(policy, colors.begin(), colors.begin() + numberOfPositions, make_uchar3(255, 255, 255));

        thrust::fill(policy, isAlive.begin(), isAlive.begin() + numberOfPositions, true);
        this->numberOfPositions = numberOfPositions;

        cudaStreamSynchronize(stream);
        FilterInvalidPositions();
    }

    void PCD::FromDevicePointers(
        const float3* d_positions,
        const float3* d_normals,
        const float4* d_colors,
        size_t numberOfPositions,
        CUstream_st* stream,
        [[maybe_unused]] cached_allocator* alloc)
    {
        resize(numberOfPositions);
        auto policy = thrust::cuda::par.on(stream);

        if (d_positions)
        {
            thrust::copy(policy, d_positions, d_positions + numberOfPositions, positions.begin());
            if (numberOfPositions > 0)
                ComputeAABBSinglePassOnStream(policy, positions, numberOfPositions, aabb.min, aabb.max);
        }

        if (d_normals)
            thrust::copy(policy, d_normals, d_normals + numberOfPositions, normals.begin());
        else
            thrust::fill(policy, normals.begin(), normals.begin() + numberOfPositions, make_float3(0.0f, 0.0f, 1.0f));

        if (d_colors)
            thrust::transform(policy, d_colors, d_colors + numberOfPositions, colors.begin(), Float4ToUChar3());
        else
            thrust::fill(policy, colors.begin(), colors.begin() + numberOfPositions, make_uchar3(255, 255, 255));

        thrust::fill(policy, isAlive.begin(), isAlive.begin() + numberOfPositions, true);
        this->numberOfPositions = numberOfPositions;

        cudaStreamSynchronize(stream);
        FilterInvalidPositions();
    }

    void PCD::FromDevicePointers(
        const float3* d_positions,
        const float3* d_normals,
        const uchar3* d_colors,
        size_t numberOfPositions,
        const cuAABB& localAABB,
        const float* inverseRT,
        CUstream_st* stream,
        [[maybe_unused]] cached_allocator* alloc)
    {
        resize(numberOfPositions);
        auto policy = thrust::cuda::par.on(stream);

        if (d_positions)
        {
            thrust::copy(policy, d_positions, d_positions + numberOfPositions, positions.begin());

            if (numberOfPositions > 0)
            {
                TransformAndAABBFilter filter;
                for (int i = 0; i < 16; ++i)
                    filter.m[i] = inverseRT[i];
                filter.aabb = localAABB;

                thrust::transform(
                    policy,
                    positions.begin(), positions.begin() + numberOfPositions,
                    positions.begin(), filter);
            }
        }

        if (d_normals)
            thrust::copy(policy, d_normals, d_normals + numberOfPositions, normals.begin());
        else
            thrust::fill(policy, normals.begin(), normals.begin() + numberOfPositions, make_float3(0.0f, 0.0f, 1.0f));

        if (d_colors)
            thrust::copy(policy, d_colors, d_colors + numberOfPositions, colors.begin());
        else
            thrust::fill(policy, colors.begin(), colors.begin() + numberOfPositions, make_uchar3(255, 255, 255));

        thrust::fill(policy, isAlive.begin(), isAlive.begin() + numberOfPositions, true);
        this->numberOfPositions = numberOfPositions;

        cudaStreamSynchronize(stream);

        const size_t countBeforeFilter = this->numberOfPositions;
        FilterInvalidPositions();

        if (countBeforeFilter == this->numberOfPositions && this->numberOfPositions > 0)
            ComputeAABBSinglePass(positions, this->numberOfPositions, aabb.min, aabb.max);
    }

    void PCD::ToHostVectors(
        std::vector<float3>& h_positions,
        std::vector<float3>& h_normals,
        std::vector<uchar3>& h_colors)
    {
        auto alive_begin = isAlive.begin();
        const size_t aliveCount = static_cast<size_t>(
            thrust::count(alive_begin, alive_begin + numberOfPositions, true));

        h_positions.resize(aliveCount);
        h_normals.resize(aliveCount);
        h_colors.resize(aliveCount);

        if (aliveCount == 0) return;

        if (aliveCount == numberOfPositions)
        {
            thrust::copy(positions.begin(), positions.begin() + aliveCount, h_positions.begin());
            thrust::copy(normals.begin(), normals.begin() + aliveCount, h_normals.begin());
            thrust::copy(colors.begin(), colors.begin() + aliveCount, h_colors.begin());
        }
        else
        {
            thrust::device_vector<float3> d_p(aliveCount);
            thrust::device_vector<float3> d_n(aliveCount);
            thrust::device_vector<uchar3> d_c(aliveCount);

            auto src = thrust::make_zip_iterator(thrust::make_tuple(
                positions.begin(), normals.begin(), colors.begin()));
            auto dst = thrust::make_zip_iterator(thrust::make_tuple(
                d_p.begin(), d_n.begin(), d_c.begin()));

            thrust::copy_if(src, src + numberOfPositions, alive_begin, dst,
                thrust::identity<bool>());

            thrust::copy(d_p.begin(), d_p.end(), h_positions.begin());
            thrust::copy(d_n.begin(), d_n.end(), h_normals.begin());
            thrust::copy(d_c.begin(), d_c.end(), h_colors.begin());
        }
    }

    void PCD::ToHostVectors(
        std::vector<float3>& h_positions,
        std::vector<float3>& h_normals,
        std::vector<float4>& h_colors)
    {
        auto alive_begin = isAlive.begin();
        const size_t aliveCount = static_cast<size_t>(
            thrust::count(alive_begin, alive_begin + numberOfPositions, true));

        h_positions.resize(aliveCount);
        h_normals.resize(aliveCount);
        h_colors.resize(aliveCount);

        if (aliveCount == 0) return;

        thrust::device_vector<uchar3> d_c3(aliveCount);

        if (aliveCount == numberOfPositions)
        {
            thrust::copy(positions.begin(), positions.begin() + aliveCount, h_positions.begin());
            thrust::copy(normals.begin(), normals.begin() + aliveCount, h_normals.begin());
            thrust::copy(colors.begin(), colors.begin() + aliveCount, d_c3.begin());
        }
        else
        {
            thrust::device_vector<float3> d_p(aliveCount);
            thrust::device_vector<float3> d_n(aliveCount);

            auto src = thrust::make_zip_iterator(thrust::make_tuple(
                positions.begin(), normals.begin(), colors.begin()));
            auto dst = thrust::make_zip_iterator(thrust::make_tuple(
                d_p.begin(), d_n.begin(), d_c3.begin()));

            thrust::copy_if(src, src + numberOfPositions, alive_begin, dst,
                thrust::identity<bool>());

            thrust::copy(d_p.begin(), d_p.end(), h_positions.begin());
            thrust::copy(d_n.begin(), d_n.end(), h_normals.begin());
        }

        thrust::device_vector<float4> d_c4(aliveCount);
        thrust::transform(d_c3.begin(), d_c3.end(), d_c4.begin(), UChar3ToFloat4());
        thrust::copy(d_c4.begin(), d_c4.end(), h_colors.begin());
    }

    void ExtractByAABBImpl(
        const thrust::device_vector<float3>& positions,
        const thrust::device_vector<float3>& normals,
        const thrust::device_vector<uchar3>& colors,
        const thrust::device_vector<bool>& isAlive,
        size_t numberOfPositions,
        const float3& regionMin,
        const float3& regionMax,
        PCD& dst)
    {
        dst.clear();
        if (numberOfPositions == 0) return;

        thrust::device_vector<bool> d_mask(numberOfPositions);
        InsideAABBMask pred{ regionMin, regionMax };

        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(positions.begin(), isAlive.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                positions.begin() + numberOfPositions,
                isAlive.begin() + numberOfPositions)),
            d_mask.begin(), pred);

        const size_t count = static_cast<size_t>(
            thrust::count(d_mask.begin(), d_mask.end(), true));

        if (count == 0) return;

        dst.resize(count);

        auto d_dst_pos = thrust::device_pointer_cast(dst.GetPositions());
        auto d_dst_nrm = thrust::device_pointer_cast(dst.GetNormals());
        auto d_dst_col = thrust::device_pointer_cast(dst.GetColors());
        auto d_dst_alive = thrust::device_pointer_cast(dst.GetIsAlive());

        auto src = thrust::make_zip_iterator(thrust::make_tuple(
            positions.begin(), normals.begin(), colors.begin()));
        auto dst_zip = thrust::make_zip_iterator(thrust::make_tuple(
            d_dst_pos, d_dst_nrm, d_dst_col));

        thrust::copy_if(src, src + numberOfPositions, d_mask.begin(), dst_zip,
            thrust::identity<bool>());

        thrust::fill(d_dst_alive, d_dst_alive + count, true);
        dst.SetNumberOfPositions(count);

        auto t_begin = thrust::make_transform_iterator(d_dst_pos, DuplicateFloat3{});
        const auto init = thrust::make_tuple(
            make_float3(FLT_MAX, FLT_MAX, FLT_MAX),
            make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX));
        auto result = thrust::reduce(t_begin, t_begin + count, init, MinMaxReducer{});
        dst.GetAABB().min = thrust::get<0>(result);
        dst.GetAABB().max = thrust::get<1>(result);
    }

    PCD PCD::ExtractByAABB(const float3& regionMin, const float3& regionMax) const
    {
        PCD result;
        ExtractByAABBImpl(positions, normals, colors, isAlive,
            numberOfPositions, regionMin, regionMax, result);
        return result;
    }

    PCD PCD::ExtractByAABB(const cuAABB& region) const
    {
        return ExtractByAABB(region.min, region.max);
    }

    void PCD::ExtractByAABB(const float3& regionMin, const float3& regionMax, PCD& out) const
    {
        ExtractByAABBImpl(positions, normals, colors, isAlive,
            numberOfPositions, regionMin, regionMax, out);
    }

    void PCD::ExtractByAABB(const cuAABB& region, PCD& out) const
    {
        ExtractByAABB(region.min, region.max, out);
    }

    void PCD::RebuildAABB()
    {
        if (numberOfPositions == 0)
        {
            aabb.min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
            aabb.max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
            return;
        }
        ComputeAABBSinglePass(positions, numberOfPositions, aabb.min, aabb.max);
    }

    void PCD::RebuildAABB(CUstream_st* stream, [[maybe_unused]] cached_allocator* alloc)
    {
        if (numberOfPositions == 0)
        {
            aabb.min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
            aabb.max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
            return;
        }
        ComputeAABBSinglePassOnStream(
            thrust::cuda::par.on(stream), positions, numberOfPositions, aabb.min, aabb.max);
    }

    void PCD::FillIsAlive(bool value)
    {
        thrust::fill(isAlive.begin(), isAlive.begin() + numberOfPositions, value);
    }

    void PCD::FillIsAlive(bool value, CUstream_st* stream, [[maybe_unused]] cached_allocator* alloc)
    {
        thrust::fill(thrust::cuda::par.on(stream),
            isAlive.begin(), isAlive.begin() + numberOfPositions, value);
    }

    void PCD::ClearLabels(CUstream_st* stream, cached_allocator* alloc)
    {
        if (numberOfPositions == 0) return;

        if (labels.size() < numberOfPositions)
            labels.resize(numberOfPositions);

        thrust::fill(labels.begin(), labels.begin() + numberOfPositions, 0u);
    }

    void PCD::ClearSubLabels(CUstream_st* stream, cached_allocator* alloc)
    {
        if (numberOfPositions == 0) return;

        if (subLabels.size() < numberOfPositions)
            subLabels.resize(numberOfPositions);

        thrust::fill(subLabels.begin(), subLabels.begin() + numberOfPositions, 0u);
    }

    void PCD::FilterInvalidPositions()
    {
        if (numberOfPositions == 0) return;

        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
            positions.begin(),
            normals.begin(),
            colors.begin(),
            isAlive.begin()));

        auto new_end = thrust::stable_partition(
            zip_begin, zip_begin + numberOfPositions, ValidPositionPred4{});

        const size_t validCount = static_cast<size_t>(new_end - zip_begin);

        if (validCount == numberOfPositions) return;

        if (validCount == 0)
        {
            numberOfPositions = 0;
            aabb.min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
            aabb.max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
            return;
        }

        numberOfPositions = validCount;
        ComputeAABBSinglePass(positions, validCount, aabb.min, aabb.max);
    }

    void PCD::Merge(const PCD& other)
    {
        if (other.numberOfPositions == 0) return;

        const size_t oldCount = numberOfPositions;
        const size_t newCount = oldCount + other.numberOfPositions;

        resize(newCount);

        thrust::copy(other.positions.begin(), other.positions.begin() + other.numberOfPositions,
            positions.begin() + oldCount);
        thrust::copy(other.normals.begin(), other.normals.begin() + other.numberOfPositions,
            normals.begin() + oldCount);
        thrust::copy(other.colors.begin(), other.colors.begin() + other.numberOfPositions,
            colors.begin() + oldCount);
        thrust::copy(other.isAlive.begin(), other.isAlive.begin() + other.numberOfPositions,
            isAlive.begin() + oldCount);

        thrust::fill(labels.begin() + oldCount, labels.begin() + newCount, 0u);
        thrust::fill(subLabels.begin() + oldCount, subLabels.begin() + newCount, 0u);

        numberOfPositions = newCount;

        aabb.min = make_float3(
            fminf(aabb.min.x, other.aabb.min.x),
            fminf(aabb.min.y, other.aabb.min.y),
            fminf(aabb.min.z, other.aabb.min.z));
        aabb.max = make_float3(
            fmaxf(aabb.max.x, other.aabb.max.x),
            fmaxf(aabb.max.y, other.aabb.max.y),
            fmaxf(aabb.max.z, other.aabb.max.z));
    }

    void PCD::CropUsingLocalRTandAABB(
        const float* inverseRT,
        const cuAABB& localAABB,
        PCD& out,
        cached_allocator* alloc,
        CUstream_st* stream)
    {
        CUDA_TS(CropUsingLocalRTandAABB);

        out.SetNumberOfPositions(0);
        out.GetAABB() = { make_float3(FLT_MAX,  FLT_MAX,  FLT_MAX),
                          make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX) };

        if (numberOfPositions == 0) return;

        auto policy = thrust::cuda::par.on(stream);

        out.resize(numberOfPositions);

        TransformAABBMask pred;
        for (int i = 0; i < 16; ++i) pred.m[i] = inverseRT[i];
        pred.aabb = localAABB;

        auto src = thrust::make_zip_iterator(thrust::make_tuple(
            positions.begin(), normals.begin(), colors.begin()));
        auto dst = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(out.GetPositions()),
            thrust::device_pointer_cast(out.GetNormals()),
            thrust::device_pointer_cast(out.GetColors())));
        auto stencil = thrust::make_zip_iterator(thrust::make_tuple(
            positions.begin(), isAlive.begin()));

        auto dst_end = thrust::copy_if(policy,
            src, src + numberOfPositions,
            stencil, dst, pred);

        const size_t count = static_cast<size_t>(dst_end - dst);

        thrust::fill(policy,
            thrust::device_pointer_cast(out.GetIsAlive()),
            thrust::device_pointer_cast(out.GetIsAlive()) + count, true);

        out.SetNumberOfPositions(count);

        if (count > 0)
            ComputeAABBSinglePassOnStream(policy, out.positions, count,
                out.GetAABB().min, out.GetAABB().max);

        CUDA_TE(CropUsingLocalRTandAABB);
    }

    void PCD::Clustering(float cellSize, float clusterDistance, CUstream_st* stream, cached_allocator* alloc)
    {
        CUDA_TS(Clustering);
        if (numberOfPositions == 0) return;
        if (labels.size() < numberOfPositions)
            labels.resize(numberOfPositions);
        sparseCells.Build(this, cellSize, stream);
        sparseCells.ApplyClustering(this, labels.data().get(), clusterDistance, stream);
        CUDA_TE(Clustering);
    }

    void PCD::Clustering(float cellSize, float clusterDistance, float angleThreshold, CUstream_st* stream, cached_allocator* alloc)
    {
        CUDA_TS(Clustering);
        if (numberOfPositions == 0) return;
        if (labels.size() < numberOfPositions)
            labels.resize(numberOfPositions);

        CUDA_TS(Clustering_Build);
        sparseCells.Build(this, cellSize, stream);
        CUDA_TE(Clustering_Build);

        CUDA_TS(Clustering_Apply);
        sparseCells.ApplyClustering(this, labels.data().get(), clusterDistance, angleThreshold, stream);
        CUDA_TE(Clustering_Apply);

        CUDA_TE(Clustering);
    }

    void PCD::ClusteringSubLabels(float cellSize, float clusterDistance, CUstream_st* stream, cached_allocator* alloc)
    {
        CUDA_TS(ClusteringSubLabels);
        if (numberOfPositions == 0) return;
        if (subLabels.size() < numberOfPositions)
            subLabels.resize(numberOfPositions);
        sparseCells.Build(this, cellSize, stream);
        sparseCells.ApplyClustering(this, subLabels.data().get(), clusterDistance, stream);
        CUDA_TE(ClusteringSubLabels);
    }

    void PCD::ClusteringSubLabels(float cellSize, float clusterDistance, float angleThreshold, CUstream_st* stream, cached_allocator* alloc)
    {
        CUDA_TS(ClusteringSubLabels);
        if (numberOfPositions == 0) return;
        if (subLabels.size() < numberOfPositions)
            subLabels.resize(numberOfPositions);

        CUDA_TS(ClusteringSubLabels_Build);
        sparseCells.Build(this, cellSize, stream);
        CUDA_TE(ClusteringSubLabels_Build);

        CUDA_TS(ClusteringSubLabels_Apply);
        sparseCells.ApplyClustering(this, subLabels.data().get(), clusterDistance, angleThreshold, stream);
        CUDA_TE(ClusteringSubLabels_Apply);

        CUDA_TE(ClusteringSubLabels);
    }

    void PCD::BuildLabelCountTable()
    {
        if (numberOfPositions == 0) return;
        BuildCountTableImpl(labels, numberOfPositions, labelCountTable);
    }

    void PCD::BuildLabelCountTable(CUstream_st* stream)
    {
        if (numberOfPositions == 0) return;
        BuildCountTableOnStreamImpl(thrust::cuda::par.on(stream), labels, numberOfPositions, labelCountTable);
    }

    void PCD::BuildPointCountTable()
    {
        if (numberOfPositions == 0) return;
        BuildPointCountTableImpl(labels, numberOfPositions, pointCountTable);
    }

    void PCD::BuildPointCountTable(CUstream_st* stream)
    {
        if (numberOfPositions == 0) return;
        BuildPointCountTableOnStreamImpl(thrust::cuda::par.on(stream), labels, numberOfPositions, pointCountTable);
    }

    void PCD::BuildSubLabelCountTable()
    {
        if (numberOfPositions == 0) return;
        BuildCountTableImpl(subLabels, numberOfPositions, subLabelCountTable);
    }

    void PCD::BuildSubLabelCountTable(CUstream_st* stream)
    {
        if (numberOfPositions == 0) return;
        BuildCountTableOnStreamImpl(thrust::cuda::par.on(stream), subLabels, numberOfPositions, subLabelCountTable);
    }

    void PCD::BuildSubLabelPointCountTable()
    {
        if (numberOfPositions == 0) return;
        BuildPointCountTableImpl(subLabels, numberOfPositions, subLabelPointCountTable);
    }

    void PCD::BuildSubLabelPointCountTable(CUstream_st* stream)
    {
        if (numberOfPositions == 0) return;
        BuildPointCountTableOnStreamImpl(thrust::cuda::par.on(stream), subLabels, numberOfPositions, subLabelPointCountTable);
    }

    __global__ void VoxelAccumulateKernel(
        const float3* __restrict__ pos,
        const float3* __restrict__ nrm,
        const uchar3* __restrict__ col,
        const bool* __restrict__ alive,
        int    count,
        float  invVoxel,
        int3   gridSize,
        float3 origin,
        float3* voxelSumPos,
        float3* voxelSumNrm,
        float3* voxelSumCol,
        int* voxelCount)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;
        if (!alive[i])  return;

        const float3 p = pos[i];
        const int    gx = max(0, min((int)((p.x - origin.x) * invVoxel), gridSize.x - 1));
        const int    gy = max(0, min((int)((p.y - origin.y) * invVoxel), gridSize.y - 1));
        const int    gz = max(0, min((int)((p.z - origin.z) * invVoxel), gridSize.z - 1));
        const int    h = (gz * gridSize.y + gy) * gridSize.x + gx;

        const float3 n = nrm[i];
        const uchar3 c = col[i];

        atomicAdd(&voxelSumPos[h].x, p.x);
        atomicAdd(&voxelSumPos[h].y, p.y);
        atomicAdd(&voxelSumPos[h].z, p.z);
        atomicAdd(&voxelSumNrm[h].x, n.x);
        atomicAdd(&voxelSumNrm[h].y, n.y);
        atomicAdd(&voxelSumNrm[h].z, n.z);
        atomicAdd(&voxelSumCol[h].x, (float)c.x);
        atomicAdd(&voxelSumCol[h].y, (float)c.y);
        atomicAdd(&voxelSumCol[h].z, (float)c.z);
        atomicAdd(&voxelCount[h], 1);
    }

    __global__ void VoxelCompactKernel(
        const float3* __restrict__ voxelSumPos,
        const float3* __restrict__ voxelSumNrm,
        const float3* __restrict__ voxelSumCol,
        const int* __restrict__ voxelCount,
        const int* __restrict__ outOffset,
        int     numVoxels,
        float3* dstPos,
        float3* dstNrm,
        uchar3* dstCol)
    {
        const int h = blockIdx.x * blockDim.x + threadIdx.x;
        if (h >= numVoxels) return;

        const int cnt = voxelCount[h];
        if (cnt == 0) return;

        const int   dst = outOffset[h];
        const float inv = 1.0f / (float)cnt;

        dstPos[dst] = make_float3(
            voxelSumPos[h].x * inv,
            voxelSumPos[h].y * inv,
            voxelSumPos[h].z * inv);

        float3 sn = voxelSumNrm[h];
        const float nl = sqrtf(sn.x * sn.x + sn.y * sn.y + sn.z * sn.z);
        if (nl > 0.0f) { sn.x /= nl; sn.y /= nl; sn.z /= nl; }
        dstNrm[dst] = sn;

        dstCol[dst] = make_uchar3(
            (unsigned char)fminf(voxelSumCol[h].x * inv, 255.0f),
            (unsigned char)fminf(voxelSumCol[h].y * inv, 255.0f),
            (unsigned char)fminf(voxelSumCol[h].z * inv, 255.0f));
    }

    void PCD::Downsample(float voxelSize, PCD& out) const
    {
        CUDA_TS(Downsample);

        out.clear();
        if (numberOfPositions == 0 || voxelSize <= 0.0f) return;

        const int3 gridSize = {
            max(1, (int)ceilf((aabb.max.x - aabb.min.x) / voxelSize) + 1),
            max(1, (int)ceilf((aabb.max.y - aabb.min.y) / voxelSize) + 1),
            max(1, (int)ceilf((aabb.max.z - aabb.min.z) / voxelSize) + 1)
        };
        const int numVoxels = gridSize.x * gridSize.y * gridSize.z;

        thrust::device_vector<float3> d_sumPos(numVoxels, make_float3(0, 0, 0));
        thrust::device_vector<float3> d_sumNrm(numVoxels, make_float3(0, 0, 0));
        thrust::device_vector<float3> d_sumCol(numVoxels, make_float3(0, 0, 0));
        thrust::device_vector<int>    d_cnt(numVoxels, 0);

        const int bs = 256;
        const int gs = ((int)numberOfPositions + bs - 1) / bs;

        VoxelAccumulateKernel << <gs, bs >> > (
            positions.data().get(),
            normals.data().get(),
            colors.data().get(),
            isAlive.data().get(),
            (int)numberOfPositions,
            1.0f / voxelSize, gridSize, aabb.min,
            thrust::raw_pointer_cast(d_sumPos.data()),
            thrust::raw_pointer_cast(d_sumNrm.data()),
            thrust::raw_pointer_cast(d_sumCol.data()),
            thrust::raw_pointer_cast(d_cnt.data()));

        thrust::device_vector<int> d_occupied(numVoxels);
        thrust::transform(d_cnt.begin(), d_cnt.end(), d_occupied.begin(),
            [] __device__(int c) { return c > 0 ? 1 : 0; });

        thrust::device_vector<int> d_offset(numVoxels);
        thrust::exclusive_scan(d_occupied.begin(), d_occupied.end(), d_offset.begin());

        const size_t validCount =
            static_cast<size_t>(d_offset.back()) +
            static_cast<size_t>(d_occupied.back());

        if (validCount == 0) return;

        out.resize(validCount);

        const int gsV = (numVoxels + bs - 1) / bs;
        VoxelCompactKernel << <gsV, bs >> > (
            thrust::raw_pointer_cast(d_sumPos.data()),
            thrust::raw_pointer_cast(d_sumNrm.data()),
            thrust::raw_pointer_cast(d_sumCol.data()),
            thrust::raw_pointer_cast(d_cnt.data()),
            thrust::raw_pointer_cast(d_offset.data()),
            numVoxels,
            out.positions.data().get(),
            out.normals.data().get(),
            out.colors.data().get());

        thrust::fill(out.isAlive.begin(), out.isAlive.begin() + validCount, true);
        out.numberOfPositions = validCount;
        out.aabb.min = aabb.min;
        out.aabb.max = aabb.max;

        CUDA_TE(Downsample);
    }

    PCD PCD::Downsample(float voxelSize) const
    {
        PCD result;
        Downsample(voxelSize, result);
        return result;
    }

#pragma endregion
}