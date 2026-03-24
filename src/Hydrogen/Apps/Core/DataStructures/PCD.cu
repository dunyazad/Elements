#include <Core/DataStructures/PCD.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

namespace Huvitz
{
    // Functor used by ExtractByAABBImpl.
    // Returns true when a point is alive AND lies within [bmin, bmax].
    struct InsideAABBMask
    {
        float3 bmin, bmax;

        __host__ __device__
            bool operator()(const thrust::tuple<float3, bool>& t) const
        {
            if (!thrust::get<1>(t))
                return false;
            const float3& p = thrust::get<0>(t);
            return p.x >= bmin.x && p.x <= bmax.x
                && p.y >= bmin.y && p.y <= bmax.y
                && p.z >= bmin.z && p.z <= bmax.z;
        }
    };

#pragma region PCD
    // 7: aabb member initializer in header handles default construction
    PCD::PCD()
    {
    }

    PCD::PCD(size_t n)
    {
        resize(n);
    }

    size_t PCD::size() const
    {
        return numberOfPositions;
    }

    size_t PCD::capacity() const
    {
        return allocated;
    }

    void PCD::resize(size_t n)
    {
        if (allocated >= n)
        {
            return;
        }

        positions.resize(n);
        normals.resize(n);
        colors.resize(n);
        isAlive.resize(n);
        allocated = n;
    }

    // 6: swap idiom to actually release device memory
    void PCD::clear()
    {
        thrust::device_vector<float3>().swap(positions);
        thrust::device_vector<float3>().swap(normals);
        thrust::device_vector<uchar3>().swap(colors);
        thrust::device_vector<bool>().swap(isAlive);
        allocated = 0;
        numberOfPositions = 0;
    }

    // 2: AABB computed on GPU via thrust::reduce after device upload
    void PCD::FromHostVectors(
        const std::vector<float3>& h_positions,
        const std::vector<float3>& h_normals,
        const std::vector<uchar3>& h_colors)
    {
        size_t n = h_positions.size();
        resize(n);

        if (!h_positions.empty())
        {
            thrust::copy(h_positions.begin(), h_positions.end(), positions.begin());

            aabb.min = thrust::reduce(positions.begin(), positions.begin() + n,
                make_float3(FLT_MAX, FLT_MAX, FLT_MAX), MinFloat3());
            aabb.max = thrust::reduce(positions.begin(), positions.begin() + n,
                make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX), MaxFloat3());
        }

        if (!h_normals.empty())
        {
            thrust::copy(h_normals.begin(), h_normals.end(), normals.begin());
        }
        else
        {
            thrust::fill(normals.begin(), normals.begin() + n, make_float3(0.0f, 0.0f, 1.0f));
        }

        if (!h_colors.empty())
        {
            thrust::copy(h_colors.begin(), h_colors.end(), colors.begin());
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
        const std::vector<uchar3>& h_colors,
        const float3& h_aabbMin,
        const float3& h_aabbMax)
    {
        size_t n = h_positions.size();
        resize(n);

        if (!h_positions.empty())
        {
            thrust::copy(h_positions.begin(), h_positions.end(), positions.begin());
        }

        if (!h_normals.empty())
        {
            thrust::copy(h_normals.begin(), h_normals.end(), normals.begin());
        }
        else
        {
            thrust::fill(normals.begin(), normals.begin() + n, make_float3(0.0f, 0.0f, 1.0f));
        }

        if (!h_colors.empty())
        {
            thrust::copy(h_colors.begin(), h_colors.end(), colors.begin());
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

    // 2: AABB computed on GPU via thrust::reduce after device upload
    void PCD::FromHostVectors(
        const std::vector<float3>& h_positions,
        const std::vector<float3>& h_normals,
        const std::vector<float4>& h_colors)
    {
        size_t n = h_positions.size();
        resize(n);

        if (!h_positions.empty())
        {
            thrust::copy(h_positions.begin(), h_positions.end(), positions.begin());

            aabb.min = thrust::reduce(positions.begin(), positions.begin() + n,
                make_float3(FLT_MAX, FLT_MAX, FLT_MAX), MinFloat3());
            aabb.max = thrust::reduce(positions.begin(), positions.begin() + n,
                make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX), MaxFloat3());
        }

        if (!h_normals.empty())
        {
            thrust::copy(h_normals.begin(), h_normals.end(), normals.begin());
        }
        else
        {
            thrust::fill(normals.begin(), normals.begin() + n, make_float3(0.0f, 0.0f, 1.0f));
        }

        if (!h_colors.empty())
        {
            thrust::device_vector<float4> d_temp_colors(h_colors.begin(), h_colors.end());
            thrust::transform(d_temp_colors.begin(), d_temp_colors.end(), colors.begin(), Float4ToUChar3());
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
        size_t n = h_positions.size();
        resize(n);

        if (!h_positions.empty())
        {
            thrust::copy(h_positions.begin(), h_positions.end(), positions.begin());
        }

        if (!h_normals.empty())
        {
            thrust::copy(h_normals.begin(), h_normals.end(), normals.begin());
        }
        else
        {
            thrust::fill(normals.begin(), normals.begin() + n, make_float3(0.0f, 0.0f, 1.0f));
        }

        if (!h_colors.empty())
        {
            thrust::device_vector<float4> d_temp_colors(h_colors.begin(), h_colors.end());
            thrust::transform(d_temp_colors.begin(), d_temp_colors.end(), colors.begin(), Float4ToUChar3());
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

    // 2: AABB computed on GPU via thrust::reduce after device upload
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

            aabb.min = thrust::reduce(positions.begin(), positions.begin() + numberOfPositions,
                make_float3(FLT_MAX, FLT_MAX, FLT_MAX), MinFloat3());
            aabb.max = thrust::reduce(positions.begin(), positions.begin() + numberOfPositions,
                make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX), MaxFloat3());
        }

        if (h_normals)
        {
            thrust::copy(h_normals, h_normals + numberOfPositions, normals.begin());
        }
        else
        {
            thrust::fill(normals.begin(), normals.begin() + numberOfPositions, make_float3(0.0f, 0.0f, 1.0f));
        }

        if (h_colors)
        {
            thrust::copy(h_colors, h_colors + numberOfPositions, colors.begin());
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
        const uchar3* h_colors,
        size_t numberOfPositions,
        const float3& h_aabbMin,
        const float3& h_aabbMax)
    {
        resize(numberOfPositions);

        if (h_positions)
        {
            thrust::copy(h_positions, h_positions + numberOfPositions, positions.begin());
        }

        if (h_normals)
        {
            thrust::copy(h_normals, h_normals + numberOfPositions, normals.begin());
        }
        else
        {
            thrust::fill(normals.begin(), normals.begin() + numberOfPositions, make_float3(0.0f, 0.0f, 1.0f));
        }

        if (h_colors)
        {
            thrust::copy(h_colors, h_colors + numberOfPositions, colors.begin());
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

    // 2: AABB computed on GPU via thrust::reduce after device upload
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

            aabb.min = thrust::reduce(positions.begin(), positions.begin() + numberOfPositions,
                make_float3(FLT_MAX, FLT_MAX, FLT_MAX), MinFloat3());
            aabb.max = thrust::reduce(positions.begin(), positions.begin() + numberOfPositions,
                make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX), MaxFloat3());
        }

        if (h_normals)
        {
            thrust::copy(h_normals, h_normals + numberOfPositions, normals.begin());
        }
        else
        {
            thrust::fill(normals.begin(), normals.begin() + numberOfPositions, make_float3(0.0f, 0.0f, 1.0f));
        }

        if (h_colors)
        {
            thrust::device_vector<float4> d_temp_colors(h_colors, h_colors + numberOfPositions);
            thrust::transform(d_temp_colors.begin(), d_temp_colors.end(), colors.begin(), Float4ToUChar3());
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
        {
            thrust::copy(h_positions, h_positions + numberOfPositions, positions.begin());
        }

        if (h_normals)
        {
            thrust::copy(h_normals, h_normals + numberOfPositions, normals.begin());
        }
        else
        {
            thrust::fill(normals.begin(), normals.begin() + numberOfPositions, make_float3(0.0f, 0.0f, 1.0f));
        }

        if (h_colors)
        {
            thrust::device_vector<float4> d_temp_colors(h_colors, h_colors + numberOfPositions);
            thrust::transform(d_temp_colors.begin(), d_temp_colors.end(), colors.begin(), Float4ToUChar3());
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

    // 3: alloc marked [[maybe_unused]] as it is not consumed in this function
    // 4: AABB reduce uses explicit FLT_MAX identity values instead of first_point
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
            {
                aabb.min = thrust::reduce(policy,
                    positions.begin(), positions.begin() + numberOfPositions,
                    make_float3(FLT_MAX, FLT_MAX, FLT_MAX), MinFloat3());
                aabb.max = thrust::reduce(policy,
                    positions.begin(), positions.begin() + numberOfPositions,
                    make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX), MaxFloat3());
            }
        }

        if (d_normals)
        {
            thrust::copy(policy, d_normals, d_normals + numberOfPositions, normals.begin());
        }
        else
        {
            thrust::fill(policy, normals.begin(), normals.begin() + numberOfPositions, make_float3(0.0f, 0.0f, 1.0f));
        }

        if (d_colors)
        {
            thrust::copy(policy, d_colors, d_colors + numberOfPositions, colors.begin());
        }
        else
        {
            thrust::fill(policy, colors.begin(), colors.begin() + numberOfPositions, make_uchar3(255, 255, 255));
        }

        thrust::fill(policy, isAlive.begin(), isAlive.begin() + numberOfPositions, true);

        this->numberOfPositions = numberOfPositions;
        FilterInvalidPositions();
    }

    // 3: alloc marked [[maybe_unused]] as it is not consumed in this function
    // 4: AABB reduce uses explicit FLT_MAX identity values instead of first_point
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
            {
                aabb.min = thrust::reduce(policy,
                    positions.begin(), positions.begin() + numberOfPositions,
                    make_float3(FLT_MAX, FLT_MAX, FLT_MAX), MinFloat3());
                aabb.max = thrust::reduce(policy,
                    positions.begin(), positions.begin() + numberOfPositions,
                    make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX), MaxFloat3());
            }
        }

        if (d_normals)
        {
            thrust::copy(policy, d_normals, d_normals + numberOfPositions, normals.begin());
        }
        else
        {
            thrust::fill(policy, normals.begin(), normals.begin() + numberOfPositions, make_float3(0.0f, 0.0f, 1.0f));
        }

        if (d_colors)
        {
            thrust::transform(policy, d_colors, d_colors + numberOfPositions, colors.begin(), Float4ToUChar3());
        }
        else
        {
            thrust::fill(policy, colors.begin(), colors.begin() + numberOfPositions, make_uchar3(255, 255, 255));
        }

        thrust::fill(policy, isAlive.begin(), isAlive.begin() + numberOfPositions, true);

        this->numberOfPositions = numberOfPositions;
        FilterInvalidPositions();
    }

    void PCD::ToHostVectors(
        std::vector<float3>& h_positions,
        std::vector<float3>& h_normals,
        std::vector<uchar3>& h_colors)
    {
        auto is_alive_start = isAlive.begin();

        size_t aliveCount = thrust::count(is_alive_start, is_alive_start + numberOfPositions, true);

        h_positions.resize(aliveCount);
        h_normals.resize(aliveCount);
        h_colors.resize(aliveCount);

        if (aliveCount == 0)
        {
            return;
        }

        if (aliveCount == numberOfPositions)
        {
            thrust::copy(positions.begin(), positions.begin() + aliveCount, h_positions.begin());
            thrust::copy(normals.begin(), normals.begin() + aliveCount, h_normals.begin());
            thrust::copy(colors.begin(), colors.begin() + aliveCount, h_colors.begin());
        }
        else
        {
            thrust::device_vector<float3> d_temp_p(aliveCount);
            thrust::device_vector<float3> d_temp_n(aliveCount);
            thrust::device_vector<uchar3> d_temp_c(aliveCount);

            thrust::copy_if(positions.begin(), positions.begin() + numberOfPositions, is_alive_start, d_temp_p.begin(), thrust::identity<bool>());
            thrust::copy_if(normals.begin(), normals.begin() + numberOfPositions, is_alive_start, d_temp_n.begin(), thrust::identity<bool>());
            thrust::copy_if(colors.begin(), colors.begin() + numberOfPositions, is_alive_start, d_temp_c.begin(), thrust::identity<bool>());

            thrust::copy(d_temp_p.begin(), d_temp_p.end(), h_positions.begin());
            thrust::copy(d_temp_n.begin(), d_temp_n.end(), h_normals.begin());
            thrust::copy(d_temp_c.begin(), d_temp_c.end(), h_colors.begin());
        }
    }

    // 5: removed unused is_alive_end variable
    void PCD::ToHostVectors(
        std::vector<float3>& h_positions,
        std::vector<float3>& h_normals,
        std::vector<float4>& h_colors)
    {
        auto is_alive_start = isAlive.begin();
        size_t aliveCount = thrust::count(is_alive_start, is_alive_start + numberOfPositions, true);

        h_positions.resize(aliveCount);
        h_normals.resize(aliveCount);
        h_colors.resize(aliveCount);

        if (aliveCount == 0)
        {
            return;
        }

        if (aliveCount == numberOfPositions)
        {
            thrust::copy(positions.begin(), positions.begin() + aliveCount, h_positions.begin());
            thrust::copy(normals.begin(), normals.begin() + aliveCount, h_normals.begin());
        }
        else
        {
            thrust::device_vector<float3> d_temp_p(aliveCount);
            thrust::device_vector<float3> d_temp_n(aliveCount);

            thrust::copy_if(positions.begin(), positions.begin() + numberOfPositions, is_alive_start, d_temp_p.begin(), thrust::identity<bool>());
            thrust::copy_if(normals.begin(), normals.begin() + numberOfPositions, is_alive_start, d_temp_n.begin(), thrust::identity<bool>());

            thrust::copy(d_temp_p.begin(), d_temp_p.end(), h_positions.begin());
            thrust::copy(d_temp_n.begin(), d_temp_n.end(), h_normals.begin());
        }

        thrust::device_vector<uchar3> d_temp_c3(aliveCount);
        if (aliveCount == numberOfPositions)
        {
            thrust::copy(colors.begin(), colors.begin() + aliveCount, d_temp_c3.begin());
        }
        else
        {
            thrust::copy_if(colors.begin(), colors.begin() + numberOfPositions, is_alive_start, d_temp_c3.begin(), thrust::identity<bool>());
        }

        thrust::device_vector<float4> d_temp_c4(aliveCount);
        thrust::transform(d_temp_c3.begin(), d_temp_c3.end(), d_temp_c4.begin(), UChar3ToFloat4());
        thrust::copy(d_temp_c4.begin(), d_temp_c4.end(), h_colors.begin());
    }

    // Helper with external linkage (no static) so CUDA on Windows accepts the
    // InsideAABBMask __host__ __device__ functor without the internal-linkage error.
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

        if (numberOfPositions == 0)
            return;

        thrust::device_vector<bool> d_mask(numberOfPositions);
        InsideAABBMask pred{ regionMin, regionMax };

        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(positions.begin(), isAlive.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(positions.begin() + numberOfPositions, isAlive.begin() + numberOfPositions)),
            d_mask.begin(),
            pred);

        size_t count = static_cast<size_t>(thrust::count(d_mask.begin(), d_mask.end(), true));

        if (count == 0)
            return;

        dst.resize(count);

        // Wrap raw device pointers so thrust can resolve the correct execution system.
        auto d_dst_pos = thrust::device_pointer_cast(dst.GetPositions());
        auto d_dst_nrm = thrust::device_pointer_cast(dst.GetNormals());
        auto d_dst_col = thrust::device_pointer_cast(dst.GetColors());
        auto d_dst_alive = thrust::device_pointer_cast(dst.GetIsAlive());

        thrust::copy_if(positions.begin(), positions.begin() + numberOfPositions,
            d_mask.begin(), d_dst_pos, thrust::identity<bool>());
        thrust::copy_if(normals.begin(), normals.begin() + numberOfPositions,
            d_mask.begin(), d_dst_nrm, thrust::identity<bool>());
        thrust::copy_if(colors.begin(), colors.begin() + numberOfPositions,
            d_mask.begin(), d_dst_col, thrust::identity<bool>());
        thrust::fill(d_dst_alive, d_dst_alive + count, true);

        dst.SetNumberOfPositions(count);

        dst.GetAABB().min = thrust::reduce(
            d_dst_pos, d_dst_pos + count,
            make_float3(FLT_MAX, FLT_MAX, FLT_MAX), MinFloat3());
        dst.GetAABB().max = thrust::reduce(
            d_dst_pos, d_dst_pos + count,
            make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX), MaxFloat3());
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
    // Functor used by FilterInvalidPositions.
    struct ValidPositionMask
    {
        __host__ __device__
            bool operator()(const float3& p) const
        {
            return p.x != FLT_MAX && p.y != FLT_MAX && p.z != FLT_MAX;
        }
    };

    void PCD::FilterInvalidPositions()
    {
        if (numberOfPositions == 0) return;

        thrust::device_vector<bool> d_mask(numberOfPositions);
        thrust::transform(
            positions.begin(), positions.begin() + numberOfPositions,
            d_mask.begin(),
            ValidPositionMask{});

        size_t validCount = static_cast<size_t>(
            thrust::count(d_mask.begin(), d_mask.end(), true));

        if (validCount == numberOfPositions) return;
        if (validCount == 0) { numberOfPositions = 0; return; }

        thrust::device_vector<float3> d_pos(validCount);
        thrust::device_vector<float3> d_nrm(validCount);
        thrust::device_vector<uchar3> d_col(validCount);

        thrust::copy_if(positions.begin(), positions.begin() + numberOfPositions,
            d_mask.begin(), d_pos.begin(), thrust::identity<bool>());
        thrust::copy_if(normals.begin(), normals.begin() + numberOfPositions,
            d_mask.begin(), d_nrm.begin(), thrust::identity<bool>());
        thrust::copy_if(colors.begin(), colors.begin() + numberOfPositions,
            d_mask.begin(), d_col.begin(), thrust::identity<bool>());

        thrust::copy(d_pos.begin(), d_pos.end(), positions.begin());
        thrust::copy(d_nrm.begin(), d_nrm.end(), normals.begin());
        thrust::copy(d_col.begin(), d_col.end(), colors.begin());
        thrust::fill(isAlive.begin(), isAlive.begin() + validCount, true);

        numberOfPositions = validCount;

        aabb.min = thrust::reduce(
            positions.begin(), positions.begin() + validCount,
            make_float3(FLT_MAX, FLT_MAX, FLT_MAX), MinFloat3());
        aabb.max = thrust::reduce(
            positions.begin(), positions.begin() + validCount,
            make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX), MaxFloat3());
    }


    // Appends all points from [other] to this PCD (regardless of isAlive state in [other]).
    // The combined AABB is expanded to cover both sets.
    void PCD::Merge(const PCD& other)
    {
        if (other.numberOfPositions == 0)
            return;

        size_t oldCount = numberOfPositions;
        size_t newCount = oldCount + other.numberOfPositions;

        resize(newCount);

        thrust::copy(other.positions.begin(), other.positions.begin() + other.numberOfPositions,
            positions.begin() + oldCount);
        thrust::copy(other.normals.begin(), other.normals.begin() + other.numberOfPositions,
            normals.begin() + oldCount);
        thrust::copy(other.colors.begin(), other.colors.begin() + other.numberOfPositions,
            colors.begin() + oldCount);
        thrust::copy(other.isAlive.begin(), other.isAlive.begin() + other.numberOfPositions,
            isAlive.begin() + oldCount);

        numberOfPositions = newCount;

        // Expand AABB to cover the merged set.
        aabb.min = make_float3(
            fminf(aabb.min.x, other.aabb.min.x),
            fminf(aabb.min.y, other.aabb.min.y),
            fminf(aabb.min.z, other.aabb.min.z));
        aabb.max = make_float3(
            fmaxf(aabb.max.x, other.aabb.max.x),
            fmaxf(aabb.max.y, other.aabb.max.y),
            fmaxf(aabb.max.z, other.aabb.max.z));
    }

    // ---------------------------------------------------------------------------
    // Voxel downsampling kernels
    // ---------------------------------------------------------------------------
    // Accumulates per-voxel sum of positions, normals, colors and point count.
    __global__ void VoxelAccumulateKernel(
        const float3* __restrict__ pos,
        const float3* __restrict__ nrm,
        const uchar3* __restrict__ col,
        const bool* __restrict__ alive,
        int count,
        float invVoxel,
        int3  gridSize,
        float3 origin,
        float3* voxelSumPos,   // [numVoxels]
        float3* voxelSumNrm,   // [numVoxels]
        float3* voxelSumCol,   // [numVoxels]  stored as float for accumulation
        int* voxelCount,    // [numVoxels]
        int* outHash)       // [count] -- hash per input point (INT_MAX if dead)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= count) return;

        if (!alive[i]) { outHash[i] = INT_MAX; return; }

        float3 p = pos[i];
        int gx = max(0, min((int)((p.x - origin.x) * invVoxel), gridSize.x - 1));
        int gy = max(0, min((int)((p.y - origin.y) * invVoxel), gridSize.y - 1));
        int gz = max(0, min((int)((p.z - origin.z) * invVoxel), gridSize.z - 1));
        int h = (gz * gridSize.y + gy) * gridSize.x + gx;
        outHash[i] = h;

        float3 n = nrm[i];
        uchar3 c = col[i];

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

    // Writes occupied voxel centroids into compact output arrays.
    // One thread per voxel; skips empty voxels.
    __global__ void VoxelCompactKernel(
        const float3* __restrict__ voxelSumPos,
        const float3* __restrict__ voxelSumNrm,
        const float3* __restrict__ voxelSumCol,
        const int* __restrict__ voxelCount,
        const int* __restrict__ outOffset,  // exclusive prefix sum of (voxelCount>0)
        int numVoxels,
        float3* dstPos,
        float3* dstNrm,
        uchar3* dstCol)
    {
        int h = blockIdx.x * blockDim.x + threadIdx.x;
        if (h >= numVoxels) return;

        int cnt = voxelCount[h];
        if (cnt == 0) return;

        int dst = outOffset[h];
        float inv = 1.0f / (float)cnt;

        dstPos[dst] = make_float3(
            voxelSumPos[h].x * inv,
            voxelSumPos[h].y * inv,
            voxelSumPos[h].z * inv);

        float3 sn = voxelSumNrm[h];
        float  nl = sqrtf(sn.x * sn.x + sn.y * sn.y + sn.z * sn.z);
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

        if (numberOfPositions == 0 || voxelSize <= 0.0f)
            return;

        int3 gridSize = {
            max(1, (int)ceilf((aabb.max.x - aabb.min.x) / voxelSize) + 1),
            max(1, (int)ceilf((aabb.max.y - aabb.min.y) / voxelSize) + 1),
            max(1, (int)ceilf((aabb.max.z - aabb.min.z) / voxelSize) + 1)
        };
        int numVoxels = gridSize.x * gridSize.y * gridSize.z;

        // Per-voxel accumulation buffers.
        thrust::device_vector<float3> d_sumPos(numVoxels, make_float3(0, 0, 0));
        thrust::device_vector<float3> d_sumNrm(numVoxels, make_float3(0, 0, 0));
        thrust::device_vector<float3> d_sumCol(numVoxels, make_float3(0, 0, 0));
        thrust::device_vector<int>    d_cnt(numVoxels, 0);
        thrust::device_vector<int>    d_hash(numberOfPositions);

        int bs = 256;
        int gs = ((int)numberOfPositions + bs - 1) / bs;

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
            thrust::raw_pointer_cast(d_cnt.data()),
            thrust::raw_pointer_cast(d_hash.data()));

        cudaDeviceSynchronize();

        // Exclusive prefix sum over occupied mask to get compact output offsets.
        thrust::device_vector<int> d_occupied(numVoxels);
        thrust::transform(d_cnt.begin(), d_cnt.end(), d_occupied.begin(),
            [] __device__(int c) { return c > 0 ? 1 : 0; });

        size_t validCount = static_cast<size_t>(
            thrust::count(d_occupied.begin(), d_occupied.end(), 1));

        if (validCount == 0) return;

        thrust::device_vector<int> d_offset(numVoxels);
        thrust::exclusive_scan(d_occupied.begin(), d_occupied.end(), d_offset.begin());

        out.resize(validCount);

        int gsV = (numVoxels + bs - 1) / bs;
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

        cudaDeviceSynchronize();

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
