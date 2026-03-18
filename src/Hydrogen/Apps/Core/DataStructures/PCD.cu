#include <Core/DataStructures/PCD.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

namespace Huvitz
{
#pragma region PCD
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

    void PCD::clear()
    {
        positions.clear();
        normals.clear();
        colors.clear();
        isAlive.clear();
        // customFloatAttributes.clear();
        allocated = 0;
        numberOfPositions = 0;
    }

    void PCD::FromHostVectors(
        const std::vector<float3>& h_positions,
        const std::vector<float3>& h_normals,
        const std::vector<uchar3>& h_colors)
    {
        float3 h_aabbMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 h_aabbMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        for (const auto& p : h_positions)
        {
            h_aabbMin = min_f3(h_aabbMin, p);
            h_aabbMax = max_f3(h_aabbMax, p);
        }

        FromHostVectors(h_positions, h_normals, h_colors, h_aabbMin, h_aabbMax);
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
    }

    void PCD::FromHostVectors(
        const std::vector<float3>& h_positions,
        const std::vector<float3>& h_normals,
        const std::vector<float4>& h_colors)
    {
        float3 h_aabbMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 h_aabbMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        for (const auto& p : h_positions)
        {
            h_aabbMin = min_f3(h_aabbMin, p);
            h_aabbMax = max_f3(h_aabbMax, p);
        }

        FromHostVectors(h_positions, h_normals, h_colors, h_aabbMin, h_aabbMax);
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
    }

    void PCD::FromHostPointers(
        const float3* h_positions,
        const float3* h_normals,
        const uchar3* h_colors,
        size_t numberOfPositions)
    {
        float3 h_aabbMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 h_aabbMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        if (h_positions)
        {
            for (size_t i = 0; i < numberOfPositions; ++i)
            {
                h_aabbMin = min_f3(h_aabbMin, h_positions[i]);
                h_aabbMax = max_f3(h_aabbMax, h_positions[i]);
            }
        }

        FromHostPointers(h_positions, h_normals, h_colors, numberOfPositions, h_aabbMin, h_aabbMax);
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
    }

    void PCD::FromHostPointers(
        const float3* h_positions,
        const float3* h_normals,
        const float4* h_colors,
        size_t numberOfPositions)
    {
        float3 h_aabbMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 h_aabbMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        if (h_positions)
        {
            for (size_t i = 0; i < numberOfPositions; ++i)
            {
                h_aabbMin = min_f3(h_aabbMin, h_positions[i]);
                h_aabbMax = max_f3(h_aabbMax, h_positions[i]);
            }
        }

        FromHostPointers(h_positions, h_normals, h_colors, numberOfPositions, h_aabbMin, h_aabbMax);
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
    }

    void PCD::FromDevicePointers(
        const float3* d_positions,
        const float3* d_normals,
        const uchar3* d_colors,
        size_t numberOfPositions,
        CUstream_st* stream,
        cached_allocator* alloc)
    {
        resize(numberOfPositions);
        auto policy = thrust::cuda::par.on(stream);

        if (d_positions)
        {
            thrust::copy(policy, d_positions, d_positions + numberOfPositions, positions.begin());

            if (numberOfPositions > 0)
            {
                float3 first_point;
                cudaMemcpyAsync(&first_point, d_positions, sizeof(float3), cudaMemcpyDeviceToHost, (cudaStream_t)stream);
                cudaStreamSynchronize((cudaStream_t)stream);

                aabb.min = thrust::reduce(policy, positions.begin(), positions.begin() + numberOfPositions, first_point, MinFloat3());
                aabb.max = thrust::reduce(policy, positions.begin(), positions.begin() + numberOfPositions, first_point, MaxFloat3());
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
    }

    void PCD::FromDevicePointers(
        const float3* d_positions,
        const float3* d_normals,
        const float4* d_colors,
        size_t numberOfPositions,
        CUstream_st* stream,
        cached_allocator* alloc)
    {
        resize(numberOfPositions);
        auto policy = thrust::cuda::par.on(stream);

        if (d_positions)
        {
            thrust::copy(policy, d_positions, d_positions + numberOfPositions, positions.begin());

            if (numberOfPositions > 0)
            {
                float3 first_point;
                cudaMemcpyAsync(&first_point, d_positions, sizeof(float3), cudaMemcpyDeviceToHost, (cudaStream_t)stream);
                cudaStreamSynchronize((cudaStream_t)stream);

                aabb.min = thrust::reduce(policy, positions.begin(), positions.begin() + numberOfPositions, first_point, MinFloat3());
                aabb.max = thrust::reduce(policy, positions.begin(), positions.begin() + numberOfPositions, first_point, MaxFloat3());
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
    }

    void PCD::ToHostVectors(
        std::vector<float3>& h_positions,
        std::vector<float3>& h_normals,
        std::vector<uchar3>& h_colors)
    {
        auto is_alive_start = isAlive.begin();
        auto is_alive_end = isAlive.begin() + numberOfPositions;

        size_t aliveCount = thrust::count(is_alive_start, is_alive_end, true);

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

    void PCD::ToHostVectors(
        std::vector<float3>& h_positions,
        std::vector<float3>& h_normals,
        std::vector<float4>& h_colors)
    {
        auto is_alive_start = isAlive.begin();
        auto is_alive_end = isAlive.begin() + numberOfPositions;
        size_t aliveCount = thrust::count(is_alive_start, is_alive_end, true);

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
#pragma endregion
}
