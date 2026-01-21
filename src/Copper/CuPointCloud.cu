#include <Copper/CuPointCloud.h>

#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

CuPointCloud::CuPointCloud() {}

CuPointCloud::CuPointCloud(size_t n)
{
    resize(n);
}

size_t CuPointCloud::size() const
{
    return points.size();
}

void CuPointCloud::resize(size_t n)
{
    points.resize(n);
    normals.resize(n);
    colors.resize(n);
    isAlive.resize(n);
}

void CuPointCloud::clear()
{
    points.clear();
    normals.clear();
    colors.clear();
    isAlive.clear();
}

float3* CuPointCloud::getPointsPtr()
{
    return thrust::raw_pointer_cast(points.data());
}

void CuPointCloud::FromHostVectors(
    const std::vector<float3>& h_points,
    const std::vector<float3>& h_normals,
    const std::vector<uchar3>& h_colors)
{
    size_t n = h_points.size();
    resize(n);

    if (!h_points.empty())
        thrust::copy(h_points.begin(), h_points.end(), points.begin());

    if (!h_normals.empty())
        thrust::copy(h_normals.begin(), h_normals.end(), normals.begin());
    else
        thrust::fill(normals.begin(), normals.end(), make_float3(0.0f, 0.0f, 1.0f));

    if (!h_colors.empty())
        thrust::copy(h_colors.begin(), h_colors.end(), colors.begin());
    else
        thrust::fill(colors.begin(), colors.end(), make_uchar3(255, 255, 255));

    thrust::fill(isAlive.begin(), isAlive.end(), true);
}

void CuPointCloud::FromHostPointers(
    const float3* h_points,
    const float3* h_normals,
    const uchar3* h_colors,
    size_t numPoints)
{
    resize(numPoints);

    if (h_points)
        thrust::copy(h_points, h_points + numPoints, points.begin());

    if (h_normals)
        thrust::copy(h_normals, h_normals + numPoints, normals.begin());
    else
        thrust::fill(normals.begin(), normals.end(), make_float3(0.0f, 0.0f, 1.0f));

    if (h_colors)
        thrust::copy(h_colors, h_colors + numPoints, colors.begin());
    else
        thrust::fill(colors.begin(), colors.end(), make_uchar3(255, 255, 255));

    thrust::fill(isAlive.begin(), isAlive.end(), true);
}

void CuPointCloud::FromHostPointers(
    const float3* h_points,
    const float3* h_normals,
    const float4* h_colors,
    size_t numPoints)
{
    resize(numPoints);

    if (h_points)
        thrust::copy(h_points, h_points + numPoints, points.begin());

    if (h_normals)
        thrust::copy(h_normals, h_normals + numPoints, normals.begin());
    else
        thrust::fill(normals.begin(), normals.end(), make_float3(0.0f, 0.0f, 1.0f));

    if (h_colors) {
        std::vector<uchar3> temp_colors(numPoints);
        for (size_t i = 0; i < numPoints; i++)
        {
            auto& c = h_colors[i];
            temp_colors[i] = {
                (unsigned char)(c.x * 255.0f),
                (unsigned char)(c.y * 255.0f),
                (unsigned char)(c.z * 255.0f)
            };
        }
        thrust::copy(temp_colors.begin(), temp_colors.end(), colors.begin());
    }
    else {
        thrust::fill(colors.begin(), colors.end(), make_uchar3(255, 255, 255));
    }

    thrust::fill(isAlive.begin(), isAlive.end(), true);
}

void CuPointCloud::ToHostVectors(
    std::vector<float3>& h_points,
    std::vector<float3>& h_normals,
    std::vector<uchar3>& h_colors)
{
    size_t aliveCount = thrust::count(isAlive.begin(), isAlive.end(), true);

    h_points.resize(aliveCount);
    h_normals.resize(aliveCount);
    h_colors.resize(aliveCount);

    if (aliveCount == 0) return;

    thrust::device_vector<float3> d_temp_p(aliveCount);
    thrust::device_vector<float3> d_temp_n(aliveCount);
    thrust::device_vector<uchar3> d_temp_c(aliveCount);

    thrust::copy_if(points.begin(), points.end(), isAlive.begin(), d_temp_p.begin(), thrust::identity<bool>());
    thrust::copy_if(normals.begin(), normals.end(), isAlive.begin(), d_temp_n.begin(), thrust::identity<bool>());
    thrust::copy_if(colors.begin(), colors.end(), isAlive.begin(), d_temp_c.begin(), thrust::identity<bool>());

    thrust::copy(d_temp_p.begin(), d_temp_p.end(), h_points.begin());
    thrust::copy(d_temp_n.begin(), d_temp_n.end(), h_normals.begin());
    thrust::copy(d_temp_c.begin(), d_temp_c.end(), h_colors.begin());
}

void CuPointCloud::ToHostVectors(
    std::vector<float3>& h_points,
    std::vector<float3>& h_normals,
    std::vector<float4>& h_colors)
{
    size_t aliveCount = thrust::count(isAlive.begin(), isAlive.end(), true);

    h_points.resize(aliveCount);
    h_normals.resize(aliveCount);
    h_colors.resize(aliveCount);

    if (aliveCount == 0) return;

    thrust::device_vector<float3> d_temp_p(aliveCount);
    thrust::device_vector<float3> d_temp_n(aliveCount);
    thrust::device_vector<uchar3> d_temp_c(aliveCount);

    thrust::copy_if(points.begin(), points.end(), isAlive.begin(), d_temp_p.begin(), thrust::identity<bool>());
    thrust::copy_if(normals.begin(), normals.end(), isAlive.begin(), d_temp_n.begin(), thrust::identity<bool>());
    thrust::copy_if(colors.begin(), colors.end(), isAlive.begin(), d_temp_c.begin(), thrust::identity<bool>());

    thrust::copy(d_temp_p.begin(), d_temp_p.end(), h_points.begin());
    thrust::copy(d_temp_n.begin(), d_temp_n.end(), h_normals.begin());

    std::vector<uchar3> temp_h_colors(aliveCount);
    thrust::copy(d_temp_c.begin(), d_temp_c.end(), temp_h_colors.begin());

    for (size_t i = 0; i < aliveCount; ++i)
    {
        uchar3 c = temp_h_colors[i];
        h_colors[i] = {
            (float)c.x / 255.0f,
            (float)c.y / 255.0f,
            (float)c.z / 255.0f,
            1.0f
        };
    }
}