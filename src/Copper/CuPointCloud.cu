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

namespace
{
    struct PickTransformOp
    {
        float3 origin;
        float3 dir;
        float thresholdSq;

        __host__ __device__
            PickTransformOp(float3 o, float3 d, float t)
            : origin(o), dir(d), thresholdSq(t* t) {
        }

        __host__ __device__
            PickResult operator()(const thrust::tuple<int, float3, bool>& t) const
        {
            int idx = thrust::get<0>(t);
            float3 p = thrust::get<1>(t);
            bool alive = thrust::get<2>(t);

            if (!alive) return { FLT_MAX, -1, make_float3(0,0,0) };

            // 1. 벡터 P - Origin
            float3 v = make_float3(p.x - origin.x, p.y - origin.y, p.z - origin.z);

            // 2. Ray 방향으로의 투영 거리 (t)
            float t_proj = v.x * dir.x + v.y * dir.y + v.z * dir.z;

            // 카메라 뒤에 있는 점은 무시
            if (t_proj < 0.0f) return { FLT_MAX, -1, make_float3(0,0,0) };

            // 3. Ray 상의 가장 가까운 점 (Closest Point on Ray)
            float3 cp = make_float3(
                origin.x + t_proj * dir.x,
                origin.y + t_proj * dir.y,
                origin.z + t_proj * dir.z
            );

            // 4. Ray와 포인트 사이의 수직 거리 제곱 계산
            float dx = p.x - cp.x;
            float dy = p.y - cp.y;
            float dz = p.z - cp.z;
            float distSq = dx * dx + dy * dy + dz * dz;

            // 허용 오차(반지름) 밖이면 무시
            if (distSq > thresholdSq) return { FLT_MAX, -1, make_float3(0,0,0) };

            // 허용 오차 안이라면 결과 반환
            // [수정] 실제 포인트 위치 p를 함께 반환
            return { t_proj, idx, p };
        }
    };

    // 두 결과 중 더 가까운(depth가 작은) 것을 선택하는 Functor
    struct PickReduceOp
    {
        __host__ __device__
            PickResult operator()(const PickResult& a, const PickResult& b) const
        {
            return (a.distance < b.distance) ? a : b;
        }
    };
}

PickResult CuPointCloud::Pick(const float3& rayOrigin, const float3& rayDir, float tolerance)
{
    if (points.empty()) return { -1, -1, make_float3(0,0,0) };

    // 1. Iterator 준비 (Index, Point, IsAlive를 묶어서 전달)
    auto zip_iter = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::counting_iterator<int>(0),
        points.begin(),
        isAlive.begin()
    ));

    // 2. Transform Reduce 실행
    PickResult initVal = { FLT_MAX, -1, make_float3(0,0,0) };

    PickResult result = thrust::transform_reduce(
        zip_iter,
        zip_iter + points.size(),
        PickTransformOp(rayOrigin, rayDir, tolerance),
        initVal,
        PickReduceOp()
    );

    return result;
}
