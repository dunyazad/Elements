#pragma once

#include <Copper/CopperCommon.h>

#include <map>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <Eigen/Core>
#include <Eigen/Dense>

struct COPPER_API PickResult
{
    float distance;
    int index;
    float3 position;
};

struct COPPER_API CuPointCloud
{
    float3 aabbMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
	float3 aabbMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    thrust::device_vector<float3> points;
    thrust::device_vector<float3> normals;
    thrust::device_vector<uchar3> colors;
    thrust::device_vector<bool> isAlive;

	std::map<std::string, thrust::device_vector<float>> customFloatAttributes;

    CuPointCloud();
    CuPointCloud(size_t n);

    size_t size() const;
    void resize(size_t n);
    void clear();

    float3* getPointsPtr();

    void FromHostVectors(
        const std::vector<float3>& h_points,
        const std::vector<float3>& h_normals,
		const std::vector<uchar3>& h_colors,
        const float3& h_aabbMin,
        const float3& h_aabbMax);

    void FromHostPointers(
        const float3* h_points,
        const float3* h_normals,
        const uchar3* h_colors,
        size_t n,
        const float3& h_aabbMin,
        const float3& h_aabbMax);

    void FromHostPointers(
        const float3* h_points,
        const float3* h_normals,
        const float4* h_colors,
        size_t n,
        const float3& h_aabbMin,
        const float3& h_aabbMax);

    void ToHostVectors(
        std::vector<float3>& h_points,
		std::vector<float3>& h_normals,
		std::vector<uchar3>& h_colors);

    void ToHostVectors(
        std::vector<float3>& h_points,
        std::vector<float3>& h_normals,
        std::vector<float4>& h_colors);

    PickResult Pick(const float3& rayOrigin, const float3& rayDir, float tolerance);

	void GlobalRegistration(CuPointCloud& targetCloud, Eigen::Matrix4f& outTransform, float maxCorrespondenceDistance = 0.1f, int maxIterations = 50);
};
