#pragma once

#include <Copper/CopperCommon.h>

#include <map>
#include <string>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct COPPER_API PickResult
{
    float distance;
    int index;
    float3 position;
};

struct COPPER_API CuPointCloud
{
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
		const std::vector<uchar3>& h_colors);

    void FromHostPointers(
        const float3* h_points,
        const float3* h_normals,
        const uchar3* h_colors,
        size_t n);

    void FromHostPointers(
        const float3* h_points,
        const float3* h_normals,
        const float4* h_colors,
        size_t n);

    void ToHostVectors(
        std::vector<float3>& h_points,
		std::vector<float3>& h_normals,
		std::vector<uchar3>& h_colors);

    void ToHostVectors(
        std::vector<float3>& h_points,
        std::vector<float3>& h_normals,
        std::vector<float4>& h_colors);

    PickResult Pick(const float3& rayOrigin, const float3& rayDir, float tolerance);
};
