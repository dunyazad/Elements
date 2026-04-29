#pragma once

#include <string>
#include <vector>

//#include <Helium/TypeDefinitions.h>
//#include <Helium/Color.hpp>

#ifdef HELIUM_EXPORTS
#define HELIUM_API __declspec(dllexport)
#else
#define HELIUM_API __declspec(dllimport)
#endif

#ifndef __CUDACC__
struct float3
{
	float x, y, z;
};

struct float4
{
	float x, y, z, w;
};
#endif

struct cuAABB
{
	float3 min;
	float3 max;
};

class HELIUM_API IVisualDebugging
{
public:
	static void Clear(const std::string& tag);
	static void ClearAll();

	static void SetVisibility(bool visible, const std::string& tag);
	static void SetVisibilityAll(bool visible);
	static void ToggleVisibility(const std::string& tag);
	static void ToggleVisibilityAll();

	static void AddLine(
		const std::string& tag,
		const float3& v0,
		const float3& v1,
		const float4& c);

	static void AddLine(
		const std::string& tag,
		const float3& v0,
		const float3& v1,
		const float4& c0,
		const float4& c1);

	static void AddTriangle(
		const std::string& tag,
		const float3& v0,
		const float3& v1,
		const float3& v2,
		const float4& c);

	static void AddTriangle(
		const std::string& tag,
		const float3& v0,
		const float3& v1,
		const float3& v2,
		const float4& c0,
		const float4& c1,
		const float4& c2);

	static void AddBox(
		const std::string& tag,
		const ::cuAABB& aabb,
		const float4& color);

	static void AddBox(
		const std::string& tag,
		const float3& center,
		const float3& dimensions,
		const float4& color);

	static void AddBox(
		const std::string& tag,
		const float3& center,
		const float3& normal,
		const float3& dimensions,
		const float4& color);

	static void AddWiredBox(
		const std::string& tag,
		const ::cuAABB& aabb,
		const float4& color);

	static void AddWiredBox(
		const std::string& tag,
		const float3& center,
		const float3& dimensions,
		const float4& color);

	static void AddWiredBox(
		const std::string& tag,
		const float3& center,
		const float3& normal,
		const float3& dimensions,
		const float4& color);

	static void AddWiredBoxBatch(
		const std::string& tag,
		const std::vector<float3>& centers,
		const float3& dimensions,
		const float4& color);

	static void AddWiredBoxBatch(
		const std::string& tag,
		const std::vector<float3>& centers,
		const std::vector<float3>& dimensions,
		const std::vector<float4>& colors);

	static void AddSphere(
		const std::string& tag,
		const float3& center,
		const float3& normal,
		float radius,
		const float4& color);

	static void AddSphereBatch(
		const std::string& tag,
		const std::vector<float3>& centers,
		float radius,
		const float4& color);

	static void AddSphereBatch(
		const std::string& tag,
		const std::vector<float3>& centers,
		const std::vector<float3>& normals,
		float radius,
		const float4& color);

	static void AddSphereBatch(
		const std::string& tag,
		const std::vector<float3>& centers,
		float radius,
		const std::vector<float4>& colors);

	static void AddSphereBatch(
		const std::string& tag,
		const std::vector<float3>& centers,
		const std::vector<float3>& normals,
		float radius,
		const std::vector<float4>& colors);

	static void AddDisk(
		const std::string& tag,
		const float3& center,
		const float3& normal,
		float radius,
		unsigned int slices,
		const float4& color,
		bool isBillboard = false);

	static void AddDiskBatch(
		const std::string& tag,
		const std::vector<float3>& positions,
		const std::vector<float3>& normals,
		float radius,
		unsigned int slices,
		const float4& color,
		bool isBillboard = false);

	static void AddDiskBatch(
		const std::string& tag,
		const std::vector<float3>& positions,
		const std::vector<float3>& normals,
		float radius,
		unsigned int slices,
		const std::vector<float4>& colors,
		bool isBillboard = false);

	static void AddCylinder(
		const std::string& tag,
		const float3& center,
		const float3& normal,
		float radius,
		float height,
		const float4& color);

	static void AddCone(
		const std::string& tag,
		const float3& center,
		const float3& normal,
		float radius,
		float height,
		const float4& color);

	static void AddCapsule(
		const std::string& tag,
		const float3& center,
		const float3& normal,
		float radius,
		float height,
		unsigned int rings,
		const float4& color);

	static void AddTorus(
		const std::string& tag,
		const float3& center,
		const float3& normal,
		float majorRadius,
		float minorRadius,
		unsigned int majorSegments,
		unsigned int minorSegments,
		const float4& color);

	static void AddTube(
		const std::string& tag,
		const float3& center,
		const float3& normal,
		float radius,
		unsigned int curveSegments,
		unsigned int radialSegments,
		const float4& color);

	static void AddArrow(
		const std::string& tag,
		const float3& start,
		const float3& direction,
		float length,
		const float4& color);

	//static void AddFrustum(
	//	const std::string& tag,
	//	const Eigen::Matrix4f& invViewProj,
	//	const float4& color);

	//static void AddFrustum(
	//	const std::string& tag,
	//	const Eigen::Matrix4f& viewMatrix,
	//	float fovDegrees,
	//	float aspectRatio,
	//	float nearPlane,
	//	float farPlane,
	//	const float4& color);

	static void AddGrid(
		const std::string& tag,
		const float3& center,
		const float3& normal,
		int divisions,
		float spacing,
		const float4& color);

	static void ClearSelectionList();
	static void AddToSelectionList(const std::string& tag);
	static unsigned int ShowNextSelection();
	static unsigned int ShowPreviousSelection();

	static float3 GetInstanceScale(const std::string& tag, size_t index = 0);
	static void SetInstanceScale(const std::string& tag, const float3& newScale, size_t index);
	static void SetInstanceScale(const std::string& tag, const float3& newScale);
};
