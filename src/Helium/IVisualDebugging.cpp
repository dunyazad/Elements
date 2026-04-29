#include "pch.h"

#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#define XYZ(v) (v).x, (v).y, (v).z
#define XYZ_(v) (v).x(), (v).y(), (v).z()
#define XYZW(v) (v).x, (v).y, (v).z, (v).w
#define XYZW_(v) (v).x(), (v).y(), (v).z(), (v).w()


void IVisualDebugging::Clear(const std::string& tag)
{
	VD::Clear(tag);
}
void IVisualDebugging::ClearAll()
{
	VD::ClearAll();
}

void IVisualDebugging::SetVisibility(bool visible, const std::string& tag)
{
	VD::SetVisibility(visible, tag);
}
void IVisualDebugging::SetVisibilityAll(bool visible)
{
	VD::SetVisibilityAll(visible);
}
void IVisualDebugging::ToggleVisibility(const std::string& tag)
{
	VD::ToggleVisibility(tag);
}
void IVisualDebugging::ToggleVisibilityAll()
{
	VD::ToggleVisibilityAll();
}

void IVisualDebugging::AddLine(
	const std::string& tag,
	const float3& v0,
	const float3& v1,
	const float4& c)
{
	VD::AddLine(tag, { XYZ(v0) }, { XYZ(v1) }, { XYZW(c) });
}

void IVisualDebugging::AddLine(
	const std::string& tag,
	const float3& v0,
	const float3& v1,
	const float4& c0,
	const float4& c1)
{
	VD::AddLine(tag, { XYZ(v0) }, { XYZ(v1) }, { XYZW(c0) }, { XYZW(c1) });
}

void IVisualDebugging::AddTriangle(
	const std::string& tag,
	const float3& v0,
	const float3& v1,
	const float3& v2,
	const float4& c)
{
	VD::AddTriangle(tag, { XYZ(v0) }, { XYZ(v1) }, { XYZ(v2) }, { XYZW(c) });
}

void IVisualDebugging::AddTriangle(
	const std::string& tag,
	const float3& v0,
	const float3& v1,
	const float3& v2,
	const float4& c0,
	const float4& c1,
	const float4& c2)
{
	VD::AddTriangle(tag, { XYZ(v0) }, { XYZ(v1) }, { XYZ(v2) }, { XYZW(c0) }, { XYZW(c1) }, { XYZW(c2) });
}

void IVisualDebugging::AddBox(
	const std::string& tag,
	const ::cuAABB& aabb,
	const float4& color)
{
	VD::AddBox(tag, { {XYZ(aabb.min)}, {XYZ(aabb.max)} }, { XYZW(color) });
}

void IVisualDebugging::AddBox(
	const std::string& tag,
	const float3& center,
	const float3& dimensions,
	const float4& color)
{
	VD::AddBox(tag, { XYZ(center) }, { XYZ(dimensions) }, { XYZW(color) });
}

void IVisualDebugging::AddBox(
	const std::string& tag,
	const float3& center,
	const float3& normal,
	const float3& dimensions,
	const float4& color)
{
	VD::AddBox(tag, { XYZ(center) }, { XYZ(normal) }, { XYZ(dimensions) }, { XYZW(color) });
}

void IVisualDebugging::AddWiredBox(
	const std::string& tag,
	const ::cuAABB& aabb,
	const float4& color)
{
	VD::AddWiredBox(tag, {{XYZ(aabb.min)}, {XYZ(aabb.max)}}, { XYZW(color) });
}

void IVisualDebugging::AddWiredBox(
	const std::string& tag,
	const float3& center,
	const float3& dimensions,
	const float4& color)
{
	VD::AddWiredBox(tag, { XYZ(center) }, { XYZ(dimensions) }, { XYZW(color) });
}

void IVisualDebugging::AddWiredBox(
	const std::string& tag,
	const float3& center,
	const float3& normal,
	const float3& dimensions,
	const float4& color)
{
	VD::AddWiredBox(tag, { XYZ(center) }, { XYZ(normal) }, { XYZ(dimensions) }, { XYZW(color) });
}

void IVisualDebugging::AddWiredBoxBatch(
	const std::string& tag,
	const std::vector<float3>& centers,
	const float3& dimensions,
	const float4& color)
{
	std::vector<Eigen::Vector3f> eigenCenters(centers.size());
	memcpy(eigenCenters.data(), centers.data(), centers.size() * sizeof(float3));
	VD::AddWiredBoxBatch(tag, eigenCenters, { XYZ(dimensions) }, { XYZW(color) });
}

void IVisualDebugging::AddWiredBoxBatch(
	const std::string& tag,
	const std::vector<float3>& centers,
	const std::vector<float3>& dimensions,
	const std::vector<float4>& colors)
{
	std::vector<Eigen::Vector3f> eigenCenters(centers.size());
	memcpy(eigenCenters.data(), centers.data(), centers.size() * sizeof(float3));

	std::vector<Eigen::Vector3f> eigenDimensions(dimensions.size());
	memcpy(eigenDimensions.data(), dimensions.data(), dimensions.size() * sizeof(float3));

	std::vector<Eigen::Vector4f> eigenColors(colors.size());
	memcpy(eigenColors.data(), colors.data(), colors.size() * sizeof(float4));

	VD::AddWiredBoxBatch(tag, eigenCenters, eigenDimensions, eigenColors);
}

void IVisualDebugging::AddSphere(
	const std::string& tag,
	const float3& center,
	const float3& normal,
	float radius,
	const float4& color)
{
	VD::AddSphere(tag, { XYZ(center) }, { XYZ(normal) }, radius, { XYZW(color) });
}

void IVisualDebugging::AddSphereBatch(
	const std::string& tag,
	const std::vector<float3>& centers,
	float radius,
	const float4& color)
{
	std::vector<Eigen::Vector3f> eigenCenters(centers.size());
	memcpy(eigenCenters.data(), centers.data(), centers.size() * sizeof(float3));

	VD::AddSphereBatch(tag, eigenCenters, radius, { XYZW(color) });
}

void IVisualDebugging::AddSphereBatch(
	const std::string& tag,
	const std::vector<float3>& centers,
	const std::vector<float3>& normals,
	float radius,
	const float4& color)
{
	std::vector<Eigen::Vector3f> eigenCenters(centers.size());
	memcpy(eigenCenters.data(), centers.data(), centers.size() * sizeof(float3));

	std::vector<Eigen::Vector3f> eigenNormals(normals.size());
	memcpy(eigenNormals.data(), normals.data(), normals.size() * sizeof(float3));

	VD::AddSphereBatch(tag, eigenCenters, eigenNormals, radius, { XYZW(color) });
}

void IVisualDebugging::AddSphereBatch(
	const std::string& tag,
	const std::vector<float3>& centers,
	float radius,
	const std::vector<float4>& colors)
{
	std::vector<Eigen::Vector3f> eigenCenters(centers.size());
	memcpy(eigenCenters.data(), centers.data(), centers.size() * sizeof(float3));

	std::vector<Eigen::Vector4f> eigenColors(colors.size());
	memcpy(eigenColors.data(), colors.data(), colors.size() * sizeof(float4));

	VD::AddSphereBatch(tag, eigenCenters, radius, eigenColors);
}

void IVisualDebugging::AddSphereBatch(
	const std::string& tag,
	const std::vector<float3>& centers,
	const std::vector<float3>& normals,
	float radius,
	const std::vector<float4>& colors)
{
	std::vector<Eigen::Vector3f> eigenCenters(centers.size());
	memcpy(eigenCenters.data(), centers.data(), centers.size() * sizeof(float3));

	std::vector<Eigen::Vector3f> eigenNormals(normals.size());
	memcpy(eigenNormals.data(), normals.data(), normals.size() * sizeof(float3));

	std::vector<Eigen::Vector4f> eigenColors(colors.size());
	memcpy(eigenColors.data(), colors.data(), colors.size() * sizeof(float4));

	VD::AddSphereBatch(tag, eigenCenters, eigenNormals, radius, eigenColors);
}

void IVisualDebugging::AddDisk(
	const std::string& tag,
	const float3& center,
	const float3& normal,
	float radius,
	unsigned int slices,
	const float4& color,
	bool isBillboard)
{
	VD::AddDisk(tag, { XYZ(center) }, { XYZ(normal) }, radius, slices, { XYZW(color) }, isBillboard);
}

void IVisualDebugging::AddDiskBatch(
	const std::string& tag,
	const std::vector<float3>& positions,
	const std::vector<float3>& normals,
	float radius,
	unsigned int slices,
	const float4& color,
	bool isBillboard)
{
	std::vector<Eigen::Vector3f> eigenPositions(positions.size());
	memcpy(eigenPositions.data(), positions.data(), positions.size() * sizeof(float3));

	std::vector<Eigen::Vector3f> eigenNormals(normals.size());
	memcpy(eigenNormals.data(), normals.data(), normals.size() * sizeof(float3));

	VD::AddDiskBatch(tag, eigenPositions, eigenNormals, radius, slices, { XYZW(color) }, isBillboard);
}

void IVisualDebugging::AddDiskBatch(
	const std::string& tag,
	const std::vector<float3>& positions,
	const std::vector<float3>& normals,
	float radius,
	unsigned int slices,
	const std::vector<float4>& colors,
	bool isBillboard)
{
	std::vector<Eigen::Vector3f> eigenPositions(positions.size());
	memcpy(eigenPositions.data(), positions.data(), positions.size() * sizeof(float3));

	std::vector<Eigen::Vector3f> eigenNormals(normals.size());
	memcpy(eigenNormals.data(), normals.data(), normals.size() * sizeof(float3));

	std::vector<Eigen::Vector4f> eigenColors(colors.size());
	memcpy(eigenColors.data(), colors.data(), colors.size() * sizeof(float4));

	VD::AddDiskBatch(tag, eigenPositions, eigenNormals, radius, slices, eigenColors, isBillboard);
}

void IVisualDebugging::AddCylinder(
	const std::string& tag,
	const float3& center,
	const float3& normal,
	float radius,
	float height,
	const float4& color)
{
	VD::AddCylinder(tag, { XYZ(center) }, { XYZ(normal) }, radius, height, { XYZW(color) });
}

void IVisualDebugging::AddCone(
	const std::string& tag,
	const float3& center,
	const float3& normal,
	float radius,
	float height,
	const float4& color)
{
	VD::AddCone(tag, { XYZ(center) }, { XYZ(normal) }, radius, height, { XYZW(color) });
}

void IVisualDebugging::AddCapsule(
	const std::string& tag,
	const float3& center,
	const float3& normal,
	float radius,
	float height,
	unsigned int rings,
	const float4& color)
{
	VD::AddCapsule(tag, { XYZ(center) }, { XYZ(normal) }, radius, height, rings, { XYZW(color) });
}

void IVisualDebugging::AddTorus(
	const std::string& tag,
	const float3& center,
	const float3& normal,
	float majorRadius,
	float minorRadius,
	unsigned int majorSegments,
	unsigned int minorSegments,
	const float4& color)
{
	VD::AddTorus(tag, { XYZ(center) }, { XYZ(normal) }, majorRadius, minorRadius, majorSegments, minorSegments, { XYZW(color) });
}

void IVisualDebugging::AddTube(
	const std::string& tag,
	const float3& center,
	const float3& normal,
	float radius,
	unsigned int curveSegments,
	unsigned int radialSegments,
	const float4& color)
{
	VD::AddTube(tag, { XYZ(center) }, { XYZ(normal) }, radius, curveSegments, radialSegments, { XYZW(color) });
}

void IVisualDebugging::AddArrow(
	const std::string& tag,
	const float3& start,
	const float3& direction,
	float length,
	const float4& color)
{
	VD::AddArrow(tag, { XYZ(start) }, { XYZ(direction) }, length, { XYZW(color) });
}

void IVisualDebugging::AddGrid(
	const std::string& tag,
	const float3& center,
	const float3& normal,
	int divisions,
	float spacing,
	const float4& color)
{
	VD::AddGrid(tag, { XYZ(center) }, { XYZ(normal) }, divisions, spacing, { XYZW(color) });
}

void IVisualDebugging::ClearSelectionList()
{
	VD::ClearSelectionList();
}

void IVisualDebugging::AddToSelectionList(const std::string& tag)
{
	VD::AddToSelectionList(tag);
}

unsigned int IVisualDebugging::ShowNextSelection()
{
	return VD::ShowNextSelection();
}

unsigned int IVisualDebugging::ShowPreviousSelection()
{
	return VD::ShowPreviousSelection();
}

float3 IVisualDebugging::GetInstanceScale(const std::string& tag, size_t index)
{
	auto scale = VD::GetInstanceScale(tag, index);
	return { XYZ_(scale) };
}

void IVisualDebugging::SetInstanceScale(const std::string& tag, const float3& newScale, size_t index)
{
	VD::SetInstanceScale(tag, { XYZ(newScale) }, index);
}

void IVisualDebugging::SetInstanceScale(const std::string& tag, const float3& newScale)
{
	VD::SetInstanceScale(tag, { XYZ(newScale) });
}
