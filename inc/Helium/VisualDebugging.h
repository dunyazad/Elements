#pragma once

#include <map>
#include <mutex>
#include <vector>
#include <functional>
#include <string>

#include <entt/entt.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <Helium/TypeDefinitions.h>
#include <Helium/Color.hpp>

using Entity = entt::entity;

class DebuggingRenderable;
class TextBlock;

class HELIUM_API VisualDebugging
{
public:
	static VisualDebugging& Instance()
	{
		static VisualDebugging instance;
		return instance;
	}

	static void Initialize();
	static void Terminate();
	static void DispatchCommands();

	static void CreateLineEntity(const std::string& tag);
	static void CreateTriangleEntity(const std::string& tag);
	static void CreateBoxEntity(const std::string& tag); // Box는 보통 1x1x1 고정
	static void CreateWiredBoxEntity(const std::string& tag);
	static void CreateSphereEntity(const std::string& tag, float radius, unsigned int slices, unsigned int stacks);
	static void CreateDiskEntity(const std::string& tag, float radius, unsigned int slices, bool isBillboard = false);
	static void CreateCylinderEntity(const std::string& tag, float radius, float height, unsigned int slices);
	static void CreateConeEntity(const std::string& tag, float radius, float height, unsigned int slices);
	static void CreateCapsuleEntity(const std::string& tag, float radius, float height, unsigned int rings);
	static void CreateTorusEntity(const std::string& tag, float majorRadius, float minorRadius, unsigned int majorSegments, unsigned int minorSegments);
	static void CreateTubeEntity(const std::string& tag, float radius, unsigned int curveSegments, unsigned int radialSegments);
	static void CreateArrowEntity(const std::string& tag);

	static void Clear(const std::string& tag);
	static void ClearAll();

	static void SetVisibility(bool visible, const std::string& tag);
	static void SetVisibilityAll(bool visible);
	static void ToggleVisibility(const std::string& tag);
	static void ToggleVisibilityAll();

	static void AddGeometryInstance(
		const std::string& tag,
		std::function<void(const std::string&)> createEntityFunc,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		const Eigen::Vector3f& scale,
		const Eigen::Vector4f& color);

	static void AddLine(
		const std::string& tag,
		const Eigen::Vector3f& v0,
		const Eigen::Vector3f& v1,
		const Eigen::Vector4f& c);

	static void AddLine(
		const std::string& tag,
		const Eigen::Vector3f& v0,
		const Eigen::Vector3f& v1,
		const Eigen::Vector4f& c0,
		const Eigen::Vector4f& c1);

	static void AddTriangle(
		const std::string& tag,
		const Eigen::Vector3f& v0,
		const Eigen::Vector3f& v1,
		const Eigen::Vector3f& v2,
		const Eigen::Vector4f& c);

	static void AddTriangle(
		const std::string& tag,
		const Eigen::Vector3f& v0,
		const Eigen::Vector3f& v1,
		const Eigen::Vector3f& v2,
		const Eigen::Vector4f& c0,
		const Eigen::Vector4f& c1,
		const Eigen::Vector4f& c2);

	static void AddBox(
		const std::string& tag,
		const AABB& aabb,
		const Eigen::Vector4f& color);

	static void AddBox(
		const std::string& tag,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& dimensions,
		const Eigen::Vector4f& color);

	static void AddBox(
		const std::string& tag,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		const Eigen::Vector3f& dimensions,
		const Eigen::Vector4f& color);

	static void AddWiredBox(
		const std::string& tag,
		const AABB& aabb,
		const Eigen::Vector4f& color);

	static void AddWiredBox(
		const std::string& tag,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& dimensions,
		const Eigen::Vector4f& color);

	static void AddWiredBox(
		const std::string& tag,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		const Eigen::Vector3f& dimensions,
		const Eigen::Vector4f& color);

	static void AddWiredBoxBatch(
		const std::string& tag,
		const std::vector<Eigen::Vector3f>& centers,
		const Eigen::Vector3f& dimensions,
		const Eigen::Vector4f& color);

	static void AddWiredBoxBatch(
		const std::string& tag,
		const std::vector<Eigen::Vector3f>& centers,
		const std::vector <Eigen::Vector3f>& dimensions,
		const std::vector<Eigen::Vector4f>& colors);

	static void AddSphere(
		const std::string& tag,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		float radius,
		const Eigen::Vector4f& color);

	static void AddSphereBatch(
		const std::string& tag,
		const std::vector<Eigen::Vector3f>& centers,
		float radius,
		const Eigen::Vector4f& color);

	static void AddSphereBatch(
		const std::string& tag,
		const std::vector<Eigen::Vector3f>& centers,
		const std::vector<Eigen::Vector3f>& normals,
		float radius,
		const Eigen::Vector4f& color);

	static void AddSphereBatch(
		const std::string& tag,
		const std::vector<Eigen::Vector3f>& centers,
		float radius,
		const std::vector<Eigen::Vector4f>& colors);

	static void AddSphereBatch(
		const std::string& tag,
		const std::vector<Eigen::Vector3f>& centers,
		const std::vector<Eigen::Vector3f>& normals,
		float radius,
		const std::vector<Eigen::Vector4f>& colors);

	static void AddDisk(
		const std::string& tag,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		float radius,
		unsigned int slices,
		const Eigen::Vector4f& color,
		bool isBillboard = false);

	static void AddDiskBatch(
		const std::string& tag,
		const std::vector<Eigen::Vector3f>& positions,
		const std::vector<Eigen::Vector3f>& normals,
		float radius,
		unsigned int slices,
		const Eigen::Vector4f& color,
		bool isBillboard = false);

	static void AddDiskBatch(
		const std::string& tag,
		const std::vector<Eigen::Vector3f>& positions,
		const std::vector<Eigen::Vector3f>& normals,
		float radius,
		unsigned int slices,
		const std::vector<Eigen::Vector4f>& colors,
		bool isBillboard = false);

	static void AddCylinder(
		const std::string& tag,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		float radius,
		float height,
		const Eigen::Vector4f& color);

	static void AddCone(
		const std::string& tag,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		float radius,
		float height,
		const Eigen::Vector4f& color);

	static void AddCapsule(
		const std::string& tag,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		float radius,
		float height,
		unsigned int rings,
		const Eigen::Vector4f& color);

	static void AddTorus(
		const std::string& tag,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		float majorRadius,
		float minorRadius,
		unsigned int majorSegments,
		unsigned int minorSegments,
		const Eigen::Vector4f& color);

	static void AddTube(
		const std::string& tag,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		float radius,
		unsigned int curveSegments,
		unsigned int radialSegments,
		const Eigen::Vector4f& color);

	static void AddArrow(
		const std::string& tag,
		const Eigen::Vector3f& start,
		const Eigen::Vector3f& direction,
		float length,
		const Eigen::Vector4f& color);

	static void AddFrustum(
		const std::string& tag,
		const Eigen::Matrix4f& invViewProj,
		const Eigen::Vector4f& color);

	static void AddFrustum(
		const std::string& tag,
		const Eigen::Matrix4f& viewMatrix,
		float fovDegrees,
		float aspectRatio,
		float nearPlane,
		float farPlane,
		const Eigen::Vector4f& color);

	static void AddGrid(
		const std::string& tag,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		int divisions,
		float spacing,
		const Eigen::Vector4f& color);

	static void ClearSelectionList();
	static void AddToSelectionList(const std::string& tag);
	static unsigned int ShowNextSelection();
	static unsigned int ShowPreviousSelection();

	static Eigen::Vector3f GetInstanceScale(const std::string& tag, size_t index = 0);
	static void SetInstanceScale(const std::string& tag, const Eigen::Vector3f& newScale, size_t index);
	static void SetInstanceScale(const std::string& tag, const Eigen::Vector3f& newScale);

private:
	static bool initialized;

	static std::mutex commandMutex;
	static std::vector<std::function<void()>> commandQueue;
	static std::vector<std::function<void()>> pendingCommands;

	static std::map<std::string, Entity> entities;
	static std::map<std::string, DebuggingRenderable*> debuggingRenderables;
	static std::map<std::string, TextBlock*> textBlocks;

	static std::vector<std::string> selectionRenderables;
	static unsigned int selectionIndex;
};
