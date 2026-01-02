#pragma once

#include <map>
#include <mutex>

#include <entt/entt.hpp>

#include <Helium/TypeDefinitions.h>
#include <Helium/Color.hpp>

using Entity = entt::entity;

class DebuggingRenderable;
class TextBlock;

class VisualDebugging
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
	static void CreateBoxEntity(const std::string& tag);
	static void CreateWiredBoxEntity(const std::string& tag);
	static void CreateSphereEntity(const std::string& tag);

	static void Clear(const std::string& tag);
	static void ClearAll();
	static void SetVisibility(bool visible, const std::string& tag);
	static void SetVisibilityAll(bool visible);
	static void ToggleVisibility(const std::string& tag);
	static void ToggleVisibilityAll();

	static void AddLine(const std::string& tag, const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector4f& c);
	static void AddLine(const std::string& tag, const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector4f& c0, const Eigen::Vector4f& c1);

	static void AddTriangle(const std::string& tag, const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, const Eigen::Vector4f& c);
	static void AddTriangle(const std::string& tag, const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, const Eigen::Vector4f& c0, const Eigen::Vector4f& c1, const Eigen::Vector4f& c2);

	static void AddBox(const std::string& tag, const AABB& aabb, const Eigen::Vector4f& color);
	static void AddBox(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& dimensions, const Eigen::Vector4f& color);
	static void AddBox(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, const Eigen::Vector3f& dimensions, const Eigen::Vector4f& color);
	
	static void AddWiredBox(const std::string& tag, const AABB& aabb, const Eigen::Vector4f& color);
	static void AddWiredBox(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& dimensions, const Eigen::Vector4f& color);
	static void AddWiredBox(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, const Eigen::Vector3f& dimensions, const Eigen::Vector4f& color);
	
	static void AddSphere(const std::string& tag, const Eigen::Vector3f& center, float radius, const Eigen::Vector4f& color);
	static void AddSphere(const std::string& tag, const Eigen::Vector3f& center, const Eigen::Vector3f& normal, float radius, const Eigen::Vector4f& color);
	
	static void ClearSelectionList();
	static void AddToSelectionList(const std::string& tag);
	static unsigned int ShowNextSelection();
	static unsigned int ShowPreviousSelection();

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
