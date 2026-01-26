#pragma once

#include <Helium/Systems/HeliumSystem.h>

#include <Eigen/Dense>

class HeliumCore;

class HELIUM_API ImmediateModeRenderSystem : public HeliumSystem
{
public:
	ImmediateModeRenderSystem(HeliumCore* core);
	virtual ~ImmediateModeRenderSystem();

	void Initialize() override;
	void Update(float dt) override;

	inline bool IsEnabled() const { return enabled; }
	inline void SetEnable(bool enable) { enabled = enable; }
	inline void ToggleEnable() { enabled = !enabled; }

	inline bool IsAxisGizmoEnabled() const { return axisGizmoEnabled; }
	inline void SetAxisGizmoEnabled(bool enable) { axisGizmoEnabled = enable; }
	inline void ToggleAxisGizmo() { axisGizmoEnabled = !axisGizmoEnabled; }

	inline bool IsCenterGizmoEnabled() const { return centerGizmoEnabled; }
	inline void SetCenterGizmoEnabled(bool enable) { centerGizmoEnabled = enable; }
	inline void ToggleCenterGizmo() { centerGizmoEnabled = !centerGizmoEnabled; }

	inline bool IsGridGizmoEnabled() const { return gridGizmoEnabled; }
	inline void SetGridGizmoEnabled(bool enable) { gridGizmoEnabled = enable; }
	inline void ToggleGridGizmo() { gridGizmoEnabled = !gridGizmoEnabled; }
	inline Eigen::Vector4f GetGridGizmoColor() const { return gridGizmoColor; }
	inline void SetGridGizmoColor(const Eigen::Vector4f& color) { gridGizmoColor = color; }
	inline Eigen::Vector3f GetGridNormal() const { return gridNormal; }
	inline void SetGridNormal(const Eigen::Vector3f& normal) { gridNormal = normal.normalized(); }
	inline float GetGridSpacing() const { return gridSpacing; }
	inline void SetGridSpacing(float spacing) { gridSpacing = spacing; }

private:
	bool enabled = true;

	bool axisGizmoEnabled = true;
	bool centerGizmoEnabled = true;
	bool gridGizmoEnabled = false;
	Eigen::Vector4f gridGizmoColor = Eigen::Vector4f(0.5f, 0.5f, 0.5f, 1.0f);
	Eigen::Vector3f gridNormal = Eigen::Vector3f::UnitY();
	float gridSpacing = 10.0f;
};
