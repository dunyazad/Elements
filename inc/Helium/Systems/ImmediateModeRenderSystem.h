#pragma once

#include <Helium/Systems/HeliumSystem.h>

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

private:
	bool enabled = true;

	bool axisGizmoEnabled = true;
	bool centerGizmoEnabled = true;
};
