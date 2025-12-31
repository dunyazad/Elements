#pragma once

#include <Helium/Systems/HeliumSystem.h>

class HeliumCore;

class ImmediateModeRenderSystem : public HeliumSystem
{
public:
	ImmediateModeRenderSystem(HeliumCore* core);
	virtual ~ImmediateModeRenderSystem();

	void Initialize() override;
	void Update(float dt) override;

	inline bool IsEnabled() const { return enabled; }
	inline void SetEnable(bool enable) { enabled = enable; }
	inline void ToggleEnable() { enabled = !enabled; }

private:
	bool enabled = true;
};
