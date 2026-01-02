#pragma once

#include <Helium/Systems/HeliumSystem.h>

#include <map>

#include <Eigen/Dense>

class HeliumCore;

class Camera;
class Renderable;
class DebuggingRenderable;
class Shader;
class Texture;

class RenderSystem : public HeliumSystem
{
public:
	RenderSystem(HeliumCore* core);
	virtual ~RenderSystem();
	void Initialize() override;
	void Update(float dt) override;
	void Render() override;
	void Shutdown() override;

	void RenderRenderables(
		float timeDelta,
		const Eigen::Matrix4f& viewMatrix,
		const Eigen::Matrix4f& perspectiveMatrix,
		const Eigen::Vector3f& eye,
		const Eigen::Vector4f& lightVector,
		const std::map<Shader*, std::vector<Renderable*>>& shadermap
	);

	void RenderDebuggingRenderables(
		float timeDelta,
		const Eigen::Matrix4f& viewMatrix,
		const Eigen::Matrix4f& perspectiveMatrix,
		const Eigen::Vector3f& eye,
		const Eigen::Vector4f& lightVector,
		const std::map<Shader*, std::vector<DebuggingRenderable*>>& shadermap
	);

	inline void SetActiveCamera(Camera* camera) { activeCamera = camera; }
	inline Camera* GetActiveCamera() const { return activeCamera; }

private:
	Camera* activeCamera = nullptr;
};
