#pragma once

#include <Helium/Systems/HeliumSystem.h>

#include <Eigen/Dense>

class HeliumCore;

class RenderSystem : public HeliumSystem
{
	public:
	RenderSystem(HeliumCore* core);
	virtual ~RenderSystem();
	void Initialize() override;
	void Update(float dt) override;
	void Render() override;
	void Shutdown() override;

protected:
	Eigen::Matrix4f viewMatrix = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f perspectiveMatrix = Eigen::Matrix4f::Identity();
	//Eigen::Vector3f eye = Eigen::Vector3f::Zero();
};
