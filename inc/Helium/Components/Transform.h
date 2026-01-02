#pragma once

#include <Eigen/Dense>
#include <set>

class Transform
{
public:
	Transform();
	~Transform();

	Transform* GetParent() const;
	void SetParent(Transform* transform);

	void AddChild(Transform* child);
	void RemoveChild(Transform* child);

	inline const Eigen::Matrix4f& GetLocalTransformMatrix() const { return localTransformMatrix; }
	inline void SetLocalTransformMatrix(const Eigen::Matrix4f& m) { localTransformMatrix = m; dirty = true; }

	inline const Eigen::Matrix4f& GetAbsoluteTransformMatrix() const { return absoluteTransformMatrix; }
	inline void SetAbsoluteTransformMatrix(const Eigen::Matrix4f& m) { absoluteTransformMatrix = m; dirty = true; }

	void UpdateAbsoluteTransformMatrix();

private:
	bool dirty = true;
	Transform* parent = nullptr;
	std::set<Transform*> children;

	Eigen::Matrix4f localTransformMatrix = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f absoluteTransformMatrix = Eigen::Matrix4f::Identity();
};
