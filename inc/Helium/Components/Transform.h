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
	void SetLocalTransformMatrix(const Eigen::Matrix4f& m); // 구현부로 이동 (dirty 처리를 위해)

	inline const Eigen::Matrix4f& GetAbsoluteTransformMatrix() const { return absoluteTransformMatrix; }
	void SetAbsoluteTransformMatrix(const Eigen::Matrix4f& m); // 구현부로 이동

	void SetLocalPosition(const Eigen::Vector3f& position);
	void SetLocalRotation(const Eigen::Quaternionf& rotation);

	Eigen::Vector3f GetLocalPosition() const;
	Eigen::Quaternionf GetLocalRotation() const;
	Eigen::Vector3f GetLocalScale() const;

	void UpdateAbsoluteTransformMatrix();

private:
	bool dirty = true;
	Transform* parent = nullptr;
	std::set<Transform*> children;

	Eigen::Matrix4f localTransformMatrix = Eigen::Matrix4f::Identity();
	Eigen::Matrix4f absoluteTransformMatrix = Eigen::Matrix4f::Identity();
};