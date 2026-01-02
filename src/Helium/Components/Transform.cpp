#include "pch.h"

#include <Helium/Components/Transform.h>

Transform::Transform()
	: dirty(true)
	, parent(nullptr)
{
}

Transform::~Transform()
{
	if (parent)
	{
		parent->children.erase(this);
	}

	for (Transform* child : children)
	{
		child->parent = nullptr;
		child->dirty = true;
	}
}

Transform* Transform::GetParent() const
{
	return parent;
}

void Transform::SetParent(Transform* newParent)
{
	if (parent == newParent) return;
	if (newParent == this) return;

	if (parent)
	{
		parent->children.erase(this);
	}

	parent = newParent;

	if (parent)
	{
		parent->children.insert(this);
	}

	dirty = true;
}

void Transform::AddChild(Transform* child)
{
	if (child)
	{
		child->SetParent(this);
	}
}

void Transform::RemoveChild(Transform* child)
{
	if (child && child->GetParent() == this)
	{
		child->SetParent(nullptr);
	}
}

void Transform::SetLocalTransformMatrix(const Eigen::Matrix4f& m)
{
	localTransformMatrix = m;
	dirty = true;
}

void Transform::SetAbsoluteTransformMatrix(const Eigen::Matrix4f& m)
{
	absoluteTransformMatrix = m;
	dirty = true;
}

void Transform::SetLocalPosition(const Eigen::Vector3f& position)
{
	localTransformMatrix.block<3, 1>(0, 3) = position;
	dirty = true;
}

void Transform::SetLocalRotation(const Eigen::Quaternionf& rotation)
{
	// 기존 스케일 추출 (행렬의 각 열 벡터의 길이)
	Eigen::Vector3f scale;
	scale.x() = localTransformMatrix.block<3, 1>(0, 0).norm();
	scale.y() = localTransformMatrix.block<3, 1>(0, 1).norm();
	scale.z() = localTransformMatrix.block<3, 1>(0, 2).norm();

	// 회전 행렬 생성
	Eigen::Matrix3f rotationMatrix = rotation.toRotationMatrix();

	// 회전 * 스케일 적용하여 3x3 영역 갱신
	localTransformMatrix.block<3, 1>(0, 0) = rotationMatrix.col(0) * scale.x();
	localTransformMatrix.block<3, 1>(0, 1) = rotationMatrix.col(1) * scale.y();
	localTransformMatrix.block<3, 1>(0, 2) = rotationMatrix.col(2) * scale.z();

	dirty = true;
}

Eigen::Vector3f Transform::GetLocalPosition() const
{
	return localTransformMatrix.block<3, 1>(0, 3);
}

Eigen::Quaternionf Transform::GetLocalRotation() const
{
	// 3x3 행렬 부분에서 회전 추출 (Scale 정규화 필요)
	Eigen::Matrix3f rotationMatrix = localTransformMatrix.block<3, 3>(0, 0);

	rotationMatrix.col(0).normalize();
	rotationMatrix.col(1).normalize();
	rotationMatrix.col(2).normalize();

	return Eigen::Quaternionf(rotationMatrix);
}

Eigen::Vector3f Transform::GetLocalScale() const
{
	return Eigen::Vector3f(
		localTransformMatrix.block<3, 1>(0, 0).norm(),
		localTransformMatrix.block<3, 1>(0, 1).norm(),
		localTransformMatrix.block<3, 1>(0, 2).norm()
	);
}

void Transform::UpdateAbsoluteTransformMatrix()
{
	if (parent)
	{
		absoluteTransformMatrix = parent->GetAbsoluteTransformMatrix() * localTransformMatrix;
	}
	else
	{
		absoluteTransformMatrix = localTransformMatrix;
	}

	dirty = false;

	for (Transform* child : children)
	{
		child->UpdateAbsoluteTransformMatrix();
	}
}