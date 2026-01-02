#include "pch.h"

#include <Helium/Components/Transform.h>

Transform::Transform()
{
}

Transform::~Transform()
{
}

Transform* Transform::GetParent() const
{
	return parent;
}

void Transform::SetParent(Transform* transform)
{
	if (parent)
	{
		parent->RemoveChild(this);
	}
	parent = transform;
	transform->AddChild(this);

	dirty = true;
}

void Transform::AddChild(Transform* child)
{
	if (child->parent && child->parent != this)
	{
		child->parent->RemoveChild(child);
	}
	children.insert(child);
	child->parent = this;

	child->dirty = true;
	dirty = true;
}

void Transform::RemoveChild(Transform* child)
{
	children.erase(child);
	child->parent = nullptr;

	child->dirty = true;
	dirty = true;
}

void Transform::UpdateAbsoluteTransformMatrix()
{
	if (parent && parent->dirty)
	{
		dirty = true;
	}

	if (!dirty) return;

	if (parent)
	{
		absoluteTransformMatrix = parent->absoluteTransformMatrix * localTransformMatrix;
	}
	else
	{
		absoluteTransformMatrix = localTransformMatrix;
	}

	for (auto child : children)
	{
		child->UpdateAbsoluteTransformMatrix();
	}

	dirty = false;
}