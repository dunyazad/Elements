#pragma once

//#include "glm_include.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/LU>

struct Ray
{
	Eigen::Vector3f origin;
	Eigen::Vector3f direction;
	Eigen::Vector3f inverseDirection;

	Ray(const Eigen::Vector3f& o, const Eigen::Vector3f& d) : origin(o), direction(d) {
		const float epsilon = 1e-6f;

		inverseDirection.x() = (std::abs(direction.x()) < epsilon) ? ((direction.x() >= 0) ? 1e20f : -1e20f) : (1.0f / direction.x());
		inverseDirection.y() = (std::abs(direction.y()) < epsilon) ? ((direction.y() >= 0) ? 1e20f : -1e20f) : (1.0f / direction.y());
		inverseDirection.z() = (std::abs(direction.z()) < epsilon) ? ((direction.z() >= 0) ? 1e20f : -1e20f) : (1.0f / direction.z());
	}

	inline bool IntersectSphere(const Eigen::Vector3f& sphereCenter, float radius, float& t) const
	{
		Eigen::Vector3f m = origin - sphereCenter;
		float b = m.dot(direction);
		float c = m.dot(m) - radius * radius;

		if (c > 0.0f && b > 0.0f) return false;

		float discr = b * b - c;

		if (discr < 0.0f) return false;

		t = -b - std::sqrt(discr);

		if (t < 0.0f) t = -b + std::sqrt(discr);

		return t >= 0.0f;
	}

	inline bool IntersectPlane(const Eigen::Vector3f& planeNormal, float planeD, float& t) const
	{
		float denom = planeNormal.dot(direction);
		if (std::abs(denom) < 1e-6f) return false;
		t = -(planeNormal.dot(origin) + planeD) / denom;
		return t >= 0.0f;
	}

	inline bool IntersectTriangle(const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, float& t) const
	{
		Eigen::Vector3f edge1 = v1 - v0;
		Eigen::Vector3f edge2 = v2 - v0;
		Eigen::Vector3f h = direction.cross(edge2);
		float a = edge1.dot(h);
		if (std::abs(a) < 1e-6f) return false;
		float f = 1.0f / a;
		Eigen::Vector3f s = origin - v0;
		float u = f * s.dot(h);
		if (u < 0.0f || u > 1.0f) return false;
		Eigen::Vector3f q = s.cross(edge1);
		float v = f * direction.dot(q);
		if (v < 0.0f || u + v > 1.0f) return false;
		t = f * edge2.dot(q);
		return t >= 0.0f;
	}
};

struct AABB
{
	Eigen::Vector3f min = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	Eigen::Vector3f max = Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	inline bool Intersects(const AABB& other) const
	{
		if (max.x() < other.min.x() || min.x() > other.max.x()) return false;
		if (max.y() < other.min.y() || min.y() > other.max.y()) return false;
		if (max.z() < other.min.z() || min.z() > other.max.z()) return false;

		return true;
	}

	inline bool Contains(const Eigen::Vector3f& p) const
	{
		return
			p.x() >= min.x() && p.x() <= max.x() &&
			p.y() >= min.y() && p.y() <= max.y() &&
			p.z() >= min.z() && p.z() <= max.z();
	}

	inline void Expand(const Eigen::Vector3f& p)
	{
		min = min.cwiseMin(p);
		max = max.cwiseMax(p);
	}

	inline void Expand(const AABB& other)
	{
		min = min.cwiseMin(other.min);
		max = max.cwiseMax(other.max);
	}

	inline bool IntersectRay(const Ray& ray, float& tNear, float& tFar) const
	{
		Eigen::Vector3f t0 = (min - ray.origin).cwiseProduct(ray.inverseDirection);
		Eigen::Vector3f t1 = (max - ray.origin).cwiseProduct(ray.inverseDirection);

		Eigen::Vector3f tMin = t0.cwiseMin(t1);
		Eigen::Vector3f tMax = t0.cwiseMax(t1);

		tNear = tMin.maxCoeff();
		tFar = tMax.minCoeff();

		return tNear <= tFar && tFar >= 0.0f;
	}
};
