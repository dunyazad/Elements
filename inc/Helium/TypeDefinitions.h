#pragma once

//#include "glm_include.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/LU>

typedef char i8;
typedef short i16;
typedef int i32;
typedef long i64;

typedef unsigned char ui8;
typedef unsigned short ui16;
typedef unsigned int ui32;
typedef unsigned long ui64;

typedef float f32;
typedef double f64;

#ifndef PRIMITIVE_MAX
#define PRIMITIVE_MAX
#define i8_max (INT8_MAX)
#define i8_min (-INT8_MAX)
#define i16_max (INT16_MAX)
#define i16_min (-INT16_MAX)
#define i32_max (INT32_MAX)
#define i32_min (-INT32_MAX)
#define i64_max (INT64_MAX)
#define i64_min (-INT64_MAX)

#define ui8_max (UINT8_MAX)
#define ui16_max (UINT16_MAX)
#define ui32_max (UINT32_MAX)
#define ui64_max (UINT64_MAX)

#define f32_max (FLT_MAX)
#define f32_min (-FLT_MAX)
#define f64_max (DBL_MAX)
#define f64_min (-DBL_MAX)
#endif

#ifndef PRIMITIVE_INVALID
#define PRIMITIVE_INVALID
#define invalid_i8 (i8_max)
#define invalid_i16 (i16_max)
#define invalid_i32 (i32_max)
#define invalid_i64 (i64_max)

#define invalid_ui8 (ui8_max)
#define invalid_ui16 (ui16_max)
#define invalid_ui32 (ui32_max)
#define invalid_ui64 (ui64_max)

#define invalid_f32 (f32_max)
#define invalid_f64 (f64_max)
#endif

struct f2
{
	f32 x = 0.0f;
	f32 y = 0.0f;

	static f2 Zero() { return f2{ 0.0f, 0.0f }; }
	static f2 Invalid() { return f2{ invalid_f32, invalid_f32 }; }
};

struct f3
{
	f32 x = 0.0f;
	f32 y = 0.0f;
	f32 z = 0.0f;

	static f3 Zero() { return f3{ 0.0f, 0.0f, 0.0f }; }
	static f3 Invalid() { return f3{ invalid_f32, invalid_f32, invalid_f32 }; }
};

struct f4
{
	f32 x = 0.0f;
	f32 y = 0.0f;
	f32 z = 0.0f;
	f32 w = 0.0f;

	static f4 Zero() { return f4{ 0.0f, 0.0f, 0.0f, 0.0f }; }
	static f4 Invalid() { return f4{ invalid_f32, invalid_f32, invalid_f32, invalid_f32 }; }
};

struct d2
{
	f64 x = 0.0;
	f64 y = 0.0;

	static d2 Zero() { return d2{ 0.0, 0.0 }; }
	static d2 Invalid() { return d2{ invalid_f64, invalid_f64 }; }
};

struct d3
{
	f64 x = 0.0;
	f64 y = 0.0;
	f64 z = 0.0;

	static d3 Zero() { return d3{ 0.0, 0.0, 0.0 }; }
	static d3 Invalid() { return d3{ invalid_f64, invalid_f64, invalid_f64 }; }
};

struct d4
{
	f64 x = 0.0;
	f64 y = 0.0;
	f64 z = 0.0;
	f64 w = 0.0;

	static d4 Zero() { return d4{ 0.0, 0.0, 0.0, 0.0 }; }
	static d4 Invalid() { return d4{ invalid_f64, invalid_f64, invalid_f64, invalid_f64 }; }
};

struct FrameEvent
{
	ui32 frameNo;
	f32 timeDelta;
};

struct FrameBufferResizeEvent
{
	i32 width = 0;
	i32 height = 0;
};

struct KeyEvent
{
	i32 keyCode = 0;
	i32 scanCode = 0;
	i32 action = 0;
	i32 mods = 0;
};

struct MousePositionEvent
{
	f64 xpos = 0.0;
	f64 ypos = 0.0;
};

struct MouseButtonEvent
{
	i32 button = 0;
	i32 action = 0;
	i32 mods = 0;
	f64 xpos = 0.0;
	f64 ypos = 0.0;
};

struct MouseWheelEvent
{
	f64 xoffset = 0.0;
	f64 yoffset = 0.0;
};

struct JoystickEvent
{
	float AxisX = 0.0f;
	float AxisY = 0.0f;
	float AxisZ = 0.0f;
	float RotX = 0.0f;
	float RotY = 0.0f;
	float RotZ = 0.0f;
	bool Buttons[16] = {
		false, false, false, false,
		false, false, false, false,
		false, false, false, false,
		false, false, false, false };
};

struct PointP
{
	f3 position = { 0.0f, 0.0f, 0.0f };
};

struct PointPN
{
	f3 position = { 0.0f, 0.0f, 0.0f };
	f3 normal = { 0.0f, 0.0f, 0.0f };
};

struct PointPNC
{
	f3 position = { 0.0f, 0.0f, 0.0f };
	f3 normal = { 0.0f, 0.0f, 0.0f };
	f3 color = { 1.0f, 1.0f, 1.0f };
};

struct Ray
{
	glm::vec3 origin;
	glm::vec3 direction;
	glm::vec3 inverseDirection;

	Ray(const glm::vec3& o, const glm::vec3& d) : origin(o), direction(d) {
		const float epsilon = 1e-6f;

		inverseDirection.x = (std::abs(direction.x) < epsilon) ? ((direction.x >= 0) ? 1e20f : -1e20f) : (1.0f / direction.x);
		inverseDirection.y = (std::abs(direction.y) < epsilon) ? ((direction.y >= 0) ? 1e20f : -1e20f) : (1.0f / direction.y);
		inverseDirection.z = (std::abs(direction.z) < epsilon) ? ((direction.z >= 0) ? 1e20f : -1e20f) : (1.0f / direction.z);
	}

	inline bool IntersectSphere(const glm::vec3& sphereCenter, float radius, float& t) const
	{
		glm::vec3 m = origin - sphereCenter;
		float b = glm::dot(m, direction);
		float c = glm::dot(m, m) - radius * radius;

		if (c > 0.0f && b > 0.0f) return false;

		float discr = b * b - c;

		if (discr < 0.0f) return false;

		t = -b - std::sqrt(discr);

		if (t < 0.0f) t = -b + std::sqrt(discr);

		return t >= 0.0f;
	}
};

struct AABB
{
	glm::vec3 min = glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX);
	glm::vec3 max = glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	inline bool Intersects(const AABB& other) const
	{
		if (max.x < other.min.x || min.x > other.max.x) return false;
		if (max.y < other.min.y || min.y > other.max.y) return false;
		if (max.z < other.min.z || min.z > other.max.z) return false;

		return true;
	}

	inline bool Contains(const glm::vec3& p) const
	{
		return
			p.x >= min.x && p.x <= max.x &&
			p.y >= min.y && p.y <= max.y &&
			p.z >= min.z && p.z <= max.z;
	}

	inline void Expand(const glm::vec3& p)
	{
		min = glm::min(min, p);
		max = glm::max(max, p);
	}

	inline void Expand(const AABB& other)
	{
		min = glm::min(min, other.min);
		max = glm::max(max, other.max);
	}

	inline bool IntersectRay(const Ray& ray, float& tNear, float& tFar) const
	{
		glm::vec3 t0 = (min - ray.origin) * ray.inverseDirection;
		glm::vec3 t1 = (max - ray.origin) * ray.inverseDirection;
		glm::vec3 tMin = glm::min(t0, t1);
		glm::vec3 tMax = glm::max(t0, t1);

		tNear = std::max(std::max(tMin.x, tMin.y), tMin.z);
		tFar = std::min(std::min(tMax.x, tMax.y), tMax.z);

		return tNear <= tFar && tFar >= 0.0f;
	}
};

namespace Eigen
{
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
}