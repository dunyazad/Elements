#include "pch.h"
#include <Helium/GeometryBuilder.h>
#include <Helium/Components/Renderable.h>
#include <vector>
#include <cmath>
#include <algorithm> // std::max

namespace
{
	const float PI = 3.14159265359f;
	const float TWO_PI = 2.0f * PI;
	const float HALF_PI = PI * 0.5f;

	template <typename T>
	void AddVertexToBuffer(T* target, const Eigen::Vector3f& position, const Eigen::Vector3f& normal, const Eigen::Vector4f& color)
	{
		target->AddVertex(position);
		target->AddNormal(normal);
		target->AddColor4(color);
	}

	template <typename T>
	void AddIndexToBuffer(T* target, unsigned int index)
	{
		target->AddIndex(index);
	}

	template <typename T>
	unsigned int GetCurrentVertexCount(T* target)
	{
		return (unsigned int)target->GetNumberOfVertices();
	}

	template <typename T>
	void BuildPlaneHelper(
		T* target,
		float width,
		float height,
		unsigned int hSegments,
		unsigned int vSegments,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		const Eigen::Vector4f& color,
		bool wireframe)
	{
		Eigen::Vector3f n = normal.normalized();
		Eigen::Vector3f tangent;

		if (std::abs(n.x()) > 0.9f)
		{
			tangent = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
		}
		else
		{
			tangent = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
		}

		tangent = (tangent - n * tangent.dot(n)).normalized();
		Eigen::Vector3f bitangent = n.cross(tangent);

		unsigned int startVertexIndex = GetCurrentVertexCount(target);
		float halfWidth = width * 0.5f;
		float halfHeight = height * 0.5f;

		for (unsigned int y = 0; y <= vSegments; ++y)
		{
			for (unsigned int x = 0; x <= hSegments; ++x)
			{
				float u = (float)x / (float)hSegments;
				float v = (float)y / (float)vSegments;

				float xPos = (u * width) - halfWidth;
				float yPos = (v * height) - halfHeight;

				Eigen::Vector3f position = center + (tangent * xPos) + (bitangent * yPos);
				AddVertexToBuffer(target, position, n, color);
			}
		}

		for (unsigned int y = 0; y < vSegments; ++y)
		{
			for (unsigned int x = 0; x < hSegments; ++x)
			{
				unsigned int i0 = startVertexIndex + (y * (hSegments + 1)) + x;
				unsigned int i1 = i0 + 1;
				unsigned int i2 = i0 + (hSegments + 1);
				unsigned int i3 = i2 + 1;

				if (wireframe)
				{
					AddIndexToBuffer(target, i0); AddIndexToBuffer(target, i1);
					AddIndexToBuffer(target, i0); AddIndexToBuffer(target, i2);

					if (x == hSegments - 1)
					{
						AddIndexToBuffer(target, i1); AddIndexToBuffer(target, i3);
					}
					if (y == vSegments - 1)
					{
						AddIndexToBuffer(target, i2); AddIndexToBuffer(target, i3);
					}
				}
				else
				{
					AddIndexToBuffer(target, i0);
					AddIndexToBuffer(target, i2);
					AddIndexToBuffer(target, i1);

					AddIndexToBuffer(target, i1);
					AddIndexToBuffer(target, i2);
					AddIndexToBuffer(target, i3);
				}
			}
		}
	}

	template <typename T>
	void BuildBoxHelper(
		T* target,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& dimension,
		const Eigen::Vector4f& color,
		bool wireframe)
	{
		Eigen::Vector3f halfDim = dimension * 0.5f;

		Eigen::Vector3f p0 = center + Eigen::Vector3f(-halfDim.x(), -halfDim.y(), halfDim.z());
		Eigen::Vector3f p1 = center + Eigen::Vector3f(halfDim.x(), -halfDim.y(), halfDim.z());
		Eigen::Vector3f p2 = center + Eigen::Vector3f(halfDim.x(), halfDim.y(), halfDim.z());
		Eigen::Vector3f p3 = center + Eigen::Vector3f(-halfDim.x(), halfDim.y(), halfDim.z());
		Eigen::Vector3f p4 = center + Eigen::Vector3f(-halfDim.x(), -halfDim.y(), -halfDim.z());
		Eigen::Vector3f p5 = center + Eigen::Vector3f(halfDim.x(), -halfDim.y(), -halfDim.z());
		Eigen::Vector3f p6 = center + Eigen::Vector3f(halfDim.x(), halfDim.y(), -halfDim.z());
		Eigen::Vector3f p7 = center + Eigen::Vector3f(-halfDim.x(), halfDim.y(), -halfDim.z());

		if (wireframe)
		{
			unsigned int startIdx = GetCurrentVertexCount(target);

			AddVertexToBuffer(target, p0, Eigen::Vector3f::Zero(), color);
			AddVertexToBuffer(target, p1, Eigen::Vector3f::Zero(), color);
			AddVertexToBuffer(target, p2, Eigen::Vector3f::Zero(), color);
			AddVertexToBuffer(target, p3, Eigen::Vector3f::Zero(), color);
			AddVertexToBuffer(target, p4, Eigen::Vector3f::Zero(), color);
			AddVertexToBuffer(target, p5, Eigen::Vector3f::Zero(), color);
			AddVertexToBuffer(target, p6, Eigen::Vector3f::Zero(), color);
			AddVertexToBuffer(target, p7, Eigen::Vector3f::Zero(), color);

			AddIndexToBuffer(target, startIdx + 0); AddIndexToBuffer(target, startIdx + 1);
			AddIndexToBuffer(target, startIdx + 1); AddIndexToBuffer(target, startIdx + 2);
			AddIndexToBuffer(target, startIdx + 2); AddIndexToBuffer(target, startIdx + 3);
			AddIndexToBuffer(target, startIdx + 3); AddIndexToBuffer(target, startIdx + 0);

			AddIndexToBuffer(target, startIdx + 4); AddIndexToBuffer(target, startIdx + 5);
			AddIndexToBuffer(target, startIdx + 5); AddIndexToBuffer(target, startIdx + 6);
			AddIndexToBuffer(target, startIdx + 6); AddIndexToBuffer(target, startIdx + 7);
			AddIndexToBuffer(target, startIdx + 7); AddIndexToBuffer(target, startIdx + 4);

			AddIndexToBuffer(target, startIdx + 0); AddIndexToBuffer(target, startIdx + 4);
			AddIndexToBuffer(target, startIdx + 1); AddIndexToBuffer(target, startIdx + 5);
			AddIndexToBuffer(target, startIdx + 2); AddIndexToBuffer(target, startIdx + 6);
			AddIndexToBuffer(target, startIdx + 3); AddIndexToBuffer(target, startIdx + 7);
		}
		else
		{
			struct Face
			{
				Eigen::Vector3f v[4];
				Eigen::Vector3f n;
			};

			Face faces[6] = {
				{ {p0, p1, p2, p3}, Eigen::Vector3f(0, 0, 1) },
				{ {p5, p4, p7, p6}, Eigen::Vector3f(0, 0, -1) },
				{ {p4, p0, p3, p7}, Eigen::Vector3f(-1, 0, 0) },
				{ {p1, p5, p6, p2}, Eigen::Vector3f(1, 0, 0) },
				{ {p3, p2, p6, p7}, Eigen::Vector3f(0, 1, 0) },
				{ {p4, p5, p1, p0}, Eigen::Vector3f(0, -1, 0) }
			};

			for (int i = 0; i < 6; ++i)
			{
				unsigned int idx = GetCurrentVertexCount(target);

				AddVertexToBuffer(target, faces[i].v[0], faces[i].n, color);
				AddVertexToBuffer(target, faces[i].v[1], faces[i].n, color);
				AddVertexToBuffer(target, faces[i].v[2], faces[i].n, color);
				AddVertexToBuffer(target, faces[i].v[3], faces[i].n, color);

				AddIndexToBuffer(target, idx + 0); AddIndexToBuffer(target, idx + 1); AddIndexToBuffer(target, idx + 2);
				AddIndexToBuffer(target, idx + 0); AddIndexToBuffer(target, idx + 2); AddIndexToBuffer(target, idx + 3);
			}
		}
	}

	template <typename T>
	void BuildSphereHelper(
		T* target,
		const Eigen::Vector3f& center,
		float radius,
		unsigned int latitudeSegments,
		unsigned int longitudeSegments,
		const Eigen::Vector4f& color,
		bool wireframe)
	{
		unsigned int startIdx = GetCurrentVertexCount(target);

		for (unsigned int y = 0; y <= latitudeSegments; ++y)
		{
			for (unsigned int x = 0; x <= longitudeSegments; ++x)
			{
				float xSegment = (float)x / (float)longitudeSegments;
				float ySegment = (float)y / (float)latitudeSegments;
				float xPos = std::cos(xSegment * 2.0f * PI) * std::sin(ySegment * PI);
				float yPos = std::cos(ySegment * PI);
				float zPos = std::sin(xSegment * 2.0f * PI) * std::sin(ySegment * PI);

				Eigen::Vector3f normal(xPos, yPos, zPos);
				Eigen::Vector3f position = center + (normal * radius);

				AddVertexToBuffer(target, position, normal, color);
			}
		}

		for (unsigned int y = 0; y < latitudeSegments; ++y)
		{
			for (unsigned int x = 0; x < longitudeSegments; ++x)
			{
				unsigned int i0 = startIdx + (y * (longitudeSegments + 1)) + x;
				unsigned int i1 = i0 + 1;
				unsigned int i2 = i0 + (longitudeSegments + 1);
				unsigned int i3 = i2 + 1;

				if (wireframe)
				{
					AddIndexToBuffer(target, i0); AddIndexToBuffer(target, i1);
					AddIndexToBuffer(target, i0); AddIndexToBuffer(target, i2);
				}
				else
				{
					if (y != 0)
					{
						AddIndexToBuffer(target, i0);
						AddIndexToBuffer(target, i1);
						AddIndexToBuffer(target, i2);
					}

					if (y != latitudeSegments - 1)
					{
						AddIndexToBuffer(target, i1);
						AddIndexToBuffer(target, i3);
						AddIndexToBuffer(target, i2);
					}
				}
			}
		}
	}

	template <typename T>
	void BuildDiskHelper(
		T* target,
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& normal,
		float radius,
		unsigned int segments,
		const Eigen::Vector4f& color,
		bool wireframe)
	{
		unsigned int centerIdx = GetCurrentVertexCount(target);
		Eigen::Vector3f n = normal.normalized();

		Eigen::Vector3f t = (std::abs(n.y()) > 0.9f) ? Eigen::Vector3f(1, 0, 0) : Eigen::Vector3f(0, 1, 0);
		Eigen::Vector3f b = n.cross(t).normalized();
		t = b.cross(n).normalized();

		AddVertexToBuffer(target, center, n, color);

		for (unsigned int i = 0; i <= segments; ++i)
		{
			float theta = (float)i / (float)segments * TWO_PI;
			float x = std::cos(theta) * radius;
			float z = std::sin(theta) * radius;
			Eigen::Vector3f pos = center + (t * x) + (b * z);
			AddVertexToBuffer(target, pos, n, color);
		}

		for (unsigned int i = 1; i <= segments; ++i)
		{
			if (wireframe)
			{
				AddIndexToBuffer(target, centerIdx + i);
				AddIndexToBuffer(target, centerIdx + i + 1);
			}
			else
			{
				AddIndexToBuffer(target, centerIdx);
				AddIndexToBuffer(target, centerIdx + i + 1);
				AddIndexToBuffer(target, centerIdx + i);
			}
		}
	}

	template <typename T>
	void BuildCylinderHelper(
		T* target,
		const Eigen::Vector3f& center,
		float radius,
		float height,
		unsigned int segments,
		const Eigen::Vector4f& color,
		bool wireframe)
	{
		float halfHeight = height * 0.5f;
		unsigned int startIdx = GetCurrentVertexCount(target);

		Eigen::Vector3f topCenter = center + Eigen::Vector3f(0, halfHeight, 0);
		Eigen::Vector3f bottomCenter = center - Eigen::Vector3f(0, halfHeight, 0);

		for (unsigned int i = 0; i <= segments; ++i)
		{
			float theta = (float)i / (float)segments * TWO_PI;
			float x = std::cos(theta);
			float z = std::sin(theta);
			Eigen::Vector3f normal(x, 0, z);

			AddVertexToBuffer(target, topCenter + normal * radius, normal, color);
			AddVertexToBuffer(target, bottomCenter + normal * radius, normal, color);
		}

		for (unsigned int i = 0; i < segments; ++i)
		{
			unsigned int top1 = startIdx + (i * 2);
			unsigned int bottom1 = startIdx + (i * 2) + 1;
			unsigned int top2 = startIdx + (i * 2) + 2;
			unsigned int bottom2 = startIdx + (i * 2) + 3;

			if (wireframe)
			{
				AddIndexToBuffer(target, top1); AddIndexToBuffer(target, top2);
				AddIndexToBuffer(target, bottom1); AddIndexToBuffer(target, bottom2);
				AddIndexToBuffer(target, top1); AddIndexToBuffer(target, bottom1);
			}
			else
			{
				AddIndexToBuffer(target, top1); AddIndexToBuffer(target, bottom1); AddIndexToBuffer(target, top2);
				AddIndexToBuffer(target, bottom1); AddIndexToBuffer(target, bottom2); AddIndexToBuffer(target, top2);
			}
		}

		BuildDiskHelper(target, topCenter, Eigen::Vector3f(0, 1, 0), radius, segments, color, wireframe);
		BuildDiskHelper(target, bottomCenter, Eigen::Vector3f(0, -1, 0), radius, segments, color, wireframe);
	}

	template <typename T>
	void BuildConeHelper(
		T* target,
		const Eigen::Vector3f& center,
		float radius,
		float height,
		unsigned int segments,
		const Eigen::Vector4f& color,
		bool wireframe)
	{
		float halfHeight = height * 0.5f;
		Eigen::Vector3f tip = center + Eigen::Vector3f(0, halfHeight, 0);
		Eigen::Vector3f baseCenter = center - Eigen::Vector3f(0, halfHeight, 0);
		unsigned int startIdx = GetCurrentVertexCount(target);

		AddVertexToBuffer(target, tip, Eigen::Vector3f(0, 1, 0), color);

		for (unsigned int i = 0; i <= segments; ++i)
		{
			float theta = (float)i / (float)segments * TWO_PI;
			float x = std::cos(theta);
			float z = std::sin(theta);

			Eigen::Vector3f dir(x, 0, z);
			Eigen::Vector3f slant = (Eigen::Vector3f(0, 1, 0).cross(dir)).cross(tip - (baseCenter + dir * radius)).normalized();

			AddVertexToBuffer(target, baseCenter + dir * radius, slant, color);
		}

		unsigned int tipIdx = startIdx;
		for (unsigned int i = 1; i <= segments; ++i)
		{
			unsigned int current = startIdx + i;
			unsigned int next = startIdx + i + 1;

			if (wireframe)
			{
				AddIndexToBuffer(target, tipIdx); AddIndexToBuffer(target, current);
				AddIndexToBuffer(target, current); AddIndexToBuffer(target, next);
			}
			else
			{
				AddIndexToBuffer(target, tipIdx);
				AddIndexToBuffer(target, next);
				AddIndexToBuffer(target, current);
			}
		}

		BuildDiskHelper(target, baseCenter, Eigen::Vector3f(0, -1, 0), radius, segments, color, wireframe);
	}

	template <typename T>
	void BuildCapsuleHelper(
		T* target,
		const Eigen::Vector3f& center,
		float radius,
		float height,
		unsigned int segments,
		unsigned int rings,
		const Eigen::Vector4f& color,
		bool wireframe)
	{
		float cylinderHeight = (std::max)(0.0f, height - 2.0f * radius);
		float halfCyl = cylinderHeight * 0.5f;

		unsigned int startIdx = GetCurrentVertexCount(target);

		for (unsigned int r = 0; r <= rings; ++r)
		{
			float v = (float)r / (float)rings;
			float phi = v * PI;

			for (unsigned int s = 0; s <= segments; ++s)
			{
				float u = (float)s / (float)segments;
				float theta = u * TWO_PI;

				float x = std::sin(phi) * std::cos(theta);
				float y = std::cos(phi);
				float z = std::sin(phi) * std::sin(theta);

				Eigen::Vector3f normal(x, y, z);
				Eigen::Vector3f pos = normal * radius;

				if (y > 0) pos.y() += halfCyl;
				else       pos.y() -= halfCyl;

				AddVertexToBuffer(target, center + pos, normal, color);
			}
		}

		for (unsigned int r = 0; r < rings; ++r)
		{
			for (unsigned int s = 0; s < segments; ++s)
			{
				unsigned int i0 = startIdx + (r * (segments + 1)) + s;
				unsigned int i1 = i0 + 1;
				unsigned int i2 = i0 + (segments + 1);
				unsigned int i3 = i2 + 1;

				if (wireframe)
				{
					AddIndexToBuffer(target, i0); AddIndexToBuffer(target, i1);
					AddIndexToBuffer(target, i0); AddIndexToBuffer(target, i2);
				}
				else
				{
					AddIndexToBuffer(target, i0); AddIndexToBuffer(target, i2); AddIndexToBuffer(target, i1);
					AddIndexToBuffer(target, i1); AddIndexToBuffer(target, i2); AddIndexToBuffer(target, i3);
				}
			}
		}
	}

	template <typename T>
	void BuildTorusHelper(
		T* target,
		const Eigen::Vector3f& center,
		float majorRadius,
		float minorRadius,
		unsigned int majorSegments,
		unsigned int minorSegments,
		const Eigen::Vector4f& color,
		bool wireframe)
	{
		unsigned int startIdx = GetCurrentVertexCount(target);

		for (unsigned int i = 0; i <= majorSegments; ++i)
		{
			float u = (float)i / (float)majorSegments * TWO_PI;
			float cu = std::cos(u);
			float su = std::sin(u);

			for (unsigned int j = 0; j <= minorSegments; ++j)
			{
				float v = (float)j / (float)minorSegments * TWO_PI;
				float cv = std::cos(v);
				float sv = std::sin(v);

				float x = (majorRadius + minorRadius * cv) * cu;
				float y = minorRadius * sv;
				float z = (majorRadius + minorRadius * cv) * su;

				Eigen::Vector3f centerTube(majorRadius * cu, 0, majorRadius * su);
				Eigen::Vector3f pos(x, y, z);
				Eigen::Vector3f normal = (pos - centerTube).normalized();

				AddVertexToBuffer(target, center + pos, normal, color);
			}
		}

		for (unsigned int i = 0; i < majorSegments; ++i)
		{
			for (unsigned int j = 0; j < minorSegments; ++j)
			{
				unsigned int i0 = startIdx + (i * (minorSegments + 1)) + j;
				unsigned int i1 = i0 + 1;
				unsigned int i2 = i0 + (minorSegments + 1);
				unsigned int i3 = i2 + 1;

				if (wireframe)
				{
					AddIndexToBuffer(target, i0); AddIndexToBuffer(target, i1);
					AddIndexToBuffer(target, i0); AddIndexToBuffer(target, i2);
				}
				else
				{
					AddIndexToBuffer(target, i0); AddIndexToBuffer(target, i2); AddIndexToBuffer(target, i1);
					AddIndexToBuffer(target, i1); AddIndexToBuffer(target, i2); AddIndexToBuffer(target, i3);
				}
			}
		}
	}

	Eigen::Vector3f CatmullRom(float t, const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Eigen::Vector3f& p3)
	{
		float t2 = t * t;
		float t3 = t2 * t;

		return 0.5f * (
			(2.0f * p1) +
			(-p0 + p2) * t +
			(2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
			(-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3
			);
	}

	template <typename T>
	void BuildTubeHelper(
		T* target,
		const std::vector<Eigen::Vector3f>& controlPoints,
		float radius,
		unsigned int curveSegments,
		unsigned int radialSegments,
		const Eigen::Vector4f& color,
		bool wireframe)
	{
		if (controlPoints.size() < 2) return;

		unsigned int startVertexIdx = GetCurrentVertexCount(target);

		// 1. 곡선 경로 생성 (Path Generation)
		std::vector<Eigen::Vector3f> pathPoints;
		std::vector<Eigen::Vector3f> tangents;

		// Catmull-Rom을 위해 앞뒤로 점을 복제하여 패딩
		std::vector<Eigen::Vector3f> paddedPoints = controlPoints;
		paddedPoints.insert(paddedPoints.begin(), controlPoints.front());
		paddedPoints.push_back(controlPoints.back());

		for (size_t i = 0; i < paddedPoints.size() - 3; ++i)
		{
			for (unsigned int j = 0; j < curveSegments; ++j)
			{
				float t = (float)j / (float)curveSegments;
				pathPoints.push_back(CatmullRom(t, paddedPoints[i], paddedPoints[i + 1], paddedPoints[i + 2], paddedPoints[i + 3]));
			}
		}
		// 마지막 점 추가
		pathPoints.push_back(paddedPoints[paddedPoints.size() - 2]);

		// 2. 접선(Tangent) 계산
		if (pathPoints.size() < 2) return; // 안전장치

		tangents.resize(pathPoints.size());

		// 첫 점 Tangent
		tangents[0] = (pathPoints[1] - pathPoints[0]).normalized();

		// 중간 점들 Tangent
		for (size_t i = 1; i < pathPoints.size() - 1; ++i)
		{
			tangents[i] = (pathPoints[i + 1] - pathPoints[i - 1]).normalized();
		}

		// 마지막 점 Tangent
		tangents.back() = (pathPoints.back() - pathPoints[pathPoints.size() - 2]).normalized();


		// 3. 프레임(Normal/Binormal) 생성 (Parallel Transport / Bishop's Frame)
		// 꼬임을 방지하기 위해 첫 프레임을 기준으로 회전시킵니다.
		std::vector<Eigen::Vector3f> normals(pathPoints.size());
		std::vector<Eigen::Vector3f> binormals(pathPoints.size());

		// 초기 프레임 설정
		Eigen::Vector3f initialTangent = tangents[0];
		Eigen::Vector3f initialNormal;

		// Tangent와 평행하지 않은 임의의 벡터를 찾아 Normal 생성
		if (std::abs(initialTangent.x()) < 0.9f) initialNormal = Eigen::Vector3f(1, 0, 0);
		else initialNormal = Eigen::Vector3f(0, 1, 0);

		normals[0] = initialTangent.cross(initialNormal).normalized();
		binormals[0] = initialTangent.cross(normals[0]).normalized();

		for (size_t i = 1; i < pathPoints.size(); ++i)
		{
			Eigen::Vector3f prevTangent = tangents[i - 1];
			Eigen::Vector3f curTangent = tangents[i];

			// 이전 Tangent를 현재 Tangent로 회전시키는 쿼터니언 계산
			Eigen::Quaternionf rotation = Eigen::Quaternionf::FromTwoVectors(prevTangent, curTangent);

			// 이전 Normal/Binormal을 회전시켜 현재 프레임 계산 (Parallel Transport)
			normals[i] = (rotation * normals[i - 1]).normalized();
			binormals[i] = curTangent.cross(normals[i]).normalized();
		}

		// 4. 버텍스 생성 (Extrusion)
		for (size_t i = 0; i < pathPoints.size(); ++i)
		{
			Eigen::Vector3f center = pathPoints[i];
			Eigen::Vector3f normal = normals[i];
			Eigen::Vector3f binormal = binormals[i];

			for (unsigned int j = 0; j <= radialSegments; ++j)
			{
				float theta = (float)j / (float)radialSegments * TWO_PI;

				float sinTheta = std::sin(theta);
				float cosTheta = std::cos(theta);

				// 원형 단면의 로컬 오프셋
				Eigen::Vector3f offset = (normal * cosTheta + binormal * sinTheta) * radius;
				Eigen::Vector3f position = center + offset;

				// 법선은 튜브 표면 바깥쪽을 향함
				Eigen::Vector3f surfaceNormal = offset.normalized();

				AddVertexToBuffer(target, position, surfaceNormal, color);
			}
		}

		// 5. 인덱스 생성
		unsigned int numRings = (unsigned int)pathPoints.size();
		unsigned int vertsPerRing = radialSegments + 1;

		for (unsigned int i = 0; i < numRings - 1; ++i)
		{
			for (unsigned int j = 0; j < radialSegments; ++j)
			{
				unsigned int current = startVertexIdx + (i * vertsPerRing) + j;
				unsigned int next = current + vertsPerRing;

				unsigned int i0 = current;
				unsigned int i1 = current + 1;
				unsigned int i2 = next;
				unsigned int i3 = next + 1;

				if (wireframe)
				{
					// Ring
					AddIndexToBuffer(target, i0); AddIndexToBuffer(target, i1);
					// Connection to next ring
					AddIndexToBuffer(target, i0); AddIndexToBuffer(target, i2);
				}
				else
				{
					AddIndexToBuffer(target, i0); AddIndexToBuffer(target, i2); AddIndexToBuffer(target, i1);
					AddIndexToBuffer(target, i1); AddIndexToBuffer(target, i2); AddIndexToBuffer(target, i3);
				}
			}
		}
	}

	template <typename T>
	void BuildArrowHelper(
		T* target,
		const Eigen::Vector3f& start,
		const Eigen::Vector3f& end,
		float stemRadius,
		float headRadius,
		float headLength,
		const Eigen::Vector4f& color,
		bool wireframe)
	{
		Eigen::Vector3f dir = (end - start);
		float length = dir.norm();
		if (length < 0.0001f) return;

		dir.normalize();

		Eigen::Quaternionf rotation = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitY(), dir);
		Eigen::Matrix3f rotMat = rotation.toRotationMatrix();

		// FIX: (std::max)를 사용하여 매크로 확장을 방지합니다.
		float stemLen = (std::max)(0.0f, length - headLength);
		Eigen::Vector3f stemCenter = start + (dir * (stemLen * 0.5f));

		unsigned int segments = 16;

		unsigned int stemStartIdx = GetCurrentVertexCount(target);
		for (unsigned int i = 0; i <= segments; ++i)
		{
			float theta = (float)i / segments * TWO_PI;
			float x = std::cos(theta) * stemRadius;
			float z = std::sin(theta) * stemRadius;

			Eigen::Vector3f normal = rotMat * Eigen::Vector3f(std::cos(theta), 0, std::sin(theta));

			AddVertexToBuffer(target, start + rotMat * Eigen::Vector3f(x, 0, z), normal, color);
			AddVertexToBuffer(target, start + rotMat * Eigen::Vector3f(x, stemLen, z), normal, color);
		}

		for (unsigned int i = 0; i < segments; ++i)
		{
			unsigned int bl = stemStartIdx + i * 2;
			unsigned int tl = bl + 1;
			unsigned int br = stemStartIdx + i * 2 + 2;
			unsigned int tr = br + 1;

			if (wireframe)
			{
				AddIndexToBuffer(target, bl); AddIndexToBuffer(target, tl);
				AddIndexToBuffer(target, bl); AddIndexToBuffer(target, br);
			}
			else
			{
				AddIndexToBuffer(target, bl); AddIndexToBuffer(target, br); AddIndexToBuffer(target, tl);
				AddIndexToBuffer(target, tl); AddIndexToBuffer(target, br); AddIndexToBuffer(target, tr);
			}
		}

		unsigned int coneStartIdx = GetCurrentVertexCount(target);
		Eigen::Vector3f tip = end;
		Eigen::Vector3f base = start + dir * stemLen;

		AddVertexToBuffer(target, tip, dir, color);

		for (unsigned int i = 0; i <= segments; ++i)
		{
			float theta = (float)i / segments * TWO_PI;
			float x = std::cos(theta) * headRadius;
			float z = std::sin(theta) * headRadius;

			Eigen::Vector3f localNormal(std::cos(theta), 0, std::sin(theta));
			Eigen::Vector3f n = rotMat * localNormal;

			AddVertexToBuffer(target, base + rotMat * Eigen::Vector3f(x, 0, z), n, color);
		}

		unsigned int tipIdx = coneStartIdx;
		for (unsigned int i = 1; i <= segments; ++i)
		{
			unsigned int cur = coneStartIdx + i;
			unsigned int next = coneStartIdx + i + 1;

			if (wireframe) {
				AddIndexToBuffer(target, tipIdx); AddIndexToBuffer(target, cur);
				AddIndexToBuffer(target, cur); AddIndexToBuffer(target, next);
			}
			else {
				AddIndexToBuffer(target, tipIdx); AddIndexToBuffer(target, next); AddIndexToBuffer(target, cur);
			}
		}
	}

	template <typename T>
	void BuildFrustumHelper(
		T* target,
		const Eigen::Matrix4f& invViewProj,
		const Eigen::Vector4f& color,
		bool wireframe)
	{
		std::vector<Eigen::Vector4f> corners = {
			{-1, -1, -1, 1}, {1, -1, -1, 1}, {1, 1, -1, 1}, {-1, 1, -1, 1},
			{-1, -1,  1, 1}, {1, -1,  1, 1}, {1, 1,  1, 1}, {-1, 1,  1, 1}
		};

		unsigned int startIdx = GetCurrentVertexCount(target);

		for (auto& v : corners)
		{
			Eigen::Vector4f worldPos = invViewProj * v;
			worldPos /= worldPos.w();
			AddVertexToBuffer(target, worldPos.head<3>(), Eigen::Vector3f::Zero(), color);
		}

		unsigned int indices[] = {
			0, 1, 1, 2, 2, 3, 3, 0,
			4, 5, 5, 6, 6, 7, 7, 4,
			0, 4, 1, 5, 2, 6, 3, 7
		};

		for (unsigned int idx : indices)
		{
			AddIndexToBuffer(target, startIdx + idx);
		}
	}

	template <typename T>
	void BuildGridHelper(
		T* target,
		float size,
		unsigned int divisions,
		const Eigen::Vector4f& color)
	{
		unsigned int startIdx = GetCurrentVertexCount(target);
		float step = size / (float)divisions;
		float halfSize = size * 0.5f;

		for (unsigned int i = 0; i <= divisions; ++i)
		{
			float x = -halfSize + (i * step);
			AddVertexToBuffer(target, Eigen::Vector3f(x, 0, -halfSize), Eigen::Vector3f(0, 1, 0), color);
			AddVertexToBuffer(target, Eigen::Vector3f(x, 0, halfSize), Eigen::Vector3f(0, 1, 0), color);

			unsigned int idx = startIdx + (i * 2);
			AddIndexToBuffer(target, idx);
			AddIndexToBuffer(target, idx + 1);
		}

		startIdx = GetCurrentVertexCount(target);
		for (unsigned int i = 0; i <= divisions; ++i)
		{
			float z = -halfSize + (i * step);
			AddVertexToBuffer(target, Eigen::Vector3f(-halfSize, 0, z), Eigen::Vector3f(0, 1, 0), color);
			AddVertexToBuffer(target, Eigen::Vector3f(halfSize, 0, z), Eigen::Vector3f(0, 1, 0), color);

			unsigned int idx = startIdx + (i * 2);
			AddIndexToBuffer(target, idx);
			AddIndexToBuffer(target, idx + 1);
		}
	}
}

void GeometryBuilder::BuildPlane(
	Renderable* renderable,
	float width,
	float height,
	unsigned int hSegments,
	unsigned int vSegments,
	const Eigen::Vector3f& center,
	const Eigen::Vector3f& normal,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildPlaneHelper(renderable, width, height, hSegments, vSegments, center, normal, color, wireframe);
}

void GeometryBuilder::BuildPlane(
	DebuggingRenderable* debuggingRenderable,
	float width,
	float height,
	unsigned int hSegments,
	unsigned int vSegments,
	const Eigen::Vector3f& center,
	const Eigen::Vector3f& normal,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildPlaneHelper(debuggingRenderable, width, height, hSegments, vSegments, center, normal, color, wireframe);
}

void GeometryBuilder::BuildBox(
	Renderable* renderable,
	const Eigen::Vector3f& center,
	const Eigen::Vector3f& dimension,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildBoxHelper(renderable, center, dimension, color, wireframe);
}

void GeometryBuilder::BuildBox(
	DebuggingRenderable* debuggingRenderable,
	const Eigen::Vector3f& center,
	const Eigen::Vector3f& dimension,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildBoxHelper(debuggingRenderable, center, dimension, color, wireframe);
}

void GeometryBuilder::BuildSphere(
	Renderable* renderable,
	const Eigen::Vector3f& center,
	float radius,
	unsigned int latitudeSegments,
	unsigned int longitudeSegments,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildSphereHelper(renderable, center, radius, latitudeSegments, longitudeSegments, color, wireframe);
}

void GeometryBuilder::BuildSphere(
	DebuggingRenderable* debuggingRenderable,
	const Eigen::Vector3f& center,
	float radius,
	unsigned int latitudeSegments,
	unsigned int longitudeSegments,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildSphereHelper(debuggingRenderable, center, radius, latitudeSegments, longitudeSegments, color, wireframe);
}

void GeometryBuilder::BuildDisk(
	Renderable* renderable,
	const Eigen::Vector3f& center,
	const Eigen::Vector3f& normal,
	float radius,
	unsigned int segments,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildDiskHelper(renderable, center, normal, radius, segments, color, wireframe);
}

void GeometryBuilder::BuildDisk(
	DebuggingRenderable* renderable,
	const Eigen::Vector3f& center,
	const Eigen::Vector3f& normal,
	float radius,
	unsigned int segments,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildDiskHelper(renderable, center, normal, radius, segments, color, wireframe);
}

void GeometryBuilder::BuildCylinder(
	Renderable* renderable,
	const Eigen::Vector3f& center,
	float radius,
	float height,
	unsigned int segments,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildCylinderHelper(renderable, center, radius, height, segments, color, wireframe);
}

void GeometryBuilder::BuildCylinder(
	DebuggingRenderable* renderable,
	const Eigen::Vector3f& center,
	float radius,
	float height,
	unsigned int segments,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildCylinderHelper(renderable, center, radius, height, segments, color, wireframe);
}

void GeometryBuilder::BuildCone(
	Renderable* renderable,
	const Eigen::Vector3f& center,
	float radius,
	float height,
	unsigned int segments,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildConeHelper(renderable, center, radius, height, segments, color, wireframe);
}

void GeometryBuilder::BuildCone(
	DebuggingRenderable* renderable,
	const Eigen::Vector3f& center,
	float radius,
	float height,
	unsigned int segments,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildConeHelper(renderable, center, radius, height, segments, color, wireframe);
}

void GeometryBuilder::BuildCapsule(
	Renderable* renderable,
	const Eigen::Vector3f& center,
	float radius,
	float height,
	unsigned int segments,
	unsigned int rings,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildCapsuleHelper(renderable, center, radius, height, segments, rings, color, wireframe);
}

void GeometryBuilder::BuildCapsule(
	DebuggingRenderable* renderable,
	const Eigen::Vector3f& center,
	float radius,
	float height,
	unsigned int segments,
	unsigned int rings,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildCapsuleHelper(renderable, center, radius, height, segments, rings, color, wireframe);
}

void GeometryBuilder::BuildTorus(
	Renderable* renderable,
	const Eigen::Vector3f& center,
	float majorRadius,
	float minorRadius,
	unsigned int majorSegments,
	unsigned int minorSegments,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildTorusHelper(renderable, center, majorRadius, minorRadius, majorSegments, minorSegments, color, wireframe);
}

void GeometryBuilder::BuildTorus(
	DebuggingRenderable* renderable,
	const Eigen::Vector3f& center,
	float majorRadius,
	float minorRadius,
	unsigned int majorSegments,
	unsigned int minorSegments,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildTorusHelper(renderable, center, majorRadius, minorRadius, majorSegments, minorSegments, color, wireframe);
}

void GeometryBuilder::BuildTube(
	Renderable* renderable,
	const std::vector<Eigen::Vector3f>& controlPoints,
	float radius,
	unsigned int curveSegments,
	unsigned int radialSegments,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildTubeHelper(renderable, controlPoints, radius, curveSegments, radialSegments, color, wireframe);
}

void GeometryBuilder::BuildTube(
	DebuggingRenderable* renderable,
	const std::vector<Eigen::Vector3f>& controlPoints,
	float radius,
	unsigned int curveSegments,
	unsigned int radialSegments,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildTubeHelper(renderable, controlPoints, radius, curveSegments, radialSegments, color, wireframe);
}

void GeometryBuilder::BuildArrow(
	Renderable* renderable,
	const Eigen::Vector3f& start,
	const Eigen::Vector3f& end,
	float stemRadius,
	float headRadius,
	float headLength,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildArrowHelper(renderable, start, end, stemRadius, headRadius, headLength, color, wireframe);
}

void GeometryBuilder::BuildArrow(
	DebuggingRenderable* renderable,
	const Eigen::Vector3f& start,
	const Eigen::Vector3f& end,
	float stemRadius,
	float headRadius,
	float headLength,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildArrowHelper(renderable, start, end, stemRadius, headRadius, headLength, color, wireframe);
}

void GeometryBuilder::BuildFrustum(
	Renderable* renderable,
	const Eigen::Matrix4f& invViewProj,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildFrustumHelper(renderable, invViewProj, color, wireframe);
}

void GeometryBuilder::BuildFrustum(
	DebuggingRenderable* renderable,
	const Eigen::Matrix4f& invViewProj,
	const Eigen::Vector4f& color,
	bool wireframe)
{
	BuildFrustumHelper(renderable, invViewProj, color, wireframe);
}

void GeometryBuilder::BuildGrid(
	Renderable* renderable,
	float size,
	unsigned int divisions,
	const Eigen::Vector4f& color)
{
	BuildGridHelper(renderable, size, divisions, color);
}

void GeometryBuilder::BuildGrid(
	DebuggingRenderable* renderable,
	float size,
	unsigned int divisions,
	const Eigen::Vector4f& color)
{
	BuildGridHelper(renderable, size, divisions, color);
}