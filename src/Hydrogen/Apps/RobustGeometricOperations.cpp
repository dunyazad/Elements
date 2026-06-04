#include "RobustGeometricOperations.h"

#include <execution>
#include <numeric>
#include <iostream>

#include <charconv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include <stb/stb_truetype.h>

namespace Eigen
{
	Matrix4f MakeTransform(
		const Vector3f& translation,
		const Vector3f& rotation_axis,
		float rotation_angle_degree,
		const Vector3f& scale)
	{
		Affine3f t = Affine3f::Identity();

		// Applied right to left: scale first, then rotate, then translate
		t.translate(translation);
		t.rotate(AngleAxisf(rotation_angle_degree * static_cast<float>(M_PI) / 180.0f, rotation_axis.normalized()));
		t.scale(scale);

		return t.matrix();
	}

	Matrix4f MakeTransformEuler(
		const Vector3f& translation,
		const Vector3f& euler_xyz_degree,
		const Vector3f& scale)
	{
		Affine3f t = Affine3f::Identity();

		t.translate(translation);
		t.rotate(AngleAxisf(euler_xyz_degree.z() * static_cast<float>(M_PI) / 180.0f, Vector3f::UnitZ()));
		t.rotate(AngleAxisf(euler_xyz_degree.y() * static_cast<float>(M_PI) / 180.0f, Vector3f::UnitY()));
		t.rotate(AngleAxisf(euler_xyz_degree.x() * static_cast<float>(M_PI) / 180.0f, Vector3f::UnitX()));
		t.scale(scale);

		return t.matrix();
	}

	Matrix4f MakeTransformManual(
		const Vector3f& translation,
		const Quaternionf& rotation,
		const Vector3f& scale)
	{
		Matrix4f m = Matrix4f::Identity();

		// Upper-left 3x3: rotation times per-axis scale (R * S)
		Matrix3f rs = rotation.toRotationMatrix();
		rs.col(0) *= scale.x();
		rs.col(1) *= scale.y();
		rs.col(2) *= scale.z();

		m.block<3, 3>(0, 0) = rs;
		m.block<3, 1>(0, 3) = translation;

		return m;
	}
}

namespace RGO
{
	// ------------------------------------------------------------
	// Distance
	// ------------------------------------------------------------

	float Distance::PointToRaySquared(const Eigen::Vector3f& p, const Eigen::Vector3f& origin, const Eigen::Vector3f& dir)
	{
		Eigen::Vector3f op = p - origin;
		float t = op.dot(dir);
		if (t < 0.0f)
		{
			return op.squaredNorm();
		}
		Eigen::Vector3f projection = origin + t * dir;
		return (p - projection).squaredNorm();
	}

	float Distance::PointToRay(const Eigen::Vector3f& p, const Eigen::Vector3f& origin, const Eigen::Vector3f& dir)
	{
		return std::sqrt(PointToRaySquared(p, origin, dir));
	}

	float Distance::PointToLineSegmentSquared(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b)
	{
		Eigen::Vector3f ab = b - a;
		float l2 = ab.squaredNorm();
		if (l2 < EPSILON * EPSILON)
		{
			return (p - a).squaredNorm();
		}
		float t = (p - a).dot(ab) / l2;
		t = std::max(0.0f, std::min(1.0f, t));
		Eigen::Vector3f projection = a + t * ab;
		return (p - projection).squaredNorm();
	}

	float Distance::PointToLineSegment(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b)
	{
		return std::sqrt(PointToLineSegmentSquared(p, a, b));
	}

	// ------------------------------------------------------------
	// Intersection
	// ------------------------------------------------------------

	bool Intersection::PointToLineSegment(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b, float epsilon)
	{
		return Distance::PointToLineSegmentSquared(p, a, b) < epsilon * epsilon;
	}

	bool Intersection::PointToRay(const Eigen::Vector3f& p, const Eigen::Vector3f& origin, const Eigen::Vector3f& dir, float epsilon)
	{
		return Distance::PointToRaySquared(p, origin, dir) < epsilon * epsilon;
	}

	Intersection::PointToTriangleResult Intersection::PointToTriangle(
		const Eigen::Vector3f& p,
		const Eigen::Vector3f& v0,
		const Eigen::Vector3f& v1,
		const Eigen::Vector3f& v2,
		float epsilon)
	{
		PointToTriangleResult result;

		// Vertex check first: vertices are the most constrained classification
		if ((p - v0).squaredNorm() < epsilon * epsilon)
		{
			result.type = PointToTriangleType::OnVertex;
			result.vertex_index = 0;
			return result;
		}
		if ((p - v1).squaredNorm() < epsilon * epsilon)
		{
			result.type = PointToTriangleType::OnVertex;
			result.vertex_index = 1;
			return result;
		}
		if ((p - v2).squaredNorm() < epsilon * epsilon)
		{
			result.type = PointToTriangleType::OnVertex;
			result.vertex_index = 2;
			return result;
		}

		// Edge check next: edges are more constrained than the interior
		if (Distance::PointToLineSegmentSquared(p, v0, v1) < epsilon * epsilon)
		{
			result.type = PointToTriangleType::OnEdge;
			result.edge_index = 0;
			return result;
		}
		if (Distance::PointToLineSegmentSquared(p, v1, v2) < epsilon * epsilon)
		{
			result.type = PointToTriangleType::OnEdge;
			result.edge_index = 1;
			return result;
		}
		if (Distance::PointToLineSegmentSquared(p, v2, v0) < epsilon * epsilon)
		{
			result.type = PointToTriangleType::OnEdge;
			result.edge_index = 2;
			return result;
		}

		// Interior check: must be near the plane AND inside the projected triangle
		Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
		float n_len = n.norm();
		if (n_len < 1e-12f)
		{
			// Degenerate triangle: nothing can be classified as inside
			return result;
		}
		n /= n_len;

		float dist = (p - v0).dot(n);
		result.plane_distance = std::abs(dist);
		if (result.plane_distance > epsilon)
		{
			return result;
		}

		Eigen::Vector3f proj = p - dist * n;
		Eigen::Vector3f c0 = (v1 - v0).cross(proj - v0);
		Eigen::Vector3f c1 = (v2 - v1).cross(proj - v1);
		Eigen::Vector3f c2 = (v0 - v2).cross(proj - v2);

		if (c0.dot(n) >= 0.0f && c1.dot(n) >= 0.0f && c2.dot(n) >= 0.0f)
		{
			result.type = PointToTriangleType::Inside;
		}

		return result;
	}

	bool Intersection::RayToTriangle(
		const Eigen::Vector3f& origin,
		const Eigen::Vector3f& dir,
		const Eigen::Vector3f& v0,
		const Eigen::Vector3f& v1,
		const Eigen::Vector3f& v2,
		Eigen::Vector3f& intersection_point,
		float epsilon)
	{
		Eigen::Vector3f edge1 = v1 - v0;
		Eigen::Vector3f edge2 = v2 - v0;
		Eigen::Vector3f h = dir.cross(edge2);
		float a = edge1.dot(h);
		if (std::abs(a) < epsilon)
			return false;
		float f = 1.0f / a;
		Eigen::Vector3f s = origin - v0;
		float u = f * s.dot(h);
		if (u < 0.0f || u > 1.0f)
			return false;
		Eigen::Vector3f q = s.cross(edge1);
		float v = f * dir.dot(q);
		if (v < 0.0f || u + v > 1.0f)
			return false;
		float t = f * edge2.dot(q);
		if (t < epsilon)
			return false;
		intersection_point = origin + t * dir;
		return true;
	}

	bool Intersection::SegmentToTriangle(
		const Eigen::Vector3f& s0,
		const Eigen::Vector3f& s1,
		const Eigen::Vector3f& v0,
		const Eigen::Vector3f& v1,
		const Eigen::Vector3f& v2,
		Eigen::Vector3f& out_point,
		float epsilon)
	{
		Eigen::Vector3f dir = s1 - s0;
		float seg_len = dir.norm();
		if (seg_len < epsilon)
			return false;
		dir /= seg_len;

		Eigen::Vector3f edge1 = v1 - v0;
		Eigen::Vector3f edge2 = v2 - v0;
		Eigen::Vector3f h = dir.cross(edge2);
		float a = edge1.dot(h);

		// Parallel to triangle plane. Coplanar overlap is handled by caller policy.
		if (std::abs(a) < 1e-8f)
			return false;

		float f = 1.0f / a;
		Eigen::Vector3f s = s0 - v0;
		float u = f * s.dot(h);
		if (u < -epsilon || u > 1.0f + epsilon)
			return false;

		Eigen::Vector3f q = s.cross(edge1);
		float v = f * dir.dot(q);
		if (v < -epsilon || u + v > 1.0f + epsilon)
			return false;

		// t must lie within the segment, not on an infinite ray
		float t = f * edge2.dot(q);
		if (t < -epsilon || t > seg_len + epsilon)
			return false;

		out_point = s0 + dir * t;
		return true;
	}

	Intersection::TriangleIntersectionResult Intersection::TriangleToTriangle(
		const Eigen::Vector3f& a0, const Eigen::Vector3f& a1, const Eigen::Vector3f& a2,
		const Eigen::Vector3f& b0, const Eigen::Vector3f& b1, const Eigen::Vector3f& b2,
		Eigen::Vector3f& intersectionA,
		Eigen::Vector3f& intersectionB,
		float epsilon)
	{
		TriangleIntersectionResult result{ TriangleTriangleIntersectionType::None, Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero() };
		intersectionA = Eigen::Vector3f::Zero();
		intersectionB = Eigen::Vector3f::Zero();

		Eigen::Vector3f nA = (a1 - a0).cross(a2 - a0);
		Eigen::Vector3f nB = (b1 - b0).cross(b2 - b0);
		if (nA.squaredNorm() < epsilon * epsilon || nB.squaredNorm() < epsilon * epsilon)
			return result;
		nA.normalize();
		nB.normalize();

		// Early out: all vertices of B strictly on one side of plane A
		float dB0 = (b0 - a0).dot(nA);
		float dB1 = (b1 - a0).dot(nA);
		float dB2 = (b2 - a0).dot(nA);
		if ((dB0 > epsilon && dB1 > epsilon && dB2 > epsilon) ||
			(dB0 < -epsilon && dB1 < -epsilon && dB2 < -epsilon))
			return result;

		// Early out: all vertices of A strictly on one side of plane B
		float dA0 = (a0 - b0).dot(nB);
		float dA1 = (a1 - b0).dot(nB);
		float dA2 = (a2 - b0).dot(nB);
		if ((dA0 > epsilon && dA1 > epsilon && dA2 > epsilon) ||
			(dA0 < -epsilon && dA1 < -epsilon && dA2 < -epsilon))
			return result;

		std::vector<Eigen::Vector3f> pts;
		pts.reserve(6);

		// Merge nearly coincident points to avoid degenerate CDT input downstream
		auto add_point = [&](const Eigen::Vector3f& p)
			{
				for (const auto& q : pts)
				{
					if ((p - q).squaredNorm() < epsilon * epsilon)
						return;
				}
				pts.push_back(p);
			};

		// Test edges of both triangles against the other triangle
		Eigen::Vector3f ip;
		if (SegmentToTriangle(a0, a1, b0, b1, b2, ip, epsilon)) add_point(ip);
		if (SegmentToTriangle(a1, a2, b0, b1, b2, ip, epsilon)) add_point(ip);
		if (SegmentToTriangle(a2, a0, b0, b1, b2, ip, epsilon)) add_point(ip);
		if (SegmentToTriangle(b0, b1, a0, a1, a2, ip, epsilon)) add_point(ip);
		if (SegmentToTriangle(b1, b2, a0, a1, a2, ip, epsilon)) add_point(ip);
		if (SegmentToTriangle(b2, b0, a0, a1, a2, ip, epsilon)) add_point(ip);

		if (pts.empty())
			return result;

		if (pts.size() == 1)
		{
			result.type = TriangleTriangleIntersectionType::Point;
			result.pointA = pts[0];
			intersectionA = pts[0];
			return result;
		}

		// Pick the farthest pair so duplicates from shared vertices cannot
		// produce a near-zero-length segment
		size_t best_i = 0;
		size_t best_j = 1;
		float best_d = -1.0f;
		for (size_t i = 0; i < pts.size(); ++i)
		{
			for (size_t j = i + 1; j < pts.size(); ++j)
			{
				float d = (pts[i] - pts[j]).squaredNorm();
				if (d > best_d)
				{
					best_d = d;
					best_i = i;
					best_j = j;
				}
			}
		}

		result.type = TriangleTriangleIntersectionType::Segment;
		result.pointA = pts[best_i];
		result.pointB = pts[best_j];
		intersectionA = result.pointA;
		intersectionB = result.pointB;
		return result;
	}

	bool Intersection::AABBtoAABB(
		const Eigen::Vector3f& a_min, const Eigen::Vector3f& a_max,
		const Eigen::Vector3f& b_min, const Eigen::Vector3f& b_max)
	{
		if (a_max.x() < b_min.x() || a_min.x() > b_max.x()) return false;
		if (a_max.y() < b_min.y() || a_min.y() > b_max.y()) return false;
		if (a_max.z() < b_min.z() || a_min.z() > b_max.z()) return false;
		return true;
	}
	// ------------------------------------------------------------
	// Mesh
	// ------------------------------------------------------------

	Mesh::Mesh()
	{
		request_vertex_status();
		request_edge_status();
		request_halfedge_status();
		request_face_status();
	}

	void Mesh::Build(const std::vector<Eigen::Vector3f>& points, const std::vector<Eigen::Vector3i>& indices)
	{
		std::vector<VertexHandle> v_handles;
		v_handles.reserve(points.size());
		for (const auto& p : points)
		{
			v_handles.push_back(add_vertex(Point(p[0], p[1], p[2])));
		}

		size_t failed_count = 0;
		for (const auto& idx : indices)
		{
			std::vector<VertexHandle> face_v_handles;
			face_v_handles.push_back(v_handles[idx[0]]);
			face_v_handles.push_back(v_handles[idx[1]]);
			face_v_handles.push_back(v_handles[idx[2]]);

			auto fh = add_face(face_v_handles);
			if (false == fh.is_valid())
			{
				++failed_count;
			}
		}

		if (failed_count > 0)
		{
			std::cout << "[Warning] Build: " << failed_count
				<< " faces rejected by add_face (non-manifold or degenerate input)." << std::endl;
		}

		BuildSpatialHashMap();
	}

	void Mesh::BuildSpatialHashMap()
	{
		hash_map.clear();

		size_t num_faces = n_faces();
		if (num_faces == 0) return;

		grid_min.setConstant(std::numeric_limits<float>::max());
		grid_max.setConstant(-std::numeric_limits<float>::max());

		struct FaceAABB
		{
			Eigen::Vector3f min;
			Eigen::Vector3f max;
		};
		std::vector<FaceAABB> face_bounds(num_faces);
		std::vector<char> face_valid(num_faces, 0);
		std::vector<int> face_indices(num_faces);
		std::iota(face_indices.begin(), face_indices.end(), 0);

		std::for_each(std::execution::par_unseq, face_indices.begin(), face_indices.end(), [&](int i)
			{
				FaceHandle fh = face_handle(i);

				// Skip faces marked deleted but not yet garbage collected
				if (status(fh).deleted())
				{
					return;
				}

				Eigen::Vector3f v0, v1, v2;
				GetFaceVertices(fh, v0, v1, v2);

				face_bounds[i].min = v0.cwiseMin(v1).cwiseMin(v2);
				face_bounds[i].max = v0.cwiseMax(v1).cwiseMax(v2);
				face_valid[i] = 1;
			});

		size_t valid_count = 0;
		for (size_t i = 0; i < num_faces; ++i)
		{
			if (0 == face_valid[i]) continue;
			grid_min = grid_min.cwiseMin(face_bounds[i].min);
			grid_max = grid_max.cwiseMax(face_bounds[i].max);
			++valid_count;
		}
		if (0 == valid_count) return;

		Eigen::Vector3f extent = grid_max - grid_min;
		grid_min -= extent * 0.01f;
		grid_max += extent * 0.01f;
		extent = grid_max - grid_min;

		float volume = extent.x() * extent.y() * extent.z();
		float ideal_cell_vol = volume / (valid_count * 0.5f);
		float c_size = std::max(0.1f, std::cbrt(ideal_cell_vol));
		grid_cell_size = Eigen::Vector3f(c_size, c_size, c_size);

		hash_map.reserve(valid_count * 2);

		for (size_t i = 0; i < num_faces; ++i)
		{
			if (0 == face_valid[i]) continue;

			Eigen::Vector3f local_min = face_bounds[i].min - grid_min;
			Eigen::Vector3f local_max = face_bounds[i].max - grid_min;

			int min_x = static_cast<int>(std::floor(local_min.x() / grid_cell_size.x()));
			int min_y = static_cast<int>(std::floor(local_min.y() / grid_cell_size.y()));
			int min_z = static_cast<int>(std::floor(local_min.z() / grid_cell_size.z()));

			int max_x = static_cast<int>(std::floor(local_max.x() / grid_cell_size.x()));
			int max_y = static_cast<int>(std::floor(local_max.y() / grid_cell_size.y()));
			int max_z = static_cast<int>(std::floor(local_max.z() / grid_cell_size.z()));

			for (int cz = min_z; cz <= max_z; ++cz)
			{
				for (int cy = min_y; cy <= max_y; ++cy)
				{
					for (int cx = min_x; cx <= max_x; ++cx)
					{
						hash_map[Eigen::Vector3i(cx, cy, cz)].push_back(static_cast<int>(i));
					}
				}
			}
		}
	}

	void Mesh::GetFaceVertices(OpenMesh::FaceHandle f_handle, Eigen::Vector3f& v0, Eigen::Vector3f& v1, Eigen::Vector3f& v2) const
	{
		auto fv_it = cfv_iter(f_handle);
		auto p0 = point(*fv_it++);
		v0 = Eigen::Vector3f(p0[0], p0[1], p0[2]);
		auto p1 = point(*fv_it++);
		v1 = Eigen::Vector3f(p1[0], p1[1], p1[2]);
		auto p2 = point(*fv_it++);
		v2 = Eigen::Vector3f(p2[0], p2[1], p2[2]);
	}

	void Mesh::QueryOverlappingFaces(const Eigen::Vector3f& aabb_min, const Eigen::Vector3f& aabb_max, std::vector<int>& out_faces) const
	{
		out_faces.clear();
		if (hash_map.empty()) return;

		Eigen::Vector3f local_min = aabb_min - grid_min;
		Eigen::Vector3f local_max = aabb_max - grid_min;

		int min_x = static_cast<int>(std::floor(local_min.x() / grid_cell_size.x()));
		int min_y = static_cast<int>(std::floor(local_min.y() / grid_cell_size.y()));
		int min_z = static_cast<int>(std::floor(local_min.z() / grid_cell_size.z()));

		int max_x = static_cast<int>(std::floor(local_max.x() / grid_cell_size.x()));
		int max_y = static_cast<int>(std::floor(local_max.y() / grid_cell_size.y()));
		int max_z = static_cast<int>(std::floor(local_max.z() / grid_cell_size.z()));

		for (int cz = min_z; cz <= max_z; ++cz)
		{
			for (int cy = min_y; cy <= max_y; ++cy)
			{
				for (int cx = min_x; cx <= max_x; ++cx)
				{
					auto it = hash_map.find(Eigen::Vector3i(cx, cy, cz));
					if (it == hash_map.end()) continue;

					out_faces.insert(out_faces.end(), it->second.begin(), it->second.end());
				}
			}
		}

		// A face spanning multiple cells appears multiple times: deduplicate
		std::sort(out_faces.begin(), out_faces.end());
		out_faces.erase(std::unique(out_faces.begin(), out_faces.end()), out_faces.end());
	}

	// File-local: applies a 4x4 transform to all points in place.
	// Returns true when the transform mirrors (negative determinant of the
	// upper-left 3x3 block), in which case the caller must flip triangle
	// winding to keep outward normals on watertight meshes.
	static bool ApplyTransform(const Eigen::Matrix4f& transform, std::vector<Eigen::Vector3f>& points)
	{
		bool is_identity = transform.isIdentity(1e-7f);
		if (false == is_identity)
		{
			for (auto& p : points)
			{
				Eigen::Vector4f q = transform * Eigen::Vector4f(p.x(), p.y(), p.z(), 1.0f);
				p = q.head<3>();
			}
		}

		float det = transform.block<3, 3>(0, 0).determinant();
		return det < 0.0f;
	}

	// File-local: swaps two indices of every triangle, flipping winding
	static void FlipWinding(std::vector<Eigen::Vector3i>& indices)
	{
		for (auto& t : indices)
		{
			std::swap(t.y(), t.z());
		}
	}

	bool Mesh::BuildBox(const Eigen::Vector3f& center, const Eigen::Vector3f& size, const Eigen::Matrix4f& transform)
	{
		if (size.x() < EPSILON || size.y() < EPSILON || size.z() < EPSILON)
		{
			std::cout << "[Error] BuildBox: all extents must be positive, got ("
				<< size.x() << ", " << size.y() << ", " << size.z() << ")" << std::endl;
			return false;
		}

		Eigen::Vector3f h = size * 0.5f;

		// Vertex layout: bit 0 = +x, bit 1 = +y, bit 2 = +z
		//
		//        6-------7
		//       /|      /|        y
		//      4-------5 |        |
		//      | 2-----|-3        +--x
		//      |/      |/        /
		//      0-------1        z (toward viewer is -z here)
		//
		std::vector<Eigen::Vector3f> points;
		points.reserve(8);
		for (int i = 0; i < 8; ++i)
		{
			points.emplace_back(
				center.x() + ((i & 1) ? h.x() : -h.x()),
				center.y() + ((i & 2) ? h.y() : -h.y()),
				center.z() + ((i & 4) ? h.z() : -h.z()));
		}

		// Two CCW triangles per face, outward normals
		std::vector<Eigen::Vector3i> indices;
		indices.reserve(12);

		// -z face (normal 0, 0, -1)
		indices.emplace_back(0, 2, 3);
		indices.emplace_back(0, 3, 1);

		// +z face (normal 0, 0, +1)
		indices.emplace_back(4, 5, 7);
		indices.emplace_back(4, 7, 6);

		// -y face (normal 0, -1, 0)
		indices.emplace_back(0, 1, 5);
		indices.emplace_back(0, 5, 4);

		// +y face (normal 0, +1, 0)
		indices.emplace_back(2, 6, 7);
		indices.emplace_back(2, 7, 3);

		// -x face (normal -1, 0, 0)
		indices.emplace_back(0, 4, 6);
		indices.emplace_back(0, 6, 2);

		// +x face (normal +1, 0, 0)
		indices.emplace_back(1, 3, 7);
		indices.emplace_back(1, 7, 5);

		// Mirroring transforms flip winding: undo it to keep normals outward
		if (ApplyTransform(transform, points))
		{
			FlipWinding(indices);
		}

		Build(points, indices);
		return true;
	}

	// Text ===========================>
	// File-local: flattens one glyph shape into closed 2D contours.
	// Quadratic and cubic curves are subdivided into fixed segments.
	// Points closer than weld_tol to the previous point are dropped so that
	// CDT never receives nearly coincident input.
	static void FlattenGlyphShape(
		const stbtt_vertex* verts,
		int num_verts,
		float scale,
		float x_offset,
		float weld_tol,
		std::vector<std::vector<Eigen::Vector2f>>& out_contours)
	{
		const int CURVE_SEGMENTS = 8;
		const float tol2 = weld_tol * weld_tol;

		std::vector<Eigen::Vector2f> contour;
		Eigen::Vector2f cursor(0.0f, 0.0f);

		auto to_world = [&](short x, short y) -> Eigen::Vector2f
			{
				return Eigen::Vector2f(static_cast<float>(x) * scale + x_offset, static_cast<float>(y) * scale);
			};

		auto push_point = [&](const Eigen::Vector2f& p)
			{
				if (false == contour.empty() && (p - contour.back()).squaredNorm() < tol2) return;
				contour.push_back(p);
			};

		auto close_contour = [&]()
			{
				if (contour.size() >= 3)
				{
					// Drop the closing duplicate if the contour returned to its start
					if ((contour.front() - contour.back()).squaredNorm() < tol2)
					{
						contour.pop_back();
					}
					if (contour.size() >= 3)
					{
						out_contours.push_back(std::move(contour));
					}
				}
				contour.clear();
			};

		for (int i = 0; i < num_verts; ++i)
		{
			const stbtt_vertex& v = verts[i];

			if (v.type == STBTT_vmove)
			{
				close_contour();
				cursor = to_world(v.x, v.y);
				push_point(cursor);
			}
			else if (v.type == STBTT_vline)
			{
				cursor = to_world(v.x, v.y);
				push_point(cursor);
			}
			else if (v.type == STBTT_vcurve)
			{
				Eigen::Vector2f p0 = cursor;
				Eigen::Vector2f c = to_world(v.cx, v.cy);
				Eigen::Vector2f p1 = to_world(v.x, v.y);

				for (int s = 1; s <= CURVE_SEGMENTS; ++s)
				{
					float t = static_cast<float>(s) / static_cast<float>(CURVE_SEGMENTS);
					float u = 1.0f - t;
					Eigen::Vector2f p = u * u * p0 + 2.0f * u * t * c + t * t * p1;
					push_point(p);
				}
				cursor = p1;
			}
			else if (v.type == STBTT_vcubic)
			{
				Eigen::Vector2f p0 = cursor;
				Eigen::Vector2f c0 = to_world(v.cx, v.cy);
				Eigen::Vector2f c1 = to_world(v.cx1, v.cy1);
				Eigen::Vector2f p1 = to_world(v.x, v.y);

				for (int s = 1; s <= CURVE_SEGMENTS; ++s)
				{
					float t = static_cast<float>(s) / static_cast<float>(CURVE_SEGMENTS);
					float u = 1.0f - t;
					Eigen::Vector2f p =
						u * u * u * p0 +
						3.0f * u * u * t * c0 +
						3.0f * u * t * t * c1 +
						t * t * t * p1;
					push_point(p);
				}
				cursor = p1;
			}
		}

		close_contour();
	}

	// File-local: constrained Delaunay triangulation of glyph contours.
	// Outer area and holes are resolved by CDT from the constraint edges,
	// so contour winding does not matter. Output triangles are CCW.
	static bool TriangulateGlyphContours(
		const std::vector<std::vector<Eigen::Vector2f>>& contours,
		std::vector<Eigen::Vector2f>& out_pts,
		std::vector<Eigen::Vector3i>& out_tris)
	{
		out_pts.clear();
		out_tris.clear();
		if (contours.empty()) return false;

		std::vector<CDT::V2d<float>> pts;
		std::vector<CDT::Edge> edges;

		for (const auto& contour : contours)
		{
			unsigned int start = static_cast<unsigned int>(pts.size());
			unsigned int n = static_cast<unsigned int>(contour.size());

			for (const auto& p : contour)
			{
				pts.push_back(CDT::V2d<float>{ p.x(), p.y() });
			}
			for (unsigned int i = 0; i < n; ++i)
			{
				edges.push_back(CDT::Edge(start + i, start + (i + 1) % n));
			}
		}

		// Exact duplicates across contours would crash insertVertices
		CDT::RemoveDuplicatesAndRemapEdges(pts, edges);

		CDT::Triangulation<float> cdt;
		try
		{
			cdt.insertVertices(pts);
			cdt.insertEdges(edges);
			cdt.eraseOuterTrianglesAndHoles();
		}
		catch (const std::exception& e)
		{
			std::cout << "[Warning] FromText: CDT failed on a glyph: " << e.what() << std::endl;
			return false;
		}

		out_pts.reserve(cdt.vertices.size());
		for (const auto& v : cdt.vertices)
		{
			out_pts.emplace_back(v.x, v.y);
		}

		out_tris.reserve(cdt.triangles.size());
		for (const auto& t : cdt.triangles)
		{
			out_tris.emplace_back(
				static_cast<int>(t.vertices[0]),
				static_cast<int>(t.vertices[1]),
				static_cast<int>(t.vertices[2]));
		}

		return false == out_tris.empty();
	}

	// File-local: extrudes a triangulated glyph into a watertight solid.
	// Front face at z = +depth/2 (CCW, normal +z), back face mirrored,
	// side walls stitched along triangulation boundary edges. Because CCW
	// triangles keep the interior on the left of each directed edge, the
	// side quad orientation is correct regardless of glyph winding.
	static void ExtrudeGlyphTo3D(
		const std::vector<Eigen::Vector2f>& pts,
		const std::vector<Eigen::Vector3i>& tris,
		float depth,
		std::vector<Eigen::Vector3f>& out_points,
		std::vector<Eigen::Vector3i>& out_indices)
	{
		int base = static_cast<int>(out_points.size());
		int num_verts = static_cast<int>(pts.size());
		float half = depth * 0.5f;

		// Front vertex i: base + i. Back vertex i: base + num_verts + i.
		for (const auto& p : pts)
		{
			out_points.emplace_back(p.x(), p.y(), half);
		}
		for (const auto& p : pts)
		{
			out_points.emplace_back(p.x(), p.y(), -half);
		}

		for (const auto& t : tris)
		{
			out_indices.emplace_back(base + t.x(), base + t.y(), base + t.z());
		}
		for (const auto& t : tris)
		{
			out_indices.emplace_back(
				base + num_verts + t.x(),
				base + num_verts + t.z(),
				base + num_verts + t.y());
		}

		// Boundary edges: undirected edges referenced by exactly one triangle.
		// The stored direction is the one appearing in the CCW triangle.
		struct EdgeRecord
		{
			int count = 0;
			int from = -1;
			int to = -1;
		};
		robin_hood::unordered_map<uint64_t, EdgeRecord> edge_records;
		edge_records.reserve(tris.size() * 3);

		auto register_edge = [&](int a, int b)
			{
				uint64_t lo = static_cast<uint64_t>(std::min(a, b));
				uint64_t hi = static_cast<uint64_t>(std::max(a, b));
				EdgeRecord& rec = edge_records[(hi << 32) | lo];
				++rec.count;
				rec.from = a;
				rec.to = b;
			};

		for (const auto& t : tris)
		{
			register_edge(t.x(), t.y());
			register_edge(t.y(), t.z());
			register_edge(t.z(), t.x());
		}

		for (const auto& kvp : edge_records)
		{
			if (kvp.second.count != 1) continue;

			int a = kvp.second.from;
			int b = kvp.second.to;

			int front_a = base + a;
			int front_b = base + b;
			int back_a = base + num_verts + a;
			int back_b = base + num_verts + b;

			out_indices.emplace_back(back_a, back_b, front_b);
			out_indices.emplace_back(back_a, front_b, front_a);
		}
	}

	// File-local: decodes one UTF-8 codepoint starting at index i.
	// Advances i past the consumed bytes. Invalid sequences yield U+FFFD
	// and consume one byte so decoding always makes progress.
	static int DecodeUtf8(const std::string& s, size_t& i)
	{
		const unsigned char b0 = static_cast<unsigned char>(s[i]);

		// 1-byte: 0xxxxxxx
		if (b0 < 0x80)
		{
			i += 1;
			return static_cast<int>(b0);
		}

		auto is_continuation = [&](size_t k) -> bool
			{
				return k < s.size() && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80;
			};

		// 2-byte: 110xxxxx 10xxxxxx
		if ((b0 & 0xE0) == 0xC0 && is_continuation(i + 1))
		{
			int cp = (static_cast<int>(b0 & 0x1F) << 6)
				| (static_cast<int>(static_cast<unsigned char>(s[i + 1]) & 0x3F));
			i += 2;
			return cp;
		}

		// 3-byte: 1110xxxx 10xxxxxx 10xxxxxx
		if ((b0 & 0xF0) == 0xE0 && is_continuation(i + 1) && is_continuation(i + 2))
		{
			int cp = (static_cast<int>(b0 & 0x0F) << 12)
				| (static_cast<int>(static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6)
				| (static_cast<int>(static_cast<unsigned char>(s[i + 2]) & 0x3F));
			i += 3;
			return cp;
		}

		// 4-byte: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
		if ((b0 & 0xF8) == 0xF0 && is_continuation(i + 1) && is_continuation(i + 2) && is_continuation(i + 3))
		{
			int cp = (static_cast<int>(b0 & 0x07) << 18)
				| (static_cast<int>(static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12)
				| (static_cast<int>(static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6)
				| (static_cast<int>(static_cast<unsigned char>(s[i + 3]) & 0x3F));
			i += 4;
			return cp;
		}

		// Invalid byte: emit replacement character, consume one byte
		i += 1;
		return 0xFFFD;
	}
	// <=========================== Text

	bool Mesh::Build3DText(const std::string& text, const std::string& font_path, float size, float depth, const Eigen::Matrix4f& transform)
	{
		if (text.empty())
		{
			std::cout << "[Error] Build3DText: empty text." << std::endl;
			return false;
		}
		if (depth < EPSILON)
		{
			std::cout << "[Error] Build3DText: depth must be positive for a solid mesh." << std::endl;
			return false;
		}

		std::ifstream file(font_path, std::ios::binary | std::ios::ate);
		if (false == file.is_open())
		{
			std::cout << "[Error] Build3DText: cannot open font: " << font_path << std::endl;
			return false;
		}
		std::streamsize file_size = file.tellg();
		file.seekg(0, std::ios::beg);
		std::vector<unsigned char> font_data(static_cast<size_t>(file_size));
		if (false == static_cast<bool>(file.read(reinterpret_cast<char*>(font_data.data()), file_size)))
		{
			std::cout << "[Error] Build3DText: font read failed: " << font_path << std::endl;
			return false;
		}

		stbtt_fontinfo font;
		if (0 == stbtt_InitFont(&font, font_data.data(), stbtt_GetFontOffsetForIndex(font_data.data(), 0)))
		{
			std::cout << "[Error] Build3DText: invalid font file: " << font_path << std::endl;
			return false;
		}

		float scale = stbtt_ScaleForPixelHeight(&font, size);
		float weld_tol = size * 1e-3f;

		std::vector<Eigen::Vector3f> points;
		std::vector<Eigen::Vector3i> indices;

		float pen_x = 0.0f;
		int prev_cp = 0;
		size_t failed_glyphs = 0;
		size_t missing_glyphs = 0;

		// The text is treated as UTF-8: iterate codepoints, not bytes
		size_t byte_pos = 0;
		while (byte_pos < text.size())
		{
			int cp = DecodeUtf8(text, byte_pos);

			if (0 != prev_cp)
			{
				pen_x += scale * static_cast<float>(stbtt_GetCodepointKernAdvance(&font, prev_cp, cp));
			}

			// Report characters the font does not cover instead of
			// silently building .notdef boxes
			if (0 == stbtt_FindGlyphIndex(&font, cp))
			{
				++missing_glyphs;
				std::cout << "[Warning] Build3DText: font has no glyph for codepoint U+"
					<< std::hex << std::uppercase << cp << std::dec << std::nouppercase << std::endl;
			}

			stbtt_vertex* glyph_verts = nullptr;
			int num_verts = stbtt_GetCodepointShape(&font, cp, &glyph_verts);

			if (num_verts > 0 && glyph_verts)
			{
				std::vector<std::vector<Eigen::Vector2f>> contours;
				FlattenGlyphShape(glyph_verts, num_verts, scale, pen_x, weld_tol, contours);
				stbtt_FreeShape(&font, glyph_verts);

				std::vector<Eigen::Vector2f> glyph_pts;
				std::vector<Eigen::Vector3i> glyph_tris;
				if (TriangulateGlyphContours(contours, glyph_pts, glyph_tris))
				{
					ExtrudeGlyphTo3D(glyph_pts, glyph_tris, depth, points, indices);
				}
				else
				{
					++failed_glyphs;
				}
			}
			else if (glyph_verts)
			{
				stbtt_FreeShape(&font, glyph_verts);
			}

			int advance = 0;
			int left_side_bearing = 0;
			stbtt_GetCodepointHMetrics(&font, cp, &advance, &left_side_bearing);
			pen_x += scale * static_cast<float>(advance);

			prev_cp = cp;
		}

		if (failed_glyphs > 0)
		{
			std::cout << "[Warning] Build3DText: " << failed_glyphs << " glyphs failed to triangulate." << std::endl;
		}

		if (points.empty() || indices.empty())
		{
			std::cout << "[Error] Build3DText: no geometry produced from text." << std::endl;
			return false;
		}

		// Mirroring transforms flip winding: undo it to keep normals outward
		if (ApplyTransform(transform, points))
		{
			FlipWinding(indices);
		}

		Build(points, indices);
		return true;
	}

	// ------------------------------------------------------------
	// Operators
	// ------------------------------------------------------------

	OperatorIntersectionLoops::OperatorIntersectionLoops(Mesh* a, Mesh* b)
		: meshA(a), meshB(b)
	{
	}

	bool OperatorIntersectionLoops::Execute()
	{
		segments.clear();
		loops.clear();
		segments_by_face_a.clear();
		segments_by_face_b.clear();

		if (nullptr == meshA || nullptr == meshB) return false;
		if (0 == meshA->n_faces() || 0 == meshB->n_faces()) return false;

		size_t num_faces_b = meshB->n_faces();

		// Each B face writes only to its own slot: no mutex needed
		std::vector<std::vector<IntersectionSegment>> per_face_segments(num_faces_b);
		std::vector<int> face_indices(num_faces_b);
		std::iota(face_indices.begin(), face_indices.end(), 0);

		std::for_each(std::execution::par_unseq, face_indices.begin(), face_indices.end(), [&](int i)
			{
				CollectSegmentsForFaceB(i, per_face_segments[i]);
			});

		size_t total = 0;
		for (const auto& list : per_face_segments)
		{
			total += list.size();
		}
		segments.reserve(total);

		for (const auto& list : per_face_segments)
		{
			segments.insert(segments.end(), list.begin(), list.end());
		}

		CanonicalizeSegments();

		BuildFaceLookupTables();

		// Canonicalization invariants are checked here, before any
		// downstream stage consumes the endpoints. A failure here means
		// CDT input safety cannot be guaranteed: stop and report.
		if (false == ValidateCanonicalization()) return false;

		BuildLoops();

		// Endpoint validation failure means co-refinement on this data
		// would leave boundary edges: report failure to the caller.
		if (false == ValidateSegmentEndpoints()) return false;

		// Open chains violate the watertight-input invariant
		for (const auto& loop : loops)
		{
			if (false == loop.closed) return false;
		}

		return true;
	}

	void OperatorIntersectionLoops::CollectSegmentsForFaceB(int face_b_index, std::vector<IntersectionSegment>& out_segments) const
	{
		out_segments.clear();

		OpenMesh::FaceHandle fb = meshB->face_handle(face_b_index);
		if (meshB->status(fb).deleted()) return;

		Eigen::Vector3f b0, b1, b2;
		meshB->GetFaceVertices(fb, b0, b1, b2);

		const Eigen::Vector3f pad(EPSILON, EPSILON, EPSILON);
		Eigen::Vector3f b_min = b0.cwiseMin(b1).cwiseMin(b2) - pad;
		Eigen::Vector3f b_max = b0.cwiseMax(b1).cwiseMax(b2) + pad;

		std::vector<int> candidates;
		meshA->QueryOverlappingFaces(b_min, b_max, candidates);
		if (candidates.empty()) return;

		for (int face_a_index : candidates)
		{
			OpenMesh::FaceHandle fa = meshA->face_handle(face_a_index);
			if (meshA->status(fa).deleted()) continue;

			Eigen::Vector3f a0, a1, a2;
			meshA->GetFaceVertices(fa, a0, a1, a2);

			// Exact AABB rejection: hash query is cell-level and over-approximates
			Eigen::Vector3f a_min = a0.cwiseMin(a1).cwiseMin(a2) - pad;
			Eigen::Vector3f a_max = a0.cwiseMax(a1).cwiseMax(a2) + pad;
			if (false == Intersection::AABBtoAABB(a_min, a_max, b_min, b_max)) continue;

			Eigen::Vector3f ipA, ipB;
			Intersection::TriangleIntersectionResult result =
				Intersection::TriangleToTriangle(a0, a1, a2, b0, b1, b2, ipA, ipB);

			// Point contacts give no constraint edge for co-refinement: skip
			if (result.type != Intersection::TriangleTriangleIntersectionType::Segment) continue;

			// Near-degenerate segments poison CDT downstream: reject here
			if ((result.pointA - result.pointB).squaredNorm() < EPSILON * EPSILON) continue;

			IntersectionSegment seg;
			seg.p0 = result.pointA;
			seg.p1 = result.pointB;
			seg.face_a = face_a_index;
			seg.face_b = face_b_index;
			out_segments.push_back(seg);
		}
	}

	void OperatorIntersectionLoops::BuildFaceLookupTables()
	{
		for (int i = 0; i < static_cast<int>(segments.size()); ++i)
		{
			segments_by_face_a[segments[i].face_a].push_back(i);
			segments_by_face_b[segments[i].face_b].push_back(i);
		}
	}

	int OperatorIntersectionLoops::WeldEndpoint(
		const Eigen::Vector3f& p,
		robin_hood::unordered_map<Eigen::Vector3i, std::vector<int>, Int3Hash, Int3Equal>& cell_to_nodes,
		std::vector<Eigen::Vector3f>& node_positions) const
	{
		Eigen::Vector3i cell = QuantizePoint(p);

		// Scan the 27 neighbor cells so that two points within EPSILON
		// but straddling a quantization cell boundary still weld together.
		// Single-cell lookup would split closed loops into false open chains.
		for (int dz = -1; dz <= 1; ++dz)
		{
			for (int dy = -1; dy <= 1; ++dy)
			{
				for (int dx = -1; dx <= 1; ++dx)
				{
					Eigen::Vector3i neighbor(cell.x() + dx, cell.y() + dy, cell.z() + dz);
					auto it = cell_to_nodes.find(neighbor);
					if (it == cell_to_nodes.end()) continue;

					for (int node_id : it->second)
					{
						if ((node_positions[node_id] - p).squaredNorm() < EPSILON * EPSILON)
						{
							return node_id;
						}
					}
				}
			}
		}

		int new_id = static_cast<int>(node_positions.size());
		node_positions.push_back(p);
		cell_to_nodes[cell].push_back(new_id);
		return new_id;
	}

	OperatorIntersectionLoops::IntersectionLoop OperatorIntersectionLoops::TraceLoopFromSegment(
		int start_segment,
		const std::vector<std::pair<int, int>>& segment_nodes,
		const std::vector<std::vector<int>>& node_to_segments,
		const std::vector<Eigen::Vector3f>& node_positions,
		std::vector<char>& segment_used) const
	{
		IntersectionLoop loop;

		segment_used[start_segment] = 1;

		std::deque<int> node_seq;
		std::deque<int> seg_seq;
		node_seq.push_back(segment_nodes[start_segment].first);
		node_seq.push_back(segment_nodes[start_segment].second);
		seg_seq.push_back(start_segment);

		// Advances one step from the given node along any unused segment.
		// Returns the next node id, or -1 if no unused segment remains.
		auto step = [&](int current_node, int& out_segment) -> int
			{
				for (int seg_idx : node_to_segments[current_node])
				{
					if (segment_used[seg_idx]) continue;

					segment_used[seg_idx] = 1;
					out_segment = seg_idx;

					return (segment_nodes[seg_idx].first == current_node)
						? segment_nodes[seg_idx].second
						: segment_nodes[seg_idx].first;
				}
				return -1;
			};

		// Walk forward from the tail
		while (true)
		{
			int seg_idx = -1;
			int next_node = step(node_seq.back(), seg_idx);
			if (next_node < 0) break;

			seg_seq.push_back(seg_idx);

			if (next_node == node_seq.front())
			{
				loop.closed = true;
				break;
			}
			node_seq.push_back(next_node);
		}

		// Open so far: walk backward from the head to capture the rest of the chain
		if (false == loop.closed)
		{
			while (true)
			{
				int seg_idx = -1;
				int next_node = step(node_seq.front(), seg_idx);
				if (next_node < 0) break;

				seg_seq.push_front(seg_idx);

				if (next_node == node_seq.back())
				{
					loop.closed = true;
					break;
				}
				node_seq.push_front(next_node);
			}
		}

		loop.points.reserve(node_seq.size());
		for (int node_id : node_seq)
		{
			loop.points.push_back(node_positions[node_id]);
		}
		loop.segment_indices.assign(seg_seq.begin(), seg_seq.end());

		return loop;
	}

	void OperatorIntersectionLoops::BuildLoops()
	{
		loops.clear();
		if (segments.empty()) return;

		std::vector<Eigen::Vector3f> node_positions;
		node_positions.reserve(segments.size() * 2);
		robin_hood::unordered_map<Eigen::Vector3i, std::vector<int>, Int3Hash, Int3Equal> cell_to_nodes;

		// Weld segment endpoints into shared nodes
		std::vector<std::pair<int, int>> segment_nodes(segments.size());
		for (size_t i = 0; i < segments.size(); ++i)
		{
			int n0 = WeldEndpoint(segments[i].p0, cell_to_nodes, node_positions);
			int n1 = WeldEndpoint(segments[i].p1, cell_to_nodes, node_positions);
			segment_nodes[i] = { n0, n1 };
		}

		// Node to incident segment adjacency
		std::vector<std::vector<int>> node_to_segments(node_positions.size());
		std::vector<char> segment_used(segments.size(), 0);

		for (size_t i = 0; i < segments.size(); ++i)
		{
			// Both endpoints welded to the same node: zero length after welding.
			// Mark used so it cannot seed or join any loop.
			if (segment_nodes[i].first == segment_nodes[i].second)
			{
				segment_used[i] = 1;
				continue;
			}
			node_to_segments[segment_nodes[i].first].push_back(static_cast<int>(i));
			node_to_segments[segment_nodes[i].second].push_back(static_cast<int>(i));
		}

		// Chain unused segments into loops
		size_t open_chain_count = 0;
		for (size_t i = 0; i < segments.size(); ++i)
		{
			if (segment_used[i]) continue;

			IntersectionLoop loop = TraceLoopFromSegment(
				static_cast<int>(i), segment_nodes, node_to_segments, node_positions, segment_used);

			if (false == loop.closed)
			{
				++open_chain_count;
			}
			loops.push_back(std::move(loop));
		}

		if (open_chain_count > 0)
		{
			std::cout << "[Warning] BuildLoops: " << open_chain_count
				<< " open chains out of " << loops.size()
				<< " curves. Watertight inputs must yield closed loops only;"
				<< " check TriangleToTriangle for missed segments." << std::endl;
		}
	}

	void OperatorIntersectionLoops::CanonicalizeSegments()
	{
		canonical_snap_count = 0;
		if (segments.empty()) return;

		// Pass 1: snap endpoints to mesh features.
		// After this pass, every endpoint near a vertex carries the exact
		// vertex coordinate and every endpoint near an edge lies exactly
		// on that edge.
		std::vector<char> p0_snapped(segments.size(), 0);
		std::vector<char> p1_snapped(segments.size(), 0);

		for (size_t i = 0; i < segments.size(); ++i)
		{
			IntersectionSegment& seg = segments[i];

			bool s0 = false;
			bool s1 = false;
			seg.p0 = CanonicalizePoint(seg.p0, seg.face_a, seg.face_b, s0);
			seg.p1 = CanonicalizePoint(seg.p1, seg.face_a, seg.face_b, s1);

			if (s0) { p0_snapped[i] = 1; ++canonical_snap_count; }
			if (s1) { p1_snapped[i] = 1; ++canonical_snap_count; }
		}

		// Pass 2: global weld. Snapped points are seeded first so that a
		// cluster containing a feature-snapped point adopts the feature
		// coordinate as its representative, never an arbitrary free point.
		std::vector<Eigen::Vector3f> node_positions;
		node_positions.reserve(segments.size() * 2);
		robin_hood::unordered_map<Eigen::Vector3i, std::vector<int>, Int3Hash, Int3Equal> cell_to_nodes;

		for (size_t i = 0; i < segments.size(); ++i)
		{
			if (p0_snapped[i]) WeldEndpoint(segments[i].p0, cell_to_nodes, node_positions);
			if (p1_snapped[i]) WeldEndpoint(segments[i].p1, cell_to_nodes, node_positions);
		}

		for (auto& seg : segments)
		{
			seg.p0 = node_positions[WeldEndpoint(seg.p0, cell_to_nodes, node_positions)];
			seg.p1 = node_positions[WeldEndpoint(seg.p1, cell_to_nodes, node_positions)];
		}

		// Pass 3: drop segments that collapsed to zero length.
		// These would otherwise seed degenerate constraint edges in CDT.
		size_t before = segments.size();
		segments.erase(
			std::remove_if(segments.begin(), segments.end(),
				[](const IntersectionSegment& s)
				{
					return (s.p0 - s.p1).squaredNorm() < EPSILON * EPSILON;
				}),
			segments.end());

		size_t removed = before - segments.size();
		std::cout << "[Info] CanonicalizeSegments: " << canonical_snap_count
			<< " endpoints snapped to features, " << removed
			<< " degenerate segments removed, " << segments.size()
			<< " segments remain." << std::endl;
	}

	bool OperatorIntersectionLoops::TrySnapToFaceVertex(
		const Eigen::Vector3f& p,
		const Mesh* mesh,
		int face_index,
		Eigen::Vector3f& out_point) const
	{
		Eigen::Vector3f v0, v1, v2;
		mesh->GetFaceVertices(mesh->face_handle(face_index), v0, v1, v2);

		const Eigen::Vector3f* verts[3] = { &v0, &v1, &v2 };

		// Pick the nearest vertex within tolerance, not the first match,
		// so the snap is deterministic when two vertices are both close.
		float best_d2 = EPSILON * EPSILON;
		int best = -1;
		for (int i = 0; i < 3; ++i)
		{
			float d2 = (p - *verts[i]).squaredNorm();
			if (d2 < best_d2)
			{
				best_d2 = d2;
				best = i;
			}
		}

		if (best < 0) return false;

		out_point = *verts[best];
		return true;
	}

	bool OperatorIntersectionLoops::TrySnapToFaceEdge(
		const Eigen::Vector3f& p,
		const Mesh* mesh,
		int face_index,
		Eigen::Vector3f& out_point) const
	{
		Eigen::Vector3f v0, v1, v2;
		mesh->GetFaceVertices(mesh->face_handle(face_index), v0, v1, v2);

		const Eigen::Vector3f* ea[3] = { &v0, &v1, &v2 };
		const Eigen::Vector3f* eb[3] = { &v1, &v2, &v0 };

		// Pick the nearest edge projection within tolerance
		float best_d2 = EPSILON * EPSILON;
		Eigen::Vector3f best_proj = Eigen::Vector3f::Zero();
		bool found = false;

		for (int i = 0; i < 3; ++i)
		{
			Eigen::Vector3f ab = *eb[i] - *ea[i];
			float l2 = ab.squaredNorm();
			if (l2 < EPSILON * EPSILON) continue;

			float t = (p - *ea[i]).dot(ab) / l2;
			t = std::max(0.0f, std::min(1.0f, t));

			Eigen::Vector3f proj = *ea[i] + t * ab;
			float d2 = (p - proj).squaredNorm();
			if (d2 < best_d2)
			{
				best_d2 = d2;
				best_proj = proj;
				found = true;
			}
		}

		if (false == found) return false;

		out_point = best_proj;
		return true;
	}

	bool OperatorIntersectionLoops::ValidateCanonicalization()
	{
		canonical_validation_failure_count = 0;

		// All three invariants are checked even if an earlier one fails,
		// so a single run reports the full picture for debugging.
		bool idempotent = ValidateCanonicalIdempotence();
		bool no_degenerate = ValidateNoDegenerateSegments();
		bool spacing_ok = ValidateCanonicalConstraintSpacing();

		bool ok = idempotent && no_degenerate && spacing_ok;

		if (ok)
		{
			std::cout << "[Info] ValidateCanonicalization: all invariants hold for "
				<< segments.size() << " segments." << std::endl;
		}
		else
		{
			std::cout << "[Error] ValidateCanonicalization: "
				<< canonical_validation_failure_count
				<< " invariant violations. CDT input safety is not guaranteed." << std::endl;
		}
		return ok;
	}

	bool OperatorIntersectionLoops::ValidateCanonicalIdempotence()
	{
		// Re-applying CanonicalizePoint to a canonical endpoint may move it
		// by a few float ULPs: edge re-projection (a + t * ab) rounds, and
		// adjacent faces sharing the same geometric edge fetch its vertices
		// in different order, rounding differently. At coordinate magnitude
		// 50, one ULP is already 3.8e-6, so an absolute tolerance below
		// that is physically unsatisfiable in float.
		//
		// The invariant that actually matters is that re-canonicalization
		// cannot flip any identity decision: movement strictly below half
		// of EPSILON can neither split a welded pair (< EPSILON) nor merge
		// a separated pair (>= EPSILON). Genuine snap-weld interference
		// moves points by about EPSILON and is still caught.
		const float move_tol = 0.5f * EPSILON;
		const float move_tol2 = move_tol * move_tol;

		size_t failures = 0;
		float max_move = 0.0f;

		for (size_t i = 0; i < segments.size(); ++i)
		{
			const IntersectionSegment& seg = segments[i];
			const Eigen::Vector3f* endpoints[2] = { &seg.p0, &seg.p1 };

			for (int e = 0; e < 2; ++e)
			{
				bool snapped = false;
				Eigen::Vector3f again = CanonicalizePoint(*endpoints[e], seg.face_a, seg.face_b, snapped);

				float moved2 = (again - *endpoints[e]).squaredNorm();
				max_move = std::max(max_move, std::sqrt(moved2));

				if (moved2 > move_tol2)
				{
					++failures;
					std::cout << "[Error] CanonicalIdempotence: segment " << i
						<< " endpoint " << e << " moved by " << std::sqrt(moved2)
						<< " on re-canonicalization (tolerance " << move_tol
						<< "). Snap and weld tolerances interfere." << std::endl;
				}
			}
		}

		// Max movement is reported even on success: a creeping increase
		// toward the tolerance is an early warning that coordinate
		// magnitudes are getting too large for float precision vs EPSILON.
		std::cout << "[Info] CanonicalIdempotence: max re-canonicalization movement "
			<< max_move << " (tolerance " << move_tol << ")." << std::endl;

		canonical_validation_failure_count += failures;
		return 0 == failures;
	}

	bool OperatorIntersectionLoops::ValidateNoDegenerateSegments()
	{
		size_t failures = 0;
		for (size_t i = 0; i < segments.size(); ++i)
		{
			float len2 = (segments[i].p0 - segments[i].p1).squaredNorm();
			if (len2 < EPSILON * EPSILON)
			{
				++failures;
				std::cout << "[Error] NoDegenerateSegments: segment " << i
					<< " has length " << std::sqrt(len2)
					<< " after canonicalization. It must have been removed." << std::endl;
			}
		}

		canonical_validation_failure_count += failures;
		return 0 == failures;
	}

	size_t OperatorIntersectionLoops::CheckConstraintSpacingForFace(
		const Mesh* mesh,
		int face_index,
		const std::vector<int>& segment_indices,
		const char* mesh_label) const
	{
		// Gather exactly the point set that per-face CDT will receive:
		// the three face vertices plus every constraint endpoint.
		Eigen::Vector3f v0, v1, v2;
		mesh->GetFaceVertices(mesh->face_handle(face_index), v0, v1, v2);

		std::vector<Eigen::Vector3f> pts;
		pts.reserve(3 + segment_indices.size() * 2);
		pts.push_back(v0);
		pts.push_back(v1);
		pts.push_back(v2);

		for (int seg_idx : segment_indices)
		{
			pts.push_back(segments[seg_idx].p0);
			pts.push_back(segments[seg_idx].p1);
		}

		// Remove bit-identical duplicates first. Identical coordinates are
		// the intended outcome of canonicalization and are safe for CDT
		// after RemoveDuplicatesAndRemapEdges.
		std::sort(pts.begin(), pts.end(),
			[](const Eigen::Vector3f& a, const Eigen::Vector3f& b)
			{
				if (a.x() != b.x()) return a.x() < b.x();
				if (a.y() != b.y()) return a.y() < b.y();
				return a.z() < b.z();
			});
		pts.erase(std::unique(pts.begin(), pts.end(),
			[](const Eigen::Vector3f& a, const Eigen::Vector3f& b)
			{
				return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
			}),
			pts.end());

		// Any remaining pair closer than EPSILON is exactly the failure
		// mode that produced degenerate CDT triangles: two points that are
		// logically distinct to CDT but geometrically nearly coincident.
		size_t failures = 0;
		for (size_t i = 0; i < pts.size(); ++i)
		{
			for (size_t j = i + 1; j < pts.size(); ++j)
			{
				float d2 = (pts[i] - pts[j]).squaredNorm();
				if (d2 < EPSILON * EPSILON)
				{
					++failures;
					std::cout << "[Error] ConstraintSpacing: " << mesh_label
						<< " face " << face_index
						<< " has two distinct constraint points only "
						<< std::sqrt(d2) << " apart:"
						<< " (" << pts[i].x() << ", " << pts[i].y() << ", " << pts[i].z() << ") and"
						<< " (" << pts[j].x() << ", " << pts[j].y() << ", " << pts[j].z() << ")."
						<< " CDT on this face would produce degenerate triangles." << std::endl;
				}
			}
		}

		return failures;
	}

	bool OperatorIntersectionLoops::ValidateCanonicalConstraintSpacing()
	{
		size_t failures = 0;

		for (const auto& kvp : segments_by_face_a)
		{
			failures += CheckConstraintSpacingForFace(meshA, kvp.first, kvp.second, "meshA");
		}
		for (const auto& kvp : segments_by_face_b)
		{
			failures += CheckConstraintSpacingForFace(meshB, kvp.first, kvp.second, "meshB");
		}

		if (0 == failures)
		{
			std::cout << "[Info] ConstraintSpacing: all per-face constraint point sets"
				<< " are safely separated (>= EPSILON) for "
				<< segments_by_face_a.size() << " A faces and "
				<< segments_by_face_b.size() << " B faces." << std::endl;
		}

		canonical_validation_failure_count += failures;
		return 0 == failures;
	}

	bool OperatorIntersectionLoops::ValidateSegmentEndpoints()
	{
		validation_failure_count = 0;

		// Canonicalization may move an endpoint by up to EPSILON, so a
		// point originally within EPSILON of the other face can now be up
		// to 2 * EPSILON away. The validation tolerance accounts for that.
		const float validate_epsilon = 2.0f * EPSILON;

		for (size_t i = 0; i < segments.size(); ++i)
		{
			const IntersectionSegment& seg = segments[i];

			Eigen::Vector3f a0, a1, a2;
			Eigen::Vector3f b0, b1, b2;
			meshA->GetFaceVertices(meshA->face_handle(seg.face_a), a0, a1, a2);
			meshB->GetFaceVertices(meshB->face_handle(seg.face_b), b0, b1, b2);

			// Every endpoint must lie on BOTH source triangles. A point
			// outside either one cannot be inserted consistently into both
			// meshes during co-refinement and will leave boundary edges.
			const Eigen::Vector3f* endpoints[2] = { &seg.p0, &seg.p1 };
			for (int e = 0; e < 2; ++e)
			{
				Intersection::PointToTriangleResult on_a =
					Intersection::PointToTriangle(*endpoints[e], a0, a1, a2, validate_epsilon);
				Intersection::PointToTriangleResult on_b =
					Intersection::PointToTriangle(*endpoints[e], b0, b1, b2, validate_epsilon);

				bool ok_a = on_a.type != Intersection::PointToTriangleType::Outside;
				bool ok_b = on_b.type != Intersection::PointToTriangleType::Outside;

				if (false == ok_a || false == ok_b)
				{
					++validation_failure_count;

					std::cout << "[Error] Segment " << i << " endpoint " << e
						<< " (" << endpoints[e]->x() << ", " << endpoints[e]->y() << ", " << endpoints[e]->z() << ")"
						<< " not on " << (ok_a ? "" : "meshA face ") << (ok_a ? "" : std::to_string(seg.face_a))
						<< (false == ok_a && false == ok_b ? " and " : "")
						<< (ok_b ? "" : "meshB face ") << (ok_b ? "" : std::to_string(seg.face_b))
						<< " (plane dist A: " << on_a.plane_distance
						<< ", B: " << on_b.plane_distance << ")" << std::endl;
				}
			}
		}

		if (validation_failure_count > 0)
		{
			std::cout << "[Error] ValidateSegmentEndpoints: " << validation_failure_count
				<< " endpoint failures out of " << segments.size() * 2
				<< " endpoints. Co-refinement on this data will leave boundary edges." << std::endl;
			return false;
		}
		else
		{
			std::cout << "[Info] ValidateSegmentEndpoints: All " << segments.size() * 2
				<< " endpoints valid. Co-refinement on this data should yield no boundary edges." << std::endl;
		}
		return true;
	}

	Eigen::Vector3f OperatorIntersectionLoops::CanonicalizePoint(
		const Eigen::Vector3f& p,
		int face_a,
		int face_b,
		bool& out_snapped) const
	{
		// Priority order matters and must be globally consistent:
		// vertex of A, vertex of B, edge of A, edge of B, then free point.
		// A vertex is the most constrained feature, so it wins over an edge.
		// Without a fixed priority, two segments sharing one logical point
		// could canonicalize it to two slightly different coordinates,
		// which is exactly the degenerate-CDT-input failure mode.
		Eigen::Vector3f snapped;

		if (TrySnapToFaceVertex(p, meshA, face_a, snapped))
		{
			out_snapped = true;
			return snapped;
		}
		if (TrySnapToFaceVertex(p, meshB, face_b, snapped))
		{
			out_snapped = true;
			return snapped;
		}
		if (TrySnapToFaceEdge(p, meshA, face_a, snapped))
		{
			out_snapped = true;
			return snapped;
		}
		if (TrySnapToFaceEdge(p, meshB, face_b, snapped))
		{
			out_snapped = true;
			return snapped;
		}

		out_snapped = false;
		return p;
	}

	OperatorCoRefine::OperatorCoRefine(Mesh* a, Mesh* b, const OperatorIntersectionLoops* loops)
		: meshA(a), meshB(b), loop_op(loops)
	{
	}

	bool OperatorCoRefine::Execute()
	{
		new_boundary_edge_count_a = 0;
		new_boundary_edge_count_b = 0;

		if (nullptr == meshA || nullptr == meshB || nullptr == loop_op) return false;

		if (loop_op->GetSegments().empty())
		{
			std::cout << "[Info] OperatorCoRefine: no intersection segments, nothing to refine." << std::endl;
			return true;
		}

		// Segments carry positions, not handles, so refining A first does
		// not invalidate the data needed to refine B.
		bool ok_a = RefineMesh(meshA, loop_op->GetSegmentIndicesByFaceA(), "meshA", new_boundary_edge_count_a);
		bool ok_b = RefineMesh(meshB, loop_op->GetSegmentIndicesByFaceB(), "meshB", new_boundary_edge_count_b);

		return ok_a && ok_b;
	}

	bool OperatorCoRefine::RefineMesh(
		Mesh* mesh,
		const robin_hood::unordered_map<int, std::vector<int>>& segment_indices_by_face,
		const char* mesh_label,
		size_t& out_new_boundary_edge_count)
	{
		out_new_boundary_edge_count = 0;

		// Record the boundary state of the input BEFORE any modification.
		// Open inputs (scan data) are allowed: pre-existing boundary edges
		// are a property of the input, not a refinement failure. The
		// invariant this stage must uphold is only that refinement CREATES
		// no boundary edges.
		std::vector<BoundaryEdge> input_boundary;
		CollectInputBoundaryEdges(mesh, input_boundary);

		robin_hood::unordered_map<int, FaceRefinementInput> inputs;
		GatherFaceInputs(segment_indices_by_face, inputs);
		PropagateEdgePointsToNeighbors(mesh, inputs);

		// Triangle soup: every 3 consecutive points form one triangle.
		// Welding happens once at the end, relying on canonicalization
		// having made logically identical points bit-identical.
		std::vector<Eigen::Vector3f> soup;
		soup.reserve(mesh->n_faces() * 3 + loop_op->GetSegments().size() * 12);

		size_t refined_faces = 0;
		size_t failed_faces = 0;
		size_t num_faces = mesh->n_faces();

		for (size_t i = 0; i < num_faces; ++i)
		{
			OpenMesh::FaceHandle fh = mesh->face_handle(static_cast<int>(i));
			if (mesh->status(fh).deleted()) continue;

			auto it = inputs.find(static_cast<int>(i));
			if (it == inputs.end())
			{
				// Untouched face: passes through unchanged
				Eigen::Vector3f v0, v1, v2;
				mesh->GetFaceVertices(fh, v0, v1, v2);
				soup.push_back(v0);
				soup.push_back(v1);
				soup.push_back(v2);
				continue;
			}

			++refined_faces;
			if (false == TriangulateFace(mesh, static_cast<int>(i), it->second, soup))
			{
				++failed_faces;
			}
		}

		std::vector<Eigen::Vector3f> points;
		std::vector<Eigen::Vector3i> indices;
		WeldTriangleSoup(soup, points, indices);

		// Triangles collapsed by welding punch holes into the surface.
		// A nonzero count means the mesh contains features smaller than
		// EPSILON, i.e. the weld tolerance is destroying geometry. That is
		// a different root cause than a refinement bug, so it is reported
		// separately here.
		size_t soup_triangles = soup.size() / 3;
		size_t collapsed = soup_triangles - indices.size();
		if (collapsed > 0)
		{
			std::cout << "[Warning] RefineMesh(" << mesh_label << "): " << collapsed
				<< " triangles collapsed during welding. The mesh has features"
				<< " smaller than EPSILON; expect holes at those spots." << std::endl;
		}

		mesh->clear();
		mesh->Build(points, indices);

		out_new_boundary_edge_count = CountNewBoundaryEdges(mesh, input_boundary, mesh_label);

		std::cout << "[Info] RefineMesh(" << mesh_label << "): " << refined_faces
			<< " faces refined, " << failed_faces << " failures, "
			<< indices.size() << " triangles total, "
			<< input_boundary.size() << " input boundary edges, "
			<< out_new_boundary_edge_count << " NEW boundary edges after rebuild." << std::endl;

		// New boundary edges mean the co-refinement guarantee is broken on
		// this data: report failure loudly instead of patching.
		return 0 == failed_faces && 0 == out_new_boundary_edge_count;
	}

	void OperatorCoRefine::GatherFaceInputs(
		const robin_hood::unordered_map<int, std::vector<int>>& segment_indices_by_face,
		robin_hood::unordered_map<int, FaceRefinementInput>& out_inputs) const
	{
		const auto& segments = loop_op->GetSegments();

		for (const auto& kvp : segment_indices_by_face)
		{
			FaceRefinementInput& input = out_inputs[kvp.first];

			for (int seg_idx : kvp.second)
			{
				const auto& seg = segments[seg_idx];
				AddUniquePoint(input.points, seg.p0);
				AddUniquePoint(input.points, seg.p1);
				input.constraints.push_back({ seg.p0, seg.p1 });
			}
		}
	}

	void OperatorCoRefine::PropagateEdgePointsToNeighbors(
		Mesh* mesh,
		robin_hood::unordered_map<int, FaceRefinementInput>& inputs) const
	{
		// A point lying on a face edge splits that edge, so the face across
		// the edge must be split at the same point even when it carries no
		// intersection segments of its own. Skipping this creates
		// T-junctions, which surface as residual boundary edges after the
		// rebuild. This was the failure mode of earlier attempts.
		struct Propagation
		{
			int target_face;
			Eigen::Vector3f point;
		};
		std::vector<Propagation> propagations;

		for (const auto& kvp : inputs)
		{
			OpenMesh::FaceHandle fh = mesh->face_handle(kvp.first);

			Eigen::Vector3f c0, c1, c2;
			mesh->GetFaceVertices(fh, c0, c1, c2);

			for (const auto& p : kvp.second.points)
			{
				// Corner points are already shared by all incident faces
				if ((p - c0).squaredNorm() < EPSILON * EPSILON) continue;
				if ((p - c1).squaredNorm() < EPSILON * EPSILON) continue;
				if ((p - c2).squaredNorm() < EPSILON * EPSILON) continue;

				for (auto heh : mesh->fh_range(fh))
				{
					const auto& pf = mesh->point(mesh->from_vertex_handle(heh));
					const auto& pt = mesh->point(mesh->to_vertex_handle(heh));
					Eigen::Vector3f ef(pf[0], pf[1], pf[2]);
					Eigen::Vector3f et(pt[0], pt[1], pt[2]);

					if (Distance::PointToLineSegmentSquared(p, ef, et) >= EPSILON * EPSILON) continue;

					OpenMesh::HalfedgeHandle opp = mesh->opposite_halfedge_handle(heh);
					if (mesh->is_boundary(opp)) break;

					OpenMesh::FaceHandle nfh = mesh->face_handle(opp);
					if (false == nfh.is_valid() || mesh->status(nfh).deleted()) break;

					propagations.push_back({ nfh.idx(), p });
					break;
				}
			}
		}

		// Applied after the scan so the map is not mutated while iterating.
		// operator[] intentionally creates entries for faces that had no
		// segments: they still need retriangulation to split their edge.
		for (const auto& prop : propagations)
		{
			AddUniquePoint(inputs[prop.target_face].points, prop.point);
		}
	}

	bool OperatorCoRefine::TriangulateFace(
		const Mesh* mesh,
		int face_index,
		const FaceRefinementInput& input,
		std::vector<Eigen::Vector3f>& out_soup) const
	{
		Eigen::Vector3f c0, c1, c2;
		mesh->GetFaceVertices(mesh->face_handle(face_index), c0, c1, c2);

		// On any failure the original triangle is emitted so the mesh
		// stays closed, and false is returned so the failure stays loud.
		auto emit_original = [&]()
			{
				out_soup.push_back(c0);
				out_soup.push_back(c1);
				out_soup.push_back(c2);
			};

		Eigen::Vector3f n = (c1 - c0).cross(c2 - c0);
		float n_len = n.norm();
		if (n_len < 1e-12f)
		{
			std::cout << "[Error] TriangulateFace: degenerate face " << face_index << std::endl;
			emit_original();
			return false;
		}
		n /= n_len;

		// Orthonormal in-plane basis. (e1, e2, n) is right-handed, so CCW
		// triangles in 2D map back to triangles whose normal matches n.
		Eigen::Vector3f e1 = (c1 - c0).normalized();
		Eigen::Vector3f e2 = n.cross(e1);

		// Point list: corners first, then unique extra points.
		// Exact-bit dedupe is correct because canonicalization made
		// logically identical points bit-identical.
		std::vector<Eigen::Vector3f> pts3;
		pts3.reserve(3 + input.points.size());
		pts3.push_back(c0);
		pts3.push_back(c1);
		pts3.push_back(c2);

		auto find_exact = [&](const Eigen::Vector3f& p) -> int
			{
				for (size_t i = 0; i < pts3.size(); ++i)
				{
					if (pts3[i].x() == p.x() && pts3[i].y() == p.y() && pts3[i].z() == p.z())
						return static_cast<int>(i);
				}
				return -1;
			};

		for (const auto& p : input.points)
		{
			if (find_exact(p) < 0) pts3.push_back(p);
		}

		std::vector<CDT::V2d<float>> pts2;
		pts2.reserve(pts3.size());
		for (const auto& p : pts3)
		{
			Eigen::Vector3f d = p - c0;
			pts2.push_back(CDT::V2d<float>{ d.dot(e1), d.dot(e2) });
		}

		std::vector<CDT::Edge> edges;
		robin_hood::unordered_set<uint64_t> edge_keys;
		auto add_edge = [&](int a, int b)
			{
				if (a == b || a < 0 || b < 0) return;
				uint64_t lo = static_cast<uint64_t>(std::min(a, b));
				uint64_t hi = static_cast<uint64_t>(std::max(a, b));
				uint64_t key = (hi << 32) | lo;
				if (edge_keys.find(key) != edge_keys.end()) return;
				edge_keys.insert(key);
				edges.push_back(CDT::Edge(static_cast<unsigned int>(a), static_cast<unsigned int>(b)));
			};

		// Face boundary edges, split by the extra points lying on them.
		// The neighbor across each edge receives the same split points via
		// PropagateEdgePointsToNeighbors, which is what prevents
		// T-junctions. Constraining the boundary also lets
		// eraseOuterTriangles trim everything outside the face while
		// keeping regions enclosed by intersection loops (it peels from
		// the outside only and does not treat inner loops as holes).
		const int corner_edges[3][2] = { { 0, 1 }, { 1, 2 }, { 2, 0 } };
		for (int eidx = 0; eidx < 3; ++eidx)
		{
			const Eigen::Vector3f& a = pts3[corner_edges[eidx][0]];
			const Eigen::Vector3f& b = pts3[corner_edges[eidx][1]];
			Eigen::Vector3f ab = b - a;
			float l2 = ab.squaredNorm();

			std::vector<std::pair<float, int>> on_edge;
			for (size_t i = 3; i < pts3.size(); ++i)
			{
				if (Distance::PointToLineSegmentSquared(pts3[i], a, b) < EPSILON * EPSILON)
				{
					float t = (l2 > 0.0f) ? (pts3[i] - a).dot(ab) / l2 : 0.0f;
					on_edge.push_back({ t, static_cast<int>(i) });
				}
			}
			std::sort(on_edge.begin(), on_edge.end());

			int prev = corner_edges[eidx][0];
			for (const auto& tp : on_edge)
			{
				add_edge(prev, tp.second);
				prev = tp.second;
			}
			add_edge(prev, corner_edges[eidx][1]);
		}

		// Intersection segment constraints
		for (const auto& con : input.constraints)
		{
			int ia = find_exact(con.first);
			int ib = find_exact(con.second);
			if (ia < 0 || ib < 0)
			{
				std::cout << "[Error] TriangulateFace: constraint endpoint missing"
					<< " from point set on face " << face_index << std::endl;
				emit_original();
				return false;
			}
			add_edge(ia, ib);
		}

		CDT::Triangulation<float> cdt;
		try
		{
			cdt.insertVertices(pts2);
			cdt.insertEdges(edges);
			cdt.eraseOuterTriangles();
		}
		catch (const std::exception& ex)
		{
			std::cout << "[Error] TriangulateFace: CDT failed on face " << face_index
				<< ": " << ex.what() << std::endl;
			emit_original();
			return false;
		}

		if (cdt.triangles.empty())
		{
			std::cout << "[Error] TriangulateFace: CDT produced no triangles"
				<< " on face " << face_index << std::endl;
			emit_original();
			return false;
		}

		// CDT should not invent vertices with this input, but if it ever
		// does they are unprojected back onto the face plane.
		std::vector<Eigen::Vector3f> out_positions;
		out_positions.reserve(cdt.vertices.size());
		for (size_t i = 0; i < cdt.vertices.size(); ++i)
		{
			if (i < pts3.size())
			{
				out_positions.push_back(pts3[i]);
			}
			else
			{
				out_positions.push_back(c0 + cdt.vertices[i].x * e1 + cdt.vertices[i].y * e2);
			}
		}

		for (const auto& t : cdt.triangles)
		{
			out_soup.push_back(out_positions[t.vertices[0]]);
			out_soup.push_back(out_positions[t.vertices[1]]);
			out_soup.push_back(out_positions[t.vertices[2]]);
		}

		return true;
	}

	void OperatorCoRefine::AddUniquePoint(std::vector<Eigen::Vector3f>& pts, const Eigen::Vector3f& p) const
	{
		// Exact comparison is intentional: canonicalization guarantees
		// logically identical points are bit-identical.
		for (const auto& q : pts)
		{
			if (q.x() == p.x() && q.y() == p.y() && q.z() == p.z()) return;
		}
		pts.push_back(p);
	}

	void OperatorCoRefine::WeldTriangleSoup(
		const std::vector<Eigen::Vector3f>& soup,
		std::vector<Eigen::Vector3f>& out_points,
		std::vector<Eigen::Vector3i>& out_indices) const
	{
		out_points.clear();
		out_indices.clear();

		robin_hood::unordered_map<Eigen::Vector3f, int, Vector3fHash, Vector3fEqual> vertex_map;
		vertex_map.reserve(soup.size());

		for (size_t i = 0; i + 2 < soup.size(); i += 3)
		{
			Eigen::Vector3i tri;
			for (int j = 0; j < 3; ++j)
			{
				const Eigen::Vector3f& p = soup[i + j];
				auto it = vertex_map.find(p);
				if (it != vertex_map.end())
				{
					tri[j] = it->second;
				}
				else
				{
					int ni = static_cast<int>(out_points.size());
					out_points.push_back(p);
					vertex_map[p] = ni;
					tri[j] = ni;
				}
			}

			// Collapsed after welding: drop
			if (tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0]) continue;
			out_indices.push_back(tri);
		}
	}

	void OperatorCoRefine::CollectInputBoundaryEdges(const Mesh* mesh, std::vector<BoundaryEdge>& out_edges) const
	{
		out_edges.clear();

		const Eigen::Vector3f pad(2.0f * EPSILON, 2.0f * EPSILON, 2.0f * EPSILON);

		for (size_t i = 0; i < mesh->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = mesh->edge_handle(static_cast<int>(i));
			if (mesh->status(eh).deleted()) continue;
			if (false == mesh->is_boundary(eh)) continue;

			OpenMesh::HalfedgeHandle heh = mesh->halfedge_handle(eh, 0);
			auto pf = mesh->point(mesh->from_vertex_handle(heh));
			auto pt = mesh->point(mesh->to_vertex_handle(heh));

			BoundaryEdge be;
			be.a = Eigen::Vector3f(pf[0], pf[1], pf[2]);
			be.b = Eigen::Vector3f(pt[0], pt[1], pt[2]);

			// Padded AABB for the cheap reject in containment tests. The
			// padding covers the snap movement of up to EPSILON plus the
			// matching tolerance.
			be.aabb_min = be.a.cwiseMin(be.b) - pad;
			be.aabb_max = be.a.cwiseMax(be.b) + pad;

			out_edges.push_back(be);
		}
	}

	bool OperatorCoRefine::IsSubEdgeOfInputBoundary(
		const Eigen::Vector3f& a,
		const Eigen::Vector3f& b,
		const std::vector<BoundaryEdge>& input_boundary) const
	{
		// A rebuilt boundary edge is legitimate when it lies ON some input
		// boundary edge. Exact pair matching would be wrong: an
		// intersection point landing on an input boundary edge legally
		// splits it into sub-edges whose endpoint pairs never existed in
		// the input. Testing both endpoints and the midpoint against the
		// input segment accepts exactly those sub-edges and nothing else.
		//
		// Tolerance is 2 * EPSILON because canonicalization may have moved
		// an endpoint by up to EPSILON off the original line.
		const float tol = 2.0f * EPSILON;
		const float tol2 = tol * tol;

		Eigen::Vector3f mid = (a + b) * 0.5f;

		for (const auto& be : input_boundary)
		{
			// Cheap reject on the padded AABB
			if (a.x() < be.aabb_min.x() && b.x() < be.aabb_min.x()) continue;
			if (a.x() > be.aabb_max.x() && b.x() > be.aabb_max.x()) continue;
			if (a.y() < be.aabb_min.y() && b.y() < be.aabb_min.y()) continue;
			if (a.y() > be.aabb_max.y() && b.y() > be.aabb_max.y()) continue;
			if (a.z() < be.aabb_min.z() && b.z() < be.aabb_min.z()) continue;
			if (a.z() > be.aabb_max.z() && b.z() > be.aabb_max.z()) continue;

			if (Distance::PointToLineSegmentSquared(a, be.a, be.b) >= tol2) continue;
			if (Distance::PointToLineSegmentSquared(b, be.a, be.b) >= tol2) continue;
			if (Distance::PointToLineSegmentSquared(mid, be.a, be.b) >= tol2) continue;

			return true;
		}

		return false;
	}

	size_t OperatorCoRefine::CountNewBoundaryEdges(
		const Mesh* mesh,
		const std::vector<BoundaryEdge>& input_boundary,
		const char* mesh_label) const
	{
		size_t new_count = 0;
		const size_t report_limit = 10;

		for (size_t i = 0; i < mesh->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = mesh->edge_handle(static_cast<int>(i));
			if (mesh->status(eh).deleted()) continue;
			if (false == mesh->is_boundary(eh)) continue;

			OpenMesh::HalfedgeHandle heh = mesh->halfedge_handle(eh, 0);
			auto pf = mesh->point(mesh->from_vertex_handle(heh));
			auto pt = mesh->point(mesh->to_vertex_handle(heh));
			Eigen::Vector3f a(pf[0], pf[1], pf[2]);
			Eigen::Vector3f b(pt[0], pt[1], pt[2]);

			if (IsSubEdgeOfInputBoundary(a, b, input_boundary)) continue;

			++new_count;
			if (new_count <= report_limit)
			{
				std::cout << "[Error] NewBoundaryEdge(" << mesh_label << "): ("
					<< a.x() << ", " << a.y() << ", " << a.z() << ") - ("
					<< b.x() << ", " << b.y() << ", " << b.z()
					<< ") was not present in the input." << std::endl;
			}
		}

		if (new_count > report_limit)
		{
			std::cout << "[Error] NewBoundaryEdge(" << mesh_label << "): "
				<< (new_count - report_limit) << " more not shown." << std::endl;
		}

		return new_count;
	}

	size_t OperatorCoRefine::CountBoundaryEdges(const Mesh* mesh) const
	{
		size_t count = 0;
		for (size_t i = 0; i < mesh->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = mesh->edge_handle(static_cast<int>(i));
			if (mesh->status(eh).deleted()) continue;
			if (mesh->is_boundary(eh)) ++count;
		}
		return count;
	}

	OperatorBoolean::OperatorBoolean(
		Type t,
		Mesh* a,
		Mesh* b,
		const OperatorIntersectionLoops* loops,
		Mesh* res)
		: type(t), meshA(a), meshB(b), loop_op(loops), result(res)
	{
	}

	bool OperatorBoolean::Execute()
	{
		result_boundary_edge_count = 0;
		endpoint_segments.clear();
		data_a = MeshSideData();
		data_b = MeshSideData();

		if (nullptr == meshA || nullptr == meshB || nullptr == loop_op || nullptr == result) return false;
		if (0 == meshA->n_faces() || 0 == meshB->n_faces()) return false;

		// Endpoint lookup table for seam reconstruction
		const auto& segments = loop_op->GetSegments();
		endpoint_segments.reserve(segments.size() * 2);
		for (int i = 0; i < static_cast<int>(segments.size()); ++i)
		{
			endpoint_segments[segments[i].p0].push_back(i);
			endpoint_segments[segments[i].p1].push_back(i);
		}

		// Input topology, for the record. It decides which later stages
		// are defined, but the stages executed below are valid for open
		// and closed inputs alike.
		size_t boundary_a = CountBoundaryEdges(meshA);
		size_t boundary_b = CountBoundaryEdges(meshB);

		std::cout << "[Info] OperatorBoolean: meshA is "
			<< (0 == boundary_a ? "closed" : "open") << " (" << boundary_a
			<< " boundary edges), meshB is "
			<< (0 == boundary_b ? "closed" : "open") << " (" << boundary_b
			<< " boundary edges)." << std::endl;

		// STAGE GATE: the pipeline is intentionally cut here. Only seam
		// reconstruction and the seam-bounded flood fill run, each with
		// its own validation, so that any failure is attributable to
		// exactly one stage. Classification (ExecuteSolidBoolean and
		// ExecuteOpenMeshTrim further down in this file) reconnects only
		// after this stage is verified on real data.
		if (false == BuildSeamEdgeFlags(meshA, data_a, "meshA")) return false;
		if (false == BuildSeamEdgeFlags(meshB, data_b, "meshB")) return false;

		if (false == ValidateSeamIntegrity(meshA, data_a, "meshA")) return false;
		if (false == ValidateSeamIntegrity(meshB, data_b, "meshB")) return false;

		if (false == BuildFacePatches(meshA, data_a, "meshA")) return false;
		if (false == BuildFacePatches(meshB, data_b, "meshB")) return false;

		ReportPatchStatistics(meshA, data_a, "meshA");
		ReportPatchStatistics(meshB, data_b, "meshB");

		std::cout << "[Info] OperatorBoolean: stage gate reached."
			<< " Seam and patch structures are built and validated."
			<< " Classification and assembly are NOT executed yet." << std::endl;

		return true;
	}

	void OperatorBoolean::ReportPatchStatistics(const Mesh* mesh, const MeshSideData& data, const char* label) const
	{
		// Per-patch face counts make a flood fill leak visible at a
		// glance: a leak merges patches, so instead of several plausibly
		// sized patches one oversized patch appears.
		std::vector<size_t> patch_faces(data.patch_count, 0);
		std::vector<size_t> patch_seam_edges(data.patch_count, 0);

		for (size_t i = 0; i < mesh->n_faces(); ++i)
		{
			OpenMesh::FaceHandle fh = mesh->face_handle(static_cast<int>(i));
			if (mesh->status(fh).deleted()) continue;

			int patch = data.face_patch[i];
			if (patch < 0) continue;
			++patch_faces[patch];
		}

		// Seam edges incident to each patch: every patch created by the
		// cut must touch the seam. A patch with zero seam edges is a
		// separate connected component, which is informative on its own.
		for (size_t i = 0; i < mesh->n_edges(); ++i)
		{
			if (0 == data.edge_is_seam[i]) continue;

			OpenMesh::EdgeHandle eh = mesh->edge_handle(static_cast<int>(i));
			OpenMesh::HalfedgeHandle h0 = mesh->halfedge_handle(eh, 0);
			OpenMesh::HalfedgeHandle h1 = mesh->halfedge_handle(eh, 1);

			if (false == mesh->is_boundary(h0))
			{
				int p = data.face_patch[mesh->face_handle(h0).idx()];
				if (p >= 0) ++patch_seam_edges[p];
			}
			if (false == mesh->is_boundary(h1))
			{
				int p = data.face_patch[mesh->face_handle(h1).idx()];
				if (p >= 0) ++patch_seam_edges[p];
			}
		}

		const int report_limit = 20;
		for (int p = 0; p < data.patch_count; ++p)
		{
			if (p >= report_limit)
			{
				std::cout << "[Info] PatchStats(" << label << "): "
					<< (data.patch_count - report_limit) << " more patches not shown." << std::endl;
				break;
			}
			std::cout << "[Info] PatchStats(" << label << "): patch " << p
				<< " has " << patch_faces[p] << " faces, touches "
				<< patch_seam_edges[p] << " seam edge sides." << std::endl;
		}
	}

	bool OperatorBoolean::ExecuteSolidBoolean()
	{
		// Disjoint or fully contained inputs have no segments. The same
		// pipeline still works: no seams means one patch per connected
		// component, classified as a whole by one ray cast each.
		MeshSideData data_a;
		MeshSideData data_b;

		if (false == BuildSeamEdgeFlags(meshA, data_a, "meshA")) return false;
		if (false == BuildSeamEdgeFlags(meshB, data_b, "meshB")) return false;

		// Seam integrity must hold BEFORE flood fill: a broken seam would
		// let inside and outside patches merge silently.
		if (false == ValidateSeamIntegrity(meshA, data_a, "meshA")) return false;
		if (false == ValidateSeamIntegrity(meshB, data_b, "meshB")) return false;

		if (false == BuildFacePatches(meshA, data_a, "meshA")) return false;
		if (false == BuildFacePatches(meshB, data_b, "meshB")) return false;

		if (false == ClassifyPatches(meshA, meshB, data_a, "meshA")) return false;
		if (false == ClassifyPatches(meshB, meshA, data_b, "meshB")) return false;

		// Cross check: classifications must flip across every seam edge.
		// This catches both flood fill leaks and wrong ray verdicts.
		if (false == ValidatePatchAdjacency(meshA, data_a, "meshA")) return false;
		if (false == ValidatePatchAdjacency(meshB, data_b, "meshB")) return false;

		// Face selection per boolean type. For Difference the kept B faces
		// bound removed volume, so their winding flips to stay outward.
		bool keep_inside_a = (type == Intersection);
		bool keep_inside_b = (type == Intersection || type == Difference);
		bool flip_b = (type == Difference);

		std::vector<Eigen::Vector3f> soup;
		soup.reserve((meshA->n_faces() + meshB->n_faces()) * 3);

		CollectFacesForBoolean(meshA, data_a, keep_inside_a, false, soup);
		CollectFacesForBoolean(meshB, data_b, keep_inside_b, flip_b, soup);

		std::vector<Eigen::Vector3f> points;
		std::vector<Eigen::Vector3i> indices;
		WeldTriangleSoup(soup, points, indices);

		if (indices.empty())
		{
			std::cout << "[Warning] OperatorBoolean: result is empty." << std::endl;
		}

		result->clear();
		result->Build(points, indices);

		result_boundary_edge_count = CountBoundaryEdges(result);

		std::cout << "[Info] OperatorBoolean: " << indices.size() << " result triangles, "
			<< result_boundary_edge_count << " boundary edges in result." << std::endl;

		return 0 == result_boundary_edge_count;
	}

	bool OperatorBoolean::ExecuteOpenMeshTrim(Mesh* open_mesh, const Mesh* solid, const char* open_label)
	{
		// All structure building and validation runs on the open mesh
		// only. The solid is used purely as the classification reference,
		// where ray parity is valid.
		MeshSideData data;

		if (false == BuildSeamEdgeFlags(open_mesh, data, open_label)) return false;
		if (false == ValidateSeamIntegrity(open_mesh, data, open_label)) return false;
		if (false == BuildFacePatches(open_mesh, data, open_label)) return false;
		if (false == ClassifyPatches(open_mesh, solid, data, open_label)) return false;
		if (false == ValidatePatchAdjacency(open_mesh, data, open_label)) return false;

		// The open mesh's own border is a property of the input and stays
		// legitimate in the result. It is recorded before assembly so the
		// result validation can separate it from pipeline-created holes.
		std::vector<BoundaryEdge> input_boundary;
		CollectInputBoundaryEdges(open_mesh, input_boundary);

		bool keep_inside = (Intersection == type);

		std::vector<Eigen::Vector3f> soup;
		soup.reserve(open_mesh->n_faces() * 3);
		CollectFacesForBoolean(open_mesh, data, keep_inside, false, soup);

		std::vector<Eigen::Vector3f> points;
		std::vector<Eigen::Vector3i> indices;
		WeldTriangleSoup(soup, points, indices);

		if (indices.empty())
		{
			std::cout << "[Warning] OperatorBoolean: trim result is empty." << std::endl;
		}

		result->clear();
		result->Build(points, indices);

		result_boundary_edge_count = CountBoundaryEdges(result);

		// A trimmed surface is open by construction: its boundary is the
		// seam (the cut line) plus the surviving pieces of the input
		// border. Only boundary edges outside those two sets are failures.
		size_t invalid = CountInvalidResultBoundaryEdges(result, input_boundary, "result");

		std::cout << "[Info] OperatorBoolean: trim kept " << indices.size()
			<< " triangles from " << open_label << ", "
			<< result_boundary_edge_count << " boundary edges, "
			<< invalid << " invalid." << std::endl;

		return 0 == invalid;
	}

	bool OperatorBoolean::BuildSeamEdgeFlags(const Mesh* mesh, MeshSideData& data, const char* label) const
	{
		const auto& segments = loop_op->GetSegments();

		data.edge_is_seam.assign(mesh->n_edges(), 0);

		size_t seam_count = 0;
		double seam_length = 0.0;

		for (size_t i = 0; i < mesh->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = mesh->edge_handle(static_cast<int>(i));
			if (mesh->status(eh).deleted()) continue;

			OpenMesh::HalfedgeHandle heh = mesh->halfedge_handle(eh, 0);
			auto pf = mesh->point(mesh->from_vertex_handle(heh));
			auto pt = mesh->point(mesh->to_vertex_handle(heh));
			Eigen::Vector3f a(pf[0], pf[1], pf[2]);
			Eigen::Vector3f b(pt[0], pt[1], pt[2]);

			// Both endpoints must be canonical intersection points
			auto it_a = endpoint_segments.find(a);
			if (it_a == endpoint_segments.end()) continue;
			auto it_b = endpoint_segments.find(b);
			if (it_b == endpoint_segments.end()) continue;

			// The edge is a seam only when it runs ALONG some segment, not
			// merely between two seam vertices (CDT diagonals do that too).
			// Testing both endpoints and the midpoint distinguishes the two
			// cases, and also accepts sub-edges of a constraint that CDT
			// split at another canonical point lying on it.
			Eigen::Vector3f mid = (a + b) * 0.5f;

			auto runs_along = [&](int seg_idx) -> bool
				{
					const auto& s = segments[seg_idx];
					if (Distance::PointToLineSegmentSquared(a, s.p0, s.p1) >= EPSILON * EPSILON) return false;
					if (Distance::PointToLineSegmentSquared(b, s.p0, s.p1) >= EPSILON * EPSILON) return false;
					if (Distance::PointToLineSegmentSquared(mid, s.p0, s.p1) >= EPSILON * EPSILON) return false;
					return true;
				};

			bool is_seam = false;
			for (int seg_idx : it_a->second)
			{
				if (runs_along(seg_idx)) { is_seam = true; break; }
			}
			if (false == is_seam)
			{
				for (int seg_idx : it_b->second)
				{
					if (runs_along(seg_idx)) { is_seam = true; break; }
				}
			}

			if (is_seam)
			{
				data.edge_is_seam[i] = 1;
				++seam_count;
				seam_length += static_cast<double>((b - a).norm());
			}
		}

		std::cout << "[Info] BuildSeamEdgeFlags(" << label << "): " << seam_count
			<< " seam edges, total length " << seam_length << "." << std::endl;
		return true;
	}

	bool OperatorBoolean::ValidateSeamIntegrity(const Mesh* mesh, const MeshSideData& data, const char* label) const
	{
		const auto& segments = loop_op->GetSegments();
		size_t failures = 0;

		// Check 1: every canonical endpoint must survive as a mesh vertex.
		// A missing endpoint means welding merged it away or CDT dropped
		// it, leaving a gap where flood fill could leak.
		robin_hood::unordered_set<Eigen::Vector3f, Vector3fBitHash, Vector3fBitEqual> vertex_set;
		vertex_set.reserve(mesh->n_vertices());
		for (size_t i = 0; i < mesh->n_vertices(); ++i)
		{
			OpenMesh::VertexHandle vh = mesh->vertex_handle(static_cast<int>(i));
			if (mesh->status(vh).deleted()) continue;
			auto p = mesh->point(vh);
			vertex_set.insert(Eigen::Vector3f(p[0], p[1], p[2]));
		}

		for (const auto& kvp : endpoint_segments)
		{
			if (vertex_set.find(kvp.first) == vertex_set.end())
			{
				++failures;
				std::cout << "[Error] SeamIntegrity(" << label << "): canonical point ("
					<< kvp.first.x() << ", " << kvp.first.y() << ", " << kvp.first.z()
					<< ") is not a vertex of the rebuilt mesh." << std::endl;
			}
		}

		// Check 2: seam degree parity. Seam curves are closed, so every
		// vertex must touch an even number of seam edges. Odd degree marks
		// a broken seam end, the exact leak this stage must rule out.
		std::vector<int> seam_degree(mesh->n_vertices(), 0);
		double seam_length = 0.0;
		for (size_t i = 0; i < mesh->n_edges(); ++i)
		{
			if (0 == data.edge_is_seam[i]) continue;

			OpenMesh::HalfedgeHandle heh = mesh->halfedge_handle(mesh->edge_handle(static_cast<int>(i)), 0);
			seam_degree[mesh->from_vertex_handle(heh).idx()] += 1;
			seam_degree[mesh->to_vertex_handle(heh).idx()] += 1;

			auto pf = mesh->point(mesh->from_vertex_handle(heh));
			auto pt = mesh->point(mesh->to_vertex_handle(heh));
			Eigen::Vector3f a(pf[0], pf[1], pf[2]);
			Eigen::Vector3f b(pt[0], pt[1], pt[2]);
			seam_length += static_cast<double>((b - a).norm());
		}

		for (size_t i = 0; i < seam_degree.size(); ++i)
		{
			if (0 != (seam_degree[i] % 2))
			{
				++failures;
				auto p = mesh->point(mesh->vertex_handle(static_cast<int>(i)));
				std::cout << "[Error] SeamIntegrity(" << label << "): vertex ("
					<< p[0] << ", " << p[1] << ", " << p[2] << ") has odd seam degree "
					<< seam_degree[i] << ": the seam is broken here." << std::endl;
			}
		}

		// Check 3: total seam length must match total segment length, since
		// every segment is carved exactly once into each mesh. Missing or
		// duplicated sub-edges surface as a mismatch here.
		double expected = 0.0;
		for (const auto& s : segments)
		{
			expected += static_cast<double>((s.p1 - s.p0).norm());
		}

		double diff = std::abs(seam_length - expected);
		double tol = std::max(1e-6, expected * 1e-3);
		if (diff > tol)
		{
			++failures;
			std::cout << "[Error] SeamIntegrity(" << label << "): seam length " << seam_length
				<< " does not match total segment length " << expected
				<< " (diff " << diff << ", tolerance " << tol << ")." << std::endl;
		}

		if (0 == failures)
		{
			std::cout << "[Info] SeamIntegrity(" << label << "): seam is closed and complete"
				<< " (length " << seam_length << " vs expected " << expected << ")." << std::endl;
			return true;
		}

		std::cout << "[Error] SeamIntegrity(" << label << "): " << failures << " failures." << std::endl;
		return false;
	}

	bool OperatorBoolean::BuildFacePatches(const Mesh* mesh, MeshSideData& data, const char* label) const
	{
		size_t num_faces = mesh->n_faces();
		data.face_patch.assign(num_faces, -1);
		data.patch_count = 0;

		std::vector<int> stack;

		for (size_t start = 0; start < num_faces; ++start)
		{
			OpenMesh::FaceHandle sfh = mesh->face_handle(static_cast<int>(start));
			if (mesh->status(sfh).deleted()) continue;
			if (data.face_patch[start] >= 0) continue;

			int patch_id = data.patch_count++;
			data.face_patch[start] = patch_id;
			stack.clear();
			stack.push_back(static_cast<int>(start));

			while (false == stack.empty())
			{
				int f = stack.back();
				stack.pop_back();

				OpenMesh::FaceHandle fh = mesh->face_handle(f);
				for (auto heh : mesh->fh_range(fh))
				{
					// Seam edges are walls: flood fill must not cross them
					if (data.edge_is_seam[mesh->edge_handle(heh).idx()]) continue;

					OpenMesh::HalfedgeHandle opp = mesh->opposite_halfedge_handle(heh);
					if (mesh->is_boundary(opp)) continue;

					OpenMesh::FaceHandle nfh = mesh->face_handle(opp);
					if (false == nfh.is_valid()) continue;
					if (mesh->status(nfh).deleted()) continue;
					if (data.face_patch[nfh.idx()] >= 0) continue;

					data.face_patch[nfh.idx()] = patch_id;
					stack.push_back(nfh.idx());
				}
			}
		}

		data.patch_side.assign(data.patch_count, FaceSide::Unknown);

		std::cout << "[Info] BuildFacePatches(" << label << "): " << data.patch_count
			<< " patches." << std::endl;
		return data.patch_count > 0;
	}

	bool OperatorBoolean::ClassifyPatches(const Mesh* mesh, const Mesh* other, MeshSideData& data, const char* label) const
	{
		// Representative face per patch: the largest one. Large faces sit
		// away from the dense triangulation near the seam, which keeps the
		// ray origin clear of the degeneracy-prone region.
		std::vector<int> rep_face(data.patch_count, -1);
		std::vector<float> rep_area(data.patch_count, -1.0f);

		for (size_t i = 0; i < mesh->n_faces(); ++i)
		{
			OpenMesh::FaceHandle fh = mesh->face_handle(static_cast<int>(i));
			if (mesh->status(fh).deleted()) continue;

			int patch = data.face_patch[i];
			if (patch < 0) continue;

			Eigen::Vector3f v0, v1, v2;
			mesh->GetFaceVertices(fh, v0, v1, v2);
			float area2 = (v1 - v0).cross(v2 - v0).squaredNorm();

			if (area2 > rep_area[patch])
			{
				rep_area[patch] = area2;
				rep_face[patch] = static_cast<int>(i);
			}
		}

		for (int p = 0; p < data.patch_count; ++p)
		{
			if (rep_face[p] < 0)
			{
				std::cout << "[Error] ClassifyPatches(" << label << "): patch " << p
					<< " has no representative face." << std::endl;
				return false;
			}

			Eigen::Vector3f v0, v1, v2;
			mesh->GetFaceVertices(mesh->face_handle(rep_face[p]), v0, v1, v2);
			Eigen::Vector3f centroid = (v0 + v1 + v2) / 3.0f;

			FaceSide side = RobustPointInMesh(centroid, other);
			if (FaceSide::Inside != side && FaceSide::Outside != side)
			{
				// OnSurface means coplanar face contact, which is a
				// separate stage by agreement: fail loudly, never guess.
				std::cout << "[Error] ClassifyPatches(" << label << "): patch " << p
					<< " could not be classified ("
					<< (side == FaceSide::OnSurface ? "OnSurface" : "Unknown")
					<< ")." << std::endl;
				return false;
			}

			data.patch_side[p] = side;
		}

		size_t inside = 0;
		for (const auto& s : data.patch_side)
		{
			if (FaceSide::Inside == s) ++inside;
		}
		std::cout << "[Info] ClassifyPatches(" << label << "): " << inside << " inside, "
			<< (data.patch_side.size() - inside) << " outside patches." << std::endl;
		return true;
	}

	OperatorBoolean::FaceSide OperatorBoolean::RobustPointInMesh(const Eigen::Vector3f& p, const Mesh* other) const
	{
		// On-surface check first: a point on the other mesh cannot be
		// classified by ray parity at all.
		const Eigen::Vector3f pad(EPSILON, EPSILON, EPSILON);
		std::vector<int> near_faces;
		other->QueryOverlappingFaces(p - pad, p + pad, near_faces);
		for (int f : near_faces)
		{
			OpenMesh::FaceHandle fh = other->face_handle(f);
			if (other->status(fh).deleted()) continue;

			Eigen::Vector3f v0, v1, v2;
			other->GetFaceVertices(fh, v0, v1, v2);
			if (Intersection::PointToTriangle(p, v0, v1, v2).type != Intersection::PointToTriangleType::Outside)
			{
				return FaceSide::OnSurface;
			}
		}

		// Bounding box of the other mesh: ray length and a cheap reject
		Eigen::Vector3f bb_min;
		Eigen::Vector3f bb_max;
		bb_min.setConstant(std::numeric_limits<float>::max());
		bb_max.setConstant(-std::numeric_limits<float>::max());
		for (size_t i = 0; i < other->n_vertices(); ++i)
		{
			OpenMesh::VertexHandle vh = other->vertex_handle(static_cast<int>(i));
			if (other->status(vh).deleted()) continue;
			auto q = other->point(vh);
			Eigen::Vector3f v(q[0], q[1], q[2]);
			bb_min = bb_min.cwiseMin(v);
			bb_max = bb_max.cwiseMax(v);
		}

		if (p.x() < bb_min.x() - EPSILON || p.x() > bb_max.x() + EPSILON ||
			p.y() < bb_min.y() - EPSILON || p.y() > bb_max.y() + EPSILON ||
			p.z() < bb_min.z() - EPSILON || p.z() > bb_max.z() + EPSILON)
		{
			return FaceSide::Outside;
		}

		float ray_len = (bb_max - bb_min).norm() * 2.0f;

		// Non-axis-aligned directions: CSG inputs are often axis-aligned,
		// so axis rays would constantly graze faces and edges.
		const Eigen::Vector3f dirs[8] = {
			Eigen::Vector3f(0.813f, 0.342f, 0.471f).normalized(),
			Eigen::Vector3f(-0.357f, 0.866f, 0.350f).normalized(),
			Eigen::Vector3f(0.275f, -0.488f, 0.828f).normalized(),
			Eigen::Vector3f(-0.629f, -0.451f, -0.633f).normalized(),
			Eigen::Vector3f(0.522f, 0.711f, -0.471f).normalized(),
			Eigen::Vector3f(-0.804f, 0.221f, 0.552f).normalized(),
			Eigen::Vector3f(0.183f, 0.620f, 0.763f).normalized(),
			Eigen::Vector3f(0.948f, -0.276f, 0.158f).normalized()
		};

		std::vector<int> candidates;
		for (int d = 0; d < 8; ++d)
		{
			Eigen::Vector3f end = p + dirs[d] * ray_len;
			Eigen::Vector3f q_min = p.cwiseMin(end) - pad;
			Eigen::Vector3f q_max = p.cwiseMax(end) + pad;
			other->QueryOverlappingFaces(q_min, q_max, candidates);

			size_t hit_count = 0;
			bool degenerate = false;

			for (int f : candidates)
			{
				OpenMesh::FaceHandle fh = other->face_handle(f);
				if (other->status(fh).deleted()) continue;

				Eigen::Vector3f v0, v1, v2;
				other->GetFaceVertices(fh, v0, v1, v2);

				Eigen::Vector3f hit;
				if (false == Intersection::RayToTriangle(p, dirs[d], v0, v1, v2, hit)) continue;

				// A hit near a vertex or an edge is counted by every face
				// sharing that feature, which corrupts the parity. Such a
				// ray is unreliable: discard it and try the next direction.
				Intersection::PointToTriangleResult cls = Intersection::PointToTriangle(hit, v0, v1, v2);
				if (cls.type != Intersection::PointToTriangleType::Inside)
				{
					degenerate = true;
					break;
				}
				++hit_count;
			}

			if (degenerate) continue;

			return (1 == (hit_count % 2)) ? FaceSide::Inside : FaceSide::Outside;
		}

		std::cout << "[Error] RobustPointInMesh: all ray directions degenerate at ("
			<< p.x() << ", " << p.y() << ", " << p.z() << ")." << std::endl;
		return FaceSide::Unknown;
	}

	bool OperatorBoolean::ValidatePatchAdjacency(const Mesh* mesh, const MeshSideData& data, const char* label) const
	{
		// Crossing the intersection curve on a watertight surface flips
		// which side of the other mesh you are on. A seam edge whose two
		// incident faces classify the same way therefore proves either a
		// flood fill leak (patches wrongly merged across a broken seam) or
		// a wrong ray verdict. Either way the boolean must not proceed.
		size_t failures = 0;

		for (size_t i = 0; i < mesh->n_edges(); ++i)
		{
			if (0 == data.edge_is_seam[i]) continue;

			OpenMesh::EdgeHandle eh = mesh->edge_handle(static_cast<int>(i));
			OpenMesh::HalfedgeHandle h0 = mesh->halfedge_handle(eh, 0);
			OpenMesh::HalfedgeHandle h1 = mesh->halfedge_handle(eh, 1);

			if (mesh->is_boundary(h0) || mesh->is_boundary(h1)) continue;

			FaceSide s0 = data.patch_side[data.face_patch[mesh->face_handle(h0).idx()]];
			FaceSide s1 = data.patch_side[data.face_patch[mesh->face_handle(h1).idx()]];

			if (s0 == s1)
			{
				++failures;
				if (failures <= 10)
				{
					std::cout << "[Error] PatchAdjacency(" << label << "): seam edge " << i
						<< " has the same classification on both sides." << std::endl;
				}
			}
		}

		if (0 == failures)
		{
			std::cout << "[Info] PatchAdjacency(" << label << "): all seam edges separate"
				<< " inside from outside." << std::endl;
			return true;
		}

		std::cout << "[Error] PatchAdjacency(" << label << "): " << failures
			<< " seam edges violate the inside/outside flip." << std::endl;
		return false;
	}

	void OperatorBoolean::CollectFacesForBoolean(
		const Mesh* mesh,
		const MeshSideData& data,
		bool keep_inside,
		bool flip_winding,
		std::vector<Eigen::Vector3f>& out_soup) const
	{
		FaceSide wanted = keep_inside ? FaceSide::Inside : FaceSide::Outside;

		for (size_t i = 0; i < mesh->n_faces(); ++i)
		{
			OpenMesh::FaceHandle fh = mesh->face_handle(static_cast<int>(i));
			if (mesh->status(fh).deleted()) continue;

			int patch = data.face_patch[i];
			if (patch < 0) continue;
			if (data.patch_side[patch] != wanted) continue;

			Eigen::Vector3f v0, v1, v2;
			mesh->GetFaceVertices(fh, v0, v1, v2);

			out_soup.push_back(v0);
			if (flip_winding)
			{
				out_soup.push_back(v2);
				out_soup.push_back(v1);
			}
			else
			{
				out_soup.push_back(v1);
				out_soup.push_back(v2);
			}
		}
	}

	void OperatorBoolean::WeldTriangleSoup(
		const std::vector<Eigen::Vector3f>& soup,
		std::vector<Eigen::Vector3f>& out_points,
		std::vector<Eigen::Vector3i>& out_indices) const
	{
		out_points.clear();
		out_indices.clear();

		robin_hood::unordered_map<Eigen::Vector3f, int, Vector3fHash, Vector3fEqual> vertex_map;
		vertex_map.reserve(soup.size());

		for (size_t i = 0; i + 2 < soup.size(); i += 3)
		{
			Eigen::Vector3i tri;
			for (int j = 0; j < 3; ++j)
			{
				const Eigen::Vector3f& p = soup[i + j];
				auto it = vertex_map.find(p);
				if (it != vertex_map.end())
				{
					tri[j] = it->second;
				}
				else
				{
					int ni = static_cast<int>(out_points.size());
					out_points.push_back(p);
					vertex_map[p] = ni;
					tri[j] = ni;
				}
			}

			// Collapsed after welding: drop
			if (tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0]) continue;
			out_indices.push_back(tri);
		}
	}

	void OperatorBoolean::CollectInputBoundaryEdges(const Mesh* mesh, std::vector<BoundaryEdge>& out_edges) const
	{
		out_edges.clear();

		const Eigen::Vector3f pad(2.0f * EPSILON, 2.0f * EPSILON, 2.0f * EPSILON);

		for (size_t i = 0; i < mesh->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = mesh->edge_handle(static_cast<int>(i));
			if (mesh->status(eh).deleted()) continue;
			if (false == mesh->is_boundary(eh)) continue;

			OpenMesh::HalfedgeHandle heh = mesh->halfedge_handle(eh, 0);
			auto pf = mesh->point(mesh->from_vertex_handle(heh));
			auto pt = mesh->point(mesh->to_vertex_handle(heh));

			BoundaryEdge be;
			be.a = Eigen::Vector3f(pf[0], pf[1], pf[2]);
			be.b = Eigen::Vector3f(pt[0], pt[1], pt[2]);

			// Padded AABB for the cheap reject in containment tests
			be.aabb_min = be.a.cwiseMin(be.b) - pad;
			be.aabb_max = be.a.cwiseMax(be.b) + pad;

			out_edges.push_back(be);
		}
	}

	bool OperatorBoolean::IsSubEdgeOfInputBoundary(
		const Eigen::Vector3f& a,
		const Eigen::Vector3f& b,
		const std::vector<BoundaryEdge>& input_boundary) const
	{
		// Same containment logic as in OperatorCoRefine: a result boundary
		// edge is legitimate when it lies ON some input boundary edge.
		// Endpoint pair matching would wrongly reject sub-edges created by
		// welding splits along the border.
		const float tol = 2.0f * EPSILON;
		const float tol2 = tol * tol;

		Eigen::Vector3f mid = (a + b) * 0.5f;

		for (const auto& be : input_boundary)
		{
			// Cheap reject on the padded AABB
			if (a.x() < be.aabb_min.x() && b.x() < be.aabb_min.x()) continue;
			if (a.x() > be.aabb_max.x() && b.x() > be.aabb_max.x()) continue;
			if (a.y() < be.aabb_min.y() && b.y() < be.aabb_min.y()) continue;
			if (a.y() > be.aabb_max.y() && b.y() > be.aabb_max.y()) continue;
			if (a.z() < be.aabb_min.z() && b.z() < be.aabb_min.z()) continue;
			if (a.z() > be.aabb_max.z() && b.z() > be.aabb_max.z()) continue;

			if (Distance::PointToLineSegmentSquared(a, be.a, be.b) >= tol2) continue;
			if (Distance::PointToLineSegmentSquared(b, be.a, be.b) >= tol2) continue;
			if (Distance::PointToLineSegmentSquared(mid, be.a, be.b) >= tol2) continue;

			return true;
		}

		return false;
	}

	bool OperatorBoolean::IsSeamSubEdge(const Eigen::Vector3f& a, const Eigen::Vector3f& b) const
	{
		// Same containment test as BuildSeamEdgeFlags: the edge must run
		// ALONG some intersection segment, endpoints and midpoint alike.
		const auto& segments = loop_op->GetSegments();
		Eigen::Vector3f mid = (a + b) * 0.5f;

		for (const auto& s : segments)
		{
			if (Distance::PointToLineSegmentSquared(a, s.p0, s.p1) >= EPSILON * EPSILON) continue;
			if (Distance::PointToLineSegmentSquared(b, s.p0, s.p1) >= EPSILON * EPSILON) continue;
			if (Distance::PointToLineSegmentSquared(mid, s.p0, s.p1) >= EPSILON * EPSILON) continue;
			return true;
		}
		return false;
	}

	size_t OperatorBoolean::CountInvalidResultBoundaryEdges(
		const Mesh* mesh,
		const std::vector<BoundaryEdge>& open_input_boundary,
		const char* label) const
	{
		// The trim result is expected to be open: along the seam where the
		// solid cut it, and along surviving pieces of the input border.
		// Any other boundary edge is a hole the pipeline created and is a
		// hard failure.
		size_t invalid = 0;
		const size_t report_limit = 10;

		for (size_t i = 0; i < mesh->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = mesh->edge_handle(static_cast<int>(i));
			if (mesh->status(eh).deleted()) continue;
			if (false == mesh->is_boundary(eh)) continue;

			OpenMesh::HalfedgeHandle heh = mesh->halfedge_handle(eh, 0);
			auto pf = mesh->point(mesh->from_vertex_handle(heh));
			auto pt = mesh->point(mesh->to_vertex_handle(heh));
			Eigen::Vector3f a(pf[0], pf[1], pf[2]);
			Eigen::Vector3f b(pt[0], pt[1], pt[2]);

			if (IsSeamSubEdge(a, b)) continue;
			if (IsSubEdgeOfInputBoundary(a, b, open_input_boundary)) continue;

			++invalid;
			if (invalid <= report_limit)
			{
				std::cout << "[Error] InvalidResultBoundary(" << label << "): ("
					<< a.x() << ", " << a.y() << ", " << a.z() << ") - ("
					<< b.x() << ", " << b.y() << ", " << b.z()
					<< ") is neither a seam sub-edge nor an input border sub-edge." << std::endl;
			}
		}

		if (invalid > report_limit)
		{
			std::cout << "[Error] InvalidResultBoundary(" << label << "): "
				<< (invalid - report_limit) << " more not shown." << std::endl;
		}

		return invalid;
	}

	size_t OperatorBoolean::CountBoundaryEdges(const Mesh* mesh) const
	{
		size_t count = 0;
		for (size_t i = 0; i < mesh->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = mesh->edge_handle(static_cast<int>(i));
			if (mesh->status(eh).deleted()) continue;
			if (mesh->is_boundary(eh)) ++count;
		}
		return count;
	}
}