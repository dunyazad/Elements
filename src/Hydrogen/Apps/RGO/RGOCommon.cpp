#include "RGOCommon.h"

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
	LogLevel Log::level = LogLevel::Info;
	bool Log::diagnostics = false;

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

	Eigen::Vector3f Distance::ClosestPointOnTriangle(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c)
	{
		Eigen::Vector3f ab = b - a;
		Eigen::Vector3f ac = c - a;
		Eigen::Vector3f ap = p - a;

		// Vertex region outside a
		float d1 = ab.dot(ap);
		float d2 = ac.dot(ap);
		if (d1 <= 0.0f && d2 <= 0.0f) return a;

		// Vertex region outside b
		Eigen::Vector3f bp = p - b;
		float d3 = ab.dot(bp);
		float d4 = ac.dot(bp);
		if (d3 >= 0.0f && d4 <= d3) return b;

		// Edge region ab
		float vc = d1 * d4 - d3 * d2;
		if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
		{
			float v = d1 / (d1 - d3);
			return a + v * ab;
		}

		// Vertex region outside c
		Eigen::Vector3f cp = p - c;
		float d5 = ab.dot(cp);
		float d6 = ac.dot(cp);
		if (d6 >= 0.0f && d5 <= d6) return c;

		// Edge region ac
		float vb = d5 * d2 - d1 * d6;
		if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
		{
			float w = d2 / (d2 - d6);
			return a + w * ac;
		}

		// Edge region bc
		float va = d3 * d6 - d5 * d4;
		if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
		{
			float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
			return b + w * (c - b);
		}

		// Face region: barycentric interior
		float denom = va + vb + vc;
		if (std::abs(denom) < 1e-20f) return a;
		float inv = 1.0f / denom;
		float v = vb * inv;
		float w = vc * inv;
		return a + ab * v + ac * w;
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

		// In-plane containment with an epsilon band. Each edge cross product
		// c = edge x (proj - edge_start) has magnitude (edge_length * signed
		// perpendicular distance of proj from that edge), and c.dot(n) is its
		// signed component along the face normal. Dividing the negative
		// overshoot by the edge length converts it to the actual in-plane
		// distance proj lies OUTSIDE that edge. A bare >= 0 test gives this
		// distance a zero tolerance, so a point projected exactly onto its
		// own face plane but a few 1e-7 beyond a triangle edge (curve samples
		// whose owning face the sampler picked one triangle over) is wrongly
		// classified Outside. Allowing an overshoot up to epsilon, the same
		// epsilon used for the perpendicular plane test above, makes the
		// in-plane and out-of-plane tolerances consistent. Points genuinely
		// outside (overshoot beyond epsilon) still return Outside.
		Eigen::Vector3f proj = p - dist * n;

		const Eigen::Vector3f ea[3] = { v0, v1, v2 };
		const Eigen::Vector3f eb[3] = { v1, v2, v0 };

		for (int i = 0; i < 3; ++i)
		{
			Eigen::Vector3f edge = eb[i] - ea[i];
			float edge_len = edge.norm();
			if (edge_len < 1e-12f) return result;

			float signed_area2 = edge.cross(proj - ea[i]).dot(n);
			float in_plane_dist = signed_area2 / edge_len;

			if (in_plane_dist < -epsilon)
			{
				return result;
			}
		}

		result.type = PointToTriangleType::Inside;
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

	bool Intersection::CoplanarSegmentToTriangle(
		const Eigen::Vector3f& s0,
		const Eigen::Vector3f& s1,
		const Eigen::Vector3f& v0,
		const Eigen::Vector3f& v1,
		const Eigen::Vector3f& v2,
		Eigen::Vector3f& out_p0,
		Eigen::Vector3f& out_p1,
		float epsilon)
	{
		Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
		float n_len = n.norm();
		if (n_len < 1e-12f) return false;
		n /= n_len;

		// Endpoints are assumed within epsilon of the plane: projecting
		// them makes the whole computation exactly planar.
		Eigen::Vector3f p0 = s0 - ((s0 - v0).dot(n)) * n;
		Eigen::Vector3f p1 = s1 - ((s1 - v0).dot(n)) * n;

		Eigen::Vector3f e1 = (v1 - v0).normalized();
		Eigen::Vector3f e2 = n.cross(e1);

		auto to2d = [&](const Eigen::Vector3f& p) -> Eigen::Vector2f
			{
				Eigen::Vector3f d = p - v0;
				return Eigen::Vector2f(d.dot(e1), d.dot(e2));
			};

		Eigen::Vector2f t0 = to2d(v0);
		Eigen::Vector2f t1 = to2d(v1);
		Eigen::Vector2f t2 = to2d(v2);
		Eigen::Vector2f q0 = to2d(p0);
		Eigen::Vector2f q1 = to2d(p1);

		// Triangle winding in 2D decides which side of an edge is inside
		float area2 = (t1 - t0).x() * (t2 - t0).y() - (t1 - t0).y() * (t2 - t0).x();
		if (std::abs(area2) < 1e-12f) return false;
		float orient = (area2 > 0.0f) ? 1.0f : -1.0f;

		// Clip the parameter interval of the segment against the three
		// edge half-planes (classic interval clipping, exact boundaries)
		float lo = 0.0f;
		float hi = 1.0f;
		const Eigen::Vector2f* tv[3] = { &t0, &t1, &t2 };
		for (int e = 0; e < 3; ++e)
		{
			Eigen::Vector2f ea = *tv[e];
			Eigen::Vector2f eb = *tv[(e + 1) % 3];
			Eigen::Vector2f dir = eb - ea;
			float len = dir.norm();
			if (len < 1e-12f) return false;

			auto signed_dist = [&](const Eigen::Vector2f& p) -> float
				{
					return orient * (dir.x() * (p.y() - ea.y()) - dir.y() * (p.x() - ea.x())) / len;
				};

			float d0 = signed_dist(q0);
			float d1 = signed_dist(q1);

			if (d0 < 0.0f && d1 < 0.0f) return false;
			if (d0 < 0.0f)
			{
				lo = std::max(lo, d0 / (d0 - d1));
			}
			else if (d1 < 0.0f)
			{
				hi = std::min(hi, d0 / (d0 - d1));
			}
		}

		if (hi <= lo) return false;

		out_p0 = p0 + (p1 - p0) * lo;
		out_p1 = p0 + (p1 - p0) * hi;
		return (out_p1 - out_p0).squaredNorm() >= epsilon * epsilon;
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
		size_t reported = 0;
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

				// Forensics: the three vertex positions of each rejected
				// face. A zero-length edge (two identical coordinates) means
				// a degenerate triangle reached Build; distinct coordinates
				// that still fail mean the face would make an edge incident
				// to three or more faces (non-manifold) at this location.
				if (reported < 12)
				{
					++reported;
					const Eigen::Vector3f& p0 = points[idx[0]];
					const Eigen::Vector3f& p1 = points[idx[1]];
					const Eigen::Vector3f& p2 = points[idx[2]];
					std::cout << std::setprecision(10);
					if (Log::Diag())
					{
						std::cout << "[Debug] Build rejected face: idx (" << idx[0] << ", "
							<< idx[1] << ", " << idx[2] << ") 3D ("
							<< p0.x() << ", " << p0.y() << ", " << p0.z() << ") ("
							<< p1.x() << ", " << p1.y() << ", " << p1.z() << ") ("
							<< p2.x() << ", " << p2.y() << ", " << p2.z() << ")"
							<< " edge lens " << (p1 - p0).norm() << " / "
							<< (p2 - p1).norm() << " / " << (p0 - p2).norm() << std::endl;
					}
					std::cout << std::setprecision(6);
				}
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

	// File-local: grid vertex index. Grid is (nx + 1) by (ny + 1),
	// row-major: index(i, j) = j * (nx + 1) + i.
	static int SineWaveGridIndex(int nx, int i, int j)
	{
		return j * (nx + 1) + i;
	}

	// File-local: vertex layout of the sine wave slab.
	// Top grid first [0, grid_count), bottom grid after [grid_count, 2 * grid_count),
	// where grid_count = (nx + 1) * (ny + 1). Top z follows the sine wave,
	// bottom z is flat at center.z - size.z / 2.
	static void AppendSineWaveSlabPoints(
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& size,
		float amplitude,
		float wave_count,
		int nx,
		int ny,
		std::vector<Eigen::Vector3f>& out_points)
	{
		float half_x = size.x() * 0.5f;
		float half_y = size.y() * 0.5f;
		float half_z = size.z() * 0.5f;

		int grid_count = (nx + 1) * (ny + 1);
		out_points.reserve(static_cast<size_t>(grid_count) * 2);

		// Top grid: sine displacement along x only
		for (int j = 0; j <= ny; ++j)
		{
			float v = static_cast<float>(j) / static_cast<float>(ny);
			float y = center.y() - half_y + size.y() * v;

			for (int i = 0; i <= nx; ++i)
			{
				float u = static_cast<float>(i) / static_cast<float>(nx);
				float x = center.x() - half_x + size.x() * u;
				float z = center.z() + half_z
					+ amplitude * std::sin(2.0f * static_cast<float>(M_PI) * wave_count * u);

				out_points.emplace_back(x, y, z);
			}
		}

		// Bottom grid: flat, same xy so the side walls are exactly vertical
		for (int j = 0; j <= ny; ++j)
		{
			float v = static_cast<float>(j) / static_cast<float>(ny);
			float y = center.y() - half_y + size.y() * v;

			for (int i = 0; i <= nx; ++i)
			{
				float u = static_cast<float>(i) / static_cast<float>(nx);
				float x = center.x() - half_x + size.x() * u;

				out_points.emplace_back(x, y, center.z() - half_z);
			}
		}
	}

	// File-local: top and bottom surface triangles, two per grid cell.
	// Top is CCW viewed from +z (outward +z normal), bottom mirrored
	// (outward -z normal). Index layout must match AppendSineWaveSlabPoints.
	static void AppendSineWaveSlabTopBottomIndices(
		int nx,
		int ny,
		std::vector<Eigen::Vector3i>& out_indices)
	{
		int bottom_offset = (nx + 1) * (ny + 1);

		for (int j = 0; j < ny; ++j)
		{
			for (int i = 0; i < nx; ++i)
			{
				int t00 = SineWaveGridIndex(nx, i, j);
				int t10 = SineWaveGridIndex(nx, i + 1, j);
				int t01 = SineWaveGridIndex(nx, i, j + 1);
				int t11 = SineWaveGridIndex(nx, i + 1, j + 1);

				out_indices.emplace_back(t00, t10, t11);
				out_indices.emplace_back(t00, t11, t01);

				int b00 = bottom_offset + t00;
				int b10 = bottom_offset + t10;
				int b01 = bottom_offset + t01;
				int b11 = bottom_offset + t11;

				out_indices.emplace_back(b00, b11, b10);
				out_indices.emplace_back(b00, b01, b11);
			}
		}
	}

	// File-local: four side walls stitching the top boundary to the bottom
	// boundary, two CCW triangles per column, outward normals. Quads on
	// the wavy edges are non-planar but remain valid as two triangles.
	// Index layout must match AppendSineWaveSlabPoints.
	static void AppendSineWaveSlabSideIndices(
		int nx,
		int ny,
		std::vector<Eigen::Vector3i>& out_indices)
	{
		int bottom_offset = (nx + 1) * (ny + 1);

		// y min side (normal -y) and y max side (normal +y)
		for (int i = 0; i < nx; ++i)
		{
			int t0 = SineWaveGridIndex(nx, i, 0);
			int t1 = SineWaveGridIndex(nx, i + 1, 0);
			int b0 = bottom_offset + t0;
			int b1 = bottom_offset + t1;

			out_indices.emplace_back(b0, b1, t1);
			out_indices.emplace_back(b0, t1, t0);

			int u0 = SineWaveGridIndex(nx, i, ny);
			int u1 = SineWaveGridIndex(nx, i + 1, ny);
			int c0 = bottom_offset + u0;
			int c1 = bottom_offset + u1;

			out_indices.emplace_back(c1, c0, u0);
			out_indices.emplace_back(c1, u0, u1);
		}

		// x min side (normal -x) and x max side (normal +x)
		for (int j = 0; j < ny; ++j)
		{
			int t0 = SineWaveGridIndex(nx, 0, j);
			int t1 = SineWaveGridIndex(nx, 0, j + 1);
			int b0 = bottom_offset + t0;
			int b1 = bottom_offset + t1;

			out_indices.emplace_back(b1, b0, t0);
			out_indices.emplace_back(b1, t0, t1);

			int u0 = SineWaveGridIndex(nx, nx, j);
			int u1 = SineWaveGridIndex(nx, nx, j + 1);
			int c0 = bottom_offset + u0;
			int c1 = bottom_offset + u1;

			out_indices.emplace_back(c0, c1, u1);
			out_indices.emplace_back(c0, u1, u0);
		}
	}

	bool Mesh::BuildSineWaveBox(
		const Eigen::Vector3f& center,
		const Eigen::Vector3f& size,
		float amplitude,
		float wave_count,
		int segments_x,
		int segments_y,
		const Eigen::Matrix4f& transform)
	{
		if (size.x() < EPSILON || size.y() < EPSILON || size.z() < EPSILON)
		{
			std::cout << "[Error] BuildSineWaveBox: all extents must be positive, got ("
				<< size.x() << ", " << size.y() << ", " << size.z() << ")" << std::endl;
			return false;
		}
		if (amplitude < 0.0f || amplitude >= size.z())
		{
			std::cout << "[Error] BuildSineWaveBox: amplitude must satisfy"
				<< " 0 <= amplitude < size.z, got " << amplitude
				<< " (size.z " << size.z() << "). Otherwise the top surface"
				<< " can dip below the bottom and self-intersect." << std::endl;
			return false;
		}
		if (segments_x < 1 || segments_y < 1)
		{
			std::cout << "[Error] BuildSineWaveBox: segments must be at least 1, got ("
				<< segments_x << ", " << segments_y << ")" << std::endl;
			return false;
		}

		std::vector<Eigen::Vector3f> points;
		AppendSineWaveSlabPoints(center, size, amplitude, wave_count, segments_x, segments_y, points);

		std::vector<Eigen::Vector3i> indices;
		indices.reserve(static_cast<size_t>(segments_x) * segments_y * 4
			+ static_cast<size_t>(segments_x + segments_y) * 4);
		AppendSineWaveSlabTopBottomIndices(segments_x, segments_y, indices);
		AppendSineWaveSlabSideIndices(segments_x, segments_y, indices);

		// Mirroring transforms flip winding: undo it to keep normals outward
		if (ApplyTransform(transform, points))
		{
			FlipWinding(indices);
		}

		Build(points, indices);
		return true;
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

	int Mesh::BuildSeamBoundedPatches(
		const std::vector<char>& edge_is_seam,
		std::vector<int>& out_face_patch) const
	{
		size_t num_faces = n_faces();
		out_face_patch.assign(num_faces, -1);
		int patch_count = 0;

		// Edge indices beyond the flag array are treated as not seam, so
		// callers may pass flags built before a topology change without
		// out of range access.
		auto is_seam = [&](int edge_idx) -> bool
			{
				return edge_idx >= 0
					&& edge_idx < static_cast<int>(edge_is_seam.size())
					&& 0 != edge_is_seam[edge_idx];
			};

		std::vector<int> stack;

		for (size_t start = 0; start < num_faces; ++start)
		{
			FaceHandle sfh = face_handle(static_cast<int>(start));
			if (status(sfh).deleted()) continue;
			if (out_face_patch[start] >= 0) continue;

			int patch_id = patch_count++;
			out_face_patch[start] = patch_id;
			stack.clear();
			stack.push_back(static_cast<int>(start));

			while (false == stack.empty())
			{
				int f = stack.back();
				stack.pop_back();

				FaceHandle fh = face_handle(f);
				for (auto heh : fh_range(fh))
				{
					// Seam edges are walls: flood fill must not cross them
					if (is_seam(edge_handle(heh).idx())) continue;

					HalfedgeHandle opp = opposite_halfedge_handle(heh);
					if (is_boundary(opp)) continue;

					FaceHandle nfh = face_handle(opp);
					if (false == nfh.is_valid()) continue;
					if (status(nfh).deleted()) continue;
					if (out_face_patch[nfh.idx()] >= 0) continue;

					out_face_patch[nfh.idx()] = patch_id;
					stack.push_back(nfh.idx());
				}
			}
		}

		return patch_count;
	}

	int Mesh::SplitMeshBySeam(
		const std::vector<char>& edge_is_seam,
		std::vector<std::vector<Eigen::Vector3f>>& out_patch_points,
		std::vector<std::vector<Eigen::Vector3i>>& out_patch_indices) const
	{
		out_patch_points.clear();
		out_patch_indices.clear();

		std::vector<int> face_patch;
		int patch_count = BuildSeamBoundedPatches(edge_is_seam, face_patch);
		if (patch_count <= 0) return 0;

		out_patch_points.resize(patch_count);
		out_patch_indices.resize(patch_count);

		// Per patch vertex weld maps. Seam vertices are shared by several
		// patches, so each patch carries its own copy of them.
		std::vector<robin_hood::unordered_map<Eigen::Vector3f, int, Vector3fHash, Vector3fEqual>> vertex_maps(patch_count);

		size_t num_faces = n_faces();
		for (size_t i = 0; i < num_faces; ++i)
		{
			FaceHandle fh = face_handle(static_cast<int>(i));
			if (status(fh).deleted()) continue;

			int p = face_patch[i];
			if (p < 0) continue;

			Eigen::Vector3f v0, v1, v2;
			GetFaceVertices(fh, v0, v1, v2);
			const Eigen::Vector3f tri_pts[3] = { v0, v1, v2 };

			Eigen::Vector3i fi;
			for (int j = 0; j < 3; ++j)
			{
				auto& vm = vertex_maps[p];
				auto it = vm.find(tri_pts[j]);
				if (it != vm.end())
				{
					fi[j] = it->second;
				}
				else
				{
					int ni = static_cast<int>(out_patch_points[p].size());
					out_patch_points[p].push_back(tri_pts[j]);
					vm[tri_pts[j]] = ni;
					fi[j] = ni;
				}
			}
			out_patch_indices[p].push_back(fi);
		}

		return patch_count;
	}

	std::vector<std::vector<OpenMesh::VertexHandle>> Mesh::GetBorderLoops() const
	{
		std::vector<std::vector<OpenMesh::VertexHandle>> out_loops;

		out_loops.clear();

		// A boundary halfedge is one whose face side is empty. Following
		// next_halfedge_handle from a boundary halfedge stays on the same
		// border and returns to the start after one full traversal. Each
		// halfedge is visited at most once via the visited mark, so a mesh
		// with several holes yields one loop per hole and the walk
		// terminates. The length cap is a hard stop against a corrupt
		// next-link cycle that never returns to the seed.
		std::vector<char> visited(n_halfedges(), 0);
		const size_t max_chain = n_halfedges() + 1;

		for (size_t i = 0; i < n_halfedges(); ++i)
		{
			OpenMesh::HalfedgeHandle start = halfedge_handle(static_cast<int>(i));
			if (status(start).deleted()) continue;
			if (false == is_boundary(start)) continue;
			if (0 != visited[start.idx()]) continue;

			std::vector<OpenMesh::VertexHandle> loop;
			OpenMesh::HalfedgeHandle heh = start;
			size_t guard = 0;

			do
			{
				visited[heh.idx()] = 1;
				loop.push_back(to_vertex_handle(heh));

				heh = next_halfedge_handle(heh);
				++guard;

				if (false == heh.is_valid()) break;
				if (guard > max_chain) break;

			} while (heh != start);

			if (loop.size() >= 3)
			{
				out_loops.push_back(std::move(loop));
			}
		}

		return out_loops;
	}

	Eigen::AlignedBox3f Mesh::GetBoundingBox() const
	{
		Eigen::AlignedBox3f box;
		box.setEmpty();

		for (size_t i = 0; i < n_faces(); ++i)
		{
			OpenMesh::FaceHandle fh = face_handle(static_cast<int>(i));
			if (status(fh).deleted()) continue;

			Eigen::Vector3f v0, v1, v2;
			GetFaceVertices(fh, v0, v1, v2);

			box.extend(v0);
			box.extend(v1);
			box.extend(v2);
		}

		return box;
	}

	bool Mesh::GeodesicShortestPath(
		OpenMesh::VertexHandle start,
		OpenMesh::VertexHandle end,
		std::vector<OpenMesh::VertexHandle>& out_path,
		float& out_distance) const
	{
		out_path.clear();
		out_distance = 0.0f;

		std::vector<float> dist;
		std::vector<int> prev;
		if (false == RunDijkstraVertexGraph(start, end, dist, prev))
		{
			std::cout << "[Warning] GeodesicShortestPath: no path between the given vertices." << std::endl;
			return false;
		}

		const int start_idx = start.idx();
		const int end_idx = end.idx();
		out_distance = dist[end_idx];

		std::vector<OpenMesh::VertexHandle> reversed;
		int cur = end_idx;
		size_t guard = 0;
		const size_t max_steps = n_vertices() + 1;

		while (cur != -1)
		{
			reversed.push_back(vertex_handle(cur));
			if (cur == start_idx) break;

			cur = prev[cur];
			if (++guard > max_steps)
			{
				std::cout << "[Error] GeodesicShortestPath: predecessor chain did not terminate." << std::endl;
				out_path.clear();
				out_distance = 0.0f;
				return false;
			}
		}

		if (reversed.empty() || reversed.back() != start)
		{
			std::cout << "[Error] GeodesicShortestPath: path reconstruction failed." << std::endl;
			out_path.clear();
			out_distance = 0.0f;
			return false;
		}

		out_path.assign(reversed.rbegin(), reversed.rend());
		return true;
	}

	bool Mesh::RunDijkstraVertexGraph(
		OpenMesh::VertexHandle start,
		OpenMesh::VertexHandle end,
		std::vector<float>& out_dist,
		std::vector<int>& out_prev) const
	{
		const size_t num_vertices = n_vertices();
		out_dist.assign(num_vertices, std::numeric_limits<float>::max());
		out_prev.assign(num_vertices, -1);

		if (false == start.is_valid() || false == end.is_valid()) return false;
		if (status(start).deleted() || status(end).deleted()) return false;

		const int start_idx = start.idx();
		const int end_idx = end.idx();
		out_dist[start_idx] = 0.0f;

		// Min-heap keyed by tentative distance. A vertex may be pushed
		// several times; a stale entry is discarded on pop because its
		// stored distance no longer matches out_dist.
		using QueueNode = std::pair<float, int>;
		std::priority_queue<QueueNode, std::vector<QueueNode>, std::greater<QueueNode>> queue;
		queue.push({ 0.0f, start_idx });

		std::vector<char> finalized(num_vertices, 0);

		while (false == queue.empty())
		{
			QueueNode node = queue.top();
			queue.pop();

			float d = node.first;
			int u = node.second;

			if (0 != finalized[u]) continue;
			if (d > out_dist[u]) continue;
			finalized[u] = 1;

			if (u == end_idx) return true;

			OpenMesh::VertexHandle uh = vertex_handle(u);
			Eigen::Vector3f pu(point(uh).data());

			for (auto voh_it = cvoh_iter(uh); voh_it.is_valid(); ++voh_it)
			{
				OpenMesh::HalfedgeHandle heh = *voh_it;
				if (status(edge_handle(heh)).deleted()) continue;

				OpenMesh::VertexHandle vh = to_vertex_handle(heh);
				if (false == vh.is_valid()) continue;
				if (status(vh).deleted()) continue;

				int v = vh.idx();
				if (0 != finalized[v]) continue;

				Eigen::Vector3f pv(point(vh).data());
				float w = (pv - pu).norm();
				float nd = out_dist[u] + w;

				if (nd < out_dist[v])
				{
					out_dist[v] = nd;
					out_prev[v] = u;
					queue.push({ nd, v });
				}
			}
		}

		return out_dist[end_idx] < std::numeric_limits<float>::max();
	}

	bool Mesh::ClosestPointOnSurface(
		const Eigen::Vector3f& query,
		Eigen::Vector3f& out_point,
		OpenMesh::FaceHandle& out_face) const
	{
		out_face = OpenMesh::FaceHandle();
		out_point = query;

		if (0 == n_faces()) return false;
		if (hash_map.empty()) return false;

		// Base radius from the grid cell. Fall back to a fraction of the
		// scene diagonal when the cell size is not set.
		float base = grid_cell_size.maxCoeff();
		if (base <= 0.0f) base = (grid_max - grid_min).norm() * 0.01f + EPSILON;

		const float diag = (grid_max - grid_min).norm() + base;
		float radius = base;

		float best_d2 = std::numeric_limits<float>::max();
		Eigen::Vector3f best_p = query;
		OpenMesh::FaceHandle best_f;

		std::vector<int> candidates;

		for (int iter = 0; iter < 32; ++iter)
		{
			Eigen::Vector3f r3(radius, radius, radius);
			QueryOverlappingFaces(query - r3, query + r3, candidates);

			best_d2 = std::numeric_limits<float>::max();
			best_f = OpenMesh::FaceHandle();

			for (int fi : candidates)
			{
				if (fi < 0 || fi >= static_cast<int>(n_faces())) continue;

				OpenMesh::FaceHandle fh = face_handle(fi);
				if (false == fh.is_valid()) continue;
				if (status(fh).deleted()) continue;

				Eigen::Vector3f v0, v1, v2;
				GetFaceVertices(fh, v0, v1, v2);

				Eigen::Vector3f cp = Distance::ClosestPointOnTriangle(query, v0, v1, v2);
				float d2 = (cp - query).squaredNorm();
				if (d2 < best_d2)
				{
					best_d2 = d2;
					best_p = cp;
					best_f = fh;
				}
			}

			// Once the best hit is within the queried box, no closer face
			// can lie outside it.
			if (best_f.is_valid() && std::sqrt(best_d2) <= radius) break;

			if (radius >= diag) break;
			radius *= 2.0f;
		}

		if (false == best_f.is_valid()) return false;

		out_point = best_p;
		out_face = best_f;
		return true;
	}

	// File-local: evaluates one centripetal Catmull-Rom span. The active
	// span is p1 -> p2; p0 and p3 are the neighbors that set the tangents.
	// Centripetal (alpha = 0.5) knot spacing prevents the cusps and
	// self-loops that uniform Catmull-Rom makes on unevenly spaced points.
	static Eigen::Vector3f CatmullRomPoint(
		const Eigen::Vector3f& p0,
		const Eigen::Vector3f& p1,
		const Eigen::Vector3f& p2,
		const Eigen::Vector3f& p3,
		float t)
	{
		auto next_knot = [](float ti, const Eigen::Vector3f& a, const Eigen::Vector3f& b) -> float
			{
				float d = std::sqrt((b - a).norm());   // chord length to the power alpha = 0.5
				if (d < 1e-6f) d = 1e-6f;               // floor; dedupe should make this unreachable
				return ti + d;
			};

		float t0 = 0.0f;
		float t1 = next_knot(t0, p0, p1);
		float t2 = next_knot(t1, p1, p2);
		float t3 = next_knot(t2, p2, p3);

		float tt = t1 + (t2 - t1) * t;

		Eigen::Vector3f a1 = ((t1 - tt) / (t1 - t0)) * p0 + ((tt - t0) / (t1 - t0)) * p1;
		Eigen::Vector3f a2 = ((t2 - tt) / (t2 - t1)) * p1 + ((tt - t1) / (t2 - t1)) * p2;
		Eigen::Vector3f a3 = ((t3 - tt) / (t3 - t2)) * p2 + ((tt - t2) / (t3 - t2)) * p3;

		Eigen::Vector3f b1 = ((t2 - tt) / (t2 - t0)) * a1 + ((tt - t0) / (t2 - t0)) * a2;
		Eigen::Vector3f b2 = ((t3 - tt) / (t3 - t1)) * a2 + ((tt - t1) / (t3 - t1)) * a3;

		return ((t2 - tt) / (t2 - t1)) * b1 + ((tt - t1) / (t2 - t1)) * b2;
	}

	// File-local: drops points coincident with the previous one within
	// EPSILON. Coincident control points give a zero knot interval, which
	// is degenerate input for the spline, so they are removed at the source.
	// For a closed loop a trailing point coincident with the first is also
	// dropped so the seam is not duplicated.
	static void DedupeConsecutivePoints(
		const std::vector<Eigen::Vector3f>& in,
		bool closed,
		std::vector<Eigen::Vector3f>& out)
	{
		out.clear();
		out.reserve(in.size());

		for (const auto& p : in)
		{
			if (false == out.empty() && (p - out.back()).squaredNorm() < EPSILON * EPSILON) continue;
			out.push_back(p);
		}

		if (closed && out.size() >= 2 && (out.front() - out.back()).squaredNorm() < EPSILON * EPSILON)
		{
			out.pop_back();
		}
	}

	// File-local: samples a Catmull-Rom chain into a dense polyline.
	// Closed chains wrap their neighbor indices. Open chains use reflected
	// phantom endpoints (2*P0 - P1 and 2*Pn-1 - Pn-2) so the end tangents
	// are well defined and no knot interval collapses. Each span emits
	// samples over [0, 1); the span end is the next span's start, so no
	// vertex is duplicated. The final open-chain endpoint is appended exactly.
	static void SampleCatmullRomChain(
		const std::vector<Eigen::Vector3f>& ctrl,
		bool closed,
		int samples_per_segment,
		std::vector<Eigen::Vector3f>& out)
	{
		out.clear();

		const int n = static_cast<int>(ctrl.size());
		if (n < 2) { out = ctrl; return; }
		if (samples_per_segment < 1) samples_per_segment = 1;

		auto P = [&](int i) -> Eigen::Vector3f
			{
				if (closed)
				{
					int m = ((i % n) + n) % n;
					return ctrl[m];
				}
				if (i < 0) return 2.0f * ctrl[0] - ctrl[1];
				if (i >= n) return 2.0f * ctrl[n - 1] - ctrl[n - 2];
				return ctrl[i];
			};

		const int span_count = closed ? n : (n - 1);
		out.reserve(static_cast<size_t>(span_count) * samples_per_segment + 1);

		for (int s = 0; s < span_count; ++s)
		{
			Eigen::Vector3f p0 = P(s - 1);
			Eigen::Vector3f p1 = P(s);
			Eigen::Vector3f p2 = P(s + 1);
			Eigen::Vector3f p3 = P(s + 2);

			for (int k = 0; k < samples_per_segment; ++k)
			{
				float t = static_cast<float>(k) / static_cast<float>(samples_per_segment);
				out.push_back(CatmullRomPoint(p0, p1, p2, p3, t));
			}
		}

		// Open chain: emit the last control point. Closed chain is cyclic,
		// so the seam is left implicit and not duplicated.
		if (false == closed)
		{
			out.push_back(ctrl[n - 1]);
		}
	}

	bool Mesh::BuildSmoothSurfaceCurve(
		const std::vector<Eigen::Vector3f>& picked_points,
		bool closed,
		int samples_per_segment,
		std::vector<Eigen::Vector3f>& out_curve,
		std::vector<OpenMesh::FaceHandle>& out_faces) const
	{
		out_curve.clear();
		out_faces.clear();

		if (picked_points.size() < 2)
		{
			std::cout << "[Warning] BuildSmoothSurfaceCurve: need at least 2 picked points." << std::endl;
			return false;
		}
		if (0 == n_faces() || hash_map.empty())
		{
			std::cout << "[Error] BuildSmoothSurfaceCurve: spatial hash is empty;"
				<< " call BuildSpatialHashMap first." << std::endl;
			return false;
		}

		std::vector<Eigen::Vector3f> ctrl;
		DedupeConsecutivePoints(picked_points, closed, ctrl);

		if (ctrl.size() < 2)
		{
			std::cout << "[Warning] BuildSmoothSurfaceCurve: fewer than 2 distinct points after dedupe." << std::endl;
			return false;
		}
		if (closed && ctrl.size() < 3)
		{
			std::cout << "[Warning] BuildSmoothSurfaceCurve: a closed loop needs at least 3 distinct points." << std::endl;
			return false;
		}

		std::vector<Eigen::Vector3f> spline;
		SampleCatmullRomChain(ctrl, closed, samples_per_segment, spline);

		out_curve.reserve(spline.size());
		out_faces.reserve(spline.size());

		for (const auto& s : spline)
		{
			Eigen::Vector3f surf;
			OpenMesh::FaceHandle fh;
			if (ClosestPointOnSurface(s, surf, fh))
			{
				out_curve.push_back(surf);
				out_faces.push_back(fh);
			}
			else
			{
				// Projection failed only when the mesh is empty, already
				// guarded above; keep the raw sample so indices stay aligned.
				out_curve.push_back(s);
				out_faces.push_back(OpenMesh::FaceHandle());
			}
		}

		return false == out_curve.empty();
	}

	OpenMesh::VertexHandle Mesh::ClosestVertex(const Eigen::Vector3f& query) const
	{
		Eigen::Vector3f surf;
		OpenMesh::FaceHandle fh;
		if (false == ClosestPointOnSurface(query, surf, fh)) return OpenMesh::VertexHandle();
		if (false == fh.is_valid()) return OpenMesh::VertexHandle();

		OpenMesh::VertexHandle best;
		float best_d2 = std::numeric_limits<float>::max();

		for (auto fv_it = cfv_iter(fh); fv_it.is_valid(); ++fv_it)
		{
			Eigen::Vector3f p(point(*fv_it).data());
			float d2 = (p - query).squaredNorm();
			if (d2 < best_d2)
			{
				best_d2 = d2;
				best = *fv_it;
			}
		}

		return best;
	}

	bool Mesh::BuildGeodesicAnchorPolyline(
		const std::vector<OpenMesh::VertexHandle>& anchors,
		bool closed,
		std::vector<Eigen::Vector3f>& out_points,
		std::vector<char>& out_is_anchor) const
	{
		out_points.clear();
		out_is_anchor.clear();

		const size_t n = anchors.size();
		if (n < 2) return false;

		const size_t seg_count = closed ? n : (n - 1);

		auto append_vertex = [&](OpenMesh::VertexHandle vh, bool anchor)
			{
				Eigen::Vector3f p(point(vh).data());
				out_points.push_back(p);
				out_is_anchor.push_back(static_cast<char>(anchor ? 1 : 0));
			};

		append_vertex(anchors[0], true);

		for (size_t s = 0; s < seg_count; ++s)
		{
			OpenMesh::VertexHandle a = anchors[s];
			OpenMesh::VertexHandle b = anchors[(s + 1) % n];

			const bool closing = closed && (s + 1 == seg_count);

			std::vector<OpenMesh::VertexHandle> path;
			float dist = 0.0f;
			if (false == GeodesicShortestPath(a, b, path, dist) || path.size() < 2)
			{
				std::cout << "[Warning] BuildGeodesicAnchorPolyline: no geodesic between"
					<< " anchors " << a.idx() << " and " << b.idx()
					<< "; bridging directly." << std::endl;

				if (false == closing)
				{
					append_vertex(b, true);
				}
				continue;
			}

			const size_t last = closing ? (path.size() - 1) : path.size();
			for (size_t k = 1; k < last; ++k)
			{
				const bool is_dest_anchor = (false == closing) && (k == path.size() - 1);
				append_vertex(path[k], is_dest_anchor);
			}
		}

		return out_points.size() >= 2;
	}

	void Mesh::SmoothCurveOnSurface(
		std::vector<Eigen::Vector3f>& curve,
		const std::vector<char>& is_anchor,
		bool closed,
		int iterations,
		float strength) const
	{
		const size_t n = curve.size();
		if (n < 3) return;
		if (iterations < 1) return;
		if (strength <= 0.0f) return;
		if (strength > 1.0f) strength = 1.0f;

		std::vector<Eigen::Vector3f> next(curve);

		for (int it = 0; it < iterations; ++it)
		{
			for (size_t i = 0; i < n; ++i)
			{
				if (i < is_anchor.size() && 0 != is_anchor[i])
				{
					next[i] = curve[i];
					continue;
				}

				if (false == closed && (0 == i || n - 1 == i))
				{
					next[i] = curve[i];
					continue;
				}

				const Eigen::Vector3f& prev = closed ? curve[(i + n - 1) % n] : curve[i - 1];
				const Eigen::Vector3f& nxt = closed ? curve[(i + 1) % n] : curve[i + 1];

				Eigen::Vector3f target = 0.5f * (prev + nxt);
				Eigen::Vector3f moved = (1.0f - strength) * curve[i] + strength * target;

				Eigen::Vector3f surf;
				OpenMesh::FaceHandle fh;
				if (ClosestPointOnSurface(moved, surf, fh)) next[i] = surf;
				else next[i] = moved;
			}

			curve.swap(next);
		}
	}

	bool Mesh::BuildGeodesicSurfaceCurve(
		const std::vector<Eigen::Vector3f>& picked_points,
		bool closed,
		int smoothing_iterations,
		float smoothing_strength,
		std::vector<Eigen::Vector3f>& out_curve,
		std::vector<OpenMesh::FaceHandle>& out_faces) const
	{
		out_curve.clear();
		out_faces.clear();

		if (picked_points.size() < 2)
		{
			std::cout << "[Warning] BuildGeodesicSurfaceCurve: need at least 2 picked points." << std::endl;
			return false;
		}
		if (0 == n_faces() || hash_map.empty())
		{
			std::cout << "[Error] BuildGeodesicSurfaceCurve: spatial hash is empty;"
				<< " call BuildSpatialHashMap first." << std::endl;
			return false;
		}

		std::vector<Eigen::Vector3f> deduped;
		DedupeConsecutivePoints(picked_points, closed, deduped);
		if (deduped.size() < 2)
		{
			std::cout << "[Warning] BuildGeodesicSurfaceCurve: fewer than 2 distinct points after dedupe." << std::endl;
			return false;
		}
		if (closed && deduped.size() < 3)
		{
			std::cout << "[Warning] BuildGeodesicSurfaceCurve: a closed loop needs at least 3 distinct points." << std::endl;
			return false;
		}

		std::vector<OpenMesh::VertexHandle> anchors;
		anchors.reserve(deduped.size());
		for (const auto& p : deduped)
		{
			OpenMesh::VertexHandle vh = ClosestVertex(p);
			if (false == vh.is_valid())
			{
				std::cout << "[Error] BuildGeodesicSurfaceCurve: a picked point has no nearby vertex." << std::endl;
				return false;
			}
			if (false == anchors.empty() && anchors.back() == vh) continue;
			anchors.push_back(vh);
		}

		if (closed && anchors.size() >= 2 && anchors.front() == anchors.back())
		{
			anchors.pop_back();
		}

		if (anchors.size() < 2)
		{
			std::cout << "[Warning] BuildGeodesicSurfaceCurve: fewer than 2 distinct anchor vertices." << std::endl;
			return false;
		}
		if (closed && anchors.size() < 3)
		{
			std::cout << "[Warning] BuildGeodesicSurfaceCurve: a closed loop needs at least 3 distinct anchor vertices." << std::endl;
			return false;
		}

		std::vector<Eigen::Vector3f> polyline;
		std::vector<char> is_anchor;
		if (false == BuildGeodesicAnchorPolyline(anchors, closed, polyline, is_anchor))
		{
			std::cout << "[Error] BuildGeodesicSurfaceCurve: failed to build geodesic polyline." << std::endl;
			return false;
		}

		SmoothCurveOnSurface(polyline, is_anchor, closed, smoothing_iterations, smoothing_strength);

		out_curve.reserve(polyline.size());
		out_faces.reserve(polyline.size());

		for (const auto& p : polyline)
		{
			Eigen::Vector3f surf;
			OpenMesh::FaceHandle fh;
			if (ClosestPointOnSurface(p, surf, fh))
			{
				out_curve.push_back(surf);
				out_faces.push_back(fh);
			}
			else
			{
				out_curve.push_back(p);
				out_faces.push_back(OpenMesh::FaceHandle());
			}
		}

		return out_curve.size() >= 2;
	}

	bool Mesh::BuildCutSeamVertexLoop(
		const std::vector<Eigen::Vector3f>& picked_points,
		std::vector<OpenMesh::VertexHandle>& out_loop) const
	{
		out_loop.clear();

		if (0 == n_faces() || hash_map.empty())
		{
			std::cout << "[Error] BuildCutSeamVertexLoop: spatial hash is empty;"
				<< " call BuildSpatialHashMap first." << std::endl;
			return false;
		}

		std::vector<Eigen::Vector3f> deduped;
		DedupeConsecutivePoints(picked_points, true, deduped);
		if (deduped.size() < 3)
		{
			std::cout << "[Warning] BuildCutSeamVertexLoop: a closed cut needs at least 3 distinct points." << std::endl;
			return false;
		}

		std::vector<OpenMesh::VertexHandle> anchors;
		anchors.reserve(deduped.size());
		for (const auto& p : deduped)
		{
			OpenMesh::VertexHandle vh = ClosestVertex(p);
			if (false == vh.is_valid())
			{
				std::cout << "[Error] BuildCutSeamVertexLoop: a picked point has no nearby vertex." << std::endl;
				return false;
			}
			if (false == anchors.empty() && anchors.back() == vh) continue;
			anchors.push_back(vh);
		}

		if (anchors.size() >= 2 && anchors.front() == anchors.back())
		{
			anchors.pop_back();
		}
		if (anchors.size() < 3)
		{
			std::cout << "[Warning] BuildCutSeamVertexLoop: fewer than 3 distinct anchor vertices." << std::endl;
			return false;
		}

		std::vector<OpenMesh::VertexHandle> loop;
		const size_t n = anchors.size();

		loop.push_back(anchors[0]);

		for (size_t s = 0; s < n; ++s)
		{
			OpenMesh::VertexHandle a = anchors[s];
			OpenMesh::VertexHandle b = anchors[(s + 1) % n];
			const bool closing = (s + 1 == n);

			std::vector<OpenMesh::VertexHandle> path;
			float dist = 0.0f;
			if (false == GeodesicShortestPath(a, b, path, dist) || path.size() < 2)
			{
				std::cout << "[Error] BuildCutSeamVertexLoop: no geodesic between anchors "
					<< a.idx() << " and " << b.idx() << "; cannot form a safe seam." << std::endl;
				out_loop.clear();
				return false;
			}

			const size_t last = closing ? (path.size() - 1) : path.size();
			for (size_t k = 1; k < last; ++k)
			{
				loop.push_back(path[k]);
			}
		}

		if (loop.size() < 3)
		{
			std::cout << "[Error] BuildCutSeamVertexLoop: seam loop too short." << std::endl;
			return false;
		}

		// Verify the loop is simple: no vertex visited twice.
		{
			robin_hood::unordered_set<int> seen;
			seen.reserve(loop.size() * 2);
			for (const auto& vh : loop)
			{
				if (false == seen.insert(vh.idx()).second)
				{
					std::cout << "[Error] BuildCutSeamVertexLoop: seam revisits vertex "
						<< vh.idx() << " (self-intersecting loop). Adjust the picked"
						<< " points so the path does not cross itself." << std::endl;
					out_loop.clear();
					return false;
				}
			}
		}

		// Verify every consecutive pair (cyclic) is a real mesh edge.
		for (size_t i = 0; i < loop.size(); ++i)
		{
			OpenMesh::VertexHandle a = loop[i];
			OpenMesh::VertexHandle b = loop[(i + 1) % loop.size()];

			bool connected = false;
			for (auto voh_it = cvoh_iter(a); voh_it.is_valid(); ++voh_it)
			{
				if (to_vertex_handle(*voh_it) == b) { connected = true; break; }
			}
			if (false == connected)
			{
				std::cout << "[Error] BuildCutSeamVertexLoop: loop vertices "
					<< a.idx() << " and " << b.idx() << " are not edge-connected."
					<< " Cannot form a seam without inserting vertices." << std::endl;
				out_loop.clear();
				return false;
			}
		}

		out_loop = std::move(loop);
		return true;
	}

	bool Mesh::BuildSeamFromVertexLoop(
		const std::vector<OpenMesh::VertexHandle>& loop,
		std::vector<char>& out_edge_is_seam) const
	{
		out_edge_is_seam.assign(n_edges(), 0);

		if (loop.size() < 3)
		{
			std::cout << "[Error] BuildSeamFromVertexLoop: loop too short." << std::endl;
			return false;
		}

		for (size_t i = 0; i < loop.size(); ++i)
		{
			OpenMesh::VertexHandle a = loop[i];
			OpenMesh::VertexHandle b = loop[(i + 1) % loop.size()];

			OpenMesh::EdgeHandle eh;
			for (auto voh_it = cvoh_iter(a); voh_it.is_valid(); ++voh_it)
			{
				if (to_vertex_handle(*voh_it) == b)
				{
					eh = edge_handle(*voh_it);
					break;
				}
			}

			if (false == eh.is_valid())
			{
				std::cout << "[Error] BuildSeamFromVertexLoop: no edge between vertices "
					<< a.idx() << " and " << b.idx() << "." << std::endl;
				out_edge_is_seam.assign(n_edges(), 0);
				return false;
			}

			out_edge_is_seam[eh.idx()] = 1;
		}

		return true;
	}

	bool Mesh::SplitByMarginLoop(
		const std::vector<Eigen::Vector3f>& picked_points,
		std::vector<Eigen::Vector3f>& out_small_points,
		std::vector<Eigen::Vector3i>& out_small_indices,
		std::vector<Eigen::Vector3f>& out_large_points,
		std::vector<Eigen::Vector3i>& out_large_indices) const
	{
		out_small_points.clear();
		out_small_indices.clear();
		out_large_points.clear();
		out_large_indices.clear();

		std::vector<OpenMesh::VertexHandle> loop;
		if (false == BuildCutSeamVertexLoop(picked_points, loop)) return false;

		std::vector<char> edge_is_seam;
		if (false == BuildSeamFromVertexLoop(loop, edge_is_seam)) return false;

		std::vector<std::vector<Eigen::Vector3f>> patch_points;
		std::vector<std::vector<Eigen::Vector3i>> patch_indices;
		int patch_count = SplitMeshBySeam(edge_is_seam, patch_points, patch_indices);

		if (2 != patch_count)
		{
			std::cout << "[Error] SplitByMarginLoop: seam produced " << patch_count
				<< " patches, expected exactly 2. The margin loop does not cleanly"
				<< " separate the mesh; adjust the picked points." << std::endl;
			return false;
		}

		auto patch_area = [](const std::vector<Eigen::Vector3f>& pts,
			const std::vector<Eigen::Vector3i>& idx) -> float
			{
				float area = 0.0f;
				for (const auto& t : idx)
				{
					const Eigen::Vector3f& a = pts[t[0]];
					const Eigen::Vector3f& b = pts[t[1]];
					const Eigen::Vector3f& c = pts[t[2]];
					area += 0.5f * (b - a).cross(c - a).norm();
				}
				return area;
			};

		float area0 = patch_area(patch_points[0], patch_indices[0]);
		float area1 = patch_area(patch_points[1], patch_indices[1]);

		int small_idx = (area0 <= area1) ? 0 : 1;
		int large_idx = 1 - small_idx;

		out_small_points = patch_points[small_idx];
		out_small_indices = patch_indices[small_idx];
		out_large_points = patch_points[large_idx];
		out_large_indices = patch_indices[large_idx];

		if (Log::AtInfo())
		{
			std::cout << "[Info] SplitByMarginLoop: split into areas "
				<< patch_area(out_small_points, out_small_indices) << " (small) and "
				<< patch_area(out_large_points, out_large_indices) << " (large)." << std::endl;
		}

		return true;
	}

	// ------------------------------------------------------------
	// Operators
	// ------------------------------------------------------------

	void ExtractMeshSoup(
		const Mesh* mesh,
		std::vector<Eigen::Vector3f>& out_points,
		std::vector<Eigen::Vector3i>& out_indices)
	{
		out_points.clear();
		out_indices.clear();
		out_points.reserve(mesh->n_vertices());

		for (size_t i = 0; i < mesh->n_vertices(); ++i)
		{
			auto p = mesh->point(mesh->vertex_handle(static_cast<int>(i)));
			out_points.push_back(Eigen::Vector3f(p[0], p[1], p[2]));
		}

		out_indices.reserve(mesh->n_faces());
		for (size_t i = 0; i < mesh->n_faces(); ++i)
		{
			OpenMesh::FaceHandle fh = mesh->face_handle(static_cast<int>(i));
			if (mesh->status(fh).deleted()) continue;

			Eigen::Vector3i tri;
			int k = 0;
			for (auto fv_it = mesh->cfv_iter(fh); fv_it.is_valid() && k < 3; ++fv_it, ++k)
			{
				tri[k] = fv_it->idx();
			}
			if (3 == k) out_indices.push_back(tri);
		}
	}

	bool ValidateTriangleSoup(
		const std::vector<Eigen::Vector3f>& points,
		const std::vector<Eigen::Vector3i>& indices,
		const char* label,
		float near_pair_radius)
	{
		size_t failures = 0;

		struct EdgeKey
		{
			int a;
			int b;
		};
		struct EdgeKeyHash
		{
			size_t operator()(const EdgeKey& k) const
			{
				return static_cast<size_t>(k.a) * 1000003ull ^ static_cast<size_t>(k.b);
			}
		};
		struct EdgeKeyEqual
		{
			bool operator()(const EdgeKey& x, const EdgeKey& y) const
			{
				return x.a == y.a && x.b == y.b;
			}
		};

		// Check 1: degenerate and duplicate triangles. The same vertex
		// triple appearing twice means a region of the surface exists
		// twice: every edge of that region becomes 4-valent.
		{
			struct TriKey
			{
				int a;
				int b;
				int c;
			};
			struct TriKeyHash
			{
				size_t operator()(const TriKey& k) const
				{
					size_t h = static_cast<size_t>(k.a);
					h = h * 1000003ull ^ static_cast<size_t>(k.b);
					h = h * 1000003ull ^ static_cast<size_t>(k.c);
					return h;
				}
			};
			struct TriKeyEqual
			{
				bool operator()(const TriKey& x, const TriKey& y) const
				{
					return x.a == y.a && x.b == y.b && x.c == y.c;
				}
			};

			robin_hood::unordered_set<TriKey, TriKeyHash, TriKeyEqual> seen;
			seen.reserve(indices.size());

			size_t degenerate = 0;
			size_t duplicates = 0;

			for (size_t i = 0; i < indices.size(); ++i)
			{
				int a = indices[i][0];
				int b = indices[i][1];
				int cc = indices[i][2];

				if (a == b || b == cc || cc == a)
				{
					++degenerate;
					continue;
				}

				TriKey key;
				key.a = std::min(a, std::min(b, cc));
				key.c = std::max(a, std::max(b, cc));
				key.b = a + b + cc - key.a - key.c;

				if (false == seen.insert(key).second)
				{
					++duplicates;
					if (duplicates <= 10)
					{
						const Eigen::Vector3f& p = points[key.a];
						std::cout << "[Error] SoupTopology(" << label << "): duplicate triangle near ("
							<< p.x() << ", " << p.y() << ", " << p.z() << ")." << std::endl;
					}
				}
			}

			if (degenerate > 0)
			{
				++failures;
				std::cout << "[Error] SoupTopology(" << label << "): " << degenerate
					<< " degenerate triangles." << std::endl;
			}
			if (duplicates > 0)
			{
				++failures;
				std::cout << "[Error] SoupTopology(" << label << "): " << duplicates
					<< " duplicate triangles." << std::endl;
			}
		}

		// Shared edge map for checks 2 and 3
		robin_hood::unordered_map<EdgeKey, int, EdgeKeyHash, EdgeKeyEqual> edge_faces;
		edge_faces.reserve(indices.size() * 3);

		for (const auto& tri : indices)
		{
			for (int e = 0; e < 3; ++e)
			{
				int a = tri[e];
				int b = tri[(e + 1) % 3];
				EdgeKey key;
				key.a = std::min(a, b);
				key.b = std::max(a, b);
				edge_faces[key] += 1;
			}
		}

		// Check 2: edge manifoldness and edge length statistics. The
		// length tiers show which edges an external tool would COLLAPSE
		// when welding at a coarser tolerance, turning their triangles
		// degenerate and the surroundings non-manifold over there.
		{
			size_t holes = 0;
			size_t complex_edges = 0;
			size_t reported = 0;

			float min_len = std::numeric_limits<float>::max();
			size_t len_t1 = 0;
			size_t len_t2 = 0;
			size_t len_t3 = 0;

			for (const auto& kvp : edge_faces)
			{
				float len = (points[kvp.first.a] - points[kvp.first.b]).norm();
				if (len < min_len) min_len = len;
				if (len < 2.0f * EPSILON) ++len_t1;
				else if (len < 10.0f * EPSILON) ++len_t2;
				else if (len < 100.0f * EPSILON) ++len_t3;

				if (2 == kvp.second) continue;

				if (kvp.second < 2) ++holes;
				else ++complex_edges;

				++reported;
				if (reported <= 10)
				{
					const Eigen::Vector3f& p0 = points[kvp.first.a];
					const Eigen::Vector3f& p1 = points[kvp.first.b];
					std::cout << "[Error] SoupTopology(" << label << "): edge ("
						<< p0.x() << ", " << p0.y() << ", " << p0.z() << ") - ("
						<< p1.x() << ", " << p1.y() << ", " << p1.z() << ") has "
						<< kvp.second << " incident faces." << std::endl;
				}
			}

			if (Log::AtInfo())
			{
				std::cout << "[Info] SoupTopology(" << label << "): min edge length " << min_len
					<< ", edges shorter than 2e/10e/100e EPSILON: "
					<< len_t1 << " / " << len_t2 << " / " << len_t3 << "." << std::endl;
			}

			if (holes > 0)
			{
				++failures;
				std::cout << "[Error] SoupTopology(" << label << "): " << holes
					<< " edges with fewer than 2 faces (holes)." << std::endl;
			}
			if (complex_edges > 0)
			{
				++failures;
				std::cout << "[Error] SoupTopology(" << label << "): " << complex_edges
					<< " edges with more than 2 faces (non-manifold)." << std::endl;
			}
		}

		// Check 3: near-coincident vertex pairs up to near_pair_radius,
		// excluding pairs already connected by an edge (those are just
		// short edges, counted above). Unconnected near pairs are
		// separate sheets passing close: exactly what a distance-welding
		// importer fuses into non-manifold features.
		{
			robin_hood::unordered_map<Eigen::Vector3i, std::vector<int>, Int3Hash, Int3Equal> grid;
			grid.reserve(points.size());

			auto quantize = [&](const Eigen::Vector3f& p) -> Eigen::Vector3i
				{
					return Eigen::Vector3i(
						static_cast<int>(std::floor(p.x() / near_pair_radius)),
						static_cast<int>(std::floor(p.y() / near_pair_radius)),
						static_cast<int>(std::floor(p.z() / near_pair_radius)));
				};

			size_t pair_t1 = 0;
			size_t pair_t2 = 0;
			size_t pair_t3 = 0;
			size_t shown = 0;

			for (int i = 0; i < static_cast<int>(points.size()); ++i)
			{
				Eigen::Vector3i cell = quantize(points[i]);
				for (int dz = -1; dz <= 1; ++dz)
				{
					for (int dy = -1; dy <= 1; ++dy)
					{
						for (int dx = -1; dx <= 1; ++dx)
						{
							auto it = grid.find(Eigen::Vector3i(cell.x() + dx, cell.y() + dy, cell.z() + dz));
							if (it == grid.end()) continue;

							for (int j : it->second)
							{
								float dist = (points[j] - points[i]).norm();
								if (dist >= near_pair_radius) continue;

								EdgeKey key;
								key.a = std::min(i, j);
								key.b = std::max(i, j);
								if (edge_faces.find(key) != edge_faces.end()) continue;

								if (dist < EPSILON) ++pair_t1;
								else if (dist < 10.0f * EPSILON) ++pair_t2;
								else ++pair_t3;

								++shown;
								if (shown <= 10)
								{
									std::cout << "[Warning] SoupTopology(" << label
										<< "): unconnected vertices " << dist << " apart at ("
										<< points[i].x() << ", " << points[i].y() << ", "
										<< points[i].z() << ")." << std::endl;
								}
							}
						}
					}
				}
				grid[cell].push_back(i);
			}

			if (pair_t1 + pair_t2 + pair_t3 > 0)
			{
				std::cout << "[Warning] SoupTopology(" << label << "): unconnected near vertex pairs"
					<< " under EPSILON/10e/" << near_pair_radius << ": "
					<< pair_t1 << " / " << pair_t2 << " / " << pair_t3
					<< ". A distance-welding importer fuses these." << std::endl;
			}
		}

		// Check 4: non-manifold (bowtie) vertices. All edges can be clean
		// while a vertex still joins two or more separate face fans, for
		// example where a glyph outline touches itself in one point.
		{
			std::vector<std::vector<int>> vertex_tris(points.size());
			for (int t = 0; t < static_cast<int>(indices.size()); ++t)
			{
				vertex_tris[indices[t][0]].push_back(t);
				vertex_tris[indices[t][1]].push_back(t);
				vertex_tris[indices[t][2]].push_back(t);
			}

			size_t bowties = 0;

			for (int v = 0; v < static_cast<int>(points.size()); ++v)
			{
				const std::vector<int>& tris = vertex_tris[v];
				if (tris.size() < 2) continue;

				std::vector<int> parent(tris.size());
				for (size_t i = 0; i < parent.size(); ++i) parent[i] = static_cast<int>(i);

				std::function<int(int)> find = [&](int x) -> int
					{
						while (parent[x] != x)
						{
							parent[x] = parent[parent[x]];
							x = parent[x];
						}
						return x;
					};

				robin_hood::unordered_map<int, int> neighbor_to_tri;
				neighbor_to_tri.reserve(tris.size() * 2);

				for (size_t i = 0; i < tris.size(); ++i)
				{
					const Eigen::Vector3i& tri = indices[tris[i]];
					for (int e = 0; e < 3; ++e)
					{
						int w = tri[e];
						if (w == v) continue;

						auto it = neighbor_to_tri.find(w);
						if (it == neighbor_to_tri.end())
						{
							neighbor_to_tri[w] = static_cast<int>(i);
						}
						else
						{
							int ra = find(static_cast<int>(i));
							int rb = find(it->second);
							if (ra != rb) parent[ra] = rb;
						}
					}
				}

				int components = 0;
				for (size_t i = 0; i < tris.size(); ++i)
				{
					if (find(static_cast<int>(i)) == static_cast<int>(i)) ++components;
				}

				if (components > 1)
				{
					++bowties;
					if (bowties <= 10)
					{
						std::cout << "[Error] SoupTopology(" << label << "): vertex ("
							<< points[v].x() << ", " << points[v].y() << ", " << points[v].z()
							<< ") joins " << components << " separate face fans (non-manifold vertex)." << std::endl;
					}
				}
			}

			if (bowties > 0)
			{
				++failures;
				std::cout << "[Error] SoupTopology(" << label << "): " << bowties
					<< " non-manifold (bowtie) vertices." << std::endl;
			}
		}

		if (0 == failures)
		{
			if (Log::AtInfo())
			{
				std::cout << "[Info] SoupTopology(" << label << "): 2-manifold, no duplicate"
					<< " or degenerate triangles, no bowtie vertices." << std::endl;
			}
			return true;
		}

		std::cout << "[Error] SoupTopology(" << label << "): " << failures
			<< " failure categories." << std::endl;
		return false;
	}
}