#include "RobustGeometricOperations.h"

namespace RGO
{
	OperatorCreateSkirt::OperatorCreateSkirt(Mesh* mesh, float skirt_height)
		: meshTarget(mesh)
		, skirt_height(skirt_height)
		, skirt_direction(Eigen::Vector3f(0.0f, 0.0f, 1.0f))
		, has_explicit_direction(false)
	{
	}
	
	OperatorCreateSkirt::OperatorCreateSkirt(Mesh* mesh, float skirt_height, const Eigen::Vector3f& direction)
		: meshTarget(mesh)
		, skirt_height(skirt_height)
		, skirt_direction(direction)
		, has_explicit_direction(true)
	{
	}

	bool OperatorCreateSkirt::BuildSkirtForLoop(
		const std::vector<OpenMesh::VertexHandle>& loop,
		const Eigen::Vector3f& offset,
		std::vector<Eigen::Vector3f>& out_points,
		std::vector<Eigen::Vector3i>& out_indices) const
	{
		size_t n = loop.size();
		if (n < 3) return false;

		// The loop vertices are gathered in boundary-halfedge order, so
		// consecutive entries loop[i] -> loop[i+1] form an existing border
		// edge running in the boundary direction. The top ring reuses the
		// border vertex coordinates; the bottom ring is the same ring
		// pushed by offset. Two triangles per column stitch them, wound so
		// the wall's outward normal is consistent with that border
		// direction (the boundary halfedge has empty face on its left, so
		// top_a -> top_b -> bottom_b keeps the wall facing outward).
		int base = static_cast<int>(out_points.size());

		for (size_t i = 0; i < n; ++i)
		{
			Eigen::Vector3f top(meshTarget->point(loop[i]).data());
			out_points.push_back(top);
		}
		for (size_t i = 0; i < n; ++i)
		{
			Eigen::Vector3f top(meshTarget->point(loop[i]).data());
			out_points.push_back(top + offset);
		}

		int top_base = base;
		int bot_base = base + static_cast<int>(n);

		for (size_t i = 0; i < n; ++i)
		{
			int ti0 = top_base + static_cast<int>(i);
			int ti1 = top_base + static_cast<int>((i + 1) % n);
			int bi0 = bot_base + static_cast<int>(i);
			int bi1 = bot_base + static_cast<int>((i + 1) % n);

			out_indices.emplace_back(ti0, ti1, bi1);
			out_indices.emplace_back(ti0, bi1, bi0);
		}

		return true;
	}

	bool OperatorCreateSkirt::Execute()
	{
		if (nullptr == meshTarget) return false;
		if (0 == meshTarget->n_faces()) return false;
		if (skirt_height < EPSILON)
		{
			std::cout << "[Error] OperatorCreateSkirt: skirt_height must be positive, got "
				<< skirt_height << "." << std::endl;
			return false;
		}

		std::vector<std::vector<OpenMesh::VertexHandle>> loops = meshTarget->GetBorderLoops();
		if (loops.empty())
		{
			std::cout << "[Warning] OperatorCreateSkirt: mesh has no border loops;"
				<< " nothing to extend. The mesh may already be closed." << std::endl;
			return false;
		}

		std::vector<Eigen::Vector3f> flat_points;
		std::vector<Eigen::Vector3i> flat_indices;
		ExtractMeshSoup(meshTarget, flat_points, flat_indices);

		size_t loops_built = 0;
		for (const auto& loop : loops)
		{
			// The skirt direction is derived from the loop itself. An
			// explicit direction, when supplied, overrides the computed one
			// for every loop; otherwise each loop gets its own outward
			// plane normal, which is correct when a mesh has several holes
			// lying in different planes.
			Eigen::Vector3f dir;
			if (has_explicit_direction)
			{
				dir = skirt_direction;
			}
			else if (false == ComputeLoopDirection(loop, dir))
			{
				std::cout << "[Warning] OperatorCreateSkirt: a border loop is degenerate;"
					<< " skipped." << std::endl;
				continue;
			}

			float dlen = dir.norm();
			if (dlen < 1e-12f) continue;
			dir /= dlen;

			Eigen::Vector3f offset = dir * skirt_height;

			if (BuildSkirtForLoop(loop, offset, flat_points, flat_indices))
			{
				++loops_built;
			}
		}

		if (Log::AtInfo())
		{
			std::cout << "[Info] OperatorCreateSkirt: extended " << loops_built
				<< " of " << loops.size() << " border loops by height "
				<< skirt_height << " (direction computed from each loop)." << std::endl;
		}

		meshTarget->clear();
		meshTarget->Build(flat_points, flat_indices);

		return loops_built > 0;
	}

	bool OperatorCreateSkirt::ComputeLoopDirection(
		const std::vector<OpenMesh::VertexHandle>& loop,
		Eigen::Vector3f& out_direction) const
	{
		size_t n = loop.size();
		if (n < 3) return false;

		// Loop centroid and Newell-method plane normal. Newell accumulates
		// the cross terms over every edge of the loop, so it yields a
		// stable area-weighted normal even when the loop is slightly
		// non-planar or has nearly collinear consecutive points, where a
		// single three-point cross product would collapse to zero.
		Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
		for (size_t i = 0; i < n; ++i)
		{
			centroid += Eigen::Vector3f(meshTarget->point(loop[i]).data());
		}
		centroid /= static_cast<float>(n);

		Eigen::Vector3f normal = Eigen::Vector3f::Zero();
		for (size_t i = 0; i < n; ++i)
		{
			Eigen::Vector3f cur(meshTarget->point(loop[i]).data());
			Eigen::Vector3f nxt(meshTarget->point(loop[(i + 1) % n]).data());

			normal.x() += (cur.y() - nxt.y()) * (cur.z() + nxt.z());
			normal.y() += (cur.z() - nxt.z()) * (cur.x() + nxt.x());
			normal.z() += (cur.x() - nxt.x()) * (cur.y() + nxt.y());
		}

		float nlen = normal.norm();
		if (nlen < 1e-12f) return false;
		normal /= nlen;

		// Sign disambiguation. The skirt must grow AWAY from the surface.
		// Averaging the centroids of the faces incident to the loop
		// vertices gives a point on the surface side of the border; the
		// vector from the loop centroid toward that point therefore points
		// into the surface, so the skirt normal must oppose it. If the
		// plane normal currently agrees with the inward direction, flip it.
		Eigen::Vector3f surface_side = Eigen::Vector3f::Zero();
		size_t face_count = 0;
		for (size_t i = 0; i < n; ++i)
		{
			for (auto vf_it = meshTarget->vf_iter(loop[i]); vf_it.is_valid(); ++vf_it)
			{
				OpenMesh::FaceHandle fh = *vf_it;
				if (meshTarget->status(fh).deleted()) continue;

				Eigen::Vector3f v0, v1, v2;
				meshTarget->GetFaceVertices(fh, v0, v1, v2);
				surface_side += (v0 + v1 + v2) / 3.0f;
				++face_count;
			}
		}

		if (face_count > 0)
		{
			surface_side /= static_cast<float>(face_count);
			Eigen::Vector3f inward = surface_side - centroid;
			if (normal.dot(inward) > 0.0f)
			{
				normal = -normal;
			}
		}

		out_direction = normal;
		return true;
	}

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
		input_boundary_edges.clear();
		coplanar_face_pairs.clear();
		open_chain_count = 0;

		if (nullptr == meshA || nullptr == meshB) return false;
		if (0 == meshA->n_faces() || 0 == meshB->n_faces()) return false;

		// Input topology diagnostics. The pipeline preserves the input
		// surface everywhere it does not cut, so defects already present
		// in an input (self-touching outlines, near-coincident sheets)
		// reappear unchanged in the result. They are reported here at
		// their true origin; the run continues, the verdict is data for
		// the caller and for the input generator.
		{
			std::vector<Eigen::Vector3f> diag_points;
			std::vector<Eigen::Vector3i> diag_indices;

			ExtractMeshSoup(meshA, diag_points, diag_indices);
			ValidateTriangleSoup(diag_points, diag_indices, "inputA", 100.0f * EPSILON);

			ExtractMeshSoup(meshB, diag_points, diag_indices);
			ValidateTriangleSoup(diag_points, diag_indices, "inputB", 100.0f * EPSILON);
		}

		// Input boundary edges are recorded up front so open chains can
		// later be judged against the input borders.
		CollectInputBoundaryEdges(meshA, "meshA");
		CollectInputBoundaryEdges(meshB, "meshB");

		size_t num_faces_b = meshB->n_faces();

		// Each B face writes only to its own slot: no mutex needed
		std::vector<std::vector<IntersectionSegment>> per_face_segments(num_faces_b);
		std::vector<std::vector<std::pair<int, int>>> per_face_coplanar(num_faces_b);
		std::vector<int> face_indices(num_faces_b);
		std::iota(face_indices.begin(), face_indices.end(), 0);

		std::for_each(std::execution::par_unseq, face_indices.begin(), face_indices.end(), [&](int i)
			{
				CollectSegmentsForFaceB(i, per_face_segments[i], per_face_coplanar[i]);
			});

		// Coplanar overlapping pairs mark the OnSurface configuration.
		// They produce no curve segment here; the overlap region boundary
		// comes from the tangent-edge and transversal cases. The pairs
		// are recorded for the OnSurface classification stage.
		for (const auto& list : per_face_coplanar)
		{
			coplanar_face_pairs.insert(coplanar_face_pairs.end(), list.begin(), list.end());
		}
		if (false == coplanar_face_pairs.empty())
		{
			if (Log::AtInfo())
			{
				std::cout << "[Info] OperatorIntersectionLoops: " << coplanar_face_pairs.size()
					<< " coplanar overlapping face pairs recorded for the OnSurface stage." << std::endl;
			}
		}

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

		// Open chains are not failures per se: on an open input the curve
		// legitimately ends at the input border. They are failures when an
		// endpoint stops in the middle of both surfaces, which means
		// segments were missed upstream.
		if (false == ValidateOpenChains()) return false;

		return true;
	}

	bool OperatorIntersectionLoops::ExecuteFromSurfaceCurve(
		Mesh* mesh,
		const std::vector<Eigen::Vector3f>& curve_points,
		const std::vector<OpenMesh::FaceHandle>& curve_faces,
		bool closed)
	{
		segments.clear();
		loops.clear();
		segments_by_face_a.clear();
		segments_by_face_b.clear();
		input_boundary_edges.clear();
		coplanar_face_pairs.clear();
		open_chain_count = 0;

		if (nullptr == mesh) return false;
		if (0 == mesh->n_faces()) return false;

		// Single-mesh mode: both face references in every segment point to
		// the same mesh, so the existing two-mesh canonicalize and validate
		// code runs unchanged with meshA == meshB.
		meshA = mesh;
		meshB = mesh;
		single_mesh_mode = true;

		CollectInputBoundaryEdges(mesh, "curveMesh");

		if (false == CollectSegmentsFromCurve(curve_points, curve_faces, closed))
		{
			std::cout << "[Error] ExecuteFromSurfaceCurve: no segments produced from curve." << std::endl;
			return false;
		}

		// From here the path is identical to the two-mesh Execute: snap and
		// weld endpoints to mesh features, build the face lookup, validate
		// CDT-input safety, then trace the loop.
		CanonicalizeSegments();

		BuildFaceLookupTables();

		if (false == ValidateCanonicalization()) return false;

		BuildLoops();

		if (false == ValidateSegmentEndpoints()) return false;

		if (false == ValidateOpenChains()) return false;

		return true;
	}

	OpenMesh::HalfedgeHandle OperatorIntersectionLoops::SharedHalfedge(
		OpenMesh::FaceHandle fa,
		OpenMesh::FaceHandle fb) const
	{
		if (false == fa.is_valid() || false == fb.is_valid())
			return OpenMesh::HalfedgeHandle();

		for (auto fh_it = meshA->cfh_iter(fa); fh_it.is_valid(); ++fh_it)
		{
			OpenMesh::HalfedgeHandle opp = meshA->opposite_halfedge_handle(*fh_it);
			if (meshA->is_boundary(opp)) continue;

			if (meshA->face_handle(opp) == fb)
			{
				return *fh_it;
			}
		}

		return OpenMesh::HalfedgeHandle();
	}

	bool OperatorIntersectionLoops::SegmentEdgeCrossing(
		OpenMesh::FaceHandle fa,
		OpenMesh::HalfedgeHandle edge_he,
		const Eigen::Vector3f& p0,
		const Eigen::Vector3f& p1,
		Eigen::Vector3f& out_point) const
	{
		Eigen::Vector3f e0(meshA->point(meshA->from_vertex_handle(edge_he)).data());
		Eigen::Vector3f e1(meshA->point(meshA->to_vertex_handle(edge_he)).data());

		// Solve the crossing in the plane of face fa. The shared edge
		// vertices e0,e1 lie exactly on fa's plane and on the neighbour's
		// plane, so returning the point parametrized ALONG the edge places
		// it on both face planes and, crucially, makes the crossing of a
		// given mesh edge identical no matter which of the two incident
		// faces computed it. That identity is what lets neighbouring curve
		// segments weld into one chain instead of tearing into open chains.
		Eigen::Vector3f fv0, fv1, fv2;
		meshA->GetFaceVertices(fa, fv0, fv1, fv2);
		Eigen::Vector3f n = (fv1 - fv0).cross(fv2 - fv0);
		float nlen = n.norm();
		if (nlen < 1e-12f) return false;
		n /= nlen;

		Eigen::Vector3f bx = (fv1 - fv0);
		float bxl = bx.norm();
		if (bxl < 1e-12f) return false;
		bx /= bxl;
		Eigen::Vector3f by = n.cross(bx);

		auto to2d = [&](const Eigen::Vector3f& p) -> Eigen::Vector2f
			{
				Eigen::Vector3f d = p - fv0;
				return Eigen::Vector2f(d.dot(bx), d.dot(by));
			};

		Eigen::Vector2f a0 = to2d(p0);
		Eigen::Vector2f a1 = to2d(p1);
		Eigen::Vector2f b0 = to2d(e0);
		Eigen::Vector2f b1 = to2d(e1);

		Eigen::Vector2f da = a1 - a0;
		Eigen::Vector2f db = b1 - b0;

		float denom = da.x() * db.y() - da.y() * db.x();
		if (std::abs(denom) < 1e-12f) return false;

		Eigen::Vector2f diff = b0 - a0;
		float t = (diff.x() * db.y() - diff.y() * db.x()) / denom;
		float u = (diff.x() * da.y() - diff.y() * da.x()) / denom;

		const float lo = -1e-3f;
		const float hi = 1.0f + 1e-3f;
		if (t < lo || t > hi || u < lo || u > hi) return false;

		// Place the point on the edge via the edge parameter u, NOT on the
		// chord via t. e0,e1 are mesh vertices, so this is bit-identical
		// across both incident faces for the same u.
		float uc = std::max(0.0f, std::min(1.0f, u));
		out_point = e0 + (e1 - e0) * uc;
		return true;
	}

	bool OperatorIntersectionLoops::WalkFaceStrip(
		OpenMesh::FaceHandle fa,
		OpenMesh::FaceHandle fb,
		const Eigen::Vector3f& p_from,
		const Eigen::Vector3f& p_to,
		std::vector<Eigen::Vector3f>& out_crossings,
		std::vector<OpenMesh::FaceHandle>& out_faces) const
	{
		out_crossings.clear();
		out_faces.clear();

		// Same face: the whole curve piece stays inside fa, no crossing.
		if (fa == fb)
		{
			out_faces.push_back(fa);
			return true;
		}

		// Find the actual mesh face path fa ... fb. A straight chord between
		// two surface samples does not follow the surface across curvature,
		// so the previous chord-vs-edge crossing solve missed the shared
		// edge even for directly adjacent faces (hops == 1 in diagnostics).
		// The BFS path is purely topological and always connects the two
		// faces; each consecutive pair shares exactly one edge, and the
		// crossing is placed on that edge.
		std::vector<OpenMesh::FaceHandle> path;
		if (false == FindFacePathBFS(fa, fb, path)) return false;
		if (path.size() < 2) return false;

		out_faces.push_back(path[0]);

		for (size_t k = 0; k + 1 < path.size(); ++k)
		{
			OpenMesh::HalfedgeHandle shared = SharedHalfedge(path[k], path[k + 1]);
			if (false == shared.is_valid()) return false;

			Eigen::Vector3f x;
			if (false == CrossingOnSharedEdge(shared, p_from, p_to, x)) return false;

			out_crossings.push_back(x);
			out_faces.push_back(path[k + 1]);
		}

		return true;
	}

	void OperatorIntersectionLoops::EmitCurveSegment(
		const Eigen::Vector3f& p0,
		const Eigen::Vector3f& p1,
		int face_index,
		bool project_p0,
		bool project_p1)
	{
		// Curve samples are true surface points off the face plane and must
		// be projected onto face_index's plane. Edge crossings already lie
		// on a shared edge (hence on both incident planes) and are passed
		// through unprojected: projecting a crossing onto one of its two
		// faces would split the single mesh-edge point into two slightly
		// different coordinates and tear the chain apart at that node.
		OpenMesh::FaceHandle f = meshA->face_handle(face_index);

		Eigen::Vector3f q0 = p0;
		Eigen::Vector3f q1 = p1;
		if (f.is_valid())
		{
			if (project_p0) q0 = ProjectToFacePlane(f, p0);
			if (project_p1) q1 = ProjectToFacePlane(f, p1);
		}

		if ((q0 - q1).squaredNorm() < EPSILON * EPSILON) return;

		IntersectionSegment seg;
		seg.p0 = q0;
		seg.p1 = q1;

		seg.face_a = face_index;
		seg.face_b = face_index;

		segments.push_back(seg);
	}

	bool OperatorIntersectionLoops::FaceVerticesSafe(
		OpenMesh::FaceHandle f,
		Eigen::Vector3f& a,
		Eigen::Vector3f& b,
		Eigen::Vector3f& c) const
	{
		if (false == f.is_valid()) return false;

		Eigen::Vector3f v[3];
		int cnt = 0;
		for (auto fv_it = meshA->cfv_iter(f); fv_it.is_valid(); ++fv_it)
		{
			if (cnt < 3)
			{
				const auto& p = meshA->point(*fv_it);
				v[cnt] = Eigen::Vector3f(p[0], p[1], p[2]);
			}
			++cnt;
		}

		if (3 != cnt) return false;

		a = v[0];
		b = v[1];
		c = v[2];
		return true;
	}

	Eigen::Vector3f OperatorIntersectionLoops::ProjectToFacePlane(
		OpenMesh::FaceHandle f,
		const Eigen::Vector3f& p) const
	{
		Eigen::Vector3f a, b, c;
		if (false == FaceVerticesSafe(f, a, b, c)) return p;

		Eigen::Vector3f n = (b - a).cross(c - a);
		float len = n.norm();
		if (len < 1e-20f) return p;
		n /= len;

		float signed_dist = (p - a).dot(n);
		return p - n * signed_dist;
	}

	bool OperatorIntersectionLoops::CollectSegmentsFromCurve(
		const std::vector<Eigen::Vector3f>& curve_points,
		const std::vector<OpenMesh::FaceHandle>& curve_faces,
		bool closed)
	{
		const size_t n = curve_points.size();
		if (n < 2 || curve_faces.size() != n)
		{
			std::cout << "[Error] CollectSegmentsFromCurve: need matching point and"
				<< " face arrays of length >= 2." << std::endl;
			return false;
		}

		// One crossing per mesh edge across the whole curve. Cleared here so
		// a fresh curve starts with no cached crossings; CrossingOnSharedEdge
		// fills it and reuses each edge's first crossing on later visits.
		edge_crossing_cache.clear();

		const size_t pair_count = closed ? n : (n - 1);
		size_t walk_failures = 0;

		for (size_t i = 0; i < pair_count; ++i)
		{
			const Eigen::Vector3f& p_from = curve_points[i];
			const Eigen::Vector3f& p_to = curve_points[(i + 1) % n];

			OpenMesh::FaceHandle fa = curve_faces[i];
			OpenMesh::FaceHandle fb = curve_faces[(i + 1) % n];

			if (false == fa.is_valid() || false == fb.is_valid())
			{
				++walk_failures;
				continue;
			}

			std::vector<Eigen::Vector3f> crossings;
			std::vector<OpenMesh::FaceHandle> strip_faces;
			if (false == WalkFaceStrip(fa, fb, p_from, p_to, crossings, strip_faces))
			{
				++walk_failures;
				EmitCurveSegment(p_from, p_to, fa.idx(), true, true);
				continue;
			}

			Eigen::Vector3f seg_start = p_from;
			bool start_is_sample = true;

			for (size_t k = 0; k < strip_faces.size(); ++k)
			{
				bool end_is_sample = (k >= crossings.size());
				Eigen::Vector3f seg_end = end_is_sample ? p_to : crossings[k];

				EmitCurveSegment(seg_start, seg_end, strip_faces[k].idx(),
					start_is_sample, end_is_sample);

				seg_start = seg_end;
				start_is_sample = false;
			}
		}

		if (walk_failures > 0)
		{
			std::cout << "[Warning] CollectSegmentsFromCurve: " << walk_failures
				<< " curve spans had invalid faces or were emitted as a single"
				<< " in-face chord." << std::endl;
		}

		if (Log::AtInfo())
		{
			std::cout << "[Info] CollectSegmentsFromCurve: " << segments.size()
				<< " per-face segments from " << n << " curve points." << std::endl;
		}

		return false == segments.empty();
	}

	void OperatorIntersectionLoops::CollectSegmentsForFaceB(
		int face_b_index,
		std::vector<IntersectionSegment>& out_segments,
		std::vector<std::pair<int, int>>& out_coplanar_pairs) const
	{
		out_segments.clear();
		out_coplanar_pairs.clear();

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

		Eigen::Vector3f nB = (b1 - b0).cross(b2 - b0);
		float nB_len = nB.norm();
		if (nB_len < 1e-12f) return;
		nB /= nB_len;

		const Eigen::Vector3f bv[3] = { b0, b1, b2 };

		auto emit = [&](const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, int face_a_index)
			{
				if ((p0 - p1).squaredNorm() < EPSILON * EPSILON) return;

				IntersectionSegment seg;
				seg.p0 = p0;
				seg.p1 = p1;
				seg.face_a = face_a_index;
				seg.face_b = face_b_index;
				out_segments.push_back(seg);
			};

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

			Eigen::Vector3f nA = (a1 - a0).cross(a2 - a0);
			float nA_len = nA.norm();
			if (nA_len < 1e-12f) continue;
			nA /= nA_len;

			const Eigen::Vector3f av[3] = { a0, a1, a2 };

			// Vertex-plane classification with explicit epsilon bands.
			// The generic transversal path is fragile exactly when
			// vertices sit ON the other plane, so those configurations
			// are routed to deterministic special cases instead.
			int b_on_idx[3];
			int b_on = 0;
			int b_pos = 0;
			int b_neg = 0;
			for (int i = 0; i < 3; ++i)
			{
				float d = (bv[i] - a0).dot(nA);
				if (std::abs(d) < EPSILON) { b_on_idx[b_on++] = i; }
				else if (d > 0.0f) ++b_pos;
				else ++b_neg;
			}

			// All of B in plane A: coplanar overlap, recorded for the
			// OnSurface stage. No curve segment exists for this pair.
			if (3 == b_on)
			{
				out_coplanar_pairs.push_back({ face_a_index, face_b_index });
				continue;
			}

			int a_on_idx[3];
			int a_on = 0;
			int a_pos = 0;
			int a_neg = 0;
			for (int i = 0; i < 3; ++i)
			{
				float d = (av[i] - b0).dot(nB);
				if (std::abs(d) < EPSILON) { a_on_idx[a_on++] = i; }
				else if (d > 0.0f) ++a_pos;
				else ++a_neg;
			}

			if (3 == a_on)
			{
				out_coplanar_pairs.push_back({ face_a_index, face_b_index });
				continue;
			}

			// Tangent edge of B lying in plane A: the intersection is a
			// sub-piece of that edge. Clip it to triangle A in 2D, which
			// is deterministic, instead of the fragile generic path.
			if (2 == b_on)
			{
				const Eigen::Vector3f& s0 = bv[b_on_idx[0]];
				const Eigen::Vector3f& s1 = bv[b_on_idx[1]];

				Eigen::Vector3f p0, p1;
				if (Intersection::CoplanarSegmentToTriangle(s0, s1, a0, a1, a2, p0, p1))
				{
					if (2 == a_on)
					{
						// Both tangent: the two on-edges are collinear on
						// the plane intersection line. Clamp to the
						// overlap of the two intervals.
						Eigen::Vector3f dir = p1 - p0;
						float dl = dir.norm();
						if (dl >= EPSILON)
						{
							dir /= dl;
							float ta0 = (av[a_on_idx[0]] - p0).dot(dir);
							float ta1 = (av[a_on_idx[1]] - p0).dot(dir);
							float lo = std::max(0.0f, std::min(ta0, ta1));
							float hi = std::min(dl, std::max(ta0, ta1));
							if (hi - lo >= EPSILON)
							{
								emit(p0 + dir * lo, p0 + dir * hi, face_a_index);
							}
						}
					}
					else
					{
						emit(p0, p1, face_a_index);
					}
				}
				continue;
			}

			// Tangent edge of A lying in plane B: symmetric case
			if (2 == a_on)
			{
				const Eigen::Vector3f& s0 = av[a_on_idx[0]];
				const Eigen::Vector3f& s1 = av[a_on_idx[1]];

				Eigen::Vector3f p0, p1;
				if (Intersection::CoplanarSegmentToTriangle(s0, s1, b0, b1, b2, p0, p1))
				{
					emit(p0, p1, face_a_index);
				}
				continue;
			}

			// A triangle that does not cross the other plane can only
			// touch it at a vertex: a point contact, no constraint edge.
			bool b_crosses = (b_pos > 0 && b_neg > 0);
			bool a_crosses = (a_pos > 0 && a_neg > 0);
			if (false == b_crosses || false == a_crosses) continue;

			// Generic transversal intersection
			Eigen::Vector3f ipA, ipB;
			Intersection::TriangleIntersectionResult result =
				Intersection::TriangleToTriangle(a0, a1, a2, b0, b1, b2, ipA, ipB);

			// Point contacts give no constraint edge for co-refinement: skip
			if (result.type != Intersection::TriangleTriangleIntersectionType::Segment) continue;

			emit(result.pointA, result.pointB, face_a_index);
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
		//
		// The NEAREST node within EPSILON is selected, not the first one
		// found. With several nodes inside the weld radius, first-found
		// assignment depends on hash iteration order and can attach an
		// endpoint to the wrong node, corrupting curve connectivity. The
		// nearest choice is deterministic, and an exact duplicate of an
		// existing node always rejoins that node (distance zero).
		int best_id = -1;
		float best_d2 = EPSILON * EPSILON;

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
						float d2 = (node_positions[node_id] - p).squaredNorm();
						if (d2 < best_d2)
						{
							best_d2 = d2;
							best_id = node_id;
						}
					}
				}
			}
		}

		if (best_id >= 0)
		{
			return best_id;
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

		std::vector<char> segment_used(segments.size(), 0);

		// Pass 0: topological deduplication.
		// The same geometric piece of the intersection curve is reported
		// by several face pairs when it lies within EPSILON of shared
		// edges of BOTH meshes at once: at a 2x2 face corner crossing the
		// tiny corner piece is produced by both diagonal face pairs,
		// bit-identical after canonicalization. The curve needs every
		// piece exactly once. A duplicate raises the degree of both its
		// nodes to an odd number, which makes closed traversal impossible
		// and tears the loop into open chains. Duplicates are excluded
		// from the topology here but remain in `segments`, so every face
		// still receives its constraints for co-refinement.
		{
			struct SegKey
			{
				Eigen::Vector3f a;
				Eigen::Vector3f b;
			};
			struct SegKeyHash
			{
				size_t operator()(const SegKey& k) const
				{
					Vector3fBitHash h;
					return h(k.a) * 1000003ull ^ h(k.b);
				}
			};
			struct SegKeyEqual
			{
				bool operator()(const SegKey& x, const SegKey& y) const
				{
					Vector3fBitEqual eq;
					return eq(x.a, y.a) && eq(x.b, y.b);
				}
			};

			auto less_xyz = [](const Eigen::Vector3f& a, const Eigen::Vector3f& b) -> bool
				{
					if (a.x() != b.x()) return a.x() < b.x();
					if (a.y() != b.y()) return a.y() < b.y();
					return a.z() < b.z();
				};

			robin_hood::unordered_map<SegKey, int, SegKeyHash, SegKeyEqual> first_by_pair;
			first_by_pair.reserve(segments.size());

			size_t duplicate_count = 0;
			const size_t report_limit = 8;

			for (int i = 0; i < static_cast<int>(segments.size()); ++i)
			{
				SegKey key;
				if (less_xyz(segments[i].p0, segments[i].p1))
				{
					key.a = segments[i].p0;
					key.b = segments[i].p1;
				}
				else
				{
					key.a = segments[i].p1;
					key.b = segments[i].p0;
				}

				auto it = first_by_pair.find(key);
				if (it == first_by_pair.end())
				{
					first_by_pair[key] = i;
					continue;
				}

				segment_used[i] = 1;
				++duplicate_count;

				if (duplicate_count <= report_limit)
				{
					if (Log::AtInfo())
					{
						std::cout << "[Info] BuildLoops: segment " << i
							<< " (faceA " << segments[i].face_a
							<< ", faceB " << segments[i].face_b
							<< ", len " << (segments[i].p0 - segments[i].p1).norm()
							<< ") duplicates segment " << it->second
							<< " (faceA " << segments[it->second].face_a
							<< ", faceB " << segments[it->second].face_b
							<< "): excluded from curve topology." << std::endl;
					}
				}
			}

			if (duplicate_count > 0)
			{
				if (Log::AtInfo())
				{
					std::cout << "[Info] BuildLoops: " << duplicate_count
						<< " duplicate segments excluded from topology, "
						<< (segments.size() - duplicate_count)
						<< " remain for loop tracing." << std::endl;
				}
			}
		}

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

		for (size_t i = 0; i < segments.size(); ++i)
		{
			// Topology duplicates marked above must not contribute edges
			if (segment_used[i]) continue;

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
		size_t open_chain_count_local = 0;
		for (size_t i = 0; i < segments.size(); ++i)
		{
			if (segment_used[i]) continue;

			IntersectionLoop loop = TraceLoopFromSegment(
				static_cast<int>(i), segment_nodes, node_to_segments, node_positions, segment_used);

			if (false == loop.closed)
			{
				++open_chain_count_local;
			}
			loops.push_back(std::move(loop));
		}

		if (open_chain_count_local > 0)
		{
			std::cout << "[Warning] BuildLoops: " << open_chain_count_local
				<< " open chains out of " << loops.size()
				<< " curves. Watertight inputs must yield closed loops only;"
				<< " check TriangleToTriangle for missed segments." << std::endl;
		}
	}

	void OperatorIntersectionLoops::CanonicalizeSegments()
	{
		canonical_snap_count = 0;
		if (segments.empty()) return;

		// Pass 1: weld RAW endpoints into logical nodes FIRST. Identity
		// is decided before any snapping, so every instance of one
		// intersection point joins the same node no matter which face
		// pair reported it. The previous order (snap per face pair, then
		// weld) let two instances of one point snap to different features
		// of different faces; the weld then had to pick one coordinate
		// arbitrarily, and re-canonicalization moved such points by up to
		// EPSILON. That interference is removed here at the root.
		std::vector<Eigen::Vector3f> node_positions;
		node_positions.reserve(segments.size() * 2);
		robin_hood::unordered_map<Eigen::Vector3i, std::vector<int>, Int3Hash, Int3Equal> cell_to_nodes;

		std::vector<std::pair<int, int>> segment_nodes(segments.size());
		for (size_t i = 0; i < segments.size(); ++i)
		{
			int n0 = WeldEndpoint(segments[i].p0, cell_to_nodes, node_positions);
			int n1 = WeldEndpoint(segments[i].p1, cell_to_nodes, node_positions);
			segment_nodes[i] = { n0, n1 };
		}

		// Pass 2: union of incident faces per node, gathered from every
		// segment touching the node.
		std::vector<std::vector<int>> node_faces_a;
		std::vector<std::vector<int>> node_faces_b;
		GatherNodeFaces(segment_nodes, node_positions.size(), node_faces_a, node_faces_b);

		// Pass 3: ONE snap decision per node against ALL candidate
		// features of all incident faces.
		std::vector<SnapClass> node_snap_class(node_positions.size(), SnapClass::Free);
		for (size_t n = 0; n < node_positions.size(); ++n)
		{
			Eigen::Vector3f snapped;
			SnapClass cls = SnapNodePosition(node_positions[n], node_faces_a[n], node_faces_b[n], snapped);
			node_snap_class[n] = cls;
			if (SnapClass::Free != cls)
			{
				node_positions[n] = snapped;
				++canonical_snap_count;
			}
		}

		// Pass 4: snapping can move two distinct nodes to within EPSILON
		// of each other (for example both onto nearby spots of one edge).
		// Such pairs are re-merged, and merged representatives are
		// re-snapped with the merged face union so the final coordinate
		// is again a single feature decision.
		std::vector<int> node_remap;
		MergeSnappedNodes(node_positions, node_snap_class, node_faces_a, node_faces_b, node_remap);

		// Pass 5: write node coordinates back. All endpoints of one node
		// are bit-identical by construction.
		for (size_t i = 0; i < segments.size(); ++i)
		{
			segments[i].p0 = node_positions[node_remap[segment_nodes[i].first]];
			segments[i].p1 = node_positions[node_remap[segment_nodes[i].second]];
		}

		// Pass 6: drop segments that collapsed to zero length. These
		// would otherwise seed degenerate constraint edges in CDT.
		size_t before = segments.size();
		segments.erase(
			std::remove_if(segments.begin(), segments.end(),
				[](const IntersectionSegment& s)
				{
					return (s.p0 - s.p1).squaredNorm() < EPSILON * EPSILON;
				}),
			segments.end());

		size_t removed = before - segments.size();
		if (Log::AtInfo())
		{
			std::cout << "[Info] CanonicalizeSegments: " << canonical_snap_count
				<< " nodes snapped to features, " << removed
				<< " degenerate segments removed, " << segments.size()
				<< " segments remain." << std::endl;
		}
	}

	void OperatorIntersectionLoops::GatherNodeFaces(
		const std::vector<std::pair<int, int>>& segment_nodes,
		size_t node_count,
		std::vector<std::vector<int>>& out_faces_a,
		std::vector<std::vector<int>>& out_faces_b) const
	{
		out_faces_a.assign(node_count, std::vector<int>());
		out_faces_b.assign(node_count, std::vector<int>());

		for (size_t i = 0; i < segments.size(); ++i)
		{
			const int nodes[2] = { segment_nodes[i].first, segment_nodes[i].second };
			for (int e = 0; e < 2; ++e)
			{
				out_faces_a[nodes[e]].push_back(segments[i].face_a);
				out_faces_b[nodes[e]].push_back(segments[i].face_b);
			}
		}

		// Sorted unique lists make the snap scan order deterministic,
		// which keeps tie-breaking on equal distances deterministic too.
		auto sort_unique = [](std::vector<int>& v)
			{
				std::sort(v.begin(), v.end());
				v.erase(std::unique(v.begin(), v.end()), v.end());
			};

		for (size_t n = 0; n < node_count; ++n)
		{
			sort_unique(out_faces_a[n]);
			sort_unique(out_faces_b[n]);
		}
	}

	bool OperatorIntersectionLoops::SnapPointToNearestVertex(
		const Eigen::Vector3f& p,
		const std::vector<int>& faces_a,
		const std::vector<int>& faces_b,
		Eigen::Vector3f& out_point) const
	{
		// Nearest vertex within EPSILON across ALL incident faces of both
		// meshes. Strict less-than keeps the first candidate on exact
		// ties, and the scan order is deterministic (sorted face lists,
		// fixed mesh order A then B).
		float best_d2 = EPSILON * EPSILON;
		bool found = false;

		auto scan = [&](const Mesh* mesh, const std::vector<int>& faces)
			{
				for (int f : faces)
				{
					Eigen::Vector3f v0, v1, v2;
					mesh->GetFaceVertices(mesh->face_handle(f), v0, v1, v2);
					const Eigen::Vector3f* verts[3] = { &v0, &v1, &v2 };

					for (int i = 0; i < 3; ++i)
					{
						float d2 = (p - *verts[i]).squaredNorm();
						if (d2 < best_d2)
						{
							best_d2 = d2;
							out_point = *verts[i];
							found = true;
						}
					}
				}
			};

		scan(meshA, faces_a);
		scan(meshB, faces_b);
		return found;
	}

	bool OperatorIntersectionLoops::SnapPointToNearestEdge(
		const Eigen::Vector3f& p,
		const std::vector<int>& faces_a,
		const std::vector<int>& faces_b,
		Eigen::Vector3f& out_point) const
	{
		// Nearest edge projection within EPSILON across ALL incident
		// faces of both meshes, deterministic for the same reasons as
		// the vertex scan.
		float best_d2 = EPSILON * EPSILON;
		bool found = false;

		auto scan = [&](const Mesh* mesh, const std::vector<int>& faces)
			{
				for (int f : faces)
				{
					Eigen::Vector3f v0, v1, v2;
					mesh->GetFaceVertices(mesh->face_handle(f), v0, v1, v2);

					const Eigen::Vector3f* ea[3] = { &v0, &v1, &v2 };
					const Eigen::Vector3f* eb[3] = { &v1, &v2, &v0 };

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
							out_point = proj;
							found = true;
						}
					}
				}
			};

		scan(meshA, faces_a);
		scan(meshB, faces_b);
		return found;
	}

	OperatorIntersectionLoops::SnapClass OperatorIntersectionLoops::SnapNodePosition(
		const Eigen::Vector3f& p,
		const std::vector<int>& faces_a,
		const std::vector<int>& faces_b,
		Eigen::Vector3f& out_point) const
	{
		// Vertex candidates win over edge candidates: a vertex is the
		// most constrained feature. The decision is iterated because an
		// edge snap can move the point into the EPSILON ball of a vertex
		// that the raw position was just outside of; the vertex must then
		// win. Vertex snaps are absorbing and a point lying on an edge
		// re-selects that edge at distance zero, so the iteration reaches
		// a fixed point in at most two steps: free -> edge -> vertex.
		Eigen::Vector3f current = p;
		SnapClass cls = SnapClass::Free;

		for (int iter = 0; iter < 2; ++iter)
		{
			Eigen::Vector3f snapped;
			if (SnapPointToNearestVertex(current, faces_a, faces_b, snapped))
			{
				out_point = snapped;
				return SnapClass::Vertex;
			}
			if (false == SnapPointToNearestEdge(current, faces_a, faces_b, snapped))
			{
				break;
			}
			current = snapped;
			cls = SnapClass::Edge;
		}

		out_point = current;
		return cls;
	}

	void OperatorIntersectionLoops::MergeSnappedNodes(
		std::vector<Eigen::Vector3f>& node_positions,
		std::vector<SnapClass>& node_snap_class,
		std::vector<std::vector<int>>& node_faces_a,
		std::vector<std::vector<int>>& node_faces_b,
		std::vector<int>& out_node_remap) const
	{
		size_t node_count = node_positions.size();
		out_node_remap.assign(node_count, -1);

		// Representatives are registered in class priority order so a
		// vertex-snapped node can never be absorbed by an edge-snapped or
		// free node: the most constrained coordinate always survives.
		std::vector<int> order(node_count);
		std::iota(order.begin(), order.end(), 0);
		std::stable_sort(order.begin(), order.end(),
			[&](int a, int b)
			{
				return static_cast<int>(node_snap_class[a]) > static_cast<int>(node_snap_class[b]);
			});

		robin_hood::unordered_map<Eigen::Vector3i, std::vector<int>, Int3Hash, Int3Equal> cell_to_reps;
		std::vector<char> rep_dirty(node_count, 0);
		std::vector<int> dirty_reps;

		auto merge_faces = [](std::vector<int>& target, const std::vector<int>& source)
			{
				target.insert(target.end(), source.begin(), source.end());
				std::sort(target.begin(), target.end());
				target.erase(std::unique(target.begin(), target.end()), target.end());
			};

		size_t merged_count = 0;

		for (int n : order)
		{
			const Eigen::Vector3f& p = node_positions[n];
			Eigen::Vector3i cell = QuantizePoint(p);

			// 27-cell scan, nearest representative within EPSILON, same
			// rationale as WeldEndpoint.
			int best_rep = -1;
			float best_d2 = EPSILON * EPSILON;

			for (int dz = -1; dz <= 1; ++dz)
			{
				for (int dy = -1; dy <= 1; ++dy)
				{
					for (int dx = -1; dx <= 1; ++dx)
					{
						auto it = cell_to_reps.find(Eigen::Vector3i(cell.x() + dx, cell.y() + dy, cell.z() + dz));
						if (it == cell_to_reps.end()) continue;

						for (int rep : it->second)
						{
							float d2 = (node_positions[rep] - p).squaredNorm();
							if (d2 < best_d2)
							{
								best_d2 = d2;
								best_rep = rep;
							}
						}
					}
				}
			}

			if (best_rep >= 0)
			{
				out_node_remap[n] = best_rep;
				++merged_count;

				// The representative inherits the member's incident faces
				// and is re-snapped below with the merged union, so the
				// final coordinate reflects one decision over all faces.
				merge_faces(node_faces_a[best_rep], node_faces_a[n]);
				merge_faces(node_faces_b[best_rep], node_faces_b[n]);
				if (0 == rep_dirty[best_rep])
				{
					rep_dirty[best_rep] = 1;
					dirty_reps.push_back(best_rep);
				}
				continue;
			}

			out_node_remap[n] = n;
			cell_to_reps[cell].push_back(n);
		}

		// Re-snap merged representatives once. Vertex snaps are absorbing
		// and edge re-selection at distance zero is stable, so one pass
		// reaches the fixed point. Residual sub-EPSILON proximities that
		// could in principle remain are exactly what ConstraintSpacing
		// validates afterwards.
		for (int rep : dirty_reps)
		{
			Eigen::Vector3f snapped;
			SnapClass cls = SnapNodePosition(node_positions[rep], node_faces_a[rep], node_faces_b[rep], snapped);
			if (SnapClass::Free != cls)
			{
				node_positions[rep] = snapped;
				node_snap_class[rep] = cls;
			}
		}

		if (merged_count > 0)
		{
			if (Log::AtInfo())
			{
				std::cout << "[Info] MergeSnappedNodes: " << merged_count
					<< " nodes merged after snapping, " << dirty_reps.size()
					<< " representatives re-snapped with merged face unions." << std::endl;
			}
		}
	}

	bool OperatorIntersectionLoops::ValidateCanonicalIdempotence()
	{
		// One snap decision exists per canonical point, so re-running the
		// node-level snap on a canonical coordinate must return that same
		// coordinate: a point on a vertex re-selects the vertex and a
		// point on an edge re-selects that edge, both at distance zero.
		// The only residual movement is float rounding of the edge
		// re-projection (a + t * ab), a few ULPs: at coordinate magnitude
		// 50 one ULP is already 3.8e-6. The tolerance of half EPSILON is
		// far above ULP noise but strict enough that any real second
		// feature decision (about EPSILON of movement) still fails loudly.
		const float move_tol = 0.5f * EPSILON;
		const float move_tol2 = move_tol * move_tol;

		// Bit-identical endpoint clusters with their incident face
		// unions. After MergeSnappedNodes these clusters correspond
		// one-to-one to the final nodes, so the union gathered here is
		// exactly the information the node-level snap consumed.
		struct PointFaces
		{
			std::vector<int> faces_a;
			std::vector<int> faces_b;
		};

		robin_hood::unordered_map<Eigen::Vector3f, PointFaces, Vector3fBitHash, Vector3fBitEqual> clusters;
		clusters.reserve(segments.size() * 2);

		for (const auto& seg : segments)
		{
			const Eigen::Vector3f* endpoints[2] = { &seg.p0, &seg.p1 };
			for (int e = 0; e < 2; ++e)
			{
				PointFaces& pf = clusters[*endpoints[e]];
				pf.faces_a.push_back(seg.face_a);
				pf.faces_b.push_back(seg.face_b);
			}
		}

		auto sort_unique = [](std::vector<int>& v)
			{
				std::sort(v.begin(), v.end());
				v.erase(std::unique(v.begin(), v.end()), v.end());
			};

		size_t failures = 0;
		float max_move = 0.0f;

		for (auto& kvp : clusters)
		{
			sort_unique(kvp.second.faces_a);
			sort_unique(kvp.second.faces_b);

			Eigen::Vector3f again;
			SnapNodePosition(kvp.first, kvp.second.faces_a, kvp.second.faces_b, again);

			float moved2 = (again - kvp.first).squaredNorm();
			max_move = std::max(max_move, std::sqrt(moved2));

			if (moved2 > move_tol2)
			{
				++failures;
				std::cout << "[Error] CanonicalIdempotence: canonical point ("
					<< kvp.first.x() << ", " << kvp.first.y() << ", " << kvp.first.z()
					<< ") moved by " << std::sqrt(moved2)
					<< " on re-snap (tolerance " << move_tol
					<< "). The node-level snap is not at a fixed point." << std::endl;
			}
		}

		if (Log::AtInfo())
		{
			std::cout << "[Info] CanonicalIdempotence: " << clusters.size()
				<< " canonical points checked, max re-snap movement "
				<< max_move << " (tolerance " << move_tol << ")." << std::endl;
		}

		canonical_validation_failure_count += failures;
		return 0 == failures;
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
			if (Log::AtInfo())
			{
				std::cout << "[Info] ValidateCanonicalization: all invariants hold for "
					<< segments.size() << " segments." << std::endl;
			}
		}
		else
		{
			std::cout << "[Error] ValidateCanonicalization: "
				<< canonical_validation_failure_count
				<< " invariant violations. CDT input safety is not guaranteed." << std::endl;
		}
		return ok;
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
			if (Log::AtInfo())
			{
				std::cout << "[Info] ConstraintSpacing: all per-face constraint point sets"
					<< " are safely separated (>= EPSILON) for "
					<< segments_by_face_a.size() << " A faces and "
					<< segments_by_face_b.size() << " B faces." << std::endl;
			}
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

					// Forensic distances: how far the endpoint really is from
					// its own triangle in 3D (clamped closest point). This
					// separates an in-plane tolerance miss (distance within a
					// few EPSILON) from a wrong curve_faces assignment
					// (distance well above EPSILON, the sample belongs to a
					// different triangle than the one recorded).
					Eigen::Vector3f cp_a = Distance::ClosestPointOnTriangle(*endpoints[e], a0, a1, a2);
					Eigen::Vector3f cp_b = Distance::ClosestPointOnTriangle(*endpoints[e], b0, b1, b2);
					float d_tri_a = (*endpoints[e] - cp_a).norm();
					float d_tri_b = (*endpoints[e] - cp_b).norm();

					std::cout << "[Error] Segment " << i << " endpoint " << e
						<< " (" << endpoints[e]->x() << ", " << endpoints[e]->y() << ", " << endpoints[e]->z() << ")"
						<< " not on " << (ok_a ? "" : "meshA face ") << (ok_a ? "" : std::to_string(seg.face_a))
						<< (false == ok_a && false == ok_b ? " and " : "")
						<< (ok_b ? "" : "meshB face ") << (ok_b ? "" : std::to_string(seg.face_b))
						<< " (plane dist A: " << on_a.plane_distance
						<< ", B: " << on_b.plane_distance
						<< "; dist to triangle A: " << d_tri_a
						<< ", B: " << d_tri_b << ")" << std::endl;
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
			if (Log::AtInfo())
			{
				std::cout << "[Info] ValidateSegmentEndpoints: All " << segments.size() * 2
					<< " endpoints valid. Co-refinement on this data should yield no boundary edges." << std::endl;
			}
		}
		return true;
	}

	void OperatorIntersectionLoops::CollectInputBoundaryEdges(const Mesh* mesh, const char* label)
	{
		size_t before = input_boundary_edges.size();

		for (size_t i = 0; i < mesh->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = mesh->edge_handle(static_cast<int>(i));
			if (mesh->status(eh).deleted()) continue;
			if (false == mesh->is_boundary(eh)) continue;

			OpenMesh::HalfedgeHandle heh = mesh->halfedge_handle(eh, 0);
			auto pf = mesh->point(mesh->from_vertex_handle(heh));
			auto pt = mesh->point(mesh->to_vertex_handle(heh));

			InputBoundaryEdge be;
			be.a = Eigen::Vector3f(pf[0], pf[1], pf[2]);
			be.b = Eigen::Vector3f(pt[0], pt[1], pt[2]);
			input_boundary_edges.push_back(be);
		}

		if (Log::AtInfo())
		{
			std::cout << "[Info] CollectInputBoundaryEdges(" << label << "): "
				<< (input_boundary_edges.size() - before) << " boundary edges." << std::endl;
		}
	}

	float OperatorIntersectionLoops::DistanceToInputBoundary(const Eigen::Vector3f& p) const
	{
		if (input_boundary_edges.empty())
		{
			return std::numeric_limits<float>::max();
		}

		float best2 = std::numeric_limits<float>::max();
		for (const auto& e : input_boundary_edges)
		{
			float d2 = Distance::PointToLineSegmentSquared(p, e.a, e.b);
			if (d2 < best2) best2 = d2;
		}
		return std::sqrt(best2);
	}

	bool OperatorIntersectionLoops::ValidateOpenChains()
	{
		open_chain_count = 0;
		size_t invalid = 0;

		// Canonicalization may move an endpoint by up to EPSILON off the
		// border it geometrically lies on, so 2 * EPSILON is the correct
		// acceptance distance, consistent with ValidateSegmentEndpoints.
		const float tol = 2.0f * EPSILON;

		for (size_t li = 0; li < loops.size(); ++li)
		{
			const IntersectionLoop& loop = loops[li];
			if (loop.closed) continue;

			++open_chain_count;

			if (loop.points.size() < 2)
			{
				++invalid;
				std::cout << "[Error] OpenChains: curve " << li
					<< " is open with fewer than 2 points." << std::endl;
				continue;
			}

			const Eigen::Vector3f& head = loop.points.front();
			const Eigen::Vector3f& tail = loop.points.back();

			float head_dist = DistanceToInputBoundary(head);
			float tail_dist = DistanceToInputBoundary(tail);
			bool head_ok = head_dist <= tol;
			bool tail_ok = tail_dist <= tol;

			if (head_ok && tail_ok)
			{
				if (Log::AtInfo())
				{
					std::cout << "[Info] OpenChains: curve " << li << " ("
						<< loop.segment_indices.size() << " segments) ends on the"
						<< " input border at both ends: accepted." << std::endl;
				}
				continue;
			}

			++invalid;
			std::cout << "[Error] OpenChains: curve " << li << " ("
				<< loop.segment_indices.size() << " segments) ends in the"
				<< " middle of the surfaces. head ("
				<< head.x() << ", " << head.y() << ", " << head.z()
				<< ") dist to input border " << head_dist << ", tail ("
				<< tail.x() << ", " << tail.y() << ", " << tail.z()
				<< ") dist " << tail_dist
				<< ". This means missed intersection segments, not an open input." << std::endl;

			// Forensics for the first few failures only, to keep the log
			// readable when many chains break at once.
			if (invalid <= 4)
			{
				DumpEndpointNeighborhood(head, "head");
				if (false == (tail.x() == head.x() && tail.y() == head.y() && tail.z() == head.z()))
				{
					DumpEndpointNeighborhood(tail, "tail");
				}
			}
		}

		if (open_chain_count > 0 && 0 == invalid)
		{
			if (Log::AtInfo())
			{
				std::cout << "[Info] OpenChains: " << open_chain_count
					<< " open chains, all ending on input borders." << std::endl;
			}
		}

		return 0 == invalid;
	}

	void OperatorIntersectionLoops::DumpEndpointNeighborhood(const Eigen::Vector3f& p, const char* tag) const
	{
		const float radius = 10.0f * EPSILON;
		const float radius2 = radius * radius;

		// Full precision is required here: the differences being hunted
		// are a few EPSILON and vanish at the default 6 significant
		// digits.
		std::cout << std::setprecision(10);
		if (Log::Diag())
		{
			std::cout << "[Debug] Neighborhood(" << tag << ") of ("
				<< p.x() << ", " << p.y() << ", " << p.z()
				<< "), radius " << radius << ":" << std::endl;
		}

		size_t exact_degree = 0;
		size_t found = 0;

		for (size_t i = 0; i < segments.size(); ++i)
		{
			const IntersectionSegment& s = segments[i];
			const Eigen::Vector3f* endpoints[2] = { &s.p0, &s.p1 };

			for (int e = 0; e < 2; ++e)
			{
				float d2 = (*endpoints[e] - p).squaredNorm();
				if (d2 > radius2) continue;

				bool exact = endpoints[e]->x() == p.x()
					&& endpoints[e]->y() == p.y()
					&& endpoints[e]->z() == p.z();
				if (exact) ++exact_degree;

				++found;
				if (Log::Diag())
				{
					std::cout << "[Debug]   segment " << i << " endpoint " << e
						<< " (" << endpoints[e]->x() << ", " << endpoints[e]->y()
						<< ", " << endpoints[e]->z() << ") dist " << std::sqrt(d2)
						<< (exact ? " EXACT" : "")
						<< " faceA " << s.face_a << " faceB " << s.face_b
						<< " len " << (s.p0 - s.p1).norm() << std::endl;
				}
			}
		}

		if (Log::Diag())
		{
			std::cout << "[Debug]   " << found << " endpoints in radius, "
				<< exact_degree << " bit-identical (node degree)." << std::endl;
		}
		std::cout << std::setprecision(6);
	}

	bool OperatorIntersectionLoops::FindFacePathBFS(
		OpenMesh::FaceHandle fa,
		OpenMesh::FaceHandle fb,
		std::vector<OpenMesh::FaceHandle>& out_path) const
	{
		out_path.clear();

		if (false == fa.is_valid() || false == fb.is_valid()) return false;

		if (fa == fb)
		{
			out_path.push_back(fa);
			return true;
		}

		// Breadth-first search over face adjacency through shared
		// non-boundary edges. The predecessor map reconstructs the face
		// sequence fa ... fb, which is guaranteed to be a chain of
		// edge-adjacent faces regardless of how the straight chord between
		// the two curve samples behaves over a curved surface.
		robin_hood::unordered_map<int, int> prev;
		prev[fa.idx()] = -1;

		std::deque<OpenMesh::FaceHandle> queue;
		queue.push_back(fa);

		const size_t max_visited = meshA->n_faces() + 1;
		size_t visited = 0;
		bool found = false;

		while (false == queue.empty())
		{
			if (++visited > max_visited) return false;

			OpenMesh::FaceHandle cur = queue.front();
			queue.pop_front();

			if (cur == fb)
			{
				found = true;
				break;
			}

			for (auto fh_it = meshA->cfh_iter(cur); fh_it.is_valid(); ++fh_it)
			{
				OpenMesh::HalfedgeHandle opp = meshA->opposite_halfedge_handle(*fh_it);
				if (meshA->is_boundary(opp)) continue;

				OpenMesh::FaceHandle nfh = meshA->face_handle(opp);
				if (false == nfh.is_valid()) continue;
				if (meshA->status(nfh).deleted()) continue;
				if (prev.find(nfh.idx()) != prev.end()) continue;

				prev[nfh.idx()] = cur.idx();
				queue.push_back(nfh);
			}
		}

		if (false == found) return false;

		// Reconstruct fa ... fb by walking predecessors back from fb.
		std::vector<OpenMesh::FaceHandle> reversed;
		int cur_idx = fb.idx();
		const size_t max_steps = meshA->n_faces() + 1;
		size_t steps = 0;

		while (cur_idx != -1)
		{
			reversed.push_back(meshA->face_handle(cur_idx));
			if (++steps > max_steps) return false;

			auto it = prev.find(cur_idx);
			if (it == prev.end()) return false;
			cur_idx = it->second;
		}

		out_path.assign(reversed.rbegin(), reversed.rend());
		return out_path.size() >= 1 && out_path.front() == fa && out_path.back() == fb;
	}

	bool OperatorIntersectionLoops::CrossingOnSharedEdge(
		OpenMesh::HalfedgeHandle shared_he,
		const Eigen::Vector3f& p_from,
		const Eigen::Vector3f& p_to,
		Eigen::Vector3f& out_point) const
	{
		// One crossing list per mesh edge, matched by direct distance rather
		// than by a quantized slot. The previous slot-keyed cache split the
		// same physical crossing across two adjacent slots when two curve
		// spans hit the edge at almost the same parameter but on opposite
		// sides of a slot boundary. That produced two crossings a few EPSILON
		// apart on the same edge, which co-refinement then carved as a tiny
		// T-junction (a NEW boundary edge) or, when welding fused them after
		// the face was already split, a zero-length edge that add_face
		// rejected. Comparing the new crossing's distance to every crossing
		// already registered on this edge removes the slot boundary entirely:
		// within 2 * EPSILON it is the same point and is reused bit-for-bit;
		// beyond that it is a genuinely different crossing and is kept apart.
		OpenMesh::EdgeHandle eh = meshA->edge_handle(shared_he);
		int edge_idx = eh.idx();

		Eigen::Vector3f e0(meshA->point(meshA->from_vertex_handle(shared_he)).data());
		Eigen::Vector3f e1(meshA->point(meshA->to_vertex_handle(shared_he)).data());

		Eigen::Vector3f edge = e1 - e0;
		float edge_len2 = edge.squaredNorm();
		if (edge_len2 < 1e-20f) return false;

		Eigen::Vector3f d1 = edge;
		Eigen::Vector3f d2 = p_to - p_from;
		Eigen::Vector3f r = e0 - p_from;

		float a = d1.dot(d1);
		float e = d2.dot(d2);
		float f = d2.dot(r);

		float s;
		if (e < 1e-20f)
		{
			s = -d1.dot(r) / a;
		}
		else
		{
			float c = d1.dot(r);
			float b = d1.dot(d2);
			float denom = a * e - b * b;
			if (std::abs(denom) > 1e-20f)
			{
				s = (b * f - c * e) / denom;
			}
			else
			{
				Eigen::Vector3f mid = 0.5f * (p_from + p_to);
				s = (mid - e0).dot(edge) / edge_len2;
			}
		}

		s = std::max(0.0f, std::min(1.0f, s));
		Eigen::Vector3f crossing = e0 + edge * s;

		// Reuse an existing crossing on this edge if the new one is within
		// 2 * EPSILON of it. The nearest existing crossing is chosen so the
		// decision is deterministic when several are registered.
		std::vector<Eigen::Vector3f>& existing = edge_crossing_cache[edge_idx];

		int best = -1;
		float best_d2 = (2.0f * EPSILON) * (2.0f * EPSILON);
		for (int i = 0; i < static_cast<int>(existing.size()); ++i)
		{
			float d2i = (existing[i] - crossing).squaredNorm();
			if (d2i < best_d2)
			{
				best_d2 = d2i;
				best = i;
			}
		}

		if (best >= 0)
		{
			out_point = existing[best];
			return true;
		}

		existing.push_back(crossing);
		out_point = crossing;
		return true;
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
			if (Log::AtInfo())
			{
				std::cout << "[Info] OperatorCoRefine: no intersection segments, nothing to refine." << std::endl;
			}
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

		if (Log::AtInfo())
		{
			std::cout << "[Info] RefineMesh(" << mesh_label << "): " << refined_faces
				<< " faces refined, " << failed_faces << " failures, "
				<< indices.size() << " triangles total, "
				<< input_boundary.size() << " input boundary edges, "
				<< out_new_boundary_edge_count << " NEW boundary edges after rebuild." << std::endl;
		}

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

		Eigen::Vector3f e1 = (c1 - c0).normalized();
		Eigen::Vector3f e2 = n.cross(e1);

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

			// Forensics: two constraint points landing on the SAME face edge
			// only slightly more than EPSILON apart split that edge into a
			// sub-segment the neighbour across the edge may not reproduce,
			// leaving a T-junction that surfaces as a NEW boundary edge after
			// the rebuild. Report any such close pair on this edge.
			for (size_t i = 1; i < on_edge.size(); ++i)
			{
				const Eigen::Vector3f& pa = pts3[on_edge[i - 1].second];
				const Eigen::Vector3f& pb = pts3[on_edge[i].second];
				float gap = (pa - pb).norm();
				if (gap < 10.0f * EPSILON)
				{
					std::cout << std::setprecision(10);
					if (Log::Diag())
					{
						std::cout << "[Debug] TriangulateFace face " << face_index
							<< " edge " << eidx << ": two on-edge constraint points "
							<< gap << " apart at (" << pa.x() << ", " << pa.y() << ", " << pa.z()
							<< ") and (" << pb.x() << ", " << pb.y() << ", " << pb.z() << ")" << std::endl;
					}
					std::cout << std::setprecision(6);
				}
			}

			int prev = corner_edges[eidx][0];
			for (const auto& tp : on_edge)
			{
				add_edge(prev, tp.second);
				prev = tp.second;
			}
			add_edge(prev, corner_edges[eidx][1]);
		}

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

	// File-local: readable classification names for logs
	static const char* FaceSideNameString(OperatorBoolean::FaceSide s)
	{
		switch (s)
		{
		case OperatorBoolean::FaceSide::Inside: return "Inside";
		case OperatorBoolean::FaceSide::Outside: return "Outside";
		case OperatorBoolean::FaceSide::OnSurfaceSame: return "OnSurfaceSame";
		case OperatorBoolean::FaceSide::OnSurfaceOpposite: return "OnSurfaceOpposite";
		default: return "Unknown";
		}
	}

	// File-local: readable boolean operation names for logs
	static const char* BooleanTypeNameString(OperatorBoolean::Type t)
	{
		switch (t)
		{
		case OperatorBoolean::Intersection: return "Intersection";
		case OperatorBoolean::Union: return "Union";
		case OperatorBoolean::Difference: return "Difference";
		default: return "UnknownType";
		}
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

		// Strategy selection from the actual topology of the inputs.
		// Solid booleans are only defined between closed volumes: an open
		// shell has no volume, ray parity against it is undefined, and
		// stitching its patches with the other mesh along the seam cannot
		// produce a volume boundary.
		size_t boundary_a = CountBoundaryEdges(meshA);
		size_t boundary_b = CountBoundaryEdges(meshB);
		bool a_closed = (0 == boundary_a);
		bool b_closed = (0 == boundary_b);

		if (Log::AtInfo())
		{
			std::cout << "[Info] OperatorBoolean: meshA is "
				<< (a_closed ? "closed" : "open") << " (" << boundary_a
				<< " boundary edges), meshB is "
				<< (b_closed ? "closed" : "open") << " (" << boundary_b
				<< " boundary edges)." << std::endl;
		}

		if (a_closed && b_closed)
		{
			if (false == BuildSeamEdgeFlags(meshA, data_a, "meshA")) return false;
			if (false == BuildSeamEdgeFlags(meshB, data_b, "meshB")) return false;

			if (false == ValidateSeamIntegrity(meshA, data_a, "meshA")) return false;
			if (false == ValidateSeamIntegrity(meshB, data_b, "meshB")) return false;

			if (false == BuildFacePatches(meshA, data_a, "meshA")) return false;
			if (false == BuildFacePatches(meshB, data_b, "meshB")) return false;

			if (false == ClassifyPatches(meshA, meshB, data_a, "meshA")) return false;
			if (false == ClassifyPatches(meshB, meshA, data_b, "meshB")) return false;

			if (false == ValidatePatchAdjacency(meshA, data_a, "meshA")) return false;
			if (false == ValidatePatchAdjacency(meshB, data_b, "meshB")) return false;

			ReportPatchStatistics(meshA, data_a, "meshA");
			ReportPatchStatistics(meshB, data_b, "meshB");

			return AssembleSolidBoolean();
		}

		if (false == a_closed && false == b_closed)
		{
			std::cout << "[Error] OperatorBoolean: both inputs are open."
				<< " At least one closed solid is required." << std::endl;
			return false;
		}

		if (Union == type)
		{
			std::cout << "[Error] OperatorBoolean: Union requires two closed solids."
				<< " An open input contributes no volume." << std::endl;
			return false;
		}

		// Exactly one mesh is open: the result is that mesh trimmed by
		// the closed solid. All structures are built and validated on the
		// open mesh only; the solid serves as the classification
		// reference, where ray parity is valid.
		Mesh* open_mesh = nullptr;
		const Mesh* solid = nullptr;
		MeshSideData* data = nullptr;
		const char* open_label = nullptr;

		if (false == a_closed)
		{
			// Intersection: the part of A inside B.
			// Difference: the part of A outside B.
			open_mesh = meshA;
			solid = meshB;
			data = &data_a;
			open_label = "meshA";
		}
		else
		{
			// meshB is the open one. Intersection is symmetric, so
			// trimming B against A is well defined. Difference (A minus
			// B) would need the volume of B, which an open B lacks.
			if (Intersection != type)
			{
				std::cout << "[Error] OperatorBoolean: Difference requires a closed meshB."
					<< " An open meshB has no volume to subtract." << std::endl;
				return false;
			}
			open_mesh = meshB;
			solid = meshA;
			data = &data_b;
			open_label = "meshB";
		}

		if (false == BuildSeamEdgeFlags(open_mesh, *data, open_label)) return false;
		if (false == ValidateSeamIntegrity(open_mesh, *data, open_label)) return false;
		if (false == BuildFacePatches(open_mesh, *data, open_label)) return false;
		if (false == ClassifyPatches(open_mesh, solid, *data, open_label)) return false;
		if (false == ValidatePatchAdjacency(open_mesh, *data, open_label)) return false;

		ReportPatchStatistics(open_mesh, *data, open_label);

		return AssembleOpenMeshTrim(open_mesh, *data, open_label);
	}

	bool OperatorBoolean::AssembleSolidBoolean()
	{
		// Face selection per boolean type, including coplanar overlap
		// rules. A coplanar overlap region exists identically on both
		// meshes, so it is taken from mesh A only, never from both:
		// - Intersection and Union keep Same regions: the coincident
		//   surface bounds the common volume and the union there.
		// - Difference drops Same regions (the subtracted volume ends
		//   exactly at A's surface, opening it there) and keeps Opposite
		//   regions of A (B only touches A from the other side, A's
		//   surface survives).
		// - Opposite regions are zero-volume face contacts for
		//   Intersection and interior walls for Union: dropped.
		// For Difference the kept B faces bound removed volume, so their
		// winding flips to stay outward.
		bool a_inside = false;
		bool a_outside = false;
		bool a_same = false;
		bool a_opposite = false;
		bool b_inside = false;
		bool b_outside = false;
		bool flip_b = false;

		switch (type)
		{
		case Intersection:
			a_inside = true;
			a_same = true;
			b_inside = true;
			break;
		case Union:
			a_outside = true;
			a_same = true;
			b_outside = true;
			break;
		case Difference:
			a_outside = true;
			a_opposite = true;
			b_inside = true;
			flip_b = true;
			break;
		}

		std::vector<Eigen::Vector3f> soup;
		soup.reserve((meshA->n_faces() + meshB->n_faces()) * 3);

		CollectFacesForBoolean(meshA, data_a, a_inside, a_outside, a_same, a_opposite, false, soup);
		size_t kept_a = soup.size() / 3;
		CollectFacesForBoolean(meshB, data_b, b_inside, b_outside, false, false, flip_b, soup);
		size_t kept_b = soup.size() / 3 - kept_a;

		// The operation and the per-side selection are logged so any run
		// is identifiable from its log alone: different operations on the
		// same data can coincidentally produce equal triangle totals.
		if (Log::AtInfo())
		{
			std::cout << "[Info] OperatorBoolean: assembling " << BooleanTypeNameString(type)
				<< ": kept " << kept_a << " faces from meshA, "
				<< kept_b << " faces from meshB." << std::endl;
		}

		std::vector<Eigen::Vector3f> points;
		std::vector<Eigen::Vector3i> indices;
		WeldTriangleSoup(soup, points, indices);

		if (indices.empty())
		{
			std::cout << "[Warning] OperatorBoolean: result is empty." << std::endl;
		}

		// Index-level manifoldness check BEFORE the halfedge build, which
		// could silently drop non-manifold faces and hide the defect.
		bool soup_ok = ValidateResultSoup(points, indices, "result");

		result->clear();
		result->Build(points, indices);

		result_boundary_edge_count = CountBoundaryEdges(result);

		if (Log::AtInfo())
		{
			std::cout << "[Info] OperatorBoolean: " << indices.size() << " result triangles, "
				<< result_boundary_edge_count << " boundary edges in result." << std::endl;
		}

		return soup_ok && 0 == result_boundary_edge_count;
	}

	bool OperatorBoolean::AssembleOpenMeshTrim(Mesh* open_mesh, const MeshSideData& data, const char* open_label)
	{
		// The open mesh's own border is a property of the input and stays
		// legitimate in the result. It is recorded before assembly so the
		// result validation can separate it from pipeline-created holes.
		std::vector<BoundaryEdge> input_boundary;
		CollectInputBoundaryEdges(open_mesh, input_boundary);

		// Intersection keeps the part of the open mesh inside the solid,
		// including pieces lying on the solid surface facing the same
		// way. Difference keeps the outside part, including pieces lying
		// on the surface facing the opposite way.
		bool keep_inside = (Intersection == type);
		bool keep_outside = (Difference == type);
		bool keep_same = keep_inside;
		bool keep_opposite = keep_outside;

		std::vector<Eigen::Vector3f> soup;
		soup.reserve(open_mesh->n_faces() * 3);
		CollectFacesForBoolean(open_mesh, data, keep_inside, keep_outside, keep_same, keep_opposite, false, soup);

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

		if (Log::AtInfo())
		{
			std::cout << "[Info] OperatorBoolean: " << BooleanTypeNameString(type)
				<< " trim kept " << indices.size()
				<< " triangles from " << open_label << ", "
				<< result_boundary_edge_count << " boundary edges, "
				<< invalid << " invalid." << std::endl;
		}

		return 0 == invalid;
	}

	bool OperatorBoolean::BuildSeamEdgeFlags(const Mesh* mesh, MeshSideData& data, const char* label) const
	{
		const auto& segments = loop_op->GetSegments();

		data.edge_is_seam.assign(mesh->n_edges(), 0);

		// The seam is identified BIT-EXACTLY: an edge is a seam edge if
		// and only if its endpoint pair equals the canonical endpoint
		// pair of some intersection segment. Canonicalization keeps
		// constraint coordinates bit-identical through carving and
		// welding, so no geometric tolerance is needed here.
		//
		// The previous tolerance test (endpoints and midpoint within
		// EPSILON of a segment) was proven wrong on pinch contacts: near
		// collinear canonical points produce CDT sliver triangles whose
		// free third edge hugs the segment within EPSILON. Those free
		// edges were flagged as seam, doubling the chain and breaking
		// degree parity.
		struct SegKey
		{
			Eigen::Vector3f a;
			Eigen::Vector3f b;
		};
		struct SegKeyHash
		{
			size_t operator()(const SegKey& k) const
			{
				Vector3fBitHash h;
				return h(k.a) * 1000003ull ^ h(k.b);
			}
		};
		struct SegKeyEqual
		{
			bool operator()(const SegKey& x, const SegKey& y) const
			{
				Vector3fBitEqual eq;
				return eq(x.a, y.a) && eq(x.b, y.b);
			}
		};

		auto less_xyz = [](const Eigen::Vector3f& a, const Eigen::Vector3f& b) -> bool
			{
				if (a.x() != b.x()) return a.x() < b.x();
				if (a.y() != b.y()) return a.y() < b.y();
				return a.z() < b.z();
			};

		auto make_key = [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b) -> SegKey
			{
				SegKey key;
				if (less_xyz(a, b))
				{
					key.a = a;
					key.b = b;
				}
				else
				{
					key.a = b;
					key.b = a;
				}
				return key;
			};

		// Value: matched flag, so every segment pair can be verified to
		// exist as a mesh edge afterwards. Duplicate segments (same
		// canonical pair from several face pairs) collapse into one entry,
		// which is correct: the mesh carves that piece once.
		robin_hood::unordered_map<SegKey, char, SegKeyHash, SegKeyEqual> segment_pairs;
		segment_pairs.reserve(segments.size());
		for (const auto& s : segments)
		{
			segment_pairs[make_key(s.p0, s.p1)] = 0;
		}

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

			auto it = segment_pairs.find(make_key(a, b));
			if (it == segment_pairs.end()) continue;

			it->second = 1;
			data.edge_is_seam[i] = 1;
			++seam_count;
			seam_length += static_cast<double>((b - a).norm());
		}

		// Reverse check: every segment must exist in the mesh as one edge
		// with exactly its canonical endpoints. A miss means the carve
		// did not deliver this constraint (or CDT split it), and the seam
		// would have a gap: fail loudly here, before flood fill.
		size_t missing = 0;
		for (const auto& kvp : segment_pairs)
		{
			if (0 != kvp.second) continue;
			++missing;
			if (missing <= 10)
			{
				std::cout << "[Error] BuildSeamEdgeFlags(" << label << "): segment ("
					<< kvp.first.a.x() << ", " << kvp.first.a.y() << ", " << kvp.first.a.z()
					<< ") - ("
					<< kvp.first.b.x() << ", " << kvp.first.b.y() << ", " << kvp.first.b.z()
					<< ") has no matching mesh edge." << std::endl;
			}
		}
		if (missing > 10)
		{
			std::cout << "[Error] BuildSeamEdgeFlags(" << label << "): "
				<< (missing - 10) << " more missing segments not shown." << std::endl;
		}

		if (Log::AtInfo())
		{
			std::cout << "[Info] BuildSeamEdgeFlags(" << label << "): " << seam_count
				<< " seam edges (bit-exact, " << segment_pairs.size()
				<< " unique segment pairs), total length " << seam_length << "." << std::endl;
		}

		return 0 == missing;
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

		// Forensics for odd-degree vertices: every incident edge is
		// dumped with full precision, its seam flag, the segment its
		// geometry matched, and the exact distances of both endpoints to
		// that segment. This separates the real carved seam chain from a
		// ghost chain (split original mesh edges hugging the curve within
		// EPSILON) that the tolerance-based seam test wrongly flags.
		auto dump_vertex = [&](int vidx, int degree)
			{
				OpenMesh::VertexHandle vh = mesh->vertex_handle(vidx);
				auto p = mesh->point(vh);
				Eigen::Vector3f a(p[0], p[1], p[2]);

				std::cout << std::setprecision(10);
				if (Log::Diag())
				{
					std::cout << "[Debug] SeamVertex(" << label << ") ("
						<< a.x() << ", " << a.y() << ", " << a.z()
						<< ") seam degree " << degree << ", incident edges:" << std::endl;
				}

				for (auto voh_it = mesh->cvoh_iter(vh); voh_it.is_valid(); ++voh_it)
				{
					OpenMesh::EdgeHandle eh = mesh->edge_handle(*voh_it);
					auto q = mesh->point(mesh->to_vertex_handle(*voh_it));
					Eigen::Vector3f b(q[0], q[1], q[2]);

					bool is_seam = 0 != data.edge_is_seam[eh.idx()];

					Eigen::Vector3f mid = (a + b) * 0.5f;
					int match = -1;
					float dist_a = -1.0f;
					float dist_b = -1.0f;
					float dist_m = -1.0f;
					for (size_t si = 0; si < segments.size(); ++si)
					{
						const auto& s = segments[si];
						float da2 = Distance::PointToLineSegmentSquared(a, s.p0, s.p1);
						if (da2 >= EPSILON * EPSILON) continue;
						float db2 = Distance::PointToLineSegmentSquared(b, s.p0, s.p1);
						if (db2 >= EPSILON * EPSILON) continue;
						float dm2 = Distance::PointToLineSegmentSquared(mid, s.p0, s.p1);
						if (dm2 >= EPSILON * EPSILON) continue;

						match = static_cast<int>(si);
						dist_a = std::sqrt(da2);
						dist_b = std::sqrt(db2);
						dist_m = std::sqrt(dm2);
						break;
					}

					if (Log::Diag())
					{
						std::cout << "[Debug]   to (" << b.x() << ", " << b.y() << ", " << b.z()
							<< ") len " << (b - a).norm()
							<< (is_seam ? " SEAM" : " ----");
						if (match >= 0)
						{
							std::cout << " runs along segment " << match
								<< " (faceA " << segments[match].face_a
								<< ", faceB " << segments[match].face_b
								<< ", len " << (segments[match].p0 - segments[match].p1).norm()
								<< ") dists " << dist_a << " / " << dist_m << " / " << dist_b;
						}
						std::cout << std::endl;
					}
				}
				std::cout << std::setprecision(6);
			};

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

		size_t odd_dumped = 0;
		for (size_t i = 0; i < seam_degree.size(); ++i)
		{
			if (0 != (seam_degree[i] % 2))
			{
				++failures;
				auto p = mesh->point(mesh->vertex_handle(static_cast<int>(i)));
				std::cout << "[Error] SeamIntegrity(" << label << "): vertex ("
					<< p[0] << ", " << p[1] << ", " << p[2] << ") has odd seam degree "
					<< seam_degree[i] << ": the seam is broken here." << std::endl;

				if (odd_dumped < 4)
				{
					++odd_dumped;
					dump_vertex(static_cast<int>(i), seam_degree[i]);
				}
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
			if (Log::AtInfo())
			{
				std::cout << "[Info] SeamIntegrity(" << label << "): seam is closed and complete"
					<< " (length " << seam_length << " vs expected " << expected << ")." << std::endl;
			}
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

		if (Log::AtInfo())
		{
			std::cout << "[Info] BuildFacePatches(" << label << "): " << data.patch_count
				<< " patches." << std::endl;
		}
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

		size_t inside = 0;
		size_t outside = 0;
		size_t on_same = 0;
		size_t on_opposite = 0;

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

			Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
			float n_len = n.norm();
			if (n_len < 1e-12f)
			{
				std::cout << "[Error] ClassifyPatches(" << label << "): patch " << p
					<< " representative face is degenerate." << std::endl;
				return false;
			}
			n /= n_len;

			FaceSide side = RobustPointInMesh(centroid, n, other);
			if (FaceSide::Unknown == side)
			{
				std::cout << "[Error] ClassifyPatches(" << label << "): patch " << p
					<< " could not be classified." << std::endl;
				return false;
			}

			data.patch_side[p] = side;

			switch (side)
			{
			case FaceSide::Inside: ++inside; break;
			case FaceSide::Outside: ++outside; break;
			case FaceSide::OnSurfaceSame: ++on_same; break;
			case FaceSide::OnSurfaceOpposite: ++on_opposite; break;
			default: break;
			}
		}

		if (Log::AtInfo())
		{
			std::cout << "[Info] ClassifyPatches(" << label << "): " << inside << " inside, "
				<< outside << " outside, " << on_same << " on-surface-same, "
				<< on_opposite << " on-surface-opposite patches." << std::endl;
		}
		return true;
	}

	OperatorBoolean::FaceSide OperatorBoolean::RobustPointInMesh(const Eigen::Vector3f& p, const Eigen::Vector3f& query_normal, const Mesh* other) const
	{
		// On-surface check first: a point on the other mesh cannot be
		// classified by ray parity at all. When the point does lie on the
		// other surface, the relative normal orientation decides between
		// the Same and Opposite on-surface classifications.
		const Eigen::Vector3f pad(EPSILON, EPSILON, EPSILON);
		std::vector<int> near_faces;
		other->QueryOverlappingFaces(p - pad, p + pad, near_faces);

		float best_abs_dot = 0.0f;
		float best_dot = 0.0f;
		bool on_surface = false;

		for (int f : near_faces)
		{
			OpenMesh::FaceHandle fh = other->face_handle(f);
			if (other->status(fh).deleted()) continue;

			Eigen::Vector3f v0, v1, v2;
			other->GetFaceVertices(fh, v0, v1, v2);
			if (Intersection::PointToTriangle(p, v0, v1, v2).type == Intersection::PointToTriangleType::Outside) continue;

			on_surface = true;

			Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
			float n_len = n.norm();
			if (n_len < 1e-12f) continue;
			n /= n_len;

			float d = n.dot(query_normal);
			if (std::abs(d) > best_abs_dot)
			{
				best_abs_dot = std::abs(d);
				best_dot = d;
			}
		}

		if (on_surface)
		{
			// Nearly perpendicular normals mean the point sits on the
			// intersection curve itself, where Same/Opposite is undefined.
			// Representative centroids must not land there: report it.
			if (best_abs_dot < 0.5f)
			{
				std::cout << "[Error] RobustPointInMesh: on-surface point ("
					<< p.x() << ", " << p.y() << ", " << p.z()
					<< ") has ambiguous orientation (|dot| " << best_abs_dot << ")." << std::endl;
				return FaceSide::Unknown;
			}
			return (best_dot > 0.0f) ? FaceSide::OnSurfaceSame : FaceSide::OnSurfaceOpposite;
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
		// Crossing the intersection curve changes the relation to the
		// other mesh, so the two patches incident to a seam edge must
		// carry DIFFERENT classifications. This generalizes the pure
		// transversal rule (Inside/Outside flip) to the OnSurface cases:
		// entering or leaving a coplanar overlap region also changes the
		// classification. Identical sides across a seam edge prove either
		// a flood fill leak or a wrong classification verdict, and the
		// boolean must not proceed.
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
						<< " has the same classification ("
						<< FaceSideNameString(s0) << ") on both sides." << std::endl;
				}
			}
		}

		if (0 == failures)
		{
			if (Log::AtInfo())
			{
				std::cout << "[Info] PatchAdjacency(" << label << "): every seam edge separates"
					<< " differently classified patches." << std::endl;
			}
			return true;
		}

		std::cout << "[Error] PatchAdjacency(" << label << "): " << failures
			<< " seam edges violate the classification change rule." << std::endl;
		return false;
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
				if (Log::AtInfo())
				{
					std::cout << "[Info] PatchStats(" << label << "): "
						<< (data.patch_count - report_limit) << " more patches not shown." << std::endl;
				}
				break;
			}
			if (Log::AtInfo())
			{
				std::cout << "[Info] PatchStats(" << label << "): patch " << p
					<< " has " << patch_faces[p] << " faces, touches "
					<< patch_seam_edges[p] << " seam edge sides, side "
					<< FaceSideNameString(data.patch_side[p]) << "." << std::endl;
			}
		}
	}

	void OperatorBoolean::CollectFacesForBoolean(
		const Mesh* mesh,
		const MeshSideData& data,
		bool keep_inside,
		bool keep_outside,
		bool keep_on_same,
		bool keep_on_opposite,
		bool flip_winding,
		std::vector<Eigen::Vector3f>& out_soup) const
	{
		for (size_t i = 0; i < mesh->n_faces(); ++i)
		{
			OpenMesh::FaceHandle fh = mesh->face_handle(static_cast<int>(i));
			if (mesh->status(fh).deleted()) continue;

			int patch = data.face_patch[i];
			if (patch < 0) continue;

			FaceSide side = data.patch_side[patch];
			bool keep =
				(FaceSide::Inside == side && keep_inside) ||
				(FaceSide::Outside == side && keep_outside) ||
				(FaceSide::OnSurfaceSame == side && keep_on_same) ||
				(FaceSide::OnSurfaceOpposite == side && keep_on_opposite);
			if (false == keep) continue;

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

	bool OperatorBoolean::ValidateResultSoup(
		const std::vector<Eigen::Vector3f>& points,
		const std::vector<Eigen::Vector3i>& indices,
		const char* label) const
	{
		// Probe radius 100 * EPSILON: external tools weld at coarser
		// tolerances than this pipeline, so near pairs and short edges up
		// to that scale explain non-manifold verdicts seen only there.
		return ValidateTriangleSoup(points, indices, label, 100.0f * EPSILON);
	}

	// File-local: closest point on a triangle to p (Ericson's method).
	static Eigen::Vector3f ClosestPointOnTriangle(
		const Eigen::Vector3f& p,
		const Eigen::Vector3f& a,
		const Eigen::Vector3f& b,
		const Eigen::Vector3f& c)
	{
		Eigen::Vector3f ab = b - a;
		Eigen::Vector3f ac = c - a;
		Eigen::Vector3f ap = p - a;

		float d1 = ab.dot(ap);
		float d2 = ac.dot(ap);
		if (d1 <= 0.0f && d2 <= 0.0f) return a;

		Eigen::Vector3f bp = p - b;
		float d3 = ab.dot(bp);
		float d4 = ac.dot(bp);
		if (d3 >= 0.0f && d4 <= d3) return b;

		float vc = d1 * d4 - d3 * d2;
		if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
		{
			float v = d1 / (d1 - d3);
			return a + v * ab;
		}

		Eigen::Vector3f cp = p - c;
		float d5 = ab.dot(cp);
		float d6 = ac.dot(cp);
		if (d6 >= 0.0f && d5 <= d6) return c;

		float vb = d5 * d2 - d1 * d6;
		if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
		{
			float w = d2 / (d2 - d6);
			return a + w * ac;
		}

		float va = d3 * d6 - d5 * d4;
		if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
		{
			float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
			return b + w * (c - b);
		}

		float denom = 1.0f / (va + vb + vc);
		float v = vb * denom;
		float w = vc * denom;
		return a + ab * v + ac * w;
	}

	OperatorRemesh::OperatorRemesh(
		Mesh* target,
		float target_edge_length,
		int iterations,
		float feature_angle_degree,
		const PassFlags& passes)
		: meshTarget(target)
		, target_edge_length(target_edge_length)
		, edge_high(4.0f / 3.0f * target_edge_length)
		, edge_low(4.0f / 5.0f * target_edge_length)
		, iterations(iterations)
		, feature_angle_degree(feature_angle_degree)
		, passes(passes)
	{
	}

	void OperatorRemesh::BuildReferenceSnapshot()
	{
		std::vector<Eigen::Vector3f> points;
		std::vector<Eigen::Vector3i> indices;
		ExtractMeshSoup(meshTarget, points, indices);

		reference_mesh.clear();
		reference_mesh.Build(points, indices);
	}

	void OperatorRemesh::ClassifyFeatureVertex(OpenMesh::VertexHandle vh, float corner_threshold)
	{
		Eigen::Vector3f p(meshTarget->point(vh).data());

		Eigen::Vector3f dir_first = Eigen::Vector3f::Zero();
		Eigen::Vector3f dir_second = Eigen::Vector3f::Zero();
		int feature_count = 0;

		for (auto voh_it = meshTarget->voh_iter(vh); voh_it.is_valid(); ++voh_it)
		{
			OpenMesh::EdgeHandle e = meshTarget->edge_handle(*voh_it);
			if (false == meshTarget->property(prop_edge_feature, e)) continue;

			Eigen::Vector3f pn(meshTarget->point(meshTarget->to_vertex_handle(*voh_it)).data());
			Eigen::Vector3f dir = pn - p;
			float len = dir.norm();
			if (len < 1e-12f) continue;
			dir /= len;

			if (0 == feature_count) dir_first = dir;
			else if (1 == feature_count) dir_second = dir;
			++feature_count;
		}

		int cls = VertexFree;
		if (2 == feature_count)
		{
			// Straight curve: outgoing directions nearly opposite (dot ~ -1).
			// A sharp bend pushes the dot above the corner threshold: pin it.
			float d = dir_first.dot(dir_second);
			cls = (d > corner_threshold) ? VertexCorner : VertexOnFeature;
		}
		else if (feature_count > 0)
		{
			// Endpoint (1) or junction (3+): always pinned.
			cls = VertexCorner;
		}

		meshTarget->property(prop_vertex_feature, vh) = cls;
	}

	void OperatorRemesh::DetectFeatureEdges()
	{
		feature_edge_count = 0;

		const float cos_threshold = std::cos(feature_angle_degree * static_cast<float>(M_PI) / 180.0f);
		corner_cos_threshold = -cos_threshold;

		for (size_t i = 0; i < meshTarget->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = meshTarget->edge_handle(static_cast<int>(i));
			if (meshTarget->status(eh).deleted()) continue;

			bool is_feature = false;

			if (meshTarget->is_boundary(eh))
			{
				is_feature = true;
			}
			else
			{
				OpenMesh::HalfedgeHandle h0 = meshTarget->halfedge_handle(eh, 0);
				OpenMesh::HalfedgeHandle h1 = meshTarget->halfedge_handle(eh, 1);

				Eigen::Vector3f a0, a1, a2;
				Eigen::Vector3f b0, b1, b2;
				meshTarget->GetFaceVertices(meshTarget->face_handle(h0), a0, a1, a2);
				meshTarget->GetFaceVertices(meshTarget->face_handle(h1), b0, b1, b2);

				Eigen::Vector3f n0 = (a1 - a0).cross(a2 - a0);
				Eigen::Vector3f n1 = (b1 - b0).cross(b2 - b0);
				float l0 = n0.norm();
				float l1 = n1.norm();
				if (l0 > 1e-12f && l1 > 1e-12f)
				{
					float cos_angle = (n0 / l0).dot(n1 / l1);
					if (cos_angle < cos_threshold) is_feature = true;
				}
			}

			meshTarget->property(prop_edge_feature, eh) = is_feature;
			if (is_feature) ++feature_edge_count;
		}

		for (size_t i = 0; i < meshTarget->n_vertices(); ++i)
		{
			OpenMesh::VertexHandle vh = meshTarget->vertex_handle(static_cast<int>(i));
			if (meshTarget->status(vh).deleted()) continue;
			ClassifyFeatureVertex(vh, corner_cos_threshold);
		}

		if (Log::AtInfo())
		{
			std::cout << "[Info] DetectFeatureEdges: " << feature_edge_count
				<< " feature edges at dihedral angle > " << feature_angle_degree
				<< " degrees." << std::endl;
		}
	}

	bool OperatorRemesh::ReprojectToReference(const Eigen::Vector3f& p, Eigen::Vector3f& out_point) const
	{
		float radius = std::max(target_edge_length, 10.0f * EPSILON);

		std::vector<int> candidates;
		for (int attempt = 0; attempt < 8; ++attempt)
		{
			Eigen::Vector3f pad(radius, radius, radius);
			reference_mesh.QueryOverlappingFaces(p - pad, p + pad, candidates);
			if (false == candidates.empty()) break;
			radius *= 2.0f;
		}

		if (candidates.empty())
		{
			out_point = p;
			return false;
		}

		// Nearest reference face by clamped closest-point distance.
		float best_d2 = std::numeric_limits<float>::max();
		Eigen::Vector3f best_cp = p;
		Eigen::Vector3f best_n = Eigen::Vector3f::Zero();
		Eigen::Vector3f best_v0 = Eigen::Vector3f::Zero();
		bool found = false;

		for (int f : candidates)
		{
			OpenMesh::FaceHandle fh = reference_mesh.face_handle(f);
			if (reference_mesh.status(fh).deleted()) continue;

			Eigen::Vector3f v0, v1, v2;
			reference_mesh.GetFaceVertices(fh, v0, v1, v2);

			Eigen::Vector3f cp = ClosestPointOnTriangle(p, v0, v1, v2);
			float d2 = (cp - p).squaredNorm();
			if (d2 < best_d2)
			{
				Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
				float nl = n.norm();
				if (nl < 1e-12f) continue;

				best_d2 = d2;
				best_cp = cp;
				best_n = n / nl;
				best_v0 = v0;
				found = true;
			}
		}

		if (false == found)
		{
			out_point = p;
			return false;
		}

		// Decide flat vs curved from the local reference normals. A flat
		// region (the glyph walls and faces) has all nearby reference
		// faces sharing the nearest face's normal: plane projection there
		// keeps the wall dead flat and stops the ripple. A curved region
		// (the round pillar of an 'l', the sine-wave top) has normals that
		// fan out: plane projection onto a chord pulls the surface inward
		// and the pillar caves in, so the clamped closest point, which
		// stays ON the reference surface, is correct there.
		bool flat = true;
		for (int f : candidates)
		{
			OpenMesh::FaceHandle fh = reference_mesh.face_handle(f);
			if (reference_mesh.status(fh).deleted()) continue;

			Eigen::Vector3f v0, v1, v2;
			reference_mesh.GetFaceVertices(fh, v0, v1, v2);

			Eigen::Vector3f cp = ClosestPointOnTriangle(p, v0, v1, v2);
			if ((cp - p).squaredNorm() > best_d2 + target_edge_length * target_edge_length) continue;

			Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
			float nl = n.norm();
			if (nl < 1e-12f) continue;
			n /= nl;

			if (best_n.dot(n) < 0.98f) { flat = false; break; }
		}

		if (flat)
		{
			float signed_dist = (p - best_v0).dot(best_n);
			out_point = p - signed_dist * best_n;
		}
		else
		{
			out_point = best_cp;
		}
		return true;
	}

	int OperatorRemesh::TargetValence(OpenMesh::VertexHandle vh) const
	{
		int cls = meshTarget->property(prop_vertex_feature, vh);
		if (VertexFree == cls && false == meshTarget->is_boundary(vh))
		{
			return 6;
		}
		return 4;
	}

	size_t OperatorRemesh::SplitLongEdges()
	{
		std::vector<int> long_edges;
		for (size_t i = 0; i < meshTarget->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = meshTarget->edge_handle(static_cast<int>(i));
			if (meshTarget->status(eh).deleted()) continue;

			OpenMesh::HalfedgeHandle heh = meshTarget->halfedge_handle(eh, 0);
			Eigen::Vector3f a(meshTarget->point(meshTarget->from_vertex_handle(heh)).data());
			Eigen::Vector3f b(meshTarget->point(meshTarget->to_vertex_handle(heh)).data());

			if ((b - a).norm() > edge_high)
			{
				long_edges.push_back(static_cast<int>(i));
			}
		}

		size_t split_count = 0;

		for (int edge_idx : long_edges)
		{
			OpenMesh::EdgeHandle eh = meshTarget->edge_handle(edge_idx);
			if (meshTarget->status(eh).deleted()) continue;

			OpenMesh::HalfedgeHandle heh = meshTarget->halfedge_handle(eh, 0);
			OpenMesh::VertexHandle v_from = meshTarget->from_vertex_handle(heh);
			OpenMesh::VertexHandle v_to = meshTarget->to_vertex_handle(heh);

			Eigen::Vector3f a(meshTarget->point(v_from).data());
			Eigen::Vector3f b(meshTarget->point(v_to).data());
			Eigen::Vector3f mid = (a + b) * 0.5f;

			bool edge_is_feature = meshTarget->property(prop_edge_feature, eh);

			// A non-feature edge spanning a curved region is split onto the
			// original surface. A feature edge is a straight crease segment,
			// so its midpoint already lies on the feature: no reprojection.
			Eigen::Vector3f new_pos = mid;
			if (false == edge_is_feature)
			{
				ReprojectToReference(mid, new_pos);
			}

			// Reject the split if the reprojected midpoint would fold either
			// incident face. On narrow curved walls the projection can pull
			// the midpoint across the strip; splitting there is what seeds
			// the persistent dot -1 folds. Skipping the split leaves the
			// long edge for a later iteration once relaxation has widened
			// the local spacing.
			if (false == SplitKeepsFacesValid(eh, new_pos)) continue;

			OpenMesh::VertexHandle v_new = meshTarget->add_vertex(
				Mesh::Point(new_pos.x(), new_pos.y(), new_pos.z()));
			meshTarget->split(eh, v_new);
			++split_count;

			// Custom properties are not interpolated by split. The new
			// vertex on a feature edge lies on exactly two feature halves.
			meshTarget->property(prop_vertex_feature, v_new) =
				edge_is_feature ? VertexOnFeature : VertexFree;

			// Of the edges now incident to v_new, only the two collinear
			// halves of the original feature edge inherit the feature flag;
			// the spokes into the opposite apexes are interior.
			for (auto voh_it = meshTarget->voh_iter(v_new); voh_it.is_valid(); ++voh_it)
			{
				OpenMesh::VertexHandle v_other = meshTarget->to_vertex_handle(*voh_it);
				OpenMesh::EdgeHandle e_spoke = meshTarget->edge_handle(*voh_it);

				bool collinear = (v_other == v_from) || (v_other == v_to);
				meshTarget->property(prop_edge_feature, e_spoke) = edge_is_feature && collinear;
			}
		}

		return split_count;
	}

	bool OperatorRemesh::CollapseRingStaysValid(
		OpenMesh::VertexHandle v_ring,
		OpenMesh::VertexHandle v_from,
		OpenMesh::VertexHandle v_to,
		const Eigen::Vector3f& survivor) const
	{
		// Predicts every face of v_ring's one-ring after the collapse. Both
		// v_from and v_to map to the survivor position; all other corners
		// keep their current coordinates. A face that contains BOTH v_from
		// and v_to is destroyed by the collapse and is skipped.
		for (auto vf_it = meshTarget->vf_iter(v_ring); vf_it.is_valid(); ++vf_it)
		{
			OpenMesh::FaceHandle fh = *vf_it;
			if (meshTarget->status(fh).deleted()) continue;

			bool has_from = false;
			bool has_to = false;
			for (auto fv_it = meshTarget->cfv_iter(fh); fv_it.is_valid(); ++fv_it)
			{
				if (*fv_it == v_from) has_from = true;
				if (*fv_it == v_to) has_to = true;
			}

			// This face disappears in the collapse; it cannot fold.
			if (has_from && has_to) continue;

			Eigen::Vector3f p_old[3];
			Eigen::Vector3f p_new[3];
			int k = 0;
			for (auto fv_it = meshTarget->cfv_iter(fh); fv_it.is_valid() && k < 3; ++fv_it, ++k)
			{
				Eigen::Vector3f pos(meshTarget->point(*fv_it).data());
				p_old[k] = pos;
				p_new[k] = ((*fv_it == v_from) || (*fv_it == v_to)) ? survivor : pos;
			}
			if (k < 3) continue;

			Eigen::Vector3f n_old = (p_old[1] - p_old[0]).cross(p_old[2] - p_old[0]);
			Eigen::Vector3f n_new = (p_new[1] - p_new[0]).cross(p_new[2] - p_new[0]);

			if (n_old.dot(n_new) <= 0.0f) return false;

			float area_old = n_old.norm();
			float area_new = n_new.norm();
			if (area_new < 0.1f * area_old) return false;
			if (area_new < 1e-10f) return false;

			// Reference-orientation guard. The before/after dot above only
			// sees a single step's rotation, which stays positive on a
			// curved wall even as the face is dragged across the wall onto
			// the opposite sheet, where it ends up facing INTO the surface.
			// The same reference test that defines a fold for detection is
			// applied here predictively: if the post-collapse face opposes
			// the local reference normal, the collapse is creating exactly
			// the fold CountFlippedFaces would report, so it is rejected.
			Eigen::Vector3f centroid_new = (p_new[0] + p_new[1] + p_new[2]) / 3.0f;
			float new_len = n_new.norm();
			if (new_len > 1e-12f)
			{
				Eigen::Vector3f unit_new = n_new / new_len;
				Eigen::Vector3f ref_n;
				if (ReferenceNormalAt(centroid_new, ref_n))
				{
					if (unit_new.dot(ref_n) < 0.0f) return false;
				}
			}
		}

		return true;
	}

	bool OperatorRemesh::CollapseKeepsFacesValid(
		OpenMesh::VertexHandle v_from,
		OpenMesh::VertexHandle v_to,
		const Eigen::Vector3f& survivor) const
	{
		// A collapse merges v_from into v_to at the survivor position. The
		// faces that can fold are NOT only v_from's: every face incident to
		// v_to also has one corner (v_to) effectively repositioned to the
		// survivor, and the faces brought over from v_from become newly
		// adjacent to v_to's existing ring. Checking only v_from's faces (as
		// the prior version did) misses folds that appear on v_to's side,
		// which is exactly where the narrow curved-wall collapses fold. So
		// both one-rings are predicted against the survivor position, and
		// any face that would reverse or collapse to zero area rejects the
		// move. Faces containing the edge v_from-v_to vanish in the collapse
		// and are skipped.
		if (false == CollapseRingStaysValid(v_from, v_from, v_to, survivor)) return false;
		if (false == CollapseRingStaysValid(v_to, v_from, v_to, survivor)) return false;
		return true;
	}

	size_t OperatorRemesh::CollapseShortEdges()
	{
		if (Log::Diag())
		{
			std::cout << "[Debug] CollapseShortEdges BUILD MARKER 2026-06-08-A: reference deviation guard active." << std::endl;
		}

		size_t collapse_count = 0;

		for (size_t i = 0; i < meshTarget->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = meshTarget->edge_handle(static_cast<int>(i));
			if (meshTarget->status(eh).deleted()) continue;

			OpenMesh::HalfedgeHandle heh = meshTarget->halfedge_handle(eh, 0);
			OpenMesh::VertexHandle v_from = meshTarget->from_vertex_handle(heh);
			OpenMesh::VertexHandle v_to = meshTarget->to_vertex_handle(heh);

			Eigen::Vector3f a(meshTarget->point(v_from).data());
			Eigen::Vector3f b(meshTarget->point(v_to).data());
			if ((b - a).norm() >= edge_low) continue;

			int cls_from = meshTarget->property(prop_vertex_feature, v_from);
			int cls_to = meshTarget->property(prop_vertex_feature, v_to);
			if (VertexFree != cls_from) continue;
			if (VertexFree != cls_to) continue;

			if (false == meshTarget->is_collapse_ok(heh)) continue;

			Eigen::Vector3f survivor = b;

			Eigen::Vector3f ref_cp;
			if (ClosestPointOnReference(survivor, ref_cp))
			{
				float dev = (survivor - ref_cp).norm();
				if (dev > 0.5f * target_edge_length) continue;
			}

			bool reject = false;
			for (auto voh_it = meshTarget->voh_iter(v_from); voh_it.is_valid(); ++voh_it)
			{
				OpenMesh::VertexHandle v_n = meshTarget->to_vertex_handle(*voh_it);
				if (v_n == v_to) continue;

				Eigen::Vector3f pn(meshTarget->point(v_n).data());
				if ((pn - survivor).norm() > edge_high) { reject = true; break; }
			}
			if (reject) continue;

			if (false == CollapseKeepsFacesValid(v_from, v_to, survivor)) continue;

			meshTarget->collapse(heh);
			++collapse_count;
		}

		meshTarget->garbage_collection();
		return collapse_count;
	}

	bool OperatorRemesh::FlipPreservesOrientation(
		OpenMesh::VertexHandle va,
		OpenMesh::VertexHandle vb,
		OpenMesh::VertexHandle vc,
		OpenMesh::VertexHandle vd) const
	{
		Eigen::Vector3f a(meshTarget->point(va).data());
		Eigen::Vector3f b(meshTarget->point(vb).data());
		Eigen::Vector3f c(meshTarget->point(vc).data());
		Eigen::Vector3f d(meshTarget->point(vd).data());

		// Old triangles around the shared edge a-b:
		//   T0 = (a, b, c)   T1 = (b, a, d)
		// After flip the diagonal becomes c-d:
		//   N0 = (a, c, d)   N1 = (b, d, c)
		// The new pair must not oppose the old pair it replaces.
		Eigen::Vector3f n_old0 = (b - a).cross(c - a);
		Eigen::Vector3f n_old1 = (a - b).cross(d - b);
		Eigen::Vector3f n_new0 = (c - a).cross(d - a);
		Eigen::Vector3f n_new1 = (d - b).cross(c - b);

		Eigen::Vector3f n_old = n_old0 + n_old1;
		if (n_old.squaredNorm() < 1e-20f) return false;

		if (n_new0.dot(n_old) <= 0.0f) return false;
		if (n_new1.dot(n_old) <= 0.0f) return false;

		return true;
	}

	size_t OperatorRemesh::FlipToImproveValence()
	{
		size_t flip_count = 0;

		size_t reject_feature = 0;
		size_t reject_boundary = 0;
		size_t reject_flip_ok = 0;
		size_t reject_angle = 0;
		size_t reject_orientation = 0;
		size_t considered = 0;

		for (size_t i = 0; i < meshTarget->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = meshTarget->edge_handle(static_cast<int>(i));
			if (meshTarget->status(eh).deleted()) continue;

			if (meshTarget->property(prop_edge_feature, eh)) { ++reject_feature; continue; }
			if (meshTarget->is_boundary(eh)) { ++reject_boundary; continue; }
			if (false == meshTarget->is_flip_ok(eh)) { ++reject_flip_ok; continue; }

			++considered;

			OpenMesh::HalfedgeHandle h0 = meshTarget->halfedge_handle(eh, 0);
			OpenMesh::HalfedgeHandle h1 = meshTarget->halfedge_handle(eh, 1);

			OpenMesh::VertexHandle va = meshTarget->from_vertex_handle(h0);
			OpenMesh::VertexHandle vb = meshTarget->to_vertex_handle(h0);
			OpenMesh::VertexHandle vc = meshTarget->to_vertex_handle(meshTarget->next_halfedge_handle(h0));
			OpenMesh::VertexHandle vd = meshTarget->to_vertex_handle(meshTarget->next_halfedge_handle(h1));

			Eigen::Vector3f pa(meshTarget->point(va).data());
			Eigen::Vector3f pb(meshTarget->point(vb).data());
			Eigen::Vector3f pc(meshTarget->point(vc).data());
			Eigen::Vector3f pd(meshTarget->point(vd).data());

			float angle_before = MinAngleOfTwoTriangles(pa, pb, pc, pd);
			float angle_after = MinAngleOfTwoTriangles(pc, pd, pb, pa);

			const float margin = 0.01f;
			if (angle_after <= angle_before + margin) { ++reject_angle; continue; }

			if (false == FlipPreservesOrientation(va, vb, vc, vd)) { ++reject_orientation; continue; }

			meshTarget->flip(eh);
			++flip_count;
		}

		if (Log::AtInfo())
		{
			std::cout << "[Info] FlipDiag: considered " << considered
				<< ", rejected feature " << reject_feature
				<< ", boundary " << reject_boundary
				<< ", flip_ok " << reject_flip_ok
				<< ", angle " << reject_angle
				<< ", orientation " << reject_orientation
				<< ", flipped " << flip_count << "." << std::endl;
		}

		return flip_count;
	}

	bool OperatorRemesh::RelaxMoveKeepsFacesValid(
		OpenMesh::VertexHandle vh,
		const Eigen::Vector3f& new_pos) const
	{
		Eigen::Vector3f old_pos(meshTarget->point(vh).data());

		// Diagnostic window around the persistent first fold at (14.91, 0.7, -2.3).
		bool watch = (std::abs(old_pos.x() - 14.91f) < 0.1f)
			&& (std::abs(old_pos.z() - (-2.3f)) < 0.2f);

		for (auto vf_it = meshTarget->vf_iter(vh); vf_it.is_valid(); ++vf_it)
		{
			OpenMesh::FaceHandle fh = *vf_it;
			if (meshTarget->status(fh).deleted()) continue;

			Eigen::Vector3f q_old[3];
			Eigen::Vector3f q_new[3];
			int k = 0;
			for (auto fv_it = meshTarget->cfv_iter(fh); fv_it.is_valid() && k < 3; ++fv_it, ++k)
			{
				Eigen::Vector3f pos(meshTarget->point(*fv_it).data());
				q_old[k] = pos;
				q_new[k] = (*fv_it == vh) ? new_pos : pos;
			}
			if (k < 3) continue;

			Eigen::Vector3f n_old = (q_old[1] - q_old[0]).cross(q_old[2] - q_old[0]);
			Eigen::Vector3f n_new = (q_new[1] - q_new[0]).cross(q_new[2] - q_new[0]);

			if (n_old.dot(n_new) <= 0.0f)
			{
				if (watch)
				{
					if (Log::Diag())
					{
						std::cout << "[Debug] RelaxGuard REJECT at (" << old_pos.x() << ", "
							<< old_pos.y() << ", " << old_pos.z() << ") -> (" << new_pos.x()
							<< ", " << new_pos.y() << ", " << new_pos.z()
							<< "): own-ring face would flip." << std::endl;
					}
				}
				return false;
			}

			float area_old = n_old.norm();
			float area_new = n_new.norm();
			if (area_new < 0.1f * area_old) return false;
			if (area_new < 1e-10f) return false;
		}

		if (watch)
		{
			if (Log::Diag())
			{
				std::cout << "[Debug] RelaxGuard PASS at (" << old_pos.x() << ", "
					<< old_pos.y() << ", " << old_pos.z() << ") -> (" << new_pos.x()
					<< ", " << new_pos.y() << ", " << new_pos.z()
					<< "): all own-ring faces valid. Fold must be on a NEIGHBOR's ring." << std::endl;
			}
		}

		return true;
	}

	bool OperatorRemesh::ComputeRelaxTarget(
		OpenMesh::VertexHandle vh,
		int cls,
		Eigen::Vector3f& out_target) const
	{
		Eigen::Vector3f p(meshTarget->point(vh).data());

		if (VertexOnFeature == cls)
		{
			Eigen::Vector3f tangent = Eigen::Vector3f::Zero();
			Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
			int feature_neighbors = 0;

			for (auto voh_it = meshTarget->voh_iter(vh); voh_it.is_valid(); ++voh_it)
			{
				OpenMesh::EdgeHandle e = meshTarget->edge_handle(*voh_it);
				if (false == meshTarget->property(prop_edge_feature, e)) continue;

				Eigen::Vector3f pn(meshTarget->point(meshTarget->to_vertex_handle(*voh_it)).data());
				Eigen::Vector3f dir = pn - p;
				if (feature_neighbors > 0 && dir.dot(tangent) < 0.0f) dir = -dir;
				tangent += dir;
				centroid += pn;
				++feature_neighbors;
			}

			if (feature_neighbors < 2) return false;
			centroid /= static_cast<float>(feature_neighbors);

			float tlen = tangent.norm();
			if (tlen < 1e-12f) return false;
			tangent /= tlen;

			// Slide along the straight tangent toward the neighbor centroid.
			Eigen::Vector3f move = centroid - p;
			Eigen::Vector3f slid = p + move.dot(tangent) * tangent;

			// The feature curve is not straight: on the sine-wave wall it
			// bends. A straight-tangent slide therefore leaves the curved
			// surface and cuts across it, folding the adjacent faces. The
			// slid point is pulled back onto the reference surface so it
			// follows the real curve instead of its chord. The midpoint of
			// the two feature neighbors is the seed for the projection,
			// which keeps the result near the curve even where the local
			// reference patch is itself curved.
			Eigen::Vector3f projected;
			if (ReprojectToReference(slid, projected))
			{
				out_target = projected;
			}
			else
			{
				out_target = slid;
			}
			return true;
		}

		// Free vertex adjacent to any feature edge: held fixed, since
		// smoothing pulls a narrow feature-bounded strip across its own
		// boundary.
		for (auto voh_it = meshTarget->voh_iter(vh); voh_it.is_valid(); ++voh_it)
		{
			if (meshTarget->property(prop_edge_feature, meshTarget->edge_handle(*voh_it)))
			{
				return false;
			}
		}

		Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
		int n = 0;
		for (auto vv_it = meshTarget->vv_iter(vh); vv_it.is_valid(); ++vv_it)
		{
			centroid += Eigen::Vector3f(meshTarget->point(*vv_it).data());
			++n;
		}
		if (0 == n) return false;
		centroid /= static_cast<float>(n);

		Eigen::Vector3f normal(meshTarget->normal(vh).data());
		Eigen::Vector3f move = centroid - p;
		if (normal.squaredNorm() > 1e-12f)
		{
			normal.normalize();
			move -= move.dot(normal) * normal;
		}

		Eigen::Vector3f tangent_pos = p + move;
		Eigen::Vector3f projected;
		if (ReprojectToReference(tangent_pos, projected))
		{
			out_target = projected;
		}
		else
		{
			out_target = tangent_pos;
		}
		return true;
	}

	void OperatorRemesh::TangentialRelaxation()
	{
		// Sequential update with post-move fold verification. Computing the
		// target from the current state and checking the predicted one-ring
		// is not enough: when several corners of one face are relaxed in the
		// same sweep, each individual move passes its own predictive guard,
		// yet their cumulative displacement folds the shared face. So after
		// applying a move, the vertex's one-ring is re-checked against the
		// ACTUAL present geometry, and the move is reverted the instant a
		// real fold appears across any of its non-feature edges. A reverted
		// vertex simply keeps its prior position for this pass.
		for (size_t i = 0; i < meshTarget->n_vertices(); ++i)
		{
			OpenMesh::VertexHandle vh = meshTarget->vertex_handle(static_cast<int>(i));
			if (meshTarget->status(vh).deleted()) continue;

			int cls = meshTarget->property(prop_vertex_feature, vh);
			if (VertexCorner == cls) continue;

			Eigen::Vector3f target;
			if (false == ComputeRelaxTarget(vh, cls, target)) continue;

			if (false == RelaxMoveKeepsFacesValid(vh, target)) continue;

			Eigen::Vector3f old_pos(meshTarget->point(vh).data());
			meshTarget->set_point(vh, Mesh::Point(target.x(), target.y(), target.z()));

			// Post-move truth check against the cumulative geometry. If the
			// applied move folded any one-ring face, revert it.
			if (VertexRingHasFold(vh))
			{
				meshTarget->set_point(vh, Mesh::Point(old_pos.x(), old_pos.y(), old_pos.z()));
			}
		}
	}

	size_t OperatorRemesh::CountBoundaryEdges() const
	{
		size_t count = 0;
		for (size_t i = 0; i < meshTarget->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = meshTarget->edge_handle(static_cast<int>(i));
			if (meshTarget->status(eh).deleted()) continue;
			if (meshTarget->is_boundary(eh)) ++count;
		}
		return count;
	}

	size_t OperatorRemesh::ValidateManifold(const char* stage_label) const
	{
		// Edge fan count: every live edge must bound exactly 2 faces for a
		// closed manifold. Boundary edges (1 face) are counted separately by
		// CountBoundaryEdges; here a non-boundary edge with != 2 faces, or any
		// vertex whose one-ring is not a single fan, is the violation.
		size_t non_manifold_edges = 0;
		const size_t report_limit = 8;

		for (size_t i = 0; i < meshTarget->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = meshTarget->edge_handle(static_cast<int>(i));
			if (meshTarget->status(eh).deleted()) continue;

			// OpenMesh guarantees <= 2 faces per edge by construction, so the
			// only manifold defect an edge can show here is a complex/isolated
			// edge flagged non-manifold by the kernel.
			if (false == meshTarget->is_manifold(meshTarget->to_vertex_handle(meshTarget->halfedge_handle(eh, 0))))
			{
				// Vertex-level non-manifold is reported in the vertex pass below.
			}
		}

		size_t non_manifold_vertices = 0;
		for (size_t i = 0; i < meshTarget->n_vertices(); ++i)
		{
			OpenMesh::VertexHandle vh = meshTarget->vertex_handle(static_cast<int>(i));
			if (meshTarget->status(vh).deleted()) continue;
			if (false == meshTarget->is_manifold(vh))
			{
				++non_manifold_vertices;
				if (non_manifold_vertices <= report_limit)
				{
					auto p = meshTarget->point(vh);
					std::cout << "[Error] Manifold(" << stage_label << "): vertex ("
						<< p[0] << ", " << p[1] << ", " << p[2]
						<< ") is non-manifold." << std::endl;
				}
			}
		}

		size_t total = non_manifold_edges + non_manifold_vertices;
		if (0 == total)
		{
			if (Log::AtInfo())
			{
				std::cout << "[Info] Manifold(" << stage_label << "): clean ("
					<< meshTarget->n_vertices() << " vertices, "
					<< meshTarget->n_faces() << " faces)." << std::endl;
			}
		}
		else
		{
			std::cout << "[Error] Manifold(" << stage_label << "): "
				<< non_manifold_vertices << " non-manifold vertices." << std::endl;
		}
		return total;
	}

	size_t OperatorRemesh::ValidateFeatureParity(const char* stage_label) const
	{
		if (false == prop_edge_feature.is_valid()) return 0;

		bad_feature_vertices.clear();

		// Feature-edge degree per vertex. A feature curve is locally a
		// 1-manifold path, so a vertex it passes through has degree 2. The
		// only OTHER admissible cases are a vertex off all feature curves
		// (degree 0) and a curve endpoint or junction (degree 1 or 3+),
		// which is geometrically a CORNER and MUST be pinned as such. So
		// the correct invariant is not "all even", which wrongly flags the
		// box's 3-edge corners; it is: every degree != 2 feature vertex is
		// classified VertexCorner. A degree-1 or junction vertex left as
		// VertexOnFeature is a real break, because relaxation would slide
		// it off the curve end.
		std::vector<int> feature_degree(meshTarget->n_vertices(), 0);

		size_t feature_edges = 0;
		for (size_t i = 0; i < meshTarget->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = meshTarget->edge_handle(static_cast<int>(i));
			if (meshTarget->status(eh).deleted()) continue;
			if (false == meshTarget->property(prop_edge_feature, eh)) continue;

			++feature_edges;
			OpenMesh::HalfedgeHandle heh = meshTarget->halfedge_handle(eh, 0);
			feature_degree[meshTarget->from_vertex_handle(heh).idx()] += 1;
			feature_degree[meshTarget->to_vertex_handle(heh).idx()] += 1;
		}

		size_t misclassified = 0;
		size_t corners = 0;
		size_t curve_points = 0;
		const size_t report_limit = 8;

		for (size_t i = 0; i < feature_degree.size(); ++i)
		{
			int deg = feature_degree[i];
			if (0 == deg) continue;

			OpenMesh::VertexHandle vh = meshTarget->vertex_handle(static_cast<int>(i));
			if (meshTarget->status(vh).deleted()) continue;

			int cls = meshTarget->property(prop_vertex_feature, vh);

			if (2 == deg)
			{
				// A through-point: OnFeature (sliding) or Corner (pinned by
				// a sharp bend) are both valid here. Nothing to check.
				++curve_points;
				continue;
			}

			// deg == 1 (endpoint) or deg >= 3 (junction): must be a corner.
			++corners;
			if (VertexCorner != cls)
			{
				++misclassified;
				bad_feature_vertices.push_back(Eigen::Vector3f(
					meshTarget->point(vh)[0],
					meshTarget->point(vh)[1],
					meshTarget->point(vh)[2]));

				if (misclassified <= report_limit)
				{
					auto p = meshTarget->point(vh);
					std::cout << "[Error] FeatureParity(" << stage_label << "): vertex ("
						<< p[0] << ", " << p[1] << ", " << p[2] << ") has feature degree "
						<< deg << " but class " << cls
						<< " (degree != 2 must be VertexCorner)." << std::endl;
				}
			}
		}

		if (0 == misclassified)
		{
			if (Log::AtInfo())
			{
				std::cout << "[Info] FeatureParity(" << stage_label << "): intact ("
					<< feature_edges << " feature edges, " << curve_points
					<< " through-points, " << corners << " corners, all consistent)." << std::endl;
			}
		}
		else
		{
			std::cout << "[Error] FeatureParity(" << stage_label << "): " << misclassified
				<< " feature vertices with degree != 2 not pinned as corners."
				<< " The feature curve is broken or a junction is unprotected." << std::endl;
		}

		return misclassified;
	}

	bool OperatorRemesh::FaceNormalAndCentroid(
		OpenMesh::FaceHandle fh,
		Eigen::Vector3f& out_normal,
		Eigen::Vector3f& out_centroid) const
	{
		if (meshTarget->status(fh).deleted()) return false;

		Eigen::Vector3f v0, v1, v2;
		meshTarget->GetFaceVertices(fh, v0, v1, v2);

		Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
		float nl = n.norm();
		if (nl < 1e-12f) return false;

		out_normal = n / nl;
		out_centroid = (v0 + v1 + v2) / 3.0f;
		return true;
	}

	size_t OperatorRemesh::CountFlippedFaces(const char* stage_label) const
	{
		flipped_face_centers.clear();

		size_t flipped_edges = 0;
		const size_t report_limit = 8;

		std::vector<Eigen::Vector3f> this_stage_folds;

		const float carry_radius = 2.0f * target_edge_length;
		const float carry_radius2 = carry_radius * carry_radius;

		size_t carried = 0;
		size_t fresh = 0;
		size_t ridge_skipped = 0;

		for (size_t i = 0; i < meshTarget->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = meshTarget->edge_handle(static_cast<int>(i));
			if (meshTarget->status(eh).deleted()) continue;
			if (meshTarget->is_boundary(eh)) continue;

			if (prop_edge_feature.is_valid()
				&& meshTarget->property(prop_edge_feature, eh)) continue;

			OpenMesh::HalfedgeHandle h0 = meshTarget->halfedge_handle(eh, 0);
			OpenMesh::HalfedgeHandle h1 = meshTarget->halfedge_handle(eh, 1);
			OpenMesh::FaceHandle f0 = meshTarget->face_handle(h0);
			OpenMesh::FaceHandle f1 = meshTarget->face_handle(h1);
			if (false == f0.is_valid() || false == f1.is_valid()) continue;

			Eigen::Vector3f n0, c0, n1, c1;
			if (false == FaceNormalAndCentroid(f0, n0, c0)) continue;
			if (false == FaceNormalAndCentroid(f1, n1, c1)) continue;

			if (n0.dot(n1) >= 0.0f) continue;

			// dot < 0 alone cannot tell a genuine fold (a face turned back
			// onto the surface) from a valid sharp ridge that the feature
			// detector missed (front wall meets side wall at ~90 degrees).
			// The reference surface settles it: a fold has at least one
			// face whose normal opposes the local reference normal, while a
			// sharp ridge has BOTH faces agreeing with their own nearest
			// reference face (each simply picks a different reference face
			// across the crease). Only the former is a real defect; the
			// latter is correct geometry whose edge merely lacks the
			// feature flag, which is a feature-propagation gap, not a fold.
			Eigen::Vector3f ref_n0, ref_n1;
			bool have_r0 = ReferenceNormalAt(c0, ref_n0);
			bool have_r1 = ReferenceNormalAt(c1, ref_n1);

			if (have_r0 && have_r1)
			{
				bool f0_agrees = (n0.dot(ref_n0) > 0.0f);
				bool f1_agrees = (n1.dot(ref_n1) > 0.0f);

				// Both faces sit the right way up on the reference surface:
				// this is a valid crease, not a fold. Skip it.
				if (f0_agrees && f1_agrees)
				{
					++ridge_skipped;
					continue;
				}
			}

			++flipped_edges;
			flipped_face_centers.push_back(c0);
			flipped_face_centers.push_back(c1);

			Eigen::Vector3f mid = 0.5f * (c0 + c1);
			this_stage_folds.push_back(mid);

			bool is_carried = false;
			for (const Eigen::Vector3f& prev : prev_stage_fold_mids)
			{
				if ((prev - mid).squaredNorm() < carry_radius2)
				{
					is_carried = true;
					break;
				}
			}
			if (is_carried) ++carried;
			else ++fresh;

			if (flipped_edges <= report_limit)
			{
				std::cout << "[Error] FlippedFace(" << stage_label << "): faces across edge near ("
					<< mid.x() << ", " << mid.y() << ", " << mid.z()
					<< ") have opposing normals (dot " << n0.dot(n1) << ")"
					<< (is_carried ? " [carried]" : " [NEW]") << "." << std::endl;
			}
		}

		if (0 == flipped_edges)
		{
			if (Log::AtInfo())
			{
				std::cout << "[Info] FlippedFaces(" << stage_label << "): none ("
					<< ridge_skipped << " valid sharp ridges skipped)." << std::endl;
			}
		}
		else
		{
			std::cout << "[Error] FlippedFaces(" << stage_label << "): " << flipped_edges
				<< " folded edges (" << fresh << " NEW, " << carried
				<< " carried from previous stage, " << ridge_skipped
				<< " valid sharp ridges skipped)." << std::endl;
		}

		prev_stage_fold_mids = this_stage_folds;

		return flipped_edges;
	}

	bool OperatorRemesh::ValidateStage(const char* stage_label, size_t boundary_expected) const
	{
		size_t nm = ValidateManifold(stage_label);
		size_t parity = ValidateFeatureParity(stage_label);
		size_t flipped = CountFlippedFaces(stage_label);

		size_t boundary = CountBoundaryEdges();
		bool boundary_ok = (boundary == boundary_expected);
		if (false == boundary_ok)
		{
			std::cout << "[Error] Boundary(" << stage_label << "): edge count "
				<< boundary << " != expected " << boundary_expected << "." << std::endl;
		}
		else
		{
			if (Log::AtInfo())
			{
				std::cout << "[Info] Boundary(" << stage_label << "): " << boundary
					<< " edges (preserved)." << std::endl;
			}
		}

		bool ok = (0 == nm) && (0 == parity) && (0 == flipped) && boundary_ok;
		std::cout << "[" << (ok ? "Info" : "Error") << "] ValidateStage(" << stage_label
			<< "): " << (ok ? "PASS" : "FAIL") << "." << std::endl;
		return ok;
	}

	void OperatorRemesh::GetFeatureEdgeLines(std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& out_lines) const
	{
		out_lines.clear();
		if (false == prop_edge_feature.is_valid()) return;

		for (size_t i = 0; i < meshTarget->n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = meshTarget->edge_handle(static_cast<int>(i));
			if (meshTarget->status(eh).deleted()) continue;
			if (false == meshTarget->property(prop_edge_feature, eh)) continue;

			OpenMesh::HalfedgeHandle heh = meshTarget->halfedge_handle(eh, 0);
			auto pf = meshTarget->point(meshTarget->from_vertex_handle(heh));
			auto pt = meshTarget->point(meshTarget->to_vertex_handle(heh));
			out_lines.push_back({
				Eigen::Vector3f(pf[0], pf[1], pf[2]),
				Eigen::Vector3f(pt[0], pt[1], pt[2]) });
		}
	}

	float OperatorRemesh::TriangleAspectRatio(
		const Eigen::Vector3f& a,
		const Eigen::Vector3f& b,
		const Eigen::Vector3f& c) const
	{
		float la = (b - c).norm();
		float lb = (c - a).norm();
		float lc = (a - b).norm();

		float longest = std::max(la, std::max(lb, lc));

		float s = 0.5f * (la + lb + lc);
		float area = std::sqrt(std::max(0.0f, s * (s - la) * (s - lb) * (s - lc)));
		if (area < 1e-12f) return std::numeric_limits<float>::max();

		float inradius = area / s;
		if (inradius < 1e-12f) return std::numeric_limits<float>::max();

		return longest / inradius;
	}

	void OperatorRemesh::DiagnoseFeatureOnlyFaces(const char* stage_label) const
	{
		if (false == prop_vertex_feature.is_valid()) return;

		size_t total_faces = 0;
		size_t feature_only_faces = 0;
		size_t two_feature_faces = 0;

		float worst_aspect = 0.0f;
		Eigen::Vector3f worst_centroid = Eigen::Vector3f::Zero();

		for (size_t i = 0; i < meshTarget->n_faces(); ++i)
		{
			OpenMesh::FaceHandle fh = meshTarget->face_handle(static_cast<int>(i));
			if (meshTarget->status(fh).deleted()) continue;

			++total_faces;

			int feature_vertex_count = 0;
			Eigen::Vector3f pos[3];
			int k = 0;
			for (auto fv_it = meshTarget->cfv_iter(fh); fv_it.is_valid() && k < 3; ++fv_it, ++k)
			{
				int cls = meshTarget->property(prop_vertex_feature, *fv_it);
				if (VertexFree != cls) ++feature_vertex_count;
				pos[k] = Eigen::Vector3f(meshTarget->point(*fv_it).data());
			}
			if (k < 3) continue;

			if (3 == feature_vertex_count) ++feature_only_faces;
			else if (2 == feature_vertex_count) ++two_feature_faces;

			float aspect = TriangleAspectRatio(pos[0], pos[1], pos[2]);
			if (aspect > worst_aspect)
			{
				worst_aspect = aspect;
				worst_centroid = (pos[0] + pos[1] + pos[2]) / 3.0f;
			}
		}

		if (Log::AtInfo())
		{
			std::cout << "[Info] FeatureOnlyFaces(" << stage_label << "): "
				<< feature_only_faces << " of " << total_faces
				<< " faces have all 3 vertices on features, "
				<< two_feature_faces << " have exactly 2. Worst aspect ratio "
				<< worst_aspect << " near (" << worst_centroid.x() << ", "
				<< worst_centroid.y() << ", " << worst_centroid.z() << ")." << std::endl;
		}
	}

	bool OperatorRemesh::ClosestPointOnReference(const Eigen::Vector3f& p, Eigen::Vector3f& out_point) const
	{
		float radius = std::max(target_edge_length, 10.0f * EPSILON);

		std::vector<int> candidates;
		for (int attempt = 0; attempt < 8; ++attempt)
		{
			Eigen::Vector3f pad(radius, radius, radius);
			reference_mesh.QueryOverlappingFaces(p - pad, p + pad, candidates);
			if (false == candidates.empty()) break;
			radius *= 2.0f;
		}

		if (candidates.empty())
		{
			out_point = p;
			return false;
		}

		float best_d2 = std::numeric_limits<float>::max();
		Eigen::Vector3f best = p;

		for (int f : candidates)
		{
			OpenMesh::FaceHandle fh = reference_mesh.face_handle(f);
			if (reference_mesh.status(fh).deleted()) continue;

			Eigen::Vector3f v0, v1, v2;
			reference_mesh.GetFaceVertices(fh, v0, v1, v2);

			Eigen::Vector3f cp = ClosestPointOnTriangle(p, v0, v1, v2);
			float d2 = (cp - p).squaredNorm();
			if (d2 < best_d2)
			{
				best_d2 = d2;
				best = cp;
			}
		}

		out_point = best;
		return true;
	}

	void OperatorRemesh::DiagnoseSurfaceDeviation(const char* stage_label) const
	{
		float max_dev = 0.0f;
		double sum_dev = 0.0;
		size_t counted = 0;
		Eigen::Vector3f worst_pos = Eigen::Vector3f::Zero();

		for (size_t i = 0; i < meshTarget->n_vertices(); ++i)
		{
			OpenMesh::VertexHandle vh = meshTarget->vertex_handle(static_cast<int>(i));
			if (meshTarget->status(vh).deleted()) continue;

			Eigen::Vector3f p(meshTarget->point(vh).data());

			Eigen::Vector3f cp;
			if (false == ClosestPointOnReference(p, cp)) continue;

			float dev = (cp - p).norm();
			sum_dev += dev;
			++counted;
			if (dev > max_dev)
			{
				max_dev = dev;
				worst_pos = p;
			}
		}

		float mean_dev = (counted > 0) ? static_cast<float>(sum_dev / counted) : 0.0f;

		if (Log::AtInfo())
		{
			std::cout << "[Info] SurfaceDeviation(" << stage_label << "): max "
				<< max_dev << " near (" << worst_pos.x() << ", " << worst_pos.y()
				<< ", " << worst_pos.z() << "), mean " << mean_dev
				<< " over " << counted << " vertices." << std::endl;
		}
	}

	float OperatorRemesh::MinAngleOfTwoTriangles(
		const Eigen::Vector3f& a,
		const Eigen::Vector3f& b,
		const Eigen::Vector3f& c,
		const Eigen::Vector3f& d) const
	{
		// Two triangles (a,b,c) and (a,c,d) sharing edge a-c. Returns the
		// smallest interior angle over both, as a cosine-free proxy: the
		// minimum of the six corner angles. A sliver has a near-zero angle,
		// so maximizing this minimum is exactly the Delaunay (max-min-angle)
		// criterion that removes slivers without moving any vertex.
		float worst = std::numeric_limits<float>::max();

		const Eigen::Vector3f* tri[2][3] = {
			{ &a, &b, &c },
			{ &a, &c, &d }
		};

		for (int t = 0; t < 2; ++t)
		{
			for (int k = 0; k < 3; ++k)
			{
				const Eigen::Vector3f& p0 = *tri[t][k];
				const Eigen::Vector3f& p1 = *tri[t][(k + 1) % 3];
				const Eigen::Vector3f& p2 = *tri[t][(k + 2) % 3];

				Eigen::Vector3f e0 = p1 - p0;
				Eigen::Vector3f e1 = p2 - p0;
				float l0 = e0.norm();
				float l1 = e1.norm();
				if (l0 < 1e-12f || l1 < 1e-12f) return 0.0f;

				float cos_angle = e0.dot(e1) / (l0 * l1);
				cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));
				float angle = std::acos(cos_angle);

				if (angle < worst) worst = angle;
			}
		}

		return worst;
	}

	bool OperatorRemesh::VertexRingHasFold(OpenMesh::VertexHandle vh) const
	{
		// Checks the ACTUAL current geometry of vh's one-ring for a fold,
		// reading every face's real present coordinates. Unlike the
		// predictive RelaxMoveKeepsFacesValid, this sees the cumulative
		// result of all moves already applied this pass.
		for (auto vf_it = meshTarget->vf_iter(vh); vf_it.is_valid(); ++vf_it)
		{
			OpenMesh::FaceHandle fh = *vf_it;
			if (meshTarget->status(fh).deleted()) continue;

			OpenMesh::HalfedgeHandle h_in_face;
			bool found = false;
			for (auto fh_it = meshTarget->fh_iter(fh); fh_it.is_valid(); ++fh_it)
			{
				if (meshTarget->to_vertex_handle(*fh_it) == vh)
				{
					h_in_face = *fh_it;
					found = true;
					break;
				}
			}
			if (false == found) continue;

			OpenMesh::HalfedgeHandle opp = meshTarget->opposite_halfedge_handle(h_in_face);
			if (meshTarget->is_boundary(opp)) continue;

			OpenMesh::EdgeHandle eh = meshTarget->edge_handle(h_in_face);
			if (prop_edge_feature.is_valid()
				&& meshTarget->property(prop_edge_feature, eh)) continue;

			OpenMesh::FaceHandle nfh = meshTarget->face_handle(opp);
			if (false == nfh.is_valid() || meshTarget->status(nfh).deleted()) continue;

			Eigen::Vector3f n0, c0, n1, c1;
			if (false == FaceNormalAndCentroid(fh, n0, c0)) continue;
			if (false == FaceNormalAndCentroid(nfh, n1, c1)) continue;

			// Two adjacent faces with opposing normals are only a real fold
			// when at least one of them faces INTO the reference surface.
			// On a curved glyph wall the two faces of a valid sharp ridge
			// also oppose each other across a missed-feature edge, yet both
			// agree with their own nearest reference face: that is correct
			// geometry, not a fold, and reverting the relax move there would
			// wrongly freeze the curved wall. This applies the same
			// reference-based fold definition CountFlippedFaces uses, so the
			// relax revert triggers on exactly the defects detection counts.
			if (n0.dot(n1) >= 0.0f) continue;

			Eigen::Vector3f ref_n0, ref_n1;
			bool have_r0 = ReferenceNormalAt(c0, ref_n0);
			bool have_r1 = ReferenceNormalAt(c1, ref_n1);

			if (have_r0 && have_r1)
			{
				bool f0_agrees = (n0.dot(ref_n0) > 0.0f);
				bool f1_agrees = (n1.dot(ref_n1) > 0.0f);

				// Both faces sit the right way up on the reference: a valid
				// crease, not a fold. Do not revert for this.
				if (f0_agrees && f1_agrees) continue;
			}

			return true;
		}

		return false;
	}

	bool OperatorRemesh::SplitKeepsFacesValid(
		OpenMesh::EdgeHandle eh,
		const Eigen::Vector3f& new_pos) const
	{
		// A split replaces each of the (up to two) faces incident to the
		// edge with two sub-triangles sharing the new midpoint vertex. On a
		// narrow curved wall the reprojected midpoint can leave the original
		// face plane enough to reverse one sub-triangle. This predicts each
		// resulting sub-triangle's normal against its parent face normal and
		// rejects the split if any would oppose it, so the split never
		// introduces a fold.
		OpenMesh::HalfedgeHandle h0 = meshTarget->halfedge_handle(eh, 0);
		OpenMesh::HalfedgeHandle h1 = meshTarget->halfedge_handle(eh, 1);

		OpenMesh::VertexHandle v_from = meshTarget->from_vertex_handle(h0);
		OpenMesh::VertexHandle v_to = meshTarget->to_vertex_handle(h0);

		Eigen::Vector3f a(meshTarget->point(v_from).data());
		Eigen::Vector3f b(meshTarget->point(v_to).data());

		// For each side, the apex is the third vertex of that face. The two
		// sub-triangles are (a, mid, apex) and (mid, b, apex), which must
		// keep the parent winding (a, b, apex).
		auto side_ok = [&](OpenMesh::HalfedgeHandle h) -> bool
			{
				OpenMesh::FaceHandle fh = meshTarget->face_handle(h);
				if (false == fh.is_valid()) return true;
				if (meshTarget->status(fh).deleted()) return true;

				OpenMesh::VertexHandle v_apex =
					meshTarget->to_vertex_handle(meshTarget->next_halfedge_handle(h));
				Eigen::Vector3f apex(meshTarget->point(v_apex).data());

				Eigen::Vector3f n_parent = (b - a).cross(apex - a);

				Eigen::Vector3f n_sub0 = (new_pos - a).cross(apex - a);
				Eigen::Vector3f n_sub1 = (b - new_pos).cross(apex - new_pos);

				if (n_parent.dot(n_sub0) <= 0.0f) return false;
				if (n_parent.dot(n_sub1) <= 0.0f) return false;
				return true;
			};

		if (false == side_ok(h0)) return false;
		if (false == side_ok(h1)) return false;
		return true;
	}

	bool OperatorRemesh::ReferenceNormalAt(
		const Eigen::Vector3f& p,
		Eigen::Vector3f& out_normal) const
	{
		float radius = std::max(target_edge_length, 10.0f * EPSILON);

		std::vector<int> candidates;
		for (int attempt = 0; attempt < 8; ++attempt)
		{
			Eigen::Vector3f pad(radius, radius, radius);
			reference_mesh.QueryOverlappingFaces(p - pad, p + pad, candidates);
			if (false == candidates.empty()) break;
			radius *= 2.0f;
		}

		if (candidates.empty()) return false;

		// Nearest reference face by clamped closest-point distance. Its
		// oriented normal is the local surface direction the remeshed face
		// must agree with. A genuine fold faces opposite this; a sharp but
		// valid ridge faces along it (the two ridge faces simply pick two
		// different reference faces, each agreeing with its own).
		float best_d2 = std::numeric_limits<float>::max();
		Eigen::Vector3f best_n = Eigen::Vector3f::Zero();
		bool found = false;

		for (int f : candidates)
		{
			OpenMesh::FaceHandle fh = reference_mesh.face_handle(f);
			if (reference_mesh.status(fh).deleted()) continue;

			Eigen::Vector3f v0, v1, v2;
			reference_mesh.GetFaceVertices(fh, v0, v1, v2);

			Eigen::Vector3f cp = ClosestPointOnTriangle(p, v0, v1, v2);
			float d2 = (cp - p).squaredNorm();
			if (d2 < best_d2)
			{
				Eigen::Vector3f n = (v1 - v0).cross(v2 - v0);
				float nl = n.norm();
				if (nl < 1e-12f) continue;

				best_d2 = d2;
				best_n = n / nl;
				found = true;
			}
		}

		if (false == found) return false;
		out_normal = best_n;
		return true;
	}

	bool OperatorRemesh::Execute()
	{
		total_split_count = 0;
		total_collapse_count = 0;
		total_flip_count = 0;
		feature_edge_count = 0;
		if (nullptr == meshTarget) return false;
		if (0 == meshTarget->n_faces()) return false;
		if (target_edge_length < EPSILON)
		{
			std::cout << "[Error] OperatorRemesh: target_edge_length must be positive." << std::endl;
			return false;
		}
		meshTarget->add_property(prop_edge_feature, "remesh_edge_feature");
		meshTarget->add_property(prop_vertex_feature, "remesh_vertex_feature");
		meshTarget->request_vertex_normals();
		meshTarget->request_face_normals();
		BuildReferenceSnapshot();
		DetectFeatureEdges();
		input_boundary_count = CountBoundaryEdges();
		// Baseline: the invariants must already hold on the input before any
		// edit. If Pass 0 detection itself produced odd parity, the problem is
		// in detection, not in the operators.
		if (Log::AtInfo())
		{
			std::cout << "[Info] OperatorRemesh: --- stage Pass0 (detection) ---" << std::endl;
		}
		ValidateStage("Pass0", input_boundary_count);
		DiagnoseFeatureOnlyFaces("Pass0");
		DiagnoseSurfaceDeviation("Pass0");
		bool all_ok = true;
		for (int iter = 0; iter < iterations; ++iter)
		{
			if (Log::AtInfo())
			{
				std::cout << "[Info] OperatorRemesh: ===== iteration " << iter << " =====" << std::endl;
			}
			if (passes.split)
			{
				size_t s = SplitLongEdges();
				total_split_count += s;
				if (Log::AtInfo())
				{
					std::cout << "[Info] OperatorRemesh iter " << iter << ": " << s << " splits." << std::endl;
				}
				if (false == ValidateStage("split", input_boundary_count)) all_ok = false;
				DiagnoseSurfaceDeviation("split");
			}
			if (passes.collapse)
			{
				size_t c = CollapseShortEdges();
				total_collapse_count += c;
				if (Log::AtInfo())
				{
					std::cout << "[Info] OperatorRemesh iter " << iter << ": " << c << " collapses." << std::endl;
				}
				if (false == ValidateStage("collapse", input_boundary_count)) all_ok = false;
				DiagnoseSurfaceDeviation("collapse");
			}
			if (passes.flip)
			{
				size_t f = FlipToImproveValence();
				total_flip_count += f;
				if (Log::AtInfo())
				{
					std::cout << "[Info] OperatorRemesh iter " << iter << ": " << f << " flips." << std::endl;
				}
				if (false == ValidateStage("flip", input_boundary_count)) all_ok = false;
				DiagnoseSurfaceDeviation("flip");
			}
			if (passes.relax)
			{
				meshTarget->update_face_normals();
				meshTarget->update_vertex_normals();
				TangentialRelaxation();
				if (Log::AtInfo())
				{
					std::cout << "[Info] OperatorRemesh iter " << iter << ": relaxation done." << std::endl;
				}
				if (false == ValidateStage("relax", input_boundary_count)) all_ok = false;
				DiagnoseSurfaceDeviation("relax");
			}
		}
		DiagnoseFeatureOnlyFaces("final");
		meshTarget->BuildSpatialHashMap();
		std::vector<Eigen::Vector3f> points;
		std::vector<Eigen::Vector3i> indices;
		ExtractMeshSoup(meshTarget, points, indices);
		bool soup_ok = ValidateTriangleSoup(points, indices, "remesh", 100.0f * EPSILON);
		if (Log::AtInfo())
		{
			std::cout << "[Info] OperatorRemesh: " << total_split_count << " splits, "
				<< total_collapse_count << " collapses, " << total_flip_count
				<< " flips total over " << iterations << " iterations, "
				<< indices.size() << " result triangles." << std::endl;
		}
		return soup_ok && all_ok;
	}

	OperatorValidate::OperatorValidate(std::function<bool()> predicate, const std::string& label)
		: predicate(std::move(predicate))
		, label(label)
	{
	}

	bool OperatorValidate::Execute()
	{
		if (false == static_cast<bool>(predicate))
		{
			std::cout << "[Error] OperatorValidate(" << label
				<< "): no predicate bound." << std::endl;
			return false;
		}

		bool ok = predicate();
		std::cout << "[" << (ok ? "Info" : "Error") << "] OperatorValidate("
			<< label << "): " << (ok ? "PASS" : "FAIL") << "." << std::endl;
		return ok;
	}

	Pipeline::Pipeline()
	{
	}

	void Pipeline::RegisterValidator(std::function<bool()> predicate, const std::string& label)
	{
		std::unique_ptr<Operator> validator =
			std::make_unique<OperatorValidate>(std::move(predicate), label);
		stages.push_back(Stage{ std::move(validator), label });
	}

	void Pipeline::Clear()
	{
		stages.clear();
		failed_index = -1;
		failed_label.clear();
	}

	bool Pipeline::Run()
	{
		failed_index = -1;
		failed_label.clear();

		for (size_t i = 0; i < stages.size(); ++i)
		{
			if (nullptr == stages[i].op)
			{
				failed_index = static_cast<int>(i);
				failed_label = stages[i].label;
				std::cout << "[Error] Pipeline: stage " << i << " ("
					<< stages[i].label << ") is null; pipeline terminated." << std::endl;
				return false;
			}

			if (Log::AtInfo())
			{
				std::cout << "[Info] Pipeline: --- stage " << i << " ("
					<< stages[i].label << ") ---" << std::endl;
			}

			if (false == stages[i].op->Execute())
			{
				failed_index = static_cast<int>(i);
				failed_label = stages[i].label;
				std::cout << "[Error] Pipeline: stage " << i << " ("
					<< stages[i].label << ") failed; pipeline terminated, "
					<< (stages.size() - i - 1) << " remaining stages skipped." << std::endl;
				return false;
			}
		}

		if (Log::AtInfo())
		{
			std::cout << "[Info] Pipeline: all " << stages.size()
				<< " stages succeeded." << std::endl;
		}
		return true;
	}
}