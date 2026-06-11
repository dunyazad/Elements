#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <execution>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <robin_hood/robin_hood.h>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include <CDT.h>

// Robust Geometric Operations:
//		mesh boolean, offsetting, etc. based on spatial hashing and exact predicates.

namespace Eigen
{
	Matrix4f MakeTransform(
		const Vector3f& translation,
		const Vector3f& rotation_axis,
		float rotation_angle_degree,
		const Vector3f& scale);

	Matrix4f MakeTransformEuler(
		const Vector3f& translation,
		const Vector3f& euler_xyz_degree,
		const Vector3f& scale);

	Matrix4f MakeTransformManual(
		const Vector3f& translation,
		const Quaternionf& rotation,
		const Vector3f& scale);
}

namespace RGO
{
	const float EPSILON = 1e-5f;

	// Runtime log and diagnostic toggle. [Error] and [Warning] always
	// print. [Info] prints only at Info level or above; [Debug]
	// diagnostics print only when diagnostics is enabled. The pipeline
	// sets these from the application before running an operation.
	enum class LogLevel
	{
		Silent = 0,
		Error = 1,
		Info = 2,
		Debug = 3
	};

	struct Log
	{
		static LogLevel level;
		static bool diagnostics;

		static bool AtInfo() { return static_cast<int>(level) >= static_cast<int>(LogLevel::Info); }
		static bool AtDebug() { return static_cast<int>(level) >= static_cast<int>(LogLevel::Debug); }
		static bool Diag() { return diagnostics; }
	};

	// Shared quantization for hashing and equality.
	// Hash and Equal must agree on the same grid, otherwise the
	// unordered_map invariant (equal keys imply equal hashes) breaks.
	inline Eigen::Vector3i QuantizePoint(const Eigen::Vector3f& v, float cell = EPSILON)
	{
		return Eigen::Vector3i(
			static_cast<int>(std::floor(v.x() / cell)),
			static_cast<int>(std::floor(v.y() / cell)),
			static_cast<int>(std::floor(v.z() / cell)));
	}

	struct Int3Hash
	{
		size_t operator()(const Eigen::Vector3i& v) const
		{
			// Cast to unsigned before multiplying to avoid signed overflow UB
			size_t h1 = static_cast<size_t>(static_cast<unsigned int>(v.x())) * 73856093ull;
			size_t h2 = static_cast<size_t>(static_cast<unsigned int>(v.y())) * 19349663ull;
			size_t h3 = static_cast<size_t>(static_cast<unsigned int>(v.z())) * 83492791ull;
			return h1 ^ h2 ^ h3;
		}
	};

	struct Int3Equal
	{
		bool operator()(const Eigen::Vector3i& a, const Eigen::Vector3i& b) const
		{
			return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
		}
	};

	struct Vector3fHash
	{
		size_t operator()(const Eigen::Vector3f& v) const
		{
			Eigen::Vector3i q = QuantizePoint(v);
			size_t h1 = static_cast<size_t>(static_cast<unsigned int>(q.x())) * 73856093ull;
			size_t h2 = static_cast<size_t>(static_cast<unsigned int>(q.y())) * 19349663ull;
			size_t h3 = static_cast<size_t>(static_cast<unsigned int>(q.z())) * 83492791ull;
			return h1 ^ h2 ^ h3;
		}
	};

	struct Vector3fEqual
	{
		bool operator()(const Eigen::Vector3f& a, const Eigen::Vector3f& b) const
		{
			// Must use the same quantization as Vector3fHash.
			// Compare components directly instead of Eigen operator== to avoid
			// instantiating deprecated std::equal_to typedefs in older Eigen.
			Eigen::Vector3i qa = QuantizePoint(a);
			Eigen::Vector3i qb = QuantizePoint(b);
			return qa.x() == qb.x() && qa.y() == qb.y() && qa.z() == qb.z();
		}
	};

	struct Vector3fBitHash
	{
		size_t operator()(const Eigen::Vector3f& v) const
		{
			// Bit-exact hash. Canonicalization guarantees that logically
			// identical points are bit-identical, so no quantization is
			// needed and no false negatives are possible.
			uint32_t bx, by, bz;
			std::memcpy(&bx, &v.x(), sizeof(uint32_t));
			std::memcpy(&by, &v.y(), sizeof(uint32_t));
			std::memcpy(&bz, &v.z(), sizeof(uint32_t));
			size_t h = static_cast<size_t>(bx) * 73856093ull;
			h ^= static_cast<size_t>(by) * 19349663ull;
			h ^= static_cast<size_t>(bz) * 83492791ull;
			return h;
		}
	};

	struct Vector3fBitEqual
	{
		bool operator()(const Eigen::Vector3f& a, const Eigen::Vector3f& b) const
		{
			return a.x() == b.x() && a.y() == b.y() && a.z() == b.z();
		}
	};

	class Distance
	{
	public:
		static float PointToRaySquared(const Eigen::Vector3f& p, const Eigen::Vector3f& origin, const Eigen::Vector3f& dir);
		static float PointToRay(const Eigen::Vector3f& p, const Eigen::Vector3f& origin, const Eigen::Vector3f& dir);
		static float PointToLineSegmentSquared(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b);
		static float PointToLineSegment(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b);

		static Eigen::Vector3f ClosestPointOnTriangle(const Eigen::Vector3f& p, const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c);
	};

	class Intersection
	{
	public:
		enum class PointToTriangleType
		{
			Outside,
			Inside,
			OnEdge,
			OnVertex
		};

		struct PointToTriangleResult
		{
			PointToTriangleType type = PointToTriangleType::Outside;

			// Distance from the point to the triangle plane (absolute value)
			float plane_distance = 0.0f;

			// For OnVertex: 0, 1, or 2 (v0, v1, v2). Otherwise -1.
			int vertex_index = -1;

			// For OnEdge: 0 (v0-v1), 1 (v1-v2), or 2 (v2-v0). Otherwise -1.
			int edge_index = -1;
		};

		enum TriangleTriangleIntersectionType
		{
			None,
			Point,
			Segment,
			Triangle,
			Coplanar
		};

		struct TriangleIntersectionResult
		{
			TriangleTriangleIntersectionType type;
			Eigen::Vector3f pointA;
			Eigen::Vector3f pointB;
		};

		static bool PointToLineSegment(
			const Eigen::Vector3f& p,
			const Eigen::Vector3f& a,
			const Eigen::Vector3f& b,
			float epsilon = EPSILON);

		static bool PointToRay(
			const Eigen::Vector3f& p,
			const Eigen::Vector3f& origin,
			const Eigen::Vector3f& dir,
			float epsilon = EPSILON);

		static PointToTriangleResult PointToTriangle(
			const Eigen::Vector3f& p,
			const Eigen::Vector3f& v0,
			const Eigen::Vector3f& v1,
			const Eigen::Vector3f& v2,
			float epsilon = EPSILON);

		static bool RayToTriangle(
			const Eigen::Vector3f& origin,
			const Eigen::Vector3f& dir,
			const Eigen::Vector3f& v0,
			const Eigen::Vector3f& v1,
			const Eigen::Vector3f& v2,
			Eigen::Vector3f& intersection_point,
			float epsilon = EPSILON);

		// Finite segment vs triangle intersection.
		// Unlike RayToTriangle, t is clamped to the segment range so points
		// beyond the segment endpoints are rejected.
		static bool SegmentToTriangle(
			const Eigen::Vector3f& s0,
			const Eigen::Vector3f& s1,
			const Eigen::Vector3f& v0,
			const Eigen::Vector3f& v1,
			const Eigen::Vector3f& v2,
			Eigen::Vector3f& out_point,
			float epsilon = EPSILON);

		// Clips a segment that lies in (or within epsilon of) the plane
		// of the given triangle against that triangle, in 2D. Endpoints
		// are first projected onto the triangle plane. Returns false when
		// the clipped piece is empty or shorter than epsilon. This is the
		// robust path for tangent configurations (a triangle edge lying
		// in the other triangle's plane), where the generic transversal
		// intersection depends on fragile on-boundary epsilon decisions.
		static bool CoplanarSegmentToTriangle(
			const Eigen::Vector3f& s0,
			const Eigen::Vector3f& s1,
			const Eigen::Vector3f& v0,
			const Eigen::Vector3f& v1,
			const Eigen::Vector3f& v2,
			Eigen::Vector3f& out_p0,
			Eigen::Vector3f& out_p1,
			float epsilon = EPSILON);

		static TriangleIntersectionResult TriangleToTriangle(
			const Eigen::Vector3f& a0, const Eigen::Vector3f& a1, const Eigen::Vector3f& a2,
			const Eigen::Vector3f& b0, const Eigen::Vector3f& b1, const Eigen::Vector3f& b2,
			Eigen::Vector3f& intersectionA,
			Eigen::Vector3f& intersectionB,
			float epsilon = EPSILON);

		static bool AABBtoAABB(
			const Eigen::Vector3f& a_min, const Eigen::Vector3f& a_max,
			const Eigen::Vector3f& b_min, const Eigen::Vector3f& b_max);
	};

	class Mesh : public OpenMesh::TriMesh_ArrayKernelT<>
	{
	public:
		Mesh();

		void Build(const std::vector<Eigen::Vector3f>& points, const std::vector<Eigen::Vector3i>& indices);
		void BuildSpatialHashMap();
		void GetFaceVertices(OpenMesh::FaceHandle f_handle, Eigen::Vector3f& v0, Eigen::Vector3f& v1, Eigen::Vector3f& v2) const;

		void QueryOverlappingFaces(const Eigen::Vector3f& aabb_min, const Eigen::Vector3f& aabb_max, std::vector<int>& out_faces) const;

		// Builds a watertight axis-aligned box centered at the given point,
		// then applies the given transform. 8 vertices, 12 triangles,
		// CCW winding viewed from outside. If the transform has a negative
		// determinant (mirroring), winding is flipped to keep normals outward.
		// All extents must be positive. Returns false on invalid input.
		bool BuildBox(
			const Eigen::Vector3f& center,
			const Eigen::Vector3f& size,
			const Eigen::Matrix4f& transform = Eigen::Matrix4f::Identity());

		// Builds a watertight solid box whose TOP surface is displaced by
		// a sine wave along x: z_top(u) = +size.z/2 + amplitude * sin(2*pi*wave_count*u),
		// u in [0,1] across the x extent. Bottom and side walls stay flat.
		// Centered at the given point, then the given transform is applied.
		// If the transform has a negative determinant (mirroring), winding
		// is flipped to keep normals outward.
		// amplitude must satisfy 0 <= amplitude < size.z so the top can
		// never dip below the bottom. segments_x / segments_y >= 1.
		// Returns false on invalid input.
		bool BuildSineWaveBox(
			const Eigen::Vector3f& center,
			const Eigen::Vector3f& size,
			float amplitude,
			float wave_count,
			int segments_x,
			int segments_y,
			const Eigen::Matrix4f& transform = Eigen::Matrix4f::Identity());

		// Builds a watertight solid 3D mesh from a text string, intended as
		// CSG input. Glyph outlines come from a TTF font, are triangulated
		// with CDT (holes handled), and extruded along z centered at z = 0,
		// then the given transform is applied. If the transform has a negative
		// determinant (mirroring), winding is flipped to keep normals outward.
		// size is the glyph pixel height in world units, depth must be > 0.
		// Returns false on font or geometry failure.
		bool Build3DText(
			const std::string& text,
			const std::string& font_path,
			float size,
			float depth,
			const Eigen::Matrix4f& transform = Eigen::Matrix4f::Identity());

		// Seam-bounded flood fill: assigns a patch id to every face,
		// treating edges flagged in edge_is_seam as walls. Edge indices
		// beyond the flag array are treated as not seam. Returns the
		// patch count.
		int BuildSeamBoundedPatches(
			const std::vector<char>& edge_is_seam,
			std::vector<int>& out_face_patch) const;

		// Splits this mesh into per-patch welded geometry, using seam
		// edges as patch boundaries. Patch ids match the flood fill order
		// of BuildSeamBoundedPatches. Returns the patch count.
		int SplitMeshBySeam(
			const std::vector<char>& edge_is_seam,
			std::vector<std::vector<Eigen::Vector3f>>& out_patch_points,
			std::vector<std::vector<Eigen::Vector3i>>& out_patch_indices) const;

		std::vector<std::vector<OpenMesh::VertexHandle>> GetBorderLoops() const;
		
		Eigen::AlignedBox3f GetBoundingBox() const;

		// Shortest path along mesh edges from start to end, computed with
		// Dijkstra over the vertex graph using Euclidean edge lengths as
		// weights. This is the edge-restricted discrete geodesic: exact on
		// the edge graph and robust, but the path is constrained to mesh
		// edges rather than crossing face interiors. out_path lists vertices
		// from start to end inclusive; out_distance is the summed edge
		// length. Returns false when a handle is invalid or deleted, or when
		// no path connects the two vertices.
		bool GeodesicShortestPath(
			OpenMesh::VertexHandle start,
			OpenMesh::VertexHandle end,
			std::vector<OpenMesh::VertexHandle>& out_path,
			float& out_distance) const;

		// Closest point on any live face to the query point. Uses the
		// spatial hash with an expanding query box: once the best distance
		// is within the box half-extent, no closer face can lie outside, so
		// the result is exact. out_face is the face holding the closest
		// point. Returns false when the mesh is empty.
		bool ClosestPointOnSurface(
			const Eigen::Vector3f& query,
			Eigen::Vector3f& out_point,
			OpenMesh::FaceHandle& out_face) const;

		// Builds a smooth surface-conforming curve from hand-picked points.
		// The points become control points of a centripetal Catmull-Rom
		// spline, sampled samples_per_segment times per span, then every
		// sample is projected onto the nearest surface point. closed makes
		// the curve a loop (the dental margin case). out_curve lists the
		// surface points; out_faces[i] is the face carrying out_curve[i].
		// Requires a built spatial hash. Returns false on degenerate input.
		bool BuildSmoothSurfaceCurve(
			const std::vector<Eigen::Vector3f>& picked_points,
			bool closed,
			int samples_per_segment,
			std::vector<Eigen::Vector3f>& out_curve,
			std::vector<OpenMesh::FaceHandle>& out_faces) const;

		// Nearest mesh vertex to the query point. Finds the closest surface
		// point first, then picks the nearest of that face's three vertices.
		// Returns an invalid handle only when the mesh is empty.
		OpenMesh::VertexHandle ClosestVertex(const Eigen::Vector3f& query) const;

		// Builds a smooth curve that never leaves the surface, from
		// hand-picked points. Each pick is snapped to a vertex; consecutive
		// picks are joined by geodesic vertex paths to form a dense polyline
		// that already lies on the surface; the polyline is then Laplacian
		// smoothed with every moved point reprojected onto the surface, while
		// the picked anchors stay fixed so the curve passes through them.
		// closed makes a loop. Requires a built spatial hash. Returns false
		// on degenerate input.
		bool BuildGeodesicSurfaceCurve(
			const std::vector<Eigen::Vector3f>& picked_points,
			bool closed,
			int smoothing_iterations,
			float smoothing_strength,
			std::vector<Eigen::Vector3f>& out_curve,
			std::vector<OpenMesh::FaceHandle>& out_faces) const;

		// Builds the geodesic vertex loop used as a cut seam from picked
		// points, the same anchors and geodesics that BuildGeodesicSurfaceCurve
		// visualizes. The loop is the unsmoothed vertex sequence so every
		// consecutive pair is a real mesh edge. out_loop lists vertices once
		// around the loop (no repeated closing vertex). Returns false if a
		// geodesic is missing, if any consecutive pair is not an edge, or if
		// the loop is not simple (revisits a vertex). closed cut only.
		bool BuildCutSeamVertexLoop(
			const std::vector<Eigen::Vector3f>& picked_points,
			std::vector<OpenMesh::VertexHandle>& out_loop) const;

		// Converts a vertex loop into a seam edge flag vector indexed by edge
		// index, sized to n_edges(). Each consecutive pair (cyclic) must be a
		// real edge; returns false otherwise. The flags feed SplitMeshBySeam.
		bool BuildSeamFromVertexLoop(
			const std::vector<OpenMesh::VertexHandle>& loop,
			std::vector<char>& out_edge_is_seam) const;

		// Splits the mesh along a picked margin loop into exactly two welded
		// patches. Fails (returns false, emits nothing) unless the seam is a
		// valid simple loop that separates the surface into two parts. The
		// patch with the smaller total triangle area is returned in the
		// _small outputs, the other in _large. Requires a built spatial hash.
		bool SplitByMarginLoop(
			const std::vector<Eigen::Vector3f>& picked_points,
			std::vector<Eigen::Vector3f>& out_small_points,
			std::vector<Eigen::Vector3i>& out_small_indices,
			std::vector<Eigen::Vector3f>& out_large_points,
			std::vector<Eigen::Vector3i>& out_large_indices) const;

	protected:
		Eigen::Vector3f grid_min;
		Eigen::Vector3f grid_max;
		Eigen::Vector3f grid_cell_size;

		robin_hood::unordered_map<Eigen::Vector3i, std::vector<int>, Int3Hash, Int3Equal> hash_map;

		// Single-source Dijkstra over the vertex graph. Fills out_dist and
		// out_prev (predecessor vertex index, -1 for none), both sized to
		// n_vertices() and indexed by vertex handle index. Stops early once
		// the target vertex is finalized. Deleted vertices and edges are
		// skipped. Returns true when end is reachable.
		bool RunDijkstraVertexGraph(
			OpenMesh::VertexHandle start,
			OpenMesh::VertexHandle end,
			std::vector<float>& out_dist,
			std::vector<int>& out_prev) const;

		// Concatenates geodesic vertex paths between consecutive anchors into
		// one polyline. out_is_anchor[i] is non-zero where out_points[i] is a
		// picked anchor. The shared join vertex between adjacent segments is
		// emitted once; a closed loop's final vertex (equal to the first) is
		// not duplicated. A missing geodesic is bridged directly and reported.
		bool BuildGeodesicAnchorPolyline(
			const std::vector<OpenMesh::VertexHandle>& anchors,
			bool closed,
			std::vector<Eigen::Vector3f>& out_points,
			std::vector<char>& out_is_anchor) const;

		// Laplacian smoothing constrained to the surface. Each non-anchor
		// point moves toward the midpoint of its neighbors by strength, then
		// is reprojected onto the nearest surface point so it cannot drift
		// off the mesh. Anchors and (for open chains) endpoints stay fixed.
		// Updates are applied Jacobi-style (all from the previous iteration).
		void SmoothCurveOnSurface(
			std::vector<Eigen::Vector3f>& curve,
			const std::vector<char>& is_anchor,
			bool closed,
			int iterations,
			float strength) const;
	};

	class Operator
	{
	public:
		virtual ~Operator() {}
		virtual bool Execute() = 0;
	};

	// Extracts the live faces of a mesh as an indexed triangle set,
	// reusing the mesh vertex indices.
	void ExtractMeshSoup(
		const Mesh* mesh,
		std::vector<Eigen::Vector3f>& out_points,
		std::vector<Eigen::Vector3i>& out_indices);

	// Topology validation of an indexed triangle set: duplicate and
	// degenerate triangles, edge manifoldness, bowtie vertices, short
	// edges and near-coincident vertex pairs up to near_pair_radius
	// (tiered, edge-connected pairs excluded). The tiers show what an
	// external tool would fuse when welding at a coarser tolerance than
	// this pipeline's EPSILON.
	bool ValidateTriangleSoup(
		const std::vector<Eigen::Vector3f>& points,
		const std::vector<Eigen::Vector3i>& indices,
		const char* label,
		float near_pair_radius);
	
	enum class IntersectionType
	{
		None,
		Vertex,
		Edge,
		Face
	};

	struct IntersectionResult
	{
		IntersectionType type = IntersectionType::None;
		float t = std::numeric_limits<float>::max();
		OpenMesh::VertexHandle vh;
		OpenMesh::EdgeHandle eh;
		OpenMesh::FaceHandle fh;
		Eigen::Vector3f hit_point;
	};
}