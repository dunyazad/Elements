#pragma once

#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstring>
#include <cstdint>
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

	protected:
		Eigen::Vector3f grid_min;
		Eigen::Vector3f grid_max;
		Eigen::Vector3f grid_cell_size;

		robin_hood::unordered_map<Eigen::Vector3i, std::vector<int>, Int3Hash, Int3Equal> hash_map;
	};

	class Operator
	{
	public:
		virtual ~Operator() {}
		virtual bool Execute() = 0;
	};

	class OperatorIntersectionLoops : public Operator
	{
	public:
		struct IntersectionSegment
		{
			Eigen::Vector3f p0;
			Eigen::Vector3f p1;
			int face_a = -1;
			int face_b = -1;
		};

		struct IntersectionLoop
		{
			// Ordered welded points along the curve.
			// For closed loops the last point connects back to the first;
			// the first point is NOT duplicated at the end.
			std::vector<Eigen::Vector3f> points;

			// Indices into GetSegments(), in traversal order
			std::vector<int> segment_indices;

			// False means an open chain. For two watertight inputs every
			// intersection curve must be closed, so an open chain signals
			// missed segments upstream and should be investigated.
			bool closed = false;
		};

		OperatorIntersectionLoops(Mesh* a, Mesh* b);

		virtual bool Execute() override;

		const std::vector<IntersectionSegment>& GetSegments() const { return segments; }
		const std::vector<IntersectionLoop>& GetLoops() const { return loops; }
		const robin_hood::unordered_map<int, std::vector<int>>& GetSegmentIndicesByFaceA() const { return segments_by_face_a; }
		const robin_hood::unordered_map<int, std::vector<int>>& GetSegmentIndicesByFaceB() const { return segments_by_face_b; }

		size_t GetValidationFailureCount() const { return validation_failure_count; }
		size_t GetCanonicalSnapCount() const { return canonical_snap_count; }
		size_t GetCanonicalValidationFailureCount() const { return canonical_validation_failure_count; }
		size_t GetOpenChainCount() const { return open_chain_count; }
		// Coplanar overlapping (face_a, face_b) pairs, recorded during
		// segment collection for the OnSurface classification stage.
		const std::vector<std::pair<int, int>>& GetCoplanarFacePairs() const { return coplanar_face_pairs; }

	protected:
		void CollectSegmentsForFaceB(
			int face_b_index,
			std::vector<IntersectionSegment>& out_segments,
			std::vector<std::pair<int, int>>& out_coplanar_pairs) const;
		void BuildFaceLookupTables();
		void BuildLoops();

		// Canonicalization: snaps segment endpoints to mesh features
		// (vertices first, then edges) and welds all endpoints globally so
		// that points meant to be identical are bit-identical. This is the
		// input points within a single face.
		void CanonicalizeSegments();

		bool TrySnapToFaceVertex(
			const Eigen::Vector3f& p,
			const Mesh* mesh,
			int face_index,
			Eigen::Vector3f& out_point) const;

		bool TrySnapToFaceEdge(
			const Eigen::Vector3f& p,
			const Mesh* mesh,
			int face_index,
			Eigen::Vector3f& out_point) const;

		Eigen::Vector3f CanonicalizePoint(
			const Eigen::Vector3f& p,
			int face_a,
			int face_b,
			bool& out_snapped) const;

		// Canonicalization validation: checks the invariants that make
		// downstream CDT safe. Must run after BuildFaceLookupTables.
		bool ValidateCanonicalization();
		bool ValidateCanonicalIdempotence();
		bool ValidateNoDegenerateSegments();
		bool ValidateCanonicalConstraintSpacing();

		size_t CheckConstraintSpacingForFace(
			const Mesh* mesh,
			int face_index,
			const std::vector<int>& segment_indices,
			const char* mesh_label) const;

		int WeldEndpoint(
			const Eigen::Vector3f& p,
			robin_hood::unordered_map<Eigen::Vector3i, std::vector<int>, Int3Hash, Int3Equal>& cell_to_nodes,
			std::vector<Eigen::Vector3f>& node_positions) const;

		IntersectionLoop TraceLoopFromSegment(
			int start_segment,
			const std::vector<std::pair<int, int>>& segment_nodes,
			const std::vector<std::vector<int>>& node_to_segments,
			const std::vector<Eigen::Vector3f>& node_positions,
			std::vector<char>& segment_used) const;

		Mesh* meshA = nullptr;
		Mesh* meshB = nullptr;

		std::vector<IntersectionSegment> segments;
		std::vector<IntersectionLoop> loops;
		robin_hood::unordered_map<int, std::vector<int>> segments_by_face_a;
		robin_hood::unordered_map<int, std::vector<int>> segments_by_face_b;

		bool ValidateSegmentEndpoints();

		size_t validation_failure_count = 0;
		size_t canonical_snap_count = 0;
		size_t canonical_validation_failure_count = 0;

		struct InputBoundaryEdge
		{
			Eigen::Vector3f a;
			Eigen::Vector3f b;
		};

		void CollectInputBoundaryEdges(const Mesh* mesh, const char* label);
		float DistanceToInputBoundary(const Eigen::Vector3f& p) const;

		// Open chain validation: an intersection curve may legitimately
		// end where it runs off the open border of an input surface, but
		// an endpoint stopping in the middle of both surfaces means
		// missed segments and is a hard failure.
		bool ValidateOpenChains();

		size_t open_chain_count = 0;

		std::vector<InputBoundaryEdge> input_boundary_edges;

		// Failure forensics: prints every canonical segment endpoint
		// within a small radius of the given point, with exact distances,
		// owning segments and faces, and the count of bit-identical
		// endpoints. Distinguishes a canonicalization split (a partner
		// node a few EPSILON away) from a genuinely missing segment
		// (nothing nearby).
		void DumpEndpointNeighborhood(const Eigen::Vector3f& p, const char* tag) const;

		std::vector<std::pair<int, int>> coplanar_face_pairs;
	};

	class OperatorCoRefine : public Operator
	{
	public:
		OperatorCoRefine(Mesh* a, Mesh* b, const OperatorIntersectionLoops* loops);

		virtual bool Execute() override;

		size_t GetNewBoundaryEdgeCountA() const { return new_boundary_edge_count_a; }
		size_t GetNewBoundaryEdgeCountB() const { return new_boundary_edge_count_b; }

	protected:
		struct FaceRefinementInput
		{
			// Unique extra points to insert into this face's triangulation
			std::vector<Eigen::Vector3f> points;

			// Intersection segments restricted to this face, as point pairs
			std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> constraints;
		};

		struct BoundaryEdge
		{
			Eigen::Vector3f a;
			Eigen::Vector3f b;
			Eigen::Vector3f aabb_min;
			Eigen::Vector3f aabb_max;
		};

		bool RefineMesh(
			Mesh* mesh,
			const robin_hood::unordered_map<int, std::vector<int>>& segment_indices_by_face,
			const char* mesh_label,
			size_t& out_new_boundary_edge_count);

		void GatherFaceInputs(
			const robin_hood::unordered_map<int, std::vector<int>>& segment_indices_by_face,
			robin_hood::unordered_map<int, FaceRefinementInput>& out_inputs) const;

		void PropagateEdgePointsToNeighbors(
			Mesh* mesh,
			robin_hood::unordered_map<int, FaceRefinementInput>& inputs) const;

		bool TriangulateFace(
			const Mesh* mesh,
			int face_index,
			const FaceRefinementInput& input,
			std::vector<Eigen::Vector3f>& out_soup) const;

		void AddUniquePoint(std::vector<Eigen::Vector3f>& pts, const Eigen::Vector3f& p) const;

		void WeldTriangleSoup(
			const std::vector<Eigen::Vector3f>& soup,
			std::vector<Eigen::Vector3f>& out_points,
			std::vector<Eigen::Vector3i>& out_indices) const;

		void CollectInputBoundaryEdges(const Mesh* mesh, std::vector<BoundaryEdge>& out_edges) const;

		bool IsSubEdgeOfInputBoundary(
			const Eigen::Vector3f& a,
			const Eigen::Vector3f& b,
			const std::vector<BoundaryEdge>& input_boundary) const;

		size_t CountNewBoundaryEdges(
			const Mesh* mesh,
			const std::vector<BoundaryEdge>& input_boundary,
			const char* mesh_label) const;

		size_t CountBoundaryEdges(const Mesh* mesh) const;

		Mesh* meshA = nullptr;
		Mesh* meshB = nullptr;
		const OperatorIntersectionLoops* loop_op = nullptr;

		size_t new_boundary_edge_count_a = 0;
		size_t new_boundary_edge_count_b = 0;
	};

	class OperatorBoolean : public Operator
	{
	public:
		enum Type { Union, Intersection, Difference };

		enum class FaceSide
		{
			Unknown,
			Inside,
			Outside,

			// The patch lies on the other mesh's surface. Same means the
			// two coincident surfaces face the same direction, Opposite
			// means they face each other or away from each other. The
			// distinction decides which boolean operations keep the patch.
			OnSurfaceSame,
			OnSurfaceOpposite
		};

		OperatorBoolean(
			Type t,
			Mesh* a,
			Mesh* b,
			const OperatorIntersectionLoops* loops,
			Mesh* result);

		virtual bool Execute() override;

		struct MeshSideData
		{
			// Per edge index: 1 when the edge lies on the intersection curve
			std::vector<char> edge_is_seam;

			// Per face index: patch id from seam-bounded flood fill
			std::vector<int> face_patch;

			int patch_count = 0;

			// Per patch id: classification against the other mesh
			std::vector<FaceSide> patch_side;
		};

		size_t GetResultBoundaryEdgeCount() const { return result_boundary_edge_count; }

		// Read access for visualization and external verification of the
		// seam and patch stage.
		const MeshSideData& GetSideDataA() const { return data_a; }
		const MeshSideData& GetSideDataB() const { return data_b; }

	protected:
		struct BoundaryEdge
		{
			Eigen::Vector3f a;
			Eigen::Vector3f b;
			Eigen::Vector3f aabb_min;
			Eigen::Vector3f aabb_max;
		};

		// Solid assembly: both inputs are closed volumes, structures in
		// data_a and data_b are already built and validated by Execute.
		// The result is a closed solid assembled from patches of both
		// meshes, with coplanar overlap regions taken from mesh A once.
		bool AssembleSolidBoolean();

		// Trim assembly: exactly one input is open. An open shell has no
		// volume, so ray parity against it is undefined and its patches
		// cannot be stitched with the other mesh into a volume boundary.
		// The only well defined result is the open mesh trimmed by the
		// closed solid, which is an open surface.
		bool AssembleOpenMeshTrim(Mesh* open_mesh, const MeshSideData& data, const char* open_label);

		void CollectFacesForBoolean(
			const Mesh* mesh,
			const MeshSideData& data,
			bool keep_inside,
			bool keep_outside,
			bool keep_on_same,
			bool keep_on_opposite,
			bool flip_winding,
			std::vector<Eigen::Vector3f>& out_soup) const;

		void ReportPatchStatistics(const Mesh* mesh, const MeshSideData& data, const char* label) const;

		bool BuildSeamEdgeFlags(const Mesh* mesh, MeshSideData& data, const char* label) const;
		bool ValidateSeamIntegrity(const Mesh* mesh, const MeshSideData& data, const char* label) const;
		bool BuildFacePatches(const Mesh* mesh, MeshSideData& data, const char* label) const;
		bool ClassifyPatches(const Mesh* mesh, const Mesh* other, MeshSideData& data, const char* label) const;
		FaceSide RobustPointInMesh(const Eigen::Vector3f& p, const Eigen::Vector3f& query_normal, const Mesh* other) const;
		bool ValidatePatchAdjacency(const Mesh* mesh, const MeshSideData& data, const char* label) const;

		void WeldTriangleSoup(
			const std::vector<Eigen::Vector3f>& soup,
			std::vector<Eigen::Vector3f>& out_points,
			std::vector<Eigen::Vector3i>& out_indices) const;

		void CollectInputBoundaryEdges(const Mesh* mesh, std::vector<BoundaryEdge>& out_edges) const;

		bool IsSubEdgeOfInputBoundary(
			const Eigen::Vector3f& a,
			const Eigen::Vector3f& b,
			const std::vector<BoundaryEdge>& input_boundary) const;

		bool IsSeamSubEdge(const Eigen::Vector3f& a, const Eigen::Vector3f& b) const;

		size_t CountInvalidResultBoundaryEdges(
			const Mesh* mesh,
			const std::vector<BoundaryEdge>& open_input_boundary,
			const char* label) const;

		size_t CountBoundaryEdges(const Mesh* mesh) const;

		Type type;
		Mesh* meshA = nullptr;
		Mesh* meshB = nullptr;
		const OperatorIntersectionLoops* loop_op = nullptr;
		Mesh* result = nullptr;

		// Canonical endpoint position to indices of segments having it as
		// an endpoint. Bit-exact keys are valid because canonicalization
		// made logically identical points bit-identical.
		robin_hood::unordered_map<Eigen::Vector3f, std::vector<int>, Vector3fBitHash, Vector3fBitEqual> endpoint_segments;

		// Seam and patch structures, kept as members so they survive
		// Execute and can be inspected and visualized stage by stage.
		MeshSideData data_a;
		MeshSideData data_b;

		size_t result_boundary_edge_count = 0;
	};

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