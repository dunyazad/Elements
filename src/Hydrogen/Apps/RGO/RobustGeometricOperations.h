#pragma once

#include "RGOCommon.h"

namespace RGO
{
	class OperatorCreateSkirt : public Operator
	{
	public:
		OperatorCreateSkirt(Mesh* mesh, float skirt_height);
		OperatorCreateSkirt(Mesh* mesh, float skirt_height, const Eigen::Vector3f& direction);
		virtual bool Execute() override;

		bool ComputeLoopDirection(
			const std::vector<OpenMesh::VertexHandle>& loop,
			Eigen::Vector3f& out_direction) const;
	private:
		bool BuildSkirtForLoop(
			const std::vector<OpenMesh::VertexHandle>& loop,
			const Eigen::Vector3f& offset,
			std::vector<Eigen::Vector3f>& out_points,
			std::vector<Eigen::Vector3i>& out_indices) const;

		Mesh* meshTarget = nullptr;
		float skirt_height = 0.0f;
		Eigen::Vector3f skirt_direction = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
		bool has_explicit_direction = false;
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

		// Node-level canonicalization. Endpoints are welded into logical
		// nodes FIRST, then ONE snap decision is made per node against
		// the union of features of all faces incident to that node.
		// Vertices win over edges across classes, the nearest candidate
		// wins within a class, and the decision is iterated to a fixed
		// point. This makes canonicalization idempotent by construction,
		// removing the snap-vs-weld interference of the previous
		// per-face-pair snapping order at the root.
		enum class SnapClass
		{
			Free = 0,
			Edge = 1,
			Vertex = 2
		};

		void GatherNodeFaces(
			const std::vector<std::pair<int, int>>& segment_nodes,
			size_t node_count,
			std::vector<std::vector<int>>& out_faces_a,
			std::vector<std::vector<int>>& out_faces_b) const;

		bool SnapPointToNearestVertex(
			const Eigen::Vector3f& p,
			const std::vector<int>& faces_a,
			const std::vector<int>& faces_b,
			Eigen::Vector3f& out_point) const;

		bool SnapPointToNearestEdge(
			const Eigen::Vector3f& p,
			const std::vector<int>& faces_a,
			const std::vector<int>& faces_b,
			Eigen::Vector3f& out_point) const;

		SnapClass SnapNodePosition(
			const Eigen::Vector3f& p,
			const std::vector<int>& faces_a,
			const std::vector<int>& faces_b,
			Eigen::Vector3f& out_point) const;

		void MergeSnappedNodes(
			std::vector<Eigen::Vector3f>& node_positions,
			std::vector<SnapClass>& node_snap_class,
			std::vector<std::vector<int>>& node_faces_a,
			std::vector<std::vector<int>>& node_faces_b,
			std::vector<int>& out_node_remap) const;

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

		bool ValidateResultSoup(
			const std::vector<Eigen::Vector3f>& points,
			const std::vector<Eigen::Vector3i>& indices,
			const char* label) const;

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

	class OperatorRemesh : public Operator
	{
	public:
		// Per-pass enable flags, so a single pass can be isolated and its
		// effect on the invariants observed in the log. Default is all on.
		struct PassFlags
		{
			bool split = true;
			bool collapse = true;
			bool flip = true;
			bool relax = true;
		};

		// target_edge_length is an absolute world-space length. Edges longer
		// than 4/3 of it are split, shorter than 4/5 are collapsed, so the
		// post-remesh edge lengths concentrate around target_edge_length.
		// iterations is the number of split/collapse/flip/relax rounds.
		// feature_angle_degree is the dihedral angle above which an interior
		// edge is treated as a sharp feature and preserved.
		OperatorRemesh(
			Mesh* target,
			float target_edge_length,
			int iterations,
			float feature_angle_degree,
			const PassFlags& passes = PassFlags());

		virtual bool Execute() override;

		size_t GetSplitCount() const { return total_split_count; }
		size_t GetCollapseCount() const { return total_collapse_count; }
		size_t GetFlipCount() const { return total_flip_count; }
		size_t GetFeatureEdgeCount() const { return feature_edge_count; }

		// Detected feature edges as endpoint pairs, for visualization.
		void GetFeatureEdgeLines(std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& out_lines) const;

		// Vertices flagged by the last ValidateFeatureParity call, for
		// visualization as spheres.
		const std::vector<Eigen::Vector3f>& GetBadFeatureVertices() const { return bad_feature_vertices; }

		// Centroids flagged by the last CountFlippedFaces call.
		const std::vector<Eigen::Vector3f>& GetFlippedFaceCenters() const { return flipped_face_centers; }

	protected:
		enum VertexFeature
		{
			// Interior vertex: free to move on the tangent plane.
			VertexFree = 0,

			// On exactly two feature edges meeting smoothly: may slide
			// along the feature-curve tangent only.
			VertexOnFeature = 1,

			// A feature curve junction, endpoint, or a sharp bend in a
			// feature curve: pinned, never moved.
			VertexCorner = 2
		};

		// Pass 0, once before iterating: freeze the input surface as the
		// reprojection target so smoothing cannot accumulate shrinkage.
		void BuildReferenceSnapshot();

		// Pass 0: flag feature edges (dihedral over threshold, plus all
		// boundary edges) and classify each vertex.
		void DetectFeatureEdges();
		void ClassifyFeatureVertex(OpenMesh::VertexHandle vh, float corner_cos_threshold);

		// Closest point on the original surface to p. Used to keep relaxed
		// interior vertices on the input shape. Returns false if empty.
		bool ReprojectToReference(const Eigen::Vector3f& p, Eigen::Vector3f& out_point) const;

		// The four operators. Split/collapse/flip return their edit count.
		size_t SplitLongEdges();
		bool CollapseRingStaysValid(
			OpenMesh::VertexHandle v_ring,
			OpenMesh::VertexHandle v_from,
			OpenMesh::VertexHandle v_to,
			const Eigen::Vector3f& survivor) const;
		bool CollapseKeepsFacesValid(
			OpenMesh::VertexHandle v_from,
			OpenMesh::VertexHandle v_to,
			const Eigen::Vector3f& survivor) const;
		size_t CollapseShortEdges();
		bool FlipPreservesOrientation(
			OpenMesh::VertexHandle va,
			OpenMesh::VertexHandle vb,
			OpenMesh::VertexHandle vc,
			OpenMesh::VertexHandle vd) const;
		size_t FlipToImproveValence();
		bool RelaxMoveKeepsFacesValid(
			OpenMesh::VertexHandle vh,
			const Eigen::Vector3f& new_pos) const;
		bool ComputeRelaxTarget(
			OpenMesh::VertexHandle vh,
			int cls,
			Eigen::Vector3f& out_target) const;
		void TangentialRelaxation();

		int TargetValence(OpenMesh::VertexHandle vh) const;

		// Verification, callable after any pass. Each returns a violation
		// count (0 == clean) and logs the first few offenders.
		size_t ValidateManifold(const char* stage_label) const;
		size_t ValidateFeatureParity(const char* stage_label) const;
		bool FaceNormalAndCentroid(
			OpenMesh::FaceHandle fh,
			Eigen::Vector3f& out_normal,
			Eigen::Vector3f& out_centroid) const;
		size_t CountFlippedFaces(const char* stage_label) const;
		size_t CountBoundaryEdges() const;

		// Runs the full invariant suite after a stage and returns true when
		// all hold. boundary_expected is the input boundary count that must
		// be preserved (0 for a closed solid).
		bool ValidateStage(const char* stage_label, size_t boundary_expected) const;

		void DiagnoseFeatureOnlyFaces(const char* stage_label) const;

		float TriangleAspectRatio(
			const Eigen::Vector3f& a,
			const Eigen::Vector3f& b,
			const Eigen::Vector3f& c) const;

		void DiagnoseSurfaceDeviation(const char* stage_label) const;

		bool ClosestPointOnReference(
			const Eigen::Vector3f& p,
			Eigen::Vector3f& out_point) const;

		bool ReferenceNormalAt(
			const Eigen::Vector3f& p,
			Eigen::Vector3f& out_normal) const;

		float MinAngleOfTwoTriangles(
			const Eigen::Vector3f& a,
			const Eigen::Vector3f& b,
			const Eigen::Vector3f& c,
			const Eigen::Vector3f& d) const;

		bool VertexRingHasFold(OpenMesh::VertexHandle vh) const;

		bool SplitKeepsFacesValid(
			OpenMesh::EdgeHandle eh,
			const Eigen::Vector3f& new_pos) const;

		Mesh* meshTarget = nullptr;
		float target_edge_length = 0.0f;
		float edge_high = 0.0f;
		float edge_low = 0.0f;
		int iterations = 0;
		float feature_angle_degree = 0.0f;
		float corner_cos_threshold = 0.0f;
		PassFlags passes;

		Mesh reference_mesh;

		OpenMesh::EPropHandleT<bool> prop_edge_feature;
		OpenMesh::VPropHandleT<int> prop_vertex_feature;

		size_t input_boundary_count = 0;
		size_t total_split_count = 0;
		size_t total_collapse_count = 0;
		size_t total_flip_count = 0;
		size_t feature_edge_count = 0;

		// Positions of vertices that failed the most recent feature-parity
		// check (degree != 2 but not pinned as a corner). Collected for
		// external visualization. Cleared at the start of each check.
		mutable std::vector<Eigen::Vector3f> bad_feature_vertices;

		// Centroids of faces flagged by the last CountFlippedFaces call,
		// for visualization. Cleared at the start of each call.
		mutable std::vector<Eigen::Vector3f> flipped_face_centers;

		mutable std::vector<Eigen::Vector3f> prev_stage_fold_mids;
	};

	class OperatorValidate : public Operator
	{
	public:
		OperatorValidate(std::function<bool()> predicate, const std::string& label);
		virtual bool Execute() override;

	private:
		std::function<bool()> predicate;
		std::string label;
	};

	class Pipeline
	{
	public:
		Pipeline();

		template <typename T, typename... Args>
		T* RegisterOperator(const std::string& label, Args&&... args)
		{
			std::unique_ptr<T> owned = std::make_unique<T>(std::forward<Args>(args)...);
			T* observer = owned.get();
			stages.push_back(Stage{ std::move(owned), label });
			return observer;
		}

		void RegisterValidator(std::function<bool()> predicate, const std::string& label);

		bool Run();
		void Clear();

		int GetFailedIndex() const { return failed_index; }
		const std::string& GetFailedLabel() const { return failed_label; }
		size_t GetStageCount() const { return stages.size(); }

	private:
		struct Stage
		{
			std::unique_ptr<Operator> op;
			std::string label;
		};

		std::vector<Stage> stages;
		int failed_index = -1;
		std::string failed_label;
	};
}