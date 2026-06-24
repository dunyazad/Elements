#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
#include <vector>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>

#include <robin_hood/robin_hood.h>

#include <Helium/Helium.h>
#include <Helium/HeliumCore.h>
#include <Helium/Serialization.hpp>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include "RGO/RobustGeometricOperations.h"
#include <OpenMesh/Core/IO/MeshIO.hh>
#include "RGO/HeliumMesh.h"


class AppMeshSplineCut : public App
{
public:
	enum OperationMode
	{
		None,
		AddControlPoint,
		RemoveControlPoint,
	} current_mode = AddControlPoint;

	struct EditState
	{
		std::vector<Eigen::Vector3f> control_points;
		bool margin_closed = false;
	};

public:
	virtual void Initialize() override
	{
		TS(LoadingMesh_A);
		meshes.emplace_back(std::make_unique<RGO::HeliumMesh>());
		RGO::HeliumMesh& mesh_a = *meshes.back();
		LoadWelded(mesh_a, "D:\\Resources\\3D\\STL\\md.stl");
		TE(LoadingMesh_A);

		mesh_a.SetOnEntityCreatedCallback([this, &mesh_a]()
			{
				Helium.CreateEventCallback<MouseButtonEvent>(mesh_a.entity, "3D",
					[this, &mesh_a](Entity entity, const MouseButtonEvent& event)
					{
						if (event.button != MouseButton::Left) return;

						// Release: end any active drag, regardless of where.
						if (event.action == 0)
						{
							EndDrag();
							return;
						}

						if (event.action != 1) return;

						auto current_renderable = Helium.GetComponent<Renderable>(entity);
						if (nullptr == current_renderable) return;

						TS(Picking);
						Eigen::Vector3f local_hit;
						bool is_hit = RaycastToLocalSurface(mesh_a, (float)event.xpos, (float)event.ypos, local_hit);
						TE(Picking);

						if (false == is_hit) return;

						HandlePress(mesh_a, local_hit, event.IsCtrlPressed());
					});

				Helium.CreateEventCallback<MousePositionEvent>(mesh_a.entity, "3D",
					[this, &mesh_a](Entity entity, const MousePositionEvent& event)
					{
						if (dragging_index < 0) return;

						Eigen::Vector3f local_hit;
						if (false == RaycastToLocalSurface(mesh_a, (float)event.xpos, (float)event.ypos, local_hit)) return;

						UpdateDraggedControlPoint(mesh_a, local_hit);
					});

				Helium.CreateEventCallback<KeyEvent>(mesh_a.entity, "3D",
					[this, &mesh_a](Entity entity, const KeyEvent& event)
					{
						if (event.action != 1) return;

						if (KeyCode::A == event.keyCode)
							current_mode = OperationMode::AddControlPoint;
						else if (KeyCode::R == event.keyCode)
							current_mode = OperationMode::RemoveControlPoint;
						else if (KeyCode::N == event.keyCode)
							current_mode = OperationMode::None;
						else if (KeyCode::C == event.keyCode)
							SetMarginClosed(mesh_a, false == margin_closed);
						else if (KeyCode::S == event.keyCode)
							SplitByMarginLine(mesh_a);
						else if (KeyCode::K == event.keyCode)
							MoveSmallPiece(Eigen::Vector3f(0.0f, 0.0f, 2.0f));
						else if (KeyCode::LeftBracket == event.keyCode)
							SaveMarginLineInputToFile(mesh_a);
						else if (KeyCode::RightBracket == event.keyCode)
							LoadMarginLineInputFromFile(mesh_a);
						else if (KeyCode::Z == event.keyCode && event.IsCtrlPressed())
							Undo(mesh_a);
						else if (KeyCode::Y == event.keyCode && event.IsCtrlPressed())
							Redo(mesh_a);
					});
			});

		Helium.AddOnInitializeCallback([this]()
			{
				auto& mesh = *meshes.back();
			});
	}

	virtual void Execute() override
	{
		Helium.AddOnUpdateCallback([this](float time_delta)
			{
				for (auto& m : this->meshes) m->Update();
			});
	}

	void LoadWelded(RGO::HeliumMesh& mesh, const std::string& path)
	{
		STLFormat stl;
		stl.Deserialize(path);

		const std::vector<Eigen::Vector3f>& pts = stl.GetPoints();
		std::vector<Eigen::Vector3f> wp;
		std::vector<Eigen::Vector3i> wi;
		robin_hood::unordered_map<Eigen::Vector3f, int, RGO::Vector3fHash, RGO::Vector3fEqual> vm;

		for (size_t i = 0; i < pts.size(); i += 3)
		{
			Eigen::Vector3i fi;
			for (int j = 0; j < 3; ++j)
			{
				const Eigen::Vector3f& p = pts[i + j];
				auto it = vm.find(p);
				if (it != vm.end()) fi[j] = it->second;
				else { int ni = (int)wp.size(); wp.push_back(p); vm[p] = ni; fi[j] = ni; }
			}
			if (fi[0] != fi[1] && fi[1] != fi[2] && fi[2] != fi[0]) wi.push_back(fi);
		}
		mesh.Build(wp, wi);
	}

	Eigen::Vector4f PatchColor(int index)
	{
		// Golden ratio hue stepping gives well separated colors for
		// any number of patches without a fixed palette.
		float h = std::fmod(static_cast<float>(index) * 0.61803398875f, 1.0f) * 6.0f;
		float s = 0.85f;
		float v = 1.0f;
		int sector = static_cast<int>(h);
		float f = h - static_cast<float>(sector);
		float p = v * (1.0f - s);
		float q = v * (1.0f - s * f);
		float t = v * (1.0f - s * (1.0f - f));
		float r;
		float g;
		float b;
		switch (sector % 6)
		{
		case 0: r = v; g = t; b = p; break;
		case 1: r = q; g = v; b = p; break;
		case 2: r = p; g = v; b = t; break;
		case 3: r = p; g = q; b = v; break;
		case 4: r = t; g = p; b = v; break;
		default: r = v; g = p; b = q; break;
		}
		return Eigen::Vector4f(r, g, b, 1.0f);
	}

	bool RaycastToLocalSurface(RGO::HeliumMesh& mesh, float xpos, float ypos, Eigen::Vector3f& out_local_hit) const
	{
		auto camera_entity = Helium.GetEntityByName("MainCamera");
		auto camera = Helium.GetComponent<Camera>(camera_entity);
		if (nullptr == camera) return false;

		Ray ray = camera->ScreenPointToRay(xpos, ypos, Helium.GetWidth(), Helium.GetHeight());

		Eigen::Vector3f local_origin = ray.origin - mesh.offset;
		RGO::IntersectionResult hit_result;
		if (false == mesh.IntersectGridRay(local_origin, ray.direction, hit_result)) return false;

		out_local_hit = hit_result.hit_point;
		return true;
	}

	void RebuildAndDrawMarginCurve(RGO::HeliumMesh& mesh)
	{
		VD::Clear("MarginLines");
		VD::Clear("MarginCurve");

		margin_curve.clear();
		margin_curve_faces.clear();

		if (control_points.size() < 2) return;
		if (margin_closed && control_points.size() < 3) return;

		std::vector<Eigen::Vector3f> curve;
		std::vector<OpenMesh::FaceHandle> faces;
		if (false == mesh.BuildGeodesicSurfaceCurve(
			control_points, margin_closed,
			margin_smoothing_iterations, margin_smoothing_strength,
			curve, faces))
		{
			return;
		}
		if (curve.size() < 2) return;

		// Keep the result so callers can retrieve the margin line.
		margin_curve = curve;
		margin_curve_faces = faces;

		// Closed loop draws blue, open chain draws green.
		const Eigen::Vector4f color = margin_closed
			? Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f)
			: Eigen::Vector4f(1.0f, 0.0f, 1.0f, 1.0f);

		size_t seg_count = margin_closed ? curve.size() : (curve.size() - 1);
		for (size_t i = 0; i < seg_count; ++i)
		{
			Eigen::Vector3f a = curve[i] + mesh.offset;
			Eigen::Vector3f b = curve[(i + 1) % curve.size()] + mesh.offset;
			VD::AddLine("MarginCurve", a, b, color);
		}
	}

	void RedrawControlPoints(RGO::HeliumMesh& mesh)
	{
		VD::Clear("ControlPoints");
		for (const auto& p : control_points)
		{
			Eigen::Vector3f world_point = p + mesh.offset;
			VD::AddSphere("ControlPoints", world_point, { 0.0f, 0.0f, 1.0f }, 0.3f,
				Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f));
		}
	}

	size_t FindBestInsertSpan(const Eigen::Vector3f& local_point) const
	{
		const size_t n = control_points.size();

		// Fewer than 2 points: nothing to insert between, append at the end.
		if (n < 2) return n;

		const size_t span_count = margin_closed ? n : (n - 1);

		size_t best_span = span_count;     // sentinel meaning "append"
		float best_d2 = std::numeric_limits<float>::max();

		// Mean control-polygon segment length sets the insertion threshold,
		// so the test scales with the mesh without a hard-coded distance.
		float total_len = 0.0f;
		for (size_t s = 0; s < span_count; ++s)
		{
			const Eigen::Vector3f& a = control_points[s];
			const Eigen::Vector3f& b = control_points[(s + 1) % n];
			total_len += (b - a).norm();
		}
		float mean_len = (span_count > 0) ? (total_len / static_cast<float>(span_count)) : 0.0f;

		// A click counts as "on a span" only within this band around it.
		// Beyond it, the click is treated as extending the chain, not as an
		// insertion: this stops a far point from being pulled between two
		// points on the opposite side of a curved loop.
		float insert_band = 0.5f * mean_len;

		for (size_t s = 0; s < span_count; ++s)
		{
			const Eigen::Vector3f& a = control_points[s];
			const Eigen::Vector3f& b = control_points[(s + 1) % n];

			float d2 = RGO::Distance::PointToLineSegmentSquared(local_point, a, b);
			if (d2 < best_d2)
			{
				best_d2 = d2;
				best_span = s;
			}
		}

		// The closest span is still too far: the click is not on the curve,
		// so append rather than inserting into an unrelated span.
		if (best_span == span_count) return n;
		if (best_d2 > insert_band * insert_band) return n;

		// Insert after the span's start point.
		return best_span + 1;
	}

	void AddControlPointAt(RGO::HeliumMesh& mesh, const Eigen::Vector3f& local_point)
	{
		PushUndoSnapshot();

		size_t insert_at = FindBestInsertSpan(local_point);

		if (insert_at >= control_points.size())
		{
			control_points.push_back(local_point);
		}
		else
		{
			control_points.insert(
				control_points.begin() + static_cast<std::ptrdiff_t>(insert_at),
				local_point);
		}

		RedrawControlPoints(mesh);
		RebuildAndDrawMarginCurve(mesh);
	}

	void RemoveNearestControlPoint(RGO::HeliumMesh& mesh, const Eigen::Vector3f& local_point)
	{
		if (control_points.empty()) return;

		PushUndoSnapshot();

		size_t best = 0;
		float best_d2 = (control_points[0] - local_point).squaredNorm();
		for (size_t i = 1; i < control_points.size(); ++i)
		{
			float d2 = (control_points[i] - local_point).squaredNorm();
			if (d2 < best_d2)
			{
				best_d2 = d2;
				best = i;
			}
		}

		control_points.erase(control_points.begin() + static_cast<std::ptrdiff_t>(best));

		RedrawControlPoints(mesh);
		RebuildAndDrawMarginCurve(mesh);
	}

	void SetMarginClosed(RGO::HeliumMesh& mesh, bool closed)
	{
		if (margin_closed == closed) return;

		PushUndoSnapshot();

		margin_closed = closed;
		RebuildAndDrawMarginCurve(mesh);
	}

	int PickControlPointForDrag(const Eigen::Vector3f& local_point) const
	{
		if (control_points.empty()) return -1;

		size_t best = 0;
		float best_d2 = (control_points[0] - local_point).squaredNorm();
		for (size_t i = 1; i < control_points.size(); ++i)
		{
			float d2 = (control_points[i] - local_point).squaredNorm();
			if (d2 < best_d2)
			{
				best_d2 = d2;
				best = i;
			}
		}

		if (best_d2 > drag_pick_radius * drag_pick_radius) return -1;
		return static_cast<int>(best);
	}

	void UpdateDraggedControlPoint(RGO::HeliumMesh& mesh, const Eigen::Vector3f& local_point)
	{
		if (dragging_index < 0) return;
		if (dragging_index >= static_cast<int>(control_points.size())) return;

		control_points[dragging_index] = local_point;

		RedrawControlPoints(mesh);
		RebuildAndDrawMarginCurve(mesh);
	}

	void EndDrag()
	{
		dragging_index = -1;
	}

	void HandlePress(RGO::HeliumMesh& mesh, const Eigen::Vector3f& local_hit, bool ctrl_pressed)
	{
		Eigen::Vector3f world_picked_point = local_hit + mesh.offset;

		if (ctrl_pressed)
		{
			auto camera_entity = Helium.GetEntityByName("MainCamera");
			Helium.GetComponent<Camera>(camera_entity)->SetTarget(world_picked_point);
			return;
		}

		// Dragging takes priority over add/remove: a press landing on an
		// existing control point starts a drag and consumes the click.
		int pick = PickControlPointForDrag(local_hit);
		if (pick >= 0)
		{
			// Snapshot once at drag start so the whole drag is one undo step.
			PushUndoSnapshot();
			dragging_index = pick;
			return;
		}

		dragging_index = -1;

		if (OperationMode::None == current_mode)
		{
			VD::Clear("PickedTriangle");
			VD::Clear("PickedTrianglePoints");
			VD::Clear("Picked");
			VD::AddSphere("Picked", world_picked_point, { 0.0f, 0.0f, 1.0f }, 0.5f,
				Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
		}
		else if (OperationMode::AddControlPoint == current_mode)
		{
			AddControlPointAt(mesh, local_hit);
		}
		else if (OperationMode::RemoveControlPoint == current_mode)
		{
			RemoveNearestControlPoint(mesh, local_hit);
		}
	}

	bool GetMarginLineLocal(std::vector<Eigen::Vector3f>& out_points, bool& out_closed) const
	{
		out_points = margin_curve;
		out_closed = margin_closed;
		return out_points.size() >= 2;
	}

	bool GetMarginLineWorld(RGO::HeliumMesh& mesh, std::vector<Eigen::Vector3f>& out_points, bool& out_closed) const
	{
		out_points.clear();
		out_points.reserve(margin_curve.size());
		for (const auto& p : margin_curve)
		{
			out_points.push_back(p + mesh.offset);
		}
		out_closed = margin_closed;
		return out_points.size() >= 2;
	}

	bool GetMarginLineFaces(std::vector<OpenMesh::FaceHandle>& out_faces) const
	{
		out_faces = margin_curve_faces;
		return out_faces.size() >= 2;
	}

	RGO::HeliumMesh* BuildPieceMesh(
		const std::vector<Eigen::Vector3f>& points,
		const std::vector<Eigen::Vector3i>& indices,
		const Eigen::Vector4f& color,
		const Eigen::Vector3f& offset)
	{
		meshes.emplace_back(std::make_unique<RGO::HeliumMesh>());
		RGO::HeliumMesh* piece = meshes.back().get();

		piece->Build(points, indices);
		piece->mesh_color = color;
		piece->offset = offset;
		piece->is_dirty = true;

		return piece;
	}

	void SplitByMarginLine(RGO::HeliumMesh& source)
	{
		if (false == margin_closed)
		{
			std::cout << "[Warning] SplitByMarginLine: close the margin loop first (press C)." << std::endl;
			return;
		}

		// TEMP forensic: print near-coincident on-edge constraint points.
		// TriangulateFace logs any pair < 10*EPSILON apart on a face edge at
		// full precision (face index, both coords, gap), which pinpoints the
		// T-junction that add_face rejects. Remove once the cause is fixed.
		RGO::Log::diagnostics = true;
		if (margin_curve.size() < 3 || margin_curve_faces.size() != margin_curve.size())
		{
			std::cout << "[Warning] SplitByMarginLine: need a built closed margin curve"
				<< " (press points, then C). Current curve is too short." << std::endl;
			return;
		}

		// Diagnose curve self-proximity before carving: folds here become
		// degree-4 nodes and open chains downstream.
		DiagnoseCurveSelfProximity(margin_curve, margin_closed);

		// Stage 1: turn the surface curve into per-face intersection
		// segments on this single mesh.
		RGO::OperatorIntersectionLoops loop_op(&source, &source);
		if (false == loop_op.ExecuteFromSurfaceCurve(&source, margin_curve, margin_curve_faces, true))
		{
			std::cout << "[Warning] SplitByMarginLine: could not build intersection"
				<< " segments from the margin curve; see messages above." << std::endl;
			return;
		}

		// Stage 2: carve the curve into the mesh. After this, source has
		// been rebuilt so the curve coincides with real mesh edges.
		// Graceful degrade: tolerate a few residual boundary edges from
		// ill-conditioned, near-tangent spots and let seam reconstruction
		// decide success (it requires exactly 2 patches). CDT failures still
		// abort.
		RGO::OperatorCoRefine corefine(&source, &source, &loop_op);
		corefine.allow_residual_boundary = true;
		if (false == corefine.Execute())
		{
			std::cout << "[Warning] SplitByMarginLine: co-refinement failed (CDT"
				<< " error); the curve could not be carved. Adjust the points." << std::endl;
			return;
		}

		size_t residual = corefine.GetNewBoundaryEdgeCountA() + corefine.GetNewBoundaryEdgeCountB();
		if (residual > 0)
		{
			std::cout << "[Warning] SplitByMarginLine: co-refinement left " << residual
				<< " residual boundary edge(s) at ill-conditioned spots; attempting"
				<< " seam reconstruction anyway (degraded carve)." << std::endl;
		}

		// The spatial hash and any cached topology are stale after refine.
		source.BuildSpatialHashMap();

		// Stage 3: flag the carved seam edges, then flood fill into patches.
		std::vector<char> edge_is_seam;
		if (false == BuildSeamFlagsFromSegments(source, loop_op, edge_is_seam))
		{
			std::cout << "[Warning] SplitByMarginLine: seam reconstruction failed after refine." << std::endl;
			return;
		}

		std::vector<std::vector<Eigen::Vector3f>> patch_points;
		std::vector<std::vector<Eigen::Vector3i>> patch_indices;
		int patch_count = source.SplitMeshBySeam(edge_is_seam, patch_points, patch_indices);

		if (2 != patch_count)
		{
			std::cout << "[Error] SplitByMarginLine: seam produced " << patch_count
				<< " patches, expected exactly 2. The margin loop does not cleanly"
				<< " separate the mesh." << std::endl;
			return;
		}

		// Stage 4: smaller patch by area is the extracted piece.
		float area0 = PatchArea(patch_points[0], patch_indices[0]);
		float area1 = PatchArea(patch_points[1], patch_indices[1]);
		int small_idx = (area0 <= area1) ? 0 : 1;
		int large_idx = 1 - small_idx;

		if (RGO::Log::AtInfo())
		{
			std::cout << "[Info] SplitByMarginLine: areas " << std::min(area0, area1)
				<< " (small) and " << std::max(area0, area1) << " (large)." << std::endl;
		}

		// Hide the carved source; show the two extracted pieces.
		source.mesh_color = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
		source.is_dirty = true;

		small_piece = BuildPieceMesh(patch_points[small_idx], patch_indices[small_idx],
			Eigen::Vector4f(1.0f, 0.4f, 0.2f, 1.0f),
			source.offset + small_piece_move);

		large_piece = BuildPieceMesh(patch_points[large_idx], patch_indices[large_idx],
			Eigen::Vector4f(0.5f, 0.7f, 1.0f, 1.0f),
			source.offset);

		VD::Clear("MarginCurve");
		VD::Clear("ControlPoints");
	}

	void MoveSmallPiece(const Eigen::Vector3f& delta)
	{
		if (nullptr == small_piece) return;
		small_piece->offset += delta;
		small_piece->is_dirty = true;
	}

	EditState CaptureState() const
	{
		EditState s;
		s.control_points = control_points;
		s.margin_closed = margin_closed;
		return s;
	}

	void PushUndoSnapshot()
	{
		undo_stack.push_back(CaptureState());

		// A fresh edit invalidates any redo history beyond this point.
		redo_stack.clear();

		// Cap the depth so a long session does not grow without bound.
		if (undo_stack.size() > max_undo_depth)
		{
			undo_stack.erase(undo_stack.begin());
		}
	}

	void ApplyState(RGO::HeliumMesh& mesh, const EditState& s)
	{
		control_points = s.control_points;
		margin_closed = s.margin_closed;

		// A restored state has no drag in progress.
		dragging_index = -1;

		RedrawControlPoints(mesh);
		RebuildAndDrawMarginCurve(mesh);
	}

	void Undo(RGO::HeliumMesh& mesh)
	{
		if (undo_stack.empty())
		{
			if (RGO::Log::AtInfo())
			{
				std::cout << "[Info] Undo: nothing to undo." << std::endl;
			}
			return;
		}

		// Current state becomes redoable.
		redo_stack.push_back(CaptureState());

		EditState s = undo_stack.back();
		undo_stack.pop_back();

		ApplyState(mesh, s);
	}

	void Redo(RGO::HeliumMesh& mesh)
	{
		if (redo_stack.empty())
		{
			if (RGO::Log::AtInfo())
			{
				std::cout << "[Info] Redo: nothing to redo." << std::endl;
			}
			return;
		}

		// Current state becomes undoable again.
		undo_stack.push_back(CaptureState());

		EditState s = redo_stack.back();
		redo_stack.pop_back();

		ApplyState(mesh, s);
	}

	bool BuildSeamFlagsFromSegments(
		RGO::HeliumMesh& mesh,
		const RGO::OperatorIntersectionLoops& loop_op,
		std::vector<char>& out_edge_is_seam)
	{
		out_edge_is_seam.assign(mesh.n_edges(), 0);

		const auto& segments = loop_op.GetSegments();
		if (segments.empty()) return false;

		// Bit-exact endpoint-pair set. Canonicalization made logically
		// identical points bit-identical, and co-refinement carved each
		// segment as a real mesh edge, so an edge is a seam edge exactly
		// when its endpoint pair equals a segment's endpoint pair.
		auto less_xyz = [](const Eigen::Vector3f& a, const Eigen::Vector3f& b) -> bool
			{
				if (a.x() != b.x()) return a.x() < b.x();
				if (a.y() != b.y()) return a.y() < b.y();
				return a.z() < b.z();
			};

		struct Key
		{
			Eigen::Vector3f a;
			Eigen::Vector3f b;
		};
		struct KeyHash
		{
			size_t operator()(const Key& k) const
			{
				RGO::Vector3fBitHash h;
				return h(k.a) * 1000003ull ^ h(k.b);
			}
		};
		struct KeyEqual
		{
			bool operator()(const Key& x, const Key& y) const
			{
				RGO::Vector3fBitEqual eq;
				return eq(x.a, y.a) && eq(x.b, y.b);
			}
		};

		auto make_key = [&](const Eigen::Vector3f& a, const Eigen::Vector3f& b) -> Key
			{
				Key k;
				if (less_xyz(a, b)) { k.a = a; k.b = b; }
				else { k.a = b; k.b = a; }
				return k;
			};

		robin_hood::unordered_map<Key, char, KeyHash, KeyEqual> seg_pairs;
		seg_pairs.reserve(segments.size());
		for (const auto& s : segments)
		{
			seg_pairs[make_key(s.p0, s.p1)] = 0;
		}

		size_t seam_count = 0;
		for (size_t i = 0; i < mesh.n_edges(); ++i)
		{
			OpenMesh::EdgeHandle eh = mesh.edge_handle(static_cast<int>(i));
			if (mesh.status(eh).deleted()) continue;

			OpenMesh::HalfedgeHandle heh = mesh.halfedge_handle(eh, 0);
			auto pf = mesh.point(mesh.from_vertex_handle(heh));
			auto pt = mesh.point(mesh.to_vertex_handle(heh));
			Eigen::Vector3f a(pf[0], pf[1], pf[2]);
			Eigen::Vector3f b(pt[0], pt[1], pt[2]);

			auto it = seg_pairs.find(make_key(a, b));
			if (it == seg_pairs.end()) continue;

			it->second = 1;
			out_edge_is_seam[i] = 1;
			++seam_count;
		}

		size_t missing = 0;
		for (const auto& kvp : seg_pairs)
		{
			if (0 == kvp.second) ++missing;
		}

		if (missing > 0)
		{
			std::cout << "[Error] BuildSeamFlagsFromSegments: " << missing
				<< " segments have no matching mesh edge after refine."
				<< " The cut seam is incomplete." << std::endl;
			return false;
		}

		if (RGO::Log::AtInfo())
		{
			std::cout << "[Info] BuildSeamFlagsFromSegments: " << seam_count
				<< " seam edges matched from " << seg_pairs.size()
				<< " unique segment pairs." << std::endl;
		}
		return true;
	}

	float PatchArea(
		const std::vector<Eigen::Vector3f>& pts,
		const std::vector<Eigen::Vector3i>& idx) const
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
	}

	bool SaveMarginLineInput(
		const std::string& path,
		const std::vector<Eigen::Vector3f>& picked_points,
		const std::vector<Eigen::Vector3f>& curve_points,
		const std::vector<OpenMesh::FaceHandle>& curve_faces,
		bool closed)
	{
		// Binary layout, little-endian, tightly packed:
		//   char[8]   magic   "RGOMARGN"
		//   uint32    version 1
		//   uint8     closed  (0 or 1)
		//   uint64    picked_count
		//   float[3]  * picked_count
		//   uint64    curve_count
		//   float[3]  * curve_count
		//   uint64    face_count           (must equal curve_count)
		//   int32     * face_count         (face index, -1 for invalid)
		// curve_faces is stored as raw face indices: the SAME mesh must be
		// loaded for these to be valid, since a FaceHandle is only an index
		// into that mesh's face array.
		if (curve_points.size() != curve_faces.size())
		{
			std::cout << "[Error] SaveMarginLineInput: curve_points (" << curve_points.size()
				<< ") and curve_faces (" << curve_faces.size()
				<< ") sizes differ; refusing to save inconsistent input." << std::endl;
			return false;
		}

		std::ofstream out(path, std::ios::binary | std::ios::trunc);
		if (false == out.is_open())
		{
			std::cout << "[Error] SaveMarginLineInput: cannot open file for writing: "
				<< path << std::endl;
			return false;
		}

		const char magic[8] = { 'R', 'G', 'O', 'M', 'A', 'R', 'G', 'N' };
		const uint32_t version = 1;
		const uint8_t closed_flag = closed ? 1 : 0;

		out.write(magic, 8);
		out.write(reinterpret_cast<const char*>(&version), sizeof(version));
		out.write(reinterpret_cast<const char*>(&closed_flag), sizeof(closed_flag));

		uint64_t picked_count = static_cast<uint64_t>(picked_points.size());
		out.write(reinterpret_cast<const char*>(&picked_count), sizeof(picked_count));
		for (const auto& p : picked_points)
		{
			float xyz[3] = { p.x(), p.y(), p.z() };
			out.write(reinterpret_cast<const char*>(xyz), sizeof(xyz));
		}

		uint64_t curve_count = static_cast<uint64_t>(curve_points.size());
		out.write(reinterpret_cast<const char*>(&curve_count), sizeof(curve_count));
		for (const auto& p : curve_points)
		{
			float xyz[3] = { p.x(), p.y(), p.z() };
			out.write(reinterpret_cast<const char*>(xyz), sizeof(xyz));
		}

		uint64_t face_count = static_cast<uint64_t>(curve_faces.size());
		out.write(reinterpret_cast<const char*>(&face_count), sizeof(face_count));
		for (const auto& fh : curve_faces)
		{
			int32_t idx = fh.is_valid() ? static_cast<int32_t>(fh.idx()) : -1;
			out.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
		}

		if (false == out.good())
		{
			std::cout << "[Error] SaveMarginLineInput: write failed (disk full or stream error): "
				<< path << std::endl;
			return false;
		}

		if (RGO::Log::AtInfo())
		{
			std::cout << "[Info] SaveMarginLineInput: wrote " << picked_count
				<< " picked points, " << curve_count << " curve points to "
				<< path << "." << std::endl;
		}
		return true;
	}

	bool LoadMarginLineInput(
		const std::string& path,
		const RGO::Mesh* mesh,
		std::vector<Eigen::Vector3f>& out_picked_points,
		std::vector<Eigen::Vector3f>& out_curve_points,
		std::vector<OpenMesh::FaceHandle>& out_curve_faces,
		bool& out_closed)
	{
		out_picked_points.clear();
		out_curve_points.clear();
		out_curve_faces.clear();
		out_closed = false;

		if (nullptr == mesh)
		{
			std::cout << "[Error] LoadMarginLineInput: a mesh is required to rebuild"
				<< " face handles from stored indices." << std::endl;
			return false;
		}

		std::ifstream in(path, std::ios::binary);
		if (false == in.is_open())
		{
			std::cout << "[Error] LoadMarginLineInput: cannot open file for reading: "
				<< path << std::endl;
			return false;
		}

		char magic[8] = { 0 };
		in.read(magic, 8);
		const char expect[8] = { 'R', 'G', 'O', 'M', 'A', 'R', 'G', 'N' };
		if (false == in.good() || 0 != std::memcmp(magic, expect, 8))
		{
			std::cout << "[Error] LoadMarginLineInput: bad magic; not an RGO margin file: "
				<< path << std::endl;
			return false;
		}

		uint32_t version = 0;
		in.read(reinterpret_cast<char*>(&version), sizeof(version));
		if (false == in.good() || 1 != version)
		{
			std::cout << "[Error] LoadMarginLineInput: unsupported version " << version
				<< " (expected 1)." << std::endl;
			return false;
		}

		uint8_t closed_flag = 0;
		in.read(reinterpret_cast<char*>(&closed_flag), sizeof(closed_flag));
		if (false == in.good())
		{
			std::cout << "[Error] LoadMarginLineInput: truncated file at closed flag." << std::endl;
			return false;
		}
		out_closed = (0 != closed_flag);

		uint64_t picked_count = 0;
		in.read(reinterpret_cast<char*>(&picked_count), sizeof(picked_count));
		if (false == in.good())
		{
			std::cout << "[Error] LoadMarginLineInput: truncated file at picked count." << std::endl;
			return false;
		}
		out_picked_points.reserve(static_cast<size_t>(picked_count));
		for (uint64_t i = 0; i < picked_count; ++i)
		{
			float xyz[3] = { 0.0f, 0.0f, 0.0f };
			in.read(reinterpret_cast<char*>(xyz), sizeof(xyz));
			if (false == in.good())
			{
				std::cout << "[Error] LoadMarginLineInput: truncated at picked point "
					<< i << "." << std::endl;
				out_picked_points.clear();
				return false;
			}
			out_picked_points.emplace_back(xyz[0], xyz[1], xyz[2]);
		}

		uint64_t curve_count = 0;
		in.read(reinterpret_cast<char*>(&curve_count), sizeof(curve_count));
		if (false == in.good())
		{
			std::cout << "[Error] LoadMarginLineInput: truncated file at curve count." << std::endl;
			out_picked_points.clear();
			return false;
		}
		out_curve_points.reserve(static_cast<size_t>(curve_count));
		for (uint64_t i = 0; i < curve_count; ++i)
		{
			float xyz[3] = { 0.0f, 0.0f, 0.0f };
			in.read(reinterpret_cast<char*>(xyz), sizeof(xyz));
			if (false == in.good())
			{
				std::cout << "[Error] LoadMarginLineInput: truncated at curve point "
					<< i << "." << std::endl;
				out_picked_points.clear();
				out_curve_points.clear();
				return false;
			}
			out_curve_points.emplace_back(xyz[0], xyz[1], xyz[2]);
		}

		uint64_t face_count = 0;
		in.read(reinterpret_cast<char*>(&face_count), sizeof(face_count));
		if (false == in.good())
		{
			std::cout << "[Error] LoadMarginLineInput: truncated file at face count." << std::endl;
			out_picked_points.clear();
			out_curve_points.clear();
			return false;
		}
		if (face_count != curve_count)
		{
			std::cout << "[Error] LoadMarginLineInput: face count " << face_count
				<< " does not match curve count " << curve_count
				<< "; file is inconsistent." << std::endl;
			out_picked_points.clear();
			out_curve_points.clear();
			return false;
		}

		const int num_faces = static_cast<int>(mesh->n_faces());
		size_t invalid_faces = 0;
		out_curve_faces.reserve(static_cast<size_t>(face_count));
		for (uint64_t i = 0; i < face_count; ++i)
		{
			int32_t idx = -1;
			in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
			if (false == in.good())
			{
				std::cout << "[Error] LoadMarginLineInput: truncated at face index "
					<< i << "." << std::endl;
				out_picked_points.clear();
				out_curve_points.clear();
				out_curve_faces.clear();
				return false;
			}

			// A stored -1 was an invalid handle at save time. A non-negative
			// index must be within this mesh's face range, otherwise the
			// file was saved against a different mesh and the handles are
			// meaningless: fail loudly rather than carve with wrong faces.
			if (idx < 0)
			{
				out_curve_faces.push_back(OpenMesh::FaceHandle());
				++invalid_faces;
			}
			else if (idx >= num_faces)
			{
				std::cout << "[Error] LoadMarginLineInput: stored face index " << idx
					<< " at position " << i << " exceeds mesh face count " << num_faces
					<< ". This file was saved against a different mesh." << std::endl;
				out_picked_points.clear();
				out_curve_points.clear();
				out_curve_faces.clear();
				return false;
			}
			else
			{
				out_curve_faces.push_back(mesh->face_handle(idx));
			}
		}

		if (RGO::Log::AtInfo())
		{
			std::cout << "[Info] LoadMarginLineInput: read " << picked_count
				<< " picked points, " << curve_count << " curve points ("
				<< invalid_faces << " invalid face handles) from " << path
				<< ", closed " << (out_closed ? "true" : "false") << "." << std::endl;
		}
		return true;
	}

	void SaveMarginLineInputToFile(RGO::HeliumMesh& mesh)
	{
		if (margin_curve.size() < 2 || margin_curve_faces.size() != margin_curve.size())
		{
			std::cout << "[Warning] SaveMarginLineInputToFile: no valid margin curve to save."
				<< " Pick points and build the curve first." << std::endl;
			return;
		}

		const std::string path = "D:\\Temp\\margin_input.bin";
		if (SaveMarginLineInput(path, control_points, margin_curve, margin_curve_faces, margin_closed))
		{
			if (RGO::Log::AtInfo())
			{
				std::cout << "[Info] SaveMarginLineInputToFile: saved current margin input to "
					<< path << "." << std::endl;
			}
		}
	}

	void LoadMarginLineInputFromFile(RGO::HeliumMesh& mesh)
	{
		const std::string path = "D:\\Temp\\margin_input.bin";

		std::vector<Eigen::Vector3f> picked;
		std::vector<Eigen::Vector3f> curve;
		std::vector<OpenMesh::FaceHandle> faces;
		bool closed = false;

		if (false == LoadMarginLineInput(path, &mesh, picked, curve, faces, closed))
		{
			std::cout << "[Warning] LoadMarginLineInputFromFile: load failed; see messages above." << std::endl;
			return;
		}

		// Replace the live edit state with the loaded input so the picked
		// points, closed flag, and the exact curve all match what was saved.
		// The curve and its faces are taken verbatim from the file, so
		// BuildGeodesicSurfaceCurve is bypassed and the co-refine input is
		// bit-identical across runs.
		control_points = picked;
		margin_closed = closed;
		margin_curve = curve;
		margin_curve_faces = faces;

		dragging_index = -1;
		undo_stack.clear();
		redo_stack.clear();

		// Redraw control points and the loaded curve directly. The curve is
		// NOT rebuilt from control points here, since the whole point is to
		// reuse the saved curve exactly; only the visualization is refreshed.
		RedrawControlPoints(mesh);
		RedrawLoadedMarginCurve(mesh);

		if (RGO::Log::AtInfo())
		{
			std::cout << "[Info] LoadMarginLineInputFromFile: loaded " << control_points.size()
				<< " picked points and " << margin_curve.size()
				<< " curve points; running split." << std::endl;
		}

		SplitByMarginLine(mesh);
	}

	void RedrawLoadedMarginCurve(RGO::HeliumMesh& mesh)
	{
		VD::Clear("MarginLines");
		VD::Clear("MarginCurve");

		if (margin_curve.size() < 2) return;

		// Draw the already-loaded curve without rebuilding it from control
		// points. Closed loop draws red, open chain draws magenta, matching
		// RebuildAndDrawMarginCurve.
		const Eigen::Vector4f color = margin_closed
			? Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f)
			: Eigen::Vector4f(1.0f, 0.0f, 1.0f, 1.0f);

		size_t seg_count = margin_closed ? margin_curve.size() : (margin_curve.size() - 1);
		for (size_t i = 0; i < seg_count; ++i)
		{
			Eigen::Vector3f a = margin_curve[i] + mesh.offset;
			Eigen::Vector3f b = margin_curve[(i + 1) % margin_curve.size()] + mesh.offset;
			VD::AddLine("MarginCurve", a, b, color);
		}
	}

	bool LoadMarginLineInput(
		const std::string& path,
		const RGO::HeliumMesh* mesh,
		std::vector<Eigen::Vector3f>& out_picked_points,
		std::vector<Eigen::Vector3f>& out_curve_points,
		std::vector<OpenMesh::FaceHandle>& out_curve_faces,
		bool& out_closed)
	{
		out_picked_points.clear();
		out_curve_points.clear();
		out_curve_faces.clear();
		out_closed = false;

		if (nullptr == mesh)
		{
			std::cout << "[Error] LoadMarginLineInput: a mesh is required to rebuild"
				<< " face handles from stored indices." << std::endl;
			return false;
		}

		std::ifstream in(path, std::ios::binary);
		if (false == in.is_open())
		{
			std::cout << "[Error] LoadMarginLineInput: cannot open file for reading: "
				<< path << std::endl;
			return false;
		}

		char magic[8] = { 0 };
		in.read(magic, 8);
		const char expect[8] = { 'R', 'G', 'O', 'M', 'A', 'R', 'G', 'N' };
		if (false == in.good() || 0 != std::memcmp(magic, expect, 8))
		{
			std::cout << "[Error] LoadMarginLineInput: bad magic; not an RGO margin file: "
				<< path << std::endl;
			return false;
		}

		uint32_t version = 0;
		in.read(reinterpret_cast<char*>(&version), sizeof(version));
		if (false == in.good() || 1 != version)
		{
			std::cout << "[Error] LoadMarginLineInput: unsupported version " << version
				<< " (expected 1)." << std::endl;
			return false;
		}

		uint8_t closed_flag = 0;
		in.read(reinterpret_cast<char*>(&closed_flag), sizeof(closed_flag));
		if (false == in.good())
		{
			std::cout << "[Error] LoadMarginLineInput: truncated file at closed flag." << std::endl;
			return false;
		}
		out_closed = (0 != closed_flag);

		uint64_t picked_count = 0;
		in.read(reinterpret_cast<char*>(&picked_count), sizeof(picked_count));
		if (false == in.good())
		{
			std::cout << "[Error] LoadMarginLineInput: truncated file at picked count." << std::endl;
			return false;
		}
		out_picked_points.reserve(static_cast<size_t>(picked_count));
		for (uint64_t i = 0; i < picked_count; ++i)
		{
			float xyz[3] = { 0.0f, 0.0f, 0.0f };
			in.read(reinterpret_cast<char*>(xyz), sizeof(xyz));
			if (false == in.good())
			{
				std::cout << "[Error] LoadMarginLineInput: truncated at picked point "
					<< i << "." << std::endl;
				out_picked_points.clear();
				return false;
			}
			out_picked_points.emplace_back(xyz[0], xyz[1], xyz[2]);
		}

		uint64_t curve_count = 0;
		in.read(reinterpret_cast<char*>(&curve_count), sizeof(curve_count));
		if (false == in.good())
		{
			std::cout << "[Error] LoadMarginLineInput: truncated file at curve count." << std::endl;
			out_picked_points.clear();
			return false;
		}
		out_curve_points.reserve(static_cast<size_t>(curve_count));
		for (uint64_t i = 0; i < curve_count; ++i)
		{
			float xyz[3] = { 0.0f, 0.0f, 0.0f };
			in.read(reinterpret_cast<char*>(xyz), sizeof(xyz));
			if (false == in.good())
			{
				std::cout << "[Error] LoadMarginLineInput: truncated at curve point "
					<< i << "." << std::endl;
				out_picked_points.clear();
				out_curve_points.clear();
				return false;
			}
			out_curve_points.emplace_back(xyz[0], xyz[1], xyz[2]);
		}

		uint64_t face_count = 0;
		in.read(reinterpret_cast<char*>(&face_count), sizeof(face_count));
		if (false == in.good())
		{
			std::cout << "[Error] LoadMarginLineInput: truncated file at face count." << std::endl;
			out_picked_points.clear();
			out_curve_points.clear();
			return false;
		}
		if (face_count != curve_count)
		{
			std::cout << "[Error] LoadMarginLineInput: face count " << face_count
				<< " does not match curve count " << curve_count
				<< "; file is inconsistent." << std::endl;
			out_picked_points.clear();
			out_curve_points.clear();
			return false;
		}

		const int num_faces = static_cast<int>(mesh->n_faces());
		size_t invalid_faces = 0;
		out_curve_faces.reserve(static_cast<size_t>(face_count));
		for (uint64_t i = 0; i < face_count; ++i)
		{
			int32_t idx = -1;
			in.read(reinterpret_cast<char*>(&idx), sizeof(idx));
			if (false == in.good())
			{
				std::cout << "[Error] LoadMarginLineInput: truncated at face index "
					<< i << "." << std::endl;
				out_picked_points.clear();
				out_curve_points.clear();
				out_curve_faces.clear();
				return false;
			}

			// A stored -1 was an invalid handle at save time. A non-negative
			// index must be within this mesh's face range; otherwise the file
			// was saved against a different mesh and the handles are
			// meaningless, so fail loudly rather than carve with wrong faces.
			if (idx < 0)
			{
				out_curve_faces.push_back(OpenMesh::FaceHandle());
				++invalid_faces;
			}
			else if (idx >= num_faces)
			{
				std::cout << "[Error] LoadMarginLineInput: stored face index " << idx
					<< " at position " << i << " exceeds mesh face count " << num_faces
					<< ". This file was saved against a different mesh." << std::endl;
				out_picked_points.clear();
				out_curve_points.clear();
				out_curve_faces.clear();
				return false;
			}
			else
			{
				out_curve_faces.push_back(mesh->face_handle(idx));
			}
		}

		if (RGO::Log::AtInfo())
		{
			std::cout << "[Info] LoadMarginLineInput: read " << picked_count
				<< " picked points, " << curve_count << " curve points ("
				<< invalid_faces << " invalid face handles) from " << path
				<< ", closed " << (out_closed ? "true" : "false") << "." << std::endl;
		}
		return true;
	}

	void DiagnoseCurveSelfProximity(
		const std::vector<Eigen::Vector3f>& curve,
		bool closed) const
	{
		const size_t n = curve.size();
		if (n < 4) return;

		// A self-proximity is two curve samples that are far apart ALONG the
		// curve (index distance large) yet close in space. That is the
		// signature of the curve folding back near itself, which co-refine
		// turns into a degree-4 node and an open chain. Sliver segments of
		// length ~1e-5 in the loop tracer come from exactly these folds.
		// Consecutive samples are skipped (their closeness is normal); only
		// pairs at least min_index_gap apart are reported.
		const size_t min_index_gap = 3;

		// Proximity threshold scaled to the mean curve sample spacing, so a
		// fold is flagged relative to how densely the curve is sampled.
		double total = 0.0;
		size_t spans = closed ? n : (n - 1);
		for (size_t i = 0; i < spans; ++i)
		{
			total += (curve[(i + 1) % n] - curve[i]).norm();
		}
		float mean_spacing = (spans > 0) ? static_cast<float>(total / spans) : 0.0f;
		float near_threshold = 0.5f * mean_spacing;

		std::cout << std::setprecision(10);
		if (RGO::Log::Diag())
		{
			std::cout << "[Debug] DiagnoseCurveSelfProximity: " << n << " samples, mean spacing "
				<< mean_spacing << ", flagging pairs closer than " << near_threshold
				<< " at least " << min_index_gap << " indices apart." << std::endl;
		}

		size_t reported = 0;
		float min_fold_dist = std::numeric_limits<float>::max();

		for (size_t i = 0; i < n; ++i)
		{
			for (size_t j = i + min_index_gap; j < n; ++j)
			{
				// In a closed curve the wrap-around neighbours are also
				// consecutive, so skip pairs whose cyclic index gap is small.
				size_t cyclic_gap = std::min(j - i, n - (j - i));
				if (cyclic_gap < min_index_gap) continue;

				float d = (curve[i] - curve[j]).norm();
				if (d >= near_threshold) continue;

				if (d < min_fold_dist) min_fold_dist = d;

				if (reported < 20)
				{
					++reported;
					if (RGO::Log::Diag())
					{
						std::cout << "[Debug]   fold: sample " << i << " (" << curve[i].x()
							<< ", " << curve[i].y() << ", " << curve[i].z() << ") and sample "
							<< j << " (" << curve[j].x() << ", " << curve[j].y() << ", "
							<< curve[j].z() << ") are " << d << " apart, "
							<< (j - i) << " indices apart." << std::endl;
					}
				}
			}
		}

		if (RGO::Log::Diag())
		{
			std::cout << "[Debug] DiagnoseCurveSelfProximity: " << reported
				<< " folds reported, closest fold distance " << min_fold_dist << "." << std::endl;
		}
		std::cout << std::setprecision(6);
	}

protected:
	std::vector<std::unique_ptr<RGO::HeliumMesh>> meshes;

	std::vector<Eigen::Vector3f> control_points;
	bool margin_closed = false;
	int margin_samples_per_segment = 16;
	int margin_smoothing_iterations = 12;
	float margin_smoothing_strength = 0.5f;

	int dragging_index = -1;
	float drag_pick_radius = 0.6f;

	std::vector<Eigen::Vector3f> margin_curve;
	std::vector<OpenMesh::FaceHandle> margin_curve_faces;

	RGO::HeliumMesh* small_piece = nullptr;
	RGO::HeliumMesh* large_piece = nullptr;
	Eigen::Vector3f small_piece_move = Eigen::Vector3f(0.0f, 0.0f, 5.0f);

	std::vector<EditState> undo_stack;
	std::vector<EditState> redo_stack;
	size_t max_undo_depth = 100;
};

REGISTER_APP(AppMeshSplineCut, "AppMeshSplineCut");
