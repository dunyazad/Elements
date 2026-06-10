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
		if (margin_curve.size() < 3 || margin_curve_faces.size() != margin_curve.size())
		{
			std::cout << "[Warning] SplitByMarginLine: need a built closed margin curve"
				<< " (press points, then C). Current curve is too short." << std::endl;
			return;
		}

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
		RGO::OperatorCoRefine corefine(&source, &source, &loop_op);
		if (false == corefine.Execute())
		{
			std::cout << "[Warning] SplitByMarginLine: co-refinement left boundary"
				<< " edges; the curve could not be carved cleanly. Adjust the points." << std::endl;
			return;
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

		std::cout << "[Info] SplitByMarginLine: areas " << std::min(area0, area1)
			<< " (small) and " << std::max(area0, area1) << " (large)." << std::endl;

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
			std::cout << "[Info] Undo: nothing to undo." << std::endl;
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
			std::cout << "[Info] Redo: nothing to redo." << std::endl;
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

		std::cout << "[Info] BuildSeamFlagsFromSegments: " << seam_count
			<< " seam edges matched from " << seg_pairs.size()
			<< " unique segment pairs." << std::endl;
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
