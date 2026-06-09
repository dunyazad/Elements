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
			? Eigen::Vector4f(0.0f, 0.0f, 1.0f, 1.0f)
			: Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f);

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

	void AddControlPointAt(RGO::HeliumMesh& mesh, const Eigen::Vector3f& local_point)
	{
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
		margin_closed = closed;
		RebuildAndDrawMarginCurve(mesh);
	}

	void HandlePick(RGO::HeliumMesh& mesh, const RGO::IntersectionResult& hit_result, bool ctrl_pressed)
	{
		Eigen::Vector3f world_picked_point = hit_result.hit_point + mesh.offset;

		if (ctrl_pressed)
		{
			auto camera_entity = Helium.GetEntityByName("MainCamera");
			Helium.GetComponent<Camera>(camera_entity)->SetTarget(world_picked_point);
		}

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
			AddControlPointAt(mesh, hit_result.hit_point);
		}
		else if (OperationMode::RemoveControlPoint == current_mode)
		{
			RemoveNearestControlPoint(mesh, hit_result.hit_point);
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

		// For an open chain, the last span is index n - 2 (joining the last
		// two points). When that span is the closest, the click is past the
		// end of the chain, so appending is the natural action rather than
		// inserting before the final point.
		if (false == margin_closed && best_span == span_count - 1)
		{
			// Decide append vs insert by which endpoint of the last span is
			// closer: nearer the very last point means extend the chain.
			const Eigen::Vector3f& a = control_points[n - 2];
			const Eigen::Vector3f& b = control_points[n - 1];
			float da2 = (local_point - a).squaredNorm();
			float db2 = (local_point - b).squaredNorm();
			if (db2 <= da2) return n;
		}

		if (span_count == best_span) return n;

		// Insert after the span's start point.
		return best_span + 1;
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

	void EndDrag()
	{
		dragging_index = -1;
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
};

REGISTER_APP(AppMeshSplineCut, "AppMeshSplineCut");
