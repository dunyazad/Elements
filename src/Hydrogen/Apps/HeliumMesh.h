#pragma once

#include <vector>
#include <limits>
#include <iostream>

#include <Eigen/Dense>

#include <Helium/Helium.h>
#include <Helium/HeliumCore.h>
#include <Helium/VisualDebugging.h>

#include "SimpleGeometryLibrary.hpp"

// Renderable mesh shared by all apps. Inherits the geometry kernel
// (SGL::Mesh) and adds Helium rendering plus interactive picking.
// Class-body member definitions are implicitly inline, so including this
// header in multiple translation units does not violate ODR.
class SGLHeliumMesh : public SGL::Mesh
{
public:
    SGLHeliumMesh() : SGL::Mesh() {}
    virtual ~SGLHeliumMesh() {}
    Entity entity = InvalidEntity;
    bool is_dirty = true;
    Eigen::Vector3f offset = Eigen::Vector3f::Zero();

    Eigen::Vector4f mesh_color = Eigen::Vector4f(0.8f, 0.8f, 0.8f, 1.0f);

    enum OperationMode
    {
        None,
        FlipEdge,
        SplitFace,
    } current_mode = None;

    void Update()
    {
        using VD = VisualDebugging;

        if (false == is_dirty) return;

        Renderable* renderable = nullptr;

        if (InvalidEntity == entity)
        {
            entity = Helium.CreateEntity("HeliumMesh");
            renderable = Helium.CreateComponent<Renderable>(entity);
            renderable->Initialize(Renderable::Triangles);
            renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));
            renderable->SetFaceCullingMode(Renderable::NoCulling);

            Helium.CreateEventCallback<KeyEvent>(entity, "Mesh", [this](Entity e, const KeyEvent& event)
                {
                    auto current_renderable = Helium.GetComponent<Renderable>(e);
                    if (nullptr == current_renderable) return;

                    if (event.action == 1 && KeyCode::D1 == event.keyCode)
                        current_renderable->SetDrawingMode(Renderable::Solid);
                    else if (event.action == 1 && KeyCode::D2 == event.keyCode)
                        current_renderable->SetDrawingMode(Renderable::WireFrame);
                    else if (event.action == 1 && KeyCode::D3 == event.keyCode)
                        current_renderable->SetDrawingMode(Renderable::WireFrameOverSolid);
                    else if (event.action == 1 && KeyCode::D4 == event.keyCode)
                        current_renderable->SetDrawingMode(Renderable::Point);
                    else if (event.action == 1 && KeyCode::D5 == event.keyCode)
                        current_renderable->SetDrawingMode(Renderable::None);
                    else if (event.action == 1 && KeyCode::F1 == event.keyCode)
                        current_mode = OperationMode::None;
                    else if (event.action == 1 && KeyCode::F2 == event.keyCode)
                        current_mode = OperationMode::FlipEdge;
                    else if (event.action == 1 && KeyCode::F3 == event.keyCode)
                        current_mode = OperationMode::SplitFace;
                });

            Helium.CreateEventCallback<MouseButtonEvent>(entity, "Mesh", [this](Entity e, const MouseButtonEvent& event)
                {
                    if (event.action == 1 && event.button == MouseButton::Left)
                    {
                        auto current_renderable = Helium.GetComponent<Renderable>(e);
                        if (nullptr == current_renderable) return;

                        auto camera_entity = Helium.GetEntityByName("MainCamera");
                        auto camera = Helium.GetComponent<Camera>(camera_entity);
                        if (nullptr == camera) return;

                        TS(Picking);
                        Ray ray = camera->ScreenPointToRay(
                            (float)event.xpos,
                            (float)event.ypos,
                            Helium.GetWidth(),
                            Helium.GetHeight()
                        );

                        Eigen::Vector3f local_origin = ray.origin - offset;

                        SGL::IntersectionResult hit_result;
                        bool is_hit = IntersectGridRay(local_origin, ray.direction, hit_result);
                        TE(Picking);

                        if (is_hit)
                        {
                            auto world_picked_point = hit_result.hit_point + offset;

                            if (event.IsCtrlPressed())
                            {
                                auto cameraEntity = Helium.GetEntityByName("MainCamera");
                                Helium.GetComponent<Camera>(cameraEntity)->SetTarget(world_picked_point);
                            }

                            if (OperationMode::None == current_mode)
                            {
                                VD::Clear("PickedTriangle");
                                VD::Clear("PickedTrianglePoints");
                            }
                            else if (OperationMode::FlipEdge == current_mode)
                            {
                                OpenMesh::EdgeHandle target_edge;
                                float min_dist = std::numeric_limits<float>::max();
                                Eigen::Vector3f target_v0, target_v1;

                                for (auto fh_it = fh_iter(hit_result.fh); fh_it.is_valid(); ++fh_it)
                                {
                                    auto v0_handle = from_vertex_handle(*fh_it);
                                    auto v1_handle = to_vertex_handle(*fh_it);

                                    Eigen::Vector3f v0(point(v0_handle).data());
                                    Eigen::Vector3f v1(point(v1_handle).data());

                                    float dist = DistanceToSegment(hit_result.hit_point, v0, v1);

                                    if (dist < min_dist)
                                    {
                                        min_dist = dist;
                                        target_edge = edge_handle(*fh_it);
                                        target_v0 = v0;
                                        target_v1 = v1;
                                    }
                                }

                                if (target_edge.is_valid())
                                {
                                    VD::Clear("Picked edge");

                                    if (is_flip_ok(target_edge) && IsConvexQuadrilateral(target_edge))
                                    {
                                        flip(target_edge);

                                        TS(BuildSpatialHashMap);
                                        BuildSpatialHashMap();
                                        TE(BuildSpatialHashMap);

                                        is_dirty = true;
                                    }
                                    else
                                    {
                                        std::cout << "[Warning] Cannot flip this edge (Boundary or Non-manifold)." << std::endl;
                                    }
                                }
                            }
                            else if (OperationMode::SplitFace == current_mode)
                            {
                                if (hit_result.type != SGL::IntersectionType::Vertex)
                                {
                                    OpenMesh::VertexHandle new_v = add_vertex(Point(hit_result.hit_point.x(), hit_result.hit_point.y(), hit_result.hit_point.z()));

                                    if (hit_result.type == SGL::IntersectionType::Face)
                                    {
                                        split(hit_result.fh, new_v);
                                        is_dirty = true;
                                    }
                                    else if (hit_result.type == SGL::IntersectionType::Edge)
                                    {
                                        split(hit_result.eh, new_v);
                                        is_dirty = true;
                                    }

                                    if (is_dirty)
                                    {
                                        TS(BuildSpatialHashMap);
                                        BuildSpatialHashMap();
                                        TE(BuildSpatialHashMap);
                                    }
                                }
                                else
                                {
                                    std::cout << "[Info] Snapped to existing vertex. Split aborted to prevent degenerate faces." << std::endl;
                                }
                            }

                            VD::Clear("Picked");
                            VD::AddSphere("Picked", world_picked_point, { 0.0f, 0.0f, 1.0f }, 0.005f, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
                        }
                    }
                });
        }
        else
        {
            renderable = Helium.GetComponent<Renderable>(entity);
        }

        if (renderable)
        {
            std::vector<Eigen::Vector3f> positions(n_vertices());
            for (auto v_it = vertices_begin(); v_it != vertices_end(); ++v_it)
            {
                auto p = point(*v_it);
                positions[v_it->idx()] = Eigen::Vector3f(p[0], p[1], p[2]) + offset;
            }

            std::vector<Eigen::Vector3f> normals(positions.size(), Eigen::Vector3f::Zero());
            std::vector<unsigned int> indices;
            indices.reserve(n_faces() * 3);

            for (auto f_it = faces_begin(); f_it != faces_end(); ++f_it)
            {
                auto fv_it = cfv_iter(*f_it);
                int idx0 = fv_it->idx(); ++fv_it;
                int idx1 = fv_it->idx(); ++fv_it;
                int idx2 = fv_it->idx();

                indices.push_back(static_cast<unsigned int>(idx0));
                indices.push_back(static_cast<unsigned int>(idx1));
                indices.push_back(static_cast<unsigned int>(idx2));

                Eigen::Vector3f normal_vector = (positions[idx1] - positions[idx0]).cross(positions[idx2] - positions[idx0]).normalized();
                normals[idx0] += normal_vector;
                normals[idx1] += normal_vector;
                normals[idx2] += normal_vector;
            }

            for (size_t i = 0; i < normals.size(); i++) normals[i].normalize();

            renderable->SetVertices(positions);
            renderable->SetNormals(normals);

            renderable->SetColors4(std::vector<Eigen::Vector4f>(positions.size(), mesh_color));
            renderable->SetIndices(indices);
            renderable->Update();
        }

        is_dirty = false;
    }
};







// HeliumMesh depends on Helium engine symbols (Entity, Renderable, Helium,
// VisualDebugging, KeyEvent, MouseButtonEvent, Camera, Ray, TS/TE, etc).
// Include this header AFTER the Helium engine headers, exactly as the old
// monolithic header was included in Apps.cpp.
// Keeping engine-dependent code out of RobustGeometricOperations.cpp keeps
// the geometry library buildable as a standalone translation unit.

#include "RobustGeometricOperations.h"

namespace RGO
{
	class HeliumMesh : public Mesh
	{
	public:
		HeliumMesh() : Mesh() {}
		virtual ~HeliumMesh() {}

		Entity entity = InvalidEntity;
		bool is_dirty = true;
		Eigen::Vector3f offset = Eigen::Vector3f::Zero();

		Eigen::Vector4f mesh_color = Eigen::Vector4f(0.8f, 0.8f, 0.8f, 1.0f);

		enum OperationMode
		{
			None,
			FlipEdge,
			SplitFace,
		} current_mode = None;

		void Update()
		{
			using VD = VisualDebugging;

			if (false == is_dirty) return;

			Renderable* renderable = nullptr;

			if (InvalidEntity == entity)
			{
				entity = Helium.CreateEntity("HeliumMesh");
				renderable = Helium.CreateComponent<Renderable>(entity);
				renderable->Initialize(Renderable::Triangles);
				renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));
				renderable->SetFaceCullingMode(Renderable::NoCulling);

				Helium.CreateEventCallback<KeyEvent>(entity, "Mesh", [this](Entity e, const KeyEvent& event)
					{
						auto current_renderable = Helium.GetComponent<Renderable>(e);
						if (nullptr == current_renderable) return;

						if (event.action == 1 && KeyCode::D1 == event.keyCode)
							current_renderable->SetDrawingMode(Renderable::Solid);
						else if (event.action == 1 && KeyCode::D2 == event.keyCode)
							current_renderable->SetDrawingMode(Renderable::WireFrame);
						else if (event.action == 1 && KeyCode::D3 == event.keyCode)
							current_renderable->SetDrawingMode(Renderable::WireFrameOverSolid);
						else if (event.action == 1 && KeyCode::D4 == event.keyCode)
							current_renderable->SetDrawingMode(Renderable::Point);
						else if (event.action == 1 && KeyCode::D5 == event.keyCode)
							current_renderable->SetDrawingMode(Renderable::None);
						else if (event.action == 1 && KeyCode::F1 == event.keyCode)
							current_mode = OperationMode::None;
						else if (event.action == 1 && KeyCode::F2 == event.keyCode)
							current_mode = OperationMode::FlipEdge;
						else if (event.action == 1 && KeyCode::F3 == event.keyCode)
							current_mode = OperationMode::SplitFace;
					});

				Helium.CreateEventCallback<MouseButtonEvent>(entity, "Mesh", [this](Entity e, const MouseButtonEvent& event)
					{
						if (event.action == 1 && event.button == MouseButton::Left)
						{
							auto current_renderable = Helium.GetComponent<Renderable>(e);
							if (nullptr == current_renderable) return;

							auto camera_entity = Helium.GetEntityByName("MainCamera");
							auto camera = Helium.GetComponent<Camera>(camera_entity);
							if (nullptr == camera) return;

							TS(Picking);
							Ray ray = camera->ScreenPointToRay(
								(float)event.xpos,
								(float)event.ypos,
								Helium.GetWidth(),
								Helium.GetHeight()
							);

							Eigen::Vector3f local_origin = ray.origin - offset;

							IntersectionResult hit_result;
							bool is_hit = IntersectGridRay(local_origin, ray.direction, hit_result);
							TE(Picking);

							if (is_hit)
							{
								auto world_picked_point = hit_result.hit_point + offset;

								if (event.IsCtrlPressed())
								{
									auto cameraEntity = Helium.GetEntityByName("MainCamera");
									Helium.GetComponent<Camera>(cameraEntity)->SetTarget(world_picked_point);
								}

								if (OperationMode::None == current_mode)
								{
									VD::Clear("PickedTriangle");
									VD::Clear("PickedTrianglePoints");
								}
								else if (OperationMode::FlipEdge == current_mode)
								{
									HandleFlipEdge(hit_result);
								}
								else if (OperationMode::SplitFace == current_mode)
								{
									HandleSplitFace(hit_result);
								}

								VD::Clear("Picked");
								VD::AddSphere("Picked", world_picked_point, { 0.0f, 0.0f, 1.0f }, 0.005f, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
							}
						}
					});
			}
			else
			{
				renderable = Helium.GetComponent<Renderable>(entity);
			}

			if (renderable)
			{
				UploadToRenderable(renderable);
			}

			is_dirty = false;
		}

		void HandleFlipEdge(const IntersectionResult& hit_result)
		{
			using VD = VisualDebugging;

			OpenMesh::EdgeHandle target_edge;
			float min_dist = std::numeric_limits<float>::max();

			for (auto fh_it = fh_iter(hit_result.fh); fh_it.is_valid(); ++fh_it)
			{
				auto v0_handle = from_vertex_handle(*fh_it);
				auto v1_handle = to_vertex_handle(*fh_it);

				Eigen::Vector3f v0(point(v0_handle).data());
				Eigen::Vector3f v1(point(v1_handle).data());

				float dist = Distance::PointToLineSegment(hit_result.hit_point, v0, v1);

				if (dist < min_dist)
				{
					min_dist = dist;
					target_edge = edge_handle(*fh_it);
				}
			}

			if (target_edge.is_valid())
			{
				VD::Clear("Picked edge");

				if (is_flip_ok(target_edge) && IsConvexQuadrilateral(target_edge))
				{
					flip(target_edge);

					TS(BuildSpatialHashMap);
					BuildSpatialHashMap();
					TE(BuildSpatialHashMap);

					is_dirty = true;
				}
				else
				{
					std::cout << "[Warning] Cannot flip this edge (Boundary or Non-manifold)." << std::endl;
				}
			}
		}

		void HandleSplitFace(const IntersectionResult& hit_result)
		{
			if (hit_result.type == IntersectionType::Vertex)
			{
				std::cout << "[Info] Snapped to existing vertex. Split aborted to prevent degenerate faces." << std::endl;
				return;
			}

			OpenMesh::VertexHandle new_v = add_vertex(Point(hit_result.hit_point.x(), hit_result.hit_point.y(), hit_result.hit_point.z()));

			if (hit_result.type == IntersectionType::Face)
			{
				split(hit_result.fh, new_v);
				is_dirty = true;
			}
			else if (hit_result.type == IntersectionType::Edge)
			{
				split(hit_result.eh, new_v);
				is_dirty = true;
			}

			if (is_dirty)
			{
				TS(BuildSpatialHashMap);
				BuildSpatialHashMap();
				TE(BuildSpatialHashMap);
			}
		}

		void UploadToRenderable(Renderable* renderable)
		{
			std::vector<Eigen::Vector3f> positions(n_vertices());
			for (auto v_it = vertices_begin(); v_it != vertices_end(); ++v_it)
			{
				auto p = point(*v_it);
				positions[v_it->idx()] = Eigen::Vector3f(p[0], p[1], p[2]) + offset;
			}

			std::vector<Eigen::Vector3f> normals(positions.size(), Eigen::Vector3f::Zero());
			std::vector<unsigned int> indices;
			indices.reserve(n_faces() * 3);

			for (auto f_it = faces_begin(); f_it != faces_end(); ++f_it)
			{
				if (status(*f_it).deleted()) continue;

				auto fv_it = cfv_iter(*f_it);
				int idx0 = fv_it->idx(); ++fv_it;
				int idx1 = fv_it->idx(); ++fv_it;
				int idx2 = fv_it->idx();

				indices.push_back(static_cast<unsigned int>(idx0));
				indices.push_back(static_cast<unsigned int>(idx1));
				indices.push_back(static_cast<unsigned int>(idx2));

				Eigen::Vector3f normal_vector = (positions[idx1] - positions[idx0]).cross(positions[idx2] - positions[idx0]);
				float len = normal_vector.norm();
				if (len > 1e-12f)
				{
					normal_vector /= len;
					normals[idx0] += normal_vector;
					normals[idx1] += normal_vector;
					normals[idx2] += normal_vector;
				}
			}

			for (size_t i = 0; i < normals.size(); i++)
			{
				float len = normals[i].norm();
				if (len > 1e-12f) normals[i] /= len;
				else normals[i] = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
			}

			renderable->SetVertices(positions);
			renderable->SetNormals(normals);

			renderable->SetColors4(std::vector<Eigen::Vector4f>(positions.size(), mesh_color));
			renderable->SetIndices(indices);
			renderable->Update();
		}

		IntersectionResult IntersectRayFaceWithSnap(const Eigen::Vector3f& origin, const Eigen::Vector3f& direction, OpenMesh::FaceHandle fh) const
		{
			IntersectionResult result;
			const float EPS = 1e-4f;

			auto fv_it = cfv_iter(fh);
			auto vh0 = *fv_it++;
			auto vh1 = *fv_it++;
			auto vh2 = *fv_it;

			Eigen::Vector3f v0(point(vh0).data());
			Eigen::Vector3f v1(point(vh1).data());
			Eigen::Vector3f v2(point(vh2).data());

			Eigen::Vector3f edge1 = v1 - v0;
			Eigen::Vector3f edge2 = v2 - v0;
			Eigen::Vector3f pvec = direction.cross(edge2);

			float det = edge1.dot(pvec);
			if (std::abs(det) < 1e-8f) return result;

			float inv_det = 1.0f / det;
			Eigen::Vector3f tvec = origin - v0;

			float u = tvec.dot(pvec) * inv_det;
			if (u < -EPS || u > 1.0f + EPS) return result;

			Eigen::Vector3f qvec = tvec.cross(edge1);
			float v = direction.dot(qvec) * inv_det;
			if (v < -EPS || u + v > 1.0f + EPS) return result;

			float t = edge2.dot(qvec) * inv_det;
			if (t <= 1e-6f) return result;

			float w = 1.0f - u - v;
			result.t = t;
			result.hit_point = origin + direction * t;

			if (w >= 1.0f - EPS) { result.type = IntersectionType::Vertex; result.vh = vh0; result.hit_point = v0; return result; }
			if (u >= 1.0f - EPS) { result.type = IntersectionType::Vertex; result.vh = vh1; result.hit_point = v1; return result; }
			if (v >= 1.0f - EPS) { result.type = IntersectionType::Vertex; result.vh = vh2; result.hit_point = v2; return result; }

			auto fh_it = cfh_iter(fh);
			auto he0 = *fh_it++;
			auto he1 = *fh_it++;
			auto he2 = *fh_it;
			if (w <= EPS) { result.type = IntersectionType::Edge; result.eh = edge_handle(he1); return result; }
			if (u <= EPS) { result.type = IntersectionType::Edge; result.eh = edge_handle(he2); return result; }
			if (v <= EPS) { result.type = IntersectionType::Edge; result.eh = edge_handle(he0); return result; }

			result.type = IntersectionType::Face;
			result.fh = fh;

			return result;
		}

		bool IntersectGridRay(const Eigen::Vector3f& origin, const Eigen::Vector3f& dir, IntersectionResult& out_hit) const
		{
			out_hit.t = std::numeric_limits<float>::max();
			out_hit.type = IntersectionType::None;

			if (hash_map.empty()) return false;

			Eigen::Vector3f inv_dir;
			inv_dir.x() = std::abs(dir.x()) < 1e-8f ? (dir.x() < 0.0f ? -1e8f : 1e8f) : 1.0f / dir.x();
			inv_dir.y() = std::abs(dir.y()) < 1e-8f ? (dir.y() < 0.0f ? -1e8f : 1e8f) : 1.0f / dir.y();
			inv_dir.z() = std::abs(dir.z()) < 1e-8f ? (dir.z() < 0.0f ? -1e8f : 1e8f) : 1.0f / dir.z();

			Eigen::Vector3f t0 = (grid_min - origin).cwiseProduct(inv_dir);
			Eigen::Vector3f t1 = (grid_max - origin).cwiseProduct(inv_dir);
			Eigen::Vector3f tmin = t0.cwiseMin(t1);
			Eigen::Vector3f tmax = t0.cwiseMax(t1);

			float t_enter = tmin.maxCoeff();
			float t_exit = tmax.minCoeff();

			if (t_enter > t_exit || t_exit < 0.0f) return false;

			t_enter = std::max(0.0f, t_enter);
			Eigen::Vector3f start_pos = origin + dir * t_enter;
			Eigen::Vector3f local_pos = start_pos - grid_min;

			int cx = static_cast<int>(std::floor(local_pos.x() / grid_cell_size.x()));
			int cy = static_cast<int>(std::floor(local_pos.y() / grid_cell_size.y()));
			int cz = static_cast<int>(std::floor(local_pos.z() / grid_cell_size.z()));

			int stepX = (dir.x() > 0.0f) ? 1 : -1;
			int stepY = (dir.y() > 0.0f) ? 1 : -1;
			int stepZ = (dir.z() > 0.0f) ? 1 : -1;

			float tDeltaX = std::abs(grid_cell_size.x() * inv_dir.x());
			float tDeltaY = std::abs(grid_cell_size.y() * inv_dir.y());
			float tDeltaZ = std::abs(grid_cell_size.z() * inv_dir.z());

			float tMaxX = t_enter + ((stepX > 0) ? ((cx + 1) * grid_cell_size.x() - local_pos.x()) * std::abs(inv_dir.x()) : (local_pos.x() - cx * grid_cell_size.x()) * std::abs(inv_dir.x()));
			float tMaxY = t_enter + ((stepY > 0) ? ((cy + 1) * grid_cell_size.y() - local_pos.y()) * std::abs(inv_dir.y()) : (local_pos.y() - cy * grid_cell_size.y()) * std::abs(inv_dir.y()));
			float tMaxZ = t_enter + ((stepZ > 0) ? ((cz + 1) * grid_cell_size.z() - local_pos.z()) * std::abs(inv_dir.z()) : (local_pos.z() - cz * grid_cell_size.z()) * std::abs(inv_dir.z()));

			bool hit = false;
			float max_t_search = t_exit + 0.1f;
			float current_t = t_enter;

			while (current_t <= max_t_search)
			{
				if (hit && out_hit.t < current_t)
				{
					break;
				}

				Eigen::Vector3i cell_pos(cx, cy, cz);
				auto it = hash_map.find(cell_pos);
				if (it != hash_map.end())
				{
					const std::vector<int>& face_list = it->second;
					for (int f_idx : face_list)
					{
						IntersectionResult res = IntersectRayFaceWithSnap(origin, dir, face_handle(f_idx));

						if (res.type != IntersectionType::None && res.t < out_hit.t)
						{
							out_hit = res;
							out_hit.fh = face_handle(f_idx);
							hit = true;
						}
					}
				}

				if (tMaxX < tMaxY) {
					if (tMaxX < tMaxZ) { cx += stepX; current_t = tMaxX; tMaxX += tDeltaX; }
					else { cz += stepZ; current_t = tMaxZ; tMaxZ += tDeltaZ; }
				}
				else {
					if (tMaxY < tMaxZ) { cy += stepY; current_t = tMaxY; tMaxY += tDeltaY; }
					else { cz += stepZ; current_t = tMaxZ; tMaxZ += tDeltaZ; }
				}
			}

			return hit;
		}

		bool IsConvexQuadrilateral(OpenMesh::EdgeHandle eh) const
		{
			if (is_boundary(eh)) return false;

			auto h0 = halfedge_handle(eh, 0);
			auto h1 = halfedge_handle(eh, 1);

			auto v_top = to_vertex_handle(next_halfedge_handle(h0));
			auto v_left = to_vertex_handle(h1);
			auto v_bottom = to_vertex_handle(next_halfedge_handle(h1));
			auto v_right = to_vertex_handle(h0);

			Eigen::Vector3f p0(point(v_top).data());
			Eigen::Vector3f p1(point(v_left).data());
			Eigen::Vector3f p2(point(v_bottom).data());
			Eigen::Vector3f p3(point(v_right).data());

			Eigen::Vector3f e0 = p1 - p0;
			Eigen::Vector3f e1 = p2 - p1;
			Eigen::Vector3f e2 = p3 - p2;
			Eigen::Vector3f e3 = p0 - p3;

			Eigen::Vector3f n_raw = e0.cross(e1) + e2.cross(e3);
			float n_len = n_raw.norm();

			// Degenerate quad: total area is effectively zero, flipping is unsafe
			if (n_len < 1e-12f) return false;

			Eigen::Vector3f n = n_raw / n_len;

			float c0 = e0.cross(e1).dot(n);
			float c1 = e1.cross(e2).dot(n);
			float c2 = e2.cross(e3).dot(n);
			float c3 = e3.cross(e0).dot(n);

			// Scale-relative tolerance: c values have area dimension, so the
			// threshold must scale with the quad area (n_len is twice the area).
			// A small negative bound tolerates nearly collinear corners.
			float tolerance = -1e-4f * n_len;

			if (c0 < tolerance || c1 < tolerance || c2 < tolerance || c3 < tolerance)
			{
				return false;
			}

			return true;
		}
	};
}