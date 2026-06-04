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

#include "RobustGeometricOperations.h"
#include <OpenMesh/Core/IO/MeshIO.hh>
#include "HeliumMesh.h"


class AppRGO : public App
{
public:
	virtual void Execute() override
	{
		TS(LoadingMesh_A);
		meshes.emplace_back(std::make_unique<RGO::HeliumMesh>());
		RGO::HeliumMesh& mesh_a = *meshes.back();
		LoadWelded(mesh_a, "D:\\Resources\\3D\\STL\\A_Maxillar.stl");
		//LoadWelded(mesh_a, "D:\\Resources\\3D\\STL\\Gadget.stl");
		//mesh_a.BuildBox(
		//	Eigen::Vector3f::Zero(),
		//	Eigen::Vector3f(50.0f, 10.0f, 4.0f),
		//	Eigen::MakeTransform(
		//		Eigen::Vector3f(25.0f, 0.0f, -2.0f),
		//		Eigen::Vector3f::UnitY(),
		//		0.0f, Eigen::Vector3f::Ones()));
		//TE(LoadingMesh_A);

		TS(LoadingMesh_B);
		meshes.emplace_back(std::make_unique<RGO::HeliumMesh>());
		RGO::HeliumMesh& mesh_b = *meshes.back();
		mesh_b.mesh_color = Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f);
		//LoadWelded(mesh_b, "D:\\Resources\\3D\\STL\\A_Tooth.stl");
		//LoadWelded(mesh_b, "D:\\Resources\\3D\\STL\\Doughnut.stl");
		//mesh_b.Build3DText("Hello 안녕 123 !@#", "..\\..\\res\\Fonts\\NanumGothic\\NanumGothic.ttf", 6.0f, 5.0f);
		TE(LoadingMesh_B);

		TS(BuildSpatialHashMap_A);
		mesh_a.BuildSpatialHashMap();
		TE(BuildSpatialHashMap_A);

		TS(BuildSpatialHashMap_B);
		mesh_b.BuildSpatialHashMap();
		TE(BuildSpatialHashMap_B);

		TS(ComputeIntersectionLoops);
		RGO::OperatorIntersectionLoops op(&mesh_a, &mesh_b);
		bool loops_ok = op.Execute();
		TE(ComputeIntersectionLoops);

		TS(CoRefine);
		bool corefine_ok = false;
		if (loops_ok)
		{
			RGO::OperatorCoRefine corefine(&mesh_a, &mesh_b, &op);
			corefine_ok = corefine.Execute();
		}
		else
		{
			std::cout << "[Error] AppRGO: intersection loops invalid, skipping co-refinement." << std::endl;
		}
		TE(CoRefine);

		TS(BooleanSeamAndPatches);
		// Stage gate: OperatorBoolean currently runs only up to seam
		// reconstruction and seam-bounded flood fill, with validation.
		// Classification and assembly stay disconnected until this stage
		// is verified on real data.
		RGO::HeliumMesh* mesh_result = nullptr;
		std::unique_ptr<RGO::OperatorBoolean> boolean_op;
		bool boolean_stage_ok = false;
		if (corefine_ok)
		{
			meshes.emplace_back(std::make_unique<RGO::HeliumMesh>());
			mesh_result = meshes.back().get();
			mesh_result->mesh_color = Eigen::Vector4f(1.0f, 0.5f, 0.0f, 1.0f);

			boolean_op = std::make_unique<RGO::OperatorBoolean>(
				RGO::OperatorBoolean::Intersection, &mesh_a, &mesh_b, &op, mesh_result);
			boolean_stage_ok = boolean_op->Execute();

			if (false == boolean_stage_ok)
			{
				std::cout << "[Error] AppRGO: seam or patch stage failed validation." << std::endl;
			}
		}
		else
		{
			std::cout << "[Error] AppRGO: co-refinement failed, skipping boolean stages." << std::endl;
		}
		TE(BooleanSeamAndPatches);

		TS(VisualizeSeamEdges);
		// Blue lines are seam edges reconstructed on each mesh. They must
		// form the same closed intersection curve on both meshes.
		if (boolean_stage_ok)
		{
			auto draw_seam = [](RGO::HeliumMesh& m, const std::vector<char>& edge_is_seam, const char* layer)
				{
					for (size_t i = 0; i < m.n_edges(); ++i)
					{
						if (0 == edge_is_seam[i]) continue;

						auto eh = m.edge_handle(static_cast<int>(i));
						if (m.status(eh).deleted()) continue;

						auto heh = m.halfedge_handle(eh, 0);
						auto p0 = m.point(m.from_vertex_handle(heh));
						auto p1 = m.point(m.to_vertex_handle(heh));
						VD::AddLine(
							layer,
							Eigen::Vector3f(p0[0], p0[1], p0[2]),
							Eigen::Vector3f(p1[0], p1[1], p1[2]),
							Eigen::Vector4f(0.0f, 0.0f, 1.0f, 1.0f));
					}
				};
			draw_seam(mesh_a, boolean_op->GetSideDataA().edge_is_seam, "SeamEdges");
			draw_seam(mesh_b, boolean_op->GetSideDataB().edge_is_seam, "SeamEdges");
			VD::SetLineWidth("SeamEdges", 5.0f);
		}
		TE(VisualizeSeamEdges);

		TS(VisualizeBoundaryEdges);
		// Any red line here is a residual boundary edge. The goal state
		// renders nothing in this layer.
		auto draw_boundary = [](RGO::HeliumMesh& m, const char* layer)
			{
				for (size_t i = 0; i < m.n_edges(); ++i)
				{
					auto eh = m.edge_handle(static_cast<int>(i));
					if (m.status(eh).deleted()) continue;
					if (false == m.is_boundary(eh)) continue;

					auto heh = m.halfedge_handle(eh, 0);
					auto p0 = m.point(m.from_vertex_handle(heh));
					auto p1 = m.point(m.to_vertex_handle(heh));
					VD::AddLine(
						layer,
						Eigen::Vector3f(p0[0], p0[1], p0[2]),
						Eigen::Vector3f(p1[0], p1[1], p1[2]),
						Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
				}
			};
		draw_boundary(mesh_a, "BoundaryEdges");
		draw_boundary(mesh_b, "BoundaryEdges");
		if (nullptr != mesh_result)
		{
			draw_boundary(*mesh_result, "BoundaryEdges");
		}
		VD::SetLineWidth("BoundaryEdges", 5.0f);
		TE(VisualizeBoundaryEdges);

		TS(VisualizePatches);
		// Each flood-fill patch becomes its own HeliumMesh with a distinct
		// color, so patch separation is verifiable visually. The original
		// meshes are cleared afterwards because the patch meshes occupy
		// the exact same geometry and would z-fight with them. All line
		// layers (seam, boundary) are drawn above, BEFORE the clear.
		if (boolean_stage_ok)
		{
			// Golden ratio hue stepping gives well separated colors for
			// any number of patches without a fixed palette.
			auto patch_color = [](int index) -> Eigen::Vector4f
				{
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
				};

			auto build_patch_meshes = [&](RGO::HeliumMesh& source, const RGO::OperatorBoolean::MeshSideData& data, int& color_index)
				{
					for (int p = 0; p < data.patch_count; ++p)
					{
						std::vector<Eigen::Vector3f> wp;
						std::vector<Eigen::Vector3i> wi;
						robin_hood::unordered_map<Eigen::Vector3f, int, RGO::Vector3fHash, RGO::Vector3fEqual> vm;

						for (size_t i = 0; i < source.n_faces(); ++i)
						{
							auto fh = source.face_handle(static_cast<int>(i));
							if (source.status(fh).deleted()) continue;
							if (data.face_patch[i] != p) continue;

							Eigen::Vector3f v0, v1, v2;
							source.GetFaceVertices(fh, v0, v1, v2);
							const Eigen::Vector3f tri_pts[3] = { v0, v1, v2 };

							Eigen::Vector3i fi;
							for (int j = 0; j < 3; ++j)
							{
								auto it = vm.find(tri_pts[j]);
								if (it != vm.end())
								{
									fi[j] = it->second;
								}
								else
								{
									int ni = static_cast<int>(wp.size());
									wp.push_back(tri_pts[j]);
									vm[tri_pts[j]] = ni;
									fi[j] = ni;
								}
							}
							wi.push_back(fi);
						}

						if (wi.empty()) continue;

						meshes.emplace_back(std::make_unique<RGO::HeliumMesh>());
						meshes.back()->mesh_color = patch_color(color_index++);
						meshes.back()->Build(wp, wi);
					}

					// Clear the source so it cannot cover the colored
					// patch meshes built from it.
					source.clear();
				};

			int color_index = 0;
			build_patch_meshes(mesh_a, boolean_op->GetSideDataA(), color_index);
			build_patch_meshes(mesh_b, boolean_op->GetSideDataB(), color_index);
		}
		TE(VisualizePatches);

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

	std::vector<std::unique_ptr<RGO::HeliumMesh>> meshes;
};

REGISTER_APP(AppRGO, "AppRGO");
