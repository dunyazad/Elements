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


class AppRGO : public App
{
public:
	virtual void Initialize() override
	{
	}

	virtual void Execute() override
	{
		TS(LoadingMesh_A);
		meshes.emplace_back(std::make_unique<RGO::HeliumMesh>());
		RGO::HeliumMesh& mesh_a = *meshes.back();
		//LoadWelded(mesh_a, "D:\\Resources\\3D\\STL\\A_Maxillar_T.stl");
		//LoadWelded(mesh_a, "D:\\Resources\\3D\\STL\\Gadget.stl");
		//LoadWelded(mesh_a, "D:\\Resources\\3D\\STL\\rabbit.stl");
		mesh_a.BuildSineWaveBox(
			Eigen::Vector3f::Zero(),
			Eigen::Vector3f(80.0f, 12.0f, 4.0f),
			1.5f, 4.0f, 120, 12,
			Eigen::MakeTransform(
				Eigen::Vector3f(25.0f, 0.0f, 0.0f),
				Eigen::Vector3f::UnitY(),
				0.0f, Eigen::Vector3f::Ones()));
		//mesh_a.BuildSineWaveBox(
		//	Eigen::Vector3f::Zero(),
		//	Eigen::Vector3f(60.0f, 12.0f, 4.0f),
		//	3.8f, 12.0f, 480, 48,
		//	Eigen::MakeTransform(
		//		Eigen::Vector3f(25.0f, 0.0f, 0.0f),
		//		Eigen::Vector3f::UnitY(),
		//		0.0f, Eigen::Vector3f::Ones()));
		TE(LoadingMesh_A);

		TS(LoadingMesh_B);
		meshes.emplace_back(std::make_unique<RGO::HeliumMesh>());
		RGO::HeliumMesh& mesh_b = *meshes.back();
		mesh_b.mesh_color = Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f);
		//LoadWelded(mesh_b, "D:\\Resources\\3D\\STL\\A_Tooth.stl");
		//LoadWelded(mesh_b, "D:\\Resources\\3D\\STL\\Doughnut.stl");
		//LoadWelded(mesh_b, "D:\\Resources\\3D\\STL\\rabbit_upside_down.stl");
		mesh_b.Build3DText("Hello 안녕 123 !@# 漢字", "..\\..\\res\\Fonts\\NanumGothic\\NanumGothic.ttf", 6.0f, 5.0f);
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

		TS(Boolean);
		// Full boolean pipeline: seam reconstruction, seam-bounded flood
		// fill, classification (including OnSurface for coplanar overlap
		// regions) and assembly, each stage validated. On success the
		// result mesh is a watertight solid (or a trimmed open surface
		// when one input is open).
		RGO::HeliumMesh* mesh_result = nullptr;
		std::unique_ptr<RGO::OperatorBoolean> boolean_op;
		bool boolean_ok = false;
		if (corefine_ok)
		{
			meshes.emplace_back(std::make_unique<RGO::HeliumMesh>());
			mesh_result = meshes.back().get();
			mesh_result->mesh_color = Eigen::Vector4f(1.0f, 0.5f, 0.0f, 1.0f);

			boolean_op = std::make_unique<RGO::OperatorBoolean>(RGO::OperatorBoolean::Union, &mesh_a, &mesh_b, &op, mesh_result);
			boolean_ok = boolean_op->Execute();

			if (false == boolean_ok)
			{
				std::cout << "[Error] AppRGO: boolean failed validation." << std::endl;
			}
		}
		else
		{
			std::cout << "[Error] AppRGO: co-refinement failed, skipping boolean stages." << std::endl;
		}
		TE(Boolean);

		
		//TS(Remesh);
		//// Isotropic remeshing of the boolean result, producing a uniform
		//// triangle distribution suitable as input to the offset stage. The
		//// result is a closed solid, so feature preservation keeps the box
		//// edges and the glyph outlines sharp; without it those creases
		//// would round off. Runs only on a validated boolean result.
		//bool remesh_ok = false;
		//if (boolean_ok && nullptr != mesh_result)
		//{
		//	// target_edge_length is absolute world units. The sine-wave box
		//	// spans 60 in x and the glyphs are about 6 tall with strokes
		//	// near 0.5 wide, so a target around 0.5 keeps glyph detail while
		//	// still simplifying the flat sine-wave regions. feature angle 30
		//	// degrees flags the box corners and glyph creases as features.
		//	RGO::OperatorRemesh remesh(mesh_result, 0.5f, 5, 50.0f);
		//	remesh_ok = remesh.Execute();

		//	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> feature_lines;
		//	remesh.GetFeatureEdgeLines(feature_lines);

		//	for (const auto& line : feature_lines)
		//	{
		//		VD::AddLine(
		//			"FeatureEdges",
		//			line.first,
		//			line.second,
		//			Eigen::Vector4f(1.0f, 0.0f, 1.0f, 1.0f));
		//	}
		//	VD::SetLineWidth("FeatureEdges", 4.0f);

		//	std::cout << "[Info] AppRGO: drew " << feature_lines.size()
		//		<< " remesh feature edges." << std::endl;

		//	TS(VisualizeBadVertices);
		//	// Yellow spheres mark vertices where the feature curve network is
		//	// broken (degree != 2 but not pinned as a corner). They localize
		//	// exactly where detection or an operator tore the curve.
		//	if (remesh_ok)
		//	{
		//		const std::vector<Eigen::Vector3f>& bad = remesh.GetBadFeatureVertices();
		//		for (const auto& v : bad)
		//		{
		//			VD::AddSphere(
		//				"BadFeatureVertices",
		//				v,
		//				Eigen::Vector3f(0.0f, 0.0f, 1.0f),
		//				0.5f,
		//				Eigen::Vector4f(1.0f, 1.0f, 0.0f, 1.0f));
		//		}
		//		std::cout << "[Info] AppRGO: drew " << bad.size()
		//			<< " bad feature vertices." << std::endl;
		//	}
		//	TE(VisualizeBadVertices);

		//	TS(VisualizeFlippedFaces);
		//	// Red spheres mark faces CountFlippedFaces judged as folded
		//	// relative to the reference surface. Used to confirm whether these
		//	// are real folds or a misjudgment of the check on thin walls.
		//	if (remesh_ok)
		//	{
		//		const std::vector<Eigen::Vector3f>& flipped = remesh.GetFlippedFaceCenters();
		//		for (const auto& c : flipped)
		//		{
		//			VD::AddSphere(
		//				"FlippedFaces",
		//				c,
		//				Eigen::Vector3f(0.0f, 0.0f, 1.0f),
		//				0.08f,
		//				Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
		//		}
		//		std::cout << "[Info] AppRGO: drew " << flipped.size()
		//			<< " flipped face centers." << std::endl;
		//	}
		//	TE(VisualizeFlippedFaces);
		//}
		//else
		//{
		//	std::cout << "[Info] AppRGO: skipping remesh (no valid boolean result)." << std::endl;
		//}
		//TE(Remesh);
		

		TS(VisualizeSeamEdges);
		// Blue lines are seam edges reconstructed on each mesh. They must
		// form the same closed intersection curve on both meshes.
		if (boolean_ok)
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
			// Seam edge flags exist only on sides the operator built:
			// both for solid booleans, only the open side for trims.
			if (boolean_op->GetSideDataA().edge_is_seam.size() == mesh_a.n_edges())
			{
				draw_seam(mesh_a, boolean_op->GetSideDataA().edge_is_seam, "SeamEdges");
			}
			if (boolean_op->GetSideDataB().edge_is_seam.size() == mesh_b.n_edges())
			{
				draw_seam(mesh_b, boolean_op->GetSideDataB().edge_is_seam, "SeamEdges");
			}
			VD::SetLineWidth("SeamEdges", 5.0f);
		}
		TE(VisualizeSeamEdges);

		TS(VisualizeBoundaryEdges);
		// Any red line on the RESULT is a failure of the pipeline. Red
		// lines on the inputs are legitimate only for open inputs.
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

		TS(VisualizeResult);
		// The boolean result occupies the same geometry as the inputs and
		// the patch meshes, so exactly ONE representation is kept visible.
		// visualize_patches = false shows the orange result solid;
		// true shows the colored per-patch decomposition instead and
		// saves each patch as an STL file. All line layers (seam,
		// boundary) are drawn above, BEFORE any clear.
		const bool visualize_patches = false;
		if (boolean_ok)
		{
			if (visualize_patches)
			{
				std::vector<std::pair<std::string, RGO::HeliumMesh*>> patch_meshes;
				int color_index = 0;
				BuildPatchMeshes(mesh_a, boolean_op->GetSideDataA(), "A", color_index, patch_meshes);
				BuildPatchMeshes(mesh_b, boolean_op->GetSideDataB(), "B", color_index, patch_meshes);

				SavePatchMeshesAsSTL(patch_meshes, "D:\\Temp\\Patch_");

				mesh_result->clear();
			}
			else
			{
				// Clear the inputs so they cannot cover the result.
				mesh_a.clear();
				mesh_b.clear();
			}

			SavePatchMeshesAsSTL({ { "Result", mesh_result } }, "D:\\Temp\\");
		}
		TE(VisualizeResult);

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

	void BuildPatchMeshes(
		RGO::HeliumMesh& source,
		const RGO::OperatorBoolean::MeshSideData& data,
		const char* label,
		int& color_index,
		std::vector<std::pair<std::string, RGO::HeliumMesh*>>& out_patches)
	{
		// Splitting itself is a Mesh capability: one shared flood fill
		// implementation serves both the boolean operator and this app.
		std::vector<std::vector<Eigen::Vector3f>> patch_points;
		std::vector<std::vector<Eigen::Vector3i>> patch_indices;
		int patch_count = source.SplitMeshBySeam(data.edge_is_seam, patch_points, patch_indices);

		for (int p = 0; p < patch_count; ++p)
		{
			if (patch_indices[p].empty()) continue;

			meshes.emplace_back(std::make_unique<RGO::HeliumMesh>());
			RGO::HeliumMesh* patch_mesh = meshes.back().get();
			patch_mesh->mesh_color = PatchColor(color_index++);
			patch_mesh->Build(patch_points[p], patch_indices[p]);

			out_patches.push_back({ std::string(label) + "_" + std::to_string(p), patch_mesh });
		}

		// Clear the source so it cannot cover the colored patch meshes
		// built from it. All line layers must be drawn before this.
		source.clear();
	}

	void SavePatchMeshesAsSTL(
		const std::vector<std::pair<std::string, RGO::HeliumMesh*>>& patches,
		const std::string& path_prefix)
	{
		OpenMesh::IO::Options write_options = OpenMesh::IO::Options::Binary;

		for (const auto& kvp : patches)
		{
			std::string path = path_prefix + kvp.first + ".stl";
			if (OpenMesh::IO::write_mesh(*kvp.second, path, write_options))
			{
				std::cout << "[Info] SavePatchMeshesAsSTL: wrote " << path << std::endl;
			}
			else
			{
				std::cout << "[Error] SavePatchMeshesAsSTL: failed to write " << path << std::endl;
			}
		}
	}

	std::vector<std::unique_ptr<RGO::HeliumMesh>> meshes;
};

REGISTER_APP(AppRGO, "AppRGO");
