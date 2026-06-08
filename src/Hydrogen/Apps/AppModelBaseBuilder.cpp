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


class AppModelBaseBuilder : public App
{
public:
	virtual void Execute() override
	{
		TS(LoadingMesh_A);
		meshes.emplace_back(std::make_unique<RGO::HeliumMesh>());
		RGO::HeliumMesh& mesh_a = *meshes.back();
		LoadWelded(mesh_a, "D:\\Resources\\3D\\STL\\maxillar_open.stl");
		TE(LoadingMesh_A);

		auto borderLoops = mesh_a.GetBorderLoops();

		for (auto& loop : borderLoops)
		{
			for (size_t i = 0; i < loop.size(); i++)
			{
				auto p0 = mesh_a.point(loop[i]);
				auto p1 = mesh_a.point(loop[(i + 1) % loop.size()]);

				printf("Border edge: (%f, %f, %f) -> (%f, %f, %f)\n", p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]);

				VD::AddLine("BorderLoop", { p0[0], p0[1], p0[2] }, { p1[0], p1[1], p1[2] }, Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f));
			}
		}

		RGO::OperatorCreateSkirt op(&mesh_a, 10.0f, {1.0f, 0.0f, 1.0f});
		op.Execute();

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

	std::vector<std::unique_ptr<RGO::HeliumMesh>> meshes;
};

REGISTER_APP(AppModelBaseBuilder, "AppModelBaseBuilder");
