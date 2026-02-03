#include "Apps.h"

class AppICP : public App
{
public:
    virtual void Execute() override
    {
		PLYFormat ply;
		if (!ply.Deserialize("D:\\Resources\\Default\\Compound.ply")) return;

		//VD::AddSphereBatch("PointCloud", ply.GetPoints(), ply.GetNormals(), 0.05f, ply.GetColors());

        CuPointCloud targetPointCloud;
		targetPointCloud.FromHostPointers(
            (float3*)ply.GetPoints().data(),
            (float3*)ply.GetNormals().data(),
            (float4*)ply.GetColors().data(),
            ply.GetPoints().size());

        CuSparseCells targetCells;
        targetCells.Build(&targetPointCloud, 0.3f);

        std::ifstream ifs("D:\\Resources\\Default\\Patches.bin", std::ios::binary);
        if (!ifs.is_open())
        {
			std::cout << "[Error] Failed to open Patches.bin" << std::endl;
            return;
        }

        CamInfo_ cam;
        Eigen::Matrix4f camRT;
        ifs.read(reinterpret_cast<char*>(&cam), sizeof(CamInfo_));
        ifs.read(reinterpret_cast<char*>(camRT.data()), sizeof(float) * 16);

        size_t numberOfPatches = 0;
        ifs.read(reinterpret_cast<char*>(&numberOfPatches), sizeof(size_t));

		robin_hood::unordered_map<size_t, std::tuple<int, Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector4f>> donwnSampling;

        for (size_t i = 0; i < numberOfPatches; i++)
        {
            TS(patch);
            size_t patchIndex = 0;
            ifs.read(reinterpret_cast<char*>(&patchIndex), sizeof(size_t));
            Eigen::Matrix4f rt0;
            ifs.read(reinterpret_cast<char*>(rt0.data()), sizeof(float) * 16);
            ifs.seekg(sizeof(float) * 16, std::ios::cur);

            size_t numPts = 0;
            ifs.read(reinterpret_cast<char*>(&numPts), sizeof(size_t));

            std::vector<Eigen::Vector3f> pts(numPts), normals(numPts);
            std::vector<Eigen::Vector3b> colors(numPts);
            ifs.read(reinterpret_cast<char*>(pts.data()), sizeof(Eigen::Vector3f) * numPts);
            ifs.read(reinterpret_cast<char*>(normals.data()), sizeof(Eigen::Vector3f) * numPts);
            ifs.read(reinterpret_cast<char*>(colors.data()), sizeof(Eigen::Vector3b) * numPts);

            size_t numPts45 = 0;
            ifs.read(reinterpret_cast<char*>(&numPts45), sizeof(size_t));
            ifs.seekg(numPts45 * (sizeof(Eigen::Vector3f) * 2), std::ios::cur);

            Eigen::Vector3f sensorPos = rt0.block<3, 1>(0, 3);
            Eigen::Matrix3f rot = rt0.block<3, 3>(0, 0);

			
            for (size_t j = 0; j < numPts; ++j)
            {
                Eigen::Vector3f pW = rot * pts[j] + sensorPos;
                Eigen::Vector3f nW = (rot * normals[j]).normalized();
                //VD::AddSphere("PointCloud", pW, nW, 0.05f, { (float)colors[j].x() / 255.0f, (float)colors[j].y() / 255.0f , (float)colors[j].z() / 255.0f , 1.0f });
				//source_points.push_back(pW);
				//source_normals.push_back(nW);
				//source_colors.push_back(Eigen::Vector4f{ (float)colors[j].x() / 255.0f, (float)colors[j].y() / 255.0f , (float)colors[j].z() / 255.0f , 1.0f });

				auto x = static_cast<size_t>(std::floor(pW.x() / 0.1f));
                auto y = static_cast<size_t>(std::floor(pW.y() / 0.1f));
                auto z = static_cast<size_t>(std::floor(pW.z() / 0.1f));
				auto hash = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
                if (donwnSampling.find(hash) == donwnSampling.end())
                {
                    donwnSampling[hash] = std::make_tuple(1, pW, nW, Eigen::Vector4f{ (float)colors[j].x() / 255.0f, (float)colors[j].y() / 255.0f, (float)colors[j].z() / 255.0f, 1.0f });
                }
                else
                {
                    auto& tup = donwnSampling[hash];
                    std::get<0>(tup) += 1;
                    std::get<1>(tup) += pW;
                    std::get<2>(tup) += nW;
					std::get<3>(tup) += Eigen::Vector4f{ (float)colors[j].x() / 255.0f, (float)colors[j].y() / 255.0f, (float)colors[j].z() / 255.0f, 1.0f };
                }
            }

            {
				//std::string patchTag = "SourcePatch_" + std::to_string(i);
                //VD::AddSphereBatch("patchTag", source_points, source_normals, 0.05f, source_colors);
                //source_points.clear();
				//source_normals.clear();
                //source_colors.clear();
            }

            TE(patch);

            //break;

        }

        for (auto& kvp : donwnSampling)
        {
			auto& [count, pSum, nSum, cSum] = kvp.second;
            Eigen::Vector3f pAvg = pSum / static_cast<float>(count);
			Eigen::Vector3f nAvg = nSum.normalized();
			Eigen::Vector4f cAvg = cSum / static_cast<float>(count);

			VD::AddSphere("DownSampled", pAvg, nAvg, 0.05f, { cAvg.x(), cAvg.y(), cAvg.z(), cAvg.w() });
        }
    }
};

REGISTER_APP(AppICP, "AppICP");
