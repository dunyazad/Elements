#include "pch.h"

#include <Helium/PointCloudProcessing/PointCloudGenerateMesh.h>

#include <execution>

#include <Helium/HeliumCore.h>
#include <Helium/SpatialPartitionings/SparseDataBlock.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

namespace PointCloudProcessing
{
	PointCloudGenerateMesh::PointCloudGenerateMesh()
		: PointCloudProcessor(PointCloudProcessorType::GenerateMesh)
	{
	}

	std::vector<uint8_t> PointCloudGenerateMesh::Process(const PointCloudProcessorParameters& parameters)
	{
		TS(PointCloudGenerateMesh);

		int pointCloudID = -1;
		int visualizationMode = 0;

		pointCloudID = parameters.GetParameter<int>("PointCloudID", pointCloudID);
		visualizationMode = parameters.GetParameter<int>("VisualizationMode", visualizationMode);

		InfoLog("", "Starting Mesh Generation (ID=%d)", pointCloudID);

		std::vector<uint8_t> outlierMarking;

		auto currentPointCloud = Helium.GetPointCloud(pointCloudID);
		if (nullptr == currentPointCloud)
		{
			ErrorLog("", "PointCloud with ID %d not found.", pointCloudID);
			return outlierMarking;
		}

		auto sparseDataBlock = Helium.GetSparseDataBlock(pointCloudID);
		if (nullptr == sparseDataBlock)
		{
			Helium.BuildSpatialPartitionings(pointCloudID);
			sparseDataBlock = Helium.GetSparseDataBlock(pointCloudID);
		}

		sparseDataBlock->Build(currentPointCloud);

		TS(Generate_Mesh);

		triangles.clear();
		holeEdges.clear();
		float isoLevel = 0.0f;

		std::vector<DataBlock*> blocks;
		blocks.reserve(sparseDataBlock->dataBlocks.size());
		for (auto& pair : sparseDataBlock->dataBlocks)
		{
			blocks.push_back(pair.second.get());
		}

		size_t estTris = blocks.size() * 96;
		triangles.reserve(estTris);

		std::unordered_map<GridKey, SNVertex, GridKeyHash> snVertices;
		snVertices.reserve(size_t(estTris * 0.6f));
		snVertices.max_load_factor(0.7f);

		std::mutex vertexMutex;
		std::mutex triMutex;

		const Eigen::Vector3i corners[8] = { {0,0,0}, {1,0,0}, {1,0,1}, {0,0,1}, {0,1,0}, {1,1,0}, {1,1,1}, {0,1,1} };
		const int edgePairs[12][2] = { {0,1}, {1,2}, {2,3}, {3,0}, {4,5}, {5,6}, {6,7}, {7,4}, {0,4}, {1,5}, {2,6}, {3,7} };

		// Pass 1: Compute vertices
		std::for_each(std::execution::par, blocks.begin(), blocks.end(), [&](DataBlock* block)
			{
				std::vector<std::pair<GridKey, SNVertex>> localVerts;
				localVerts.reserve(64);

				Eigen::Vector3f diff = block->blockMin - sparseDataBlock->gridOrigin;
				int startGx = (int)(diff.x() / sparseDataBlock->voxelSize + 0.5f);
				int startGy = (int)(diff.y() / sparseDataBlock->voxelSize + 0.5f);
				int startGz = (int)(diff.z() / sparseDataBlock->voxelSize + 0.5f);

				for (int z = 0; z < SpatialPartitioningConfiguration::voxelsPerBlockAxis; ++z)
				{
					for (int y = 0; y < SpatialPartitioningConfiguration::voxelsPerBlockAxis; ++y)
					{
						for (int x = 0; x < SpatialPartitioningConfiguration::voxelsPerBlockAxis; ++x)
						{
							int gx = startGx + x;
							int gy = startGy + y;
							int gz = startGz + z;

							float dists[8];
							Eigen::Vector3f colors[8];
							Eigen::Vector3f normals[8];
							int insideCount = 0;
							bool allValid = true;

							for (int i = 0; i < 8; ++i)
							{
								const auto* v = sparseDataBlock->GetVoxelByIndex(gx + corners[i].x(), gy + corners[i].y(), gz + corners[i].z());
								if (!v || !v->valid)
								{
									allValid = false;
									break;
								}
								dists[i] = v->signedDistance;
								colors[i] = v->color;
								normals[i] = v->normal;
								if (dists[i] < isoLevel) insideCount++;
							}

							if (!allValid || insideCount == 0 || insideCount == 8) continue;

							Eigen::Vector3f avgPos(0.0f, 0.0f, 0.0f);
							Eigen::Vector3f avgColor(0.0f, 0.0f, 0.0f);
							Eigen::Vector3f avgNormal(0.0f, 0.0f, 0.0f);
							int intersections = 0;

							for (int e = 0; e < 12; ++e)
							{
								int idx1 = edgePairs[e][0];
								int idx2 = edgePairs[e][1];
								if ((dists[idx1] < isoLevel) != (dists[idx2] < isoLevel))
								{
									float t = (isoLevel - dists[idx1]) / (dists[idx2] - dists[idx1]);
									Eigen::Vector3f p1 = sparseDataBlock->gridOrigin + Eigen::Vector3f((float)gx + corners[idx1].x(), (float)gy + corners[idx1].y(), (float)gz + corners[idx1].z()) * sparseDataBlock->voxelSize;
									Eigen::Vector3f p2 = sparseDataBlock->gridOrigin + Eigen::Vector3f((float)gx + corners[idx2].x(), (float)gy + corners[idx2].y(), (float)gz + corners[idx2].z()) * sparseDataBlock->voxelSize;

									avgPos += (p1 * (1.0f - t) + p2 * t);
									avgColor += (colors[idx1] * (1.0f - t) + colors[idx2] * t);
									avgNormal += (normals[idx1] * (1.0f - t) + normals[idx2] * t);
									intersections++;
								}
							}

							if (intersections > 0)
							{
								SNVertex v;
								v.pos = avgPos / (float)intersections;
								v.color = avgColor / (float)intersections;
								v.normal = avgNormal.normalized();
								localVerts.push_back({ {gx, gy, gz}, v });
							}
						}
					}
				}

				if (!localVerts.empty())
				{
					std::lock_guard<std::mutex> lock(vertexMutex);
					for (const auto& kv : localVerts) snVertices[kv.first] = kv.second;
				}
			});

		// Pass 2: Generate triangles
		std::for_each(std::execution::par, blocks.begin(), blocks.end(), [&](DataBlock* block)
			{
				std::vector<Triangle> localTris;
				localTris.reserve(128);

				Eigen::Vector3f diff = block->blockMin - sparseDataBlock->gridOrigin;
				int startGx = (int)(diff.x() / sparseDataBlock->voxelSize + 0.5f);
				int startGy = (int)(diff.y() / sparseDataBlock->voxelSize + 0.5f);
				int startGz = (int)(diff.z() / sparseDataBlock->voxelSize + 0.5f);

				for (int z = 0; z < SpatialPartitioningConfiguration::voxelsPerBlockAxis; ++z)
				{
					for (int y = 0; y < SpatialPartitioningConfiguration::voxelsPerBlockAxis; ++y)
					{
						for (int x = 0; x < SpatialPartitioningConfiguration::voxelsPerBlockAxis; ++x)
						{
							int gx = startGx + x;
							int gy = startGy + y;
							int gz = startGz + z;

							const auto* vCurr = sparseDataBlock->GetVoxelByIndex(gx, gy, gz);
							if (!vCurr || !vCurr->valid) continue;
							bool bCurr = vCurr->signedDistance < isoLevel;

							auto AddQuadLoc = [&](const GridKey& k1, const GridKey& k2, const GridKey& k3, const GridKey& k4, bool flip)
								{
									if (snVertices.count(k1) && snVertices.count(k2) && snVertices.count(k3) && snVertices.count(k4))
									{
										const auto& v0 = flip ? snVertices[k4] : snVertices[k1];
										const auto& v1 = flip ? snVertices[k3] : snVertices[k2];
										const auto& v2 = flip ? snVertices[k2] : snVertices[k3];
										const auto& v3 = flip ? snVertices[k1] : snVertices[k4];

										Triangle t1, t2;
										t1.v[0] = v0.pos; t1.v[1] = v1.pos; t1.v[2] = v2.pos;
										t1.c[0] = v0.color; t1.c[1] = v1.color; t1.c[2] = v2.color;
										t1.n[0] = v0.normal; t1.n[1] = v1.normal; t1.n[2] = v2.normal;

										t2.v[0] = v0.pos; t2.v[1] = v2.pos; t2.v[2] = v3.pos;
										t2.c[0] = v0.color; t2.c[1] = v2.color; t2.c[2] = v3.color;
										t2.n[0] = v0.normal; t2.n[1] = v2.normal; t2.n[2] = v3.normal;

										localTris.push_back(t1);
										localTris.push_back(t2);
									}
								};

							const auto* vX = sparseDataBlock->GetVoxelByIndex(gx + 1, gy, gz);
							if (vX && vX->valid && (bCurr != (vX->signedDistance < isoLevel)))
								AddQuadLoc({ gx, gy - 1, gz - 1 }, { gx, gy, gz - 1 }, { gx, gy, gz }, { gx, gy - 1, gz }, !bCurr);

							const auto* vY = sparseDataBlock->GetVoxelByIndex(gx, gy + 1, gz);
							if (vY && vY->valid && (bCurr != (vY->signedDistance < isoLevel)))
								AddQuadLoc({ gx - 1, gy, gz - 1 }, { gx, gy, gz - 1 }, { gx, gy, gz }, { gx - 1, gy, gz }, bCurr);

							const auto* vZ = sparseDataBlock->GetVoxelByIndex(gx, gy, gz + 1);
							if (vZ && vZ->valid && (bCurr != (vZ->signedDistance < isoLevel)))
								AddQuadLoc({ gx - 1, gy - 1, gz }, { gx, gy - 1, gz }, { gx, gy, gz }, { gx - 1, gy, gz }, !bCurr);
						}
					}
				}

				if (!localTris.empty())
				{
					std::lock_guard<std::mutex> lock(triMutex);
					triangles.insert(triangles.end(), localTris.begin(), localTris.end());
				}
			});

		TE(Generate_Mesh);

		for (auto& t : triangles)
		{
			VD::AddTriangle("Mesh",
				t.v[0], t.v[1], t.v[2],
				{ t.c[0].x(), t.c[0].y(), t.c[0].z(), 1.0f },
				{ t.c[0].x(), t.c[0].y(), t.c[0].z(), 1.0f },
				{ t.c[0].x(), t.c[0].y(), t.c[0].z(), 1.0f });
		}

		{
			float tol = 0.0001f;
			std::map<std::pair<int, int>, int> edges;
			std::unordered_map<GridKey, int, GridKeyHash> vMap;
			vMap.reserve(triangles.size());
			int vCount = 0;
			std::vector<int> triIndices; triIndices.reserve(triangles.size() * 3);
			std::vector<Eigen::Vector3f> tempVerts; tempVerts.reserve(triangles.size());

			for (const auto& t : triangles)
			{
				for (int i = 0; i < 3; ++i)
				{
					GridKey key = { (int)(t.v[i].x() / tol), (int)(t.v[i].y() / tol), (int)(t.v[i].z() / tol) };
					if (vMap.find(key) == vMap.end())
					{
						vMap[key] = vCount++;
						tempVerts.push_back(t.v[i]);
					}
					triIndices.push_back(vMap[key]);
				}
			}
			for (size_t i = 0; i < triIndices.size(); i += 3)
			{
				int idx[3] = { triIndices[i], triIndices[i + 1], triIndices[i + 2] };
				for (int k = 0; k < 3; ++k)
				{
					int a = idx[k]; int b = idx[(k + 1) % 3];
					if (a > b) std::swap(a, b);
					edges[{a, b}]++;
				}
			}
			holeEdges.clear();
			for (auto& kv : edges)
			{
				if (kv.second == 1)
				{
					holeEdges.push_back({ tempVerts[kv.first.first], tempVerts[kv.first.second] });

					VD::AddLine("HoleEdges",
						tempVerts[kv.first.first],
						tempVerts[kv.first.second],
						{ 1.0f, 0.0f, 0.0f, 1.0f });
				}
			}
		}

		TE(PointCloudGenerateMesh);

		return outlierMarking;
	}
};
