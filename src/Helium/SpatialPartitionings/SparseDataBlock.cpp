#include "pch.h"

#include <execution>
#include <numeric>

#include <Helium/SpatialPartitionings/SparseDataBlock.h>
#include <Helium/PointCloud.h>
#include <Helium/SpatialPartitionings/Morton3D.hpp>

#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#undef min
#undef max

void DataBlock::Initialize()
{
	for (int i = 0; i < SpatialPartitioningConfiguration::voxelsPerBlock; ++i)
	{
		voxels[i] = SparseDataBlockVoxel();
	}
}

SparseDataBlockVoxel* SparseDataBlock::GetVoxelByIndex(int gx, int gy, int gz)
{
	int bx = (int)floor((float)gx / SpatialPartitioningConfiguration::voxelsPerBlockAxis);
	int by = (int)floor((float)gy / SpatialPartitioningConfiguration::voxelsPerBlockAxis);
	int bz = (int)floor((float)gz / SpatialPartitioningConfiguration::voxelsPerBlockAxis);

	Eigen::Vector3f blockMin = gridOrigin + Eigen::Vector3f((float)bx * blockSizePerAxis, (float)by * blockSizePerAxis, (float)bz * blockSizePerAxis);
	auto key = Morton3D::EncodeFromPosition(blockMin + Eigen::Vector3f::Constant(voxelSize * 0.1f), gridOrigin, blockSizePerAxis);

	auto it = dataBlocks.find(key);
	if (it == dataBlocks.end()) return nullptr;

	int lx = gx % SpatialPartitioningConfiguration::voxelsPerBlockAxis;
	if (lx < 0) lx += SpatialPartitioningConfiguration::voxelsPerBlockAxis;

	int ly = gy % SpatialPartitioningConfiguration::voxelsPerBlockAxis;
	if (ly < 0) ly += SpatialPartitioningConfiguration::voxelsPerBlockAxis;

	int lz = gz % SpatialPartitioningConfiguration::voxelsPerBlockAxis;
	if (lz < 0) lz += SpatialPartitioningConfiguration::voxelsPerBlockAxis;

	return &it->second->voxels[lz * SpatialPartitioningConfiguration::voxelsPerBlockAxis * SpatialPartitioningConfiguration::voxelsPerBlockAxis + ly * SpatialPartitioningConfiguration::voxelsPerBlockAxis + lx];
}

void SparseDataBlock::Build(const PointCloud* pc)
{
	auto aabbMin = pc->GetAABB().min;

	blockSizePerAxis = voxelSize * SpatialPartitioningConfiguration::voxelsPerBlockAxis;

	gridOrigin.x() = std::floor(aabbMin.x() / blockSizePerAxis) * blockSizePerAxis;
	gridOrigin.y() = std::floor(aabbMin.y() / blockSizePerAxis) * blockSizePerAxis;
	gridOrigin.z() = std::floor(aabbMin.z() / blockSizePerAxis) * blockSizePerAxis;

	TS(Occupy);

	float truncDist = std::max(voxelSize * 4.0f, 0.15f);

	size_t numberOfPoints = pc->Size();
	std::vector<size_t> indices(numberOfPoints);
	std::iota(indices.begin(), indices.end(), 0);

	auto points = pc->GetPositions();
	auto normals = pc->GetNormals();
	auto colors = pc->GetColors();
	
	{
		TS(Pass1_Alloc);

		using KeyPair = std::pair<DataBlockKey, Eigen::Vector3f>;

		std::vector<KeyPair> keysToAllocate;
		keysToAllocate.reserve(numberOfPoints * 2);
		std::mutex vecMutex;

		std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i)
			{
				std::vector<KeyPair> localKeys;
				localKeys.reserve(8);

				Eigen::Vector3f p = points[i];
				Eigen::Vector3f vecFromOrigin = p - gridOrigin;

				int centerGx = (int)std::floor(vecFromOrigin.x() / voxelSize);
				int centerGy = (int)std::floor(vecFromOrigin.y() / voxelSize);
				int centerGz = (int)std::floor(vecFromOrigin.z() / voxelSize);

				int bx = centerGx / SpatialPartitioningConfiguration::voxelsPerBlockAxis;
				int by = centerGy / SpatialPartitioningConfiguration::voxelsPerBlockAxis;
				int bz = centerGz / SpatialPartitioningConfiguration::voxelsPerBlockAxis;

				Eigen::Vector3f blockMin = gridOrigin + Eigen::Vector3f((float)bx * blockSizePerAxis, (float)by * blockSizePerAxis, (float)bz * blockSizePerAxis);
				auto key = Morton3D::EncodeFromPosition(blockMin + Eigen::Vector3f::Constant(voxelSize * 0.1f), gridOrigin, blockSizePerAxis);
				localKeys.push_back({ key, blockMin });

				Eigen::Vector3f localP = p - blockMin;

				float margin = truncDist + voxelSize * 0.5f;

				bool nearX_Neg = localP.x() < margin;
				bool nearX_Pos = localP.x() > blockSizePerAxis - margin;
				bool nearY_Neg = localP.y() < margin;
				bool nearY_Pos = localP.y() > blockSizePerAxis - margin;
				bool nearZ_Neg = localP.z() < margin;
				bool nearZ_Pos = localP.z() > blockSizePerAxis - margin;

				if (nearX_Neg || nearX_Pos || nearY_Neg || nearY_Pos || nearZ_Neg || nearZ_Pos)
				{
					int dx_min = nearX_Neg ? -1 : 0;
					int dx_max = nearX_Pos ? 1 : 0;
					int dy_min = nearY_Neg ? -1 : 0;
					int dy_max = nearY_Pos ? 1 : 0;
					int dz_min = nearZ_Neg ? -1 : 0;
					int dz_max = nearZ_Pos ? 1 : 0;

					for (int dz = dz_min; dz <= dz_max; ++dz)
					{
						for (int dy = dy_min; dy <= dy_max; ++dy)
						{
							for (int dx = dx_min; dx <= dx_max; ++dx)
							{
								if (dx == 0 && dy == 0 && dz == 0) continue;

								Eigen::Vector3f nbMin = gridOrigin + Eigen::Vector3f((float)(bx + dx) * blockSizePerAxis, (float)(by + dy) * blockSizePerAxis, (float)(bz + dz) * blockSizePerAxis);
								auto nKey = Morton3D::EncodeFromPosition(nbMin + Eigen::Vector3f::Constant(voxelSize * 0.1f), gridOrigin, blockSizePerAxis);
								localKeys.push_back({ nKey, nbMin });
							}
						}
					}
				}

				if (!localKeys.empty())
				{
					std::lock_guard<std::mutex> lock(vecMutex);
					keysToAllocate.insert(keysToAllocate.end(), localKeys.begin(), localKeys.end());
				}
			});

		std::sort(std::execution::par, keysToAllocate.begin(), keysToAllocate.end(),
			[](const KeyPair& a, const KeyPair& b) { return a.first < b.first; });

		auto last = std::unique(std::execution::par, keysToAllocate.begin(), keysToAllocate.end(),
			[](const KeyPair& a, const KeyPair& b) { return a.first == b.first; });

		keysToAllocate.erase(last, keysToAllocate.end());

		for (const auto& kv : keysToAllocate)
		{
			if (dataBlocks.find(kv.first) == dataBlocks.end())
			{
				dataBlocks[kv.first] = std::make_unique<DataBlock>();
				dataBlocks[kv.first]->Initialize();
				dataBlocks[kv.first]->blockMin = kv.second;
			}
		}
		TE(Pass1_Alloc);
	}

	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i)
		{
			auto p = points[i];
			Eigen::Vector3f n = (normals.empty()) ? Eigen::Vector3f(0.0f, 1.0f, 0.0f) : normals[i];
			Eigen::Vector3f c = (colors.empty()) ? Eigen::Vector3f(1.0f, 1.0f, 1.0f) : colors[i].head<3>();
			
			if (c.x() > 1.0f) c /= 255.0f;

			Eigen::Vector3f vecFromOrigin = p - gridOrigin;
			int centerGx = (int)std::floor(vecFromOrigin.x() / voxelSize);
			int centerGy = (int)std::floor(vecFromOrigin.y() / voxelSize);
			int centerGz = (int)std::floor(vecFromOrigin.z() / voxelSize);

			DataBlockKey lastKey = (DataBlockKey)-1;
			DataBlock* cachedBlock = nullptr;

			for (int dz = -1; dz <= 1; ++dz)
			{
				for (int dy = -1; dy <= 1; ++dy)
				{
					for (int dx = -1; dx <= 1; ++dx)
					{
						int gx = centerGx + dx;
						int gy = centerGy + dy;
						int gz = centerGz + dz;

						Eigen::Vector3f voxelCenter = gridOrigin + Eigen::Vector3f((gx + 0.5f) * voxelSize, (gy + 0.5f) * voxelSize, (gz + 0.5f) * voxelSize);
						float dist = (p - voxelCenter).norm();
						if (dist > truncDist) continue;

						int bx = (int)floor((float)gx / SpatialPartitioningConfiguration::voxelsPerBlockAxis);
						int by = (int)floor((float)gy / SpatialPartitioningConfiguration::voxelsPerBlockAxis);
						int bz = (int)floor((float)gz / SpatialPartitioningConfiguration::voxelsPerBlockAxis);

						float currBlockSize = blockSizePerAxis;
						Eigen::Vector3f blockMin = gridOrigin + Eigen::Vector3f(bx * currBlockSize, by * currBlockSize, bz * currBlockSize);
						auto key = Morton3D::EncodeFromPosition(blockMin + Eigen::Vector3f::Constant(voxelSize * 0.1f), gridOrigin, currBlockSize);

						DataBlock* targetBlock = nullptr;
						if (key == lastKey && cachedBlock)
						{
							targetBlock = cachedBlock;
						}
						else
						{
							auto it = dataBlocks.find(key);
							if (it != dataBlocks.end())
							{
								targetBlock = it->second.get();
								lastKey = key; cachedBlock = targetBlock;
							}
						}

						if (targetBlock)
						{
							int lx = gx % SpatialPartitioningConfiguration::voxelsPerBlockAxis; if (lx < 0) lx += SpatialPartitioningConfiguration::voxelsPerBlockAxis;
							int ly = gy % SpatialPartitioningConfiguration::voxelsPerBlockAxis; if (ly < 0) ly += SpatialPartitioningConfiguration::voxelsPerBlockAxis;
							int lz = gz % SpatialPartitioningConfiguration::voxelsPerBlockAxis; if (lz < 0) lz += SpatialPartitioningConfiguration::voxelsPerBlockAxis;

							float weight = 1.0f - (dist / truncDist);
							float sdf = std::clamp((voxelCenter - p).dot(n), -truncDist, truncDist);

							std::lock_guard<std::mutex> lock(targetBlock->blockMutex);
							SparseDataBlockVoxel& voxel = targetBlock->voxels[lz * SpatialPartitioningConfiguration::voxelsPerBlockAxis * SpatialPartitioningConfiguration::voxelsPerBlockAxis + ly * SpatialPartitioningConfiguration::voxelsPerBlockAxis + lx];

							if (voxel.weight <= 0.0001f)
							{
								voxel.signedDistance = sdf;
								voxel.color = c;
								voxel.normal = n;
								voxel.weight = weight;
								voxel.valid = true;
								voxel.divergence = 0.0f;
							}
							else
							{
								float newW = voxel.weight + weight;

								Eigen::Vector3f currentDir = voxel.normal.normalized();
								float dotVal = std::clamp(currentDir.dot(n), -1.0f, 1.0f);
								float newDiv = 1.0f - dotVal;

								voxel.divergence = (voxel.divergence * voxel.weight + newDiv * weight) / newW;

								voxel.signedDistance = (voxel.signedDistance * voxel.weight + sdf * weight) / newW;
								voxel.color = (voxel.color * voxel.weight + c * weight) / newW;
								voxel.normal = (voxel.normal * voxel.weight + n * weight) / newW;
								voxel.weight = newW;
							}
						}
					}
				}
			}
		});
	TE(Occupy);
}

void SparseDataBlock::FromPointsData(const std::vector<Eigen::Vector3f>& points,
	const std::vector<Eigen::Vector3f>& normals,
	const std::vector<Eigen::Vector3f>& colors,
	const Eigen::Vector3f& aabbMin)
{
	blockSizePerAxis = voxelSize * SpatialPartitioningConfiguration::voxelsPerBlockAxis;

	gridOrigin.x() = std::floor(aabbMin.x() / blockSizePerAxis) * blockSizePerAxis;
	gridOrigin.y() = std::floor(aabbMin.y() / blockSizePerAxis) * blockSizePerAxis;
	gridOrigin.z() = std::floor(aabbMin.z() / blockSizePerAxis) * blockSizePerAxis;

	TS(Occupy);

	float truncDist = std::max(voxelSize * 4.0f, 0.15f);

	size_t numberOfPoints = points.size();
	std::vector<size_t> indices(numberOfPoints);
	std::iota(indices.begin(), indices.end(), 0);

	{
		TS(Pass1_Alloc);

		using KeyPair = std::pair<DataBlockKey, Eigen::Vector3f>;

		std::vector<KeyPair> keysToAllocate;
		keysToAllocate.reserve(numberOfPoints * 2);
		std::mutex vecMutex;

		std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i)
			{
				std::vector<KeyPair> localKeys;
				localKeys.reserve(8);

				Eigen::Vector3f p = points[i];
				Eigen::Vector3f vecFromOrigin = p - gridOrigin;

				int centerGx = (int)std::floor(vecFromOrigin.x() / voxelSize);
				int centerGy = (int)std::floor(vecFromOrigin.y() / voxelSize);
				int centerGz = (int)std::floor(vecFromOrigin.z() / voxelSize);

				int bx = centerGx / SpatialPartitioningConfiguration::voxelsPerBlockAxis;
				int by = centerGy / SpatialPartitioningConfiguration::voxelsPerBlockAxis;
				int bz = centerGz / SpatialPartitioningConfiguration::voxelsPerBlockAxis;

				Eigen::Vector3f blockMin = gridOrigin + Eigen::Vector3f((float)bx * blockSizePerAxis, (float)by * blockSizePerAxis, (float)bz * blockSizePerAxis);
				auto key = Morton3D::EncodeFromPosition(blockMin + Eigen::Vector3f::Constant(voxelSize * 0.1f), gridOrigin, blockSizePerAxis);
				localKeys.push_back({ key, blockMin });

				Eigen::Vector3f localP = p - blockMin;

				float margin = truncDist + voxelSize * 0.5f;

				bool nearX_Neg = localP.x() < margin;
				bool nearX_Pos = localP.x() > blockSizePerAxis - margin;
				bool nearY_Neg = localP.y() < margin;
				bool nearY_Pos = localP.y() > blockSizePerAxis - margin;
				bool nearZ_Neg = localP.z() < margin;
				bool nearZ_Pos = localP.z() > blockSizePerAxis - margin;

				if (nearX_Neg || nearX_Pos || nearY_Neg || nearY_Pos || nearZ_Neg || nearZ_Pos)
				{
					int dx_min = nearX_Neg ? -1 : 0;
					int dx_max = nearX_Pos ? 1 : 0;
					int dy_min = nearY_Neg ? -1 : 0;
					int dy_max = nearY_Pos ? 1 : 0;
					int dz_min = nearZ_Neg ? -1 : 0;
					int dz_max = nearZ_Pos ? 1 : 0;

					for (int dz = dz_min; dz <= dz_max; ++dz)
					{
						for (int dy = dy_min; dy <= dy_max; ++dy)
						{
							for (int dx = dx_min; dx <= dx_max; ++dx)
							{
								if (dx == 0 && dy == 0 && dz == 0) continue;

								Eigen::Vector3f nbMin = gridOrigin + Eigen::Vector3f((float)(bx + dx) * blockSizePerAxis, (float)(by + dy) * blockSizePerAxis, (float)(bz + dz) * blockSizePerAxis);
								auto nKey = Morton3D::EncodeFromPosition(nbMin + Eigen::Vector3f::Constant(voxelSize * 0.1f), gridOrigin, blockSizePerAxis);
								localKeys.push_back({ nKey, nbMin });
							}
						}
					}
				}

				if (!localKeys.empty())
				{
					std::lock_guard<std::mutex> lock(vecMutex);
					keysToAllocate.insert(keysToAllocate.end(), localKeys.begin(), localKeys.end());
				}
			});

		std::sort(std::execution::par, keysToAllocate.begin(), keysToAllocate.end(),
			[](const KeyPair& a, const KeyPair& b) { return a.first < b.first; });

		auto last = std::unique(std::execution::par, keysToAllocate.begin(), keysToAllocate.end(),
			[](const KeyPair& a, const KeyPair& b) { return a.first == b.first; });

		keysToAllocate.erase(last, keysToAllocate.end());

		for (const auto& kv : keysToAllocate)
		{
			if (dataBlocks.find(kv.first) == dataBlocks.end())
			{
				dataBlocks[kv.first] = std::make_unique<DataBlock>();
				dataBlocks[kv.first]->Initialize();
				dataBlocks[kv.first]->blockMin = kv.second;
			}
		}
		TE(Pass1_Alloc);
	}

	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i)
		{
			auto p = points[i];
			Eigen::Vector3f n = (normals.empty()) ? Eigen::Vector3f(0, 1, 0) : normals[i];
			Eigen::Vector3f c = (colors.empty()) ? Eigen::Vector3f(1, 1, 1) : colors[i];

			if (c.x() > 1.0f) c /= 255.0f;

			Eigen::Vector3f vecFromOrigin = p - gridOrigin;
			int centerGx = (int)std::floor(vecFromOrigin.x() / voxelSize);
			int centerGy = (int)std::floor(vecFromOrigin.y() / voxelSize);
			int centerGz = (int)std::floor(vecFromOrigin.z() / voxelSize);

			DataBlockKey lastKey = (DataBlockKey)-1;
			DataBlock* cachedBlock = nullptr;

			for (int dz = -1; dz <= 1; ++dz)
			{
				for (int dy = -1; dy <= 1; ++dy)
				{
					for (int dx = -1; dx <= 1; ++dx)
					{
						int gx = centerGx + dx;
						int gy = centerGy + dy;
						int gz = centerGz + dz;

						Eigen::Vector3f voxelCenter = gridOrigin + Eigen::Vector3f((gx + 0.5f) * voxelSize, (gy + 0.5f) * voxelSize, (gz + 0.5f) * voxelSize);
						float dist = (p - voxelCenter).norm();
						if (dist > truncDist) continue;

						int bx = (int)floor((float)gx / SpatialPartitioningConfiguration::voxelsPerBlockAxis);
						int by = (int)floor((float)gy / SpatialPartitioningConfiguration::voxelsPerBlockAxis);
						int bz = (int)floor((float)gz / SpatialPartitioningConfiguration::voxelsPerBlockAxis);

						float currBlockSize = blockSizePerAxis;
						Eigen::Vector3f blockMin = gridOrigin + Eigen::Vector3f(bx * currBlockSize, by * currBlockSize, bz * currBlockSize);
						auto key = Morton3D::EncodeFromPosition(blockMin + Eigen::Vector3f::Constant(voxelSize * 0.1f), gridOrigin, currBlockSize);

						DataBlock* targetBlock = nullptr;
						if (key == lastKey && cachedBlock)
						{
							targetBlock = cachedBlock;
						}
						else
						{
							auto it = dataBlocks.find(key);
							if (it != dataBlocks.end())
							{
								targetBlock = it->second.get();
								lastKey = key; cachedBlock = targetBlock;
							}
						}

						if (targetBlock)
						{
							int lx = gx % SpatialPartitioningConfiguration::voxelsPerBlockAxis; if (lx < 0) lx += SpatialPartitioningConfiguration::voxelsPerBlockAxis;
							int ly = gy % SpatialPartitioningConfiguration::voxelsPerBlockAxis; if (ly < 0) ly += SpatialPartitioningConfiguration::voxelsPerBlockAxis;
							int lz = gz % SpatialPartitioningConfiguration::voxelsPerBlockAxis; if (lz < 0) lz += SpatialPartitioningConfiguration::voxelsPerBlockAxis;

							float weight = 1.0f - (dist / truncDist);
							float sdf = std::clamp((voxelCenter - p).dot(n), -truncDist, truncDist);

							std::lock_guard<std::mutex> lock(targetBlock->blockMutex);
							SparseDataBlockVoxel& voxel = targetBlock->voxels[lz * SpatialPartitioningConfiguration::voxelsPerBlockAxis * SpatialPartitioningConfiguration::voxelsPerBlockAxis + ly * SpatialPartitioningConfiguration::voxelsPerBlockAxis + lx];

							if (voxel.weight <= 0.0001f)
							{
								voxel.signedDistance = sdf;
								voxel.color = c;
								voxel.normal = n;
								voxel.weight = weight;
								voxel.valid = true;
								voxel.divergence = 0.0f;
							}
							else
							{
								float newW = voxel.weight + weight;

								Eigen::Vector3f currentDir = voxel.normal.normalized();
								float dotVal = std::clamp(currentDir.dot(n), -1.0f, 1.0f);
								float newDiv = 1.0f - dotVal;

								voxel.divergence = (voxel.divergence * voxel.weight + newDiv * weight) / newW;

								voxel.signedDistance = (voxel.signedDistance * voxel.weight + sdf * weight) / newW;
								voxel.color = (voxel.color * voxel.weight + c * weight) / newW;
								voxel.normal = (voxel.normal * voxel.weight + n * weight) / newW;
								voxel.weight = newW;
							}
						}
					}
				}
			}
		});
	TE(Occupy);
}

void SparseDataBlock::Visualize()
{
	for (const auto& pair : dataBlocks)
	{
		const auto& block = pair.second;

		// Block ¿µ¿ª °è»ê (Eigen)
		Eigen::Vector3f blockMax = block->blockMin + Eigen::Vector3f::Constant(blockSizePerAxis);

		// Eigen Å¸ÀÔÀ» ±×´ë·Î Àü´Þ
		VD::AddWiredBox("Blocks", { block->blockMin, blockMax }, Color::yellow());

		for (int z = 0; z < SpatialPartitioningConfiguration::voxelsPerBlockAxis; ++z)
		{
			for (int y = 0; y < SpatialPartitioningConfiguration::voxelsPerBlockAxis; ++y)
			{
				for (int x = 0; x < SpatialPartitioningConfiguration::voxelsPerBlockAxis; ++x)
				{
					int index = z * SpatialPartitioningConfiguration::voxelsPerBlockAxis * SpatialPartitioningConfiguration::voxelsPerBlockAxis +
						y * SpatialPartitioningConfiguration::voxelsPerBlockAxis +
						x;

					const SparseDataBlockVoxel& voxel = block->voxels[index];
					if (voxel.valid)
					{
						// Voxel ¿µ¿ª °è»ê (Eigen)
						// °³º° °ö¼À ´ë½Å º¤ÅÍ ½ºÄ®¶ó °ö¼À È°¿ë
						Eigen::Vector3f offset((float)x, (float)y, (float)z);
						Eigen::Vector3f vMin = block->blockMin + (offset * voxelSize);
						Eigen::Vector3f vMax = vMin + Eigen::Vector3f::Constant(voxelSize);

						VD::AddWiredBox("Voxels", { vMin, vMax }, Color::red());
					}
				}
			}
		}
	}
}
