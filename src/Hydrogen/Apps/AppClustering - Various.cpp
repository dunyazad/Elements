#include "Apps.h"

#include <execution>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>
#include <unordered_map>

#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

namespace Eigen {
	template <typename Type, int Size>
	using Vector = Matrix<Type, Size, 1>;
	using Vector3b = Vector<unsigned char, 3>;
	using Vector3ui = Vector<unsigned int, 3>;
}

struct FlatClustering
{
	// -------------------------------------------------------
	// 1. 기본 데이터 구조 및 유틸리티
	// -------------------------------------------------------
	float voxelSize = 0.1f;

	std::vector<uint64_t> blockKeys;
	std::vector<Eigen::Vector3f> blockMinCorners;
	std::vector<unsigned int> blockOffsets;
	std::vector<unsigned int> blockCounts;
	std::vector<unsigned int> sortedIndices;

	struct KeyIndex {
		uint64_t key;
		unsigned int index;
	};
	std::vector<KeyIndex> proxy;

	inline uint64_t GetBlockKey(const Eigen::Vector3f& point, float invVoxelSize) const
	{
		int64_t xi = static_cast<int64_t>(std::floor(point.x() * invVoxelSize));
		int64_t yi = static_cast<int64_t>(std::floor(point.y() * invVoxelSize));
		int64_t zi = static_cast<int64_t>(std::floor(point.z() * invVoxelSize));

		return ((uint64_t)(xi & 0x1FFFFF) << 42) |
			((uint64_t)(yi & 0x1FFFFF) << 21) |
			((uint64_t)(zi & 0x1FFFFF));
	}

	inline uint64_t GetKeyFromIndices(int64_t xi, int64_t yi, int64_t zi) const
	{
		return ((uint64_t)(xi & 0x1FFFFF) << 42) |
			((uint64_t)(yi & 0x1FFFFF) << 21) |
			((uint64_t)(zi & 0x1FFFFF));
	}

	// -------------------------------------------------------
	// 2. Build: 공간 해싱 및 정렬
	// -------------------------------------------------------
	void Build(const std::vector<Eigen::Vector3f>& points, float blockSize = 0.3f)
	{
		this->voxelSize = blockSize;
		const size_t pointCount = points.size();
		if (pointCount == 0)
		{
			blockKeys.clear();
			return;
		}

		const float invVoxelSize = 1.0f / blockSize;

		if (proxy.size() != pointCount) proxy.resize(pointCount);
		if (sortedIndices.size() != pointCount) sortedIndices.resize(pointCount);

		const Eigen::Vector3f* pPoints = points.data();
		KeyIndex* pProxy = proxy.data();

		// 병렬 키 생성
		std::for_each(std::execution::par, pProxy, pProxy + pointCount, [this, pPoints, pProxy, invVoxelSize](KeyIndex& item) {
			size_t i = &item - pProxy;
			item.key = GetBlockKey(pPoints[i], invVoxelSize);
			item.index = static_cast<unsigned int>(i);
			});

		// 병렬 정렬
		std::sort(std::execution::par, proxy.begin(), proxy.end(), [](const KeyIndex& a, const KeyIndex& b) {
			return a.key < b.key;
			});

		blockKeys.clear();
		blockMinCorners.clear();
		blockOffsets.clear();
		blockCounts.clear();

		if (pointCount > 0)
		{
			uint64_t currentKey = proxy[0].key;
			unsigned int currentOffset = 0;

			for (size_t i = 0; i < pointCount; ++i)
			{
				sortedIndices[i] = proxy[i].index;

				if (proxy[i].key != currentKey)
				{
					unsigned int count = static_cast<unsigned int>(i) - currentOffset;

					blockKeys.push_back(currentKey);
					blockOffsets.push_back(currentOffset);
					blockCounts.push_back(count);

					int64_t xi = (int64_t)((currentKey >> 42) & 0x1FFFFF); if (xi & 0x100000) xi |= ~0x1FFFFF;
					int64_t yi = (int64_t)((currentKey >> 21) & 0x1FFFFF); if (yi & 0x100000) yi |= ~0x1FFFFF;
					int64_t zi = (int64_t)(currentKey & 0x1FFFFF); if (zi & 0x100000) zi |= ~0x1FFFFF;

					blockMinCorners.emplace_back(
						static_cast<float>(xi) * blockSize,
						static_cast<float>(yi) * blockSize,
						static_cast<float>(zi) * blockSize);

					currentKey = proxy[i].key;
					currentOffset = static_cast<unsigned int>(i);
				}
			}
			// 마지막 블록 처리
			blockKeys.push_back(currentKey);
			blockOffsets.push_back(currentOffset);
			blockCounts.push_back(static_cast<unsigned int>(pointCount) - currentOffset);
		}
	}

	// -------------------------------------------------------
	// 3. Robust Surface (Point-wise Weighted MLS)
	// -------------------------------------------------------
	struct RobustParams {
		float searchRadius = 0.2f;      // 이웃 탐색 반경
		float gaussSigma = 0.1f;        // 가중치 감소 폭
		int iterCount = 2;              // 반복 횟수
		bool recomputeNormals = true;   // 법선 재계산 여부
	};

	void ComputeRobustSurface(std::vector<Eigen::Vector3f>& points, std::vector<Eigen::Vector3f>& normals, RobustParams params)
	{
		if (blockKeys.empty() || points.empty()) return;

		// 블록 인덱싱 맵핑
		std::unordered_map<uint64_t, size_t> keyToBlockIdx;
		keyToBlockIdx.reserve(blockKeys.size());
		for (size_t i = 0; i < blockKeys.size(); ++i) {
			keyToBlockIdx[blockKeys[i]] = i;
		}

		const float sqRadius = params.searchRadius * params.searchRadius;
		const float invSigmaSq = 1.0f / (params.gaussSigma * params.gaussSigma);
		const float invVoxelSize = 1.0f / voxelSize;

		// 반복 적용 (Iterative MLS)
		for (int iter = 0; iter < params.iterCount; ++iter)
		{
			std::vector<Eigen::Vector3f> newPoints = points;
			std::vector<Eigen::Vector3f> newNormals = normals;

			std::for_each(std::execution::par, sortedIndices.begin(), sortedIndices.end(), [&](unsigned int pIdx)
				{
					const Eigen::Vector3f& currentPos = points[pIdx];

					// 1. Grid 기반 이웃 탐색
					uint64_t key = GetBlockKey(currentPos, invVoxelSize);
					int64_t xi = (int64_t)((key >> 42) & 0x1FFFFF); if (xi & 0x100000) xi |= ~0x1FFFFF;
					int64_t yi = (int64_t)((key >> 21) & 0x1FFFFF); if (yi & 0x100000) yi |= ~0x1FFFFF;
					int64_t zi = (int64_t)(key & 0x1FFFFF);         if (zi & 0x100000) zi |= ~0x1FFFFF;

					std::vector<unsigned int> neighbors;
					neighbors.reserve(64);

					for (int64_t dz = -1; dz <= 1; ++dz) {
						for (int64_t dy = -1; dy <= 1; ++dy) {
							for (int64_t dx = -1; dx <= 1; ++dx) {
								uint64_t nKey = GetKeyFromIndices(xi + dx, yi + dy, zi + dz);
								auto it = keyToBlockIdx.find(nKey);
								if (it != keyToBlockIdx.end()) {
									size_t bIdx = it->second;
									unsigned int start = blockOffsets[bIdx];
									unsigned int count = blockCounts[bIdx];
									for (unsigned int k = 0; k < count; ++k) {
										unsigned int nIdx = sortedIndices[start + k];
										if ((points[nIdx] - currentPos).squaredNorm() < sqRadius) {
											neighbors.push_back(nIdx);
										}
									}
								}
							}
						}
					}

					if (neighbors.size() < 5) return;

					// 2. 가중치 PCA
					Eigen::Vector3f weightedCentroid = Eigen::Vector3f::Zero();
					float totalWeight = 0.0f;
					std::vector<float> weights(neighbors.size());

					for (size_t i = 0; i < neighbors.size(); ++i) {
						float distSq = (points[neighbors[i]] - currentPos).squaredNorm();
						float w = std::exp(-distSq * invSigmaSq);
						weights[i] = w;
						weightedCentroid += points[neighbors[i]] * w;
						totalWeight += w;
					}
					if (totalWeight < 1e-6f) return;
					weightedCentroid /= totalWeight;

					Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
					for (size_t i = 0; i < neighbors.size(); ++i) {
						Eigen::Vector3f diff = points[neighbors[i]] - weightedCentroid;
						covariance += weights[i] * (diff * diff.transpose());
					}

					Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
					Eigen::Vector3f planeNormal = solver.eigenvectors().col(0);

					// 법선 방향 일관성 (Flip check)
					if (normals[pIdx].dot(planeNormal) < 0) {
						planeNormal = -planeNormal;
					}

					// 3. 투영 및 법선 갱신
					float dist = planeNormal.dot(currentPos - weightedCentroid);
					newPoints[pIdx] = currentPos - (planeNormal * dist);

					if (params.recomputeNormals) {
						newNormals[pIdx] = planeNormal;
					}
				});

			points = newPoints;
			if (params.recomputeNormals) {
				normals = newNormals;
			}
		}
	}

	// -------------------------------------------------------
	// 4. PFOR: 아웃라이어 제거
	// -------------------------------------------------------
	std::vector<uint8_t> ComputePFOR(const std::vector<Eigen::Vector3f>& points, float distanceThreshold = 0.085f)
	{
		const size_t pointCount = points.size();
		std::vector<uint8_t> outlierMarking(pointCount, 0);
		if (blockKeys.empty()) return outlierMarking;

		std::unordered_map<uint64_t, size_t> keyToBlockIdx;
		for (size_t i = 0; i < blockKeys.size(); ++i) keyToBlockIdx[blockKeys[i]] = i;

		std::vector<size_t> blockIndices(blockKeys.size());
		std::iota(blockIndices.begin(), blockIndices.end(), 0);

		std::for_each(std::execution::par, blockIndices.begin(), blockIndices.end(), [&](size_t bIdx)
			{
				uint64_t key = blockKeys[bIdx];
				unsigned int offset = blockOffsets[bIdx];
				unsigned int count = blockCounts[bIdx];

				std::vector<unsigned int> neighborIndices;

				// 이웃 블록 수집
				int64_t xi = (int64_t)((key >> 42) & 0x1FFFFF); if (xi & 0x100000) xi |= ~0x1FFFFF;
				int64_t yi = (int64_t)((key >> 21) & 0x1FFFFF); if (yi & 0x100000) yi |= ~0x1FFFFF;
				int64_t zi = (int64_t)(key & 0x1FFFFF); if (zi & 0x100000) zi |= ~0x1FFFFF;

				for (int64_t dz = -1; dz <= 1; ++dz) {
					for (int64_t dy = -1; dy <= 1; ++dy) {
						for (int64_t dx = -1; dx <= 1; ++dx) {
							uint64_t nKey = GetKeyFromIndices(xi + dx, yi + dy, zi + dz);
							auto it = keyToBlockIdx.find(nKey);
							if (it != keyToBlockIdx.end()) {
								size_t nIdx = it->second;
								unsigned int nOff = blockOffsets[nIdx];
								unsigned int nCnt = blockCounts[nIdx];
								for (unsigned int p = 0; p < nCnt; ++p) neighborIndices.push_back(sortedIndices[nOff + p]);
							}
						}
					}
				}

				if (neighborIndices.size() < 3) {
					for (unsigned int p = 0; p < count; ++p) outlierMarking[sortedIndices[offset + p]] = 1;
					return;
				}

				// PCA 계산
				Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
				for (unsigned int idx : neighborIndices) centroid += points[idx];
				centroid /= static_cast<float>(neighborIndices.size());

				Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
				for (unsigned int idx : neighborIndices) {
					Eigen::Vector3f d = points[idx] - centroid;
					covariance += d * d.transpose();
				}

				Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
				Eigen::Vector3f planeNormal = solver.eigenvectors().col(0);

				// 거리 필터링
				for (unsigned int p = 0; p < count; ++p) {
					unsigned int pIdx = sortedIndices[offset + p];
					float dist = std::abs(planeNormal.dot(points[pIdx] - centroid));
					if (dist > distanceThreshold) outlierMarking[pIdx] = 1;
				}
			});

		return outlierMarking;
	}

	// -------------------------------------------------------
	// 5. ExtractClusters: Region Growing 클러스터링
	// -------------------------------------------------------
	std::vector<std::vector<unsigned int>> ExtractClustersWithPFOR(
		const std::vector<Eigen::Vector3f>& points,
		const std::vector<Eigen::Vector3f>& normals,
		const std::vector<uint8_t>& outlierMarking,
		float distThreshold = 0.1f,
		float normalThreshold = 0.9f)
	{
		std::vector<std::vector<unsigned int>> clusters;
		if (blockKeys.empty() || points.size() != outlierMarking.size() || points.size() != normals.size())
		{
			return clusters;
		}

		const float sqDistThreshold = distThreshold * distThreshold;
		std::unordered_map<uint64_t, size_t> keyToBlockIdx;
		for (size_t i = 0; i < blockKeys.size(); ++i) keyToBlockIdx[blockKeys[i]] = i;

		std::vector<bool> blockVisited(blockKeys.size(), false);

		for (size_t i = 0; i < blockKeys.size(); ++i)
		{
			if (blockVisited[i]) continue;

			unsigned int offset = blockOffsets[i];
			unsigned int count = blockCounts[i];
			bool hasValidPoint = false;

			// 유효 포인트 존재 여부 체크
			for (unsigned int p = 0; p < count; ++p) {
				if (outlierMarking[sortedIndices[offset + p]] == 0) {
					hasValidPoint = true;
					break;
				}
			}

			if (!hasValidPoint) {
				blockVisited[i] = true;
				continue;
			}

			std::vector<unsigned int> currentCluster;
			std::vector<size_t> blockQueue;

			blockQueue.push_back(i);
			blockVisited[i] = true;

			size_t head = 0;
			while (head < blockQueue.size())
			{
				size_t currBlockIdx = blockQueue[head++];
				uint64_t key = blockKeys[currBlockIdx];
				unsigned int curOffset = blockOffsets[currBlockIdx];
				unsigned int curCount = blockCounts[currBlockIdx];

				// 현재 블록 포인트 추가
				for (unsigned int p = 0; p < curCount; ++p) {
					unsigned int pIdx = sortedIndices[curOffset + p];
					if (outlierMarking[pIdx] == 0) {
						currentCluster.push_back(pIdx);
					}
				}

				// 이웃 블록 탐색
				int64_t xi = (int64_t)((key >> 42) & 0x1FFFFF); if (xi & 0x100000) xi |= ~0x1FFFFF;
				int64_t yi = (int64_t)((key >> 21) & 0x1FFFFF); if (yi & 0x100000) yi |= ~0x1FFFFF;
				int64_t zi = (int64_t)(key & 0x1FFFFF); if (zi & 0x100000) zi |= ~0x1FFFFF;

				for (int64_t dz = -1; dz <= 1; ++dz) {
					for (int64_t dy = -1; dy <= 1; ++dy) {
						for (int64_t dx = -1; dx <= 1; ++dx) {
							if (dx == 0 && dy == 0 && dz == 0) continue;

							uint64_t neighborKey = GetKeyFromIndices(xi + dx, yi + dy, zi + dz);
							auto it = keyToBlockIdx.find(neighborKey);

							if (it != keyToBlockIdx.end() && !blockVisited[it->second])
							{
								size_t neighborBlockIdx = it->second;
								unsigned int nOffset = blockOffsets[neighborBlockIdx];
								unsigned int nCount = blockCounts[neighborBlockIdx];
								bool isConnected = false;

								// 블록 간 연결성 체크
								for (unsigned int p1 = 0; p1 < curCount; ++p1) {
									unsigned int idx1 = sortedIndices[curOffset + p1];
									if (outlierMarking[idx1] == 1) continue;

									const Eigen::Vector3f& pt1 = points[idx1];
									const Eigen::Vector3f& nm1 = normals[idx1];

									for (unsigned int p2 = 0; p2 < nCount; ++p2) {
										unsigned int idx2 = sortedIndices[nOffset + p2];
										if (outlierMarking[idx2] == 1) continue;

										if ((pt1 - points[idx2]).squaredNorm() <= sqDistThreshold) {
											if (std::abs(nm1.dot(normals[idx2])) >= normalThreshold) {
												isConnected = true;
												break;
											}
										}
									}
									if (isConnected) break;
								}

								if (isConnected) {
									blockVisited[neighborBlockIdx] = true;
									blockQueue.push_back(neighborBlockIdx);
								}
							}
						}
					}
				}
			}

			if (!currentCluster.empty()) {
				clusters.push_back(std::move(currentCluster));
			}
		}
		return clusters;
	}
};

class AppClustering : public App
{
public:
	virtual void Execute() override
	{
		// 1. 데이터 로드
		PLYFormat ply;
		ply.Deserialize("D:\\Resources\\Debug\\3D\\BasePoints.ply");
		if (ply.GetPoints().empty()) return;

		TS(Total);

		static FlatClustering clustering;

		// 2. 공간 분할 (Build)
		// voxelSize 0.3f는 대략적인 Grid 크기입니다. 데이터 밀도에 맞춰 조정하세요.
		TS(Build);
		clustering.Build(ply.GetPoints(), 0.3f);
		TE(Build);

		// 3. Robust Surface Smoothing (Weighted MLS)
		// 기존 Block-wise MLS 대신 이 방식을 사용하여 경계면 아티팩트를 제거합니다.
		TS(RobustSmoothing);
		FlatClustering::RobustParams params;
		params.searchRadius = 0.2f;      // VoxelSize와 비슷하거나 약간 작게
		params.gaussSigma = 0.1f;        // searchRadius의 절반 정도
		params.iterCount = 2;            // 2회 반복이면 충분히 부드러워짐
		params.recomputeNormals = true;  // 중요: 평탄화된 면에 맞춰 법선 갱신

		clustering.ComputeRobustSurface(ply.GetPoints(), ply.GetNormals(), params);
		TE(RobustSmoothing);

		//// 중간 확인용 시각화 (스무딩 결과)
		//VD::AddSphereBatch(
		//	"SmoothedPointCloud",
		//	ply.GetPoints(),
		//	ply.GetNormals(),
		//	0.05f,
		//	ply.GetColors());

		//ply.Serialize("D:\\Resources\\Debug\\3D\\BasePoints_Smoothed.ply");

		//return;

		// 4. 아웃라이어 제거 (PFOR)
		TS(PFOR_Compute);
		// 평면 추정 후 거리가 0.1f 이상 떨어진 점들을 아웃라이어로 마킹
		auto outliers = clustering.ComputePFOR(ply.GetPoints(), 0.05f);
		TE(PFOR_Compute);

		// 5. 클러스터링 (Extract)
		TS(Extract);
		// distThreshold: 0.1f (연결 거리)
		// normalThreshold: 0.9f (약 25도 이내의 법선 차이만 허용)
		auto clusters = clustering.ExtractClustersWithPFOR(
			ply.GetPoints(),
			ply.GetNormals(),
			outliers,
			0.1f,
			0.9f);
		TE(Extract);

		TE(Total);

		// 6. 결과 시각화
		// 각 클러스터를 랜덤 색상으로 표시
		auto colors = Color::GetContrastingColorsWithoutBWRGB(128); // 128개의 랜덤 색상

		for (size_t i = 0; i < clusters.size(); ++i)
		{
			// 너무 작은 노이즈 클러스터는 시각화 제외 (옵션)
			if (clusters[i].size() < 10) continue;

			Eigen::Vector4f randomColor = colors[i % colors.size()];
			std::vector<Eigen::Vector3f> clusterPoints;
			clusterPoints.reserve(clusters[i].size());

			for (auto idx : clusters[i]) {
				clusterPoints.push_back(ply.GetPoints()[idx]);
			}

			VD::AddSphereBatch("Cluster_" + std::to_string(i), clusterPoints, 0.05f, randomColor);
		}
	}
};

REGISTER_APP(AppClustering, "AppClustering");