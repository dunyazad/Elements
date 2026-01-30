#include "Apps.h"

class AppVoxelDataBaseMemoryUsageCheck : public App
{
	public:
		virtual void Execute() override
		{
			auto start_time = std::chrono::high_resolution_clock::now();

			auto [initial_used, total_gpu] = CheckDeviceMemory("초기 상태");

			VVV::VoxelDataBase voxel_db;

			uint32_t max_blocks = 65536;

			size_t block_bytes = sizeof(VVV::VoxelBlock) * (size_t)max_blocks;
			size_t hash_bytes = sizeof(uint64_t) * (size_t)max_blocks;
			size_t theory_total = block_bytes + hash_bytes + sizeof(uint32_t);

			printf("\n>>> [메모리 분석: 할당 예측]\n");
			printf("    - 설정 블록 수       : %u 개\n", max_blocks);
			printf("    - 예상 메모리 점유   : %.4f GB\n", theory_total / (1024.0 * 1024.0 * 1024.0));

			VVV_Allocate(voxel_db, max_blocks);
			auto [after_alloc_used, ignore1] = CheckDeviceMemory("할당 완료");

			PLYFormat ply;
			if (!ply.Deserialize("D:\\Resources\\Debug\\3D\\VoxelValues_Unlock.ply"))
			{
				printf("!!! PLY 파일 로드 실패\n");
				VVV_Free(voxel_db);
				return;
			}

			size_t n_points = ply.GetPoints().size();
			std::vector<VVV::Vector3f> points(n_points);
			std::vector<VVV::Vector3b> colors(n_points);

			for (size_t i = 0; i < n_points; i++)
			{
				auto& p = ply.GetPoints()[i];
				points[i] = { p.x(), p.y(), p.z() };

				if (!ply.GetColors().empty())
				{
					auto& c = ply.GetColors()[i];
					colors[i].x = static_cast<uint8_t>(c.x() * 255.0f);
					colors[i].y = static_cast<uint8_t>(c.y() * 255.0f);
					colors[i].z = static_cast<uint8_t>(c.z() * 255.0f);
				}
				else
				{
					colors[i] = { 255, 255, 255 };
				}
			}

			float b_size = 0.8f;

			printf("\n>>> [GPU 연산] 복셀 데이터 생성 중...\n");
			TS(VVV_UpdateVoxelFromPoints);
			VVV_UpdateVoxelFromPoints(voxel_db, points.data(), colors.data(), (uint32_t)n_points, b_size, 1);
			cudaDeviceSynchronize();
			TE(VVV_UpdateVoxelFromPoints);

			CheckDeviceMemory("업데이트 완료");

			uint32_t max_out = 50000000;
			std::vector<VVV::ExtractedVoxel> host_out(max_out);
			uint32_t final_cnt = VVV_ExtractActiveVoxelsToHost(voxel_db, b_size, host_out.data(), max_out);

			if (final_cnt > 0)
			{
				// 복셀 한 변의 길이 (Full Size)
				float v_draw = (b_size / 8.0f) * 0.9f;

				uint32_t limit = (final_cnt > max_out) ? max_out : final_cnt;

				for (uint32_t i = 0; i < limit; i++)
				{
					//if (host_out[i].weight >= 1.0f)
					{
						Eigen::Vector3f center(host_out[i].position.x, host_out[i].position.y, host_out[i].position.z);
						Eigen::Vector4f col(host_out[i].color[0] / 255.f, host_out[i].color[1] / 255.f, host_out[i].color[2] / 255.f, 1.f);

						//if (0 < center.x() || 0 < center.y() || 0 < center.z())
						//	continue;
						VD::AddWiredBox("Voxels", center, Eigen::Vector3f(v_draw, v_draw, v_draw), col);
					}
				}

				std::vector<uint64_t> host_hash_table(max_blocks);
				cudaMemcpy(host_hash_table.data(), voxel_db.d_hashTable, sizeof(uint64_t) * max_blocks, cudaMemcpyDeviceToHost);

				for (uint32_t i = 0; i < max_blocks; ++i)
				{
					uint64_t m_key = host_hash_table[i];
					if (m_key != 0 && m_key != 0xFFFFFFFFFFFFFFFFULL)
					{
						VVV::Morton64 morton(m_key);
						VVV::Vector3f b_pos = morton.ToPosition(b_size);
						Eigen::Vector3f block_center(b_pos.x, b_pos.y, b_pos.z);

						//if (0 < block_center.x() || 0 < block_center.y() || 0 < block_center.z())
						//	continue;
						VD::AddWiredBox("LDE_SparseDataBlocks", block_center, Eigen::Vector3f(b_size, b_size, b_size), Eigen::Vector4f(0, 1, 0, 0.2f));
					}
				}
			}

			uint32_t active_blocks_count = 0;
			cudaMemcpy(&active_blocks_count, voxel_db.d_blockCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);

			printf("\n>>> [최종 리포트]\n");
			printf("    - 추출된 복셀 수     : %u 개\n", final_cnt);
			printf("    - 해시 적재율        : %.2f%% (%u / %u)\n",
				(double)active_blocks_count / max_blocks * 100.0, active_blocks_count, max_blocks);

			VVV_Free(voxel_db);
			auto [final_used, ignore3] = CheckDeviceMemory("해제 완료");

			printf("\n>>> [메모리 점검]\n");
			printf("    - 잔류 누수량        : %.4f MB\n", (final_used - initial_used) / (1024.0 * 1024.0));
			printf("    - 총 소요 시간       : %.4fs\n", std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count());
			printf("==========================================================\n");
		}

};

REGISTER_APP(AppVoxelDataBaseMemoryUsageCheck, "AppVoxelDataBaseMemoryUsageCheck");
