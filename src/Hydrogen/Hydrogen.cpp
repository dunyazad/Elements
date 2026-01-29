#include "framework.h"
#include "Hydrogen.h"

#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include <algorithm>
#include <atomic>
#include <execution>
#include <filesystem>
#include <iostream>
#include <thread>
#include <vector>

#include <Helium/Serialization.hpp>
#include <Helium/DeviceInformation.h>

#include <ShellScalingApi.h>
#pragma comment(lib, "Shcore.lib")

#include <Monitor.h>

#include <Copper/CuVoxelStreaming.h>

#include <robin_hood/robin_hood.h>

#include <VVV/VVV.h>
#pragma comment(lib, "VVV.lib")

#include <Eigen/Core>
#include <Eigen/Dense>

#include <Helium/Serialization.hpp>

namespace Eigen {
	template <typename Type, int Size>
	using Vector = Matrix<Type, Size, 1>;

	using Vector3b = Vector<unsigned char, 3>;
	using Vector3ui = Vector<unsigned int, 3>;
}

#define MAX_LOADSTRING 100

HINSTANCE hInst;
WCHAR szTitle[MAX_LOADSTRING];
WCHAR szWindowClass[MAX_LOADSTRING];

std::atomic<bool> g_isRendering = true;
std::atomic<bool> g_resizeRequested = false;
std::atomic<int> g_resizeWidth = 0;
std::atomic<int> g_resizeHeight = 0;

std::thread g_renderingThread;

void BusinessLogic();

ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
	_In_opt_ HINSTANCE hPrevInstance,
	_In_ LPWSTR    lpCmdLine,
	_In_ int       nCmdShow)
{
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);

	SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);

	LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
	LoadStringW(hInstance, IDC_HYDROGEN, szWindowClass, MAX_LOADSTRING);
	MyRegisterClass(hInstance);

	if (!InitInstance(hInstance, nCmdShow))
	{
		return FALSE;
	}

	HWND hWnd = FindWindowW(szWindowClass, szTitle);

	std::string cpuId = CPUInformation::GetCpuID();
	std::string vendor = CPUInformation::GetVendorString();

	if ("000906A3BFEBFBFF" == cpuId)
	{
		AllocConsole();
		freopen("CONOUT$", "w", stdout);
		MaximizeConsoleWindowOnMonitor(3);
		MaximizeWindowOnMonitor(hWnd, 2);
	}
	else
	{
		AllocConsole();
		freopen("CONOUT$", "w", stdout);
		SetConsoleToHalfOfScreen(0, 1);
		MaximizeWindowOnMonitor(hWnd, 1);
	}

	std::cout << "CPU Vendor: " << vendor << std::endl;
	std::cout << "CPU ID (Signature): " << cpuId << std::endl;

	Cu_Initialize();

	g_renderingThread = std::thread([&]() {
		He_Initialize(hWnd, 0);
		He_InitializeScene3D();

		RECT rect;
		if (GetClientRect(hWnd, &rect))
		{
			int width = rect.right - rect.left;
			int height = rect.bottom - rect.top;
			if (width > 0 && height > 0)
			{
				He_Resize(width, height);
			}
		}

		{
			cudaFree(0);

			BusinessLogic();
		}

		{
			auto entity = Helium.CreateEntity("main");
			Helium.CreateEventCallback<KeyEvent>(entity, "3D", [](Entity e, const KeyEvent& event) {
				if (event.action == 1 && KeyCode::Plus == event.keyCode)
				{
					auto scale = VD::GetInstanceScale("LDE");
					VD::SetInstanceScale("LDE", scale * 1.1f);
				}
				else if (event.action == 1 && KeyCode::Minus == event.keyCode)
				{
					auto scale = VD::GetInstanceScale("LDE");
					VD::SetInstanceScale("LDE", scale * 0.9f);
				}
				else if (event.action == 0 && KeyCode::Tilde == event.keyCode)
				{
					VD::ToggleVisibility("LDE_SparseDataBlocks");
				}
				});
		}

		while (g_isRendering)
		{
			if (g_resizeRequested)
			{
				g_resizeRequested = false;
				if (g_resizeWidth > 0 && g_resizeHeight > 0)
				{
					He_Resize(g_resizeWidth, g_resizeHeight);
				}
			}
			He_Update(0.016f);
			He_Render();
		}
		He_Shutdown();
		});

	HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_HYDROGEN));
	MSG msg;
	while (GetMessage(&msg, nullptr, 0, 0))
	{
		if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}
	g_isRendering = false;
	if (g_renderingThread.joinable()) g_renderingThread.join();
	Cu_Shutdown();
	return (int)msg.wParam;
}

ATOM MyRegisterClass(HINSTANCE hInstance)
{
	WNDCLASSEXW wcex;
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;
	wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_HYDROGEN));
	wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_HYDROGEN);
	wcex.lpszClassName = szWindowClass;
	wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));
	return RegisterClassExW(&wcex);
}

BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
	hInst = hInstance;
	HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);
	if (!hWnd) return FALSE;
	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);
	return TRUE;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_KEYDOWN:
		if (VK_ESCAPE == wParam) PostMessage(hWnd, WM_CLOSE, 0, 0);
	case WM_KEYUP:
	case WM_SYSKEYDOWN:
	case WM_SYSKEYUP:
	case WM_MOUSEMOVE:
	case WM_LBUTTONDOWN:
	case WM_LBUTTONUP:
	case WM_RBUTTONDOWN:
	case WM_RBUTTONUP:
	case WM_MBUTTONDOWN:
	case WM_MBUTTONUP:
	case WM_MOUSEWHEEL:
		He_ProcessMessage(message, wParam, lParam);
		break;
	}
	switch (message)
	{
	case WM_COMMAND:
	{
		int wmId = LOWORD(wParam);
		switch (wmId)
		{
		case IDM_EXIT: DestroyWindow(hWnd); break;
		default: return DefWindowProc(hWnd, message, wParam, lParam);
		}
	}
	break;
	case WM_SIZE:
		g_resizeWidth = LOWORD(lParam);
		g_resizeHeight = HIWORD(lParam);
		g_resizeRequested = true;
		break;
	case WM_DESTROY: PostQuitMessage(0); break;
	default: return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	switch (message)
	{
	case WM_INITDIALOG: return (INT_PTR)TRUE;
	case WM_COMMAND:
		if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
		{
			EndDialog(hDlg, LOWORD(wParam));
			return (INT_PTR)TRUE;
		}
		break;
	}
	return (INT_PTR)FALSE;
}

namespace robin_hood
{
	template <>
	struct hash<VVV::Morton64>
	{
		size_t operator()(const VVV::Morton64& m) const noexcept
		{
			return static_cast<size_t>(m.code);
		}
	};
}

typedef struct CamInfo_
{
	float cfx;
	float cfy;
	float ccx;
	float ccy;
	int cx;
	int cy;
	int img_width;
	int img_height;
	double R[9];
	double T[3];
	Eigen::Vector3f dlpPos;
	Eigen::Vector3f camPos;
	Eigen::Matrix3f invMatTilt;
	Eigen::Matrix3f matTilt;

	Eigen::Matrix4f GetViewMatrix(const CamInfo_& info)
	{
		Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

		for (int i = 0; i < 3; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				view(i, j) = (float)info.R[i * 3 + j];
			}
			view(i, 3) = (float)info.T[i];
		}

		return view;
	}

	Eigen::Matrix4f GetProjectionMatrix(const CamInfo_& info, float n, float f)
	{
		Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();

		proj(0, 0) = 2.0f * info.cfx / info.img_width;
		proj(0, 2) = 1.0f - (2.0f * info.ccx / info.img_width);
		proj(1, 1) = 2.0f * info.cfy / info.img_height;
		proj(1, 2) = (2.0f * info.ccy / info.img_height) - 1.0f;
		proj(2, 2) = -(f + n) / (f - n);
		proj(2, 3) = -(2.0f * f * n) / (f - n);
		proj(3, 2) = -1.0f;

		return proj;
	}
} CamInfo_;

void BusinessLogic()
{
	printf("\n==========================================================\n");
	printf("[디버그] 박스 크기 교정 및 블록 최적화 모드 시작\n");
	printf("==========================================================\n");

	auto start_time = std::chrono::high_resolution_clock::now();

	// 1. 초기 상태 측정
	auto [initial_used, total_gpu] = CheckDeviceMemory("초기 상태");

	VVV::VoxelDataBase voxel_db;

	// 실제 사용량(3.4만) 대비 효율적인 64K 할당
	uint32_t max_blocks = 65536;

	size_t block_bytes = sizeof(VVV::VoxelBlock) * (size_t)max_blocks;
	size_t hash_bytes = sizeof(uint64_t) * (size_t)max_blocks;
	size_t theory_total = block_bytes + hash_bytes + sizeof(uint32_t);

	printf("\n>>> [메모리 분석: 할당 예측]\n");
	printf("    - 설정 블록 수       : %u 개\n", max_blocks);
	printf("    - 예상 메모리 점유   : %.4f GB\n", theory_total / (1024.0 * 1024.0 * 1024.0));

	// 2. GPU 메모리 할당
	VVV_Allocate(voxel_db, max_blocks);
	auto [after_alloc_used, ignore1] = CheckDeviceMemory("할당 완료");

	// 3. PLY 데이터 로드
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

	// 4. GPU 업데이트 (시간 측정 포함)
	printf("\n>>> [GPU 연산] 복셀 데이터 생성 중...\n");
	TS(VVV_UpdateVoxelFromPoints);
	VVV_UpdateVoxelFromPoints(voxel_db, points.data(), colors.data(), (uint32_t)n_points, b_size, 1);
	cudaDeviceSynchronize();
	TE(VVV_UpdateVoxelFromPoints);

	CheckDeviceMemory("업데이트 완료");

	// 5. 데이터 추출
	uint32_t max_out = 10000000;
	std::vector<VVV::ExtractedVoxel> host_out(max_out);
	uint32_t final_cnt = VVV_ExtractActiveVoxelsToHost(voxel_db, b_size, host_out.data(), max_out);

	// 6. 가시화 (Full Dimension 적용)
	if (final_cnt > 0)
	{
		// 복셀 한 변의 길이 (Full Size)
		float v_draw = (b_size / 8.0f) * 0.9f;

		uint32_t limit = (final_cnt > max_out) ? max_out : final_cnt;

		printf("\n>>> [가시화] 박스 크기 교정 적용 중...\n");

		// 6-1. 복셀 그리기
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

		// 6-2. 블록 경계 그리기
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

	// 7. 분석 리포트 및 정리
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
