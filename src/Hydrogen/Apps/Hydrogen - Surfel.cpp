#include "framework.h"
#include "Hydrogen.h"

#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include <algorithm>
#include <atomic>
#include <execution>
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

void BusinessLogic()
{
	std::vector<std::vector<Eigen::Vector3f>> loadedPositions;
	std::vector<std::vector<Eigen::Vector3f>> loadedNormals;

	size_t numberOfPatches = 1953;
	std::string filePath = "D:\\Resources\\CaseDataVoxelStreaming\\Patches\\";

	// 1. 데이터 로딩 (순차)
	for (size_t i = 0; i < numberOfPatches; i++)
	{
		std::stringstream ss;
		ss << std::setw(4) << std::setfill('0') << i;
		std::string patchFileName = filePath + "patch" + ss.str() + ".ply";
		std::cout << "Loading patch file: " << patchFileName << std::endl;
		PLYFormat ply;
		ply.Deserialize(patchFileName);

		loadedPositions.emplace_back(std::move(ply.GetPoints()));
		loadedNormals.emplace_back(std::move(ply.GetNormals()));
	}

	struct DownSampleVoxel
	{
		Eigen::Vector3f position_sum{ 0.f, 0.f, 0.f };
		Eigen::Vector3f normal_sum{ 0.f, 0.f, 0.f };
		size_t          count = 0;
	};

	// 각 패치별 독립 맵 (Lock-Free 용)
	std::vector<robin_hood::unordered_flat_map<VVV::Morton64, DownSampleVoxel>> patchLocalMaps(numberOfPatches);
	std::vector<size_t> patchIndices(numberOfPatches);
	std::iota(patchIndices.begin(), patchIndices.end(), 0);
	std::vector<double> patchTimes(numberOfPatches, 0.0);
	std::mutex logMutex;

	// 2. Accumulating (완전 병렬 + 실시간 로그)
	TS(Accumulating);
	std::for_each(std::execution::par, patchIndices.begin(), patchIndices.end(), [&](size_t i)
		{
			auto beginTime = std::chrono::high_resolution_clock::now();
			auto& localMap = patchLocalMaps[i];

			const auto& points = loadedPositions[i];
			const auto& normals = loadedNormals[i];
			localMap.reserve(points.size() / 4);

			for (size_t j = 0; j < points.size(); j++)
			{
				const auto& p = points[j];
				if (p.x() > 100000.f) continue;

				auto key = VVV::Morton64::FromPosition({ p.x(), p.y(), p.z() }, 0.05f);
				auto& v = localMap[key];
				v.position_sum += p;
				v.normal_sum += normals[j];
				v.count++;
			}

			auto endTime = std::chrono::high_resolution_clock::now();
			double duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime).count() / 1000.0;
			patchTimes[i] = duration;

			{
				std::lock_guard<std::mutex> lock(logMutex);
				std::cout << "Processed patch " << std::setw(4) << i
					<< " | Points: " << std::setw(6) << points.size()
					<< " | Time: " << std::fixed << std::setprecision(3) << duration << " ms" << std::endl;
			}
		});
	TE(Accumulating);

	// 3. Merging (Shard 단위 병렬 병합 + 실시간 로그)
	TS(Merging);
	constexpr size_t NUM_SHARDS = 128;
	struct FinalShard {
		robin_hood::unordered_flat_map<VVV::Morton64, DownSampleVoxel> map;
	};
	std::vector<FinalShard> finalShards(NUM_SHARDS);
	std::vector<size_t> shardIndices(NUM_SHARDS);
	std::iota(shardIndices.begin(), shardIndices.end(), 0);

	std::for_each(std::execution::par, shardIndices.begin(), shardIndices.end(), [&](size_t sIdx)
		{
			auto beginTime = std::chrono::high_resolution_clock::now();
			auto& targetShard = finalShards[sIdx].map;
			targetShard.reserve(200000);

			for (size_t pIdx = 0; pIdx < numberOfPatches; ++pIdx)
			{
				for (auto const& [key, v] : patchLocalMaps[pIdx])
				{
					if ((key.code & (NUM_SHARDS - 1)) == sIdx)
					{
						auto& dest = targetShard[key];
						dest.position_sum += v.position_sum;
						dest.normal_sum += v.normal_sum;
						dest.count += v.count;
					}
				}
			}

			auto endTime = std::chrono::high_resolution_clock::now();
			double duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime).count() / 1000.0;

			{
				std::lock_guard<std::mutex> lock(logMutex);
				std::cout << "Merged Shard " << std::setw(3) << sIdx
					<< " | Final Voxels: " << std::setw(7) << targetShard.size()
					<< " | Time: " << std::fixed << std::setprecision(3) << duration << " ms" << std::endl;
			}
		});
	TE(Merging);

	// Max Latency 분석 출력
	double maxTime = 0.0;
	size_t maxIdx = 0;
	for (size_t i = 0; i < patchTimes.size(); ++i)
	{
		if (patchTimes[i] > maxTime)
		{
			maxTime = patchTimes[i];
			maxIdx = i;
		}
	}
	std::cout << "\n--- Performance Summary ---" << std::endl;
	std::cout << "Max Latency (Accumulate): Patch " << maxIdx << " with " << maxTime << " ms" << std::endl;
	std::cout << "Max Latency Patch Point Count: " << loadedPositions[maxIdx].size() << std::endl;
	std::cout << "---------------------------\n" << std::endl;

	// 4. Drawing (순차 처리)
	TS(Drawing);
	size_t totalVoxels = 0;
	for (size_t sIdx = 0; sIdx < NUM_SHARDS; ++sIdx)
	{
		auto& shard = finalShards[sIdx];
		totalVoxels += shard.map.size();
		for (auto const& [key, v] : shard.map)
		{
			Eigen::Vector3f p = v.position_sum / static_cast<float>(v.count);
			Eigen::Vector3f n = v.normal_sum / static_cast<float>(v.count);
			VD::AddDisk("DownSampleVoxels", p, n, 0.025f, 16, Color::white(), false);
		}
		std::cout << "Drawing Shard " << sIdx << "..." << std::endl;
	}
	TE(Drawing);

	std::cout << "Total Downsampled Voxels: " << totalVoxels << std::endl;
}

void BusinessLogic_Sequential()
{
	struct DownSampleVoxel
	{
		Eigen::Vector3f position_sum{ 0.f, 0.f, 0.f };
		Eigen::Vector3f normal_sum{ 0.f, 0.f, 0.f };
		Eigen::Vector4f color_sum{ 0.f, 0.f, 0.f, 0.0f };
		size_t          count = 0;
	};

	robin_hood::unordered_flat_map<VVV::Morton64, size_t> downSampleVolume;
	std::vector<DownSampleVoxel> downSampleVoxels;
	downSampleVolume.reserve(20000000);
	downSampleVoxels.reserve(20000000);

	auto Accumulate = [&downSampleVoxels, &downSampleVolume](
		const VVV::Morton64& key,
		const Eigen::Vector3f& pos,
		const Eigen::Vector3f& normal,
		const Eigen::Vector4f& color)
		{
			auto it = downSampleVolume.find(key);

			if (it == downSampleVolume.end())
			{
				size_t idx = downSampleVoxels.size();
				downSampleVolume[key] = idx;

				DownSampleVoxel v;
				v.position_sum = pos;
				v.normal_sum = normal;
				v.color_sum = color;
				v.count = 1;

				downSampleVoxels.push_back(v);
			}
			else
			{
				auto& v = downSampleVoxels[it->second];
				v.position_sum += pos;
				v.normal_sum += normal;
				v.color_sum += color;
				v.count++;
			}
		};

	size_t numberOfPatches = 1953;
	std::string filePath = "D:\\Resources\\CaseDataVoxelStreaming\\Patches\\";
	for (size_t i = 0; i < numberOfPatches; i++)
	{
		std::stringstream ss;
		ss << std::setw(4) << std::setfill('0') << i;
		std::string patchFileName = filePath + "patch" + ss.str() + ".ply";
		std::cout << "Loading patch file: " << patchFileName << std::endl;
		PLYFormat ply;
		ply.Deserialize(patchFileName);

		TS(Accumulate);
		for (size_t i = 0; i < ply.GetPoints().size(); i++)
		{
			auto& p = ply.GetPoints()[i];
			if(p.x() > 100000.f || p.y() > 100000.f || p.z() > 100000.f)
			{
				continue;
			}
			auto& n = ply.GetNormals()[i];
			auto c = Color::white();

			Accumulate(
				VVV::Morton64::FromPosition({p.x(), p.y(), p.z()}, 0.05f),
				p,
				n,
				c);
		}
		TE(Accumulate);
	}

	TS(Drawing);
	for (auto& kvp : downSampleVolume)
	{
		auto& v = downSampleVoxels[kvp.second];
		Eigen::Vector3f p = v.position_sum / static_cast<float>(v.count);
		Eigen::Vector3f n = v.normal_sum / static_cast<float>(v.count);
		Eigen::Vector4f c = v.color_sum / static_cast<float>(v.count);

		VD::AddDisk("DownSampleVoxels", p, n, 0.025f, 16, c, false);
	}
	TE(Drawing);

	std::cout << "Downsample voxels: " << downSampleVoxels.size() << std::endl;
}

void BusinessLogic_Parallel()
{
	struct DownSampleVoxel {
		Eigen::Vector3f position_sum{ 0.f, 0.f, 0.f };
		Eigen::Vector3f normal_sum{ 0.f, 0.f, 0.f };
		Eigen::Vector4f color_sum{ 0.f, 0.f, 0.f, 0.0f };
		size_t count = 0;
	};

	// 1. Sharding 설정 (2의 거듭제곱이 성능에 유리)
	constexpr size_t NUM_SHARDS = 64;
	std::vector<robin_hood::unordered_flat_map<VVV::Morton64, size_t>> shardMaps(NUM_SHARDS);
	std::vector<std::vector<DownSampleVoxel>> shardVoxels(NUM_SHARDS);

	// 각 맵에 대해 reserve (전체 2000만개를 shard 수로 나눔)
	for (size_t i = 0; i < NUM_SHARDS; ++i) {
		shardMaps[i].reserve(400000);
		shardVoxels[i].reserve(400000);
	}

	// 각 shard에 안전하게 접근하기 위한 뮤텍스 (robin_hood가 스레드 세이프하지 않으므로 필요)
	std::vector<std::mutex> shardMutexes(NUM_SHARDS);

	size_t numberOfPatches = 1953;
	std::vector<size_t> patchIndices(numberOfPatches);
	std::iota(patchIndices.begin(), patchIndices.end(), 0);

	// 2. 패치 로딩 및 Accumulate 병렬화 (std::execution::par)
	TS(Accumulating);
	std::for_each(std::execution::par, patchIndices.begin(), patchIndices.end(), [&](size_t i) {
		std::stringstream ss;
		ss << std::setw(4) << std::setfill('0') << i;
		std::string patchFileName = "D:\\Resources\\CaseDataVoxelStreaming\\Patches\\patch" + ss.str() + ".ply";

		PLYFormat ply;
		ply.Deserialize(patchFileName);

		for (size_t j = 0; j < ply.GetPoints().size(); j++) {
			auto& p = ply.GetPoints()[j];
			if (p.x() > 100000.f || p.y() > 100000.f || p.z() > 100000.f) continue;

			auto& n = ply.GetNormals()[j];
			auto key = VVV::Morton64::FromPosition({ p.x(), p.y(), p.z() }, 0.05f);

			// 3. Morton 코드를 이용한 Shard 결정 (매우 중요)
			size_t shardIdx = key.code % NUM_SHARDS;

			// 특정 shard에 대해서만 lock을 걸어 경합을 최소화
			std::lock_guard<std::mutex> lock(shardMutexes[shardIdx]);

			auto& currentMap = shardMaps[shardIdx];
			auto& currentVoxels = shardVoxels[shardIdx];

			auto it = currentMap.find(key);
			if (it == currentMap.end()) {
				currentMap[key] = currentVoxels.size();
				currentVoxels.push_back({ p, n, {1.f, 1.f, 1.f, 1.f}, 1 });
			}
			else {
				auto& v = currentVoxels[it->second];
				v.position_sum += p;
				v.normal_sum += n;
				v.color_sum += Eigen::Vector4f(1, 1, 1, 1);
				v.count++;
			}
		}
		});
	TE(Accumulating);

	// 4. Drawing 단계도 병렬로 VD에 추가
	TS(Drawing);
	std::vector<size_t> shardIndices(NUM_SHARDS);
	std::iota(shardIndices.begin(), shardIndices.end(), 0);

	std::for_each(std::execution::par, shardIndices.begin(), shardIndices.end(), [&](size_t sIdx) {
		auto& currentMap = shardMaps[sIdx];
		auto& currentVoxels = shardVoxels[sIdx];

		for (auto& kvp : currentMap) {
			auto& v = currentVoxels[kvp.second];
			Eigen::Vector3f p = v.position_sum / static_cast<float>(v.count);
			Eigen::Vector3f n = v.normal_sum / static_cast<float>(v.count);
			Eigen::Vector4f c = v.color_sum / static_cast<float>(v.count);

			// VisualDebugging은 내부적으로 thread-safe해야 함 (일반적으로 VD는 대기열 방식이라 괜찮음)
			VD::AddDisk("DownSampleVoxels", p, n, 0.025f, 16, c, false);
		}
		});
	TE(Drawing);
}
