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
	VVV::VoxelDataBase voxelDB;
	VVV::IntegrationParams params;
	params.voxelSize = 0.05f;
	params.truncDist = 0.15f;
	params.maxWeight = 100.0f;

	// 1. 파일 열기
	std::ifstream ifs("D:\\Resources\\Debug\\3D\\Patches.bin", std::ios::binary);
	if (!ifs.is_open())
	{
		std::cout << "Failed to open Patches.bin" << std::endl;
		return;
	}

	// 2. 헤더 정보 로드
	CamInfo_ cam;
	Eigen::Matrix4f camRT_matrix;
	ifs.read(reinterpret_cast<char*>(&cam), sizeof(CamInfo_));
	ifs.read(reinterpret_cast<char*>(camRT_matrix.data()), sizeof(float) * 16);

	size_t numberOfPatches = 0;
	ifs.read(reinterpret_cast<char*>(&numberOfPatches), sizeof(size_t));

	std::cout << "Starting integration for " << numberOfPatches << " patches..." << std::endl;

	// 3. 패치 순회 및 통합
	for (size_t i = 0; i < numberOfPatches; i++)
	{
		size_t patchIndex = 0;
		ifs.read(reinterpret_cast<char*>(&patchIndex), sizeof(size_t));

		Eigen::Matrix4f rt0 = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f rt45 = Eigen::Matrix4f::Identity();
		ifs.read(reinterpret_cast<char*>(rt0.data()), sizeof(float) * 16);
		ifs.read(reinterpret_cast<char*>(rt45.data()), sizeof(float) * 16);

		// 데이터 로드용 임시 벡터
		std::vector<Eigen::Vector3f> points0;
		std::vector<Eigen::Vector3f> normals0;
		std::vector<Eigen::Vector3b> colors;

		size_t numberOfPoints0 = 0;
		ifs.read(reinterpret_cast<char*>(&numberOfPoints0), sizeof(size_t));

		points0.resize(numberOfPoints0);
		normals0.resize(numberOfPoints0);
		colors.resize(numberOfPoints0);

		ifs.read(reinterpret_cast<char*>(points0.data()), sizeof(Eigen::Vector3f) * numberOfPoints0);
		ifs.read(reinterpret_cast<char*>(normals0.data()), sizeof(Eigen::Vector3f) * numberOfPoints0);
		ifs.read(reinterpret_cast<char*>(colors.data()), sizeof(Eigen::Vector3b) * numberOfPoints0);

		// 45도 데이터는 위치만 읽고 스킵 (필요시 points0와 동일하게 처리 가능)
		size_t numberOfPoints45 = 0;
		ifs.read(reinterpret_cast<char*>(&numberOfPoints45), sizeof(size_t));
		ifs.seekg(numberOfPoints45 * (sizeof(Eigen::Vector3f) * 2), std::ios::cur);

		// 4. VoxelDataBase에 통합
		// VVV::Vector3f 타입 호환을 위해 변환 후 전달
		std::vector<VVV::Vector3f> integrationPoints;
		integrationPoints.reserve(numberOfPoints0);
		for (const auto& p : points0)
		{
			integrationPoints.push_back({ p.x(), p.y(), p.z() });
		}

		// 카메라 위치(cam.camPos)를 기준으로 통합 수행
		VVV::Vector3f cameraPos = { cam.camPos.x(), cam.camPos.y(), cam.camPos.z() };
		voxelDB.integrate(integrationPoints, cameraPos, params, static_cast<uint32_t>(i));

		if (i % 100 == 0)
		{
			std::cout << "Integrated patch: " << i << " / " << numberOfPatches << std::endl;
		}
	}

	std::cout << "Integration complete. Compacting database..." << std::endl;
	voxelDB.compact();
}

void BusinessLogic_Backup()
{
	VVV::VoxelDataBase voxelDB;

	std::ifstream ifs("D:\\Resources\\Debug\\3D\\Patches.bin", std::ios::binary);

	CamInfo_ cam;
	Eigen::Matrix4f camRT_matrix;
	ifs.read(reinterpret_cast<char*>(&cam), sizeof(CamInfo_));
	ifs.read(reinterpret_cast<char*>(camRT_matrix.data()), sizeof(float) * 16);

	size_t numberOfPatches = 0;
	ifs.read(reinterpret_cast<char*>(&numberOfPatches), sizeof(size_t));

	std::vector<Eigen::Vector3f> allPoints;
	std::vector<Eigen::Vector3f> allNormals;
	std::vector<Eigen::Vector4f> allColors;

	for (size_t i = 0; i < numberOfPatches; i++)
	{
		size_t patchIndex = 0;
		ifs.read(reinterpret_cast<char*>(&patchIndex), sizeof(size_t)); 
		
		Eigen::Matrix4f rt0 = Eigen::Matrix4f::Identity();
		Eigen::Matrix4f rt45 = Eigen::Matrix4f::Identity();
		ifs.read(reinterpret_cast<char*>(rt0.data()), sizeof(float) * 16);
		ifs.read(reinterpret_cast<char*>(rt45.data()), sizeof(float) * 16);

		std::vector<Eigen::Vector3f> points0;
		std::vector<Eigen::Vector3f> normals0;
		std::vector<Eigen::Vector3f> points45;
		std::vector<Eigen::Vector3f> normals45;
		std::vector<Eigen::Vector3b> colors;

		size_t numberOfPoints0 = 0;
		ifs.read(reinterpret_cast<char*>(&numberOfPoints0), sizeof(size_t));
		points0.resize(numberOfPoints0);
		normals0.resize(numberOfPoints0);
		colors.resize(numberOfPoints0);
		ifs.read(reinterpret_cast<char*>(points0.data()), sizeof(Eigen::Vector3f) * numberOfPoints0);
		ifs.read(reinterpret_cast<char*>(normals0.data()), sizeof(Eigen::Vector3f) * numberOfPoints0);
		ifs.read(reinterpret_cast<char*>(colors.data()), sizeof(Eigen::Vector3b) * numberOfPoints0);

		size_t numberOfPoints45 = 0;
		ifs.read(reinterpret_cast<char*>(&numberOfPoints45), sizeof(size_t));
		points45.resize(numberOfPoints45);
		normals45.resize(numberOfPoints45);
		ifs.read(reinterpret_cast<char*>(points45.data()), sizeof(Eigen::Vector3f) * numberOfPoints45);
		ifs.read(reinterpret_cast<char*>(normals45.data()), sizeof(Eigen::Vector3f) * numberOfPoints45);

		std::vector<Eigen::Vector4f> color4s;
		color4s.resize(colors.size());
		for (size_t cidx = 0; cidx < colors.size(); ++cidx)
		{
			color4s[cidx] = Eigen::Vector4f(
				float(colors[cidx].x()) / 255.f,
				float(colors[cidx].y()) / 255.f,
				float(colors[cidx].z()) / 255.f,
				1.0f);
		}

		allPoints.reserve(allPoints.size() + points0.size());
		allNormals.reserve(allNormals.size() + normals0.size());
		allColors.reserve(allColors.size() + colors.size());
		allPoints.insert(allPoints.end(), points0.begin(), points0.end());
		allNormals.insert(allNormals.end(), normals0.begin(), normals0.end());
		allColors.insert(allColors.end(), color4s.begin(), color4s.end());

		if (i % 100 == 0)
		{
			std::string name = "LDE_Patch_" + std::to_string(i);
			VD::AddDiskBatch(
				name,
				allPoints,
				allNormals,
				0.05f,
				16,
				allColors,
				false);

			allPoints.clear();
			allNormals.clear();
			allColors.clear();
		}
	}
}

void BusinessLogic_OLD()
{
	constexpr float SURFEL_VOXEL_SIZE = 0.05f;   // surfel downsample
	constexpr float TSDF_VOXEL_SIZE = 0.05f;   // tsdf grid
	constexpr float TSDF_TRUNC = 0.15f;   // truncation (3 * voxel)
	constexpr size_t MIN_SURFEL_COUNT = 2;

	struct AccVoxel
	{
		Eigen::Vector3f pos_sum{ 0,0,0 };
		Eigen::Vector3f nrm_sum{ 0,0,0 };
		size_t count = 0;
	};

	struct TsdfVoxel
	{
		float tsdf = 0.0f;
		float weight = 0.0f;
	};

	size_t numberOfPatches = 1953;

	std::string basePath = "D:\\Resources\\CaseDataVoxelStreaming\\Patches\\";
	std::string cachePath = basePath + "Cache\\";
	if (!std::filesystem::exists(cachePath))
		std::filesystem::create_directories(cachePath);

	std::vector<std::vector<Eigen::Vector3f>> loadedPos;
	std::vector<std::vector<Eigen::Vector3f>> loadedNrm;

	loadedPos.resize(numberOfPatches);
	loadedNrm.resize(numberOfPatches);

	std::vector<size_t> patchIdx(numberOfPatches);
	std::iota(patchIdx.begin(), patchIdx.end(), 0);

	TS(DataLoading);
	std::for_each(std::execution::par, patchIdx.begin(), patchIdx.end(),
		[&](size_t i)
		{
			std::stringstream ss;
			ss << std::setw(4) << std::setfill('0') << i;
			std::string base = "patch" + ss.str();

			std::string posBin = cachePath + base + ".pos.bin";
			std::string nrmBin = cachePath + base + ".nor.bin";

			if (std::filesystem::exists(posBin))
			{
				std::ifstream ip(posBin, std::ios::binary);
				std::ifstream in(nrmBin, std::ios::binary);

				size_t c;
				ip.read(reinterpret_cast<char*>(&c), sizeof(size_t));
				loadedPos[i].resize(c);
				ip.read(reinterpret_cast<char*>(loadedPos[i].data()),
					c * sizeof(Eigen::Vector3f));

				in.read(reinterpret_cast<char*>(&c), sizeof(size_t));
				loadedNrm[i].resize(c);
				in.read(reinterpret_cast<char*>(loadedNrm[i].data()),
					c * sizeof(Eigen::Vector3f));
			}
			else
			{
				PLYFormat ply;
				ply.Deserialize(basePath + base + ".ply");
				loadedPos[i] = ply.GetPoints();
				loadedNrm[i] = ply.GetNormals();
			}
		});
	TE(DataLoading);

	// ============================================================
	// 4. Surfel accumulation (voxel downsample)
	// ============================================================
	robin_hood::unordered_flat_map<VVV::Morton64, AccVoxel> surfelMap;
	surfelMap.reserve(300000);

	TS(SurfelAccum);
	for (size_t i = 0; i < numberOfPatches; ++i)
	{
		const auto& pts = loadedPos[i];
		const auto& nrms = loadedNrm[i];

		for (size_t j = 0; j < pts.size(); ++j)
		{
			const auto& p = pts[j];
			if (p.x() > 100000.f) continue;

			auto key = VVV::Morton64::FromPosition(
				{ p.x(), p.y(), p.z() }, SURFEL_VOXEL_SIZE);

			auto& v = surfelMap[key];
			v.pos_sum += p;
			v.nrm_sum += nrms[j];
			v.count++;
		}
	}
	TE(SurfelAccum);

	// ============================================================
	// 5. Surfel visualization (debug only)
	// ============================================================
	TS(SurfelDraw);
	std::vector<Eigen::Vector3f> surfelPos;
	std::vector<Eigen::Vector3f> surfelNrm;

	for (auto& [k, v] : surfelMap)
	{
		if (v.count < MIN_SURFEL_COUNT) continue;

		Eigen::Vector3f p = v.pos_sum / float(v.count);
		Eigen::Vector3f n = v.nrm_sum.normalized();

		surfelPos.push_back(p);
		surfelNrm.push_back(n);
	}

	VD::AddDiskBatch(
		"Surfels_Debug",
		surfelPos,
		surfelNrm,
		SURFEL_VOXEL_SIZE * 0.6f,
		16,
		Color::white(),
		false
	);
	TE(SurfelDraw);

	// ============================================================
	// 6. TSDF integration (core)
	// ============================================================
	robin_hood::unordered_flat_map<VVV::Morton64, TsdfVoxel> tsdf;
	tsdf.reserve(500000);

	TS(TSDFIntegrate);
	for (size_t i = 0; i < numberOfPatches; ++i)
	{
		const auto& pts = loadedPos[i];
		const auto& nrms = loadedNrm[i];

		for (size_t j = 0; j < pts.size(); ++j)
		{
			const Eigen::Vector3f& p = pts[j];
			const Eigen::Vector3f& n = nrms[j].normalized();

			for (float t = -TSDF_TRUNC; t <= TSDF_TRUNC; t += TSDF_VOXEL_SIZE)
			{
				Eigen::Vector3f x = p + n * t;
				auto key = VVV::Morton64::FromPosition(
					{ x.x(), x.y(), x.z() }, TSDF_VOXEL_SIZE);

				float sdf = t / TSDF_TRUNC;
				sdf = std::clamp(sdf, -1.0f, 1.0f);

				auto& v = tsdf[key];
				float w = 1.0f;

				v.tsdf = (v.tsdf * v.weight + sdf * w) / (v.weight + w);
				v.weight += w;
			}
		}
	}
	TE(TSDFIntegrate);

	// ============================================================
	// 7. Zero-crossing visualization (mesh 대신)
	// ============================================================
	TS(TSDFDraw);
	std::vector<Eigen::Vector3f> tsdfPts;
	std::vector<Eigen::Vector3f> tsdfNrms;

	for (auto& [k, v] : tsdf)
	{
		if (v.weight < 3.0f) continue;
		if (std::abs(v.tsdf) > 0.1f) continue;

		auto p = k.ToPosition(TSDF_VOXEL_SIZE);
		tsdfPts.push_back({ p.x, p.y, p.z });
		tsdfNrms.push_back(Eigen::Vector3f::UnitZ());
	}

	VD::AddDiskBatch(
		"TSDF_ZeroCross",
		tsdfPts,
		tsdfNrms,
		TSDF_VOXEL_SIZE * 0.5f,
		12,
		Color::green(),
		false
	);
	TE(TSDFDraw);

	std::cout << "Surfels: " << surfelPos.size()
		<< " | TSDF voxels: " << tsdf.size()
		<< std::endl;

	// ============================================================
// 8. Save results to file (EXPLICIT)
// ============================================================

	{
		PLYFormat plyOut;

		// ---- Surfel debug 저장 ----
		for (size_t i = 0; i < surfelPos.size(); ++i)
		{
			plyOut.AddPoint(surfelPos[i]);
			plyOut.AddNormal(
				surfelNrm[i].x(),
				surfelNrm[i].y(),
				surfelNrm[i].z()
			);
			plyOut.AddColor(1.0f, 1.0f, 1.0f, 1.0f);
		}

		plyOut.Serialize(
			"D:\\Resources\\CaseDataVoxelStreaming\\Patches\\Surfels_Debug.ply"
		);
	}

	{
		PLYFormat plyOut;

		// ---- TSDF zero-cross 저장 ----
		for (size_t i = 0; i < tsdfPts.size(); ++i)
		{
			plyOut.AddPoint(tsdfPts[i]);
			plyOut.AddNormal(0.0f, 0.0f, 1.0f); // placeholder normal
			plyOut.AddColor(0.0f, 1.0f, 0.0f, 1.0f);
		}

		plyOut.Serialize(
			"D:\\Resources\\CaseDataVoxelStreaming\\Patches\\TSDF_ZeroCross.ply"
		);
	}

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
