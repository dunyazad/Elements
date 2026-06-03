#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS

#include "framework.h"
#include "Hydrogen.h"

#include <algorithm>
#include <atomic>
#include <execution>
#include <filesystem>
#include <iostream>
#include <thread>
#include <vector>

#include <ShellScalingApi.h>
#pragma comment(lib, "Shcore.lib")

#include <Helium/Helium.h>
#include <Helium/HeliumCommon.h>
#include <Helium/HeliumCore.h>
#include <Helium/DeviceInformation.h>
#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include <Copper/Copper.h>
#include <Copper/CuPointCloud.h>
#include <Copper/CuSparseCells.h>
#include <Copper/OperatorCollection/CuOperatorCollection.h>
#include <Copper/CuVoxelStreaming.h>

#include <robin_hood/robin_hood.h>

#include <Monitor.h>

#include <Argon/Argon.h>

#include "Apps/Apps.h"

//const std::string appName = "AppTextMesh";
//const std::string appName = "AppSGL";
const std::string appName = "AppHalfEdgeMesh";
//const std::string appName = "AppManifold";
//const std::string appName = "AppSTL";
//const std::string appName = "AppSimple";
//const std::string appName = "AppTSDF";
//const std::string appName = "AppTSDFDevice";
//const std::string appName = "AppVoxelDataBaseMemoryUsageCheck";
//const std::string appName = "AppClustering";
//const std::string appName = "AppClusteringDevice";
//const std::string appName = "AppSDFFiltering";
//const std::string appName = "AppMorphology";
//const std::string appName = "AppMorphologyDebug";
//const std::string appName = "AppICP";
//const std::string appName = "AppVVV";
//const std::string appName = "AppVoxelDataBase";
//const std::string appName = "AppVoxelStreaming";
//const std::string appName = "AppMergePointClouds";
//const std::string appName = "AppMergePointClouds_Local";
//const std::string appName = "AppOperationPointCloudMerge";
//const std::string appName = "AppVoxelCache";
//const std::string appName = "AppICP";
//const std::string appName = "AppDataFrames";

#define MAX_LOADSTRING 100

HINSTANCE hInst;
WCHAR szTitle[MAX_LOADSTRING];
WCHAR szWindowClass[MAX_LOADSTRING];

std::atomic<bool> g_isRendering = true;
std::atomic<bool> g_resizeRequested = false;
std::atomic<int> g_resizeWidth = 0;
std::atomic<int> g_resizeHeight = 0;

std::thread g_renderingThread;

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
		auto largestMonitorIndex = MaximizeWindowOnLargestMonitor(hWnd);

		AllocConsole();
		freopen("CONOUT$", "w", stdout);
		MaximizeConsoleWindowOnSmallestMonitor(true);
	}
	else
	{
		auto largestMonitorIndex = MaximizeWindowOnLargestMonitor(hWnd);

		AllocConsole();
		freopen("CONOUT$", "w", stdout);
		MaximizeConsoleWindowOnSmallestMonitor(true);
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

		cudaFree(0);
		Apps::Run(appName);

		{
			auto entity = Helium.CreateEntity("main");
			Helium.CreateEventCallback<KeyEvent>(entity, "3D", [](Entity e, const KeyEvent& event) {
				if (event.action == 1 && KeyCode::Space == event.keyCode)
				{
					Helium.GetImmediateModeRenderSystem()->ToggleEnable();
				}
				else if (event.action == 1 && KeyCode::Plus == event.keyCode)
				{
					auto scale = VD::GetInstanceScale("PointCloud");
					VD::SetInstanceScale("PointCloud", scale * 1.1f);
				}
				else if (event.action == 1 && KeyCode::Minus == event.keyCode)
				{
					auto scale = VD::GetInstanceScale("PointCloud");
					VD::SetInstanceScale("PointCloud", scale * 0.9f);
				}
				else if (event.action == 0 && KeyCode::Tilde == event.keyCode)
				{
					VD::ToggleVisibility("PointCloud");
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
