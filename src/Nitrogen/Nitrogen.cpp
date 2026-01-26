#include "framework.h"
#include "Nitrogen.h"

#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include <atomic>
#include <thread>

#include <Helium/Serialization.hpp>
#include <Helium/DeviceInformation.h>

#include "TextMeshGenerator.h"

#include <ShellScalingApi.h>
#pragma comment(lib, "Shcore.lib")

#include <Monitor.h>

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
	LoadStringW(hInstance, IDC_NITROGEN, szWindowClass, MAX_LOADSTRING);
	MyRegisterClass(hInstance);

	if (!InitInstance(hInstance, nCmdShow))
	{
		return FALSE;
	}

	HWND hWnd = FindWindowW(szWindowClass, szTitle);

	std::string cpuId = CPUInformation::GetCpuID();
	std::string vendor = CPUInformation::GetVendorString();

	if ("000B0671BFEBFBFF" != cpuId)
	{
		AllocConsole();
		freopen("CONOUT$", "w", stdout);
		MaximizeConsoleWindowOnMonitor(3);

		MaximizeWindowOnMonitor(hWnd, 2);

		std::cout << "CPU Vendor: " << vendor << std::endl;
		std::cout << "CPU ID (Signature): " << cpuId << std::endl;
	}
	else
	{
		AllocConsole();
		freopen("CONOUT$", "w", stdout);
		//MaximizeConsoleWindowOnMonitor(0);
		SetConsoleToHalfOfScreen(0, 1);

		MaximizeWindowOnMonitor(hWnd, 1);

		std::cout << "CPU Vendor: " << vendor << std::endl;
		std::cout << "CPU ID (Signature): " << cpuId << std::endl;
	}

	Cu_Initialize();

	g_renderingThread = std::thread([&]() {
		He_Initialize(hWnd, 0);

		He_InitializeScene2D();

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
			auto entity = Helium.CreateEntity("Button");
			auto button = Helium.CreateComponent<GUIRectangle>(entity, 100.0f, 100.0f, 200.0f, 50.0f, Color::blue());
		}

		{
			auto entity = Helium.CreateEntity("ButtonText");
			auto button = Helium.CreateComponent<GUIText>(entity, 100.0f, 100.0f, 32.0f, Color::white(), "Button", TextHAlign::Center, TextVAlign::Middle);
		}

		{
			auto entity = Helium.CreateEntity("Circle");
			auto circle = Helium.CreateComponent<GUICircle>(entity, 500.0f, 500.0f, 50.0f, Color::red());
		}


		{
			auto entity = Helium.CreateEntity("3DTextBooleanExample");
			auto renderable = Helium.CreateComponent<Renderable>(entity);
			renderable->Initialize(Renderable::Triangles);

			// 1. Generator 초기화
			TextMeshGenerator generator;
			// 경로 주의! 실제 폰트 파일이 있어야 합니다.
			if (!generator.LoadFont("../../res/Fonts/NanumGothic/NanumGothic.ttf")) {
				return;
			}

			// 2. 3D 텍스트 생성 (Manifold 객체)
			// 텍스트: "HELIUM", 두께: 20.0f, 크기: 0.5f
			auto textManifold = generator.Create3DText("HELIUM", 20.0f, 0.5f);

			// 3. 타겟 큐브 생성 (Manifold 객체)
			// 크기 100x100x100 큐브
			auto cubeManifold = manifold::Manifold::Cube({ 100.0f, 100.0f, 50.0f }, true);

			// 4. 위치 조정
			// 텍스트를 큐브 중앙으로 이동 (Rotate 등도 가능)
			// Manifold는 Translate, Rotate, Scale 함수를 제공합니다.
			textManifold = textManifold.Translate({ -80.0f, -10.0f, 15.0f });

			// 5. Boolean 연산 (각인: Cube - Text)
			// (-) 연산자가 차집합(Difference)입니다.
			auto resultManifold = cubeManifold - textManifold;
			// 합치려면: auto resultManifold = cubeManifold + textManifold;

			// 6. 결과를 Helium 메쉬로 변환하여 렌더링
			manifold::MeshGL resultMesh = resultManifold.GetMeshGL();

			// Helium::Mesh 구조체에 데이터 복사
			std::vector<Eigen::Vector3f> positions;
			std::vector<Eigen::Vector3f> normals;
			std::vector<Eigen::Vector4f> colors;
			std::vector<unsigned int> indices;

			// A. 정점(Vertices) 변환
			// meshGL.vertProperties는 [x, y, z, x, y, z...] 순서로 들어있음
			size_t numVerts = resultMesh.vertProperties.size() / 3;

			positions.reserve(numVerts);
			normals.reserve(numVerts);
			colors.reserve(numVerts);

			for (size_t i = 0; i < numVerts; ++i)
			{
				float x = resultMesh.vertProperties[i * 3 + 0];
				float y = resultMesh.vertProperties[i * 3 + 1];
				float z = resultMesh.vertProperties[i * 3 + 2];

				// 1. 위치
				positions.emplace_back(x, y, z);

				// 2. 법선 (Normal)
				// Manifold 결과에는 Normal이 없습니다. 
				// Boolean 연산 후에는 면이 쪼개지므로 Flat Shading을 위해 
				// 나중에 면(Face) 단위로 계산하거나, 임시로 기본값(Z-up)을 넣습니다.
				// *제대로 하려면: 각 삼각형 면의 Normal을 계산해서 정점에 할당해야 함 (Flat Shading)
				normals.emplace_back(0.0f, 0.0f, 1.0f);

				// 3. 색상 (Color) - 흰색
				colors.emplace_back(1.0f, 1.0f, 1.0f, 1.0f);
			}

			// B. 인덱스(Indices) 변환
			indices.reserve(resultMesh.triVerts.size());
			for (const auto& idx : resultMesh.triVerts)
			{
				indices.push_back((unsigned int)idx);
			}

			for (size_t i = 0; i < indices.size(); i += 3) {
				unsigned int i0 = indices[i];
				unsigned int i1 = indices[i+1];
				unsigned int i2 = indices[i+2];

				Eigen::Vector3f v0 = positions[i0];
				Eigen::Vector3f v1 = positions[i1];
				Eigen::Vector3f v2 = positions[i2];

				Eigen::Vector3f faceNormal = (v1 - v0).cross(v2 - v0).normalized();
				normals[i0] = faceNormal;
				normals[i1] = faceNormal;
				normals[i2] = faceNormal;
			}

			renderable->SetVertices(positions);
			renderable->SetNormals(normals);
			renderable->SetColors4(colors);
			renderable->SetIndices(indices);
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

	HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_NITROGEN));
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
	if (g_renderingThread.joinable())
	{
		g_renderingThread.join();
	}

	CheckDeviceMemory("Before Shutdown");

	Cu_Shutdown();

	CheckDeviceMemory("After Shutdown");

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
	wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_NITROGEN));
	wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_NITROGEN);
	wcex.lpszClassName = szWindowClass;
	wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

	return RegisterClassExW(&wcex);
}

BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
	hInst = hInstance;

	HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

	if (!hWnd)
	{
		return FALSE;
	}

	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);

	return TRUE;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_KEYDOWN:
		if (VK_ESCAPE == wParam)
		{
			PostMessage(hWnd, WM_CLOSE, 0, 0);
		}
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
		case IDM_ABOUT:
			DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
			break;
		case IDM_EXIT:
			DestroyWindow(hWnd);
			break;
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
	}
	break;
	case WM_ERASEBKGND:
		return 1;
	case WM_PAINT:
	{
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hWnd, &ps);
		EndPaint(hWnd, &ps);
	}
	break;
	case WM_SIZE:
		// 리사이즈 요청
		g_resizeWidth = LOWORD(lParam);
		g_resizeHeight = HIWORD(lParam);
		g_resizeRequested = true;
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	switch (message)
	{
	case WM_INITDIALOG:
		return (INT_PTR)TRUE;

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
