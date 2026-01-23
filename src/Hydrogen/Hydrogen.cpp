#include "framework.h"
#include "Hydrogen.h"

#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include <atomic>
#include <thread>

#include <Helium/Serialization.hpp>

#include <ShellScalingApi.h>
#pragma comment(lib, "Shcore.lib")

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

	AllocConsole();
	freopen("CONOUT$", "w", stdout);

	CheckDeviceMemory("Initial");

	Cu_Initialize();

	PLYFormat ply;
	ply.Deserialize("D:\\Temp\\PLY\\DensityEstimation\\Model.ply");

	TS(MakePointCloud);

	CuPointCloud pointCloud;
	pointCloud.FromHostPointers(
		(float3*)ply.GetPoints().data(),
		(float3*)ply.GetNormals().data(),
		(float4*)ply.GetColors().data(),
		ply.GetPoints().size()
	);

	TE(MakePointCloud);

	TS(Build);

	CuSparseDataBlock sparseDataBlock;
	sparseDataBlock.Build(&pointCloud);

	TE(Build);

	//{
	//	//CuOperatorCollection operatorCollection;

	//	CuOperatorPointCloudKDE op;
	//	CuOperatorParameters params;
	//	params.SetParameter<CuPointCloud*>("pointCloud", &pointCloud);
	//	params.SetParameter<CuSparseDataBlock*>("sparseDataBlock", &sparseDataBlock);
	//	params.SetParameter<int>("k", 30);
	//	params.SetParameter<float>("bandwidth", 0.2f);
	//	std::vector<float> densities;
	//	TS(Execute);
	//	op.Execute(params, densities);
	//	TE(Execute);

	//	std::vector<float> h_densities(densities.size());
	//	thrust::copy(densities.begin(), densities.end(), h_densities.begin());

	//	thrust::host_vector<float3> h_points = pointCloud.points;
	//	thrust::host_vector<float3> h_normals = pointCloud.normals;
	//	thrust::host_vector<uchar3> h_colors = pointCloud.colors;

	//	auto [densityMin, densityMax] = std::minmax_element(h_densities.begin(), h_densities.end());
	//	printf("Density Min: %f, Max: %f\n", *densityMin, *densityMax);

	//	for (size_t i = 0; i < pointCloud.size(); i++)
	//	{
	//		auto& p = h_points[i];
	//		auto& n = h_normals[i];
	//		auto& c = h_colors[i];

	//		if (h_densities[i] < 35.0f)
	//		{
	//			VD::AddSphere("KDE",
	//				{ p.x, p.y, p.z },
	//				{ n.x, n.y, n.z },
	//				0.05f,
	//				{ 1.0f, 0.0f, 0.0f, 1.0f });
	//		}
	//		else
	//		{
	//			VD::AddSphere("KDE",
	//				{ p.x, p.y, p.z },
	//				{ n.x, n.y, n.z },
	//				0.05f,
	//				{ (float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f, 1.0f });
	//		}
	//	}
	//}

	{
		TS(LDE);

		CuOperatorPointCloudLDE op;
		CuOperatorParameters params;
		params.SetParameter<CuPointCloud*>("pointCloud", &pointCloud);
		params.SetParameter<CuSparseDataBlock*>("sparseDataBlock", &sparseDataBlock);
		params.SetParameter<float>("radius", 0.5f);
		std::vector<float> densities;
		TS(Execute);
		op.Execute(params, densities);
		TE(Execute);

		std::vector<float> h_densities(densities.size());
		thrust::copy(densities.begin(), densities.end(), h_densities.begin());

		thrust::host_vector<float3> h_points = pointCloud.points;
		thrust::host_vector<float3> h_normals = pointCloud.normals;
		thrust::host_vector<uchar3> h_colors = pointCloud.colors;

		auto [densityMin, densityMax] = std::minmax_element(h_densities.begin(), h_densities.end());
		printf("Density Min: %f, Max: %f\n", *densityMin, *densityMax);

		std::vector<float3> lowDensityPoints;
		std::vector<float3> lowDensityNormals;
		std::vector<uchar3> lowDensityColors;

		for (size_t i = 0; i < pointCloud.size(); i++)
		{
			auto& p = h_points[i];
			auto& n = h_normals[i];
			auto& c = h_colors[i];

			if (h_densities[i] < 85.0f)
			{
				VD::AddDisk("LDE_LowDensityPoints", { p.x, p.y, p.z }, { n.x, n.y, n.z }, 0.1f, 16, Color::red(), true);

				lowDensityPoints.push_back(p);
				lowDensityNormals.push_back(n);
				lowDensityColors.push_back(c);
			}
			else
			{
				VD::AddDisk("LDE", { p.x, p.y, p.z }, { n.x, n.y, n.z }, 0.01f, 16, { (float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f, 1.0f }, true);
			}
		}

		TS(BuildArrowBlocks);
		CuPointCloud lowDensityPointCloud;
		lowDensityPointCloud.FromHostVectors(lowDensityPoints, lowDensityNormals, lowDensityColors);

		CuSparseDataBlock sparseDataBlockForLowDensityPointCloud;
		sparseDataBlockForLowDensityPointCloud.Build(&lowDensityPointCloud, 10.0f);
		TE(BuildArrowBlocks);

		////////sparseDataBlockForLowDensityPointCloud.ColorizePointsByCell(&lowDensityPointCloud);
		//////// -> 이후 myCloud를 렌더링하면 알록달록하게 복셀 단위로 구분되어 보임

		//////// 3. 그리드(복셀) 가시화 (Wireframe)
		//////std::vector<std::pair<float3, float3>> boxes = sparseDataBlockForLowDensityPointCloud.GetActiveCellBounds();

		//////for (const auto& box : boxes)
		//////{
		//////	float3 minP = box.first;
		//////	float3 maxP = box.second;

		//////	VD::AddWiredBox("LDE_SparseDataBlocks",
		//////		{ (minP.x + maxP.x) * 0.5f, (minP.y + maxP.y) * 0.5f, (minP.z + maxP.z) * 0.5f },
		//////		{ maxP.x - minP.x, maxP.y - minP.y, maxP.z - minP.z },
		//////		Color::green());

		//////	// 렌더링 엔진의 Line Drawing 함수 호출 (예시)
		//////	// DrawWireBox(minP, maxP, Color::Green); 

		//////	// 혹은 GL_LINES 등을 이용해 12개의 선분을 그립니다.
		//////	// (min.x, min.y, min.z) -> (max.x, min.y, min.z) ...
		//////}

		TS(GetActiveCellStats);
		auto cellStats = sparseDataBlockForLowDensityPointCloud.GetActiveCellStats(&lowDensityPointCloud);
		TE(GetActiveCellStats);

		for (const auto& cellStat : cellStats)
		{
			VD::AddWiredBox("LDE_SparseDataBlocks",
				{ (cellStat.cellMin.x + cellStat.cellMax.x) * 0.5f,
				  (cellStat.cellMin.y + cellStat.cellMax.y) * 0.5f,
				  (cellStat.cellMin.z + cellStat.cellMax.z) * 0.5f },
				{ cellStat.cellMax.x - cellStat.cellMin.x,
				  cellStat.cellMax.y - cellStat.cellMin.y,
				  cellStat.cellMax.z - cellStat.cellMin.z },
				Color::green());

			// cellStat.pointCentroid 위치에서 가장 먼 코너에서 부터 cellStat.pointCentroid 방향으로 화살표 그리기
			float3 corner;
			corner.x = (fabsf(cellStat.pointCentroid.x - cellStat.cellMin.x) > fabsf(cellStat.pointCentroid.x - cellStat.cellMax.x)) ? cellStat.cellMin.x : cellStat.cellMax.x;
			corner.y = (fabsf(cellStat.pointCentroid.y - cellStat.cellMin.y) > fabsf(cellStat.pointCentroid.y - cellStat.cellMax.y)) ? cellStat.cellMin.y : cellStat.cellMax.y;
			corner.z = (fabsf(cellStat.pointCentroid.z - cellStat.cellMin.z) > fabsf(cellStat.pointCentroid.z - cellStat.cellMax.z)) ? cellStat.cellMin.z : cellStat.cellMax.z;
			VD::AddArrow("LDE_LowDensityPointNormals",
				{ corner.x, corner.y, corner.z },
				{ cellStat.pointCentroid.x - corner.x,
				  cellStat.pointCentroid.y - corner.y,
				  cellStat.pointCentroid.z - corner.z },
				5.0f,
				Color::blue());


			//VD::AddArrow("LDE_LowDensityPointNormals",
			//	{ cellStat.pointCentroid.x, cellStat.pointCentroid.y, cellStat.pointCentroid.z },
			//	{ cellStat.pcaNormal.x, cellStat.pcaNormal.y, cellStat.pcaNormal.z },
			//	5.0f,
			//	Color::blue());
		}
		TE(LDE);
	}

	//{
	//	TS(NND);
	//	sparseDataBlock.ApplyNND(&pointCloud, 30);
	//	TE(NND);

	//	std::vector<float3> filteredPoints;
	//	std::vector<float3> filteredNormals;
	//	std::vector<float4> filteredColors;
	//	pointCloud.ToHostVectors(filteredPoints, filteredNormals, filteredColors);

	//	PLYFormat outPly;
	//	for (size_t i = 0; i < filteredPoints.size(); i++)
	//	{
	//		outPly.AddPoint(filteredPoints[i].x, filteredPoints[i].y, filteredPoints[i].z);
	//		outPly.AddNormal(filteredNormals[i].x, filteredNormals[i].y, filteredNormals[i].z);
	//		outPly.AddColor(filteredColors[i].x, filteredColors[i].y, filteredColors[i].z, 1.0f);
	//	}
	//	outPly.Serialize("D:\\Temp\\PLY\\DensityEstimation\\Model_NND.ply");
	//}

	//{
	//	TS(LDE);
	//	sparseDataBlock.ApplyLDE(&pointCloud, 0.5f);
	//	TE(LDE);

	//	std::vector<float3> filteredPoints;
	//	std::vector<float3> filteredNormals;
	//	std::vector<float4> filteredColors;
	//	pointCloud.ToHostVectors(filteredPoints, filteredNormals, filteredColors);

	//	PLYFormat outPly;
	//	for (size_t i = 0; i < filteredPoints.size(); i++)
	//	{
	//		outPly.AddPoint(filteredPoints[i].x, filteredPoints[i].y, filteredPoints[i].z);
	//		outPly.AddNormal(filteredNormals[i].x, filteredNormals[i].y, filteredNormals[i].z);
	//		outPly.AddColor(filteredColors[i].x, filteredColors[i].y, filteredColors[i].z, 1.0f);
	//	}
	//	outPly.Serialize("D:\\Temp\\PLY\\DensityEstimation\\Model_LDE.ply");
	//}

	//{
	//	TS(KDE);
	//	sparseDataBlock.ApplyKDE(&pointCloud, 0.2f);
	//	TE(KDE);

	//	std::vector<float3> filteredPoints;
	//	std::vector<float3> filteredNormals;
	//	std::vector<float4> filteredColors;
	//	pointCloud.ToHostVectors(filteredPoints, filteredNormals, filteredColors);

	//	//PLYFormat outPly;
	//	for (size_t i = 0; i < filteredPoints.size(); i++)
	//	{
	//		//outPly.AddPoint(filteredPoints[i].x, filteredPoints[i].y, filteredPoints[i].z);
	//		//outPly.AddNormal(filteredNormals[i].x, filteredNormals[i].y, filteredNormals[i].z);
	//		//outPly.AddColor(filteredColors[i].x, filteredColors[i].y, filteredColors[i].z, 1.0f);

	//		VD::AddSphere("KDE",
	//			{ filteredPoints[i].x, filteredPoints[i].y, filteredPoints[i].z },
	//			{ filteredNormals[i].x, filteredNormals[i].y, filteredNormals[i].z },
	//			0.05f,
	//			{ filteredColors[i].x, filteredColors[i].y, filteredColors[i].z, 1.0f });
	//	}
	//	//outPly.Serialize("D:\\Temp\\PLY\\DensityEstimation\\Model_KDE.ply");
	//}

	//{
	//	TS(PFOR);
	//	sparseDataBlock.ApplyPFOR(&pointCloud, 30, 0.07f);
	//	TE(PFOR);

	//	std::vector<float3> filteredPoints;
	//	std::vector<float3> filteredNormals;
	//	std::vector<float4> filteredColors;
	//	pointCloud.ToHostVectors(filteredPoints, filteredNormals, filteredColors);

	//	PLYFormat outPly;
	//	for (size_t i = 0; i < filteredPoints.size(); i++)
	//	{
	//		outPly.AddPoint(filteredPoints[i].x, filteredPoints[i].y, filteredPoints[i].z);
	//		outPly.AddNormal(filteredNormals[i].x, filteredNormals[i].y, filteredNormals[i].z);
	//		outPly.AddColor(filteredColors[i].x, filteredColors[i].y, filteredColors[i].z, 1.0f);
	//	}
	//	outPly.Serialize("D:\\Temp\\PLY\\DensityEstimation\\Model_PFOR.ply");
	//}

	//{
	//	TS(SOR);
	//	sparseDataBlock.ApplySOR(&pointCloud, 30, 1.0f);
	//	TE(SOR);

	//	std::vector<float3> filteredPoints;
	//	std::vector<float3> filteredNormals;
	//	std::vector<float4> filteredColors;
	//	pointCloud.ToHostVectors(filteredPoints, filteredNormals, filteredColors);

	//	PLYFormat outPly;
	//	for (size_t i = 0; i < filteredPoints.size(); i++)
	//	{
	//		outPly.AddPoint(filteredPoints[i].x, filteredPoints[i].y, filteredPoints[i].z);
	//		outPly.AddNormal(filteredNormals[i].x, filteredNormals[i].y, filteredNormals[i].z);
	//		outPly.AddColor(filteredColors[i].x, filteredColors[i].y, filteredColors[i].z, 1.0f);
	//	}
	//	outPly.Serialize("D:\\Temp\\PLY\\DensityEstimation\\Model_SOR.ply");
	//}

	g_renderingThread = std::thread([hWnd]() {
		He_Initialize(hWnd, 0);

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
			auto entity = Helium.CreateEntity("main");
			Helium.CreateEventCallback<KeyEvent>(entity, "3D", [](Entity e, const KeyEvent& event) {
				if (event.action == 1 && KeyCode::Plus == event.keyCode)
				{
					auto scale = VD::GetInstanceScale("LDE");
					printf("Scale X: %f\n", scale.x());

					VD::SetInstanceScale("LDE", scale * 1.1f);
				}
				else if (event.action == 1 && KeyCode::Minus == event.keyCode)
				{
					auto scale = VD::GetInstanceScale("LDE");
					printf("Scale X: %f\n", scale.x());

					VD::SetInstanceScale("LDE", scale * 0.9f);
				}
				else if (event.action == 0 && KeyCode::Tilde == event.keyCode)
				{
					//VD::ToggleVisibility("LDE_LowDensityPoints");
					//VD::ToggleVisibility("LDE_LowDensityPointNormals");
					VD::ToggleVisibility("LDE_SparseDataBlocks");
				}
				});
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
