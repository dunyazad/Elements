#include "framework.h"
#include "Hydrogen.h"

#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include <atomic>
#include <thread>

#include <Helium/Serialization.hpp>
#include <Helium/DeviceInformation.h>

#include <ShellScalingApi.h>
#pragma comment(lib, "Shcore.lib")

#include <Monitor.h>

#include "TextMeshGenerator.h"

#include <mapbox/earcut.hpp>

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
				else if (event.action == 0 && KeyCode::D1 == event.keyCode)
				{
					VD::ToggleVisibility("LDE");
				}
				else if (event.action == 0 && KeyCode::D2 == event.keyCode)
				{
					VD::ToggleVisibility("LDE_LowDensityPointNormals");
				}
				});

			Helium.CreateEventCallback<MouseButtonEvent>(entity, "3D", [&](Entity e, const MouseButtonEvent& event) {
				if (event.action == 1 && event.button == MouseButton::Left)
				{
					if (event.IsCtrlPressed())
					{
						auto entity = Helium.GetEntityByName("MainCamera");
						auto camera = Helium.GetComponent<Camera>(entity);
						if (nullptr == camera) return;

						Eigen::Matrix4f viewMatrix = camera->GetViewMatrix();
						Eigen::Matrix4f projMatrix = camera->GetProjectionMatrix();

						float mouseX = (float)event.xpos;
						float mouseY = (float)event.ypos;
						float screenW = (float)Helium.GetWidth();
						float screenH = (float)Helium.GetHeight();

						float x = (2.0f * mouseX) / screenW - 1.0f;
						float y = 1.0f - (2.0f * mouseY) / screenH;
						float z = 1.0f;

						Eigen::Vector4f rayClip(x, y, -1.0f, 1.0f);
						Eigen::Matrix4f projInv = projMatrix.inverse();
						Eigen::Vector4f rayView = projInv * rayClip;

						rayView = Eigen::Vector4f(rayView.x(), rayView.y(), -1.0f, 0.0f);
						Eigen::Matrix4f viewInv = viewMatrix.inverse();
						Eigen::Vector4f rayWorld4 = viewInv * rayView;
						Eigen::Vector3f rayDir(rayWorld4.x(), rayWorld4.y(), rayWorld4.z());
						rayDir.normalize();

						Eigen::Vector3f rayOrigin = viewInv.block<3, 1>(0, 3);

						auto toFloat3 = [](const Eigen::Vector3f& v) { return make_float3(v.x(), v.y(), v.z()); };

						float tolerance = 0.05f;

						/*PickResult pickResult = pointCloud.Pick(
							toFloat3(rayOrigin),
							toFloat3(rayDir),
							tolerance
						);

						if (pickResult.index != -1)
						{
							auto cameraManipulator = Helium.GetComponent<CameraManipulatorTrackball>(entity);
							if (nullptr != cameraManipulator)
							{
								cameraManipulator->SetCenter(Eigen::Vector3f(pickResult.position.x, pickResult.position.y, pickResult.position.z));
							}
						}*/
					}
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

		{
				auto LoadManifoldFromPLYFormat = [](const std::string& path) -> manifold::Manifold
				{
					// 1. 사용자 정의 PLYFormat 사용
					PLYFormat plyLoader;
					if (!plyLoader.Deserialize(path))
					{
						printf("!!! [ERROR] PLYFormat Deserialize failed: %s\n", path.c_str());
						return manifold::Manifold();
					}

					const auto& points = plyLoader.GetPoints();
					const auto& indices = plyLoader.GetTriangleIndices();

					if (points.empty())
					{
						printf("!!! [ERROR] PLY has no vertices.\n");
						return manifold::Manifold();
					}

					printf(">>> [PLY INFO] Loaded Vertices: %llu, Triangles: %llu\n", points.size(), indices.size());

					// 2. Manifold::MeshGL 구조로 데이터 변환
					manifold::MeshGL meshGL;
					meshGL.numProp = 3; // Position (x, y, z)
					meshGL.vertProperties.reserve(points.size() * 3);
					meshGL.triVerts.reserve(indices.size() * 3);

					// (1) 정점 변환 (Eigen::Vector3f -> std::vector<float>)
					for (const auto& p : points)
					{
						meshGL.vertProperties.push_back(p.x());
						meshGL.vertProperties.push_back(p.y());
						meshGL.vertProperties.push_back(p.z());
					}

					// (2) 인덱스 변환 (Eigen::Vector3i -> std::vector<uint32_t>)
					for (const auto& tri : indices)
					{
						meshGL.triVerts.push_back((uint32_t)tri.x());
						meshGL.triVerts.push_back((uint32_t)tri.y());
						meshGL.triVerts.push_back((uint32_t)tri.z());
					}

					// 3. Manifold 객체 생성
					// 데이터 유효성 검사 (Manifold는 닫힌 메쉬를 선호하지만, 열린 메쉬도 처리는 가능)
					return manifold::Manifold(meshGL);
				};

				auto CreateClosedBaseMesh = [](const manifold::MeshGL& inputMesh, float baseHeightOffset) -> manifold::Manifold
				{
					// 1. 데이터 복사
					std::vector<float> vertProps = inputMesh.vertProperties;
					std::vector<uint32_t> triVerts = inputMesh.triVerts;
					int numProps = inputMesh.numProp;
					int numVerts = (int)vertProps.size() / numProps;

					// -------------------------------------------------------------------------
					// [핵심 수정] 경계선(Boundary) 찾기 로직
					// 원리: 모든 삼각형의 엣지를 (u->v) 방향으로 저장했을 때,
					// 반대 방향(v->u) 짝이 없는 엣지가 바로 경계선입니다.
					// -------------------------------------------------------------------------

					// (1) 모든 Directed Edge 수집
					// set을 사용해 중복을 제거하고 검색 속도를 확보합니다.
					std::set<std::pair<int, int>> allDirectedEdges;

					for (size_t i = 0; i < triVerts.size(); i += 3) {
						int v0 = triVerts[i];
						int v1 = triVerts[i + 1];
						int v2 = triVerts[i + 2];

						// 삼각형의 Winding Order(CCW) 그대로 저장
						allDirectedEdges.insert({ v0, v1 });
						allDirectedEdges.insert({ v1, v2 });
						allDirectedEdges.insert({ v2, v0 });
					}

					// (2) 짝(Twin)이 없는 엣지만 골라내서 연결 리스트(Next Map) 구성
					std::map<int, int> boundaryNext; // Current Vertex -> Next Vertex

					for (const auto& edge : allDirectedEdges) {
						int u = edge.first;
						int v = edge.second;

						// 반대 방향(v->u)이 존재하는지 확인
						if (allDirectedEdges.find({ v, u }) == allDirectedEdges.end()) {
							// 반대 방향이 없다면, 이것이 '경계선'이며 방향도 올바른 상태임
							boundaryNext[u] = v;
						}
					}

					if (boundaryNext.empty()) {
						printf("!!! [ERROR] 경계선을 찾을 수 없습니다. (이미 닫힌 메쉬일 수 있음)\n");
						return manifold::Manifold(inputMesh);
					}

					// 3. 루프 추적 (Boundary Loop Linking)
					// 끊어진 곳 없이 하나의 루프로 연결합니다.
					std::vector<int> boundaryLoop;

					// 시작점 찾기 (아무거나 하나 잡음)
					int startVert = boundaryNext.begin()->first;
					int curr = startVert;

					// 루프 순회
					size_t safeGuard = 0;
					do {
						boundaryLoop.push_back(curr);

						if (boundaryNext.find(curr) == boundaryNext.end()) {
							printf("!!! [ERROR] 경계선이 끊겨 있습니다. (Non-Manifold Mesh 가능성)\n");
							break;
						}

						curr = boundaryNext[curr];
						safeGuard++;

					} while (curr != startVert && safeGuard < boundaryNext.size() + 10);

					printf(">>> [INFO] 경계선 추출 완료: %d개 정점\n", (int)boundaryLoop.size());

					// 4. 바닥면 생성 (Z값 내리기)
					float minZ = 1e9f;
					for (int i = 0; i < numVerts; ++i) {
						float z = vertProps[i * numProps + 2];
						if (z < minZ) minZ = z;
					}
					float baseZ = minZ - baseHeightOffset;

					int baseStartIndex = numVerts; // 기존 정점 끝번호부터 바닥 정점 시작

					// 바닥 정점 데이터 추가
					for (int idx : boundaryLoop) {
						// 기존 속성 복사
						for (int k = 0; k < numProps; ++k) {
							vertProps.push_back(vertProps[idx * numProps + k]);
						}
						// Z값 변경
						vertProps.back() = baseZ;
						vertProps[vertProps.size() - numProps + 2] = baseZ; // Z index safe access
					}

					// 5. 벽(Wall) 생성 (Quads -> Triangles)
					// boundaryLoop의 정점들은 이미 CCW(반시계) 순서로 정렬되어 있습니다.
					int loopSize = (int)boundaryLoop.size();
					for (int i = 0; i < loopSize; ++i) {
						int top1 = boundaryLoop[i];
						int top2 = boundaryLoop[(i + 1) % loopSize];

						int bot1 = baseStartIndex + i;
						int bot2 = baseStartIndex + ((i + 1) % loopSize);

						// [Wall Triangles]
						// 1. Top1 -> Bot1 -> Top2
						triVerts.push_back(top1);
						triVerts.push_back(bot1);
						triVerts.push_back(top2);

						// 2. Top2 -> Bot1 -> Bot2
						triVerts.push_back(top2);
						triVerts.push_back(bot1);
						triVerts.push_back(bot2);
					}

					// 6. 바닥(Cap) 뚜껑 닫기 (Earcut)
					std::vector<std::vector<std::array<double, 2>>> polygon;
					std::vector<std::array<double, 2>> contour;

					for (int i = 0; i < loopSize; ++i) {
						int idx = boundaryLoop[i];
						// 바닥 뚜껑은 위에서 봤을 때 구멍을 막는 것이므로
						// X, Y 좌표를 그대로 씁니다.
						float x = vertProps[idx * numProps];
						float y = vertProps[idx * numProps + 1];
						contour.push_back({ (double)x, (double)y });
					}
					polygon.push_back(contour);

					// 삼각화 수행
					std::vector<uint32_t> capIndices = mapbox::earcut<uint32_t>(polygon);

					// 바닥 인덱스 추가 (Normal 방향 주의: 바닥은 아래를 봐야 함)
					// 원본 루프가 CCW였다면, 바닥면은 뒤집어야(CW) 아래를 향합니다.
					for (size_t i = 0; i < capIndices.size(); i += 3) {
						// Earcut 결과 인덱스 + 오프셋
						uint32_t idx0 = baseStartIndex + capIndices[i];
						uint32_t idx1 = baseStartIndex + capIndices[i + 1];
						uint32_t idx2 = baseStartIndex + capIndices[i + 2];

						// 순서를 바꿔서(0-2-1) Normal을 아래로 향하게 함
						triVerts.push_back(idx0);
						triVerts.push_back(idx2);
						triVerts.push_back(idx1);
					}

					// 7. 최종 Manifold 생성
					manifold::MeshGL resultMesh;
					resultMesh.numProp = numProps;
					resultMesh.vertProperties = vertProps;
					resultMesh.triVerts = triVerts;

					return manifold::Manifold(resultMesh);
				};

				auto entity = Helium.CreateEntity("PlyBooleanExample");
				auto renderable = Helium.CreateComponent<Renderable>(entity);
				renderable->Initialize(Renderable::Triangles);
				renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

				// 1. PLY 로드
				//std::string plyPath = "../../res/Models/bunny.ply";
				//std::string plyPath = "D:\\Resources\\3D\\PLY\\Stanford_Bunny.ply";
				std::string plyPath = "D:\\Resources\\3D\\PLY\\mesh_cap.ply";
				printf(">>> [STEP 1] Loading PLY using PLYFormat class...\n");

				manifold::Manifold plyManifold = LoadManifoldFromPLYFormat(plyPath);

				if (plyManifold.NumVert() == 0) return;

				plyManifold = CreateClosedBaseMesh(plyManifold.GetMeshGL(), 10.0f);

				// 2. 큐브 생성
				float cubeSize = 500.0f;
				manifold::Manifold cubeManifold = manifold::Manifold::Cube({ cubeSize, cubeSize * 0.1f, cubeSize }, true);

				// 3. [정렬 및 스케일링] 
				auto plyBox = plyManifold.BoundingBox();

				// [오류 수정] * 0.5f -> * 0.5 (double)
				// Manifold의 vec3는 double 정밀도를 사용하므로 연산 시 double을 써야 합니다.
				manifold::vec3 plyCenter = (plyBox.min + plyBox.max) * 0.5;

				// (1) 중앙 정렬
				printf(">>> [ALIGN] Centering PLY Mesh...\n");
				plyManifold = plyManifold.Translate({ -plyCenter.x, -plyCenter.y + 5.0f, -plyCenter.z });

				// (2) 스케일 자동 조정
				// std::max는 같은 타입끼리 비교해야 하므로 float/double 혼용 주의
				double dx = plyBox.max.x - plyBox.min.x;
				double dy = plyBox.max.y - plyBox.min.y;
				double dz = plyBox.max.z - plyBox.min.z;
				double plyMaxDim = std::max({ dx, dy, dz });

				if (plyMaxDim > 0.001)
				{
					double targetSize = (double)cubeSize * 0.8;
					double scaleFactor = targetSize / plyMaxDim;

					printf(">>> [SCALE] Auto-scaling PLY by factor: %.4f (Original Size: %.2f -> Target: %.2f)\n",
						scaleFactor, plyMaxDim, targetSize);

					// Scale 함수는 vec3(double)을 인자로 받습니다.
					plyManifold = plyManifold.Scale({ scaleFactor, scaleFactor, scaleFactor });
				}

				// 4. Boolean 연산
				printf(">>> [STEP 2] Running Boolean Difference (Cube - Ply)...\n");
				printf("    Cube Verts: %d, Ply Verts: %d\n", cubeManifold.NumVert(), plyManifold.NumVert());

				manifold::Manifold result = cubeManifold - plyManifold;

				if (result.NumVert() == 0)
				{
					printf("!!! [WARN] Difference result is empty. Trying Union for debugging...\n");
					result = cubeManifold + plyManifold;
				}

				printf(">>> [STEP 3] Operation Complete. Result Verts: %d\n", result.NumVert());

				//result.Simplify();

				// 5. 렌더링 데이터 변환
				manifold::MeshGL resultMesh = result.GetMeshGL();

				std::vector<Eigen::Vector3f> flatPositions;
				std::vector<Eigen::Vector3f> flatNormals;
				std::vector<Eigen::Vector4f> flatColors;
				std::vector<unsigned int> flatIndices;

				auto GetPos = [&](int idx) -> Eigen::Vector3f {
					return Eigen::Vector3f(
						(float)resultMesh.vertProperties[idx * 3 + 0], // double -> float 명시적 캐스팅
						(float)resultMesh.vertProperties[idx * 3 + 1],
						(float)resultMesh.vertProperties[idx * 3 + 2]
					);
					};

				// [경고 수정] printf %llu 사용 (size_t 대응)
				printf(">>> [DEBUG] Generating Render Data (Triangles: %llu)...\n", resultMesh.triVerts.size() / 3);

				for (size_t i = 0; i < resultMesh.triVerts.size(); i += 3)
				{
					unsigned int i0 = resultMesh.triVerts[i + 0];
					unsigned int i1 = resultMesh.triVerts[i + 1];
					unsigned int i2 = resultMesh.triVerts[i + 2];

					Eigen::Vector3f v0 = GetPos(i0);
					Eigen::Vector3f v1 = GetPos(i1);
					Eigen::Vector3f v2 = GetPos(i2);

					Eigen::Vector3f faceNormal = (v1 - v0).cross(v2 - v0).normalized();

					flatPositions.push_back(v0);
					flatPositions.push_back(v1);
					flatPositions.push_back(v2);

					flatNormals.push_back(faceNormal);
					flatNormals.push_back(faceNormal);
					flatNormals.push_back(faceNormal);

					flatColors.push_back({ 0.9f, 0.9f, 0.9f, 1.0f });
					flatColors.push_back({ 0.9f, 0.9f, 0.9f, 1.0f });
					flatColors.push_back({ 0.9f, 0.9f, 0.9f, 1.0f });

					unsigned int currentIdx = (unsigned int)flatIndices.size();
					flatIndices.push_back(currentIdx + 0);
					flatIndices.push_back(currentIdx + 1);
					flatIndices.push_back(currentIdx + 2);
				}

				renderable->SetVertices(flatPositions);
				renderable->SetNormals(flatNormals);
				renderable->SetColors4(flatColors);
				renderable->SetIndices(flatIndices);

				Helium.CreateEventCallback<KeyEvent>(entity, "3D", [](Entity e, const KeyEvent& event) {
					if (event.action == 1 && KeyCode::Space == event.keyCode)
					{
						auto renderable = Helium.GetComponent<Renderable>(e);
						renderable->NextDrawingMode();
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
