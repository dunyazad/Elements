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
	
	if("000B0671BFEBFBFF" != cpuId)
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

	CuPointCloud pointCloud;
	CuSparseDataBlock sparseDataBlock;

#if 0
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
#endif // 0

	std::thread thread([&]() {
			TS(Loading);
			PLYFormat ply;
			ply.Deserialize("D:\\Temp\\PLY\\DensityEstimation\\Model.ply");
			TE(Loading);

			TS(MakePointCloud);

			//CuPointCloud pointCloud;
			pointCloud.FromHostPointers(
				(float3*)ply.GetPoints().data(),
				(float3*)ply.GetNormals().data(),
				(float4*)ply.GetColors().data(),
				ply.GetPoints().size()
			);

			TE(MakePointCloud);

			TS(Build);

			sparseDataBlock.Build(&pointCloud);

			TE(Build);

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
					/////////////////////////////////VD::AddDisk("LDE_LowDensityPoints", { p.x, p.y, p.z }, { n.x, n.y, n.z }, 0.1f, 16, Color::red(), true);

					lowDensityPoints.push_back(p);
					lowDensityNormals.push_back(n);
					lowDensityColors.push_back(c);
				}
				else
				{
					VD::AddDisk("LDE", { p.x, p.y, p.z }, { n.x, n.y, n.z }, 0.05f, 16, { (float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f, 1.0f }, true);
				}
			}

			TS(BuildArrowBlocks);
			CuPointCloud lowDensityPointCloud;
			lowDensityPointCloud.FromHostVectors(lowDensityPoints, lowDensityNormals, lowDensityColors);

			CuSparseDataBlock sparseDataBlockForLowDensityPointCloud;
			sparseDataBlockForLowDensityPointCloud.Build(&lowDensityPointCloud, 10.0f);
			TE(BuildArrowBlocks);

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

			CuOperatorPointCloudClustering clusteringOp;
			CuOperatorParameters clusteringParams;

			clusteringParams.SetParameter<CuPointCloud*>("pointCloud", &lowDensityPointCloud);

			clusteringParams.SetParameter<CuSparseDataBlock*>("sparseDataBlock", &sparseDataBlockForLowDensityPointCloud);
			clusteringParams.SetParameter<float>("radius", 3.0f);
			clusteringParams.SetParameter<int>("minClusterSize", 10);
			clusteringParams.SetParameter<int>("maxClusterSize", 2000000000);

			std::vector<uint64_t> clusterLabels;

			TS(Clustering);
			clusteringOp.Execute(clusteringParams, clusterLabels);
			TE(Clustering);

			thrust::host_vector<float3> h_lowDensityPoints = lowDensityPointCloud.points;
			thrust::host_vector<float3> h_lowDensityNormals = lowDensityPointCloud.normals;
			thrust::host_vector<uchar3> h_lowDensityColors = lowDensityPointCloud.colors;

			auto colors = Color::GetContrastingColorsWithoutBWRGB(100);
			for (size_t i = 0; i < clusterLabels.size(); i++)
			{
				const auto& p = h_lowDensityPoints[i];
				const auto& n = h_lowDensityNormals[i];
				auto c = h_lowDensityColors[i];
				uint64_t label = clusterLabels[i];
				c = { (unsigned char)(colors[label % colors.size()].x() * 255.0f),
					  (unsigned char)(colors[label % colors.size()].y() * 255.0f),
					  (unsigned char)(colors[label % colors.size()].z() * 255.0f) };

				VD::AddDisk(
					"LDE_ClusteredLowDensityPoints",
					{ p.x, p.y, p.z },
					{ n.x, n.y, n.z },
					0.1f,
					16,
					{ (float)c.x / 255.0f, (float)c.y / 255.0f, (float)c.z / 255.0f, 1.0f },
					true);

				//printf("Point %zu: Cluster %llu\n", i, label);
			}
		});

#if 0
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
#endif // 0


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

						PickResult pickResult = pointCloud.Pick(
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
						}
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
			auto entity = Helium.CreateEntity("3DTextBooleanExample");
			auto renderable = Helium.CreateComponent<Renderable>(entity);
			renderable->Initialize(Renderable::Triangles);
			renderable->AddShader(Helium.CreateShader("Default", File("../../res/Shaders/Default.vs"), File("../../res/Shaders/Default.fs")));

			// 1. Generator 초기화
			TextMeshGenerator generator;
			std::string fontPath = "C:\\Windows\\Fonts\\arial.ttf";
			//std::string fontPath = "../../res/Fonts/NanumGothic/NanumGothic.ttf";
			if (!generator.LoadFont(fontPath)) {
				printf("!!! [ERROR] Font Load Failed: %s\n", fontPath.c_str());
				return;
			}

			float targetScale = 0.1f; // 0.05f -> 1.0f 변경 테스트
			float depth = 20.0f;

			// 2. 3D 텍스트 생성
			printf(">>> [DEBUG] Generating Text Mesh ('Hydrogen')...\n");
			auto textManifold = generator.Create3DText("Hydrogen", depth, targetScale);

			// [중요 확인 포인트 1] 텍스트 자체가 생성되었는가?
			int textVerts = textManifold.NumVert();
			printf(">>> [CHECK 1] Text Manifold Verts: %d\n", textVerts);

			if (textVerts == 0) {
				printf("!!! [CRITICAL] Text Mesh is EMPTY! (Check Scale or Font)\n");
				// 빈 메쉬라도 죽지 않게 빈 Manifold 유지
			}

			// 중앙 정렬
			auto bbox = textManifold.BoundingBox();
			float textCenterX = (bbox.max.x + bbox.min.x) * 0.5f;
			float textCenterY = (bbox.max.y + bbox.min.y) * 0.5f;

			// 큐브 앞면으로 이동 (Z = 25.0f 근처)
			textManifold = textManifold.Translate({ -textCenterX, -textCenterY, 50.0f });

			// 3. 타겟 큐브 생성
			manifold::Manifold cubeManifold = manifold::Manifold::Cube({ 200.0f, 200.0f, 100.0f }, true);
			printf(">>> [CHECK 2] Cube Verts: %d\n", cubeManifold.NumVert());

			// 4. Boolean 연산 (합집합)
			printf(">>> [DEBUG] Running Boolean Operation...\n");
			manifold::Manifold resultManifold = cubeManifold - textManifold;

			printf(">>> [CHECK 3] Result Manifold Verts: %d\n", resultManifold.NumVert());

			// 5. 결과를 MeshGL로 변환
			manifold::MeshGL resultMesh = resultManifold.GetMeshGL();
			int triCount = (int)resultMesh.triVerts.size() / 3;
			printf(">>> [CHECK 4] Result Mesh Triangles: %d\n", triCount);

			if (triCount == 0) {
				printf("!!! [CRITICAL] Final Mesh has 0 Triangles! Loop will NOT run.\n");
			}
			else {
				// -------------------------------------------------------------------------
				// Flat Shading 루프
				// -------------------------------------------------------------------------
				std::vector<Eigen::Vector3f> flatPositions;
				std::vector<Eigen::Vector3f> flatNormals;
				std::vector<Eigen::Vector4f> flatColors;
				std::vector<unsigned int> flatIndices;

				auto GetPos = [&](int idx) -> Eigen::Vector3f {
					return Eigen::Vector3f(
						resultMesh.vertProperties[idx * 3 + 0],
						resultMesh.vertProperties[idx * 3 + 1],
						resultMesh.vertProperties[idx * 3 + 2]
					);
					};

				printf(">>> [DEBUG] Starting Vertex Loop...\n");

				for (size_t i = 0; i < resultMesh.triVerts.size(); i += 3)
				{
					unsigned int i0 = resultMesh.triVerts[i + 0];
					unsigned int i1 = resultMesh.triVerts[i + 1];
					unsigned int i2 = resultMesh.triVerts[i + 2];

					Eigen::Vector3f v0 = GetPos(i0);
					Eigen::Vector3f v1 = GetPos(i1);
					Eigen::Vector3f v2 = GetPos(i2);

					// 첫 번째 삼각형만 출력
					if (i == 0) {
						printf(">>> [LOOP RUNNING] V0: %f, %f, %f\n", v0.x(), v0.y(), v0.z());
					}

					Eigen::Vector3f faceNormal = (v1 - v0).cross(v2 - v0).normalized();

					flatPositions.push_back(v0);
					flatPositions.push_back(v1);
					flatPositions.push_back(v2);

					flatNormals.push_back(faceNormal);
					flatNormals.push_back(faceNormal);
					flatNormals.push_back(faceNormal);

					flatColors.push_back({ 1.0f, 1.0f, 1.0f, 1.0f });
					flatColors.push_back({ 1.0f, 1.0f, 1.0f, 1.0f });
					flatColors.push_back({ 1.0f, 1.0f, 1.0f, 1.0f });

					unsigned int currentIdx = (unsigned int)flatIndices.size();
					flatIndices.push_back(currentIdx + 0);
					flatIndices.push_back(currentIdx + 1);
					flatIndices.push_back(currentIdx + 2);

					// 디버깅: 선 그리기 (필요하면 주석 해제)
					// VD::AddLine("Normal", v0, v0 + faceNormal * 5.0f, Color::red());
				}

				printf(">>> [DEBUG] Loop Finished. Total Verts Generated: %d\n", (int)flatPositions.size());

				// 6. Renderable에 데이터 설정
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
