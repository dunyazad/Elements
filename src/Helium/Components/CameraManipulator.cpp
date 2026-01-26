#include "pch.h"
#include <Helium/Components/CameraManipulator.h>
#include <Helium/Components/Camera.h>
#include <Helium/HeliumCore.h>
#include <Helium/PointCloud.h>
#include <cmath>

#define NOMINMAX
#include <Windows.h>
#undef min
#undef max

void CameraManipulatorBase::LoadSettings()
{
	std::ifstream ifs("CameraManipulatorSettings.json");
	if (ifs.is_open())
	{
		ifs >> settings;
		ifs.close();
	}
}

void CameraManipulatorBase::SaveSettings()
{
	std::ofstream ofs("CameraManipulatorSettings.json");
	if (ofs.is_open())
	{
		ofs << settings.dump(4);
		ofs.close();
	}
}

json CameraManipulatorBase::GetSetting(const std::string& key, const std::string& subKey) const
{
	if (settings.contains(key))
	{
		return settings[key][subKey];
	}
	return nullptr;
}

void CameraManipulatorBase::StoreSetting(const std::string& key, const std::string& subKey)
{
	if (!camera) return;

	Eigen::Vector3f eye = camera->GetEye();
	Eigen::Vector3f target = camera->GetTarget();
	Eigen::Vector3f up = camera->GetUp();
	int projectionMode = static_cast<int>(camera->GetProjectionMode());

	json data;
	data["eye"] = { eye.x(), eye.y(), eye.z() };
	data["target"] = { target.x(), target.y(), target.z() };
	data["up"] = { up.x(), up.y(), up.z() };
	data["projectionMode"] = projectionMode;

	if (camera->GetProjectionMode() == Camera::Perspective)
	{
		data["fov"] = camera->GetPerspectiveSettings().GetFovy();
	}
	else
	{
		auto& ortho = camera->GetOrthogonalSettings();
		data["ortho"] = { ortho.GetLeft(), ortho.GetRight(), ortho.GetBottom(), ortho.GetTop() };
	}

	settings[key][subKey] = data;

	SaveSettings();
}

void CameraManipulatorBase::RestoreSetting(const std::string& key, const std::string& subKey)
{
	auto setting = GetSetting(key, subKey);
	if (false == setting.is_null())
	{
		if (!camera) return;

		Eigen::Vector3f eye, target, up;
		eye.x() = setting["eye"][0];
		eye.y() = setting["eye"][1];
		eye.z() = setting["eye"][2];

		target.x() = setting["target"][0];
		target.y() = setting["target"][1];
		target.z() = setting["target"][2];

		up.x() = setting["up"][0];
		up.y() = setting["up"][1];
		up.z() = setting["up"][2];

		auto projectionMode = static_cast<Camera::ProjectionMode>(setting["projectionMode"].get<int>());

		camera->SetEye(eye);
		camera->SetTarget(target);
		camera->SetUp(up);
		camera->SetProjectionMode(projectionMode);

		if (projectionMode == Camera::Perspective && setting.contains("fov"))
		{
			camera->GetPerspectiveSettings().SetFovy(setting["fov"].get<float>());
		}
		else if (projectionMode == Camera::Orthogonal && setting.contains("ortho"))
		{
			auto& ortho = camera->GetOrthogonalSettings();
			ortho.SetLeft(setting["ortho"][0]);
			ortho.SetRight(setting["ortho"][1]);
			ortho.SetBottom(setting["ortho"][2]);
			ortho.SetTop(setting["ortho"][3]);
		}

		camera->SetDirty(true);
		SyncRadius();
	}
}

Eigen::Vector3f CameraManipulatorTrackball::UnProject(const Eigen::Vector3f& winCoords, const Eigen::Matrix4f& view, const Eigen::Matrix4f& proj, const Eigen::Vector4f& viewport)
{
	// 1. Map to NDC (-1 ~ 1)
	Eigen::Vector4f tmp;
	tmp[0] = (winCoords[0] - viewport[0]) / viewport[2] * 2.0f - 1.0f;
	tmp[1] = (winCoords[1] - viewport[1]) / viewport[3] * 2.0f - 1.0f;
	tmp[2] = winCoords[2] * 2.0f - 1.0f;
	tmp[3] = 1.0f;

	// 2. Inverse Transform
	Eigen::Matrix4f inv = (proj * view).inverse();
	Eigen::Vector4f obj = inv * tmp;

	// 3. Perspective Divide
	if (obj[3] != 0.0f)
		obj /= obj[3];

	return Eigen::Vector3f(obj[0], obj[1], obj[2]);
}

CameraManipulatorBase::CameraManipulatorBase()
{
	LoadSettings();
}

CameraManipulatorBase::~CameraManipulatorBase()
{
	SaveSettings();
}

CameraManipulatorOrbit::CameraManipulatorOrbit() {}
CameraManipulatorOrbit::~CameraManipulatorOrbit() {}

CameraManipulatorTrackball::CameraManipulatorTrackball()
	: radius(50.0f),
	mouseSensitivity(0.005f),
	mousePanningSensitivity(0.01f),
	isLButtonPressed(false),
	isMButtonPressed(false),
	isRButtonPressed(false)
{
}

CameraManipulatorTrackball::~CameraManipulatorTrackball() {}

void CameraManipulatorTrackball::OnMousePosition(const MousePositionEvent& event)
{
	if (nullptr == camera) return;

	float dx = (float)(event.xpos - lastMousePositionX);
	float dy = (float)(event.ypos - lastMousePositionY);

	if (isRButtonPressed)
	{
		float angleYaw = -dx * mouseSensitivity;
		float anglePitch = -dy * mouseSensitivity;

		Eigen::Vector3f eye = camera->GetEye();
		Eigen::Vector3f target = camera->GetTarget();
		Eigen::Vector3f up0 = camera->GetUp().normalized();

		float dist = (eye - target).norm();

		Eigen::Vector3f viewDir = (orbitRot * Eigen::Vector3f(0, 0, 1)).normalized();

		Eigen::Vector3f right = up0.cross(viewDir).normalized();

		Eigen::Quaternionf qYaw(Eigen::AngleAxisf(angleYaw, up0));
		Eigen::Quaternionf qPitch(Eigen::AngleAxisf(anglePitch, right));

		Eigen::Vector3f testView = (qPitch * viewDir).normalized();

		float cosPitch = testView.dot(up0);
		if (std::abs(cosPitch) < 1.0f - minPitchCos)
		{
			orbitRot = (qYaw * qPitch) * orbitRot;
		}
		else
		{
			orbitRot = qYaw * orbitRot;
		}

		orbitRot.normalize();

		Eigen::Vector3f finalView = (orbitRot * Eigen::Vector3f(0, 0, 1)).normalized();

		Eigen::Vector3f finalRight = up0.cross(finalView).normalized();

		Eigen::Vector3f finalUp = finalView.cross(finalRight).normalized();

		Eigen::Vector3f newEye = target + finalView * dist;

		camera->SetEye(newEye);
		camera->SetUp(finalUp);
	}

	if (isMButtonPressed)
	{
		float panX = -dx * mousePanningSensitivity;
		float panY = dy * mousePanningSensitivity;

		Eigen::Matrix4f invView = camera->GetViewMatrix().inverse();

		Eigen::Vector3f screenRight = invView.col(0).head<3>().normalized();
		Eigen::Vector3f screenUp = invView.col(1).head<3>().normalized();

		Eigen::Vector3f eye = camera->GetEye();
		Eigen::Vector3f target = camera->GetTarget();

		this->radius = (eye - target).norm();

		Eigen::Vector3f offset = screenRight * panX * radius * mouseSensitivity * 10.0f +
			screenUp * panY * radius * mouseSensitivity * 10.0f;

		camera->SetEye(eye + offset);
		camera->SetTarget(target + offset);
	}

	lastMousePositionX = (float)event.xpos;
	lastMousePositionY = (float)event.ypos;
}

void CameraManipulatorTrackball::OnMouseButton(const MouseButtonEvent& event)
{
	if (nullptr == camera) return;

	if (event.button == MouseButton::Left)      isLButtonPressed = (event.action == 1);
	else if (event.button == MouseButton::Right) isRButtonPressed = (event.action == 1);
	else if (event.button == MouseButton::Middle) isMButtonPressed = (event.action == 1);
}

void CameraManipulatorTrackball::OnMouseWheel(const MouseWheelEvent& event)
{
	if (nullptr == camera) return;

	bool isShiftPressed = (0 != pressedKeys.count(KeyCode::Shift) || (0 != pressedKeys.count(KeyCode::LeftShift) || 0 != pressedKeys.count(KeyCode::RightShift)));

	if (camera->GetProjectionMode() == Camera::Perspective)
	{
		if (isShiftPressed)
		{
			auto& settings = camera->GetPerspectiveSettings();
			float fovyDeg = settings.GetFovy();

			//if (event.yoffset < 0) fovyDeg += 1.0f;
			//else if (event.yoffset > 0) fovyDeg -= 1.0f;

			fovyDeg += (float)event.yoffset;

			if (fovyDeg < 1.0f) fovyDeg = 1.0f;
			if (fovyDeg > 179.0f) fovyDeg = 179.0f;

			settings.SetFovy(fovyDeg);
			camera->SetDirty(true);

			InfoLog("FOV", "Camera FOV: %.2f\n", fovyDeg);
		}
		else
		{
			auto eye = camera->GetEye();
			auto target = camera->GetTarget();
			Eigen::Vector3f viewVec = eye - target;

			radius = viewVec.norm();

			if (event.yoffset < 0) radius *= 1.1f;
			else if (event.yoffset > 0) radius *= 0.9f;

			auto& settings = camera->GetPerspectiveSettings();
			if (radius < settings.GetZNear()) radius = settings.GetZNear();
			if (radius > settings.GetZFar()) radius = settings.GetZFar();

			Eigen::Vector3f viewDir = viewVec.normalized();
			camera->SetEye(target + viewDir * radius);
		}
	}
	else if (camera->GetProjectionMode() == Camera::Orthogonal)
	{
		float scaleFactor = (event.yoffset > 0) ? 0.9f : 1.1f;
		auto& ortho = camera->GetOrthogonalSettings();

		float currentWidth = ortho.GetRight() - ortho.GetLeft();
		if (currentWidth < 0.01f && scaleFactor < 1.0f) return;

		ortho.SetTop(ortho.GetTop() * scaleFactor);
		ortho.SetBottom(ortho.GetBottom() * scaleFactor);
		ortho.SetLeft(ortho.GetLeft() * scaleFactor);
		ortho.SetRight(ortho.GetRight() * scaleFactor);

		camera->SetDirty(true);
	}
}

void CameraManipulatorTrackball::OnKey(const KeyEvent& event)
{
	KeyCode key = event.keyCode;

	if (event.action == 1) pressedKeys.insert(key);
	else if (event.action == 0) pressedKeys.erase(key);

	if (nullptr == camera) return;

	Eigen::Vector3f eye = camera->GetEye();
	Eigen::Vector3f target = camera->GetTarget();
	Eigen::Vector3f up = camera->GetUp();

	Eigen::Vector3f viewDir = (target - eye).normalized();
	Eigen::Vector3f right = up.cross(viewDir).normalized();

	float moveStep = 0.2f;

	if ((key == KeyCode::W) && event.action != 0)
	{
		eye += viewDir * moveStep;
		target += viewDir * moveStep;

		camera->SetEye(eye);
		camera->SetTarget(target);
	}
	else if ((KeyCode::S == key) && event.action != 0)
	{
		eye -= viewDir * moveStep;
		target -= viewDir * moveStep;

		camera->SetEye(eye);
		camera->SetTarget(target);
	}
	else if ((KeyCode::A == key) && event.action != 0)
	{
		eye -= right * moveStep;
		target -= right * moveStep;

		camera->SetEye(eye);
		camera->SetTarget(target);
	}
	else if ((KeyCode::D == key) && event.action != 0)
	{
		eye += right * moveStep;
		target += right * moveStep;

		camera->SetEye(eye);
		camera->SetTarget(target);
	}
	else if ((KeyCode::R == key) && event.action == 1)
	{
		Reset();
	}
	else if ((key == KeyCode::P) && event.action == 1)
	{
		if (camera->GetProjectionMode() == Camera::Perspective)
		{
			float dist = (target - eye).norm();

			int width = HeliumCore::GetStaticInstance().GetWidth();
			int height = HeliumCore::GetStaticInstance().GetHeight();
			if (height == 0) height = 1;
			float aspect = (float)width / (float)height;

			float fovyDeg = camera->GetPerspectiveSettings().GetFovy();

			// 높이(h) = 2 * 거리 * tan(FOV / 2)
			float h = 2.0f * dist * std::tan((fovyDeg * DEG2RAD) * 0.5f);

			float w = h * aspect;

			auto& ortho = camera->GetOrthogonalSettings();
			ortho.SetTop(h * 0.5f);
			ortho.SetBottom(-h * 0.5f);
			ortho.SetRight(w * 0.5f);
			ortho.SetLeft(-w * 0.5f);

			float zRange = std::max(1000.0f, dist * 10.0f);
			ortho.SetZNear(-zRange);
			ortho.SetZFar(zRange);

			camera->SetProjectionMode(Camera::Orthogonal);
		}
		else
		{
			camera->SetProjectionMode(Camera::Perspective);
		}
		return;
	}
	else if((KeyCode::F9 == key) && (event.action == 0))
	{
		auto selectedPointCloud = Helium.GetSelectedPointCloud();
		if (nullptr == selectedPointCloud) return;

		auto key = selectedPointCloud->GetFileName();
		auto subKey = "F9";

		if (event.IsCtrlPressed())
		{
			StoreSetting(key, subKey);
		}
		else
		{
			RestoreSetting(key, subKey);
		}
	}
	else if ((KeyCode::F10 == key) && (event.action == 0))
	{
		auto selectedPointCloud = Helium.GetSelectedPointCloud();
		if (nullptr == selectedPointCloud) return;

		auto key = selectedPointCloud->GetFileName();
		auto subKey = "F10";

		if (event.IsCtrlPressed())
		{
			StoreSetting(key, subKey);
		}
		else
		{
			RestoreSetting(key, subKey);
		}
	}
	else if ((KeyCode::F11 == key) && (event.action == 0))
	{
		auto selectedPointCloud = Helium.GetSelectedPointCloud();
		if (nullptr == selectedPointCloud) return;

		auto key = selectedPointCloud->GetFileName();
		auto subKey = "F11";

		if (event.IsCtrlPressed())
		{
			StoreSetting(key, subKey);
		}
		else
		{
			RestoreSetting(key, subKey);
		}
	}
	else if ((KeyCode::F12 == key) && (event.action == 0))
	{
		auto selectedPointCloud = Helium.GetSelectedPointCloud();
		if (nullptr == selectedPointCloud) return;

		auto key = selectedPointCloud->GetFileName();
		auto subKey = "F12";

		if (event.IsCtrlPressed())
		{
			StoreSetting(key, subKey);
		}
		else
		{
			RestoreSetting(key, subKey);
		}
	}
}

void CameraManipulatorTrackball::PushCameraHistory()
{
	if (!camera) return;
	cameraHistory.push_back({ camera->GetEye(), camera->GetTarget(), camera->GetUp(), radius });
	JumpCameraHistory((int)cameraHistory.size() - 1);
}

void CameraManipulatorTrackball::PopCameraHistory()
{
	if (cameraHistory.size() <= 1) return;
	cameraHistory.pop_back();
	if (cameraHistoryIndex >= cameraHistory.size())
	{
		JumpCameraHistory((int)cameraHistory.size() - 1);
	}
}

void CameraManipulatorTrackball::JumpCameraHistory(int index)
{
	if (index < 0 || index >= (int)cameraHistory.size() || !camera) return;
	cameraHistoryIndex = index;

	auto [eye, target, up, savedRadius] = cameraHistory[cameraHistoryIndex];
	this->radius = savedRadius;

	camera->SetEye(eye);
	camera->SetTarget(target);
	camera->SetUp(up);
}

void CameraManipulatorTrackball::JumpToPreviousCameraHistory()
{
	if (cameraHistoryIndex > 0) JumpCameraHistory((int)cameraHistoryIndex - 1);
}

void CameraManipulatorTrackball::JumpToNextCameraHistory()
{
	if (cameraHistoryIndex < cameraHistory.size() - 1) JumpCameraHistory((int)cameraHistoryIndex + 1);
}

void CameraManipulatorTrackball::Reset()
{
	if (cameraHistory.empty() || !camera) return;

	auto [eye, target, up, savedRadius] = cameraHistory.front();
	this->radius = savedRadius;

	cameraHistory.clear();
	cameraHistory.push_back({ eye, target, up, savedRadius });
	cameraHistoryIndex = 0;

	camera->SetEye(eye);
	camera->SetTarget(target);
	camera->SetUp(up);
	camera->SetProjectionMode(Camera::Perspective);
	camera->GetPerspectiveSettings().SetFovy(45.0f);

	Eigen::Vector3f viewDir = (camera->GetEye() - camera->GetTarget()).normalized();
	orbitRot = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0, 0, 1), viewDir);
	orbitRot.normalize();
}

void CameraManipulatorTrackball::SyncRadius()
{
	if (!camera) return;
	this->radius = (camera->GetEye() - camera->GetTarget()).norm();

	Eigen::Vector3f viewDir = (camera->GetEye() - camera->GetTarget()).normalized();

	orbitRot = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0, 0, 1), viewDir);
	orbitRot.normalize();
}

void CameraManipulatorTrackball::MakeDefault()
{
	if (!camera) return;
	cameraHistory.clear();
	cameraHistory.push_back({ camera->GetEye(), camera->GetTarget(), camera->GetUp(), radius });
	cameraHistoryIndex = 0;
}

void CameraManipulatorTrackball::SetCamera(Camera* camera)
{
	this->camera = camera;
	if (camera) PushCameraHistory();

	Eigen::Vector3f viewDir = (camera->GetEye() - camera->GetTarget()).normalized();
	orbitRot = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0, 0, 1), viewDir);
	orbitRot.normalize();
}

void CameraManipulatorTrackball::SetCenter(const Eigen::Vector3f& center)
{
	if (!camera) return;

	auto delta = center - camera->GetTarget();
	auto newEye = camera->GetEye() + delta;
	camera->SetEye(newEye);
	camera->SetTarget(center);

	Eigen::Vector3f viewDir = (newEye - center).normalized();
	Eigen::Vector3f right = camera->GetUp().cross(viewDir).normalized();
	Eigen::Vector3f newUp = viewDir.cross(right).normalized(); // cross 순서 (Right x ViewDir = Up)

	camera->SetUp(newUp);
	camera->SetDirty(true);
}

void CameraManipulatorTrackball::SetCenterFromScreenPoint(float x, float y, float depth, int screenWidth, int screenHeight)
{
	if (!camera) return;

	Eigen::Matrix4f view = camera->GetViewMatrix();
	Eigen::Matrix4f proj = camera->GetProjectionMatrix();
	Eigen::Vector4f viewport(0.0f, 0.0f, (float)screenWidth, (float)screenHeight);

	Eigen::Vector3f winCoords(x, (float)screenHeight - y - 1.0f, depth);

	Eigen::Vector3f worldPos = UnProject(winCoords, view, proj, viewport);

	camera->SetTarget(worldPos);
	this->radius = (camera->GetEye() - worldPos).norm();
	camera->SetDirty(true);
}

CameraManipulator2DOrtho::CameraManipulator2DOrtho()
{
}

CameraManipulator2DOrtho::~CameraManipulator2DOrtho()
{
}

void CameraManipulator2DOrtho::Reset()
{
	if (!camera) return;

	// 1. 카메라를 Z축 상단에서 아래를 내려다보도록 설정 (2D 평면 뷰)
	// 드로잉 앱에서는 보통 XY 평면을 사용하므로 Eye(0,0,100), Target(0,0,0), Up(0,1,0)
	camera->SetEye(Eigen::Vector3f(0.0f, 0.0f, 100.0f));
	camera->SetTarget(Eigen::Vector3f(0.0f, 0.0f, 0.0f));
	camera->SetUp(Eigen::Vector3f(0.0f, 1.0f, 0.0f));

	// 2. Orthographic 모드 강제 설정
	camera->SetProjectionMode(Camera::Orthogonal);

	// 3. 기본 줌 영역 설정 (화면 비율에 맞춤)
	float width = (float)HeliumCore::GetStaticInstance().GetWidth();
	float height = (float)HeliumCore::GetStaticInstance().GetHeight();
	if (height < 1.0f) height = 1.0f;

	// 화면 해상도의 절반 크기로 초기 월드 영역 설정 (예시)
	float halfW = width * 0.5f;
	float halfH = height * 0.5f;

	auto& ortho = camera->GetOrthogonalSettings();
	ortho.SetLeft(-halfW);
	ortho.SetRight(halfW);
	ortho.SetBottom(-halfH);
	ortho.SetTop(halfH);
	ortho.SetZNear(-1000.0f); // 2D 레이어 깊이 확보
	ortho.SetZFar(1000.0f);

	camera->SetDirty(true);
}

void CameraManipulator2DOrtho::OnMouseButton(const MouseButtonEvent& event)
{
	// 휠 버튼(Middle) 또는 우클릭(Right)을 Panning으로 사용
	if (event.button == MouseButton::Middle || event.button == MouseButton::Right)
	{
		isPanning = (event.action == 1); // Press=1, Release=0
	}
}

void CameraManipulator2DOrtho::OnMousePosition(const MousePositionEvent& event)
{
	if (!camera) return;

	double currentX = event.xpos;
	double currentY = event.ypos;

	if (isPanning)
	{
		// 1. 마우스 델타 계산 (스크린 픽셀 단위)
		double dx = currentX - lastMouseX;
		double dy = currentY - lastMouseY;

		// 2. 현재 뷰포트 및 Ortho 설정 가져오기
		int screenW = HeliumCore::GetStaticInstance().GetWidth();
		int screenH = HeliumCore::GetStaticInstance().GetHeight();

		auto& ortho = camera->GetOrthogonalSettings();
		float worldWidth = ortho.GetRight() - ortho.GetLeft();
		float worldHeight = ortho.GetTop() - ortho.GetBottom();

		// 3. 픽셀 당 월드 좌표 비율 계산
		// 마우스를 오른쪽으로 드래그하면 카메라는 왼쪽으로 이동해야 화면이 따라옴 (반대 방향)
		float pixelToWorldX = worldWidth / (float)screenW;
		float pixelToWorldY = worldHeight / (float)screenH;

		// 4. 이동 벡터 계산 (카메라 Right, Up 벡터 기준)
		// 2D Ortho라 해도 회전이 들어갈 수 있으므로 카메라의 Local Axis를 사용
		Eigen::Matrix4f view = camera->GetViewMatrix();
		// View Matrix의 역행렬에서 Right(col 0), Up(col 1) 추출
		Eigen::Matrix4f invView = view.inverse();
		Eigen::Vector3f camRight = invView.col(0).head<3>().normalized();
		Eigen::Vector3f camUp = invView.col(1).head<3>().normalized();

		// dx가 양수(마우스 우측 이동) -> 카메라는 좌측 이동 -> -dx
		// dy가 양수(마우스 하단 이동, 윈도우 좌표계) -> 카메라는 상단 이동 -> +dy (OpenGL 좌표계 고려)
		// * 주의: GLFW/Window 좌표계에서 Y는 아래로 증가하지만, 
		//   카메라 이동 시 마우스를 내리면 화면이 올라가야 하므로(카메라는 내려감) 
		//   좌표계 변환에 유의. 보통 Panning은 마우스 델타의 반대방향으로 Eye/Target 이동.

		Eigen::Vector3f translation = -camRight * (float)dx * pixelToWorldX
			+ camUp * (float)dy * pixelToWorldY;

		camera->SetEye(camera->GetEye() + translation);
		camera->SetTarget(camera->GetTarget() + translation);
		camera->SetDirty(true);
	}

	lastMouseX = currentX;
	lastMouseY = currentY;
}

void CameraManipulator2DOrtho::OnMouseWheel(const MouseWheelEvent& event)
{
	if (!camera) return;
	if (camera->GetProjectionMode() != Camera::Orthogonal) return;

	// 1. 줌 스케일 결정 (Wheel Up: 확대(0.9), Wheel Down: 축소(1.1))
	float scaleFactor = (event.yoffset > 0) ? 0.9f : 1.1f;

	auto& ortho = camera->GetOrthogonalSettings();

	// 최소/최대 줌 제한 (너무 작아지거나 커지는 것 방지)
	float currentW = ortho.GetRight() - ortho.GetLeft();
	if (currentW < 0.1f && scaleFactor < 1.0f) return; // 너무 확대됨
	if (currentW > 100000.0f && scaleFactor > 1.0f) return; // 너무 축소됨

	// 2. 마우스 커서 위치 기준으로 줌 (Zoom to Cursor) 구현
	// 현재 마우스 위치가 월드 좌표계의 어디인지 비율(Ratio) 계산
	int screenW = HeliumCore::GetStaticInstance().GetWidth();
	int screenH = HeliumCore::GetStaticInstance().GetHeight();

	// 마우스 포인터의 화면상 비율 (0.0 ~ 1.0)
	// Y좌표는 윈도우(Top-Left 0,0)와 OpenGL(Bottom-Left 0,0) 차이 고려
	float ratioX = (float)lastMouseX / (float)screenW;
	float ratioY = 1.0f - ((float)lastMouseY / (float)screenH);

	float left = ortho.GetLeft();
	float right = ortho.GetRight();
	float bottom = ortho.GetBottom();
	float top = ortho.GetTop();

	float w = right - left;
	float h = top - bottom;

	// 마우스가 가리키는 현재 월드 좌표 (투영 전의 논리적 위치)
	float mouseWorldX = left + w * ratioX;
	float mouseWorldY = bottom + h * ratioY;

	// 3. 새로운 너비/높이 계산
	float newW = w * scaleFactor;
	float newH = h * scaleFactor;

	// 4. 마우스 월드 좌표가 줌 후에도 같은 화면 비율 위치에 오도록 Bounds 재설정
	// mouseWorldX = newLeft + newW * ratioX
	// => newLeft = mouseWorldX - newW * ratioX
	float newLeft = mouseWorldX - (newW * ratioX);
	float newRight = newLeft + newW;

	float newBottom = mouseWorldY - (newH * ratioY);
	float newTop = newBottom + newH;

	ortho.SetLeft(newLeft);
	ortho.SetRight(newRight);
	ortho.SetBottom(newBottom);
	ortho.SetTop(newTop);

	camera->SetDirty(true);
}

void CameraManipulator2DOrtho::OnKey(const KeyEvent& event)
{
	// 필요 시 키보드 단축키 추가 (예: F 키를 누르면 원점 복귀 등)
	if (event.keyCode == KeyCode::F && event.action == 1)
	{
		Reset();
	}
}
