#include "pch.h"
#include <Helium/Components/CameraManipulator.h>
#include <Helium/Components/Camera.h>
#include <Helium/HeliumCore.h>
#include <cmath>

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

CameraManipulatorBase::CameraManipulatorBase() {}
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
		float angleX = -dx * mouseSensitivity;
		float angleY = -dy * mouseSensitivity;

		Eigen::Vector3f eye = camera->GetEye();
		Eigen::Vector3f target = camera->GetTarget();
		Eigen::Vector3f up = camera->GetUp();

		Eigen::Vector3f viewDir = (eye - target).normalized();
		Eigen::Vector3f right = up.cross(viewDir).normalized();

		Eigen::Quaternionf rotX = Eigen::Quaternionf(Eigen::AngleAxisf(angleX, up));
		Eigen::Quaternionf rotY = Eigen::Quaternionf(Eigen::AngleAxisf(angleY, right));

		Eigen::Quaternionf rot = rotX * rotY;

		Eigen::Vector3f rotatedViewDir = (rot * viewDir).normalized();
		Eigen::Vector3f rotatedUp = (rot * up).normalized();

		float currentDist = (eye - target).norm();
		Eigen::Vector3f newEye = target + rotatedViewDir * currentDist;

		camera->SetEye(newEye);
		camera->SetUp(rotatedUp);
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

	if (event.button == 0)      isLButtonPressed = (event.action == 1);
	else if (event.button == 1) isRButtonPressed = (event.action == 1);
	else if (event.button == 2) isMButtonPressed = (event.action == 1);
}

void CameraManipulatorTrackball::OnMouseWheel(const MouseWheelEvent& event)
{
	if (nullptr == camera) return;

	bool isShiftPressed = (0 != pressedKeys.count(GLFW_KEY_LEFT_SHIFT) || 0 != pressedKeys.count(GLFW_KEY_RIGHT_SHIFT));

	if (camera->GetProjectionMode() == Camera::Perspective)
	{
		if (isShiftPressed)
		{
			auto& settings = camera->GetPerspectiveSettings();
			float fovyDeg = settings.GetFovy() * RAD2DEG;

			if (event.yoffset < 0) fovyDeg += 1.0f;
			else if (event.yoffset > 0) fovyDeg -= 1.0f;

			if (fovyDeg < 1.0f) fovyDeg = 1.0f;
			if (fovyDeg > 179.0f) fovyDeg = 179.0f;

			settings.SetFovy(fovyDeg * DEG2RAD);
			camera->SetDirty(true);
		}
		else
		{
			// [FIX] 현재 카메라의 실제 위치를 기반으로 radius 동기화
			auto eye = camera->GetEye();
			auto target = camera->GetTarget();
			Eigen::Vector3f viewVec = eye - target;

			// 현재 거리를 radius로 갱신
			radius = viewVec.norm();

			if (event.yoffset < 0) radius *= 1.1f;
			else if (event.yoffset > 0) radius *= 0.9f;

			auto& settings = camera->GetPerspectiveSettings();
			if (radius < settings.GetZNear()) radius = settings.GetZNear();
			if (radius > settings.GetZFar()) radius = settings.GetZFar();

			Eigen::Vector3f viewDir = viewVec.normalized();
			camera->SetEye(target + viewDir * radius);

			He_Log(HE_LOG_DEBUG, "", "Camera radius: %f", radius);
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
	int key = event.keyCode;

	if (event.action == 1) pressedKeys.insert(key);
	else if (event.action == 0) pressedKeys.erase(key);

	if (nullptr == camera) return;

	Eigen::Vector3f eye = camera->GetEye();
	Eigen::Vector3f target = camera->GetTarget();
	Eigen::Vector3f up = camera->GetUp();

	Eigen::Vector3f viewDir = (target - eye).normalized();
	Eigen::Vector3f right = up.cross(viewDir).normalized();

	float moveStep = 0.2f;

	if (key == GLFW_KEY_W && event.action != 0)
	{
		eye += viewDir * moveStep;
		target += viewDir * moveStep;
	}
	else if (key == GLFW_KEY_S && event.action != 0)
	{
		eye -= viewDir * moveStep;
		target -= viewDir * moveStep;
	}
	else if (key == GLFW_KEY_A && event.action != 0)
	{
		eye -= right * moveStep;
		target -= right * moveStep;
	}
	else if (key == GLFW_KEY_D && event.action != 0)
	{
		eye += right * moveStep;
		target += right * moveStep;
	}
	else if (key == GLFW_KEY_R && event.action == 1)
	{
		Reset();
		return;
	}
	else if (key == GLFW_KEY_P && event.action == 1)
	{
		// Perspective <-> Orthogonal 전환
		if (camera->GetProjectionMode() == Camera::Perspective)
		{
			float dist = (target - eye).norm();

			int width = HeliumCore::GetStaticInstance().GetWidth();
			int height = HeliumCore::GetStaticInstance().GetHeight();
			if (height == 0) height = 1;
			float aspect = (float)width / (float)height;

			float fovy = camera->GetPerspectiveSettings().GetFovy();
			float h = 2.0f * dist * std::tan(fovy * 0.5f);
			float w = h * aspect;

			auto& ortho = camera->GetOrthogonalSettings();
			ortho.SetTop(h * 0.5f);
			ortho.SetBottom(-h * 0.5f);
			ortho.SetRight(w * 0.5f);
			ortho.SetLeft(-w * 0.5f);

			camera->SetProjectionMode(Camera::Orthogonal);
		}
		else
		{
			camera->SetProjectionMode(Camera::Perspective);
		}
		return;
	}

	camera->SetEye(eye);
	camera->SetTarget(target);
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
}

void CameraManipulatorTrackball::SyncRadius()
{
	if (!camera) return;
	this->radius = (camera->GetEye() - camera->GetTarget()).norm();
}

void CameraManipulatorTrackball::MakeDefault()
{
	if (!camera) return;
	cameraHistory.clear();
	cameraHistory.push_back({ camera->GetEye(), camera->GetTarget(), camera->GetUp(), radius });
	cameraHistoryIndex = 0;
}

void CameraManipulatorTrackball::SetCenter(const Eigen::Vector3f& center)
{
	if (!camera) return;

	auto delta = center - camera->GetTarget();
	auto newEye = camera->GetEye() + delta;
	camera->SetEye(newEye);
	camera->SetTarget(center);

	// Up 벡터 재계산 (Eigen)
	Eigen::Vector3f viewDir = (newEye - center).normalized();
	Eigen::Vector3f right = camera->GetUp().cross(viewDir).normalized();
	Eigen::Vector3f newUp = viewDir.cross(right).normalized(); // cross 순서 주의 (Right x ViewDir = Up)

	// 위 공식이 헷갈린다면, Gram-Schmidt 직교화와 유사하게:
	// newUp = (right x (newEye - center)).normalized()
	newUp = right.cross((newEye - center).normalized()).normalized();

	camera->SetUp(newUp);
	camera->SetDirty(true);
}

void CameraManipulatorTrackball::SetCenterFromScreenPoint(float x, float y, float depth, int screenWidth, int screenHeight)
{
	if (!camera) return;

	Eigen::Matrix4f view = camera->GetViewMatrix();
	Eigen::Matrix4f proj = camera->GetProjectionMatrix();
	Eigen::Vector4f viewport(0.0f, 0.0f, (float)screenWidth, (float)screenHeight);

	// Y축 뒤집기 (화면 좌표계 -> GL 좌표계)
	Eigen::Vector3f winCoords(x, (float)screenHeight - y - 1.0f, depth);

	// 직접 구현한 UnProject 사용
	Eigen::Vector3f worldPos = UnProject(winCoords, view, proj, viewport);

	camera->SetTarget(worldPos);
	this->radius = (camera->GetEye() - worldPos).norm();
	camera->SetDirty(true);
}