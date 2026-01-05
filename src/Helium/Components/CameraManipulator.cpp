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

	if (event.button == 0)      isLButtonPressed = (event.action == 1);
	else if (event.button == 1) isRButtonPressed = (event.action == 1);
	else if (event.button == 2) isMButtonPressed = (event.action == 1);
}

void CameraManipulatorTrackball::OnMouseWheel(const MouseWheelEvent& event)
{
	if (nullptr == camera) return;

	bool isShiftPressed = (0 != pressedKeys.count(VK_SHIFT) || (0 != pressedKeys.count(VK_LSHIFT) || 0 != pressedKeys.count(VK_RSHIFT)));

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

	if ((key == 'W' || key == 'w') && event.action != 0)
	{
		eye += viewDir * moveStep;
		target += viewDir * moveStep;

		camera->SetEye(eye);
		camera->SetTarget(target);
	}
	else if ((key == 'S' || key == 's') && event.action != 0)
	{
		eye -= viewDir * moveStep;
		target -= viewDir * moveStep;

		camera->SetEye(eye);
		camera->SetTarget(target);
	}
	else if ((key == 'A' || key == 'a') && event.action != 0)
	{
		eye -= right * moveStep;
		target -= right * moveStep;

		camera->SetEye(eye);
		camera->SetTarget(target);
	}
	else if ((key == 'D' || key == 'd') && event.action != 0)
	{
		eye += right * moveStep;
		target += right * moveStep;

		camera->SetEye(eye);
		camera->SetTarget(target);
	}
	else if ((key == 'R' || key == 'r') && event.action == 1)
	{
		Reset();
	}
	else if ((key == 'P' || key == 'p') && event.action == 1)
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
	else if((VK_F9 == key) && (event.action == 0))
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
	else if ((VK_F10 == key) && (event.action == 0))
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
	else if ((VK_F11 == key) && (event.action == 0))
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
	else if ((VK_F12 == key) && (event.action == 0))
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