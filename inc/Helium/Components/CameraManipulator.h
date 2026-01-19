#pragma once

#pragma warning(disable: 4251)

#include <vector>
#include <set>
#include <unordered_set>
#include <tuple>
#include <Eigen/Dense>

#include <Helium/HeliumEvents.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

class Camera;

class HELIUM_API CameraManipulatorBase
{
public:
	CameraManipulatorBase();
	virtual ~CameraManipulatorBase();

	virtual void PushCameraHistory() = 0;
	virtual void PopCameraHistory() = 0;
	virtual void JumpCameraHistory(int index) = 0;
	virtual void JumpToPreviousCameraHistory() = 0;
	virtual void JumpToNextCameraHistory() = 0;
	virtual void Reset() = 0;
	virtual void SyncRadius() = 0;

	inline Camera* GetCamera() const { return camera; }
	inline void SetCamera(Camera* camera) { this->camera = camera; }

	void LoadSettings();
	void SaveSettings();
	json GetSetting(const std::string& key, const std::string& subKey) const;
	void StoreSetting(const std::string& key, const std::string& subKey);
	void RestoreSetting(const std::string& key, const std::string& subKey);

protected:
	Camera* camera = nullptr;

	json settings;
};

class HELIUM_API CameraManipulatorOrbit : public CameraManipulatorBase
{
public:
	CameraManipulatorOrbit();
	virtual ~CameraManipulatorOrbit();

	virtual void PushCameraHistory() override {}
	virtual void PopCameraHistory() override {}
	virtual void JumpCameraHistory(int index) override {}
	virtual void JumpToPreviousCameraHistory() override {}
	virtual void JumpToNextCameraHistory() override {}
	virtual void Reset() override {}
	virtual void SyncRadius() override {};

	inline float GetAzimuth() { return azimuth; }
	inline void SetAzimuth(float azimuth) { this->azimuth = azimuth; }
	inline float GetElevation() { return elevation; }
	inline void SetElevation(float elevation) { this->elevation = elevation; }
	inline float GetRadius() { return radius; }
	inline void SetRadius(float radius) { this->radius = radius; }
	inline float GetMouseSensitivity() { return mouseSensitivity; }
	inline void SetMouseSensitivity(float mouseSensitivity) { this->mouseSensitivity = mouseSensitivity; }
	inline float GetMouseWheelSensitivity() { return mouseWheelSensitivity; }
	inline void SetMouseWheelSensitivity(float mouseWheelSensitivity) { this->mouseWheelSensitivity = mouseWheelSensitivity; }

protected:
	std::set<int> pressedKeys;

	double lastMousePositionX = 0.0;
	double lastMousePositionY = 0.0;

	bool isLButtonPressed = false;
	bool isMButtonPressed = false;
	bool isRButtonPressed = false;

	float azimuth = 0.0f;
	float elevation = 0.0f;
	float radius = 50.0f;
	float mouseSensitivity = 0.2f;
	float mousePanningSensitivity = 0.1f;
	float mouseWheelSensitivity = 0.5f;
};

class HELIUM_API CameraManipulatorTrackball : public CameraManipulatorBase
{
public:
	CameraManipulatorTrackball();
	virtual ~CameraManipulatorTrackball();

	inline float GetRadius() { return radius; }
	inline void SetRadius(float radius) { this->radius = radius; }
	inline float GetMouseSensitivity() { return mouseSensitivity; }
	inline void SetMouseSensitivity(float mouseSensitivity) { this->mouseSensitivity = mouseSensitivity; }
	inline float GetMouseWheelSensitivity() { return mouseWheelSensitivity; }
	inline void SetMouseWheelSensitivity(float mouseWheelSensitivity) { this->mouseWheelSensitivity = mouseWheelSensitivity; }

	void OnMousePosition(const MousePositionEvent& event);
	void OnMouseButton(const MouseButtonEvent& event);
	void OnMouseWheel(const MouseWheelEvent& event);
	void OnKey(const KeyEvent& event);

	virtual void PushCameraHistory() override;
	virtual void PopCameraHistory() override;
	virtual void JumpCameraHistory(int index) override;
	virtual void JumpToPreviousCameraHistory() override;
	virtual void JumpToNextCameraHistory() override;
	virtual void Reset() override;
	virtual void SyncRadius() override;
	void MakeDefault();

	void SetCamera(Camera* camera);

	void SetCenter(const Eigen::Vector3f& center);
	void SetCenterFromScreenPoint(float x, float y, float depth, int screenWidth, int screenHeight);

private:
	Eigen::Vector3f UnProject(const Eigen::Vector3f& winCoords, const Eigen::Matrix4f& view, const Eigen::Matrix4f& proj, const Eigen::Vector4f& viewport);

private:
	float lastMousePositionX = 0.0f;
	float lastMousePositionY = 0.0f;

	bool isLButtonPressed = false;
	bool isMButtonPressed = false;
	bool isRButtonPressed = false;

	float radius = 50.0f;
	float mouseSensitivity = 0.005f;
	float mousePanningSensitivity = 0.01f;
	float mouseWheelSensitivity = 0.5f;

	Eigen::Quaternionf cameraRotation = Eigen::Quaternionf::Identity();

	std::unordered_set<KeyCode> pressedKeys;

	std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector3f, float>> cameraHistory;
	size_t cameraHistoryIndex = 0;

	Eigen::Quaternionf orbitRot = Eigen::Quaternionf::Identity();
	float minPitchCos = 0.01f;
};
