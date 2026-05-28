#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#define BUILD_AppTSDFDevice

#define _SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS // 추가

// Windows 관련 타입(HWND, UINT 등)을 위해 반드시 포함
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include <iostream>
#include <map>
#include <string>

//#include <Helium/Helium.h>
//#include <Helium/HeliumLog.h>
//#include <Helium/HeliumCore.h>
//#include <Helium/Serialization.hpp>
//#include <Helium/DeviceInformation.h>
//
//#include <Helium/Components/GUI/GUIComponent.h>
//
//#include <Helium/VisualDebugging.h>
//using VD = VisualDebugging;

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <nvapi.h>
#include <NvApiDriverSettings.h>

#pragma comment(lib, "nvapi64.lib")

//#include <Copper/Copper.h>
//#include <Copper/CuPointCloud.h>
//#include <Copper/CuSparseCells.h>
//#include <Copper/OperatorCollection/CuOperatorCollection.h>
//#include <Copper/CuVoxelStreaming.h>
//
//#include <robin_hood/robin_hood.h>

//
//#include <Eigen/Core>
//#include <Eigen/Dense>


namespace Eigen
{
    template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
    class Matrix;

    typedef Matrix<float, 4, 4, 0, 4, 4> Matrix4f;
    typedef Matrix<float, 3, 1, 0, 3, 1> Vector3f;
    typedef Matrix<uint8_t, 3, 1, 0, 3, 1> Vector3b;
    typedef Matrix<int32_t, 3, 1, 0, 3, 1> Vector3i;
    typedef Matrix<uint32_t, 3, 1, 0, 3, 1> Vector3ui;
}

class NvDriverSetting {
public:
	NvDriverSetting() {
		m_previous_gpu_performance = PREFERRED_PSTATE_DEFAULT;
	}

	~NvDriverSetting() {
	}

	bool forceGPUPerformance()
	{
		NvAPI_Status status;

		NvDRSSessionHandle hSession = 0;
		status = NvAPI_DRS_CreateSession(&hSession);
		if (status != NVAPI_OK)
			return false;

		// (2) load all the system settings into the session
		status = NvAPI_DRS_LoadSettings(hSession);
		if (status != NVAPI_OK)
			return false;

		NvDRSProfileHandle hProfile = 0;
		status = NvAPI_DRS_GetBaseProfile(hSession, &hProfile);
		if (status != NVAPI_OK)
			return false;

		NVDRS_SETTING drsGet = { 0, };
		drsGet.version = NVDRS_SETTING_VER;
		status = NvAPI_DRS_GetSetting(hSession, hProfile, PREFERRED_PSTATE_ID, &drsGet);
		if (status != NVAPI_OK)
			return false;

		if (drsGet.u32CurrentValue == PREFERRED_PSTATE_PREFER_MAX) {
			m_previous_gpu_performance = PREFERRED_PSTATE_PREFER_MAX;
		}
		else {
			NVDRS_SETTING drsSetting = { 0, };
			drsSetting.version = NVDRS_SETTING_VER;
			drsSetting.settingId = PREFERRED_PSTATE_ID;
			drsSetting.settingType = NVDRS_DWORD_TYPE;
			drsSetting.u32CurrentValue = PREFERRED_PSTATE_PREFER_MAX;
			m_previous_gpu_performance = PREFERRED_PSTATE_PREFER_MAX;

			status = NvAPI_DRS_SetSetting(hSession, hProfile, &drsSetting);
			if (status != NVAPI_OK)
				return false;

			status = NvAPI_DRS_SaveSettings(hSession);
			if (status != NVAPI_OK)
				return false;
		}

		// (6) We clean up. This is analogous to doing a free()
		NvAPI_DRS_DestroySession(hSession);
		hSession = 0;

		return true;
	}

	bool restoreGPUPerformance()
	{
		NvAPI_Status status;

		NvDRSSessionHandle hSession = 0;
		status = NvAPI_DRS_CreateSession(&hSession);
		if (status != NVAPI_OK)
			return false;

		// (2) load all the system settings into the session
		status = NvAPI_DRS_LoadSettings(hSession);
		if (status != NVAPI_OK)
			return false;

		NvDRSProfileHandle hProfile = 0;
		status = NvAPI_DRS_GetBaseProfile(hSession, &hProfile);
		if (status != NVAPI_OK)
			return false;

		NVDRS_SETTING drsGet = { 0, };
		drsGet.version = NVDRS_SETTING_VER;
		status = NvAPI_DRS_GetSetting(hSession, hProfile, PREFERRED_PSTATE_ID, &drsGet);
		if (status != NVAPI_OK)
			return false;

		if (drsGet.u32CurrentValue != m_previous_gpu_performance) {
			NVDRS_SETTING drsSetting = { 0, };
			drsSetting.version = NVDRS_SETTING_VER;
			drsSetting.settingId = PREFERRED_PSTATE_ID;
			drsSetting.settingType = NVDRS_DWORD_TYPE;
			drsSetting.u32CurrentValue = m_previous_gpu_performance;

			status = NvAPI_DRS_SetSetting(hSession, hProfile, &drsSetting);
			if (status != NVAPI_OK)
				return false;

			status = NvAPI_DRS_SaveSettings(hSession);
			if (status != NVAPI_OK)
				return false;
		}

		// (6) We clean up. This is analogous to doing a free()
		NvAPI_DRS_DestroySession(hSession);
		hSession = 0;

		return true;
	}

protected:
    unsigned long	m_previous_gpu_performance;
};

class App
{
public:
    virtual ~App()
    {
    }

    virtual void Execute() = 0;

protected:
	NvDriverSetting nvDriverSetting;
};

class Apps
{
public:
	static std::map<std::string, App*>& GetRegistry()
	{
		static std::map<std::string, App*> instance;
		return instance;
	}

	static void Add(const std::string& name, App* app)
	{
		GetRegistry()[name] = app;
	}

	static void Run(const std::string& name)
	{
		//auto t = std::thread([name]()
		{
			auto& registry = GetRegistry();
			auto it = registry.find(name);

			if (it != registry.end())
			{
				it->second->Execute();
			}
			else
			{
				std::cout << "오류: '" << name << "' 을(를) 찾을 수 없습니다." << std::endl;
			}
		}
		//);
		//t.detach();
	}

	static void Clear()
	{
		for (auto const& [name, app] : GetRegistry())
		{
			delete app;
		}
		GetRegistry().clear();
	}
};

template <typename T>
class AppRegistrar
{
public:
    AppRegistrar(const std::string& name)
    {
        Apps::Add(name, new T());
    }
};

#define REGISTER_APP(ClassName, AppName) static AppRegistrar<ClassName> global_##ClassName##_registrar(AppName);

//// --- 사용 예시 ---
//
//class CudaMemCheck : public App
//{
//public:
//    void Execute() override
//    {
//        std::cout << "CudaMemCheck 실행 중..." << std::endl;
//    }
//};
//
//REGISTER_APP(CudaMemCheck, "CudaMemCheck");
//
//class NetworkMonitor : public App
//{
//public:
//    void Execute() override
//    {
//        std::cout << "NetworkMonitor 실행 중..." << std::endl;
//    }
//};
//
//REGISTER_APP(NetworkMonitor, "NetworkMonitor");
