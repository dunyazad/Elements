#pragma once

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

#include <Helium/Helium.h>
#include <Helium/HeliumLog.h>
#include <Helium/HeliumCore.h>

#include <Helium/Components/GUI/GUIComponent.h>

#include <Helium/VisualDebugging.h>
using VD = VisualDebugging;

#include <Copper/Copper.h>
#include <Copper/CuPointCloud.h>
#include <Copper/CuSparseCells.h>
#include <Copper/OperatorCollection/CuOperatorCollection.h>

#include <Helium/Serialization.hpp>
#include <Helium/DeviceInformation.h>

#include <Copper/CuVoxelStreaming.h>

#include <robin_hood/robin_hood.h>

#include <VVV/VVV.h>
#pragma comment(lib, "VVV.lib")

#include <Eigen/Core>
#include <Eigen/Dense>

#include <Helium/Serialization.hpp>

namespace Eigen {
    template <typename Type, int Size>
    using Vector = Matrix<Type, Size, 1>;

    using Vector3b = Vector<unsigned char, 3>;
    using Vector3ui = Vector<unsigned int, 3>;
}

class App
{
public:
    virtual ~App()
    {
    }

    virtual void Execute() = 0;
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