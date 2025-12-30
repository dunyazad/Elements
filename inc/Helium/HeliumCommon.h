#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

//#define _HAS_STD_BYTE 0

#include <cstdio>
#include <stdio.h>
#include <limits.h>

#include <algorithm>
#include <any>
#include <atomic>
#include <bitset>

#define _USE_MATH_DEFINES
#include <cmath>

#include <chrono>
#include <cstddef>
#include <execution>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <thread>
#include <tuple>
#include <typeinfo>
#include <typeindex>
#include <unordered_map>
#include <unordered_set>
#include <vector>



#define SAFE_DELETE(p) { if(nullptr != (p)) { delete (p); (p) = nullptr; } }
#define SAFE_DELETE_ARRAY(p) { if(nullptr != (p)) { delete[] (p); (p) = nullptr; } }


enum HeliumLogLevel
{
	HE_LOG_INFO = 0,
	HE_LOG_WARN = 1,
	HE_LOG_ERROR = 2,
	HE_LOG_DEBUG = 3
};

typedef void(*HeliumLogCallback)(int level, const char* key, const char* value);

void He_Log(const char* fmt, ...);
void He_Log(const char* key, const char* fmt, ...);
void He_Log(HeliumLogLevel level, const char* fmt, ...);
void He_Log(HeliumLogLevel level, const char* key, const char* fmt, ...);

#include <Helium/Color.hpp>
#include <Helium/Morton3D.hpp>
#include <Helium/MarchingCubesTables.h>

#include <omp.h>

#include <entt/entt.hpp>
#ifdef _HAS_STD_BYTE
#undef _HAS_STD_BYTE
#endif
#define _HAS_STD_BYTE 0

using Entity = entt::entity;
#define InvalidEntity ((Entity)ui32_max)

using Registry = entt::registry;
using Dispatcher = entt::dispatcher;

const std::string EmptyString = "";

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef NO_BYTE
#define NO_BYTE
#endif

#include <windows.h>
#include <shellapi.h>

#define DIRECTINPUT_VERSION 0x0800
#include <dinput.h>
#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")

#include <RxTx/RxTx.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32  // Win32 관련 기능을 활성화
#include <GLFW/glfw3native.h>

#include "imgui.h"
#include "imgui_internal.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "implot.h"

#define Feather libFeather::GetStaticInstance()

#include <Helium/TypeDefinitions.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/LU>

namespace Eigen {
	template <typename Type, int Size>
	using Vector = Matrix<Type, Size, 1>;

	using Vector3b = Vector<unsigned char, 3>;
	using Vector3ui = Vector<unsigned int, 3>;
}

//#include <stb_connected_components.h>
//#include <stb_c_lexer.h>
//#include <stb_divide.h>
//#include <stb_ds.h>
//#include <stb_dxt.h>
//#include <stb_easy_font.h>
//#include <stb_herringbone_wang_tile.h>
//#include <stb_hexwave.h>
//#include <stb_image.h>
//#include <stb_image_resize2.h>
//#include <stb_image_write.h>
//#include <stb_include.h>
//#include <stb_leakcheck.h>
//#include <stb_perlin.h>
//#include <stb_rect_pack.h>
//#include <stb_sprintf.h>
//#include <stb_textedit.h>
//#include <stb_tilemap_editor.h>
//#include <stb_truetype.h>
//#include <stb_voxel_render.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#ifndef PI
#define PI 3.14159265358979323846
#endif
#define DEG2RAD (PI/180)
#define RAD2DEG (180/PI)

#ifndef XYZ
#define XYZ(v) (v).x, (v).y, (v).z
#endif
#ifndef XYZW
#define XYZW(v) (v).x, (v).y, (v).z, (v).w
#endif

namespace Time
{
	std::chrono::steady_clock::time_point Now();

	uint64_t Microseconds(std::chrono::steady_clock::time_point& from, std::chrono::steady_clock::time_point& now);

	std::chrono::steady_clock::time_point End(std::chrono::steady_clock::time_point& from, const std::string& message = "", int number = -1);

	std::string DateTime();
}

std::string Miliseconds(const std::chrono::steady_clock::time_point beginTime, const char* tag);

#define TS(name) auto time_##name = std::chrono::high_resolution_clock::now();
#define TE(name) std::cout << Miliseconds(time_##name, #name) << std::endl;

#define alog(...) printf("\033[38;5;1m\033[48;5;15m(^(OO)^) /V/\033[0m\t" __VA_ARGS__)
#define alogt(tag, ...) printf("\033[38;5;1m\033[48;5;15m [%d] (^(OO)^) /V/\033[0m\t" tag, __VA_ARGS__)

#define aerr(...) printf("\033[38;5;15m\033[48;5;1m(x(OO)x) /V/ [ERROR] @%s:%d\033[0m\t" __VA_ARGS__, __FILE__, __LINE__)
