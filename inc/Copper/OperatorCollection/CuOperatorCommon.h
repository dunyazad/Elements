#pragma once

#include <Copper/CopperCommon.h>

#include <any>
#include <map>
#include <string>

struct CuPointCloud;

struct COPPER_API CuOperatorParameters
{
    std::map<std::string, std::any> parameters;

    template <typename T>
    void SetParameter(const std::string& name, const T& value)
    {
        parameters[name] = value;
    }

    template <typename T>
    T GetParameter(const std::string& name, const T& defaultValue) const
    {
        auto it = parameters.find(name);
        if (it != parameters.end())
            return std::any_cast<T>(it->second);
        return defaultValue;
    }
};
