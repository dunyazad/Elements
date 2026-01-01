#pragma once

#include <Helium/HeliumCommon.h>

#include <Eigen/Dense>

class Shader
{
public:
    Shader(const std::string& name, const std::string& vertexSrc, const std::string& fragmentSrc);
    ~Shader();

    void Bind() const;
    void Unbind() const;

    void SetInt(const std::string& name, int value);
    void SetFloat(const std::string& name, float value);

    void SetVec3(const std::string& name, const Eigen::Vector3f& value);
    void SetVec4(const std::string& name, const Eigen::Vector4f& value);
    void SetMat4(const std::string& name, const Eigen::Matrix4f& value);

    inline unsigned int GetID() const { return rendererID; }
    inline const std::string& GetName() const { return name; }

private:
    unsigned int rendererID;
    std::string name;

    unsigned int CompileShader(unsigned int type, const std::string& source);
    int GetUniformLocation(const std::string& name);
};
