#include "pch.h"
#include <Helium/Components/Shader.h>
#include <Helium/HeliumCore.h>

Shader::Shader(const std::string& name, const std::string& vertexSrc, const std::string& fragmentSrc)
    : m_Name(name)
{
    unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexSrc);
    unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentSrc);

    m_RendererID = glCreateProgram();
    glAttachShader(m_RendererID, vs);
    glAttachShader(m_RendererID, fs);
    glLinkProgram(m_RendererID);

    int success;
    char infoLog[512];
    glGetProgramiv(m_RendererID, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_RendererID, 512, NULL, infoLog);
        Helium.Log("Shader", "Link Error (%s): %s", name.c_str(), infoLog);
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
}

Shader::~Shader()
{
    glDeleteProgram(m_RendererID);
}

unsigned int Shader::CompileShader(unsigned int type, const std::string& source)
{
    unsigned int id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, message);

        Helium.Log("Shader", "Compile Error (%s): %s",
            (type == GL_VERTEX_SHADER ? "Vertex" : "Fragment"), message);

        glDeleteShader(id);
        return 0;
    }
    return id;
}

void Shader::Bind() const { glUseProgram(m_RendererID); }
void Shader::Unbind() const { glUseProgram(0); }

int Shader::GetUniformLocation(const std::string& name)
{
    return glGetUniformLocation(m_RendererID, name.c_str());
}

void Shader::SetInt(const std::string& name, int value)
{
    glUniform1i(GetUniformLocation(name), value);
}

void Shader::SetFloat(const std::string& name, float value)
{
    glUniform1f(GetUniformLocation(name), value);
}

void Shader::SetVec3(const std::string& name, const Eigen::Vector3f& value)
{
    glUniform3fv(GetUniformLocation(name), 1, value.data());
}

void Shader::SetVec4(const std::string& name, const Eigen::Vector4f& value)
{
    glUniform4fv(GetUniformLocation(name), 1, value.data());
}

void Shader::SetMat4(const std::string& name, const Eigen::Matrix4f& value)
{
    glUniformMatrix4fv(GetUniformLocation(name), 1, GL_FALSE, value.data());
}
