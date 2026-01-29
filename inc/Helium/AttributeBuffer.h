#pragma once

#include <Helium/HeliumCommon.h>

#include <vector>
#include <memory>
#include <glad/glad.h>
#include <Eigen/Dense>
#include <Helium/GraphicsBuffer.h>

template<typename T>
class HELIUM_API AttributeBuffer
{
public:
    AttributeBuffer() = default;
    ~AttributeBuffer() = default;

    void Initialize(GLuint newAttribIndex, GLenum newTarget = GL_ARRAY_BUFFER, GLenum newUsage = GL_STATIC_DRAW)
    {
        attributeIndex = newAttribIndex;
        target = newTarget;
        usage = newUsage;

        gpuBuffer = std::make_unique<GraphicsBuffer>(target);
    }

    void Clear()
    {
        data.clear();
        dirty = true;
    }

    void Reserve(size_t capacity)
    {
        data.reserve(capacity);
	}

    size_t AddData(const T& element)
    {
        data.push_back(element);
        dirty = true;

		return data.size() - 1;
    }

    size_t AddDatas(const std::vector<T>& elements)
    {
        data.insert(data.end(), elements.begin(), elements.end());
        dirty = true;

		return data.size() - 1;
    }

    void SetData(unsigned int index, const T& element)
    {
        if (index >= data.size()) return;
        data[index] = element;
        dirty = true;
    }

    void SetDatas(const std::vector<T>& elements)
    {
        data = elements;
        dirty = true;
	}

    void Update()
    {
        if (data.empty() || !gpuBuffer) return;

        if (dirty)
        {
            gpuBuffer->SetData(data, usage);

            if (target == GL_ARRAY_BUFFER && attributeIndex != 0xFFFFFFFF)
            {
                gpuBuffer->Bind();
                SetupAttributePointer();

                if (useInstancing)
                {
                    if constexpr (std::is_same_v<T, Eigen::Matrix4f>)
                    {
                        for (int i = 0; i < 4; i++)
                            glVertexAttribDivisor(attributeIndex + i, 1);
                    }
                    else
                    {
                        glVertexAttribDivisor(attributeIndex, 1);
                    }
                }
            }
            dirty = false;
        }
    }

    std::vector<T>& GetCpuData() { return data; }
    const std::vector<T>& GetCpuData() const { return data; }
    size_t Size() const { return data.size(); }

	T& operator[](size_t index) { return data[index]; }
	const T& operator[](size_t index) const { return data[index]; }

    inline void SetUseInstancing(bool use) { if (use != useInstancing) { useInstancing = use; dirty = true; } }

private:
    void SetupAttributePointer()
    {
        if constexpr (std::is_same_v<T, unsigned int> || std::is_same_v<T, int>)
        {
            glEnableVertexAttribArray(attributeIndex);
            glVertexAttribIPointer(attributeIndex, 1, GL_UNSIGNED_INT, sizeof(T), (void*)0);
        }
        else if constexpr (std::is_same_v<T, Eigen::Vector2f>)
        {
            glEnableVertexAttribArray(attributeIndex);
            glVertexAttribPointer(attributeIndex, 2, GL_FLOAT, GL_FALSE, sizeof(T), (void*)0);
        }
        else if constexpr (std::is_same_v<T, Eigen::Vector3f>)
        {
            glEnableVertexAttribArray(attributeIndex);
            glVertexAttribPointer(attributeIndex, 3, GL_FLOAT, GL_FALSE, sizeof(T), (void*)0);
        }
        else if constexpr (std::is_same_v<T, Eigen::Vector4f>)
        {
            glEnableVertexAttribArray(attributeIndex);
            glVertexAttribPointer(attributeIndex, 4, GL_FLOAT, GL_FALSE, sizeof(T), (void*)0);
        }
        else if constexpr (std::is_same_v<T, Eigen::Matrix4f>)
        {
            for (int i = 0; i < 4; i++)
            {
                glEnableVertexAttribArray(attributeIndex + i);
                glVertexAttribPointer(attributeIndex + i, 4, GL_FLOAT, GL_FALSE, sizeof(T), (void*)(sizeof(float) * i * 4));
            }
        }
        else
        {
            glEnableVertexAttribArray(attributeIndex);
            glVertexAttribPointer(attributeIndex, sizeof(T) / sizeof(float), GL_FLOAT, GL_FALSE, sizeof(T), (void*)0);
        }
    }

private:
    std::unique_ptr<GraphicsBuffer> gpuBuffer;
    std::vector<T> data;

    GLuint attributeIndex = 0xFFFFFFFF;
    GLenum target = GL_ARRAY_BUFFER;
    GLenum usage = GL_STATIC_DRAW;

    bool dirty = true;
    bool useInstancing = false;
};
