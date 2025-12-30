#pragma once

#include <vector>
#include <glad/glad.h> 

class GraphicsBuffer
{
public:
    GraphicsBuffer(GLenum target);

    ~GraphicsBuffer();

    GraphicsBuffer(const GraphicsBuffer&) = delete;
    GraphicsBuffer& operator=(const GraphicsBuffer&) = delete;

    GraphicsBuffer(GraphicsBuffer&& other) noexcept;
    GraphicsBuffer& operator=(GraphicsBuffer&& other) noexcept;

    void Bind() const;

    void Unbind() const;

    void SetData(const void* data, GLsizeiptr size, GLenum usage);

    void SetSubData(const void* data, GLsizeiptr size, GLintptr offset);

    void BindBufferBase(GLuint index) const;

    template <typename T>
    void SetData(const std::vector<T>& data, GLenum usage)
    {
        SetData(data.data(), data.size() * sizeof(T), usage);
    }

    GLuint GetID() const { return m_RendererID; }
    GLenum GetTarget() const { return m_Target; }
    GLsizeiptr GetSize() const { return m_Size; }

private:
    GLuint m_RendererID;
    GLenum m_Target;
    GLsizeiptr m_Size;
};
