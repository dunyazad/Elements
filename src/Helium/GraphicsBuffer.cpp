#include "pch.h"

#include <Helium/GraphicsBuffer.h>

GraphicsBuffer::GraphicsBuffer(GLenum target)
    : m_RendererID(0)
    , m_Target(target)
    , m_Size(0)
{
    glGenBuffers(1, &m_RendererID);
}

GraphicsBuffer::~GraphicsBuffer()
{
    if (m_RendererID != 0)
    {
        glDeleteBuffers(1, &m_RendererID);
    }
}

GraphicsBuffer::GraphicsBuffer(GraphicsBuffer&& other) noexcept
    : m_RendererID(other.m_RendererID)
    , m_Target(other.m_Target)
    , m_Size(other.m_Size)
{
    other.m_RendererID = 0;
    other.m_Size = 0;
}

GraphicsBuffer& GraphicsBuffer::operator=(GraphicsBuffer&& other) noexcept
{
    if (this != &other)
    {
        if (m_RendererID != 0)
        {
            glDeleteBuffers(1, &m_RendererID);
        }

        m_RendererID = other.m_RendererID;
        m_Target = other.m_Target;
        m_Size = other.m_Size;

        other.m_RendererID = 0;
        other.m_Size = 0;
    }
    return *this;
}

void GraphicsBuffer::Bind() const
{
    glBindBuffer(m_Target, m_RendererID);
}

void GraphicsBuffer::Unbind() const
{
    glBindBuffer(m_Target, 0);
}

void GraphicsBuffer::SetData(const void* data, GLsizeiptr size, GLenum usage)
{
    Bind();
    glBufferData(m_Target, size, data, usage);
    m_Size = size;
}

void GraphicsBuffer::SetSubData(const void* data, GLsizeiptr size, GLintptr offset)
{
    Bind();
    glBufferSubData(m_Target, offset, size, data);
}

void GraphicsBuffer::BindBufferBase(GLuint index) const
{
    // Usually for GL_UNIFORM_BUFFER or GL_SHADER_STORAGE_BUFFER
    glBindBufferBase(m_Target, index, m_RendererID);
}