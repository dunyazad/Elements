#include "pch.h"

#include <Helium/GraphicsBuffer.h>

GraphicsBuffer::GraphicsBuffer(GLenum target)
    : rendererID(0)
    , target(target)
    , size(0)
{
    glGenBuffers(1, &rendererID);
}

GraphicsBuffer::~GraphicsBuffer()
{
    if (rendererID != 0)
    {
        glDeleteBuffers(1, &rendererID);
    }
}

GraphicsBuffer::GraphicsBuffer(GraphicsBuffer&& other) noexcept
    : rendererID(other.rendererID)
    , target(other.target)
    , size(other.size)
{
    other.rendererID = 0;
    other.size = 0;
}

GraphicsBuffer& GraphicsBuffer::operator=(GraphicsBuffer&& other) noexcept
{
    if (this != &other)
    {
        if (rendererID != 0)
        {
            glDeleteBuffers(1, &rendererID);
        }

        rendererID = other.rendererID;
        target = other.target;
        size = other.size;

        other.rendererID = 0;
        other.size = 0;
    }
    return *this;
}

void GraphicsBuffer::Bind() const
{
    glBindBuffer(target, rendererID);
}

void GraphicsBuffer::Unbind() const
{
    glBindBuffer(target, 0);
}

void GraphicsBuffer::SetData(const void* data, GLsizeiptr size, GLenum usage)
{
    Bind();
    glBufferData(target, size, data, usage);
    size = size;
}

void GraphicsBuffer::SetSubData(const void* data, GLsizeiptr size, GLintptr offset)
{
    Bind();
    glBufferSubData(target, offset, size, data);
}

void GraphicsBuffer::BindBufferBase(GLuint index) const
{
    // Usually for GL_UNIFORM_BUFFER or GL_SHADER_STORAGE_BUFFER
    glBindBufferBase(target, index, rendererID);
}