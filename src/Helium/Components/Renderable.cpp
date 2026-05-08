#include "pch.h"
#include <Helium/Components/Renderable.h>
#include <Helium/Components/Shader.h>

Renderable::Renderable()
{
}

Renderable::~Renderable()
{
    if (vao != 0) glDeleteVertexArrays(1, &vao);
}

void Renderable::Initialize(GeometryMode mode)
{
    geometryMode = mode;

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    vertices.Initialize(0);
    normals.Initialize(1);
    colors3.Initialize(2);
    colors4.Initialize(2);
    uvs.Initialize(3);

    indices.Initialize(0xFFFFFFFF, GL_ELEMENT_ARRAY_BUFFER);

    instanceColors.Initialize(4);
    instanceNormals.Initialize(5);
    instanceTransforms.Initialize(6);

    glBindVertexArray(0);
}

bool Renderable::IsInstancingEnabled() const
{
    return instancingEnabled;
}

void Renderable::EnableInstancing(bool enable)
{
    instancingEnabled = enable;

    instanceColors.SetUseInstancing(enable);
    instanceNormals.SetUseInstancing(enable);
    instanceTransforms.SetUseInstancing(enable);
}

void Renderable::ReserveInstances(size_t capacity)
{
    instanceTransforms.Reserve(capacity);
    instanceColors.Reserve(capacity);
    instanceNormals.Reserve(capacity);
}

void Renderable::Clear()
{
    vertices.Clear();
    normals.Clear();
    colors3.Clear();
    colors4.Clear();
    uvs.Clear();
    indices.Clear();
}

void Renderable::ClearInstancingData()
{
    instanceTransforms.Clear();
    instanceColors.Clear();
    instanceNormals.Clear();
    numberOfInstances = 0;
}

void Renderable::Update()
{
    glBindVertexArray(vao);

    vertices.Update();
    normals.Update();
    if (colors4.Size() > 0)
    {
        colors4.Update();
    }
    else
    {
        colors3.Update();
    }
    uvs.Update();
    indices.Update();

    if (instancingEnabled)
    {
        instanceColors.Update();
        instanceNormals.Update();
        instanceTransforms.Update();
    }

    glBindVertexArray(0);
}

size_t Renderable::AddShader(Shader* shader)
{
    shaders.push_back(shader);
	return shaders.size() - 1;
}

Shader* Renderable::GetActiveShader() const
{
    if (shaders.empty() || activeShaderIndex >= shaders.size())
        return nullptr;
    return shaders[activeShaderIndex];
}

void Renderable::SetActiveShaderIndex(unsigned int index)
{
    if (index < shaders.size())
        activeShaderIndex = index;
}

void Renderable::Draw()
{
    if (!visible || drawingMode == None) return;
    if (vertices.Size() == 0 && indices.Size() == 0) return;

    Shader* shader = GetActiveShader();
    if (shader) shader->Bind();

    glBindVertexArray(vao);

    GLint oldPolygonMode[2];
    glGetIntegerv(GL_POLYGON_MODE, oldPolygonMode);

	GLint oldCullFaceMode;
    glGetIntegerv(GL_CULL_FACE_MODE, &oldCullFaceMode);
    switch (faceCullingMode)
    {
    case NoCulling:
        glDisable(GL_CULL_FACE);
        break;
    case FrontFace:
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT);
        break;
    case BackFace:
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        break;
    case FrontAndBack:
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT_AND_BACK);
        break;
    default:
        break;
	}

	switch (drawingMode)
    {
    case Solid:
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        DrawImplementation();
        break;
    case WireFrame:
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        DrawImplementation();
        break;
    case Point:
        glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
        glPointSize(3.0f);
        DrawImplementation();
        glPointSize(1.0f);
        break;
    case WireFrameOverSolid:
        SetForcedColor(false);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0f, 1.0f);
        DrawImplementation();
        glDisable(GL_POLYGON_OFFSET_FILL);

        SetForcedColor(true, Eigen::Vector3f(0.0f, 0.0f, 0.0f));
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        DrawImplementation();

        SetForcedColor(false);
        break;
    default:
        break;
    }

    glPolygonMode(GL_FRONT_AND_BACK, oldPolygonMode[0]);
    glCullFace(oldCullFaceMode);
    glBindVertexArray(0);
}

void Renderable::DrawImplementation()
{
    Shader* shader = GetActiveShader();
    if (shader)
    {
        shader->SetInt("useSolidColor", useForcedColor ? 1 : 0);
        if (useForcedColor)
        {
            shader->SetVector3f("solidColor", forcedColor);
        }
    }

    GLsizei count = (indices.Size() > 0) ? (GLsizei)indices.Size() : (GLsizei)vertices.Size();
    bool useIndices = (indices.Size() > 0);

    if (instancingEnabled)
    {
        if (numberOfInstances > 0)
        {
            if (useIndices)
            {
                glDrawElementsInstanced(geometryMode, count, GL_UNSIGNED_INT, 0, numberOfInstances);
            }
            else
            {
                glDrawArraysInstanced(geometryMode, 0, count, numberOfInstances);
            }
        }
    }
    else
    {
        if (useIndices)
        {
            glDrawElements(geometryMode, count, GL_UNSIGNED_INT, 0);
        }
        else
        {
            glDrawArrays(geometryMode, 0, count);
        }
    }
}

void Renderable::AddInstanceTransform(const Eigen::Matrix4f& transform)
{
    instanceTransforms.AddData(transform);
    IncreaseNumberOfInstances();
}

void Renderable::AddInstanceColor(const Eigen::Vector4f& color)
{
    instanceColors.AddData(color);
}

void Renderable::AddInstanceNormal(const Eigen::Vector3f& normal)
{
    instanceNormals.AddData(normal);
}

inline Eigen::Matrix4f Renderable::GetInstanceTransform(size_t index) const
{
    if (index < instanceTransforms.Size())
        return instanceTransforms[index];
    return Eigen::Matrix4f::Identity();
}

void Renderable::SetInstanceTransform(unsigned int index, const Eigen::Matrix4f& transform)
{
	instanceTransforms.SetData(index, transform);
}

void Renderable::SetInstanceColor(unsigned int index, const Eigen::Vector4f& color)
{
	instanceColors.SetData(index, color);
}

void Renderable::SetInstanceNormal(unsigned int index, const Eigen::Vector3f& normal)
{
	instanceNormals.SetData(index, normal);
}