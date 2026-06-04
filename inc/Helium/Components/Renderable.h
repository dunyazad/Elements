#pragma once

#include <Helium/HeliumCommon.h>

#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <Helium/AttributeBuffer.h>

class Shader;

class HELIUM_API Renderable
{
public:
    enum GeometryMode
    {
        Points = GL_POINTS,
        Lines = GL_LINES,
        LineLoop = GL_LINE_LOOP,
        LineStrip = GL_LINE_STRIP,
        Triangles = GL_TRIANGLES,
        TriangleStrip = GL_TRIANGLE_STRIP,
        TriangleFan = GL_TRIANGLE_FAN,
        Quads = GL_QUADS
    };

    enum DrawingMode
    {
        None,
        Solid,
        WireFrame,
        WireFrameOverSolid,
        Point,
        NumberOfDrawingModes
    };

    enum FaceCullingMode
    {
        NoCulling,
        FrontFace,
        BackFace,
        FrontAndBack
    };

public:
    Renderable();
    virtual ~Renderable();

    void Initialize(GeometryMode mode = Triangles);

    virtual void Update();
    virtual void Draw();

    virtual void Clear();
    virtual void ClearInstancingData();

    inline size_t GetNumberOfVertices() const
    {
        return vertices.Size();
    }

    inline size_t GetNumberOfNormals() const
    {
        return normals.Size();
    }

    inline size_t GetNumberOfColors3() const
    {
        return colors3.Size();
    }

    inline size_t GetNumberOfColors4() const
    {
        return colors4.Size();
    }

    inline size_t GetNumberOfUVs() const
    {
        return uvs.Size();
    }

    inline size_t GetNumberOfIndices() const
    {
        return indices.Size();
    }

    inline Eigen::Vector3f GetVertex(unsigned int index) const
    {
        return vertices.GetCpuData()[index];
    }

    inline Eigen::Vector3f GetNormal(unsigned int index) const
    {
        return normals.GetCpuData()[index];
    }

    inline Eigen::Vector3f GetColor3(unsigned int index) const
    {
        return colors3.GetCpuData()[index];
    }

    inline Eigen::Vector4f GetColor4(unsigned int index) const
    {
        return colors4.GetCpuData()[index];
    }

    inline Eigen::Vector2f GetUV(unsigned int index) const
    {
        return uvs.GetCpuData()[index];
    }

    inline unsigned int GetIndex(unsigned int index) const
    {
        return indices.GetCpuData()[index];
    }

    inline size_t AddVertex(const Eigen::Vector3f& v)
    {
        dirty = true;
        return vertices.AddData(v);
    }

    inline size_t AddNormal(const Eigen::Vector3f& n)
    {
        dirty = true;
        return normals.AddData(n);
    }

    inline size_t AddColor3(const Eigen::Vector3f& c)
    {
        dirty = true;
        return colors3.AddData(c);
    }

    inline size_t AddColor4(const Eigen::Vector4f& c)
    {
        dirty = true;
        return colors4.AddData(c);
    }

    inline size_t AddUV(const Eigen::Vector2f& uv)
    {
        dirty = true;
        return uvs.AddData(uv);
    }

    inline size_t AddIndex(unsigned int i)
    {
        dirty = true;
        return indices.AddData(i);
    }

    inline void SetVertex(unsigned int index, const Eigen::Vector3f& v)
    {
        dirty = true;
        vertices.SetData(index, v);
    }

    inline void SetNormal(unsigned int index, const Eigen::Vector3f& n)
    {
        dirty = true;
        normals.SetData(index, n);
    }

    inline void SetColor3(unsigned int index, const Eigen::Vector3f& c)
    {
        dirty = true;
        colors3.SetData(index, c);
    }

    inline void SetColor4(unsigned int index, const Eigen::Vector4f& c)
    {
        dirty = true;
        colors4.SetData(index, c);
    }

    inline void SetUV(unsigned int index, const Eigen::Vector2f& uv)
    {
        dirty = true;
        uvs.SetData(index, uv);
    }

    inline void SetIndex(unsigned int index, unsigned int i)
    {
        dirty = true;
        indices.SetData(index, i);
    }

    inline void SetVertices(const std::vector<Eigen::Vector3f>& list)
    {
        dirty = true;
        vertices.SetDatas(list);
    }

    inline void SetNormals(const std::vector<Eigen::Vector3f>& list)
    {
        dirty = true;
        normals.SetDatas(list);
    }

    inline void SetColors3(const std::vector<Eigen::Vector3f>& list)
    {
        dirty = true;
        colors3.SetDatas(list);
    }

    inline void SetColors4(const std::vector<Eigen::Vector4f>& list)
    {
        dirty = true;
        colors4.SetDatas(list);
    }

    inline void SetUVs(const std::vector<Eigen::Vector2f>& list)
    {
        dirty = true;
        uvs.SetDatas(list);
    }

    inline void SetIndices(const std::vector<unsigned int>& list)
    {
        dirty = true;
        indices.SetDatas(list);
    }

    inline void ClearVertices()
    {
        dirty = true;
        vertices.Clear();
    }

    inline void ClearNormals()
    {
        dirty = true;
        normals.Clear();
    }

    inline void ClearColors3()
    {
        dirty = true;
        colors3.Clear();
    }

    inline void ClearColors4()
    {
        dirty = true;
        colors4.Clear();
    }

    inline void ClearUVs()
    {
        dirty = true;
        uvs.Clear();
    }

    inline void ClearIndices()
    {
        dirty = true;
        indices.Clear();
    }

    inline void AddVertices(const std::vector<Eigen::Vector3f>& list)
    {
        dirty = true;
        vertices.AddDatas(list);
    }

    inline void AddNormals(const std::vector<Eigen::Vector3f>& list)
    {
        dirty = true;
        normals.AddDatas(list);
    }

    inline void AddColors3(const std::vector<Eigen::Vector3f>& list)
    {
        dirty = true;
        colors3.AddDatas(list);
    }

    inline void AddColors4(const std::vector<Eigen::Vector4f>& list)
    {
        dirty = true;
        colors4.AddDatas(list);
    }

    inline void AddUVs(const std::vector<Eigen::Vector2f>& list)
    {
        dirty = true;
        uvs.AddDatas(list);
    }

    inline void AddIndices(const std::vector<unsigned int>& list)
    {
        dirty = true;
        indices.AddDatas(list);
    }

    bool IsInstancingEnabled() const;
    void EnableInstancing(bool enable = true);
    void ReserveInstances(size_t capacity);
    void AddInstanceTransform(const Eigen::Matrix4f& transform);
    void AddInstanceColor(const Eigen::Vector4f& color);
    void AddInstanceNormal(const Eigen::Vector3f& normal);

    Eigen::Matrix4f GetInstanceTransform(size_t index) const;
    void SetInstanceTransform(unsigned int index, const Eigen::Matrix4f& transform);
    void SetInstanceColor(unsigned int index, const Eigen::Vector4f& color);
    void SetInstanceNormal(unsigned int index, const Eigen::Vector3f& normal);

    inline void IncreaseNumberOfInstances()
    {
        numberOfInstances++;
        if (!instancingEnabled)
        {
            EnableInstancing();
        }
    }

    inline void SetNumberOfInstances(unsigned int n)
    {
        numberOfInstances = n;
    }

    inline unsigned int GetNumberOfInstances() const
    {
        return numberOfInstances;
    }

    size_t AddShader(Shader* shader);
    Shader* GetActiveShader() const;
    void SetActiveShaderIndex(unsigned int index);

    inline GeometryMode GetGeometryMode() const
    {
        return geometryMode;
    }

    inline void SetGeometryMode(GeometryMode mode)
    {
        geometryMode = mode;
    }

    inline DrawingMode GetDrawingMode() const
    {
        return drawingMode;
    }

    inline void SetDrawingMode(DrawingMode mode)
    {
        drawingMode = mode;
    }

    inline void NextDrawingMode()
    {
        drawingMode = (DrawingMode)((drawingMode + 1) % NumberOfDrawingModes);
    }

    inline FaceCullingMode GetFaceCullingMode() const
    {
        return faceCullingMode;
    }

    inline void SetFaceCullingMode(FaceCullingMode mode)
    {
        faceCullingMode = mode;
    }

    inline void SetVisible(bool isVisible)
    {
        visible = isVisible;
    }

    inline bool IsVisible() const
    {
        return visible;
    }

    inline void SetUseAlpha(bool use)
    {
        useAlpha = use;
    }

    inline bool IsUsingAlpha() const
    {
        return useAlpha;
    }

    inline const AttributeBuffer<Eigen::Vector3f>& GetVertexBuffer() const
    {
        return vertices;
    }

    inline const AttributeBuffer<Eigen::Vector3f>& GetNormalBuffer() const
    {
        return normals;
    }

    inline const AttributeBuffer<Eigen::Vector3f>& GetColor3Buffer() const
    {
        return colors3;
    }

    inline const AttributeBuffer<Eigen::Vector4f>& GetColor4Buffer() const
    {
        return colors4;
    }

    inline const AttributeBuffer<Eigen::Vector2f>& GetUVBuffer() const
    {
        return uvs;
    }

    inline const AttributeBuffer<unsigned int>& GetIndexBuffer() const
    {
        return indices;
    }

    inline size_t GetInstanceCount() const
    {
        return instanceTransforms.Size();
    }

    inline void SetForcedColor(bool enable, const Eigen::Vector3f& color = Eigen::Vector3f(1.0f, 1.0f, 1.0f))
    {
        useForcedColor = enable;
        forcedColor = color;
    }

    void SetLineWidth(float width) { lineWidth = width; }
    float GetLineWidth() const { return lineWidth; }

protected:
    void DrawImplementation();

private:
    bool visible = true;
    bool dirty = true;
    bool useAlpha = false;

    unsigned int activeShaderIndex = 0;
    std::vector<Shader*> shaders;

    GLuint vao = 0;

    GeometryMode geometryMode = Triangles;
    DrawingMode drawingMode = Solid;
    FaceCullingMode faceCullingMode = BackFace;

    AttributeBuffer<unsigned int> indices;
    AttributeBuffer<Eigen::Vector3f> vertices;
    AttributeBuffer<Eigen::Vector3f> normals;
    AttributeBuffer<Eigen::Vector3f> colors3;
    AttributeBuffer<Eigen::Vector4f> colors4;
    AttributeBuffer<Eigen::Vector2f> uvs;

    AttributeBuffer<Eigen::Matrix4f> instanceTransforms;
    AttributeBuffer<Eigen::Vector4f> instanceColors;
    AttributeBuffer<Eigen::Vector3f> instanceNormals;

    unsigned int numberOfInstances = 0;
    bool instancingEnabled = false;

    bool useForcedColor = false;
    Eigen::Vector3f forcedColor = Eigen::Vector3f(1.0f, 1.0f, 1.0f);

    float lineWidth = 1.0f;
};

class HELIUM_API DebuggingRenderable : public Renderable
{
public:
};
