#pragma once
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <Helium/AttributeBuffer.h>

class Shader;

class Renderable
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

public:
    Renderable();
    virtual ~Renderable();

    void Initialize(GeometryMode mode = Triangles);

    virtual void Update();
    virtual void Draw();

    virtual void Clear();
    virtual void ClearInstancingData();

    // Data Access
    void AddVertex(const Eigen::Vector3f& v) { vertices.AddData(v); }
    void AddNormal(const Eigen::Vector3f& n) { normals.AddData(n); }
    void AddColor3(const Eigen::Vector3f& c) { colors3.AddData(c); }
    void AddColor4(const Eigen::Vector4f& c) { colors4.AddData(c); }
    void AddUV(const Eigen::Vector2f& uv) { uvs.AddData(uv); }
    void AddIndex(unsigned int i) { indices.AddData(i); }

    // Batch Data Access
    void AddVertices(const std::vector<Eigen::Vector3f>& list) { vertices.AddDatas(list); }
    void AddNormals(const std::vector<Eigen::Vector3f>& list) { normals.AddDatas(list); }
    void AddColors3(const std::vector<Eigen::Vector3f>& list) { colors3.AddDatas(list); }
    void AddColors4(const std::vector<Eigen::Vector4f>& list) { colors4.AddDatas(list); }
    void AddUVs(const std::vector<Eigen::Vector2f>& list) { uvs.AddDatas(list); }
    void AddIndices(const std::vector<unsigned int>& list) { indices.AddDatas(list); }

    // Instancing
    void EnableInstancing(bool enable = true);
    void AddInstanceTransform(const Eigen::Matrix4f& transform);
    void AddInstanceColor(const Eigen::Vector4f& color);
    void AddInstanceNormal(const Eigen::Vector3f& normal);

    void IncreaseNumberOfInstances() { numberOfInstances++; if (!instancingEnabled) EnableInstancing(); }
    void SetNumberOfInstances(unsigned int n) { numberOfInstances = n; }
    unsigned int GetNumberOfInstances() const { return numberOfInstances; }

    // Shader Management
    void AddShader(Shader* shader);
    Shader* GetActiveShader() const;
    void SetActiveShaderIndex(unsigned int index);

    // Settings
    void SetGeometryMode(GeometryMode mode) { geometryMode = mode; }
    void SetDrawingMode(DrawingMode mode) { drawingMode = mode; }
    void NextDrawingMode() { drawingMode = (DrawingMode)((drawingMode + 1) % NumberOfDrawingModes); }

    void SetVisible(bool isVisible) { visible = isVisible; }
    bool IsVisible() const { return visible; }

    void SetUseAlpha(bool use) { useAlpha = use; }
    bool IsUsingAlpha() const { return useAlpha; }

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

    // Basic Buffers
    AttributeBuffer<unsigned int>    indices;
    AttributeBuffer<Eigen::Vector3f> vertices;
    AttributeBuffer<Eigen::Vector3f> normals;
    AttributeBuffer<Eigen::Vector3f> colors3;
    AttributeBuffer<Eigen::Vector4f> colors4;
    AttributeBuffer<Eigen::Vector2f> uvs;

    // Instancing Buffers
    AttributeBuffer<Eigen::Matrix4f> instanceTransforms;
    AttributeBuffer<Eigen::Vector4f> instanceColors;
    AttributeBuffer<Eigen::Vector3f> instanceNormals;

    unsigned int numberOfInstances = 0;
    bool instancingEnabled = false;
};
