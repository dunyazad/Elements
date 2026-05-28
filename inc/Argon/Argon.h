#pragma once

struct Vertex
{
    float x;
    float y;
    float z;
    int halfedgeIndex;
};

struct Halfedge
{
    unsigned int vertexIndex;
    unsigned int nextHalfedgeIndex;
    unsigned int oppositeHalfedgeIndex;
    unsigned int faceIndex;
};

struct Face
{
    int halfedgeIndex;
};
