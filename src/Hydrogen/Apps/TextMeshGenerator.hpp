#pragma once

#include <string>
#include <vector>
#include <array>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <Eigen/Dense>
#include <CDT.h>

#include <stb/stb_truetype.h>   // implementation defined in exactly one .cpp

#include "SimpleGeometryLibrary.hpp"

// Generates a closed, welded solid SGL::Mesh from a UTF-8 string using a
// TrueType/OpenType font. Pipeline per glyph:
//   stb outline -> flatten beziers -> 2D contours (outer + holes)
//   -> CDT front face (holes auto-removed) -> extrude to solid
// Glyphs are laid out left to right and merged into one mesh.
class TextMeshGenerator
{
public:
    TextMeshGenerator()
    {
        fontInfo = new stbtt_fontinfo();
        fontBuffer = nullptr;
    }

    ~TextMeshGenerator()
    {
        if (fontBuffer) delete[] fontBuffer;
        if (fontInfo) delete fontInfo;
    }

    // Load a .ttf/.otf from raw bytes already in memory.
    bool LoadFontFromMemory(const unsigned char* data, size_t size)
    {
        if (data == nullptr || size == 0) return false;
        if (fontBuffer) delete[] fontBuffer;
        fontBuffer = new unsigned char[size];
        std::memcpy(fontBuffer, data, size);
        if (stbtt_InitFont(fontInfo, fontBuffer, 0) == 0)
        {
            std::printf("!!! [Font] stbtt_InitFont failed\n");
            return false;
        }
        loaded_ = true;
        return true;
    }

    bool IsLoaded() const { return loaded_; }

    // Build a solid mesh for the whole string into 'out'.
    //   scale    : multiplied directly onto font units (existing convention)
    //   depth    : extrusion thickness along +Z
    //   flatness : max bezier chord error in model units (smaller = smoother)
    // 'out' is cleared, rebuilt, welded, and left closed for boolean use.
    void Create3DText(SGL::Mesh& out, const std::string& text,
        float depth, float scale, float flatness)
    {
        std::printf("\n>>> [Text] '%s' depth=%.3f scale=%.6f flatness=%.4f\n",
            text.c_str(), depth, scale, flatness);

        std::vector<Eigen::Vector3f> all_pts;
        std::vector<Eigen::Vector3i> all_idx;

        float xCursor = 0.0f;
        int built = 0;

        for (size_t i = 0; i < text.length(); ++i)
        {
            int cp = GetUtf8Codepoint(text, i);
            if (cp == 0) continue;

            // space: advance only
            if (cp == 32)
            {
                int adv = 0, lsb = 0;
                stbtt_GetCodepointHMetrics(fontInfo, cp, &adv, &lsb);
                xCursor += adv * scale;
                continue;
            }

            std::vector<Contour> contours;
            float advance = 0.0f;
            bool ok = ExtractGlyph(cp, scale, flatness, contours, advance);

            if (ok && !contours.empty())
            {
                std::vector<Eigen::Vector2f> front_pts;
                std::vector<Eigen::Vector3i> front_tris;
                TriangulateFront(contours, front_pts, front_tris);

                if (!front_tris.empty())
                {
                    ExtrudeToSolid(contours, front_pts, front_tris,
                        xCursor, depth, all_pts, all_idx);
                    built++;
                    std::printf("    code=%d loops=%zu front_tris=%zu adv=%.3f\n",
                        cp, contours.size(), front_tris.size(), advance);
                }
                else
                {
                    std::printf("    code=%d FRONT EMPTY (CDT produced no triangles)\n", cp);
                }
            }
            else
            {
                std::printf("    code=%d no contours (ok=%d)\n", cp, (int)ok);
            }

            xCursor += advance;
        }

        std::printf(">>> [Text] glyphs built=%d total pts=%zu tris=%zu\n",
            built, all_pts.size(), all_idx.size());

        out.clear();
        if (all_idx.empty()) return;

        out.Build(all_pts, all_idx);

        // weld coincident vertices so the solid is closed and manifold
        Eigen::Vector3f bmin(1e30f, 1e30f, 1e30f), bmax(-1e30f, -1e30f, -1e30f);
        for (const auto& p : all_pts) { bmin = bmin.cwiseMin(p); bmax = bmax.cwiseMax(p); }
        float diag = (bmax - bmin).norm();
        if (diag < 1e-6f) diag = 1.0f;
        out.WeldVerticesByPosition(diag * 1e-5f);

        auto d = out.Diagnose();
        std::printf(">>> [Text] welded V=%d F=%d boundary_edges=%d watertight=%s\n",
            d.num_vertices, d.num_faces, d.boundary_edges,
            d.is_watertight ? "YES" : "NO");
    }

private:
    // One closed 2D contour in model units.
    struct Contour
    {
        std::vector<Eigen::Vector2f> pts;   // no duplicated closing point
        bool is_hole = false;
    };

    // --- UTF-8 ---------------------------------------------------------
    // Decode one codepoint at text[i]; advance i past continuation bytes
    // (the caller's ++i moves to the next character). 0 on malformed input.
    static int GetUtf8Codepoint(const std::string& text, size_t& i)
    {
        unsigned char c = (unsigned char)text[i];
        int cp = 0;
        if (c < 0x80) { cp = c; }
        else if ((c & 0xE0) == 0xC0)
        {
            if (i + 1 >= text.length()) return 0;
            cp = ((c & 0x1F) << 6) | ((unsigned char)text[i + 1] & 0x3F);
            i += 1;
        }
        else if ((c & 0xF0) == 0xE0)
        {
            // 3 bytes; Korean syllables live here
            if (i + 2 >= text.length()) return 0;
            cp = ((c & 0x0F) << 12) |
                (((unsigned char)text[i + 1] & 0x3F) << 6) |
                ((unsigned char)text[i + 2] & 0x3F);
            i += 2;
        }
        else if ((c & 0xF8) == 0xF0)
        {
            if (i + 3 >= text.length()) return 0;
            cp = ((c & 0x07) << 18) |
                (((unsigned char)text[i + 1] & 0x3F) << 12) |
                (((unsigned char)text[i + 2] & 0x3F) << 6) |
                ((unsigned char)text[i + 3] & 0x3F);
            i += 3;
        }
        return cp;
    }

    // --- glyph outline -------------------------------------------------
    // Extract one glyph's contours, beziers flattened. 'advance' is the pen
    // step in model units. Returns false for missing/empty glyphs.
    bool ExtractGlyph(int codepoint, float scale, float flatness,
        std::vector<Contour>& contours, float& advance) const
    {
        contours.clear();
        advance = 0.0f;

        int adv = 0, lsb = 0;
        stbtt_GetCodepointHMetrics(fontInfo, codepoint, &adv, &lsb);
        advance = adv * scale;

        int glyph = stbtt_FindGlyphIndex(fontInfo, codepoint);
        if (glyph == 0) return false;

        stbtt_vertex* verts = nullptr;
        int nverts = stbtt_GetGlyphShape(fontInfo, glyph, &verts);
        if (nverts <= 0 || verts == nullptr)
        {
            if (verts) stbtt_FreeShape(fontInfo, verts);
            return false;
        }

        // recursive quadratic flattening by chord deviation
        std::function<void(std::vector<Eigen::Vector2f>&,
            Eigen::Vector2f, Eigen::Vector2f, Eigen::Vector2f, int)> flatten_quad =
            [&](std::vector<Eigen::Vector2f>& dst,
                Eigen::Vector2f p0, Eigen::Vector2f pc, Eigen::Vector2f p1, int depth)
            {
                Eigen::Vector2f mid_curve = 0.25f * p0 + 0.5f * pc + 0.25f * p1;
                Eigen::Vector2f mid_chord = 0.5f * (p0 + p1);
                if (depth >= 16 || (mid_curve - mid_chord).squaredNorm() < flatness * flatness)
                {
                    dst.push_back(p1);
                    return;
                }
                Eigen::Vector2f p01 = 0.5f * (p0 + pc);
                Eigen::Vector2f pc1 = 0.5f * (pc + p1);
                Eigen::Vector2f mid = 0.5f * (p01 + pc1);
                flatten_quad(dst, p0, p01, mid, depth + 1);
                flatten_quad(dst, mid, pc1, p1, depth + 1);
            };

        auto fu = [&](short x, short y) -> Eigen::Vector2f
            {
                return Eigen::Vector2f(x * scale, y * scale);
            };

        Contour current;
        Eigen::Vector2f cursor(0.0f, 0.0f);

        for (int i = 0; i < nverts; ++i)
        {
            const stbtt_vertex& v = verts[i];
            if (v.type == STBTT_vmove)
            {
                if (current.pts.size() >= 3) contours.push_back(current);
                current = Contour();
                cursor = fu(v.x, v.y);
                current.pts.push_back(cursor);
            }
            else if (v.type == STBTT_vline)
            {
                cursor = fu(v.x, v.y);
                current.pts.push_back(cursor);
            }
            else if (v.type == STBTT_vcurve)
            {
                Eigen::Vector2f p1 = fu(v.x, v.y);
                Eigen::Vector2f pc = fu(v.cx, v.cy);
                flatten_quad(current.pts, cursor, pc, p1, 0);
                cursor = p1;
            }
            else if (v.type == STBTT_vcubic)
            {
                Eigen::Vector2f p0 = cursor;
                Eigen::Vector2f c0 = fu(v.cx, v.cy);
                Eigen::Vector2f c1 = fu(v.cx1, v.cy1);
                Eigen::Vector2f p1 = fu(v.x, v.y);
                const int seg = 12;
                for (int j = 1; j <= seg; ++j)
                {
                    float t = j / (float)seg, it = 1.0f - t;
                    Eigen::Vector2f p =
                        it * it * it * p0 +
                        3.0f * it * it * t * c0 +
                        3.0f * it * t * t * c1 +
                        t * t * t * p1;
                    current.pts.push_back(p);
                }
                cursor = p1;
            }
        }
        if (current.pts.size() >= 3) contours.push_back(current);

        stbtt_FreeShape(fontInfo, verts);

        // drop duplicated closing point
        for (auto& c : contours)
            if (c.pts.size() >= 2 &&
                (c.pts.front() - c.pts.back()).squaredNorm() < 1e-12f)
                c.pts.pop_back();

        if (contours.empty()) return false;

        // classify outer vs hole by signed area relative to the largest loop
        float max_abs = 0.0f, outer_sign = 1.0f;
        for (auto& c : contours)
        {
            float a = SignedArea(c.pts);
            if (std::abs(a) > max_abs) { max_abs = std::abs(a); outer_sign = (a >= 0.0f) ? 1.0f : -1.0f; }
        }
        for (auto& c : contours)
        {
            float a = SignedArea(c.pts);
            c.is_hole = ((a >= 0.0f ? 1.0f : -1.0f) != outer_sign);
        }
        return true;
    }

    static float SignedArea(const std::vector<Eigen::Vector2f>& p)
    {
        float a = 0.0f;
        for (size_t k = 0; k < p.size(); ++k)
        {
            const Eigen::Vector2f& u = p[k];
            const Eigen::Vector2f& w = p[(k + 1) % p.size()];
            a += u.x() * w.y() - w.x() * u.y();
        }
        return 0.5f * a;
    }

    // --- front face triangulation (CDT) --------------------------------
    // Triangulate the region bounded by all contours; holes are removed by
    // CDT's eraseOuterTrianglesAndHoles using contour edges as constraints.
    void TriangulateFront(const std::vector<Contour>& contours,
        std::vector<Eigen::Vector2f>& out_pts,
        std::vector<Eigen::Vector3i>& out_tris) const
    {
        out_pts.clear();
        out_tris.clear();

        std::vector<CDT::V2d<float>> verts;
        std::vector<CDT::Edge> edges;

        // append each contour's points and its closing edge ring
        for (const auto& c : contours)
        {
            if (c.pts.size() < 3) continue;
            CDT::VertInd base = static_cast<CDT::VertInd>(verts.size());
            for (const auto& p : c.pts)
            {
                CDT::V2d<float> v; v.x = p.x(); v.y = p.y();
                verts.push_back(v);
            }
            CDT::VertInd n = static_cast<CDT::VertInd>(c.pts.size());
            for (CDT::VertInd k = 0; k < n; ++k)
                edges.push_back(CDT::Edge(base + k, base + (k + 1) % n));
        }

        if (verts.size() < 3) return;

        try
        {
            CDT::Triangulation<float> cdt(
                CDT::VertexInsertionOrder::Auto,
                CDT::IntersectingConstraintEdges::TryResolve,
                0.0f);

            CDT::DuplicatesInfo di = CDT::RemoveDuplicatesAndRemapEdges(verts, edges);

            cdt.insertVertices(verts);
            cdt.insertEdges(edges);
            cdt.eraseOuterTrianglesAndHoles();

            out_pts.resize(cdt.vertices.size());
            for (size_t i = 0; i < cdt.vertices.size(); ++i)
                out_pts[i] = Eigen::Vector2f(cdt.vertices[i].x, cdt.vertices[i].y);

            for (const auto& tri : cdt.triangles)
                out_tris.push_back(Eigen::Vector3i(
                    static_cast<int>(tri.vertices[0]),
                    static_cast<int>(tri.vertices[1]),
                    static_cast<int>(tri.vertices[2])));
        }
        catch (const std::exception& e)
        {
            std::printf("    [CDT] exception: %s\n", e.what());
            out_pts.clear();
            out_tris.clear();
        }
    }

    // Extrude to a closed solid using a SHARED vertex index space, so the
    // result is watertight without relying on a post-weld. front_pts are the
    // CDT output vertices; front_tris index into them. We build, for each
    // front vertex, a front (z=0) and back (z=depth) copy, and stitch side
    // walls along boundary edges of the triangulation using those same copies.
    void ExtrudeToSolid(const std::vector<Contour>& /*contours*/,
        const std::vector<Eigen::Vector2f>& front_pts,
        const std::vector<Eigen::Vector3i>& front_tris,
        float xOffset, float depth,
        std::vector<Eigen::Vector3f>& pts,
        std::vector<Eigen::Vector3i>& idx) const
    {
        int base = static_cast<int>(pts.size());
        int nfp = static_cast<int>(front_pts.size());

        // front copies [base .. base+nfp), back copies [base+nfp .. base+2*nfp)
        for (const auto& p : front_pts)
            pts.push_back(Eigen::Vector3f(p.x() + xOffset, p.y(), 0.0f));
        for (const auto& p : front_pts)
            pts.push_back(Eigen::Vector3f(p.x() + xOffset, p.y(), depth));

        auto F = [&](int i) { return base + i; };          // front index
        auto B = [&](int i) { return base + nfp + i; };    // back index

        // front winding sign (to give front -Z, back +Z consistently)
        bool front_ccw = true;
        if (!front_tris.empty())
        {
            const Eigen::Vector3i& t = front_tris[0];
            const Eigen::Vector2f& a = front_pts[t[0]];
            const Eigen::Vector2f& b = front_pts[t[1]];
            const Eigen::Vector2f& c = front_pts[t[2]];
            float cr = (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
            front_ccw = (cr > 0.0f);
        }

        // front + back caps, sharing the front_pts indices
        for (const auto& t : front_tris)
        {
            if (front_ccw)
            {
                idx.push_back(Eigen::Vector3i(F(t[0]), F(t[2]), F(t[1])));  // front -> -Z
                idx.push_back(Eigen::Vector3i(B(t[0]), B(t[1]), B(t[2])));  // back  -> +Z
            }
            else
            {
                idx.push_back(Eigen::Vector3i(F(t[0]), F(t[1]), F(t[2])));
                idx.push_back(Eigen::Vector3i(B(t[0]), B(t[2]), B(t[1])));
            }
        }

        // side walls along BOUNDARY edges of the triangulation. A directed
        // edge (i->j) that appears in exactly one triangle is a silhouette/
        // contour edge -> needs a wall. Count each undirected edge's uses.
        std::map<std::pair<int, int>, int> edge_count;
        auto bump = [&](int a, int b)
            {
                auto key = std::make_pair(std::min(a, b), std::max(a, b));
                edge_count[key]++;
            };
        for (const auto& t : front_tris)
        {
            bump(t[0], t[1]); bump(t[1], t[2]); bump(t[2], t[0]);
        }

        // for orientation, walk each triangle's directed edges; emit a wall
        // for the edge whose undirected count==1 (boundary of the glyph)
        for (const auto& t : front_tris)
        {
            int v[3] = { t[0], t[1], t[2] };
            for (int e = 0; e < 3; ++e)
            {
                int a = v[e], b = v[(e + 1) % 3];
                auto key = std::make_pair(std::min(a, b), std::max(a, b));
                if (edge_count[key] != 1) continue;   // interior edge, skip

                // directed edge (a->b) as wound in this triangle. The wall
                // quad is Fa,Fb,Bb,Ba; choose winding from front_ccw so the
                // wall normal points outward.
                int fa = F(a), fb = F(b), ba = B(a), bb = B(b);
                if (front_ccw)
                {
                    idx.push_back(Eigen::Vector3i(fa, fb, bb));
                    idx.push_back(Eigen::Vector3i(fa, bb, ba));
                }
                else
                {
                    idx.push_back(Eigen::Vector3i(fa, bb, fb));
                    idx.push_back(Eigen::Vector3i(fa, ba, bb));
                }
            }
        }
    }

    unsigned char* fontBuffer = nullptr;
    stbtt_fontinfo* fontInfo = nullptr;
    bool loaded_ = false;
};
