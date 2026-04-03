#!/usr/bin/env python3
"""
Step 4 -- Geometric Consolidation of Wall Primitives
=====================================================

Takes VLM-classified wall primitives from Step 3 and consolidates them
into clean wall geometry with proper topology.

Phases:
  4.1 — Flatten + Merge: combine multi-style primitives, snap float-noise
        endpoints, merge collinear micro-segments via unary_union + linemerge
  4.2 — T-junction Snap: extend near-miss endpoints to nearby wall segments
  4.3 — Collinear Bridge: fill small gaps between collinear segments
  4.4 — Short-stub Cleanup: remove isolated short segments (noise)

Input:  Step 3 classification result  (wall_primitives: {style_key: [prims]})
Output: Consolidated wall edges (LineStrings) and wall polygons
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import linemerge, unary_union

# ── constants ────────────────────────────────────────────────────────

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))

# Hardcoded confirmed-good classification results from Step 3.
HARDCODED_GOOD: dict[str, int] = {
    "2026-03-27_Test-2_v4":                2,
    "2026.01.29_-_FMOC_A_-_50%_CD_copy":  1,
    "main_st_ex":                          1,
}


# ── geometry helpers ─────────────────────────────────────────────────

def _extract_linestrings(geom) -> list[LineString]:
    """Recursively extract all non-empty LineStrings from any Shapely geom."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if geom.length > 0 else []
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if g.length > 0]
    if hasattr(geom, "geoms"):
        out: list[LineString] = []
        for g in geom.geoms:
            out.extend(_extract_linestrings(g))
        return out
    return []


def _extract_polygons(geom) -> list[Polygon]:
    """Recursively extract all non-empty Polygons from any Shapely geom."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area > 0 else []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if g.area > 0]
    if hasattr(geom, "geoms"):
        out: list[Polygon] = []
        for g in geom.geoms:
            out.extend(_extract_polygons(g))
        return out
    return []


def _line_angle(coords) -> float | None:
    """Angle in [0, π) of the chord from first to last coordinate."""
    if len(coords) < 2:
        return None
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    if dx * dx + dy * dy < 1e-12:
        return None
    return math.atan2(dy, dx) % math.pi


def _are_collinear(
    line_a: LineString,
    line_b: LineString,
    angle_tol: float = 0.05,
    perp_tol: float = 2.0,
) -> bool:
    """True if *line_a* and *line_b* are roughly on the same infinite line."""
    dir_a = _line_angle(list(line_a.coords))
    dir_b = _line_angle(list(line_b.coords))
    if dir_a is None or dir_b is None:
        return False
    diff = abs(dir_a - dir_b)
    diff = min(diff, math.pi - diff)
    if diff > angle_tol:
        return False
    # Perpendicular distance from midpoint-of-B to direction-of-A
    mid_b = line_b.interpolate(0.5, normalized=True)
    p0 = line_a.coords[0]
    cos_a, sin_a = math.cos(dir_a), math.sin(dir_a)
    perp_dist = abs(-(mid_b.y - p0[1]) * cos_a + (mid_b.x - p0[0]) * sin_a)
    return perp_dist < perp_tol


# ── Phase 4.1: Flatten + Merge ──────────────────────────────────────

def phase_4_1_flatten_merge(
    wall_primitives: dict[str, list[dict]],
    pt_per_inch: float = 72.0,
    coord_precision: int = 2,
) -> tuple[list[LineString], list[Polygon]]:
    """
    Flatten all wall primitives from every approved style into unified
    Shapely geometry.

    * Lines → round coords (remove float noise) → ``unary_union`` →
      ``linemerge`` to fuse connected micro-segments.
    * Polygons → ``unary_union`` to merge overlapping / adjacent shapes.

    Returns ``(merged_lines, merged_polygons)``.
    """
    raw_lines: list[LineString] = []
    raw_polys: list[Polygon] = []

    for _style_key, prims in wall_primitives.items():
        for prim in prims:
            pts = prim["points"]
            kind = prim["type"]

            if kind == "line" and len(pts) >= 2:
                rounded = [
                    (round(pts[0][0], coord_precision),
                     round(pts[0][1], coord_precision)),
                    (round(pts[1][0], coord_precision),
                     round(pts[1][1], coord_precision)),
                ]
                if rounded[0] != rounded[1]:
                    raw_lines.append(LineString(rounded))

            elif kind == "polygon" and len(pts) >= 3:
                rounded = [
                    (round(p[0], coord_precision),
                     round(p[1], coord_precision))
                    for p in pts
                ]
                try:
                    poly = Polygon(rounded)
                    if poly.is_valid and poly.area > 0:
                        raw_polys.append(poly)
                except Exception:
                    pass

            elif kind == "curve" and len(pts) >= 4:
                p0 = (round(pts[0][0], coord_precision),
                      round(pts[0][1], coord_precision))
                p3 = (round(pts[-1][0], coord_precision),
                      round(pts[-1][1], coord_precision))
                if p0 != p3:
                    raw_lines.append(LineString([p0, p3]))

    # ---- merge lines ------------------------------------------------
    if raw_lines:
        merged = unary_union(raw_lines)
        if isinstance(merged, MultiLineString):
            merged = linemerge(merged)
        lines = _extract_linestrings(merged)
    else:
        lines = []

    # ---- merge polygons ---------------------------------------------
    if raw_polys:
        merged_p = unary_union(raw_polys)
        polys = _extract_polygons(merged_p)
    else:
        polys = []

    return lines, polys


# ── Phase 4.2: T-junction Snap ──────────────────────────────────────

def phase_4_2_t_junction_snap(
    lines: list[LineString],
    pt_per_inch: float = 72.0,
    t_junction_tol: float | None = None,
) -> tuple[list[LineString], int]:
    """
    Snap line endpoints that nearly touch another line's body.

    For each endpoint within *t_junction_tol* of another line (but not
    already touching it), move the endpoint to the nearest point on that
    line.  Then re-merge so the new connection is reflected.

    Returns ``(snapped_lines, snap_count)``.
    """
    if t_junction_tol is None:
        t_junction_tol = 0.1 * pt_per_inch  # ~7.2 pt at 72 ppi

    if len(lines) < 2:
        return list(lines), 0

    # Build STRtree for fast spatial queries
    from shapely import STRtree
    tree = STRtree(lines)

    snap_count = 0
    new_coords: list[list[tuple[float, float]]] = [
        list(line.coords) for line in lines
    ]

    for i, line_i in enumerate(lines):
        for end_idx in (0, -1):
            pt = Point(line_i.coords[end_idx])

            # Query only lines within t_junction_tol
            nearby_idxs = tree.query(pt.buffer(t_junction_tol))

            best_dist = t_junction_tol
            best_point: tuple[float, float] | None = None

            for j in nearby_idxs:
                if j == i:
                    continue
                line_j = lines[j]
                dist = line_j.distance(pt)
                if dist >= best_dist or dist < 0.01:
                    continue
                proj = line_j.project(pt)
                nearest = line_j.interpolate(proj)
                d_start = Point(line_j.coords[0]).distance(nearest)
                d_end   = Point(line_j.coords[-1]).distance(nearest)
                if d_start < 0.5 or d_end < 0.5:
                    continue
                best_dist = dist
                best_point = (nearest.x, nearest.y)

            if best_point is not None:
                idx = 0 if end_idx == 0 else len(new_coords[i]) - 1
                new_coords[i][idx] = best_point
                snap_count += 1

    # Rebuild and re-merge
    rebuilt: list[LineString] = []
    for coords in new_coords:
        ls = LineString(coords)
        if ls.length > 0:
            rebuilt.append(ls)

    if rebuilt:
        merged = unary_union(rebuilt)
        if isinstance(merged, MultiLineString):
            merged = linemerge(merged)
        rebuilt = _extract_linestrings(merged)

    return rebuilt, snap_count


# ── Phase 4.3: Collinear Bridge ─────────────────────────────────────

def phase_4_3_collinear_bridge(
    lines: list[LineString],
    pt_per_inch: float = 72.0,
    bridge_tol: float | None = None,
    angle_tol: float = 0.05,
) -> tuple[list[LineString], int]:
    """
    Bridge small gaps between collinear line segments.

    Two endpoints that are (a) within *bridge_tol*, (b) on collinear
    lines, and (c) not separated by a crossing wall are joined with a
    new segment.

    Returns ``(bridged_lines, bridge_count)``.
    """
    if bridge_tol is None:
        bridge_tol = 0.15 * pt_per_inch  # ~10.8 pt at 72 ppi

    if len(lines) < 2:
        return list(lines), 0

    # Collect endpoints: (coord, line_idx, end_flag)
    ep_data: list[tuple[tuple[float, float], int, int]] = []
    for i, line in enumerate(lines):
        coords = list(line.coords)
        ep_data.append((coords[0], i, 0))
        ep_data.append((coords[-1], i, 1))

    ep_coords = np.array([e[0] for e in ep_data])
    tree = cKDTree(ep_coords)

    bridges: list[LineString] = []
    bridged_pairs: set[tuple[int, int]] = set()

    for idx in range(len(ep_data)):
        pt, line_i, _end_i = ep_data[idx]
        nearby = tree.query_ball_point(pt, bridge_tol)
        for jdx in nearby:
            pt_j, line_j, _end_j = ep_data[jdx]
            if line_i >= line_j:
                continue
            pair = (line_i, line_j)
            if pair in bridged_pairs:
                continue
            dist = math.dist(pt, pt_j)
            if dist < 0.01 or dist >= bridge_tol:
                continue
            if not _are_collinear(lines[line_i], lines[line_j], angle_tol):
                continue
            # Anti-hallway guard: reject if another line crosses the gap
            bridge = LineString([pt, pt_j])
            crosses = any(
                lines[k].crosses(bridge)
                for k in range(len(lines))
                if k != line_i and k != line_j
            )
            if crosses:
                continue
            bridges.append(bridge)
            bridged_pairs.add(pair)

    if not bridges:
        return list(lines), 0

    merged = unary_union(lines + bridges)
    if isinstance(merged, MultiLineString):
        merged = linemerge(merged)
    return _extract_linestrings(merged), len(bridges)


# ── Phase 4.4: Short-stub Cleanup ───────────────────────────────────

def phase_4_4_stub_cleanup(
    lines: list[LineString],
    pt_per_inch: float = 72.0,
    min_stub_length: float | None = None,
) -> tuple[list[LineString], int]:
    """
    Remove isolated short line segments (stubs).

    A stub is a line shorter than *min_stub_length* whose **both**
    endpoints are degree-1 (not shared with any other line).  This is
    intentionally conservative — only truly disconnected noise is
    removed.

    Returns ``(cleaned_lines, stubs_removed)``.
    """
    if min_stub_length is None:
        min_stub_length = 0.15 * pt_per_inch  # ~10.8 pt at 72 ppi

    if not lines:
        return list(lines), 0

    removed = 0
    result = list(lines)
    prev_count = -1

    # Iterate until stable (removing stubs may reveal new stubs)
    while len(result) != prev_count:
        prev_count = len(result)

        # Count how many lines touch each rounded endpoint
        ep_count: dict[tuple[float, float], int] = defaultdict(int)
        for line in result:
            coords = list(line.coords)
            ep_count[(round(coords[0][0], 1), round(coords[0][1], 1))] += 1
            ep_count[(round(coords[-1][0], 1), round(coords[-1][1], 1))] += 1

        kept: list[LineString] = []
        for line in result:
            if line.length >= min_stub_length:
                kept.append(line)
                continue
            coords = list(line.coords)
            s = (round(coords[0][0], 1), round(coords[0][1], 1))
            e = (round(coords[-1][0], 1), round(coords[-1][1], 1))
            if ep_count[s] <= 1 and ep_count[e] <= 1:
                removed += 1
            else:
                kept.append(line)

        result = kept

    return result, removed


# ── Main consolidation entry point ──────────────────────────────────

def consolidate_walls(
    classification_result: dict[str, Any],
    pt_per_inch: float = 72.0,
) -> dict[str, Any]:
    """
    Run all four consolidation phases on a Step 3 classification result.

    Parameters
    ----------
    classification_result : dict
        Must contain ``wall_primitives``, ``candidate_id``, ``bounding_box``.
    pt_per_inch : float
        PDF points per drawing-inch (default 72, standard PDF).

    Returns
    -------
    dict with ``candidate_id``, ``bounding_box``, ``edges``, ``polygons``,
    ``stats``.
    """
    wall_prims = classification_result.get("wall_primitives", {})
    input_count = sum(len(v) for v in wall_prims.values())

    # Phase 4.1 — Flatten + Merge
    lines, polys = phase_4_1_flatten_merge(wall_prims, pt_per_inch)
    post_merge_count = len(lines)

    # Phase 4.2 — T-junction Snap
    lines, snap_count = phase_4_2_t_junction_snap(lines, pt_per_inch)

    # Phase 4.3 — Collinear Bridge
    lines, bridge_count = phase_4_3_collinear_bridge(lines, pt_per_inch)

    # Phase 4.4 — Short-stub Cleanup
    lines, stub_count = phase_4_4_stub_cleanup(lines, pt_per_inch)

    # Serialize edges
    edges = []
    for line in lines:
        coords = [list(c) for c in line.coords]
        edges.append({"coords": coords, "length": line.length})
    edges.sort(key=lambda e: -e["length"])

    # Serialize polygons
    poly_out = []
    for poly in polys:
        coords = [list(c) for c in poly.exterior.coords]
        poly_out.append({"coords": coords, "area": poly.area})
    poly_out.sort(key=lambda p: -p["area"])

    return {
        "candidate_id":  classification_result.get("candidate_id"),
        "bounding_box":  classification_result.get("bounding_box"),
        "edges":         edges,
        "polygons":      poly_out,
        "stats": {
            "input_primitive_count":  input_count,
            "post_merge_line_count":  post_merge_count,
            "output_edge_count":      len(edges),
            "output_polygon_count":   len(poly_out),
            "total_edge_length":      sum(e["length"] for e in edges),
            "total_polygon_area":     sum(p["area"] for p in poly_out),
            "t_junctions_snapped":    snap_count,
            "bridges_added":          bridge_count,
            "stubs_removed":          stub_count,
        },
    }


# ── Overlay rendering ───────────────────────────────────────────────

def render_consolidation_overlay(
    pdf_path: str,
    candidate: dict,
    consolidated: dict,
    output_path: str,
    page_number: int = 0,
    render_scale: float = 2.0,
    padding: float = 30.0,
) -> None:
    """
    Visual overlay: green wall edges / filled-green wall polygons on a
    faded PDF page, cropped to the candidate bounding box.
    """
    import fitz  # lazy import — avoids numpy ABI warning at module level

    doc  = fitz.open(pdf_path)
    page = doc[page_number]

    # Fade background
    bg = page.new_shape()
    bg.draw_rect(page.rect)
    bg.finish(fill=(1, 1, 1), fill_opacity=0.70)
    bg.commit()

    GREEN = (0.0, 0.8, 0.0)

    # Consolidated edges
    shape = page.new_shape()
    for edge in consolidated["edges"]:
        coords = edge["coords"]
        for k in range(len(coords) - 1):
            shape.draw_line(fitz.Point(coords[k]), fitz.Point(coords[k + 1]))
            shape.finish(color=GREEN, width=2.0)
    shape.commit()

    # Consolidated polygons (filled)
    shape = page.new_shape()
    for poly in consolidated["polygons"]:
        pts = poly["coords"]
        if len(pts) >= 3:
            fp = [fitz.Point(c) for c in pts]
            shape.draw_polyline(fp + [fp[0]])
            shape.finish(color=GREEN, fill=GREEN,
                         width=0.5, fill_opacity=0.5)
    shape.commit()

    # Crop to candidate bbox
    bb = candidate["bounding_box"]
    clip = fitz.Rect(
        bb[0] - padding, bb[1] - padding,
        bb[2] + padding, bb[3] + padding,
    ) & page.rect

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    page.get_pixmap(
        matrix=fitz.Matrix(render_scale, render_scale),
        clip=clip, alpha=False,
    ).save(output_path)
    doc.close()


# ── I/O helpers ─────────────────────────────────────────────────────

def load_classification(plan_name: str, candidate_id: int) -> dict:
    """Load a specific candidate from Step 3's classifications.json."""
    path = os.path.join(
        PROJECT_ROOT, "output", plan_name, "step3", "classifications.json",
    )
    with open(path) as f:
        results = json.load(f)
    for r in results:
        if r["candidate_id"] == candidate_id:
            return r
    raise ValueError(f"candidate {candidate_id} not found in {path}")


# ── Runner ──────────────────────────────────────────────────────────

KNOWN_PDF_PATHS: dict[str, str] = {
    "2026-03-27_Test-2_v4": os.path.join(
        PROJECT_ROOT, "example_plans", "2026-03-27_Test-2_v4.pdf",
    ),
    "2026.01.29_-_FMOC_A_-_50%_CD_copy": os.path.join(
        PROJECT_ROOT, "example_plans",
        "2026.01.29 - FMOC_A - 50% CD copy.pdf",
    ),
    "main_st_ex": os.path.join(
        PROJECT_ROOT, "example_plans", "main_st_ex.pdf",
    ),
}


def run(
    plan_name: str | None = None,
    candidate_id: int | None = None,
    pt_per_inch: float = 72.0,
    pdf_path: str | None = None,
    output_dir: str | None = None,
) -> list[dict]:
    """
    Run Step 4 consolidation.

    If *plan_name* is ``None``, processes every entry in
    ``HARDCODED_GOOD``.
    """
    targets: dict[str, int] = {}
    if plan_name is not None:
        if candidate_id is None:
            candidate_id = HARDCODED_GOOD.get(plan_name)
            if candidate_id is None:
                raise ValueError(
                    f"No hardcoded candidate for '{plan_name}'; "
                    "specify candidate_id explicitly"
                )
        targets[plan_name] = candidate_id
    else:
        targets = dict(HARDCODED_GOOD)

    all_results: list[dict] = []

    for pname, cid in targets.items():
        print(f"\nStep 4: consolidating {pname} candidate {cid}")
        classification = load_classification(pname, cid)
        result = consolidate_walls(classification, pt_per_inch)

        # Output directory
        out_dir = output_dir or os.path.join(
            PROJECT_ROOT, "output", pname, "step4",
        )
        os.makedirs(out_dir, exist_ok=True)

        # Save JSON
        out_json = os.path.join(out_dir, f"candidate_{cid}_consolidated.json")
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

        s = result["stats"]
        print(f"  input:  {s['input_primitive_count']} primitives")
        print(
            f"  merge:  {s['post_merge_line_count']} lines, "
            f"{s['output_polygon_count']} polygons"
        )
        print(
            f"  snap={s['t_junctions_snapped']}  bridge={s['bridges_added']}  "
            f"stubs_removed={s['stubs_removed']}"
        )
        print(
            f"  output: {s['output_edge_count']} edges "
            f"({s['total_edge_length']:.0f} pt), "
            f"{s['output_polygon_count']} polygons "
            f"({s['total_polygon_area']:.0f} pt²)"
        )
        print(f"  saved:  {out_json}")

        # Render overlay if we can find the PDF
        resolved_pdf = pdf_path or KNOWN_PDF_PATHS.get(pname)
        if resolved_pdf and os.path.isfile(resolved_pdf):
            overlay_path = os.path.join(
                out_dir, f"candidate_{cid}_overlay.png",
            )
            render_consolidation_overlay(
                resolved_pdf, classification, result, overlay_path,
            )
            print(f"  overlay: {overlay_path}")

        all_results.append(result)

    return all_results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Running all hardcoded-good candidates …")
        run()
    else:
        run(
            plan_name=sys.argv[1],
            candidate_id=int(sys.argv[2]) if len(sys.argv) > 2 else None,
        )
