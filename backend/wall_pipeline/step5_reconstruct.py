#!/usr/bin/env python3
"""
Step 5 -- Wall Polygon Reconstruction
======================================

Takes consolidated wall geometry from Step 4 and produces filled wall
rectangles — the format takeoff software expects.

1. **Decompose** polyline edges into individual straight segments.
2. **Pair** parallel segments that are close together (the two sides
   of a wall) → filled rectangle between them.
3. **Offset** unpaired segments by a default wall thickness →
   single-edge wall rectangle.
4. **Pass through** existing polygons from Step 4 (already filled shapes).
5. **Measure** linear footage (LF) per wall and total.

Output: wall polygon list with LF, thickness, and source metadata.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

# ── constants ────────────────────────────────────────────────────────

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, ".."))

HARDCODED_GOOD: dict[str, int] = {
    "2026-03-27_Test-2_v4":                2,
    "2026.01.29_-_FMOC_A_-_50%_CD_copy":  1,
    "main_st_ex":                          1,
}

# PDF paths for overlay rendering
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


# ── 1. Segment Decomposition ────────────────────────────────────────

def decompose_edges_to_segments(edges: list[dict]) -> list[dict]:
    """
    Break Step 4 polyline edges into individual straight segments.

    Each segment gets: start, end, length, angle (0..π), midpoint,
    edge_idx (which edge it came from).
    """
    segments: list[dict] = []
    for idx, edge in enumerate(edges):
        coords = edge["coords"]
        for i in range(len(coords) - 1):
            sx, sy = coords[i]
            ex, ey = coords[i + 1]
            dx, dy = ex - sx, ey - sy
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            angle = math.atan2(dy, dx) % math.pi  # normalize to [0, π)
            segments.append({
                "start":    [sx, sy],
                "end":      [ex, ey],
                "length":   length,
                "angle":    angle,
                "midpoint": [(sx + ex) / 2, (sy + ey) / 2],
                "edge_idx": idx,
            })
    return segments


# ── 2. Parallel Pairing ─────────────────────────────────────────────

def _perp_distance(seg_a: dict, seg_b: dict) -> float:
    """Perpendicular distance between two parallel segments' midpoints."""
    angle = seg_a["angle"]
    # Perpendicular direction
    nx, ny = -math.sin(angle), math.cos(angle)
    dx = seg_b["midpoint"][0] - seg_a["midpoint"][0]
    dy = seg_b["midpoint"][1] - seg_a["midpoint"][1]
    return abs(dx * nx + dy * ny)


def _parallel_overlap(seg_a: dict, seg_b: dict) -> float:
    """
    Overlap ratio along the parallel direction.

    Projects both segments onto the dominant axis and returns the ratio
    of overlap length to the shorter segment's length.
    """
    angle = seg_a["angle"]
    ax, ay = math.cos(angle), math.sin(angle)

    def project(seg):
        s = seg["start"][0] * ax + seg["start"][1] * ay
        e = seg["end"][0] * ax + seg["end"][1] * ay
        return (min(s, e), max(s, e))

    a_lo, a_hi = project(seg_a)
    b_lo, b_hi = project(seg_b)

    overlap = max(0, min(a_hi, b_hi) - max(a_lo, b_lo))
    shorter = min(a_hi - a_lo, b_hi - b_lo)
    if shorter < 1e-6:
        return 0.0
    return overlap / shorter


def pair_parallel_segments(
    segments: list[dict],
    max_thickness: float = 50.0,
    min_overlap: float = 0.3,
    angle_tol: float = 0.08,
) -> tuple[list[dict], list[dict]]:
    """
    Find pairs of parallel segments that form wall sides.

    Returns ``(pairs, unpaired)`` where each pair is a dict with
    ``seg_a``, ``seg_b``, ``thickness``.
    """
    if not segments:
        return [], []

    n = len(segments)
    midpoints = np.array([s["midpoint"] for s in segments])
    tree = cKDTree(midpoints)

    paired_idxs: set[int] = set()
    pairs: list[dict] = []

    # Sort longest first — prefer pairing long segments
    order = sorted(range(n), key=lambda i: -segments[i]["length"])

    for i in order:
        if i in paired_idxs:
            continue
        seg_a = segments[i]
        # Search within max_thickness radius
        candidates = tree.query_ball_point(seg_a["midpoint"],
                                           max_thickness + seg_a["length"] / 2)

        best_j: int | None = None
        best_dist = max_thickness + 1

        for j in candidates:
            if j == i or j in paired_idxs:
                continue
            seg_b = segments[j]

            # Angle check
            adiff = abs(seg_a["angle"] - seg_b["angle"])
            adiff = min(adiff, math.pi - adiff)
            if adiff > angle_tol:
                continue

            # Perpendicular distance = wall thickness
            perp = _perp_distance(seg_a, seg_b)
            if perp < 0.5 or perp > max_thickness:
                continue

            # Overlap check
            if _parallel_overlap(seg_a, seg_b) < min_overlap:
                continue

            if perp < best_dist:
                best_dist = perp
                best_j = j

        if best_j is not None:
            paired_idxs.add(i)
            paired_idxs.add(best_j)
            pairs.append({
                "seg_a":     segments[i],
                "seg_b":     segments[best_j],
                "thickness": best_dist,
            })

    unpaired = [segments[i] for i in range(n) if i not in paired_idxs]
    return pairs, unpaired


# ── 3. Wall Polygon Construction ────────────────────────────────────

def build_wall_polygon(seg_a: dict, seg_b: dict) -> Polygon:
    """
    Build a filled rectangle from two paired parallel segments.

    Clips both segments to their overlapping region along the parallel
    axis so the result is a proper rectangle (no diagonal connecting
    edges from misaligned endpoints).
    """
    angle = seg_a["angle"]
    ax, ay = math.cos(angle), math.sin(angle)
    # Perpendicular direction
    nx, ny = -math.sin(angle), math.cos(angle)

    def project_along(pt):
        return pt[0] * ax + pt[1] * ay

    # Find overlap region along the parallel axis
    a_s, a_e = project_along(seg_a["start"]), project_along(seg_a["end"])
    b_s, b_e = project_along(seg_b["start"]), project_along(seg_b["end"])
    a_lo, a_hi = min(a_s, a_e), max(a_s, a_e)
    b_lo, b_hi = min(b_s, b_e), max(b_s, b_e)

    clip_lo = max(a_lo, b_lo)
    clip_hi = min(a_hi, b_hi)

    if clip_hi <= clip_lo:
        # No overlap — fallback to raw endpoints
        pts = [seg_a["start"], seg_a["end"],
               seg_b["end"], seg_b["start"]]
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly

    # Project each segment's centerline onto the perpendicular axis
    perp_a = seg_a["midpoint"][0] * nx + seg_a["midpoint"][1] * ny
    perp_b = seg_b["midpoint"][0] * nx + seg_b["midpoint"][1] * ny

    # Build a clean rectangle from the clipped overlap
    pts = [
        [clip_lo * ax + perp_a * nx, clip_lo * ay + perp_a * ny],
        [clip_hi * ax + perp_a * nx, clip_hi * ay + perp_a * ny],
        [clip_hi * ax + perp_b * nx, clip_hi * ay + perp_b * ny],
        [clip_lo * ax + perp_b * nx, clip_lo * ay + perp_b * ny],
    ]
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def offset_segment_to_wall(seg: dict, thickness: float = 6.0) -> Polygon:
    """
    Create a wall rectangle by offsetting a single segment by *thickness*.

    The offset is applied symmetrically (half to each side of the segment).
    """
    angle = seg["angle"]
    # Perpendicular direction
    nx, ny = -math.sin(angle), math.cos(angle)
    half = thickness / 2

    sx, sy = seg["start"]
    ex, ey = seg["end"]

    pts = [
        [sx + nx * half, sy + ny * half],
        [ex + nx * half, ey + ny * half],
        [ex - nx * half, ey - ny * half],
        [sx - nx * half, sy - ny * half],
    ]
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


# ── 4. Polygon passthrough helpers ──────────────────────────────────

def _polygon_lf(coords: list[list[float]]) -> tuple[float, float]:
    """
    Estimate the linear footage of a wall polygon.

    Returns ``(lf, thickness)`` — LF is the longer pair of parallel
    sides; thickness is the shorter pair.
    """
    if len(coords) < 4:
        return 0.0, 0.0
    # Compute side lengths
    sides = []
    pts = coords if coords[0] != coords[-1] else coords[:-1]
    n = len(pts)
    for i in range(n):
        dx = pts[(i + 1) % n][0] - pts[i][0]
        dy = pts[(i + 1) % n][1] - pts[i][1]
        sides.append(math.hypot(dx, dy))

    if not sides:
        return 0.0, 0.0

    sides.sort(reverse=True)
    # For a rectangle: 2 long sides + 2 short sides
    # LF = average of the two longest sides
    lf = (sides[0] + sides[1]) / 2 if len(sides) >= 2 else sides[0]
    thickness = sides[-1] if len(sides) >= 4 else 0.0
    return lf, thickness


# ── 5. Main reconstruction entry point ──────────────────────────────

def reconstruct_walls(
    consolidated: dict[str, Any],
    pt_per_inch: float = 72.0,
    max_wall_thickness: float | None = None,
    default_thickness: float | None = None,
    min_segment_length: float = 5.0,
) -> dict[str, Any]:
    """
    Reconstruct filled wall polygons from Step 4 consolidated geometry.

    Parameters
    ----------
    consolidated : dict
        Step 4 output with ``edges``, ``polygons``, ``candidate_id``,
        ``bounding_box``.
    pt_per_inch : float
        PDF points per inch (default 72).
    max_wall_thickness : float or None
        Maximum perpendicular distance to consider as a wall pair.
        Defaults to ``0.5 * pt_per_inch`` (~36 pt).
    default_thickness : float or None
        Thickness for single-edge offset walls.
        Defaults to ``0.1 * pt_per_inch`` (~7.2 pt).
    min_segment_length : float
        Segments shorter than this are skipped.

    Returns
    -------
    dict with ``candidate_id``, ``bounding_box``, ``walls``,
    ``total_lf_pt``, ``total_lf_inches``, ``stats``.
    """
    if max_wall_thickness is None:
        max_wall_thickness = 0.5 * pt_per_inch
    if default_thickness is None:
        default_thickness = 0.1 * pt_per_inch

    edges = consolidated.get("edges", [])
    polygons = consolidated.get("polygons", [])

    walls: list[dict] = []

    # ---- Decompose edges → segments --------------------------------
    all_segs = decompose_edges_to_segments(edges)
    # Filter very short segments
    all_segs = [s for s in all_segs if s["length"] >= min_segment_length]

    # ---- Pair parallel segments ------------------------------------
    pairs, unpaired = pair_parallel_segments(
        all_segs,
        max_thickness=max_wall_thickness,
        min_overlap=0.3,
    )

    paired_count = 0
    for pair in pairs:
        poly = build_wall_polygon(pair["seg_a"], pair["seg_b"])
        if poly.is_valid and poly.area > 0:
            lf = (pair["seg_a"]["length"] + pair["seg_b"]["length"]) / 2
            walls.append({
                "polygon":      [list(c) for c in poly.exterior.coords],
                "thickness":    pair["thickness"],
                "length_pt":    lf,
                "length_inches": lf / pt_per_inch,
                "source":       "paired",
            })
            paired_count += 1

    # ---- Offset unpaired segments ----------------------------------
    offset_count = 0
    for seg in unpaired:
        poly = offset_segment_to_wall(seg, thickness=default_thickness)
        if poly.is_valid and poly.area > 0:
            walls.append({
                "polygon":      [list(c) for c in poly.exterior.coords],
                "thickness":    default_thickness,
                "length_pt":    seg["length"],
                "length_inches": seg["length"] / pt_per_inch,
                "source":       "offset",
            })
            offset_count += 1

    # ---- Polygon passthrough ---------------------------------------
    polygon_count = 0
    for poly_data in polygons:
        coords = poly_data["coords"]
        lf, thickness = _polygon_lf(coords)
        walls.append({
            "polygon":      coords,
            "thickness":    thickness,
            "length_pt":    lf,
            "length_inches": lf / pt_per_inch,
            "source":       "polygon",
        })
        polygon_count += 1

    # ---- Totals ----------------------------------------------------
    total_lf_pt = sum(w["length_pt"] for w in walls)

    return {
        "candidate_id":  consolidated.get("candidate_id"),
        "bounding_box":  consolidated.get("bounding_box"),
        "walls":         walls,
        "total_lf_pt":   total_lf_pt,
        "total_lf_inches": total_lf_pt / pt_per_inch,
        "stats": {
            "paired_count":     paired_count,
            "offset_count":     offset_count,
            "polygon_count":    polygon_count,
            "total_wall_count": paired_count + offset_count + polygon_count,
        },
    }


# ── Overlay rendering ───────────────────────────────────────────────

def render_wall_overlay(
    pdf_path: str,
    consolidated: dict,
    walls_result: dict,
    output_path: str,
    page_number: int = 0,
    render_scale: float = 2.0,
    padding: float = 30.0,
) -> None:
    """
    Takeoff-style overlay: filled green wall rectangles on faded PDF.
    """
    import fitz

    doc  = fitz.open(pdf_path)
    page = doc[page_number]

    # Fade background
    bg = page.new_shape()
    bg.draw_rect(page.rect)
    bg.finish(fill=(1, 1, 1), fill_opacity=0.70)
    bg.commit()

    GREEN = (0.0, 0.75, 0.0)

    for wall in walls_result["walls"]:
        pts = wall["polygon"]
        if len(pts) < 3:
            continue
        shape = page.new_shape()
        fp = [fitz.Point(c[0], c[1]) for c in pts]
        # Close the polygon
        if fp[0] != fp[-1]:
            fp.append(fp[0])
        shape.draw_polyline(fp)
        shape.finish(color=GREEN, fill=GREEN,
                     width=0.3, fill_opacity=1.0)
        shape.commit()

    # Crop to candidate bbox
    bb = consolidated["bounding_box"]
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

def load_consolidated(plan_name: str, candidate_id: int) -> dict:
    """Load Step 4 consolidated result for a candidate."""
    path = os.path.join(
        PROJECT_ROOT, "output", plan_name, "step4",
        f"candidate_{candidate_id}_consolidated.json",
    )
    with open(path) as f:
        return json.load(f)


# ── Runner ──────────────────────────────────────────────────────────

def run(
    plan_name: str | None = None,
    candidate_id: int | None = None,
    pt_per_inch: float = 72.0,
    pdf_path: str | None = None,
    output_dir: str | None = None,
) -> list[dict]:
    """
    Run Step 5 wall reconstruction.

    If *plan_name* is ``None``, processes every entry in
    ``HARDCODED_GOOD``.
    """
    targets: dict[str, int] = {}
    if plan_name is not None:
        if candidate_id is None:
            candidate_id = HARDCODED_GOOD.get(plan_name)
            if candidate_id is None:
                raise ValueError(
                    f"No hardcoded candidate for '{plan_name}'"
                )
        targets[plan_name] = candidate_id
    else:
        targets = dict(HARDCODED_GOOD)

    all_results: list[dict] = []

    for pname, cid in targets.items():
        print(f"\nStep 5: reconstructing walls — {pname} candidate {cid}")
        consolidated = load_consolidated(pname, cid)
        result = reconstruct_walls(consolidated, pt_per_inch)

        out_dir = output_dir or os.path.join(
            PROJECT_ROOT, "output", pname, "step5",
        )
        os.makedirs(out_dir, exist_ok=True)

        out_json = os.path.join(out_dir, f"candidate_{cid}_walls.json")
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)

        s = result["stats"]
        print(f"  walls: {s['total_wall_count']} "
              f"(paired={s['paired_count']}, offset={s['offset_count']}, "
              f"polygon={s['polygon_count']})")
        print(f"  total LF: {result['total_lf_pt']:.0f} pt "
              f"= {result['total_lf_inches']:.1f} in")
        print(f"  saved: {out_json}")

        resolved_pdf = pdf_path or KNOWN_PDF_PATHS.get(pname)
        if resolved_pdf and os.path.isfile(resolved_pdf):
            overlay_path = os.path.join(
                out_dir, f"candidate_{cid}_wall_overlay.png",
            )
            render_wall_overlay(
                resolved_pdf, consolidated, result, overlay_path,
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
