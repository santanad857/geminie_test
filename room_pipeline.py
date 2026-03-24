#!/usr/bin/env python3
"""
Vector Ray-Casting Room Detection Pipeline
===========================================

Deterministic, vector-based pipeline that:
1. Extracts orthogonal wall border segments from the PDF via wall_pipeline.
2. Identifies exposed (degree-1) endpoints — the doorjambs.
3. Casts bounded rays from each jamb to seal doorway openings.
4. Assembles closed polygons via graph cycle detection.
5. Labels rooms using Claude Vision + point-in-polygon.

Usage::

    python room_pipeline.py --pdf plan.pdf [--scale 96] [--max-door-ft 20]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import anthropic
import cv2
import fitz  # PyMuPDF
import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

import wall_pipeline


# ═══════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class WallSegment:
    """An orthogonal wall-border line in PDF-point space."""
    p1: tuple[float, float]
    p2: tuple[float, float]
    source: str = "border"  # 'border' | 'ray'


@dataclass
class ExposedEndpoint:
    """A degree-1 endpoint (doorjamb) with its outward direction."""
    point: tuple[float, float]
    direction: tuple[float, float]   # unit cardinal vector
    seg_idx: int                     # index into wall segment list


@dataclass
class RoomPolygon:
    """A closed room polygon with optional label."""
    room_id: int
    name: str
    vertices: list[tuple[float, float]]
    area_sqpt: float
    area_sqft: float
    color: tuple[float, float, float]
    centroid: tuple[float, float]


_PALETTE: list[tuple[float, float, float]] = [
    (0.93, 0.35, 0.35), (0.30, 0.68, 0.97), (0.36, 0.85, 0.36),
    (0.97, 0.78, 0.22), (0.73, 0.33, 0.85), (0.97, 0.56, 0.22),
    (0.22, 0.87, 0.87), (0.87, 0.47, 0.68), (0.58, 0.78, 0.40),
    (0.46, 0.38, 0.76), (0.87, 0.68, 0.47), (0.38, 0.72, 0.60),
]


# ═══════════════════════════════════════════════════════════════════════════
#  Step 0 — Wall Segment Extraction
# ═══════════════════════════════════════════════════════════════════════════


def _vlm_seed_in_region(
    page: fitz.Page,
    clip_rect: fitz.Rect,
    dpi: int = 150,
) -> fitz.Rect:
    """Send a CROPPED region of the page to Claude VLM so it seeds a
    wall inside the main floor plan, not the option-plan insets.

    Returns a seed ``fitz.Rect`` in **full-page** PDF-point coordinates.
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
    img_bytes = pix.tobytes("png")
    img_b64 = base64.b64encode(img_bytes).decode()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = (
        f"This is a CROPPED portion of an architectural floor plan "
        f"({pix.width} x {pix.height} pixels).\n\n"
        "Identify ONE long, straight wall segment that is clearly "
        "visible, at least ~80 pixels long, and away from door or "
        "window openings.\n\n"
        "Return ONLY a JSON object with the bounding box in pixel "
        "coordinates:\n"
        '{"x1": <left>, "y1": <top>, "x2": <right>, "y2": <bottom>}\n\n'
        "Return ONLY the JSON object, nothing else."
    )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": img_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )

    text = resp.content[0].text
    m = re.search(r"\{[^}]+\}", text)
    if not m:
        raise ValueError(f"VLM did not return JSON: {text}")
    bbox = json.loads(m.group())

    # Clamp to cropped image bounds.
    bbox["x1"] = max(0, min(bbox["x1"], pix.width))
    bbox["y1"] = max(0, min(bbox["y1"], pix.height))
    bbox["x2"] = max(0, min(bbox["x2"], pix.width))
    bbox["y2"] = max(0, min(bbox["y2"], pix.height))

    print(f"  VLM cropped-pixel bbox: ({bbox['x1']},{bbox['y1']})"
          f" -> ({bbox['x2']},{bbox['y2']})")

    # Map cropped pixels → full-page PDF points.
    s = 72.0 / dpi
    return fitz.Rect(
        clip_rect.x0 + bbox["x1"] * s,
        clip_rect.y0 + bbox["y1"] * s,
        clip_rect.x0 + bbox["x2"] * s,
        clip_rect.y0 + bbox["y2"] * s,
    )


def _merge_colinear_segments(
    segments: list[WallSegment],
    tol: float = 1.5,
) -> list[WallSegment]:
    """Merge co-linear, overlapping or adjacent orthogonal segments.

    Groups horizontal segments by rounded y-coordinate and vertical
    segments by rounded x-coordinate, then merges overlapping intervals.
    """
    from collections import defaultdict

    h_groups: dict[float, list[tuple[float, float]]] = defaultdict(list)
    v_groups: dict[float, list[tuple[float, float]]] = defaultdict(list)

    for seg in segments:
        x1, y1 = seg.p1
        x2, y2 = seg.p2
        if abs(y2 - y1) < tol:  # horizontal
            key = round((y1 + y2) / 2.0 / tol) * tol
            h_groups[key].append((min(x1, x2), max(x1, x2)))
        elif abs(x2 - x1) < tol:  # vertical
            key = round((x1 + x2) / 2.0 / tol) * tol
            v_groups[key].append((min(y1, y2), max(y1, y2)))

    merged: list[WallSegment] = []

    for y_key, intervals in h_groups.items():
        intervals.sort()
        cur_lo, cur_hi = intervals[0]
        for lo, hi in intervals[1:]:
            if lo <= cur_hi + tol:
                cur_hi = max(cur_hi, hi)
            else:
                if cur_hi - cur_lo > tol:
                    merged.append(WallSegment(p1=(cur_lo, y_key), p2=(cur_hi, y_key)))
                cur_lo, cur_hi = lo, hi
        if cur_hi - cur_lo > tol:
            merged.append(WallSegment(p1=(cur_lo, y_key), p2=(cur_hi, y_key)))

    for x_key, intervals in v_groups.items():
        intervals.sort()
        cur_lo, cur_hi = intervals[0]
        for lo, hi in intervals[1:]:
            if lo <= cur_hi + tol:
                cur_hi = max(cur_hi, hi)
            else:
                if cur_hi - cur_lo > tol:
                    merged.append(WallSegment(p1=(x_key, cur_lo), p2=(x_key, cur_hi)))
                cur_lo, cur_hi = lo, hi
        if cur_hi - cur_lo > tol:
            merged.append(WallSegment(p1=(x_key, cur_lo), p2=(x_key, cur_hi)))

    return merged


def extract_wall_segments(
    pdf_path: str,
    page_index: int = 0,
    focus_rect: Optional[tuple[float, float, float, float]] = None,
    width_range: tuple[float, float] = (0.65, 0.95),
    ortho_tol_deg: float = 5.0,
) -> list[WallSegment]:
    """Extract wall border segments directly from the PDF vector data.

    Collects all black, orthogonal line segments whose stroke width falls
    within *width_range* (targeting the 0.84 pt wall-border lines), then
    merges co-linear connected segments into long wall-face lines.

    No VLM or fingerprint matching is needed — this is a pure geometric
    extraction.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    drawings = page.get_drawings()
    doc.close()

    def _extract_ortho(w_min: float, w_max: float) -> list[WallSegment]:
        """Pull orthogonal segments within a width band."""
        out: list[WallSegment] = []
        for d in drawings:
            if d.get("color") != (0.0, 0.0, 0.0):
                continue
            w = d.get("width") or 0
            if w < w_min or w > w_max:
                continue
            for item in d["items"]:
                if item[0] != "l":
                    continue
                x1, y1 = item[1].x, item[1].y
                x2, y2 = item[2].x, item[2].y
                if focus_rect is not None:
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    if not (focus_rect[0] <= mx <= focus_rect[2] and
                            focus_rect[1] <= my <= focus_rect[3]):
                        continue
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                if length < 0.5:
                    continue
                angle = math.degrees(math.atan2(abs(dy), abs(dx)))
                if angle > ortho_tol_deg and angle < (90 - ortho_tol_deg):
                    continue
                if angle <= ortho_tol_deg:
                    y2 = y1
                else:
                    x2 = x1
                out.append(WallSegment(p1=(x1, y1), p2=(x2, y2)))
        return out

    # ── Tier 1: heavy wall borders (0.84 pt) ────────────────────────
    w_min, w_max = width_range
    primary = _extract_ortho(w_min, w_max)
    print(f"  Tier-1 (wall borders {w_min}–{w_max} pt): {len(primary)}")

    # ── Tier 2: thinner lines (0.20–0.30 pt) kept ONLY if an endpoint
    #    is adjacent to a Tier-1 segment endpoint.  This captures wall
    #    outline continuations while rejecting standalone dimension lines.
    thin = _extract_ortho(0.20, 0.30)
    if primary and thin:
        # Build KDTree of all Tier-1 endpoints.
        t1_pts = []
        for s in primary:
            t1_pts.append(s.p1)
            t1_pts.append(s.p2)
        t1_arr = np.array(t1_pts, dtype=np.float64)
        t1_tree = KDTree(t1_arr)

        adjacency_radius = 5.0  # PDF points
        kept = 0
        for s in thin:
            d1, _ = t1_tree.query(s.p1)
            d2, _ = t1_tree.query(s.p2)
            if d1 < adjacency_radius or d2 < adjacency_radius:
                primary.append(s)
                kept += 1
        print(f"  Tier-2 (thin lines near walls): {kept} kept / {len(thin)} total")

    segments = _merge_colinear_segments(primary, tol=1.5)
    print(f"  After co-linear merge: {len(segments)}")
    return segments


def extract_walls_via_wall_pipeline(
    pdf_path: str,
    page_index: int = 0,
    ortho_tol_deg: float = 5.0,
) -> list[WallSegment]:
    """Extract wall border segments using wall_pipeline's VLM-guided
    fingerprinting, then convert to orthogonal WallSegment objects.

    This leverages wall_pipeline's accurate hatch+border detection to
    avoid picking up dimension lines and other non-wall geometry.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    drawings = page.get_drawings()

    # ── VLM seed + fingerprint (with retry, same as wall_pipeline.main) ──
    MAX_VLM_ATTEMPTS = 3
    fp = None
    for attempt in range(1, MAX_VLM_ATTEMPTS + 1):
        print(f"  VLM wall seed attempt {attempt}/{MAX_VLM_ATTEMPTS} …")
        try:
            seed_rect = wall_pipeline.vlm_identify_wall(page, dpi=150)
            print(f"  Seed rect (PDF pts): {seed_rect}")
            fp = wall_pipeline.extract_fingerprint(drawings, seed_rect)
            print(f"  Hatch colour : {tuple(round(c, 4) for c in fp['hatch_color'])}")
            print(f"  Hatch angle  : {fp['hatch_angle']}°")
            print(f"  Border widths: {fp['border_widths']}")
            break
        except ValueError as e:
            print(f"  {e}")
            if attempt == MAX_VLM_ATTEMPTS:
                raise RuntimeError(
                    f"Could not extract wall fingerprint after {MAX_VLM_ATTEMPTS} attempts."
                )

    # ── Global match ─────────────────────────────────────────────────────
    hatches, borders = wall_pipeline.find_all_walls(drawings, fp)
    print(f"  Matched hatches: {len(hatches)}, borders: {len(borders)}")
    doc.close()

    # ── Extract segments from ALL wall_pipeline drawings ─────────────
    # Use every vector that wall_pipeline identified as a wall: both
    # hatches and borders.  Extract orthogonal line segments from their
    # drawing items, plus edges from hatch bounding rects.
    raw_segments: list[WallSegment] = []

    # Line segments from all matched drawings (hatches + borders).
    all_wall_drawings = list(hatches) + list(borders)
    for _, d in all_wall_drawings:
        for item in d["items"]:
            if item[0] == "l":
                x1, y1 = item[1].x, item[1].y
                x2, y2 = item[2].x, item[2].y
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                if length < 0.5:
                    continue
                angle = math.degrees(math.atan2(abs(dy), abs(dx)))
                if angle > ortho_tol_deg and angle < (90 - ortho_tol_deg):
                    continue
                if angle <= ortho_tol_deg:
                    y2 = y1
                else:
                    x2 = x1
                raw_segments.append(WallSegment(p1=(x1, y1), p2=(x2, y2)))
            elif item[0] == "re":
                r = item[1]
                raw_segments.append(WallSegment(p1=(r.x0, r.y0), p2=(r.x1, r.y0)))
                raw_segments.append(WallSegment(p1=(r.x1, r.y0), p2=(r.x1, r.y1)))
                raw_segments.append(WallSegment(p1=(r.x1, r.y1), p2=(r.x0, r.y1)))
                raw_segments.append(WallSegment(p1=(r.x0, r.y1), p2=(r.x0, r.y0)))

    print(f"  Orthogonal segments from wall drawings: {len(raw_segments)}")

    # Hatch bounding rect edges — infer wall faces from hatch extent.
    hatch_edge_segs: list[WallSegment] = []
    for _, d in hatches:
        r = d["rect"]
        rw = r.x1 - r.x0
        rh = r.y1 - r.y0
        if rw < 1.0 or rh < 1.0:
            continue
        if rw > rh * 2:
            hatch_edge_segs.append(WallSegment(p1=(r.x0, r.y0), p2=(r.x1, r.y0)))
            hatch_edge_segs.append(WallSegment(p1=(r.x0, r.y1), p2=(r.x1, r.y1)))
        elif rh > rw * 2:
            hatch_edge_segs.append(WallSegment(p1=(r.x0, r.y0), p2=(r.x0, r.y1)))
            hatch_edge_segs.append(WallSegment(p1=(r.x1, r.y0), p2=(r.x1, r.y1)))
        else:
            hatch_edge_segs.append(WallSegment(p1=(r.x0, r.y0), p2=(r.x1, r.y0)))
            hatch_edge_segs.append(WallSegment(p1=(r.x1, r.y0), p2=(r.x1, r.y1)))
            hatch_edge_segs.append(WallSegment(p1=(r.x1, r.y1), p2=(r.x0, r.y1)))
            hatch_edge_segs.append(WallSegment(p1=(r.x0, r.y1), p2=(r.x0, r.y0)))

    print(f"  Hatch rect edge segments: {len(hatch_edge_segs)}")
    print(f"  Total raw segments (borders only): {len(raw_segments)}")

    # Graph path: borders only (precise junction connectivity).
    merged_borders = _merge_colinear_segments(raw_segments, tol=1.5)
    print(f"  Borders after co-linear merge: {len(merged_borders)}")
    centerlines = _collapse_to_centerlines(merged_borders, max_wall_thickness=5.0)
    centerlines = _merge_colinear_segments(centerlines, tol=3.0)
    print(f"  Centerlines (for graph): {len(centerlines)}")

    # Raster path: borders + hatch edges (maximum wall coverage).
    all_raw = raw_segments + hatch_edge_segs
    merged_all = _merge_colinear_segments(all_raw, tol=1.5)
    print(f"  All merged (for raster fallback): {len(merged_all)}")

    return centerlines, merged_all


def _collapse_to_centerlines(
    segments: list[WallSegment],
    max_wall_thickness: float = 5.0,
) -> list[WallSegment]:
    """Collapse pairs of parallel wall border segments into single
    centerline segments.

    Architectural wall borders come as two parallel lines (inner and outer
    face).  This finds parallel pairs within *max_wall_thickness* distance
    that overlap along their length, and replaces each pair with one
    segment at the midpoint coordinate spanning the union of their ranges.
    """
    from collections import defaultdict

    # Separate into horizontal and vertical.
    h_segs: list[tuple[float, float, float, int]] = []  # (y, x_lo, x_hi, orig_idx)
    v_segs: list[tuple[float, float, float, int]] = []  # (x, y_lo, y_hi, orig_idx)
    other: list[WallSegment] = []

    for i, seg in enumerate(segments):
        dx = abs(seg.p2[0] - seg.p1[0])
        dy = abs(seg.p2[1] - seg.p1[1])
        if dy < 0.5 and dx > 0.5:  # horizontal
            y = (seg.p1[1] + seg.p2[1]) / 2
            h_segs.append((y, min(seg.p1[0], seg.p2[0]), max(seg.p1[0], seg.p2[0]), i))
        elif dx < 0.5 and dy > 0.5:  # vertical
            x = (seg.p1[0] + seg.p2[0]) / 2
            v_segs.append((x, min(seg.p1[1], seg.p2[1]), max(seg.p1[1], seg.p2[1]), i))
        else:
            other.append(seg)

    consumed: set[int] = set()  # original indices that got merged
    result: list[WallSegment] = []

    # ── Collapse horizontal pairs ────────────────────────────────────
    h_segs.sort(key=lambda s: s[0])  # sort by y
    used_h: set[int] = set()  # index into h_segs
    for i in range(len(h_segs)):
        if i in used_h:
            continue
        y_i, xlo_i, xhi_i, oi = h_segs[i]
        # Look for a parallel partner close in y.
        best_j = None
        best_overlap = 0
        for j in range(i + 1, len(h_segs)):
            if j in used_h:
                continue
            y_j, xlo_j, xhi_j, oj = h_segs[j]
            if y_j - y_i > max_wall_thickness:
                break  # sorted by y, no more candidates
            # Check x-overlap.
            overlap = min(xhi_i, xhi_j) - max(xlo_i, xlo_j)
            if overlap > min(xhi_i - xlo_i, xhi_j - xlo_j) * 0.5:
                if overlap > best_overlap:
                    best_j = j
                    best_overlap = overlap
        if best_j is not None:
            y_j, xlo_j, xhi_j, oj = h_segs[best_j]
            mid_y = (y_i + y_j) / 2
            merged_xlo = min(xlo_i, xlo_j)
            merged_xhi = max(xhi_i, xhi_j)
            result.append(WallSegment(p1=(merged_xlo, mid_y), p2=(merged_xhi, mid_y)))
            used_h.add(i)
            used_h.add(best_j)
            consumed.add(oi)
            consumed.add(oj)
        else:
            # No partner — keep the original.
            result.append(segments[oi])
            consumed.add(oi)

    # ── Collapse vertical pairs ──────────────────────────────────────
    v_segs.sort(key=lambda s: s[0])  # sort by x
    used_v: set[int] = set()
    for i in range(len(v_segs)):
        if i in used_v:
            continue
        x_i, ylo_i, yhi_i, oi = v_segs[i]
        best_j = None
        best_overlap = 0
        for j in range(i + 1, len(v_segs)):
            if j in used_v:
                continue
            x_j, ylo_j, yhi_j, oj = v_segs[j]
            if x_j - x_i > max_wall_thickness:
                break
            overlap = min(yhi_i, yhi_j) - max(ylo_i, ylo_j)
            if overlap > min(yhi_i - ylo_i, yhi_j - ylo_j) * 0.5:
                if overlap > best_overlap:
                    best_j = j
                    best_overlap = overlap
        if best_j is not None:
            x_j, ylo_j, yhi_j, oj = v_segs[best_j]
            mid_x = (x_i + x_j) / 2
            merged_ylo = min(ylo_i, ylo_j)
            merged_yhi = max(yhi_i, yhi_j)
            result.append(WallSegment(p1=(mid_x, merged_ylo), p2=(mid_x, merged_yhi)))
            used_v.add(i)
            used_v.add(best_j)
            consumed.add(oi)
            consumed.add(oj)
        else:
            result.append(segments[oi])
            consumed.add(oi)

    # Add back any non-orthogonal segments.
    result.extend(other)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1 — Endpoint Degree Calculation
# ═══════════════════════════════════════════════════════════════════════════


def _point_to_segment_dist(
    pt: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """Minimum distance from *pt* to the line segment *p1*–*p2*."""
    px, py = pt
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def find_exposed_endpoints(
    segments: list[WallSegment],
    epsilon: float = 3.0,
) -> list[ExposedEndpoint]:
    """Return degree-1 endpoints (doorjambs) using KDTree spatial analysis.

    An endpoint is *exposed* if no other endpoint is within *epsilon*
    AND it does not lie on the interior of another segment (T-junction).
    """
    # Collect all endpoints: index i → segment i//2, endpoint i%2.
    pts: list[tuple[float, float]] = []
    for seg in segments:
        pts.append(seg.p1)
        pts.append(seg.p2)

    if not pts:
        return []

    arr = np.array(pts, dtype=np.float64)
    tree = KDTree(arr)

    exposed: list[ExposedEndpoint] = []
    for i, pt in enumerate(pts):
        neighbors = tree.query_ball_point(pt, r=epsilon)
        if len(neighbors) <= 1:
            # Degree-1 candidate.  Check it isn't a T-junction.
            seg_idx = i // 2
            is_t_junction = False
            for j, seg in enumerate(segments):
                if j == seg_idx:
                    continue
                if _point_to_segment_dist(pt, seg.p1, seg.p2) < epsilon:
                    is_t_junction = True
                    break
            if is_t_junction:
                continue

            # Compute outward direction.
            seg = segments[seg_idx]
            if i % 2 == 0:
                # pt is seg.p1 → outward is away from p2.
                direction = _cardinal_direction(seg.p1, seg.p2, outward=True)
            else:
                # pt is seg.p2 → outward is away from p1.
                direction = _cardinal_direction(seg.p2, seg.p1, outward=True)

            exposed.append(ExposedEndpoint(point=pt, direction=direction, seg_idx=seg_idx))

    print(f"  Exposed endpoints (doorjambs): {len(exposed)}")
    return exposed


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2 — Direction Vector & Ray Generation
# ═══════════════════════════════════════════════════════════════════════════


def _cardinal_direction(
    p_end: tuple[float, float],
    p_start: tuple[float, float],
    outward: bool = True,
) -> tuple[float, float]:
    """Return the cardinal unit direction from *p_start* toward *p_end*
    (or opposite if *outward* is ``True``)."""
    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    if outward:
        dx, dy = -dx, -dy
    if abs(dx) >= abs(dy):
        return (1.0, 0.0) if dx > 0 else (-1.0, 0.0)
    else:
        return (0.0, 1.0) if dy > 0 else (0.0, -1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  Step 3 — Bounded Ray Casting & Intersection
# ═══════════════════════════════════════════════════════════════════════════


def _ray_segment_intersection(
    ray_origin: tuple[float, float],
    ray_dir: tuple[float, float],
    seg_p1: tuple[float, float],
    seg_p2: tuple[float, float],
    max_t: float,
    tol: float = 0.5,
) -> Optional[float]:
    """Return the ray parameter *t* at the intersection point, or ``None``.

    Only considers intersections where 0 < t <= max_t and the hit point
    falls within the segment bounds (with tolerance).
    """
    ox, oy = ray_origin
    dx, dy = ray_dir
    x1, y1 = seg_p1
    x2, y2 = seg_p2

    # Horizontal ray (dy == 0) hitting a vertical segment (x1 == x2).
    if dy == 0 and x1 == x2:
        if dx == 0:
            return None
        t = (x1 - ox) / dx
        if t <= 0 or t > max_t:
            return None
        hit_y = oy  # ray is horizontal
        min_y, max_y = min(y1, y2), max(y1, y2)
        if min_y - tol <= hit_y <= max_y + tol:
            return t
        return None

    # Vertical ray (dx == 0) hitting a horizontal segment (y1 == y2).
    if dx == 0 and y1 == y2:
        if dy == 0:
            return None
        t = (y1 - oy) / dy
        if t <= 0 or t > max_t:
            return None
        hit_x = ox  # ray is vertical
        min_x, max_x = min(x1, x2), max(x1, x2)
        if min_x - tol <= hit_x <= max_x + tol:
            return t
        return None

    # Parallel cases (same orientation) → no intersection.
    return None


def cast_rays_and_find_doors(
    exposed: list[ExposedEndpoint],
    segments: list[WallSegment],
    max_door_width: float,
) -> list[WallSegment]:
    """Cast rays from exposed endpoints.  Return new segments that seal
    doorway openings.

    Implements the *handshake* rule: opposing rays that meet mid-gap
    connect directly.  Rays that hit an existing wall create a segment
    to the hit point.  Rays that reach *max_door_width* without a hit
    are discarded (exterior openings).
    """
    door_segs: list[WallSegment] = []
    used: set[int] = set()  # indices of exposed endpoints already paired

    # ── Sub-step A: Handshake detection ──────────────────────────────
    # Group exposed endpoints by their perpendicular coordinate and axis.
    # Horizontal rays (dir x≠0): group by y-coordinate (rounded).
    # Vertical rays   (dir y≠0): group by x-coordinate (rounded).
    from collections import defaultdict

    groups: dict[tuple[float, str], list[tuple[int, ExposedEndpoint]]] = defaultdict(list)
    for idx, ep in enumerate(exposed):
        if ep.direction[1] == 0:
            # Horizontal ray → perpendicular coord is y.
            key = (round(ep.point[1], 1), "h")
        else:
            # Vertical ray → perpendicular coord is x.
            key = (round(ep.point[0], 1), "v")
        groups[key].append((idx, ep))

    for key, members in groups.items():
        if len(members) < 2:
            continue
        axis = key[1]
        # Sort by the parallel coordinate.
        if axis == "h":
            members.sort(key=lambda m: m[1].point[0])
        else:
            members.sort(key=lambda m: m[1].point[1])

        # Try to pair adjacent opposing-direction rays.
        for a in range(len(members)):
            if members[a][0] in used:
                continue
            idx_a, ep_a = members[a]
            for b in range(a + 1, len(members)):
                if members[b][0] in used:
                    continue
                idx_b, ep_b = members[b]
                # Must have opposite directions.
                if (ep_a.direction[0] + ep_b.direction[0] != 0 or
                        ep_a.direction[1] + ep_b.direction[1] != 0):
                    continue
                # Distance must be within max_door_width.
                gap = math.hypot(
                    ep_a.point[0] - ep_b.point[0],
                    ep_a.point[1] - ep_b.point[1],
                )
                if gap > max_door_width:
                    break  # sorted, so no further pair will be closer
                # Handshake! Connect them.
                door_segs.append(WallSegment(
                    p1=ep_a.point, p2=ep_b.point, source="ray",
                ))
                used.add(idx_a)
                used.add(idx_b)
                break

    # ── Sub-step B: Remaining rays → cast against wall segments ──────
    for idx, ep in enumerate(exposed):
        if idx in used:
            continue
        best_t: Optional[float] = None
        for seg in segments:
            t = _ray_segment_intersection(
                ep.point, ep.direction, seg.p1, seg.p2, max_door_width,
            )
            if t is not None and (best_t is None or t < best_t):
                best_t = t
        if best_t is not None:
            hit = (
                ep.point[0] + best_t * ep.direction[0],
                ep.point[1] + best_t * ep.direction[1],
            )
            door_segs.append(WallSegment(p1=ep.point, p2=hit, source="ray"))
            used.add(idx)

    print(f"  Door-sealing segments created: {len(door_segs)} "
          f"({sum(1 for d in door_segs if d.source == 'ray')} rays, "
          f"{len(used)} endpoints used / {len(exposed)} total)")
    return door_segs


# ═══════════════════════════════════════════════════════════════════════════
#  Step 4 — Polygon Assembly
# ═══════════════════════════════════════════════════════════════════════════


def _snap(val: float, grid: float = 0.5) -> float:
    return round(val / grid) * grid


def _polygon_area(verts: list[tuple[float, float]]) -> float:
    """Signed area via the Shoelace formula."""
    n = len(verts)
    area = 0.0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _centroid(verts: list[tuple[float, float]]) -> tuple[float, float]:
    """Centroid of a polygon."""
    n = len(verts)
    cx = sum(v[0] for v in verts) / n
    cy = sum(v[1] for v in verts) / n
    return (cx, cy)


def _weld_endpoints(
    segments: list[WallSegment],
    weld_radius: float = 4.0,
) -> list[WallSegment]:
    """Cluster nearby endpoints using KDTree and snap each cluster to its
    centroid.  This bridges small gaps at wall junctions without changing
    wall_pipeline output.
    """
    # Collect all endpoints.
    pts: list[tuple[float, float]] = []
    for seg in segments:
        pts.append(seg.p1)
        pts.append(seg.p2)

    if not pts:
        return segments

    arr = np.array(pts, dtype=np.float64)
    tree = KDTree(arr)

    # Union-find to cluster nearby points.
    parent = list(range(len(pts)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for i, pt in enumerate(pts):
        neighbors = tree.query_ball_point(pt, r=weld_radius)
        for j in neighbors:
            if j != i:
                _union(i, j)

    # Compute centroid of each cluster.
    from collections import defaultdict
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(pts)):
        clusters[_find(i)].append(i)

    remap: dict[int, tuple[float, float]] = {}
    for root, members in clusters.items():
        cx = sum(pts[m][0] for m in members) / len(members)
        cy = sum(pts[m][1] for m in members) / len(members)
        centroid = (cx, cy)
        for m in members:
            remap[m] = centroid

    # Rebuild segments with welded endpoints.
    welded: list[WallSegment] = []
    for i, seg in enumerate(segments):
        new_p1 = remap[2 * i]
        new_p2 = remap[2 * i + 1]
        if new_p1 == new_p2:
            continue  # collapsed to a point
        welded.append(WallSegment(p1=new_p1, p2=new_p2, source=seg.source))

    return welded


def _stitch_t_junctions(
    segments: list[WallSegment],
    tolerance: float = 5.0,
) -> list[WallSegment]:
    """For each segment endpoint, check if it is close to the body of a
    perpendicular segment.  If so, split the target segment at the
    projection point and extend the source endpoint to it, creating a
    proper T-junction node.
    """
    # Collect all endpoints with their segment index.
    endpoints: list[tuple[int, int, tuple[float, float]]] = []  # (seg_idx, 0|1, point)
    for i, seg in enumerate(segments):
        endpoints.append((i, 0, seg.p1))
        endpoints.append((i, 1, seg.p2))

    # Build KDTree of all endpoints to find isolated ones (degree-1 candidates).
    all_pts = np.array([ep[2] for ep in endpoints], dtype=np.float64)
    if len(all_pts) == 0:
        return segments
    ep_tree = KDTree(all_pts)

    splits: dict[int, list[tuple[float, float]]] = {}  # seg_idx → split points
    moves: dict[tuple[int, int], tuple[float, float]] = {}  # (seg_idx, end) → new point

    for i, seg in enumerate(segments):
        for end_idx, pt in [(0, seg.p1), (1, seg.p2)]:
            # Check if this endpoint is near the body of another segment.
            for j, other in enumerate(segments):
                if j == i:
                    continue
                # Are the segments perpendicular?
                dx_i = seg.p2[0] - seg.p1[0]
                dy_i = seg.p2[1] - seg.p1[1]
                dx_j = other.p2[0] - other.p1[0]
                dy_j = other.p2[1] - other.p1[1]
                # One must be ~horizontal, the other ~vertical.
                i_horiz = abs(dy_i) < abs(dx_i) * 0.1 if abs(dx_i) > 0.5 else False
                i_vert = abs(dx_i) < abs(dy_i) * 0.1 if abs(dy_i) > 0.5 else False
                j_horiz = abs(dy_j) < abs(dx_j) * 0.1 if abs(dx_j) > 0.5 else False
                j_vert = abs(dx_j) < abs(dy_j) * 0.1 if abs(dy_j) > 0.5 else False
                if not ((i_horiz and j_vert) or (i_vert and j_horiz)):
                    continue

                dist = _point_to_segment_dist(pt, other.p1, other.p2)
                if dist < tolerance and dist > 0.5:
                    # Project pt onto the other segment.
                    ox, oy = other.p1
                    odx = other.p2[0] - other.p1[0]
                    ody = other.p2[1] - other.p1[1]
                    seg_len_sq = odx * odx + ody * ody
                    if seg_len_sq < 0.01:
                        continue
                    t = ((pt[0] - ox) * odx + (pt[1] - oy) * ody) / seg_len_sq
                    t = max(0.0, min(1.0, t))
                    proj = (ox + t * odx, oy + t * ody)

                    # Move the endpoint to the projection.
                    moves[(i, end_idx)] = proj
                    # Mark the target segment for splitting.
                    if j not in splits:
                        splits[j] = []
                    splits[j].append(proj)
                    break  # only stitch to nearest perpendicular

    # Apply moves.
    new_segments: list[WallSegment] = []
    for i, seg in enumerate(segments):
        p1 = moves.get((i, 0), seg.p1)
        p2 = moves.get((i, 1), seg.p2)
        if i in splits:
            # Split this segment at all projection points.
            # Order split points along the segment.
            dx = seg.p2[0] - seg.p1[0]
            dy = seg.p2[1] - seg.p1[1]
            length_sq = dx * dx + dy * dy
            all_split_pts = [p1] + splits[i] + [p2]
            if length_sq > 0:
                all_split_pts.sort(key=lambda p: (p[0] - seg.p1[0]) * dx + (p[1] - seg.p1[1]) * dy)
            for k in range(len(all_split_pts) - 1):
                sp1 = all_split_pts[k]
                sp2 = all_split_pts[k + 1]
                if math.hypot(sp2[0] - sp1[0], sp2[1] - sp1[1]) > 0.5:
                    new_segments.append(WallSegment(p1=sp1, p2=sp2, source=seg.source))
        else:
            new_segments.append(WallSegment(p1=p1, p2=p2, source=seg.source))

    return new_segments


def _build_footprint(
    segments: list[WallSegment],
    scale_ratio: float = 96.0,
) -> Optional[ShapelyPolygon]:
    """Build the building footprint by buffering all wall segments and
    taking their union.  The result is a polygon representing the building
    perimeter.

    The buffer distance is derived from the scale so it bridges typical
    door openings (~3 ft real) while keeping the footprint tight.
    """
    # 3 real feet in PDF points at this scale.
    buffer_dist = 3.0 * 12.0 / scale_ratio * 72.0
    # Clamp to a reasonable range.
    buffer_dist = max(10.0, min(buffer_dist, 40.0))

    lines = []
    for seg in segments:
        if seg.source == "ray":
            continue  # skip door-sealing rays
        try:
            ls = ShapelyLineString([seg.p1, seg.p2])
            if ls.length > 0.5:
                lines.append(ls.buffer(buffer_dist))
        except Exception:
            continue

    if not lines:
        return None

    union = unary_union(lines)

    # Take the largest polygon from the union.
    if union.geom_type == "MultiPolygon":
        largest = max(union.geoms, key=lambda g: g.area)
    elif union.geom_type == "Polygon":
        largest = union
    else:
        return None

    # Use the exterior ring of the largest connected component.
    # This naturally follows the building outline including concavities.
    return ShapelyPolygon(largest.exterior)


def assemble_room_polygons(
    wall_segments: list[WallSegment],
    door_segments: list[WallSegment],
    scale_ratio: float,
    min_area_sqft: float = 25.0,
    footprint: Optional[ShapelyPolygon] = None,
    raw_wall_segments: Optional[list[WallSegment]] = None,
    raw_door_segments: Optional[list[WallSegment]] = None,
) -> list[list[tuple[float, float]]]:
    """Build a planar graph from all segments and extract closed room
    polygons via the minimum cycle basis.

    Polygons smaller than *min_area_sqft* are discarded (wall-thickness
    artifacts).  The single largest polygon (exterior) is also discarded.

    *raw_wall_segments*, if provided, are the original dual-border
    segments (before centerline collapse) used by the raster fallback
    for better wall coverage.
    """
    # ── Phase 1: Weld endpoints and stitch T-junctions ───────────────
    all_segs = wall_segments + door_segments
    print(f"  Pre-weld segments: {len(all_segs)}")
    all_segs = _stitch_t_junctions(all_segs, tolerance=5.0)
    print(f"  After T-junction stitching: {len(all_segs)}")
    all_segs = _weld_endpoints(all_segs, weld_radius=3.5)
    print(f"  After endpoint welding: {len(all_segs)}")

    grid = 0.5  # reduced from 2.0 — welding handles the coarse alignment

    # Build node mapping: snapped coordinates → integer ID.
    coord_to_id: dict[tuple[float, float], int] = {}
    id_to_coord: dict[int, tuple[float, float]] = {}

    def _node(x: float, y: float) -> int:
        key = (_snap(x, grid), _snap(y, grid))
        if key not in coord_to_id:
            nid = len(coord_to_id)
            coord_to_id[key] = nid
            id_to_coord[nid] = key
        return coord_to_id[key]

    G = nx.Graph()
    for seg in all_segs:
        n1 = _node(seg.p1[0], seg.p1[1])
        n2 = _node(seg.p2[0], seg.p2[1])
        if n1 == n2:
            continue
        length = math.hypot(
            id_to_coord[n1][0] - id_to_coord[n2][0],
            id_to_coord[n1][1] - id_to_coord[n2][1],
        )
        G.add_edge(n1, n2, weight=length)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Area threshold in square PDF points.
    sqpt_to_sqft = (scale_ratio / 72.0) ** 2 / 144.0
    min_area_sqpt = min_area_sqft / sqpt_to_sqft

    # Minimum cycle basis.
    try:
        cycles = nx.minimum_cycle_basis(G, weight="weight")
    except Exception:
        cycles = []

    print(f"  Raw cycles found: {len(cycles)}")

    # Convert cycles to ordered vertex lists.
    polygons: list[list[tuple[float, float]]] = []
    for cycle_nodes in cycles:
        if len(cycle_nodes) < 3:
            continue
        ordered = _order_cycle(cycle_nodes, G)
        verts = [id_to_coord[n] for n in ordered]
        area = abs(_polygon_area(verts))
        if area < min_area_sqpt:
            continue
        polygons.append(verts)

    # Discard the largest polygon (exterior boundary).
    if polygons:
        areas = [abs(_polygon_area(p)) for p in polygons]
        max_idx = areas.index(max(areas))
        exterior = polygons.pop(max_idx)
        print(f"  Discarded exterior polygon ({abs(_polygon_area(exterior)) * sqpt_to_sqft:.0f} ft²)")

    # ── Filter by building footprint ─────────────────────────────────
    if footprint is not None and polygons:
        before = len(polygons)
        polygons = [
            p for p in polygons
            if footprint.contains(ShapelyPoint(_centroid(p)))
        ]
        print(f"  Footprint filter: {before} → {len(polygons)}")

    print(f"  Room polygons after filtering: {len(polygons)}")

    # Fall back to raster if graph results are insufficient.
    # Criterion: fewer than 4 room-sized polygons (> 20 sqft each).
    big_rooms = [p for p in polygons if abs(_polygon_area(p)) * sqpt_to_sqft > 20]
    if len(big_rooms) < 4:
        print(f"  Graph found only {len(big_rooms)} room-sized polygons — using raster …")
        raster_door = raw_door_segments if raw_door_segments is not None else door_segments
        raster_segs = (raw_wall_segments or wall_segments) + raster_door
        raster_polys = _fallback_raster_polygons(raster_segs, scale_ratio, footprint=footprint)
        if len(raster_polys) > len(big_rooms):
            polygons = raster_polys

    return polygons


def _order_cycle(
    cycle_nodes: list[int],
    G: nx.Graph,
) -> list[int]:
    """Order the nodes of a cycle by walking the subgraph edges."""
    node_set = set(cycle_nodes)
    sub = G.subgraph(node_set)
    start = cycle_nodes[0]
    ordered = [start]
    visited = {start}
    current = start
    for _ in range(len(cycle_nodes) - 1):
        for nbr in sub.neighbors(current):
            if nbr not in visited:
                ordered.append(nbr)
                visited.add(nbr)
                current = nbr
                break
    return ordered


def _fallback_raster_polygons(
    all_segs: list[WallSegment],
    scale_ratio: float,
    footprint: Optional[ShapelyPolygon] = None,
    render_dpi: int = 150,
    line_thickness: int = 6,
    close_kernel: int = 50,
) -> list[list[tuple[float, float]]]:
    """Raster room detection using only extracted wall+door segments.
    Draws segments on a clean canvas, applies morphological close to
    bridge corner gaps, seals with footprint boundary, then uses
    exterior flood-fill to identify rooms.
    """
    if not all_segs:
        return []
    xs = [s.p1[0] for s in all_segs] + [s.p2[0] for s in all_segs]
    ys = [s.p1[1] for s in all_segs] + [s.p2[1] for s in all_segs]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    scale = render_dpi / 72.0
    pad = int(40 * scale) + 10
    w = int((x_max - x_min) * scale) + 2 * pad
    h = int((y_max - y_min) * scale) + 2 * pad

    # ── Build footprint mask ─────────────────────────────────────────
    if footprint is not None:
        fp_coords = list(footprint.exterior.coords)
        fp_pixels = np.array([
            (int((x - x_min) * scale) + pad, int((y - y_min) * scale) + pad)
            for x, y in fp_coords
        ], dtype=np.int32)
        footprint_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(footprint_mask, [fp_pixels], 255)
    else:
        all_pts = []
        for seg in all_segs:
            all_pts.append((int((seg.p1[0] - x_min) * scale) + pad,
                            int((seg.p1[1] - y_min) * scale) + pad))
            all_pts.append((int((seg.p2[0] - x_min) * scale) + pad,
                            int((seg.p2[1] - y_min) * scale) + pad))
        hull_pts = cv2.convexHull(np.array(all_pts, dtype=np.int32))
        footprint_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(footprint_mask, [hull_pts], 255)

    # ── Draw only extracted wall + door-ray segments ─────────────────
    canvas = np.zeros((h, w), dtype=np.uint8)
    for seg in all_segs:
        pt1 = (int((seg.p1[0] - x_min) * scale) + pad,
               int((seg.p1[1] - y_min) * scale) + pad)
        pt2 = (int((seg.p2[0] - x_min) * scale) + pad,
               int((seg.p2[1] - y_min) * scale) + pad)
        cv2.line(canvas, pt1, pt2, 255, line_thickness)

    # ── Morphological close to bridge small corner gaps ──────────────
    if close_kernel > 1:
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, 1))
        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_kernel))
        closed_h = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k_h)
        closed_v = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k_v)
        canvas = cv2.bitwise_or(closed_h, closed_v)
        k_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k_sm)

    # ── Draw footprint boundary + canvas border to seal exterior ─────
    if footprint is not None:
        cv2.polylines(canvas, [fp_pixels], isClosed=True, color=255, thickness=line_thickness)
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), 255, 2)

    cv2.imwrite("raster_debug.png", canvas)

    # ── Seal exterior with thick footprint boundary + flood-fill ────
    # Draw the footprint boundary as a very thick wall to guarantee
    # no gaps for the flood-fill to leak through.
    if footprint is not None:
        cv2.polylines(canvas, [fp_pixels], isClosed=True, color=255,
                      thickness=line_thickness * 3)
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), 255, 3)

    cv2.imwrite("raster_debug.png", canvas)

    # Flood-fill from corner — marks all exterior as 128.
    flood = canvas.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (1, 1), 128)

    # Rooms = pixels still 0 (not wall=255, not exterior=128).
    rooms_mask = np.zeros_like(canvas)
    rooms_mask[flood == 0] = 255

    # Clip to tightly eroded footprint — the footprint buffer extends
    # ~27 pt beyond walls, creating a perimeter band that connects all
    # rooms.  Erode by the buffer distance to bring it inside the outer
    # wall faces.
    if footprint is not None:
        buffer_px = int(3.0 * 12.0 / scale_ratio * 72.0 * scale)  # match footprint buffer
        k_fp_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (buffer_px, buffer_px))
        fp_tight = cv2.erode(footprint_mask, k_fp_erode)
        rooms_mask = cv2.bitwise_and(rooms_mask, fp_tight)

    # Break doorway connections: morphological opening erodes thin
    # passages (doorways ~3ft) while preserving rooms (8+ ft).
    open_radius = int(1.5 * 12.0 / scale_ratio * 72.0 * scale)
    open_radius = max(5, min(open_radius, 40))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (open_radius, open_radius))
    rooms_mask = cv2.morphologyEx(rooms_mask, cv2.MORPH_OPEN, k_open)

    cv2.imwrite("raster_flood_debug.png", rooms_mask)

    contours, _ = cv2.findContours(rooms_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    inv_scale = 72.0 / render_dpi
    sqpt_to_sqft = (scale_ratio / 72.0) ** 2 / 144.0
    min_area_sqpt = 25.0 / sqpt_to_sqft

    polygons: list[list[tuple[float, float]]] = []
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        area_sqpt = area_px * (inv_scale ** 2)
        if area_sqpt < min_area_sqpt:
            continue
        pts = cnt.reshape(-1, 2)
        verts = [
            (float(p[0] - pad) * inv_scale + x_min,
             float(p[1] - pad) * inv_scale + y_min)
            for p in pts
        ]
        polygons.append(verts)

    # Filter by footprint.
    if footprint is not None and polygons:
        polygons = [
            p for p in polygons
            if footprint.contains(ShapelyPoint(_centroid(p)))
        ]

    print(f"  Raster fallback found {len(polygons)} polygons")
    return polygons


# ═══════════════════════════════════════════════════════════════════════════
#  Step 5 — VLM Semantic Labeling
# ═══════════════════════════════════════════════════════════════════════════


def _render_page_png(pdf_path: str, dpi: int, page_index: int = 0) -> tuple[bytes, int, int]:
    """Render a PDF page to PNG bytes.  Returns (png_bytes, w, h)."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    png = pix.tobytes("png")
    w, h = pix.width, pix.height
    doc.close()
    return png, w, h


def label_rooms_with_vlm(
    pdf_path: str,
    polygons: list[list[tuple[float, float]]],
    scale_ratio: float,
    dpi: int = 150,
    page_index: int = 0,
) -> list[RoomPolygon]:
    """Use Claude Vision to identify room labels, then assign each label
    to the polygon that contains it via point-in-polygon testing."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set.")

    png_bytes, full_w, full_h = _render_page_png(pdf_path, dpi, page_index)
    img_b64 = base64.b64encode(png_bytes).decode()

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = (
        "You are a construction estimator reading an architectural floor plan. "
        "Identify every room label in the MAIN (largest) floor plan drawing "
        "and return the approximate pixel centre (x, y) of each label.\n\n"
        "CRITICAL: Ignore labels inside smaller option/alternate/detail insets.\n\n"
        "Room labels include: BEDROOM, KITCHEN, BATH, LIVING, DINING, "
        "GARAGE, LAUNDRY, OFFICE, FOYER, PANTRY, CLOSET, HALL, PORCH, "
        "DECK, ENTRY, UTILITY, DEN, STUDY, MUDROOM, W.I.C., WIC, MECH, "
        "STORAGE, NOOK, LOFT, BONUS, MASTER, POWDER, FAMILY, GREAT, "
        "FLEX, OWNER'S, 2-CAR GARAGE, COVERED PATIO, and similar.\n\n"
        "Return ONLY a JSON array: "
        '[{"name": "...", "cx": <pixel_x>, "cy": <pixel_y>}, ...]\n'
        "No markdown fences, no commentary."
    )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": (
                    f"Image is {full_w}x{full_h} px at {dpi} DPI. "
                    "Identify room labels from the MAIN floor plan only.")},
            ],
        }],
        temperature=0.0,
    )

    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()
    labels: list[dict] = json.loads(raw)

    # Scale Claude coordinates (max 1568 px) → original pixel → PDF points.
    max_claude = 1568
    longest = max(full_w, full_h)
    px_scale = longest / max_claude if longest > max_claude else 1.0
    px_to_pt = 72.0 / dpi

    sqpt_to_sqft = (scale_ratio / 72.0) ** 2 / 144.0

    # Build Shapely polygons for containment testing.
    shapely_polys = []
    for verts in polygons:
        try:
            shapely_polys.append(ShapelyPolygon(verts))
        except Exception:
            shapely_polys.append(None)

    results: list[RoomPolygon] = []
    assigned_names: dict[int, str] = {}

    for lbl in labels:
        cx_px = float(lbl["cx"]) * px_scale
        cy_px = float(lbl["cy"]) * px_scale
        cx_pt = cx_px * px_to_pt
        cy_pt = cy_px * px_to_pt
        name = str(lbl.get("name", "UNKNOWN")).strip()

        for pi, sp in enumerate(shapely_polys):
            if sp is not None and sp.contains(ShapelyPoint(cx_pt, cy_pt)):
                if pi not in assigned_names:
                    assigned_names[pi] = name
                break

    for pi, verts in enumerate(polygons):
        area_sqpt = abs(_polygon_area(verts))
        area_sqft = area_sqpt * sqpt_to_sqft
        name = assigned_names.get(pi, f"ROOM-{pi + 1}")
        cent = _centroid(verts)
        results.append(RoomPolygon(
            room_id=pi + 1,
            name=name,
            vertices=verts,
            area_sqpt=area_sqpt,
            area_sqft=area_sqft,
            color=_PALETTE[pi % len(_PALETTE)],
            centroid=cent,
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Output
# ═══════════════════════════════════════════════════════════════════════════


def overlay_rooms_on_pdf(
    pdf_path: str,
    rooms: list[RoomPolygon],
    door_segs: list[WallSegment],
    output_path: str = "room_overlay.pdf",
    page_index: int = 0,
    fill_opacity: float = 0.30,
) -> str:
    """Draw room polygons, labels, and door-sealing segments on the PDF."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    for room in rooms:
        if len(room.vertices) < 3:
            continue
        pts = [fitz.Point(v[0], v[1]) for v in room.vertices]
        shape = page.new_shape()
        shape.draw_polyline(pts + [pts[0]])
        r, g, b = room.color
        shape.finish(
            color=(r * 0.6, g * 0.6, b * 0.6),
            fill=(r, g, b),
            fill_opacity=fill_opacity,
            width=0.5,
        )
        shape.commit()

        # Room label.
        cx, cy = room.centroid
        half_w = max(40, len(room.name) * 3.5)
        rect = fitz.Rect(cx - half_w, cy - 6, cx + half_w, cy + 6)
        page.insert_textbox(rect, room.name, fontsize=7, fontname="helv",
                            color=(0, 0, 0), align=fitz.TEXT_ALIGN_CENTER)

    # Door-sealing rays in dashed blue.
    for ds in door_segs:
        shape = page.new_shape()
        shape.draw_line(fitz.Point(ds.p1[0], ds.p1[1]),
                        fitz.Point(ds.p2[0], ds.p2[1]))
        shape.finish(color=(0.1, 0.3, 0.9), width=0.8, dashes="[4 2]")
        shape.commit()

    doc.save(output_path)
    doc.close()
    return output_path


def render_debug_image(
    pdf_path: str,
    wall_segments: list[WallSegment],
    door_segments: list[WallSegment],
    exposed: list[ExposedEndpoint],
    rooms: list[RoomPolygon],
    output_path: str = "room_debug.png",
    page_index: int = 0,
    dpi: int = 150,
) -> str:
    """Debug visualisation: walls white, exposed endpoints red, door rays
    green, room fills coloured.  Handles page rotation correctly."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    rot_mat = page.rotation_matrix
    # Canvas matches the rendered (rotated) page dimensions.
    scale = dpi / 72.0
    w = int(page.rect.width * scale) + 1
    h = int(page.rect.height * scale) + 1
    doc.close()
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)

    def _pt(x: float, y: float) -> tuple[int, int]:
        rp = fitz.Point(x, y) * rot_mat
        return (int(rp.x * scale), int(rp.y * scale))

    # Room fills.
    for room in rooms:
        pts = np.array([_pt(v[0], v[1]) for v in room.vertices], dtype=np.int32)
        col = tuple(int(c * 180) for c in reversed(room.color))
        cv2.fillPoly(canvas, [pts], col)

    # Wall segments (white).
    for seg in wall_segments:
        cv2.line(canvas, _pt(*seg.p1), _pt(*seg.p2), (255, 255, 255), 1)

    # Door segments (green).
    for seg in door_segments:
        cv2.line(canvas, _pt(*seg.p1), _pt(*seg.p2), (0, 220, 0), 2)

    # Exposed endpoints (red dots).
    for ep in exposed:
        cv2.circle(canvas, _pt(*ep.point), 4, (0, 0, 255), -1)

    cv2.imwrite(output_path, canvas)
    return output_path


def render_ray_debug(
    pdf_path: str,
    wall_segments: list[WallSegment],
    door_segments: list[WallSegment],
    exposed: list[ExposedEndpoint],
    footprint: Optional[ShapelyPolygon],
    page_index: int = 0,
    output_path: str = "ray_debug.png",
    dpi: int = 150,
) -> str:
    """Render the PDF page with wall segments, exposed endpoints, ray
    cast door-sealing segments, and building footprint overlaid.

    - PDF page: faded background
    - Wall segments: white lines
    - Exposed endpoints: red circles with direction arrows
    - Door-sealing rays: bright green lines with endpoint dots
    - Building footprint: yellow dashed outline
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    rot_mat = page.rotation_matrix
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    # Convert to BGR for OpenCV; fade the background.
    if pix.n >= 3:
        canvas = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    else:
        canvas = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    canvas = (canvas.astype(np.float32) * 0.35).astype(np.uint8)

    scale = dpi / 72.0
    doc.close()

    def _pt(x: float, y: float) -> tuple[int, int]:
        rp = fitz.Point(x, y) * rot_mat
        return (int(rp.x * scale), int(rp.y * scale))

    # Building footprint (yellow outline).
    if footprint is not None:
        fp_coords = list(footprint.exterior.coords)
        fp_pts = np.array([_pt(x, y) for x, y in fp_coords], dtype=np.int32)
        cv2.polylines(canvas, [fp_pts], True, (0, 220, 220), 2)

    # Wall segments (white, thin).
    for seg in wall_segments:
        cv2.line(canvas, _pt(*seg.p1), _pt(*seg.p2), (255, 255, 255), 1)

    # Exposed endpoints (red circles + direction arrows).
    for ep in exposed:
        pt = _pt(*ep.point)
        cv2.circle(canvas, pt, 5, (0, 0, 255), -1)
        # Arrow showing ray direction.
        arrow_len = 15
        tip = (pt[0] + int(ep.direction[0] * arrow_len),
               pt[1] + int(ep.direction[1] * arrow_len))
        cv2.arrowedLine(canvas, pt, tip, (0, 0, 255), 2, tipLength=0.4)

    # Door-sealing ray segments (bright green, thick).
    for seg in door_segments:
        p1 = _pt(*seg.p1)
        p2 = _pt(*seg.p2)
        cv2.line(canvas, p1, p2, (0, 255, 0), 2)
        cv2.circle(canvas, p1, 3, (0, 255, 0), -1)
        cv2.circle(canvas, p2, 3, (0, 255, 0), -1)

    cv2.imwrite(output_path, canvas)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════


def load_env() -> None:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as fh:
        for raw in fh:
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def compute_max_door_width(scale_ratio: float, max_real_feet: float = 20.0) -> float:
    """Convert a real-world maximum doorway width (in feet) to PDF points."""
    real_inches = max_real_feet * 12.0
    paper_inches = real_inches / scale_ratio
    return paper_inches * 72.0


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════════════════


def run_pipeline(
    pdf_path: str,
    *,
    scale_ratio: float = 96.0,
    epsilon: float = 3.0,
    max_door_ft: float = 20.0,
    output_pdf: str = "room_overlay.pdf",
    output_debug: str = "room_debug.png",
    page_index: int = 0,
) -> list[RoomPolygon]:
    """Execute the full vector ray-casting room detection pipeline."""

    print("=" * 62)
    print("   VECTOR RAY-CASTING ROOM DETECTION")
    print("=" * 62)

    max_dw = compute_max_door_width(scale_ratio, max_door_ft)
    print(f"  Scale      : {scale_ratio}  (max door = {max_dw:.1f} pt ≈ {max_door_ft} ft)")

    # ── Step 0: Wall segments (via wall_pipeline fingerprinting) ────
    print("\n[0/5] Extracting wall border segments via wall_pipeline …")
    segments, raw_segments = extract_walls_via_wall_pipeline(pdf_path, page_index)

    # ── Step 1: Exposed endpoints ────────────────────────────────────
    print("\n[1/5] Finding exposed endpoints (doorjambs) …")
    exposed = find_exposed_endpoints(segments, epsilon)

    # ── Step 2 is implicit (direction already computed in Step 1) ────

    # ── Step 3: Ray casting ──────────────────────────────────────────
    print("\n[3/5] Casting rays to seal doorways …")
    door_segs = cast_rays_and_find_doors(exposed, segments, max_dw)

    # Also find exposed endpoints and cast rays on the full merged data
    # (borders + hatch edges) so the raster fallback has all doorways sealed.
    print("  Finding doorways on full merged data for raster …")
    exposed_full = find_exposed_endpoints(raw_segments, epsilon)
    door_segs_full = cast_rays_and_find_doors(exposed_full, raw_segments, max_dw)

    # ── Build footprint from wall segment geometry ─────────────────
    print("\n  Building footprint from wall vectors …")
    footprint = _build_footprint(raw_segments, scale_ratio=scale_ratio)
    if footprint is not None:
        sqpt_to_sqft = (scale_ratio / 72.0) ** 2 / 144.0
        print(f"  Footprint area: {footprint.area * sqpt_to_sqft:.0f} ft²")
    else:
        print("  WARNING: Could not build footprint")

    # ── Step 4: Polygon assembly ─────────────────────────────────────
    print("\n[4/5] Assembling room polygons …")
    os.environ["_ROOM_PIPELINE_PDF"] = pdf_path
    raw_polygons = assemble_room_polygons(segments, door_segs, scale_ratio,
                                          footprint=footprint,
                                          raw_wall_segments=raw_segments,
                                          raw_door_segments=door_segs_full)

    if not raw_polygons:
        print("\n  No room polygons found.")
        return []

    # ── Step 5: VLM labeling ─────────────────────────────────────────
    print("\n[5/5] Labeling rooms with Claude Vision …")
    rooms = label_rooms_with_vlm(pdf_path, raw_polygons, scale_ratio,
                                 page_index=page_index)

    # ── Outputs ──────────────────────────────────────────────────────
    print("\n  Generating outputs …")
    overlay_rooms_on_pdf(pdf_path, rooms, door_segs, output_pdf, page_index)
    print(f"  PDF   → {output_pdf}")
    render_debug_image(pdf_path, segments, door_segs, exposed, rooms,
                       output_debug, page_index)
    print(f"  Debug → {output_debug}")
    render_ray_debug(pdf_path, raw_segments, door_segs, exposed, footprint,
                     page_index, "ray_debug.png")
    print(f"  Rays  → ray_debug.png")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("-" * 62)
    print(f"  {'ROOM':<22s}  {'AREA':>10s}  VERTICES")
    print("-" * 62)
    for rm in rooms:
        print(f"  {rm.name:<22s}  {rm.area_sqft:>8.1f} ft²  {len(rm.vertices):>5d}")
    total = sum(r.area_sqft for r in rooms)
    print("-" * 62)
    print(f"  {'TOTAL':<22s}  {total:>8.1f} ft²")
    print("-" * 62)
    print("  Done.\n")

    return rooms


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Vector ray-casting room detection for floor plans.",
    )
    ap.add_argument("--pdf", default="./352 AA copy 2.pdf")
    ap.add_argument("--scale", type=float, default=96.0,
                    help='Scale ratio (96 = 1/8"=1\'-0").')
    ap.add_argument("--epsilon", type=float, default=3.0,
                    help="Endpoint clustering tolerance in PDF pts.")
    ap.add_argument("--max-door-ft", type=float, default=20.0,
                    help="Max doorway width in real feet.")
    ap.add_argument("--output-pdf", default="room_overlay.pdf")
    ap.add_argument("--output-debug", default="room_debug.png")

    args = ap.parse_args()
    load_env()

    rooms = run_pipeline(
        pdf_path=args.pdf,
        scale_ratio=args.scale,
        epsilon=args.epsilon,
        max_door_ft=args.max_door_ft,
        output_pdf=args.output_pdf,
        output_debug=args.output_debug,
    )
    return 0 if rooms else 1


if __name__ == "__main__":
    sys.exit(main())
