#!/usr/bin/env python3
"""
wall_extractor.py — Wall Geometry Extraction Pipeline
=====================================================

Extracts wall footprint polygons from 2D architectural PDF vector data
using a three-phase deterministic geometric pipeline (no ML/VLM):

    Phase 1: Line Consolidation (de-fragmentation)
    Phase 2: Path Fill Analysis (explicit wall polygons from fills/hatching)
    Phase 3: Ray-Casting & Spatial Graph (parallel-line pairing + T-junction solving)

Usage:
    python wall_extractor.py <pdf_path> [page_num]

Dependencies:
    pip install PyMuPDF numpy shapely networkx
"""

import math
import sys
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Set

import fitz
import numpy as np
from shapely.geometry import (
    LineString, Polygon, MultiPolygon, box,
    GeometryCollection,
)
from shapely.ops import unary_union
from shapely.strtree import STRtree
import networkx as nx


# ================================================================
#  CONFIGURATION
# ================================================================
# All real-world dimensions are in inches.  SCALE_FACTOR converts
# between PDF points and real-world inches.
#
# Common architectural scales → SCALE_FACTOR (pts / real-inch):
#   1/8"  = 1'-0"  →  0.75
#   3/16" = 1'-0"  →  1.125
#   1/4"  = 1'-0"  →  1.5     ← most common for floor plans
#   3/8"  = 1'-0"  →  2.25
#   1/2"  = 1'-0"  →  3.0
# ----------------------------------------------------------------

SCALE_FACTOR: float = 1.5          # PDF pts per real inch  (1/4" = 1'-0")

MERGE_TOLERANCE: float = 1.0       # (in) max gap bridged when merging
                                    #      collinear line fragments

MIN_WALL_THICKNESS: float = 3.25    # (in) walls thinner than this are ignored
                                    #      (std 2×4 stud = 3.5")
MAX_WALL_THICKNESS: float = 14.0   # (in) walls thicker than this are ignored
                                    #      (captures CMU / double stud)

MIN_WALL_LENGTH: float = 18.0      # (in) walls shorter than 1.5 feet are
                                    #      almost certainly symbols / noise

# -- Phase 1 tolerances --
ANGLE_TOL: float    = 0.02         # (rad, ~1.15°)  collinearity angle window
PERP_DIST_TOL: float = 0.5        # (pts)  perpendicular offset window
MIN_LINE_LENGTH: float = 3.0      # (pts)  discard consolidated lines shorter
                                    #        than this (noise)

# -- Phase 2 tolerances --
HATCH_BUFFER: float  = 1.5        # (pts)  buffer for clustering hatch fills
MAX_FILL_AREA: float = 50.0       # (pts²) fills > this are checked as direct
                                    #        wall rectangles; smaller → hatch

# -- Phase 3 tolerances --
RAY_INTERVAL: float       = 2.0   # (pts)  spacing of perpendicular sample rays
MIN_CONSECUTIVE_HITS: int = 3     # min consecutive rays confirming a pair
HIT_DIST_VARIANCE: float  = 1.0   # (pts)  max spread in hit distances


# ================================================================
#  UTILITIES
# ================================================================

def _merge_intervals(
    intervals: List[Tuple[float, float]],
    tolerance: float,
) -> List[Tuple[float, float]]:
    """Merge sorted 1-D intervals whose gap ≤ *tolerance*."""
    if not intervals:
        return []
    merged = []
    lo, hi = intervals[0]
    for s, e in intervals[1:]:
        if s <= hi + tolerance:
            hi = max(hi, e)
        else:
            merged.append((lo, hi))
            lo, hi = s, e
    merged.append((lo, hi))
    return merged


def _mrr_edges(poly: Polygon) -> Tuple[float, float]:
    """Return (shorter_edge, longer_edge) of the min-area rotated bounding rect.

    Falls back to the axis-aligned bounding box when the rotated
    rectangle computation produces NaN (e.g. very thin polygons).
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrr = poly.minimum_rotated_rectangle
    c = list(mrr.exterior.coords)
    e1 = math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1])
    e2 = math.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1])
    if math.isnan(e1) or math.isnan(e2) or math.isinf(e1) or math.isinf(e2):
        # Fallback to axis-aligned bounding box
        b = poly.bounds
        w, h = b[2] - b[0], b[3] - b[1]
        return (min(w, h), max(w, h))
    return (min(e1, e2), max(e1, e2))


def min_bounding_thickness(poly: Polygon) -> float:
    """Shorter side of the minimum-area rotated bounding rectangle."""
    return _mrr_edges(poly)[0]


def max_bounding_length(poly: Polygon) -> float:
    """Longer side of the minimum-area rotated bounding rectangle."""
    return _mrr_edges(poly)[1]


# ================================================================
#  PHASE 1 — LINE CONSOLIDATION  (De-fragmentation)
# ================================================================
# Architectural PDFs often render a single visual line as dozens of
# tiny disconnected vector segments.  Phase 1:
#   1. Extracts every straight line segment from get_drawings().
#   2. Classifies each as Horizontal / Vertical / Diagonal.
#   3. Groups collinear segments (same infinite line within tolerance).
#   4. Merges overlapping / close segments into unified LineStrings.
# ================================================================

def phase1_consolidate_lines(drawings: List[dict]) -> List[LineString]:
    """Return a de-fragmented list of :class:`LineString` objects."""

    merge_pts = MERGE_TOLERANCE * SCALE_FACTOR

    # ---- 1a. Extract & classify every segment ----
    #  H → (y,  x_lo, x_hi)
    #  V → (x,  y_lo, y_hi)
    #  D → (θ,  ρ,    p1,   p2)           θ ∈ [0,π), ρ ≥ 0
    h_segs: List[Tuple[float, float, float]] = []
    v_segs: List[Tuple[float, float, float]] = []
    d_segs: List[Tuple[float, float, np.ndarray, np.ndarray]] = []

    for d in drawings:
        color = d.get("color")
        if color is None:                       # fill-only path
            continue
        if max(color) > 0.5:                    # skip light/colored strokes
            continue

        for item in d["items"]:
            # Collect (p1, p2) pairs from line items and rect edges
            point_pairs: List[Tuple[np.ndarray, np.ndarray]] = []

            if item[0] == "l":
                point_pairs.append((
                    np.array([item[1].x, item[1].y]),
                    np.array([item[2].x, item[2].y]),
                ))
            elif item[0] == "re":
                r = item[1]
                corners = [
                    (r.x0, r.y0), (r.x1, r.y0),
                    (r.x1, r.y1), (r.x0, r.y1),
                ]
                for k in range(4):
                    point_pairs.append((
                        np.array(corners[k]),
                        np.array(corners[(k + 1) % 4]),
                    ))

            for p1, p2 in point_pairs:
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = math.hypot(dx, dy)
                if length < 0.05:
                    continue

                slope_angle = math.atan2(abs(dy), abs(dx))

                if slope_angle < ANGLE_TOL:
                    # ---- Horizontal ----
                    y = (p1[1] + p2[1]) * 0.5
                    h_segs.append((y, min(p1[0], p2[0]), max(p1[0], p2[0])))

                elif slope_angle > math.pi / 2 - ANGLE_TOL:
                    # ---- Vertical ----
                    x = (p1[0] + p2[0]) * 0.5
                    v_segs.append((x, min(p1[1], p2[1]), max(p1[1], p2[1])))

                else:
                    # ---- Diagonal — Hough normal form (θ, ρ) ----
                    direction = (p2 - p1) / length
                    if direction[0] < 0 or (
                        abs(direction[0]) < 1e-10 and direction[1] < 0
                    ):
                        direction = -direction
                    normal = np.array([-direction[1], direction[0]])
                    rho = float(np.dot(normal, p1))
                    if rho < 0:
                        rho = -rho
                        normal = -normal
                    theta = float(math.atan2(direction[1], direction[0]) % math.pi)
                    d_segs.append((theta, rho, p1.copy(), p2.copy()))

    lines: List[LineString] = []

    # ---- 1b. Group & merge horizontal segments ----
    h_segs.sort()
    i = 0
    while i < len(h_segs):
        j = i + 1
        while j < len(h_segs) and h_segs[j][0] - h_segs[i][0] < PERP_DIST_TOL:
            j += 1
        y_avg = float(np.mean([h_segs[k][0] for k in range(i, j)]))
        intervals = sorted((h_segs[k][1], h_segs[k][2]) for k in range(i, j))
        for lo, hi in _merge_intervals(intervals, merge_pts):
            if hi - lo >= MIN_LINE_LENGTH:
                lines.append(LineString([(lo, y_avg), (hi, y_avg)]))
        i = j

    # ---- 1c. Group & merge vertical segments ----
    v_segs.sort()
    i = 0
    while i < len(v_segs):
        j = i + 1
        while j < len(v_segs) and v_segs[j][0] - v_segs[i][0] < PERP_DIST_TOL:
            j += 1
        x_avg = float(np.mean([v_segs[k][0] for k in range(i, j)]))
        intervals = sorted((v_segs[k][1], v_segs[k][2]) for k in range(i, j))
        for lo, hi in _merge_intervals(intervals, merge_pts):
            if hi - lo >= MIN_LINE_LENGTH:
                lines.append(LineString([(x_avg, lo), (x_avg, hi)]))
        i = j

    # ---- 1d. Group & merge diagonal segments ----
    d_segs.sort(key=lambda s: (s[0], s[1]))
    i = 0
    while i < len(d_segs):
        j = i + 1
        while j < len(d_segs):
            if (abs(d_segs[j][0] - d_segs[i][0]) < ANGLE_TOL
                    and abs(d_segs[j][1] - d_segs[i][1]) < PERP_DIST_TOL):
                j += 1
            else:
                break

        # Best-fit direction for the group
        directions = []
        for k in range(i, j):
            dv = d_segs[k][3] - d_segs[k][2]
            ln = np.linalg.norm(dv)
            if ln > 0:
                dv = dv / ln
                if dv[0] < 0 or (abs(dv[0]) < 1e-10 and dv[1] < 0):
                    dv = -dv
                directions.append(dv)
        if not directions:
            i = j
            continue

        avg_dir = np.mean(directions, axis=0)
        avg_dir = avg_dir / np.linalg.norm(avg_dir)
        avg_normal = np.array([-avg_dir[1], avg_dir[0]])
        avg_rho = float(np.mean([d_segs[k][1] for k in range(i, j)]))
        ref_pt = avg_normal * avg_rho

        # Re-project every endpoint onto the group direction
        intervals = []
        for k in range(i, j):
            t1 = float(np.dot(d_segs[k][2] - ref_pt, avg_dir))
            t2 = float(np.dot(d_segs[k][3] - ref_pt, avg_dir))
            intervals.append((min(t1, t2), max(t1, t2)))
        intervals.sort()

        for lo, hi in _merge_intervals(intervals, merge_pts):
            if hi - lo >= MIN_LINE_LENGTH:
                pA = ref_pt + lo * avg_dir
                pB = ref_pt + hi * avg_dir
                lines.append(LineString([pA.tolist(), pB.tolist()]))
        i = j

    return lines


# ================================================================
#  PHASE 2 — PATH FILL ANALYSIS  (Explicit wall polygons)
# ================================================================
# Some architects draw walls as solid fills or hatched polygons.
# Phase 2 captures these "easy wins":
#   1. Extracts filled closed paths (rects, polygons).
#   2. Large fills whose thickness is in [MIN, MAX] → direct walls.
#   3. Small fills (hatch triangles) → cluster via buffer-union,
#      then filter clustered regions by wall dimensions.
#   4. Removes Phase-1 lines that overlap captured wall borders.
# ================================================================

def phase2_fill_analysis(
    drawings: List[dict],
    phase1_lines: List[LineString],
) -> Tuple[List[Polygon], List[LineString]]:
    """Return ``(wall_polygons, remaining_lines)``."""

    min_wall_pts = MIN_WALL_THICKNESS * SCALE_FACTOR
    max_wall_pts = MAX_WALL_THICKNESS * SCALE_FACTOR

    # ---- 2a. Extract every filled shape ----
    fills: List[Polygon] = []

    for d in drawings:
        if d.get("fill") is None:
            continue

        # Rect items → box polygon
        for item in d["items"]:
            if item[0] == "re":
                r = item[1]
                poly = box(r.x0, r.y0, r.x1, r.y1)
                if poly.is_valid and poly.area > 0.01:
                    fills.append(poly)

        # Line-segment paths → polygon from vertices
        pts: List[Tuple[float, float]] = []
        for item in d["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if not pts or (
                    abs(pts[-1][0] - p1.x) > 0.01
                    or abs(pts[-1][1] - p1.y) > 0.01
                ):
                    pts.append((p1.x, p1.y))
                pts.append((p2.x, p2.y))

        if len(pts) >= 3:
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            try:
                poly = Polygon(pts)
                if poly.is_valid and poly.area > 0.01:
                    fills.append(poly)
            except Exception:
                pass

    # ---- 2b. Classify: direct wall fills vs. hatch elements ----
    direct_walls: List[Polygon] = []
    hatch_elements: List[Polygon] = []

    for poly in fills:
        area = poly.area
        if area > MAX_FILL_AREA:
            try:
                thickness = min_bounding_thickness(poly)
                length = max_bounding_length(poly)
                if (min_wall_pts <= thickness <= max_wall_pts
                        and length > thickness * 1.5):
                    direct_walls.append(poly)
            except Exception:
                pass
        elif area < MAX_FILL_AREA:
            # Small fills are hatch-pattern candidates — but reject any
            # whose bounding box is already wider than a plausible wall
            bx = poly.bounds  # (minx, miny, maxx, maxy)
            bbox_w = bx[2] - bx[0]
            bbox_h = bx[3] - bx[1]
            if min(bbox_w, bbox_h) < max_wall_pts * 2:
                hatch_elements.append(poly)

    # ---- 2c. Cluster hatch elements into wall regions ----
    hatch_walls: List[Polygon] = []

    if hatch_elements:
        buffered = [h.buffer(HATCH_BUFFER) for h in hatch_elements]
        merged = unary_union(buffered)

        candidates: List[Polygon] = []
        if isinstance(merged, Polygon):
            candidates = [merged]
        elif isinstance(merged, (MultiPolygon, GeometryCollection)):
            candidates = [g for g in merged.geoms if isinstance(g, Polygon)]

        for cand in candidates:
            # Shrink back partially so the region approximates true wall bounds
            shrunk = cand.buffer(-HATCH_BUFFER * 0.5)
            if shrunk.is_empty:
                continue
            polys = (
                [shrunk] if isinstance(shrunk, Polygon)
                else [g for g in shrunk.geoms if isinstance(g, Polygon)]
                     if isinstance(shrunk, (MultiPolygon, GeometryCollection))
                else []
            )
            for poly in polys:
                if not isinstance(poly, Polygon) or poly.area < 1.0:
                    continue
                try:
                    thickness = min_bounding_thickness(poly)
                    length = max_bounding_length(poly)
                except Exception:
                    continue
                # Must have wall-like aspect ratio AND thickness
                if (min_wall_pts * 0.5 <= thickness <= max_wall_pts * 1.5
                        and length > thickness * 2.0
                        and length < 2000):  # sanity: no sheet-spanning regions
                    hatch_walls.append(poly)

    all_fill_walls = direct_walls + hatch_walls

    # ---- 2d. Remove Phase-1 lines consumed by fill-wall borders ----
    remaining = phase1_lines
    if all_fill_walls:
        boundary_buf = unary_union(
            [w.boundary.buffer(1.0) for w in all_fill_walls]
        )
        remaining = []
        for line in phase1_lines:
            if line.length < 0.01:
                continue
            overlap = line.intersection(boundary_buf)
            frac = overlap.length / line.length if line.length > 0 else 0
            if frac < 0.8:
                remaining.append(line)

    return all_fill_walls, remaining


# ================================================================
#  PHASE 3 — RAY-CASTING & SPATIAL GRAPH
# ================================================================
# For walls drawn as un-filled parallel lines, simple 1:1 matching
# fails at T-junctions.  Phase 3:
#   1. For each consolidated line, cast perpendicular rays at regular
#      intervals up to MAX_WALL_THICKNESS distance.
#   2. If multiple consecutive rays hit the same target line at a
#      consistent distance → confirmed wall pair.
#   3. Build a quadrilateral polygon from each pair.
#   4. Build a NetworkX graph (nodes = wall polygons, edges = touch /
#      intersect) and union each connected subgraph for seamless
#      wall footprint polygons.
# ================================================================

def phase3_ray_casting(lines: List[LineString]) -> List[Polygon]:
    """Return wall polygons discovered by perpendicular ray-casting."""

    if len(lines) < 2:
        return []

    min_wall_pts = MIN_WALL_THICKNESS * SCALE_FACTOR
    max_wall_pts = MAX_WALL_THICKNESS * SCALE_FACTOR

    # Spatial index for fast intersection queries
    tree = STRtree(lines)

    confirmed_pairs: Set[Tuple[int, int]] = set()
    wall_polygons: List[Polygon] = []

    for src_i, src_line in enumerate(lines):
        coords = np.array(src_line.coords)
        p1, p2 = coords[0], coords[-1]
        seg_vec = p2 - p1
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len < MIN_LINE_LENGTH:
            continue

        direction = seg_vec / seg_len
        normal = np.array([-direction[1], direction[0]])

        n_rays = max(2, int(seg_len / RAY_INTERVAL))

        # Accumulate hits:  target_idx → [(position_along_src, signed_dist)]
        hit_map: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

        for ri in range(n_rays + 1):
            t = ri / n_rays
            origin = p1 + t * seg_vec

            ray_end_pos = origin + normal * max_wall_pts
            ray_end_neg = origin - normal * max_wall_pts
            ray_line = LineString([ray_end_neg.tolist(), ray_end_pos.tolist()])

            candidate_indices = tree.query(ray_line)

            for ci in candidate_indices:
                ci = int(ci)
                if ci == src_i:
                    continue

                target = lines[ci]
                isect = ray_line.intersection(target)
                if isect.is_empty:
                    continue

                # Extract intersection point
                if isect.geom_type == "Point":
                    ip = np.array([isect.x, isect.y])
                elif isect.geom_type in ("MultiPoint", "GeometryCollection"):
                    pts = [g for g in isect.geoms if g.geom_type == "Point"]
                    if not pts:
                        continue
                    dists = [
                        np.linalg.norm(np.array([g.x, g.y]) - origin)
                        for g in pts
                    ]
                    best = pts[int(np.argmin(dists))]
                    ip = np.array([best.x, best.y])
                else:
                    continue

                signed_dist = float(np.dot(ip - origin, normal))
                abs_dist = abs(signed_dist)

                if min_wall_pts <= abs_dist <= max_wall_pts:
                    hit_map[ci].append((t * seg_len, signed_dist))

        # ---- Analyse hit sequences per target ----
        for tgt_i, hits in hit_map.items():
            if len(hits) < MIN_CONSECUTIVE_HITS:
                continue

            pair_key = (min(src_i, tgt_i), max(src_i, tgt_i))
            if pair_key in confirmed_pairs:
                continue

            hits.sort(key=lambda h: h[0])
            distances = [abs(h[1]) for h in hits]

            # Longest run of consistent perpendicular distances
            best_start, best_len = 0, 1
            run_start = 0
            for k in range(1, len(distances)):
                if abs(distances[k] - distances[run_start]) <= HIT_DIST_VARIANCE:
                    if k - run_start + 1 > best_len:
                        best_start = run_start
                        best_len = k - run_start + 1
                else:
                    run_start = k

            if best_len < MIN_CONSECUTIVE_HITS:
                continue

            avg_dist = float(np.mean(
                distances[best_start : best_start + best_len]
            ))
            if not (min_wall_pts <= avg_dist <= max_wall_pts):
                continue

            confirmed_pairs.add(pair_key)

            wall = _build_wall_polygon(
                src_line,
                lines[tgt_i],
                hits[best_start][0],
                hits[best_start + best_len - 1][0],
            )
            if wall is not None:
                wall_polygons.append(wall)

    # Return raw wall quadrilaterals — merging is done by the pipeline
    # with orientation awareness so perpendicular walls stay separate.
    return wall_polygons


def _build_wall_polygon(
    line1: LineString,
    line2: LineString,
    t_start: float,
    t_end: float,
) -> Optional[Polygon]:
    """Build a quadrilateral wall polygon from two parallel lines.

    *t_start* / *t_end* are positions (pts) along *line1* that define
    the overlap region.
    """
    c1 = np.array(line1.coords)
    c2 = np.array(line2.coords)
    p1a, p1b = c1[0], c1[-1]
    p2a, p2b = c2[0], c2[-1]

    dir1 = p1b - p1a
    len1 = float(np.linalg.norm(dir1))
    if len1 < 0.01:
        return None
    dir1_n = dir1 / len1

    dir2 = p2b - p2a
    len2 = float(np.linalg.norm(dir2))
    if len2 < 0.01:
        return None
    dir2_n = dir2 / len2

    t_start = max(0.0, t_start)
    t_end = min(len1, t_end)
    if t_end - t_start < MIN_LINE_LENGTH:
        return None

    # Two corners on line1
    a1 = p1a + t_start * dir1_n
    a2 = p1a + t_end * dir1_n

    # Corresponding two corners on line2 (perpendicular projection)
    s1 = float(np.clip(np.dot(a1 - p2a, dir2_n), 0, len2))
    s2 = float(np.clip(np.dot(a2 - p2a, dir2_n), 0, len2))
    b1 = p2a + s1 * dir2_n
    b2 = p2a + s2 * dir2_n

    try:
        poly = Polygon([a1.tolist(), a2.tolist(), b2.tolist(), b1.tolist()])
        if poly.is_valid and poly.area > 1.0:
            return poly
        poly = poly.buffer(0)  # attempt to fix self-intersection
        if isinstance(poly, Polygon) and poly.is_valid and poly.area > 1.0:
            return poly
    except Exception:
        pass
    return None


def _wall_params(poly: Polygon) -> Tuple[float, float]:
    """Return ``(orientation_angle, centerline_offset)`` of a wall polygon.

    *orientation_angle* is in [0, π) — the direction of the wall's long axis.
    *centerline_offset* is the signed perpendicular distance from the origin
    to the wall's centerline, measured along the wall's normal direction.
    These two values together uniquely identify which *logical wall* a
    polygon segment belongs to.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mrr = poly.minimum_rotated_rectangle
            c = list(mrr.exterior.coords)
            e1 = np.array([c[1][0] - c[0][0], c[1][1] - c[0][1]])
            e2 = np.array([c[2][0] - c[1][0], c[2][1] - c[1][1]])
            long_edge = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
            angle = float(math.atan2(long_edge[1], long_edge[0]) % math.pi)
            if math.isnan(angle):
                raise ValueError
        except Exception:
            b = poly.bounds
            angle = 0.0 if (b[2] - b[0]) >= (b[3] - b[1]) else math.pi / 2

    # Normal direction (perpendicular to long axis)
    normal = np.array([-math.sin(angle), math.cos(angle)])
    cx, cy = poly.centroid.x, poly.centroid.y
    offset = float(cx * normal[0] + cy * normal[1])
    return angle, offset


def _merge_wall_graph(
    wall_polygons: List[Polygon],
    angle_tol: float = 0.15,        # ~8.6° — merge only roughly-parallel walls
    offset_tol: float = 3.0,        # pts — merge only walls on the same centerline
) -> List[Polygon]:
    """Merge touching wall polygons via a NetworkX graph.

    Two walls are connected **only** if they:
      1. Have similar orientation (within *angle_tol*).
      2. Share the same centerline (perpendicular offset within *offset_tol*).
      3. Geometrically touch or nearly touch (< 1.5 pts).

    This correctly merges broken segments of the same wall at T-junctions
    while keeping adjacent parallel walls (e.g., corridor walls) separate.
    """
    if not wall_polygons:
        return []

    params = [_wall_params(p) for p in wall_polygons]

    G = nx.Graph()
    G.add_nodes_from(range(len(wall_polygons)))

    tree = STRtree(wall_polygons)

    for i, poly in enumerate(wall_polygons):
        buffered = poly.buffer(1.5)
        candidates = tree.query(buffered)
        for j in candidates:
            j = int(j)
            if j <= i:
                continue
            # Check orientation compatibility
            a_diff = abs(params[i][0] - params[j][0])
            a_diff = min(a_diff, math.pi - a_diff)
            if a_diff > angle_tol:
                continue
            # Check centerline alignment
            if abs(params[i][1] - params[j][1]) > offset_tol:
                continue
            if (wall_polygons[j].intersects(poly)
                    or wall_polygons[j].distance(poly) < 1.5):
                G.add_edge(i, j)

    result: List[Polygon] = []
    for component in nx.connected_components(G):
        merged = unary_union([wall_polygons[idx] for idx in component])
        if isinstance(merged, Polygon):
            result.append(merged)
        elif isinstance(merged, (MultiPolygon, GeometryCollection)):
            result.extend(
                g for g in merged.geoms if isinstance(g, Polygon)
            )
    return result


# ================================================================
#  PIPELINE
# ================================================================

def extract_walls(
    pdf_path: str,
    page_num: int = 0,
) -> List[Polygon]:
    """Main entry point — extract wall footprint polygons from a PDF page.

    Parameters
    ----------
    pdf_path : str
        Path to the architectural PDF.
    page_num : int
        Zero-based page index (default ``0``).

    Returns
    -------
    list[Polygon]
        Cleaned, distinct wall-footprint polygons (Shapely).
    """
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        raise ValueError(
            f"Page {page_num} does not exist (document has {len(doc)} pages)"
        )
    page = doc[page_num]
    drawings = page.get_drawings()

    print(f"[info] Page {page_num}: size={page.rect}, rotation={page.rotation}")
    print(f"[info] Drawing paths: {len(drawings)}")
    print(
        f"[info] Scale: {SCALE_FACTOR} pts/in  |  "
        f"wall range: {MIN_WALL_THICKNESS}\"–{MAX_WALL_THICKNESS}\""
    )

    # ---- Phase 1 ----
    print("\n=== Phase 1: Line Consolidation ===")
    consolidated = phase1_consolidate_lines(drawings)
    print(f"  Consolidated lines: {len(consolidated)}")

    # ---- Phase 2 ----
    print("\n=== Phase 2: Fill Analysis ===")
    fill_walls, remaining = phase2_fill_analysis(drawings, consolidated)
    print(f"  Wall polygons from fills: {len(fill_walls)}")
    print(f"  Lines remaining for Phase 3: {len(remaining)}")

    # ---- Phase 3 ----
    print("\n=== Phase 3: Ray-Casting & Spatial Graph ===")
    ray_walls = phase3_ray_casting(remaining)
    print(f"  Wall polygons from ray casting: {len(ray_walls)}")

    # ---- Pre-filter: validate individual wall polygons ----
    min_wall_pts = MIN_WALL_THICKNESS * SCALE_FACTOR
    max_wall_pts = MAX_WALL_THICKNESS * SCALE_FACTOR
    min_len_pts = MIN_WALL_LENGTH * SCALE_FACTOR

    mb = page.mediabox
    page_xmin, page_ymin = mb.x0 - 5, mb.y0 - 5
    page_xmax, page_ymax = mb.x1 + 5, mb.y1 + 5

    all_walls = fill_walls + ray_walls
    valid_walls: List[Polygon] = []
    for wall in all_walls:
        b = wall.bounds
        if (b[2] < page_xmin or b[0] > page_xmax
                or b[3] < page_ymin or b[1] > page_ymax):
            continue
        try:
            t = min_bounding_thickness(wall)
            l = max_bounding_length(wall)
        except Exception:
            continue
        if min_wall_pts <= t <= max_wall_pts and l >= min_len_pts:
            valid_walls.append(wall)

    print(f"\n=== Validation & Merge ===")
    print(f"  Walls passing thickness/length filter: {len(valid_walls)}")

    # ---- Merge collinear touching walls (orientation-aware) ----
    merged = _merge_wall_graph(valid_walls)
    print(f"  After orientation-aware merge: {len(merged)}")

    # ---- Deduplicate: remove walls mostly contained in a larger wall ----
    merged.sort(key=lambda p: p.area, reverse=True)
    keep = [True] * len(merged)
    tree = STRtree(merged)
    for i in range(len(merged)):
        if not keep[i]:
            continue
        candidates = tree.query(merged[i])
        for j in candidates:
            j = int(j)
            if j <= i or not keep[j]:
                continue
            # j is a smaller polygon — check containment
            overlap = merged[i].intersection(merged[j])
            if overlap.area > 0.6 * merged[j].area:
                keep[j] = False

    final = [merged[i] for i in range(len(merged)) if keep[i]]
    print(f"  After deduplication: {len(final)}")

    doc.close()
    return final


# ================================================================
#  OVERLAY RENDERING
# ================================================================

def render_overlay(
    pdf_path: str,
    walls: List[Polygon],
    output_path: str = "walls_overlay.pdf",
    page_num: int = 0,
    fill_color: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    fill_opacity: float = 0.35,
    stroke_color: Tuple[float, float, float] = (0.8, 0.0, 0.0),
    stroke_width: float = 0.5,
) -> str:
    """Render detected wall polygons as a coloured overlay on the original PDF.

    Opens *pdf_path*, draws each wall polygon as a semi-transparent
    filled shape on top of the existing page content, and saves the
    result to *output_path*.

    Parameters
    ----------
    pdf_path : str
        Source architectural PDF.
    walls : list[Polygon]
        Wall polygons (Shapely) in the page's coordinate space.
    output_path : str
        Where to write the annotated PDF (default ``"walls_overlay.pdf"``).
    page_num : int
        Page index to annotate.
    fill_color : tuple
        RGB fill for the wall highlight (0-1 per channel).
    fill_opacity : float
        Opacity of the wall fill (0 = invisible, 1 = opaque).
    stroke_color : tuple
        RGB stroke colour for the wall outline.
    stroke_width : float
        Line width for the wall outline (PDF points).

    Returns
    -------
    str
        The path of the written overlay PDF.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    for wall in walls:
        # Extract exterior ring coordinates
        coords = list(wall.exterior.coords)
        if len(coords) < 3:
            continue

        # Build a PyMuPDF Shape for this polygon
        shape = page.new_shape()

        # Move to the first vertex, then draw lines to the rest
        shape.draw_polyline([fitz.Point(x, y) for x, y in coords])

        shape.finish(
            color=stroke_color,
            fill=fill_color,
            fill_opacity=fill_opacity,
            stroke_opacity=min(fill_opacity + 0.3, 1.0),
            width=stroke_width,
            closePath=True,
        )
        shape.commit()

    doc.save(output_path)
    doc.close()
    return output_path


# ================================================================
#  CLI
# ================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <pdf_path> [page_num]")
        sys.exit(1)

    _pdf = sys.argv[1]
    _page = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    walls = extract_walls(_pdf, _page)

    print(f"\n{'=' * 60}")
    print(f" RESULTS: {len(walls)} wall polygon(s)")
    print(f"{'=' * 60}")

    for idx, wall in enumerate(walls):
        try:
            thickness_pts = min_bounding_thickness(wall)
            length_pts = max_bounding_length(wall)
        except Exception:
            thickness_pts = length_pts = 0.0
        real_t = thickness_pts / SCALE_FACTOR if SCALE_FACTOR else 0
        real_l = length_pts / SCALE_FACTOR if SCALE_FACTOR else 0
        b = wall.bounds

        print(f"\n  Wall {idx + 1}:")
        print(f"    Thickness : {real_t:.1f}\"  ({thickness_pts:.1f} pts)")
        print(f"    Length    : {real_l / 12:.1f}'  ({length_pts:.1f} pts)")
        print(f"    Area      : {wall.area:.0f} pts²")
        print(f"    Bounds    : ({b[0]:.1f}, {b[1]:.1f}) – ({b[2]:.1f}, {b[3]:.1f})")
