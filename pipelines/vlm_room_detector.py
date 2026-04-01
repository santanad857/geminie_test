#!/usr/bin/env python3
"""
Endpoint Connection Detection
==============================

Identifies exposed wall endpoints (doorjambs), then connects each to
its nearest neighbor using greedy distance-based matching with a
no-crossing constraint.

Algorithm:
    Phase 1: Clean geometry, extract exposed (degree-1) endpoints
    Phase 2: Greedy nearest-neighbor matching (no crossings allowed)
    Phase 3: Render overlay on PDF

Usage::

    from pipelines.vlm_room_detector import run_opening_detection

    connections = run_opening_detection("plan.pdf", scale_ratio=96.0)

CLI::

    python -m pipelines.vlm_room_detector --pdf plan.pdf --scale 96
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
from scipy.spatial import KDTree

from pipelines.room_pipeline import (
    WallSegment,
    _merge_colinear_segments,
    _stitch_t_junctions,
    _weld_endpoints,
    extract_wall_segments_for_room_pipeline,
    find_exposed_endpoints,
    load_env,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class VerifiedConnection:
    """A connection between two exposed endpoints."""

    endpoint_a: tuple[float, float]
    endpoint_b: tuple[float, float]
    distance: float


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1 — Pre-Computation & Endpoint Extraction
# ═══════════════════════════════════════════════════════════════════════════


def _extend_dangling_stubs(
    segments: list[WallSegment],
    max_stub_len: float = 25.0,
    max_extension: float = 50.0,
    epsilon: float = 5.0,
) -> list[WallSegment]:
    """Extend short dangling stubs along their direction until they hit
    a perpendicular segment body.

    A "dangling stub" is a short segment (length < *max_stub_len*) with
    at least one endpoint that has no neighbor within *epsilon*.  The
    dangling end is extended along the segment's direction (ray-cast)
    up to *max_extension* until it intersects a perpendicular segment.
    """
    pts: list[tuple[float, float]] = []
    for seg in segments:
        pts.append(seg.p1)
        pts.append(seg.p2)
    if not pts:
        return segments

    arr = np.array(pts, dtype=np.float64)
    tree = KDTree(arr)

    result = list(segments)
    extended = 0

    for i, seg in enumerate(segments):
        seg_len = math.hypot(seg.p2[0] - seg.p1[0], seg.p2[1] - seg.p1[1])
        if seg_len > max_stub_len or seg_len < 0.5:
            continue

        # Check which end(s) are dangling.
        for end_idx, (dangling_pt, anchor_pt) in enumerate([
            (seg.p1, seg.p2),
            (seg.p2, seg.p1),
        ]):
            neighbors = tree.query_ball_point(dangling_pt, r=epsilon)
            # Dangling = only its own endpoints nearby (count ≤ 2 from same seg).
            other_neighbors = [
                n for n in neighbors
                if n // 2 != i  # different segment
            ]
            if other_neighbors:
                continue  # not dangling

            # Ray direction: from anchor toward dangling end.
            dx = dangling_pt[0] - anchor_pt[0]
            dy = dangling_pt[1] - anchor_pt[1]
            length = math.hypot(dx, dy)
            if length < 0.1:
                continue
            dx /= length
            dy /= length

            # Determine if this stub is H or V.
            is_h = abs(dx) > abs(dy)

            # Cast ray from dangling_pt along (dx, dy), find nearest
            # perpendicular segment body intersection.
            best_dist = max_extension + 1
            best_hit = None

            for j, other in enumerate(segments):
                if j == i:
                    continue
                ox1, oy1 = other.p1
                ox2, oy2 = other.p2
                odx = ox2 - ox1
                ody = oy2 - oy1
                other_is_h = abs(odx) > abs(ody)

                # Must be perpendicular.
                if is_h == other_is_h:
                    continue

                # H stub hitting V segment: solve for t where
                #   dangling_pt.x + t*dx == other.x
                # V stub hitting H segment: solve for t where
                #   dangling_pt.y + t*dy == other.y
                if is_h:
                    # Other is vertical: x is constant.
                    other_x = (ox1 + ox2) / 2
                    if abs(dx) < 0.01:
                        continue
                    t = (other_x - dangling_pt[0]) / dx
                    if t <= 0 or t > max_extension:
                        continue
                    hit_y = dangling_pt[1] + t * dy
                    min_y = min(oy1, oy2)
                    max_y = max(oy1, oy2)
                    if hit_y < min_y - 2 or hit_y > max_y + 2:
                        continue
                else:
                    # Other is horizontal: y is constant.
                    other_y = (oy1 + oy2) / 2
                    if abs(dy) < 0.01:
                        continue
                    t = (other_y - dangling_pt[1]) / dy
                    if t <= 0 or t > max_extension:
                        continue
                    hit_x = dangling_pt[0] + t * dx
                    min_x = min(ox1, ox2)
                    max_x = max(ox1, ox2)
                    if hit_x < min_x - 2 or hit_x > max_x + 2:
                        continue

                if t < best_dist:
                    best_dist = t
                    best_hit = (
                        dangling_pt[0] + t * dx,
                        dangling_pt[1] + t * dy,
                    )

            if best_hit is not None:
                # Replace the segment with the extended version.
                if end_idx == 0:
                    result[i] = WallSegment(p1=best_hit, p2=seg.p2, source=seg.source)
                else:
                    result[i] = WallSegment(p1=seg.p1, p2=best_hit, source=seg.source)
                extended += 1
                break  # only extend one end

    if extended:
        print(f"  Stub extensions: {extended}")
    return result


def cleanup_geometry(
    segments: list[WallSegment],
    *,
    merge_tol: float = 1.5,
    weld_radius: float = 3.5,
    stitch_tolerance: float = 5.0,
) -> list[WallSegment]:
    """Consolidated geometric cleanup: collinear merge, T-junction
    stitching, endpoint welding, stub extension, final merge."""
    cleaned = _merge_colinear_segments(segments, tol=merge_tol)
    cleaned = _stitch_t_junctions(cleaned, tolerance=stitch_tolerance)
    cleaned = _weld_endpoints(cleaned, weld_radius=weld_radius)
    cleaned = _extend_dangling_stubs(cleaned)
    cleaned = _merge_colinear_segments(cleaned, tol=merge_tol)
    return cleaned


def extract_exposed_coords(
    segments: list[WallSegment],
    epsilon: float = 5.0,
) -> list[tuple[tuple[float, float], int]]:
    """Extract exposed (degree-1) endpoint coordinates with parent
    segment index.  These are the doorjamb endpoints — NOT wall
    corners where two segments meet."""
    exposed = find_exposed_endpoints(segments, epsilon)
    return [(ep.point, ep.seg_idx) for ep in exposed]


def dedup_endpoints(
    exposed_coords: list[tuple[tuple[float, float], int]],
    radius: float = 10.0,
) -> list[tuple[tuple[float, float], int]]:
    """Merge exposed endpoints within *radius* into their centroid.

    Endpoints closer than *radius* are duplicates (e.g. two border-line
    endpoints that collapsed to nearly the same centerline point).
    Each cluster is replaced by a single (centroid, first_seg_idx) entry.
    """
    if len(exposed_coords) < 2:
        return list(exposed_coords)

    points = [c[0] for c in exposed_coords]
    seg_indices = [c[1] for c in exposed_coords]
    arr = np.array(points, dtype=np.float64)
    tree = KDTree(arr)

    # Union-find to cluster nearby points.
    parent = list(range(len(points)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(points)):
        neighbors = tree.query_ball_point(points[i], r=radius)
        for j in neighbors:
            if j != i:
                _union(i, j)

    # Build clusters and compute centroids.
    from collections import defaultdict
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(points)):
        clusters[_find(i)].append(i)

    deduped: list[tuple[tuple[float, float], int]] = []
    for members in clusters.values():
        cx = sum(points[m][0] for m in members) / len(members)
        cy = sum(points[m][1] for m in members) / len(members)
        deduped.append(((cx, cy), seg_indices[members[0]]))

    return deduped


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2 — Greedy Nearest-Neighbor Matching
# ═══════════════════════════════════════════════════════════════════════════


def _ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Cross-product sign: positive if A→B→C is counter-clockwise."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    """True if segment a1–a2 properly crosses segment b1–b2
    (intersection at interior points of both segments, not at shared
    endpoints)."""
    d1 = _ccw(a1, a2, b1)
    d2 = _ccw(a1, a2, b2)
    d3 = _ccw(b1, b2, a1)
    d4 = _ccw(b1, b2, a2)

    # Both endpoints of each segment must be on opposite sides of the
    # other segment's line (strict inequality — excludes touching).
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False


def _connection_through_empty_space(
    pt_a: tuple[float, float],
    pt_b: tuple[float, float],
    segments: list[WallSegment],
    wall_proximity: float = 3.0,
    max_wall_fraction: float = 0.3,
    n_samples: int = 10,
) -> bool:
    """Check that a connection primarily travels through empty space.

    Samples the interior 80% of the connection (skipping the first and
    last 10% where the line naturally touches the parent walls at each
    endpoint) and checks proximity to ALL wall segments.

    A connection that runs along a wall will have most interior samples
    near wall geometry and is rejected.

    Returns True if the connection is mostly through empty space.
    """
    wall_hits = 0
    for k in range(n_samples):
        # Sample the interior 80%: t ranges from 0.10 to 0.90
        t = 0.10 + 0.80 * (k / max(n_samples - 1, 1))
        sx = pt_a[0] + t * (pt_b[0] - pt_a[0])
        sy = pt_a[1] + t * (pt_b[1] - pt_a[1])

        near_wall = False
        for seg in segments:
            x1, y1 = seg.p1
            x2, y2 = seg.p2
            dx, dy = x2 - x1, y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 0.01:
                d = math.hypot(sx - x1, sy - y1)
            else:
                t_proj = max(0.0, min(1.0, ((sx - x1) * dx + (sy - y1) * dy) / seg_len_sq))
                px, py = x1 + t_proj * dx, y1 + t_proj * dy
                d = math.hypot(sx - px, sy - py)
            if d < wall_proximity:
                near_wall = True
                break

        if near_wall:
            wall_hits += 1

    return (wall_hits / n_samples) <= max_wall_fraction


def match_endpoints(
    exposed_coords: list[tuple[tuple[float, float], int]],
    segments: list[WallSegment],
    max_distance: float,
) -> list[VerifiedConnection]:
    """Greedy nearest-neighbor matching with constraints:

    1. Build all pairwise distances between exposed endpoints.
    2. Sort by distance ascending.
    3. For each pair (closest first):
       - Skip if either endpoint is already consumed.
       - Skip if this connection would cross any accepted connection.
       - Skip if the connection doesn't primarily go through empty space.
       - Otherwise accept it.

    Returns connections sorted by distance.
    """
    if len(exposed_coords) < 2:
        return []

    points = [c[0] for c in exposed_coords]
    seg_indices = [c[1] for c in exposed_coords]
    arr = np.array(points, dtype=np.float64)
    tree = KDTree(arr)

    # Build all candidate pairs within max_distance, sorted by distance.
    candidates: list[tuple[float, int, int]] = []  # (dist, i, j)
    for i, pt in enumerate(points):
        neighbors = tree.query_ball_point(pt, r=max_distance)
        for j in neighbors:
            if j <= i:
                continue
            dist = math.hypot(points[i][0] - points[j][0],
                              points[i][1] - points[j][1])
            candidates.append((dist, i, j))

    candidates.sort()

    # Greedy assignment.
    consumed: set[int] = set()
    accepted: list[VerifiedConnection] = []

    for dist, i, j in candidates:
        if i in consumed or j in consumed:
            continue

        # Check crossing against all accepted connections.
        crosses = False
        for conn in accepted:
            if _segments_cross(points[i], points[j],
                               conn.endpoint_a, conn.endpoint_b):
                crosses = True
                break

        if crosses:
            continue

        # Check that connection primarily goes through empty space.
        if not _connection_through_empty_space(
            points[i], points[j], segments,
        ):
            continue

        consumed.add(i)
        consumed.add(j)
        accepted.append(VerifiedConnection(
            endpoint_a=points[i],
            endpoint_b=points[j],
            distance=dist,
        ))

    return accepted


# ═══════════════════════════════════════════════════════════════════════════
#  Output — Connection Overlay
# ═══════════════════════════════════════════════════════════════════════════


_CONNECTION_COLOR = (0.1, 0.7, 0.2)  # green


def render_connections_overlay(
    pdf_path: str,
    wall_segments: list[WallSegment],
    exposed_coords: list[tuple[tuple[float, float], int]],
    connections: list[VerifiedConnection],
    output_path: str,
    page_index: int = 0,
) -> str:
    """Draw wall segments, exposed endpoints, and connections on the PDF.

    - Walls: thin gray lines
    - Exposed endpoints: red dots
    - Connections: green lines
    """
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    # ── Wall segments (thin gray) ───────────────────────────────────
    for seg in wall_segments:
        shape = page.new_shape()
        shape.draw_line(fitz.Point(*seg.p1), fitz.Point(*seg.p2))
        shape.finish(color=(0.6, 0.6, 0.6), width=0.5)
        shape.commit()

    # ── Exposed endpoints (red dots) ────────────────────────────────
    for coord, _ in exposed_coords:
        shape = page.new_shape()
        shape.draw_circle(fitz.Point(*coord), 2.5)
        shape.finish(color=(0.9, 0.1, 0.1), fill=(0.9, 0.1, 0.1))
        shape.commit()

    # ── Connections (green) ─────────────────────────────────────────
    for conn in connections:
        shape = page.new_shape()
        shape.draw_line(fitz.Point(*conn.endpoint_a),
                        fitz.Point(*conn.endpoint_b))
        shape.finish(color=_CONNECTION_COLOR, width=1.8)
        shape.commit()

        for pt in (conn.endpoint_a, conn.endpoint_b):
            shape = page.new_shape()
            shape.draw_circle(fitz.Point(*pt), 2.0)
            shape.finish(color=_CONNECTION_COLOR, fill=_CONNECTION_COLOR)
            shape.commit()

    doc.save(output_path)
    doc.close()
    return output_path


def _render_labeled_endpoints(
    pdf_path: str,
    wall_segments: list[WallSegment],
    exposed_coords: list[tuple[tuple[float, float], int]],
    output_path: str,
    page_index: int = 0,
) -> str:
    """Render PDF with wall segments and every exposed endpoint labeled
    with its coordinates for debugging."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    for seg in wall_segments:
        shape = page.new_shape()
        shape.draw_line(fitz.Point(*seg.p1), fitz.Point(*seg.p2))
        shape.finish(color=(0.5, 0.5, 0.5), width=0.8)
        shape.commit()

    for pt, si in exposed_coords:
        shape = page.new_shape()
        shape.draw_circle(fitz.Point(*pt), 3)
        shape.finish(color=(1, 0, 0), fill=(1, 0, 0))
        shape.commit()
        page.insert_text(
            fitz.Point(pt[0] + 4, pt[1] + 1),
            f"({pt[0]:.0f},{pt[1]:.0f})",
            fontsize=4,
            color=(1, 0, 0),
        )

    doc.save(output_path)
    doc.close()
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════


def compute_max_opening_pt(
    scale_ratio: float,
    max_real_feet: float = 12.0,
) -> float:
    """Convert real-world maximum opening width (feet) to PDF points."""
    return max_real_feet * 12.0 / scale_ratio * 72.0


def run_opening_detection(
    pdf_path: str,
    *,
    page_index: int = 0,
    scale_ratio: float = 96.0,
    wall_vectors_path: Optional[str] = None,
    wall_vector_mode: str = "auto",
    max_opening_ft: float = 12.0,
    epsilon: float = 5.0,
    focus_x_min: Optional[float] = None,
    exclude_points: Optional[list[tuple[float, float]]] = None,
    force_connections: Optional[list[tuple[tuple[float, float], tuple[float, float]]]] = None,
    output_pdf: str = "output/opening_detection/connections.pdf",
) -> list[VerifiedConnection]:
    """Detect openings between exposed wall endpoints using greedy
    nearest-neighbor matching and render a PDF overlay.

    Returns a list of VerifiedConnection objects.
    """
    print("=" * 62)
    print("   ENDPOINT CONNECTION DETECTION")
    print("=" * 62)

    max_opening_pt = compute_max_opening_pt(scale_ratio, max_opening_ft)
    print(f"  Scale       : {scale_ratio}")
    print(f"  Max opening : {max_opening_pt:.1f} pt ≈ {max_opening_ft} ft")

    # ── Phase 1: Clean geometry, find exposed endpoints ─────────────
    print("\n[Phase 1] Wall extraction & endpoint detection …")
    segments, raw_segments, wall_source = extract_wall_segments_for_room_pipeline(
        pdf_path,
        page_index=page_index,
        wall_vectors_path=wall_vectors_path,
        wall_vector_mode=wall_vector_mode,
    )
    print(f"  Source: {wall_source}")
    print(f"  Input segments: {len(segments)}")

    cleaned = cleanup_geometry(segments, merge_tol=1.5)
    print(f"  After cleanup: {len(cleaned)}")

    exposed_raw = extract_exposed_coords(cleaned, epsilon=epsilon)
    print(f"  Exposed endpoints (raw): {len(exposed_raw)}")
    exposed = dedup_endpoints(exposed_raw, radius=10.0)
    print(f"  After dedup (10pt radius): {len(exposed)}")

    # Filter to main floor plan — drop option inset endpoints.
    if focus_x_min is not None:
        before = len(exposed)
        exposed = [(pt, si) for pt, si in exposed if pt[0] >= focus_x_min]
        print(f"  After focus filter (x >= {focus_x_min}): {len(exposed)} (dropped {before - len(exposed)})")

    # Drop manually excluded endpoints.
    if exclude_points:
        before = len(exposed)
        exposed = [
            (pt, si) for pt, si in exposed
            if not any(abs(pt[0] - ex[0]) < 2 and abs(pt[1] - ex[1]) < 2 for ex in exclude_points)
        ]
        print(f"  After exclusions: {len(exposed)} (dropped {before - len(exposed)})")

    # ── Phase 2: Greedy nearest-neighbor matching ───────────────────
    print("\n[Phase 2] Nearest-neighbor matching …")
    connections = match_endpoints(exposed, cleaned, max_opening_pt)

    # Add forced connections (for testing scenarios with ideal wall vectors).
    if force_connections:
        for pt_a, pt_b in force_connections:
            dist = math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])
            connections.append(VerifiedConnection(
                endpoint_a=pt_a, endpoint_b=pt_b, distance=dist,
            ))
        print(f"  Forced connections: {len(force_connections)}")

    unmatched = len(exposed) - len(connections) * 2
    print(f"  Connections: {len(connections)}")
    print(f"  Unmatched endpoints: {unmatched}")

    for conn in connections:
        print(
            f"    ({conn.endpoint_a[0]:.0f},{conn.endpoint_a[1]:.0f}) → "
            f"({conn.endpoint_b[0]:.0f},{conn.endpoint_b[1]:.0f}) "
            f"d={conn.distance:.0f}pt"
        )

    # ── Render overlay ──────────────────────────────────────────────
    d = os.path.dirname(output_pdf)
    if d:
        os.makedirs(d, exist_ok=True)

    render_connections_overlay(
        pdf_path, cleaned, exposed, connections, output_pdf, page_index,
    )
    print(f"\n  Connections → {output_pdf}")

    # Always generate labeled endpoint debug PDF alongside the connections file.
    base = os.path.splitext(output_pdf)[0]
    labeled_pdf = base + "_endpoints.pdf"
    _render_labeled_endpoints(pdf_path, cleaned, exposed, labeled_pdf, page_index)
    print(f"  Labeled     → {labeled_pdf}")
    print("  Done.\n")

    return connections


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Endpoint connection detection for architectural floor plans.",
    )
    ap.add_argument("--pdf", default="./data/352 AA copy 2.pdf")
    ap.add_argument("--scale", type=float, default=96.0,
                    help='Scale ratio (96 = 1/8"=1\'-0").')
    ap.add_argument("--max-opening-ft", type=float, default=12.0,
                    help="Max opening width in real feet.")
    ap.add_argument("--wall-vectors", default=None,
                    help="Path to JSON wall vectors.")
    ap.add_argument("--wall-vector-mode",
                    choices=("auto", "centerline", "border"), default="auto")
    ap.add_argument("--output",
                    default="output/opening_detection/connections.pdf")

    args = ap.parse_args()
    load_env()

    connections = run_opening_detection(
        pdf_path=args.pdf,
        scale_ratio=args.scale,
        max_opening_ft=args.max_opening_ft,
        wall_vectors_path=args.wall_vectors,
        wall_vector_mode=args.wall_vector_mode,
        output_pdf=args.output,
    )
    return 0 if connections else 1


if __name__ == "__main__":
    sys.exit(main())
