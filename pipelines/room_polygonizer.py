#!/usr/bin/env python3
"""
room_polygonizer.py — Closed Room Polygon Extraction from Unclosed Wall Segments
=================================================================================

Ingests a raw list of 2D line segments (representing unclosed walls in a floor
plan) and outputs a list of perfectly closed, non-overlapping polygons
representing distinct rooms.

Algorithm:
    Step 1: Ingestion & R-tree spatial index initialization
    Step 2: Gap healing via ray projection from dangling endpoints
    Step 3: Noding (unary_union) & polygonization (shapely.ops.polygonize)
    Step 4: Post-processing — filter wall-thickness artifacts & exterior polygon

Dependencies:
    pip install shapely rtree

Usage::

    from shapely.geometry import LineString
    from pipelines.room_polygonizer import extract_rooms

    walls = [
        LineString([(0, 0), (10, 0)]),
        LineString([(10, 0), (10, 10)]),
        ...
    ]
    rooms = extract_rooms(walls)
    # rooms: List[Polygon] — closed, valid, non-overlapping room polygons
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

from rtree import index
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)
from shapely.ops import polygonize, unary_union


# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

MAX_GAP_TOLERANCE: float = 48.0    # max distance (units) to extend a
                                    # dangling wall to find an intersection

MIN_ROOM_AREA: float = 4.0         # minimum polygon area to keep
                                    # (filters wall thicknesses & columns)

COORD_PRECISION: int = 4            # decimal places for endpoint rounding
                                    # (mitigates floating-point drift)


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1 — Ingestion & Spatial Index Initialization
# ═══════════════════════════════════════════════════════════════════════════


def _build_spatial_index(
    lines: List[LineString],
) -> Tuple[index.Index, dict[int, LineString]]:
    """Build an R-tree spatial index from line segment bounding boxes.

    Returns:
        (rtree_index, id_to_line) — the spatial index and a dictionary
        mapping R-tree integer IDs back to the original LineString objects
        for O(1) retrieval during intersection checks.
    """
    idx = index.Index()
    id_to_line: dict[int, LineString] = {}

    for i, line in enumerate(lines):
        idx.insert(i, line.bounds)
        id_to_line[i] = line

    return idx, id_to_line


# ═══════════════════════════════════════════════════════════════════════════
#  Step 2 — Gap Healing (Ray Projection)
# ═══════════════════════════════════════════════════════════════════════════


def _round_coord(coord: Tuple[float, float]) -> Tuple[float, float]:
    """Round a 2D coordinate to ``COORD_PRECISION`` decimal places to
    counter floating-point drift in floor-plan vectors."""
    return (round(coord[0], COORD_PRECISION), round(coord[1], COORD_PRECISION))


def _find_dangling_nodes(
    lines: List[LineString],
) -> List[Tuple[Tuple[float, float], Tuple[float, float], int]]:
    """Find all dangling nodes (endpoints appearing exactly once).

    Returns a list of ``(dangling_coord, raw_coord, parent_line_index)``
    triples.  ``dangling_coord`` is the rounded coordinate used for
    counting; ``raw_coord`` is the original unrounded coordinate for
    geometric operations.
    """
    # Count occurrences of each rounded endpoint across all lines.
    endpoint_counts: Counter[Tuple[float, float]] = Counter()
    # Map rounded coord → (raw_coord, parent_line_index).
    endpoint_info: dict[Tuple[float, float], Tuple[Tuple[float, float], int]] = {}

    for i, line in enumerate(lines):
        coords = list(line.coords)
        for raw in (coords[0], coords[-1]):
            rounded = _round_coord(raw)
            endpoint_counts[rounded] += 1
            endpoint_info[rounded] = (raw, i)

    # Dangling = count of exactly 1.
    dangling: List[Tuple[Tuple[float, float], Tuple[float, float], int]] = []
    for rounded, count in endpoint_counts.items():
        if count == 1:
            raw, line_idx = endpoint_info[rounded]
            dangling.append((rounded, raw, line_idx))

    return dangling


def _compute_ray_direction(
    parent: LineString,
    dangling_raw: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """Compute the unit direction vector for a ray extending from a
    dangling endpoint, pointing *away* from the parent line's interior.

    The direction runs from the adjacent coordinate toward the dangling
    coordinate (i.e. the trajectory the wall was heading when it stopped).
    """
    coords = list(parent.coords)
    start_rounded = _round_coord(coords[0])
    end_rounded = _round_coord(coords[-1])
    dangle_rounded = _round_coord(dangling_raw)

    if start_rounded == dangle_rounded:
        # Dangling at the start of the line.
        dx = coords[0][0] - coords[1][0]
        dy = coords[0][1] - coords[1][1]
    elif end_rounded == dangle_rounded:
        # Dangling at the end of the line.
        dx = coords[-1][0] - coords[-2][0]
        dy = coords[-1][1] - coords[-2][1]
    else:
        return None

    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-10:
        return None

    return (dx / length, dy / length)


def _closest_intersection_point(
    ray: LineString,
    origin: Tuple[float, float],
    candidate: LineString,
) -> Optional[Tuple[Point, float]]:
    """Return the closest intersection point between *ray* and *candidate*
    line, along with its distance from *origin*.

    Handles Point, MultiPoint, LineString, and GeometryCollection results
    from ``shapely.intersection()``.
    """
    intersection = ray.intersection(candidate)

    if intersection.is_empty:
        return None

    # Collect all candidate points from the intersection geometry.
    points: List[Point] = []
    geom_type = intersection.geom_type

    if geom_type == "Point":
        points.append(intersection)
    elif geom_type == "MultiPoint":
        points.extend(intersection.geoms)
    elif geom_type in ("LineString", "MultiLineString", "GeometryCollection"):
        # Collinear overlap or mixed collection — extract all vertices.
        if hasattr(intersection, "geoms"):
            for geom in intersection.geoms:
                if geom.geom_type == "Point":
                    points.append(geom)
                elif hasattr(geom, "coords"):
                    points.extend(Point(c) for c in geom.coords)
        elif hasattr(intersection, "coords"):
            points.extend(Point(c) for c in intersection.coords)

    if not points:
        return None

    origin_pt = Point(origin)
    best_point: Optional[Point] = None
    best_dist = float("inf")

    for pt in points:
        dist = origin_pt.distance(pt)
        # Skip near-zero distances (self-intersection at origin).
        if dist > 1e-3 and dist < best_dist:
            best_dist = dist
            best_point = pt

    if best_point is None:
        return None

    return (best_point, best_dist)


def _heal_gaps(
    lines: List[LineString],
    spatial_idx: index.Index,
    id_to_line: dict[int, LineString],
    max_gap: float = MAX_GAP_TOLERANCE,
) -> List[LineString]:
    """Heal gaps by extending dangling endpoints along their trajectory.

    For each dangling node:
      1. Calculate the directional vector from the parent line.
      2. Create a ray extending along that vector by *max_gap*.
      3. Use the R-tree to find candidate lines whose bounding boxes
         intersect the ray's bounding box.
      4. Use ``shapely.intersection()`` to find the exact closest hit.
      5. Create a new LineString bridging the dangling node and the hit.

    Returns:
        A list of new healing LineString segments to append to the
        main line collection.
    """
    dangling = _find_dangling_nodes(lines)
    healed: List[LineString] = []

    for _rounded, raw_coord, line_idx in dangling:
        parent = lines[line_idx]

        direction = _compute_ray_direction(parent, raw_coord)
        if direction is None:
            continue

        dx, dy = direction

        # Create ray from the dangling node, extending by max_gap.
        ray_end = (raw_coord[0] + dx * max_gap, raw_coord[1] + dy * max_gap)
        ray = LineString([raw_coord, ray_end])

        # Spatial query: candidates whose bounding boxes overlap the ray.
        candidates = list(spatial_idx.intersection(ray.bounds))

        # Find the closest exact intersection across all candidates.
        best_point: Optional[Point] = None
        best_dist = float("inf")

        for cand_id in candidates:
            if cand_id == line_idx:
                continue  # skip the parent line
            cand_line = id_to_line[cand_id]

            result = _closest_intersection_point(ray, raw_coord, cand_line)
            if result is not None:
                pt, dist = result
                if dist < best_dist:
                    best_dist = dist
                    best_point = pt

        if best_point is not None and best_dist <= max_gap:
            heal_line = LineString([raw_coord, (best_point.x, best_point.y)])
            if heal_line.length > 1e-3:
                healed.append(heal_line)

    return healed


# ═══════════════════════════════════════════════════════════════════════════
#  Step 3 — Noding & Polygonization (The Core Extraction)
# ═══════════════════════════════════════════════════════════════════════════


def _extract_lines_from_geometry(geom) -> List[LineString]:
    """Recursively extract all LineString components from any Shapely
    geometry, handling collinear-overlap artifacts from ``unary_union``.

    Filters out Points, Polygons, and zero-length lines so that the
    result is safe to pass directly into ``polygonize()``.
    """
    lines: List[LineString] = []

    if geom is None or geom.is_empty:
        return lines

    geom_type = geom.geom_type

    if geom_type == "LineString":
        if geom.length > 0:
            lines.append(geom)
    elif geom_type == "MultiLineString":
        for sub in geom.geoms:
            if sub.length > 0:
                lines.append(sub)
    elif geom_type == "GeometryCollection":
        for sub in geom.geoms:
            lines.extend(_extract_lines_from_geometry(sub))
    # Silently ignore Point, MultiPoint, Polygon, etc.

    return lines


def _node_and_polygonize(lines: List[LineString]) -> List[Polygon]:
    """Node the line graph and extract minimal closed polygons.

    1. ``unary_union()`` natively planarizes the graph — breaking all
       crossing lines into distinct segments at their intersection points
       and removing duplicates.
    2. ``polygonize()`` automatically traverses the planar graph and
       returns a generator of all minimal closed Polygon objects.

    Returns:
        A list of Polygon objects (raw, before area filtering).
    """
    if not lines:
        return []

    # Node the graph: planarize intersections & remove duplicates.
    noded = unary_union(lines)

    # Extract strictly LineString/MultiLineString components.
    noded_lines = _extract_lines_from_geometry(noded)

    if not noded_lines:
        return []

    # Extract minimal cycles as polygons.
    return list(polygonize(noded_lines))


# ═══════════════════════════════════════════════════════════════════════════
#  Step 4 — Post-Processing & Filtering
# ═══════════════════════════════════════════════════════════════════════════


def _filter_polygons(
    polygons: List[Polygon],
    min_area: float = MIN_ROOM_AREA,
) -> List[Polygon]:
    """Filter raw polygons to produce the final room list.

    1. **Filter wall artifacts**: discard any polygon with
       ``.area < min_area`` (wall-thickness strips, columns, slivers).
    2. **Filter the exterior**: if the single largest polygon *contains*
       the majority of other polygons (i.e. it is the bounding exterior
       of the floor plan), discard it.  If the largest polygon does NOT
       enclose the others, it is a legitimate room (e.g. a garage or
       great room) and is kept.

    ``polygonize`` guarantees non-overlapping output, so no overlap
    removal is needed.
    """
    if not polygons:
        return []

    # Filter by minimum area.
    filtered = [p for p in polygons if p.area >= min_area]

    if len(filtered) <= 1:
        return filtered

    # Identify the largest polygon.
    max_idx = max(range(len(filtered)), key=lambda i: filtered[i].area)
    largest = filtered[max_idx]

    # Containment test: does the largest polygon enclose most others?
    # If yes, it is the exterior boundary and should be removed.
    # If no, it is a real room (garage, great room, etc.) — keep it.
    other_count = len(filtered) - 1
    contained = 0
    for i, p in enumerate(filtered):
        if i == max_idx:
            continue
        # representative_point() is guaranteed inside the polygon,
        # unlike centroid which can fall outside concave shapes.
        if largest.contains(p.representative_point()):
            contained += 1

    if contained > other_count * 0.5:
        print(f"  [polygonizer] Removing exterior polygon "
              f"(area={largest.area:.0f}, contains {contained}/{other_count} rooms)")
        filtered.pop(max_idx)

    return filtered


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════


def sqft_to_sqpt(sqft: float, scale_ratio: float) -> float:
    """Convert square feet to square PDF points for a given scale ratio.

    Useful for computing a ``min_room_area`` threshold in coordinate-space
    units from a human-readable square-footage value.

    Example::

        # At 1/8" = 1'-0" (scale_ratio=96):
        sqft_to_sqpt(25, 96.0)  # → ~2025 sqpt
    """
    sqpt_per_sqft = 1.0 / ((scale_ratio / 72.0) ** 2 / 144.0)
    return sqft * sqpt_per_sqft


def extract_rooms(
    wall_lines: List[LineString],
    *,
    max_gap_tolerance: float = MAX_GAP_TOLERANCE,
    min_room_area: float = MIN_ROOM_AREA,
    min_room_area_sqft: Optional[float] = None,
    scale_ratio: Optional[float] = None,
) -> List[Polygon]:
    """Extract closed room polygons from unclosed wall line segments.

    This is the main entry point for the module.

    Args:
        wall_lines: Raw wall vectors as Shapely ``LineString`` objects.
        max_gap_tolerance: Maximum distance (in coordinate units) to
            extend a dangling wall endpoint to find an intersection.
        min_room_area: Minimum polygon area in **coordinate-space square
            units** to keep.  Ignored when *min_room_area_sqft* and
            *scale_ratio* are both provided.
        min_room_area_sqft: Minimum polygon area in **square feet**.
            Requires *scale_ratio* to convert to coordinate units.
            When provided, overrides *min_room_area*.
        scale_ratio: Architectural scale ratio (e.g. 96.0 for
            1/8" = 1'-0").  Required when using *min_room_area_sqft*.

    Returns:
        A list of Shapely ``Polygon`` objects representing distinct rooms.
        Guaranteed to be closed, valid, and non-overlapping (by virtue
        of ``shapely.ops.polygonize``).
    """
    if not wall_lines:
        return []

    # Resolve area threshold.
    effective_min_area = min_room_area
    if min_room_area_sqft is not None and scale_ratio is not None:
        effective_min_area = sqft_to_sqpt(min_room_area_sqft, scale_ratio)
        print(f"  [polygonizer] Area threshold: {min_room_area_sqft} ft² "
              f"= {effective_min_area:.0f} sqpt (scale {scale_ratio})")

    # ── Validate input: keep only non-degenerate LineStrings ─────────
    valid_lines: List[LineString] = []
    for line in wall_lines:
        if not isinstance(line, LineString):
            continue
        if line.is_empty or line.length < 1e-10:
            continue
        valid_lines.append(line)

    if not valid_lines:
        return []

    print(f"  [polygonizer] Input: {len(valid_lines)} valid line segments")

    # ── Step 1: Build R-tree spatial index ───────────────────────────
    spatial_idx, id_to_line = _build_spatial_index(valid_lines)
    print(f"  [polygonizer] R-tree spatial index built ({len(valid_lines)} entries)")

    # ── Step 2: Gap healing (ray projection) ─────────────────────────
    healed = _heal_gaps(
        valid_lines, spatial_idx, id_to_line, max_gap_tolerance
    )
    print(f"  [polygonizer] Gap healing: {len(healed)} bridge segments created")

    # Combine original + healed lines for noding.
    all_lines = valid_lines + healed

    # Update the spatial index with healed lines (for downstream use).
    next_id = len(valid_lines)
    for hl in healed:
        spatial_idx.insert(next_id, hl.bounds)
        id_to_line[next_id] = hl
        next_id += 1

    # ── Step 3: Node & polygonize ────────────────────────────────────
    polygons = _node_and_polygonize(all_lines)
    print(f"  [polygonizer] Polygonize: {len(polygons)} raw polygons extracted")

    # ── Step 4: Post-processing ──────────────────────────────────────
    rooms = _filter_polygons(polygons, effective_min_area)
    print(f"  [polygonizer] After filtering: {len(rooms)} room polygons")

    return rooms
