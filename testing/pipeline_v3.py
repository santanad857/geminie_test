#!/usr/bin/env python3
"""
Pipeline v3 — Exhaustive Pattern Extraction + VLM Classification
================================================================

Strategy:
  1. Pre-process: extract drawings, filter dashed/transparent, handle curves.
  2. Enumerate every distinct visual pattern (fill colors, stroke classes,
     hatching signatures) — mechanical, exhaustive, zero assumptions.
  3. For each pattern, build a wall network using connectivity graphs.
  4. For each candidate overlay, ask a VLM: "are these walls, or nonsense?"
  5. Merge VLM-approved patterns into a unified graph, expand and bridge.

Generalizes to any floor plan by separating pattern extraction (mechanical)
from pattern classification (semantic, VLM-driven).

Usage:
    python testing/pipeline_v3.py [path_to_pdf ...]
"""

import base64
import json
import math
import os
import sys
from collections import Counter, defaultdict

import openai
import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env():
    """Load key=value pairs from .env in the project root."""
    for candidate in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    ]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            break


# ---------------------------------------------------------------------------
# Known scales (pt_per_inch) — will be replaced by auto-detection later
# ---------------------------------------------------------------------------

KNOWN_SCALES = {
    "352 AA copy 2.pdf": 0.75,      # 1/8" = 1'-0"
    "second_floor_352.pdf": 0.75,    # 1/8" = 1'-0"
    "custom_floor_plan.pdf": 1.5,    # 1/4" = 1'-0"
    "main_st_ex.pdf": 0.6,          # 1" = 10'-0"
    "main_st_ex2.pdf": 0.6,         # 1" = 10'-0"
}
DEFAULT_PT_PER_INCH = 1.0


# ---------------------------------------------------------------------------
# Architectural constants (real-world inches)
# ---------------------------------------------------------------------------

MIN_WALL_THICKNESS_IN = 2.0    # thinnest partition wall (2x4 stud)
MAX_WALL_THICKNESS_IN = 14.0   # thickest wall (double stud / CMU)
MIN_WALL_LENGTH_IN = 4.0       # shortest meaningful wall segment
DOOR_WIDTH_IN = 18.0           # standard door opening → bridge epsilon
CONNECTIVITY_IN = 3.0          # endpoint snapping distance
MIN_PATTERN_COUNT = 5          # minimum drawings per pattern to consider


def compute_thresholds(pt_per_inch):
    """Derive scale-aware thresholds from architectural constants."""
    ppi = pt_per_inch
    return {
        "min_thickness": max(MIN_WALL_THICKNESS_IN * ppi, 0.8),
        "max_thickness": MAX_WALL_THICKNESS_IN * ppi,
        "min_length": MIN_WALL_LENGTH_IN * ppi,
        "connectivity_eps": max(CONNECTIVITY_IN * ppi, 1.5),
        "bridge_eps": DOOR_WIDTH_IN * ppi,
        # max_bridge_gap is added per-page in process_pdf
    }


# ---------------------------------------------------------------------------
# Spatial grid for O(1) proximity queries
# ---------------------------------------------------------------------------

class SpatialGrid:
    """Hash grid mapping (x, y) → indices for fast neighbour lookups."""

    def __init__(self, cell_size=6.0):
        self.cell_size = max(cell_size, 0.1)
        self.cells = defaultdict(list)

    def _cell(self, x, y):
        return (int(math.floor(x / self.cell_size)),
                int(math.floor(y / self.cell_size)))

    def insert(self, x, y, index):
        self.cells[self._cell(x, y)].append((x, y, index))

    def query(self, x, y, epsilon, exclude_index=None):
        """Return set of indices with a point within *epsilon*."""
        cx, cy = self._cell(x, y)
        r = max(1, int(math.ceil(epsilon / self.cell_size)))
        result = set()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for px, py, idx in self.cells.get((cx + dx, cy + dy), []):
                    if idx != exclude_index and math.hypot(px - x, py - y) <= epsilon:
                        result.add(idx)
        return result


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

# Global endpoint cache — avoids repeated extraction for 86K+ drawings
_endpoint_cache = {}


def _extract_endpoints(drawing, idx=None):
    """Return list of (x, y) endpoints, including Bezier start/end.

    If *idx* is provided, caches results for repeated calls.
    """
    if idx is not None and idx in _endpoint_cache:
        return _endpoint_cache[idx]
    pts = []
    for item in drawing["items"]:
        if item[0] == "l":
            pts.append((item[1].x, item[1].y))
            pts.append((item[2].x, item[2].y))
        elif item[0] == "re":
            r = item[1]
            pts.extend([(r.x0, r.y0), (r.x1, r.y0),
                        (r.x1, r.y1), (r.x0, r.y1)])
        elif item[0] == "qu":
            q = item[1]
            for p in (q.ul, q.ur, q.lr, q.ll):
                pts.append((p.x, p.y))
        elif item[0] == "c":
            pts.append((item[1].x, item[1].y))   # start
            pts.append((item[4].x, item[4].y))   # end
    if idx is not None:
        _endpoint_cache[idx] = pts
    return pts


def _extract_segments(drawing):
    """Return list of (x1, y1, x2, y2) segments, linearizing Bezier curves."""
    segs = []
    for item in drawing["items"]:
        if item[0] == "l":
            segs.append((item[1].x, item[1].y, item[2].x, item[2].y))
        elif item[0] == "re":
            r = item[1]
            segs.append((r.x0, r.y0, r.x1, r.y0))
            segs.append((r.x1, r.y0, r.x1, r.y1))
            segs.append((r.x1, r.y1, r.x0, r.y1))
            segs.append((r.x0, r.y1, r.x0, r.y0))
        elif item[0] == "qu":
            q = item[1]
            pts = [q.ul, q.ur, q.lr, q.ll]
            for i in range(4):
                p1, p2 = pts[i], pts[(i + 1) % 4]
                segs.append((p1.x, p1.y, p2.x, p2.y))
        elif item[0] == "c":
            p0, p1, p2, p3 = item[1], item[2], item[3], item[4]
            prev = p0
            for j in range(1, 5):
                t = j / 4.0
                mt = 1 - t
                x = mt**3*p0.x + 3*mt**2*t*p1.x + 3*mt*t**2*p2.x + t**3*p3.x
                y = mt**3*p0.y + 3*mt**2*t*p1.y + 3*mt*t**2*p2.y + t**3*p3.y
                segs.append((prev.x, prev.y, x, y))
                prev = fitz.Point(x, y)
    return segs


def _dominant_angle(items):
    """Return the dominant line angle (0-180) from a drawing's items list."""
    angles = []
    for item in items:
        if item[0] == "l":
            dx = item[2].x - item[1].x
            dy = item[2].y - item[1].y
            if math.hypot(dx, dy) > 0.05:
                angles.append(math.degrees(math.atan2(dy, dx)) % 180)
    if not angles:
        return None
    bins = Counter(round(a / 5) * 5 for a in angles)
    return bins.most_common(1)[0][0]


def _angle_close(a, b, tol=10):
    """Check if two angles (0-180) are within *tol* degrees."""
    diff = abs(a - b)
    return diff <= tol or (180 - diff) <= tol


def _is_dashed(drawing):
    """True if the drawing uses a dashed stroke pattern."""
    dashes = drawing.get("dashes")
    if not dashes:
        return False
    return dashes.strip() != "[] 0"


def _is_wall_shaped(rect, thresholds):
    """True if rect has wall-like proportions (long and thin)."""
    narrow = min(rect.width, rect.height)
    wide = max(rect.width, rect.height)
    return (narrow >= thresholds["min_thickness"]
            and narrow <= thresholds["max_thickness"]
            and wide >= thresholds["min_length"])


def _is_wall_fill(drawing, thresholds=None):
    """True if drawing is structurally a wall fill (not an arrowhead).

    Accepts: rectangles, quads, 4+ segment polygons, and large 3-segment
    shapes (wall junctions, L-shapes). Rejects only small 3-segment
    triangles (arrowheads on dimension lines).
    """
    items = drawing["items"]
    for item in items:
        if item[0] in ("re", "qu"):
            return True
    if len(items) >= 4:
        return True
    # 3-item fills: accept if large (wall junction), reject if small (arrow)
    if len(items) == 3:
        r = drawing["rect"]
        area = r.width * r.height
        # Arrowheads are tiny; wall junctions are substantial
        if thresholds:
            min_area = thresholds["min_thickness"] * thresholds["min_length"]
        else:
            min_area = 4.0  # fallback
        return area >= min_area
    return False


def _round_color(c):
    """Quantize an RGB tuple to 2 decimal places for grouping."""
    return (round(c[0], 2), round(c[1], 2), round(c[2], 2))


# ---------------------------------------------------------------------------
# Phase 0: Pre-processing
# ---------------------------------------------------------------------------

def prefilter_drawings(drawings):
    """Return set of valid drawing indices after discarding noise.

    Removes: dashed strokes, nearly-invisible elements, white-only fills,
    empty drawings. Semi-transparent fills (opacity 0.5-0.95) are KEPT —
    some plans use semi-transparent fills as their wall rendering style.
    """
    valid = set()
    for i, d in enumerate(drawings):
        if _is_dashed(d):
            continue
        # Only filter truly invisible elements (opacity < 0.1)
        # Semi-transparent fills (e.g., 0.75) are legitimate wall styles
        s_op = d.get("stroke_opacity")
        f_op = d.get("fill_opacity")
        has_fill = d.get("fill") is not None
        has_stroke = d.get("color") is not None
        if has_stroke and not has_fill and s_op is not None and s_op < 0.1:
            continue
        if has_fill and not has_stroke and f_op is not None and f_op < 0.1:
            continue
        if has_fill and has_stroke:
            if (s_op is not None and s_op < 0.1
                    and f_op is not None and f_op < 0.1):
                continue
        fill = d.get("fill")
        color = d.get("color")
        width = d.get("width") or 0
        if fill == (1.0, 1.0, 1.0) and (not color or width < 0.1):
            continue
        if not d.get("items"):
            continue
        valid.add(i)
    return valid


# ---------------------------------------------------------------------------
# Phase 1: Pattern enumeration
# ---------------------------------------------------------------------------

def enumerate_patterns(drawings, valid_indices, thresholds):
    """Mechanically enumerate every distinct visual pattern.

    Returns list of pattern dicts sorted by count (descending).
    Pattern types: 'fill' (by color), 'hatch' (thin angled strokes),
    'stroke' (by color + width).
    """
    patterns = []

    # --- Fill patterns: group by fill color ---
    fill_groups = defaultdict(set)
    for i in valid_indices:
        d = drawings[i]
        fill = d.get("fill")
        if fill and fill != (1.0, 1.0, 1.0):
            fill_groups[_round_color(fill)].add(i)

    for color, indices in fill_groups.items():
        wall_shaped = {
            i for i in indices
            if _is_wall_shaped(drawings[i]["rect"], thresholds)
            and _is_wall_fill(drawings[i], thresholds)
        }
        if len(wall_shaped) < MIN_PATTERN_COUNT:
            continue
        r, g, b = color
        patterns.append({
            "type": "fill",
            "fill_color": color,
            "stroke_color": None,
            "stroke_width": None,
            "indices": wall_shaped,
            "count": len(wall_shaped),
            "label": f"fill({r:.2f},{g:.2f},{b:.2f}) x{len(wall_shaped)}",
        })

    # --- Hatching: thin, non-black, angled, multi-segment strokes ---
    hatch_indices_all = set()
    hatch_groups = defaultdict(set)
    for i in valid_indices:
        d = drawings[i]
        color = d.get("color")
        width = d.get("width")
        if (color and width and 0 < width < 0.2
                and color != (0.0, 0.0, 0.0)
                and len(d["items"]) >= 3):
            key = (_round_color(color), round(width, 2))
            hatch_groups[key].add(i)

    for (color, width), indices in hatch_groups.items():
        if len(indices) < 10:
            continue
        ref_idx = max(indices, key=lambda i: len(drawings[i]["items"]))
        angle = _dominant_angle(drawings[ref_idx]["items"])
        if angle is None:
            continue
        consistent = set()
        for i in indices:
            dom = _dominant_angle(drawings[i]["items"])
            if dom is None or _angle_close(dom, angle):
                consistent.add(i)
        if len(consistent) < 10:
            continue
        hatch_indices_all.update(consistent)
        r, g, b = color
        patterns.append({
            "type": "hatch",
            "fill_color": None,
            "stroke_color": color,
            "stroke_width": width,
            "hatch_angle": angle,
            "indices": consistent,
            "count": len(consistent),
            "label": (f"hatch({r:.2f},{g:.2f},{b:.2f}) w={width} "
                      f"~{angle:.0f}deg x{len(consistent)}"),
        })

    # --- Stroke patterns: group by (color, width), excluding hatching ---
    stroke_groups = defaultdict(set)
    for i in valid_indices:
        if i in hatch_indices_all:
            continue
        d = drawings[i]
        color = d.get("color")
        width = d.get("width")
        if color and width and width >= 0.3:
            key = (_round_color(color), round(width, 1))
            stroke_groups[key].add(i)

    for (color, width), indices in stroke_groups.items():
        if len(indices) < MIN_PATTERN_COUNT:
            continue
        r, g, b = color
        patterns.append({
            "type": "stroke",
            "fill_color": None,
            "stroke_color": color,
            "stroke_width": width,
            "indices": indices,
            "count": len(indices),
            "label": f"stroke({r:.2f},{g:.2f},{b:.2f}) w={width} x{len(indices)}",
        })

    patterns.sort(key=lambda p: -p["count"])
    return patterns


# ---------------------------------------------------------------------------
# Connectivity graph utilities
# ---------------------------------------------------------------------------

def _build_connectivity_graph(drawings, indices, epsilon=3.0):
    """Build adjacency dict {index: {neighbours}} using spatial grid."""
    grid = SpatialGrid(cell_size=max(2 * epsilon, 1.0))
    adjacency = defaultdict(set)
    for idx in indices:
        adjacency[idx]  # ensure node exists
        for px, py in _extract_endpoints(drawings[idx], idx):
            neighbours = grid.query(px, py, epsilon, exclude_index=idx)
            for n_idx in neighbours:
                adjacency[idx].add(n_idx)
                adjacency[n_idx].add(idx)
            grid.insert(px, py, idx)
    return adjacency


def _connected_components(adjacency):
    """BFS connected components. Returns list of sets, largest first."""
    visited = set()
    components = []
    for node in adjacency:
        if node in visited:
            continue
        component = set()
        queue = [node]
        while queue:
            n = queue.pop()
            if n in visited:
                continue
            visited.add(n)
            component.add(n)
            queue.extend(adjacency[n] - visited)
        components.append(component)
    return sorted(components, key=len, reverse=True)


def _gap_bridge_components(drawings, all_comps, exclude_set,
                           bridge_eps, max_bridge_gap,
                           candidate_indices=None):
    """Merge disconnected components via intermediate bridging vectors.

    A non-excluded drawing whose endpoints touch two different components
    (within bridge_eps) causes those components to merge — provided their
    bounding boxes are within max_bridge_gap.

    *candidate_indices*: if provided, only scan these indices for bridges
    (instead of all drawings). Critical for plans with 86K+ drawings.
    """
    if len(all_comps) <= 1:
        return all_comps

    comp_grid = SpatialGrid(cell_size=max(2 * bridge_eps, 1.0))
    comp_bboxes = {}
    for comp_id, comp in enumerate(all_comps):
        xs, ys = [], []
        for idx in comp:
            for px, py in _extract_endpoints(drawings[idx], idx):
                comp_grid.insert(px, py, comp_id)
                xs.append(px)
                ys.append(py)
        if xs:
            comp_bboxes[comp_id] = (min(xs), min(ys), max(xs), max(ys))

    # Iterate only candidate indices (or all drawings if not specified)
    scan = candidate_indices if candidate_indices is not None else range(len(drawings))

    merges = set()
    for i in scan:
        if i in exclude_set:
            continue
        d = drawings[i]
        endpoints = _extract_endpoints(d, i)
        if not endpoints:
            continue
        d_rect = d["rect"]
        if max(d_rect.width, d_rect.height) > max_bridge_gap:
            continue
        near_comps = set()
        for px, py in endpoints:
            near_comps.update(comp_grid.query(px, py, bridge_eps))
        if len(near_comps) >= 2:
            comp_list = sorted(near_comps)
            for a in range(len(comp_list)):
                for b in range(a + 1, len(comp_list)):
                    ca, cb = comp_list[a], comp_list[b]
                    if ca in comp_bboxes and cb in comp_bboxes:
                        ba, bb = comp_bboxes[ca], comp_bboxes[cb]
                        dx = max(0, max(ba[0], bb[0]) - min(ba[2], bb[2]))
                        dy = max(0, max(ba[1], bb[1]) - min(ba[3], bb[3]))
                        if math.hypot(dx, dy) <= max_bridge_gap:
                            merges.add((ca, cb))

    if not merges:
        return all_comps

    parent = list(range(len(all_comps)))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in merges:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    merged_groups = defaultdict(set)
    for comp_id, comp in enumerate(all_comps):
        merged_groups[_find(comp_id)].update(comp)

    return sorted(merged_groups.values(), key=len, reverse=True)


# ---------------------------------------------------------------------------
# Phase 2: Per-pattern wall network
# ---------------------------------------------------------------------------

def build_pattern_network(drawings, pattern, thresholds, valid_indices=None):
    """Build wall network for a single pattern.

    Returns set of wall drawing indices that survived connectivity filtering
    and gap-bridging.
    """
    indices = pattern["indices"]
    eps = thresholds["connectivity_eps"]
    if pattern["type"] == "hatch":
        eps *= 1.5  # hatching rects may not share exact endpoints

    # Build connectivity graph
    adj = _build_connectivity_graph(drawings, indices, epsilon=eps)
    comps = _connected_components(adj)

    # Gap-bridge to merge components separated by doors/windows
    index_set = set(indices)
    bridge_eps = thresholds["bridge_eps"]
    max_gap = thresholds.get("max_bridge_gap", bridge_eps * 10)
    comps = _gap_bridge_components(
        drawings, comps, index_set, bridge_eps, max_gap,
        candidate_indices=valid_indices)

    # Keep components with >= 2 drawings
    wall_indices = set()
    for comp in comps:
        if len(comp) >= 2:
            wall_indices.update(comp)

    return wall_indices


# ---------------------------------------------------------------------------
# Phase 3: VLM classification
# ---------------------------------------------------------------------------

def _auto_dpi(page, max_dim=3000, base_dpi=200):
    """Return DPI that keeps largest rendered dimension under max_dim."""
    page_max = max(page.rect.width, page.rect.height)
    cap = max_dim / page_max * 72
    dpi = min(base_dpi, int(cap)) if base_dpi > cap else base_dpi
    return max(dpi, 72)


def _vlm_call(client, img_bytes, prompt, max_tokens=512):
    """Send image + prompt to OpenAI GPT-4o, return parsed JSON dict."""
    img_b64 = base64.b64encode(img_bytes).decode()
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = resp.choices[0].message.content
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"VLM did not return JSON: {text}")
    return json.loads(text[start:end + 1])


def vlm_evaluate_pattern(client, overlay_path, pattern, pt_per_inch=1.0):
    """Ask VLM whether a pattern overlay represents architectural walls.

    Returns dict with: is_walls, wall_pct, false_positive_categories, reasoning.
    """
    img_bytes = open(overlay_path, "rb").read()
    ptype = pattern["type"]
    label = pattern["label"]

    # Scale context
    inches_per_foot = 12.0
    scale_desc = f"{pt_per_inch:.2f} PDF points per real-world inch"

    # Pattern-type-specific guidance
    if ptype == "fill":
        type_context = (
            "The red elements are FILLED POLYGONS (solid colored shapes). "
            "In architectural floor plans viewed from above, walls appear as "
            "long, narrow filled rectangles — the cross-section of the wall. "
            "Do NOT confuse wall cross-sections with fixtures or appliances. "
            "Fixtures (toilets, sinks, tubs) are typically small, compact, "
            "and located INSIDE rooms. Wall fills are elongated, form "
            "connected networks, and define room boundaries."
        )
    elif ptype == "hatch":
        type_context = (
            "The red elements are HATCHING PATTERNS (closely-spaced diagonal "
            "lines). In architectural plans, hatching between parallel lines "
            "indicates wall sections — the fill pattern shows the wall material."
        )
    else:
        type_context = (
            "The red elements are STROKED LINES (outlines/edges). "
            "In floor plans, wall edges are typically solid lines that form "
            "connected room boundaries. Non-wall strokes include dimension "
            "lines, door swings (arcs), text, and annotation leaders."
        )

    prompt = (
        f"This is an architectural floor plan (scale: {scale_desc}). "
        f"Certain elements are highlighted in RED.\n\n"
        f"Pattern: '{label}'\n"
        f"{type_context}\n\n"
        "Questions:\n"
        "1. Do the red elements primarily represent architectural WALLS "
        "(structural elements defining rooms and spaces)?\n"
        "2. List categories of NON-WALL elements highlighted in red "
        "(e.g., text, dimension lines, door swings, furniture, fixtures, "
        "title block). If none, return an empty list.\n"
        "3. What percentage of red-highlighted area is actual walls?\n\n"
        "Return ONLY JSON:\n"
        '{"is_walls": true/false, '
        '"wall_pct": <0-100>, '
        '"false_positive_categories": ["...", ...], '
        '"reasoning": "<1-2 sentences>"}'
    )
    return _vlm_call(client, img_bytes, prompt, max_tokens=512)


# ---------------------------------------------------------------------------
# Phase 4: Unified graph & expansion
# ---------------------------------------------------------------------------

def build_unified_graph(drawings, merged_indices, valid_indices, thresholds):
    """Rebuild connectivity across all approved patterns, gap-bridge, expand.

    Returns final set of wall drawing indices.
    """
    if not merged_indices:
        return set()

    eps = thresholds["connectivity_eps"]
    bridge_eps = thresholds["bridge_eps"]
    max_gap = thresholds.get("max_bridge_gap", bridge_eps * 10)

    wall_set = set(merged_indices)

    # --- Step 1: Rebuild unified connectivity and gap-bridge ---
    adj = _build_connectivity_graph(drawings, wall_set, epsilon=eps)
    comps = _connected_components(adj)

    n_comps_before = len(comps)
    comps = _gap_bridge_components(
        drawings, comps, wall_set, bridge_eps, max_gap,
        candidate_indices=valid_indices)

    wall_set = set()
    for comp in comps:
        if len(comp) >= 2:
            wall_set.update(comp)

    if len(comps) < n_comps_before:
        print(f"  Unified bridging: {n_comps_before} → {len(comps)} components")

    pre_expand = len(wall_set)

    # --- Step 2: Iterative expansion (3 rounds) ---
    for round_num in range(3):
        # Build endpoint grid from current wall set
        wall_grid = SpatialGrid(cell_size=max(2 * eps, 1.0))
        for idx in wall_set:
            for px, py in _extract_endpoints(drawings[idx], idx):
                wall_grid.insert(px, py, idx)

        expand_eps = eps * 2  # slightly wider for fill expansion
        added = set()

        for i in valid_indices:
            if i in wall_set:
                continue
            d = drawings[i]
            fill = d.get("fill")
            color = d.get("color")
            width = d.get("width") or 0

            # (a) Wall-shaped fills of any color → endpoint proximity
            if (fill and fill != (1.0, 1.0, 1.0)
                    and _is_wall_shaped(d["rect"], thresholds)
                    and _is_wall_fill(d, thresholds)):
                for px, py in _extract_endpoints(d, i):
                    if wall_grid.query(px, py, expand_eps):
                        added.add(i)
                        break
                continue

            # (b) Strokes overlapping wall rects → border recovery
            if color and width >= 0.3:
                dr = d["rect"]
                pad = (-eps, -eps, eps, eps)
                for px, py in _extract_endpoints(d, i):
                    nearby = wall_grid.query(px, py, eps * 3)
                    for wi in nearby:
                        if dr.intersects(drawings[wi]["rect"] + pad):
                            added.add(i)
                            break
                    if i in added:
                        break

        if not added:
            break

        # Safety: stop if expansion is suspiciously large
        if len(added) > max(len(wall_set) * 2, 500):
            print(f"  WARNING: expansion round {round_num+1} tried to add "
                  f"{len(added)} drawings — stopping (possible FP chain)")
            break

        wall_set.update(added)

    if len(wall_set) > pre_expand:
        print(f"  Expansion: +{len(wall_set) - pre_expand} drawings "
              f"({round_num + 1} rounds)")

    # --- Step 3: Prune isolated small components (remove singletons) ---
    adj_final = _build_connectivity_graph(drawings, wall_set, epsilon=eps)
    comps_final = _connected_components(adj_final)
    wall_set = set()
    for comp in comps_final:
        if len(comp) >= 2:
            wall_set.update(comp)

    return wall_set


# ---------------------------------------------------------------------------
# Phase 5: Overlay rendering
# ---------------------------------------------------------------------------

def _draw_item(shape, item):
    """Draw a single PyMuPDF drawing item onto a Shape, handling all types."""
    if item[0] == "l":
        shape.draw_line(item[1], item[2])
    elif item[0] == "re":
        shape.draw_rect(item[1])
    elif item[0] == "qu":
        shape.draw_quad(item[1])
    elif item[0] == "c":
        # Approximate cubic Bezier with line segments
        p0, p1, p2, p3 = item[1], item[2], item[3], item[4]
        n = 8
        prev = p0
        for i in range(1, n + 1):
            t = i / n
            mt = 1 - t
            x = (mt**3 * p0.x + 3 * mt**2 * t * p1.x
                 + 3 * mt * t**2 * p2.x + t**3 * p3.x)
            y = (mt**3 * p0.y + 3 * mt**2 * t * p1.y
                 + 3 * mt * t**2 * p2.y + t**3 * p3.y)
            shape.draw_line(prev, fitz.Point(x, y))
            prev = fitz.Point(x, y)


def generate_overlay(pdf_path, drawings, wall_indices, output_path, dpi=200):
    """Render floor plan with wall drawings highlighted in red."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    dpi = _auto_dpi(page, max_dim=3000, base_dpi=dpi)

    # Separate fills from strokes for different rendering styles
    fill_drawings = []
    stroke_by_w = defaultdict(list)

    for idx in wall_indices:
        d = drawings[idx]
        if d.get("fill") and d["fill"] != (1.0, 1.0, 1.0):
            fill_drawings.append(d)
        else:
            w = d.get("width") or 0.6
            for item in d["items"]:
                stroke_by_w[w].append(item)

    # Fills: red fill + red outline
    for d in fill_drawings:
        s = page.new_shape()
        for item in d["items"]:
            _draw_item(s, item)
        s.finish(color=(1, 0, 0), fill=(1, 0, 0), width=0.5)
        s.commit()

    # Strokes: red, original width
    for w, items in stroke_by_w.items():
        s = page.new_shape()
        for item in items:
            _draw_item(s, item)
        s.finish(color=(1, 0, 0), width=w)
        s.commit()

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    pix.save(output_path)
    doc.close()
    return output_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_pdf(pdf_path, tag=None, pt_per_inch=None):
    """Run the full v3 pipeline on a single PDF.

    Outputs go to output/{tag}/ with per-pattern overlays, VLM evaluations,
    final merged overlay, and summary JSON.
    """
    if tag is None:
        tag = os.path.splitext(os.path.basename(pdf_path))[0].replace(" ", "_")

    basename = os.path.basename(pdf_path)
    if pt_per_inch is None:
        pt_per_inch = KNOWN_SCALES.get(basename, DEFAULT_PT_PER_INCH)
    thresholds = compute_thresholds(pt_per_inch)

    out_dir = os.path.join("output", tag)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"  WALL DETECTION v3 — {basename}")
    print(f"  Scale: {pt_per_inch} pt/inch")
    print("=" * 60)

    # ===== Phase 0: Load & pre-process =====
    _endpoint_cache.clear()
    print("\n[Phase 0] Loading PDF and pre-processing...")
    doc = fitz.open(pdf_path)
    page = doc[0]
    drawings = page.get_drawings()
    print(f"  Page: {page.rect}  rotation={page.rotation}")
    print(f"  Total drawings: {len(drawings)}")

    # Add page-derived threshold
    thresholds["max_bridge_gap"] = max(page.rect.width, page.rect.height) / 3

    valid = prefilter_drawings(drawings)
    print(f"  After prefilter: {len(valid)}/{len(drawings)} valid "
          f"({len(drawings) - len(valid)} removed)")

    # Log layer info if present
    layer_counts = Counter(d.get("layer", "") for d in drawings)
    non_empty = {k: v for k, v in layer_counts.items() if k}
    if non_empty:
        print(f"  Layers: {dict(non_empty)}")

    # Log type distribution
    type_counts = Counter(d.get("type", "?") for d in drawings)
    print(f"  Drawing types: {dict(type_counts)}")

    # ===== Phase 1: Enumerate patterns =====
    print("\n[Phase 1] Enumerating visual patterns...")
    patterns = enumerate_patterns(drawings, valid, thresholds)
    print(f"  Found {len(patterns)} candidate patterns:")
    for i, p in enumerate(patterns[:12]):
        print(f"    {i+1}. [{p['type']:6s}] {p['label']}")

    if not patterns:
        print("  No patterns found. Cannot continue.")
        doc.close()
        return

    # ===== Phase 2: Per-pattern wall networks =====
    print("\n[Phase 2] Building wall networks per pattern...")
    pattern_results = []  # (pattern, wall_indices, overlay_path)

    # Ensure all fill/hatch patterns are evaluated (they're fewer and more
    # likely walls) plus up to 8 stroke patterns
    MAX_STROKE_PATTERNS = 8
    eval_indices = []
    n_stroke = 0
    for i, p in enumerate(patterns):
        if p["type"] in ("fill", "hatch"):
            eval_indices.append(i)
        elif n_stroke < MAX_STROKE_PATTERNS:
            eval_indices.append(i)
            n_stroke += 1

    for i in eval_indices:
        pattern = patterns[i]
        wall_idx = build_pattern_network(drawings, pattern, thresholds, valid)
        if len(wall_idx) < 2:
            print(f"  Pattern {i+1} ({pattern['label']}): "
                  f"{len(wall_idx)} connected — skipping")
            continue

        overlay_path = os.path.join(out_dir, f"pattern_{i+1}_overlay.png")
        generate_overlay(pdf_path, drawings, wall_idx, overlay_path)
        print(f"  Pattern {i+1}: {len(wall_idx)}/{pattern['count']} "
              f"drawings → {os.path.basename(overlay_path)}")
        pattern_results.append((pattern, wall_idx, overlay_path))

    if not pattern_results:
        print("  No patterns produced viable networks.")
        doc.close()
        return

    # ===== Phase 3: VLM classification =====
    print("\n[Phase 3] VLM evaluating each pattern overlay...")
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    approved_indices = set()
    approved_labels = []

    for i, (pattern, wall_idx, overlay_path) in enumerate(pattern_results):
        try:
            eval_result = vlm_evaluate_pattern(
                client, overlay_path, pattern, pt_per_inch)
        except Exception as e:
            print(f"    VLM error for pattern {i+1}: {e}")
            eval_result = {
                "is_walls": False, "wall_pct": 0,
                "false_positive_categories": [],
                "reasoning": f"VLM error: {e}",
            }

        # Save evaluation
        eval_path = os.path.join(out_dir, f"pattern_{i+1}_eval.json")
        with open(eval_path, "w") as f:
            json.dump(eval_result, f, indent=2)

        is_walls = eval_result.get("is_walls", False)
        wall_pct = eval_result.get("wall_pct", 0)
        fp_cats = eval_result.get("false_positive_categories", [])
        reasoning = eval_result.get("reasoning", "")

        # Approval based on wall_pct — the actual signal.
        # Tiered: fills/hatch have a looser bar (more structural).
        ptype = pattern["type"]
        if ptype in ("fill", "hatch"):
            approved = is_walls and wall_pct >= 60
        else:
            approved = is_walls and wall_pct >= 80

        status = "APPROVED" if approved else "rejected"
        print(f"    Pattern {i+1}: {status} — walls={is_walls}, "
              f"pct={wall_pct}, FPs={fp_cats}")
        if reasoning:
            print(f"      reason: {reasoning[:100]}")

        if approved:
            approved_indices.update(wall_idx)
            approved_labels.append(pattern["label"])

    # Fallback: use highest wall_pct if nothing approved
    if not approved_indices:
        print("\n  No patterns approved — falling back to highest wall_pct")
        best_pct = -1
        best_i = 0
        for i in range(len(pattern_results)):
            eval_path = os.path.join(out_dir, f"pattern_{i+1}_eval.json")
            try:
                with open(eval_path) as f:
                    ev = json.load(f)
                if ev.get("wall_pct", 0) > best_pct:
                    best_pct = ev["wall_pct"]
                    best_i = i
            except Exception:
                pass
        _, wall_idx, _ = pattern_results[best_i]
        approved_indices = set(wall_idx)
        approved_labels = [pattern_results[best_i][0]["label"]]
        print(f"  Fallback: pattern {best_i+1} "
              f"({approved_labels[0]}, wall_pct={best_pct})")

    print(f"\n  Approved: {len(approved_labels)} patterns, "
          f"{len(approved_indices)} drawings before unification")

    # ===== Phase 4: Unified graph & expansion =====
    print("\n[Phase 4] Building unified graph...")
    final_indices = build_unified_graph(
        drawings, approved_indices, valid, thresholds)
    print(f"  Final wall drawings: {len(final_indices)}")

    # ===== Phase 5: Output =====
    print("\n[Phase 5] Generating final overlay...")
    final_path = os.path.join(out_dir, "final_overlay.png")
    generate_overlay(pdf_path, drawings, final_indices, final_path)
    print(f"  Saved: {final_path}")

    # Counts by type
    n_fills = sum(1 for i in final_indices
                  if drawings[i].get("fill")
                  and drawings[i]["fill"] != (1.0, 1.0, 1.0))
    n_strokes = len(final_indices) - n_fills

    # Summary JSON
    summary = {
        "pdf": basename,
        "pt_per_inch": pt_per_inch,
        "total_drawings": len(drawings),
        "valid_after_prefilter": len(valid),
        "patterns_enumerated": len(patterns),
        "patterns_evaluated": len(pattern_results),
        "patterns_approved": len(approved_labels),
        "approved_labels": approved_labels,
        "wall_drawings_before_unification": len(approved_indices),
        "wall_drawings_final": len(final_indices),
        "fill_drawings": n_fills,
        "stroke_drawings": n_strokes,
        "wall_coverage_pct": round(
            len(final_indices) / max(len(drawings), 1) * 100, 1),
        "thresholds": {k: round(v, 3) for k, v in thresholds.items()},
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print("\n" + "-" * 60)
    print("  SUMMARY")
    print("-" * 60)
    print(f"  Scale (pt/inch)        : {pt_per_inch}")
    print(f"  Total drawings         : {len(drawings)}")
    print(f"  Valid (after prefilter) : {len(valid)}")
    print(f"  Patterns enumerated    : {len(patterns)}")
    print(f"  Patterns approved      : {len(approved_labels)}")
    for label in approved_labels:
        print(f"    - {label}")
    print(f"  Wall drawings (final)  : {len(final_indices)}")
    print(f"    Fills                : {n_fills}")
    print(f"    Strokes              : {n_strokes}")
    print(f"  Coverage               : "
          f"{len(final_indices) / max(len(drawings), 1) * 100:.1f}%")

    doc.close()
    print("\nDone!\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_PDFS = [
    "./example_plans/352 AA copy 2.pdf",
    "./example_plans/second_floor_352.pdf",
    "./example_plans/main_st_ex.pdf",
    "./example_plans/main_st_ex2.pdf",
    "./example_plans/custom_floor_plan.pdf",
]


def main():
    load_env()
    pdfs = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_PDFS
    for pdf_path in pdfs:
        if not os.path.exists(pdf_path):
            print(f"Skipping {pdf_path} (not found)")
            continue
        try:
            process_pdf(pdf_path)
        except Exception as e:
            import traceback
            print(f"\n  ERROR processing {pdf_path}: {e}")
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
