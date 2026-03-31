#!/usr/bin/env python3
"""
Wall Detection Pipeline for Architectural Floor Plans
======================================================

Strategy:
  1. Extract all vector paths from the PDF via PyMuPDF.
  2. Use a single VLM call (Claude) to identify ONE wall segment (seed).
  3. Extract a generalised "fingerprint" of wall vectors from that seed region
     using stroke-class ranking (color + width) with contained-endpoint
     filtering — works regardless of whether walls are hatched, solid-filled,
     or outline-only.
  4. Match the fingerprint across every vector in the document using
     connectivity-based filtering (spatial graph, connected components).
  5. Render an overlay image highlighting all detected walls.

The VLM is used exactly once to bootstrap the process.  Everything after
that is deterministic geometric matching.

Usage:
    python wall_pipeline.py [path_to_pdf]
"""

import base64
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import anthropic
import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env():
    """Load key=value pairs from .env beside this script."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def render_page(page, dpi=150):
    """Render a page to PNG bytes.  Returns (png_bytes, width_px, height_px)."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png"), pix.width, pix.height


# ---------------------------------------------------------------------------
# Spatial grid for efficient proximity queries
# ---------------------------------------------------------------------------

class SpatialGrid:
    """Hash grid mapping (x, y) → drawing indices for O(1) neighbour lookups."""

    def __init__(self, cell_size=6.0):
        self.cell_size = cell_size
        self.cells = defaultdict(list)

    def _cell(self, x, y):
        return (int(math.floor(x / self.cell_size)),
                int(math.floor(y / self.cell_size)))

    def insert(self, x, y, index):
        self.cells[self._cell(x, y)].append((x, y, index))

    def query(self, x, y, epsilon, exclude_index=None):
        """Return set of drawing indices with a point within *epsilon*."""
        cx, cy = self._cell(x, y)
        result = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for px, py, idx in self.cells.get((cx + dx, cy + dy), []):
                    if idx != exclude_index and math.hypot(px - x, py - y) <= epsilon:
                        result.add(idx)
        return result


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _extract_segments(drawing):
    """Return list of (x1, y1, x2, y2) line segments from a drawing."""
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
    return segs


def _extract_endpoints(drawing):
    """Return list of (x, y) endpoints from a drawing."""
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
    return pts


def _classify_segment(x1, y1, x2, y2, angle_tol=5.0):
    """Classify a segment as 'H', 'V', or 'D' (diagonal)."""
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 0.05:
        return None
    angle = abs(math.degrees(math.atan2(dy, dx))) % 180
    if angle <= angle_tol or angle >= 180 - angle_tol:
        return "H"
    if abs(angle - 90) <= angle_tol:
        return "V"
    return "D"


def _is_contained(x1, y1, x2, y2, rect):
    """True if BOTH endpoints lie inside *rect*."""
    return (rect.x0 <= x1 <= rect.x1 and rect.y0 <= y1 <= rect.y1 and
            rect.x0 <= x2 <= rect.x1 and rect.y0 <= y2 <= rect.y1)


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


# ---------------------------------------------------------------------------
# Scale-aware thresholds
# ---------------------------------------------------------------------------

KNOWN_SCALES = {
    "352 AA copy 2.pdf": 0.75,
    "second_floor_352.pdf": 0.75,
    "custom_floor_plan.pdf": 1.5,
    "main_st_ex.pdf": 0.6,
    "main_st_ex2.pdf": 0.6,
}
DEFAULT_PT_PER_INCH = 1.0


def compute_thresholds(pt_per_inch):
    ppi = pt_per_inch
    return {
        "min_thickness": max(2.0 * ppi, 0.8),
        "max_thickness": 14.0 * ppi,
        "min_length": 4.0 * ppi,
        "min_aspect": 1.0,
        "large_fill_min": 12.0 * ppi,
        "fill_eps": 6.0 * ppi,
        "stroke_eps": max(2.0 * ppi, 1.5),
        "hatch_cluster_eps": 4.0 * ppi,
        "bridge_eps": 18.0 * ppi,
        "max_bridge_gap": 200.0 * ppi,
        "fill_expand_eps": 12.0 * ppi,
        "mc_expand_eps": 8.0 * ppi,
        "hatch_pad": 2.0 * ppi,
        "border_intersect_pad": max(1.0 * ppi, 0.5),
        "seed_expand": max(36.0 * ppi, 20),
        "snap_search": max(18.0 * ppi, 15),
    }


# ---------------------------------------------------------------------------
# Step 2 — VLM seed locator (enhanced)
# ---------------------------------------------------------------------------

GRID_ROWS = 3
GRID_COLS = 3


def _render_grid_image(page, dpi):
    """Render the page with a labelled grid overlay.

    Returns (png_bytes, w_px, h_px, cell_labels) where cell_labels maps
    e.g. "B2" -> fitz.Rect in PDF-point space.
    """
    doc_tmp = fitz.open()  # blank doc
    doc_tmp.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
    gpage = doc_tmp[0]

    pw, ph = gpage.rect.width, gpage.rect.height
    cw, ch = pw / GRID_COLS, ph / GRID_ROWS
    cell_labels = {}

    # Draw grid lines
    shape = gpage.new_shape()
    for c in range(1, GRID_COLS):
        x = c * cw
        shape.draw_line(fitz.Point(x, 0), fitz.Point(x, ph))
    for r in range(1, GRID_ROWS):
        y = r * ch
        shape.draw_line(fitz.Point(0, y), fitz.Point(pw, y))
    shape.finish(color=(1, 0, 0), width=1.5)
    shape.commit()

    # Label each cell
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            label = f"{chr(65 + r)}{c + 1}"
            cell_rect = fitz.Rect(c * cw, r * ch, (c + 1) * cw, (r + 1) * ch)
            cell_labels[label] = cell_rect
            # Place label in upper-left of cell
            lx = cell_rect.x0 + 5
            ly = cell_rect.y0 + 18
            fontsize = max(10, min(24, int(min(cw, ch) / 10)))
            gpage.insert_text(fitz.Point(lx, ly), label,
                              fontsize=fontsize, color=(1, 0, 0))

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = gpage.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc_tmp.close()
    return img_bytes, pix.width, pix.height, cell_labels


def _auto_dpi(page, max_dim=3000, base_dpi=150):
    """Return a DPI that keeps the largest rendered dimension under *max_dim*."""
    page_max = max(page.rect.width, page.rect.height)
    cap = max_dim / page_max * 72
    dpi = min(base_dpi, int(cap)) if base_dpi > cap else base_dpi
    return max(dpi, 72)


def _vlm_call(client, img_bytes, prompt, max_tokens=512):
    """Send one image + prompt to Claude and return the parsed JSON dict."""
    img_b64 = base64.b64encode(img_bytes).decode()
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = resp.content[0].text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"VLM did not return JSON: {text}")
    return json.loads(text[start : end + 1])


# ---- Pass 1: grid localization -------------------------------------------

def _vlm_grid_pass(page, dpi):
    """Ask the VLM which grid cell contains interior walls.

    Returns (cell_rect, cell_label).
    """
    dpi = _auto_dpi(page, max_dim=3000, base_dpi=dpi)
    if dpi < 150:
        print(f"  (large page — grid render at {dpi} DPI)")

    img_bytes, w_px, h_px, cell_labels = _render_grid_image(page, dpi)
    labels_list = ", ".join(sorted(cell_labels.keys()))

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = (
        f"This is an architectural floor plan ({w_px}x{h_px} pixels) with a "
        f"red {GRID_ROWS}x{GRID_COLS} grid overlay. "
        f"The cells are labelled: {labels_list}.\n\n"
        "Which grid cell contains clearly visible INTERIOR wall segments "
        "that are part of a floor plan drawing? Pick a cell where you can "
        "see wall lines INSIDE a building — not the title block, not "
        "margins, not notes, not the outermost building border.\n\n"
        "Return ONLY a JSON object:\n"
        '{"cell": "<label like B2>"}'
    )

    result = _vlm_call(client, img_bytes, prompt, max_tokens=128)
    cell = result.get("cell", "").strip().upper()
    if cell not in cell_labels:
        raise ValueError(
            f"VLM returned invalid cell '{cell}'. Valid: {labels_list}")

    print(f"  Pass 1 — VLM picked cell {cell}")
    return cell_labels[cell], cell


# ---- Pass 2: zoomed tight wall selection ---------------------------------

def _vlm_zoom_pass(page, cell_rect):
    """Crop to *cell_rect*, render at high DPI, ask the VLM for 1-3 tight
    wall bounding boxes.

    Returns list of (seed_rect_in_page_coords, vlm_hints) tuples.
    """
    # Compute DPI so the crop's max dimension is ~1500 px
    crop_w = cell_rect.width
    crop_h = cell_rect.height
    crop_dpi = int(1500 / max(crop_w, crop_h) * 72)
    crop_dpi = max(crop_dpi, 100)

    mat = fitz.Matrix(crop_dpi / 72, crop_dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=cell_rect)
    img_bytes = pix.tobytes("png")
    w_px, h_px = pix.width, pix.height
    print(f"  Pass 2 — zoomed crop {w_px}x{h_px}px at {crop_dpi} DPI")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = (
        f"This is a zoomed-in section of an architectural floor plan "
        f"({w_px}x{h_px} pixels).\n\n"
        "Identify 1 to 3 individual wall segments in this image. "
        "For each wall:\n"
        "- Draw a TIGHT bounding box where the wall fills most of the box\n"
        "- Each box should contain ONE wall segment (a single straight "
        "section of wall), not a whole room or multiple walls\n"
        "- Pick walls that are clearly visible, straight, and at least "
        "50 pixels long\n"
        "- Pick walls away from doors and windows\n"
        "- x1 < x2 and y1 < y2, with some padding (5-15px per side)\n\n"
        "Return ONLY a JSON object:\n"
        '{"walls": [\n'
        '  {"x1": ..., "y1": ..., "x2": ..., "y2": ..., '
        '"style": "<e.g. two parallel black lines with diagonal hatching, '
        'solid black rectangle, thick black line>"},\n'
        "  ...\n"
        "]}"
    )

    result = _vlm_call(client, img_bytes, prompt, max_tokens=512)
    walls = result.get("walls", [])
    if not walls:
        raise ValueError("VLM returned no wall boxes in zoom pass")

    # Convert crop-pixel coords → page PDF-point coords
    s = 72.0 / crop_dpi
    seeds = []
    for w in walls:
        x1 = max(0, min(w["x1"], w_px))
        y1 = max(0, min(w["y1"], h_px))
        x2 = max(0, min(w["x2"], w_px))
        y2 = max(0, min(w["y2"], h_px))
        page_rect = fitz.Rect(
            cell_rect.x0 + x1 * s,
            cell_rect.y0 + y1 * s,
            cell_rect.x0 + x2 * s,
            cell_rect.y0 + y2 * s,
        )
        hints = {"style": w.get("style", "")}
        seeds.append((page_rect, hints))
        print(f"    wall: {page_rect}  style={hints['style'][:60]}")

    return seeds


# ---- Seed refinement: snap VLM boxes to actual wall vectors --------------

def _snap_one_seed(seed_rect, drawings):
    """Snap a single VLM seed rect to nearby wall-like vector drawings.

    The VLM returns approximate pixel coordinates for wall bounding boxes.
    This finds the actual PDF vector elements near that estimate, filtered
    by the seed's orientation, and returns a refined bounding box.

    For oriented seeds (tall-and-narrow or wide-and-short), the cross-axis
    position is snapped to actual vectors while the along-axis extent from
    the VLM is preserved — the VLM is typically accurate about *how long*
    the wall is but imprecise about *exactly where* it sits.
    """
    SNAP = 30  # PDF-point search expansion beyond seed edges
    search = seed_rect + (-SNAP, -SNAP, SNAP, SNAP)

    # Determine seed orientation
    w, h = seed_rect.width, seed_rect.height
    cx = (seed_rect.x0 + seed_rect.x1) / 2
    cy = (seed_rect.y0 + seed_rect.y1) / 2
    if h > 2 * w:
        orient = "V"
    elif w > 2 * h:
        orient = "H"
    else:
        orient = None

    # Collect wall-like segments with their orientation
    seg_info = []  # (fitz.Rect, segment_orient)
    for d in drawings:
        if not d["rect"].intersects(search):
            continue
        c = d.get("color")
        dw = d.get("width") or 0
        f = d.get("fill")

        if c and dw >= 0.3:
            for item in d["items"]:
                if item[0] == "l":
                    p1, p2 = item[1], item[2]
                    if math.hypot(p2.x - p1.x, p2.y - p1.y) < 3:
                        continue
                    so = _classify_segment(p1.x, p1.y, p2.x, p2.y)
                    r = fitz.Rect(min(p1.x, p2.x), min(p1.y, p2.y),
                                  max(p1.x, p2.x), max(p1.y, p2.y))
                    if r.intersects(search):
                        seg_info.append((r, so))
                elif item[0] == "re":
                    r = fitz.Rect(item[1])
                    if r.intersects(search):
                        ro = "V" if r.height > 2 * r.width else (
                             "H" if r.width > 2 * r.height else None)
                        seg_info.append((r, ro))
        elif f == (0.0, 0.0, 0.0) and c is None:
            r = fitz.Rect(d["rect"])
            if r.intersects(search):
                ro = "V" if r.height > 2 * r.width else (
                     "H" if r.width > 2 * r.height else None)
                seg_info.append((r, ro))

    if not seg_info:
        return seed_rect

    # Filter: keep segments matching the seed orientation AND near its axis
    filtered = []
    for r, so in seg_info:
        rcx = (r.x0 + r.x1) / 2
        rcy = (r.y0 + r.y1) / 2
        if orient == "V":
            if so not in ("V", None):
                continue
            if abs(rcx - cx) > SNAP:
                continue
        elif orient == "H":
            if so not in ("H", None):
                continue
            if abs(rcy - cy) > SNAP:
                continue
        filtered.append(r)

    if not filtered:
        # Fallback: nearest segments regardless of orientation
        seg_info.sort(key=lambda x: math.hypot(
            (x[0].x0 + x[0].x1) / 2 - cx,
            (x[0].y0 + x[0].y1) / 2 - cy))
        filtered = [r for r, _ in seg_info[:5]]

    # Compute bounding box of matched segments, clipped to search area
    result = fitz.Rect(filtered[0])
    for r in filtered[1:]:
        result |= r
    result &= search

    if result.is_empty or result.width < 0.5 or result.height < 0.5:
        return seed_rect

    # For oriented seeds: snap the cross-axis to vectors, preserve VLM extent
    if orient == "V":
        result = fitz.Rect(
            result.x0 - 2,
            min(result.y0, seed_rect.y0) - 2,
            result.x1 + 2,
            max(result.y1, seed_rect.y1) + 2,
        )
    elif orient == "H":
        result = fitz.Rect(
            min(result.x0, seed_rect.x0) - 2,
            result.y0 - 2,
            max(result.x1, seed_rect.x1) + 2,
            result.y1 + 2,
        )
    else:
        result += (-2, -2, 2, 2)

    return result


def _refine_seeds(seeds, drawings):
    """Snap all VLM seed rects to nearby wall vectors for precision."""
    refined = []
    for seed_rect, hints in seeds:
        snapped = _snap_one_seed(seed_rect, drawings)
        if snapped != seed_rect:
            print(f"    snapped: {seed_rect}")
            print(f"         -> {snapped}")
        refined.append((snapped, hints))
    return refined


# ---- Public API ----------------------------------------------------------

def vlm_identify_wall(page, dpi=150):
    """Two-pass VLM wall locator.

    Pass 1: grid-based coarse localization (which cell has walls?).
    Pass 2: zoomed crop with tight bbox selection (1-3 individual walls).

    Returns list of (seed_rect, vlm_hints) tuples.
    """
    cell_rect, cell_label = _vlm_grid_pass(page, dpi)
    seeds = _vlm_zoom_pass(page, cell_rect)
    # Attach the cell label to each hint for debug overlay
    for _, hints in seeds:
        hints["cell"] = cell_label
    return seeds


def save_debug_vlm_seed(pdf_path, seeds, dpi=150,
                        output_path="debug_vlm_seed.png"):
    """Render the page with each seed box in green and the grid cell in gray."""
    doc = fitz.open(pdf_path)
    page = doc[0]
    dpi = _auto_dpi(page, max_dim=3000, base_dpi=dpi)

    pw, ph = page.rect.width, page.rect.height
    cw, ch = pw / GRID_COLS, ph / GRID_ROWS

    # Draw light grid
    shape = page.new_shape()
    for c in range(1, GRID_COLS):
        shape.draw_line(fitz.Point(c * cw, 0), fitz.Point(c * cw, ph))
    for r in range(1, GRID_ROWS):
        shape.draw_line(fitz.Point(0, r * ch), fitz.Point(pw, r * ch))
    shape.finish(color=(0.8, 0.8, 0.8), width=0.3)
    shape.commit()

    # Draw each seed rect in green
    for i, (rect, hints) in enumerate(seeds):
        s = page.new_shape()
        s.draw_rect(rect)
        s.finish(color=(0, 0.8, 0), width=2.0)
        s.commit()
        # Number label
        page.insert_text(
            fitz.Point(rect.x0 + 2, rect.y0 + 10),
            str(i + 1), fontsize=8, color=(0, 0.8, 0))

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    pix.save(output_path)
    doc.close()
    return output_path


# ---------------------------------------------------------------------------
# Step 3 — Fingerprint extraction (generalised)
# ---------------------------------------------------------------------------

def _is_wall_shaped(rect, thresholds=None):
    """True if *rect* has wall-like proportions (long and thin)."""
    if thresholds is None:
        thresholds = compute_thresholds(DEFAULT_PT_PER_INCH)
    min_thickness = thresholds["min_thickness"]
    max_thickness = thresholds["max_thickness"]
    min_length = thresholds["min_length"]
    min_aspect = thresholds["min_aspect"]
    narrow = min(rect.width, rect.height)
    wide = max(rect.width, rect.height)
    if narrow < min_thickness or narrow > max_thickness:
        return False
    if wide < min_length:
        return False
    if wide / max(narrow, 0.1) < min_aspect:
        return False
    return True


def _is_wall_fill(drawing, thresholds=None):
    """True if *drawing* is structurally a wall fill (not an arrow/symbol).

    Wall fills are rectangles (``re``/``qu`` items) or closed polygonal
    paths (4+ line items).  Arrow fills are triangular (exactly 3 ``l``
    items).  This check prevents arrowheads from entering the wall
    candidate set.
    """
    if thresholds is None:
        thresholds = compute_thresholds(DEFAULT_PT_PER_INCH)
    large_fill_min = thresholds["large_fill_min"]
    # Large fills are always wall candidates regardless of shape
    r = drawing["rect"]
    if max(r.width, r.height) >= large_fill_min:
        return True
    # Small fills: accept rectangles, quads, or 4+ segment closed paths
    items = drawing["items"]
    for item in items:
        if item[0] in ("re", "qu"):
            return True
    # 4+ line segments = rectangular/polygonal wall junction
    # 3 line segments = triangular arrow
    if len(items) >= 4:
        return True
    return False


def extract_fingerprint(drawings, seed_rect, vlm_hints=None, thresholds=None):
    """Derive the wall vector signature from drawings inside *seed_rect*.

    Tries three strategies in order of signal distinctiveness:
      A. Hatched walls — detect hatch pattern, find adjacent border lines.
      B. Filled walls  — wall-shaped filled rectangles (with or without stroke).
      C. Stroked walls — stroke-class ranking (fallback for thick-stroke plans).
    """
    if thresholds is None:
        thresholds = compute_thresholds(DEFAULT_PT_PER_INCH)
    EXPAND = thresholds["seed_expand"]
    expanded = seed_rect + (-EXPAND, -EXPAND, EXPAND, EXPAND)
    region = [d for d in drawings if d["rect"].intersects(expanded)]
    print(f"  {len(region)} drawings in seed region (expansion={EXPAND}pt)")

    # --- Strategy A: detect hatching ---
    hatch = None
    hatch_candidates = [
        d for d in region
        if d.get("color")
        and d["color"] != (0.0, 0.0, 0.0)
        and len(d["items"]) >= 3
        and 0 < (d.get("width") or 0) < 0.2
    ]
    if hatch_candidates:
        ref = max(hatch_candidates, key=lambda d: len(d["items"]))
        angle = _dominant_angle(ref["items"])
        hatch = {
            "color": ref["color"],
            "width": ref["width"],
            "angle": angle if angle is not None else 45,
        }

    if hatch:
        print("  Strategy A: hatching detected — finding adjacent borders")
        hatch_rects = [d["rect"] for d in hatch_candidates]
        border_ws = Counter()
        for d in region:
            c = d.get("color")
            w = d.get("width") or 0
            if c != (0.0, 0.0, 0.0) or w <= 0.3:
                continue
            for hr in hatch_rects:
                if d["rect"].intersects(hr + (-5, -5, 5, 5)):
                    border_ws[round(w, 1)] += 1
                    break

        if not border_ws:
            raise ValueError(
                "Hatching found but no adjacent border lines. "
                "The VLM may have pointed to a non-wall area."
            )

        edge_width = border_ws.most_common(1)[0][0]
        edge_color = (0.0, 0.0, 0.0)
        all_border_widths = sorted(border_ws.keys())
        secondary_widths = [w for w in all_border_widths if w != edge_width]
        wall_style = "hatched"
        fill_color = None
        print(f"  Border widths near hatch: {dict(border_ws)}")

        return {
            "edge_color": edge_color,
            "edge_width": edge_width,
            "secondary_edge_widths": secondary_widths,
            "fill_color": fill_color,
            "hatch": hatch,
            "wall_style": wall_style,
        }

    # --- Strategy B: filled wall rectangles ---
    wall_fills = [
        d for d in region
        if d.get("fill") and d["fill"] != (1.0, 1.0, 1.0)
        and _is_wall_shaped(d["rect"], thresholds)
    ]

    if len(wall_fills) >= 2:
        # Determine the dominant fill color
        fill_colors = Counter(d["fill"] for d in wall_fills)
        fill_color = fill_colors.most_common(1)[0][0]
        count = fill_colors[fill_color]
        print(f"  Strategy B: {count} wall-shaped filled rects "
              f"(fill={fill_color})")

        return {
            "edge_color": None,
            "edge_width": None,
            "secondary_edge_widths": [],
            "fill_color": fill_color,
            "hatch": None,
            "wall_style": "solid_fill",
        }

    # --- Strategy C: stroke-class ranking (fallback) ---
    print("  Strategy C: stroke-class ranking (width >= 0.3)")
    class_lengths = defaultdict(float)

    for d in region:
        c = d.get("color")
        w = d.get("width")
        if not c or not w or w < 0.3:
            continue
        sc_key = (round(c[0], 2), round(c[1], 2), round(c[2], 2)), round(w, 1)
        for seg in _extract_segments(d):
            x1, y1, x2, y2 = seg
            if not _is_contained(x1, y1, x2, y2, expanded):
                continue
            seg_len = math.hypot(x2 - x1, y2 - y1)
            if seg_len < 5.0:
                continue
            orient = _classify_segment(x1, y1, x2, y2)
            if orient in ("H", "V"):
                class_lengths[sc_key] += seg_len

    if class_lengths:
        ranked = sorted(class_lengths.items(), key=lambda x: -x[1])
        (edge_color, edge_width), top_length = ranked[0]

        print(f"  Top stroke class: color={edge_color}, width={edge_width}, "
              f"length={top_length:.1f}pt")
        if len(ranked) > 1:
            print(f"  Runner-up classes:")
            for (sc, sw), ln in ranked[1:4]:
                print(f"    color={sc}, width={sw}, length={ln:.1f}pt")

        secondary_widths = [
            sc_width for (sc_color, sc_width), length in ranked[1:]
            if sc_color == edge_color and length > top_length * 0.3
        ]

        return {
            "edge_color": edge_color,
            "edge_width": edge_width,
            "secondary_edge_widths": secondary_widths,
            "fill_color": None,
            "hatch": None,
            "wall_style": "thick_stroke" if edge_width > 2.0 else "outline",
        }

    raise ValueError(
        "No structural lines or filled walls found in seed region. "
        "The VLM may have pointed to a non-wall area."
    )


# ---------------------------------------------------------------------------
# Step 4 — Connectivity-based global matching
# ---------------------------------------------------------------------------

def _build_connectivity_graph(drawings, indices, epsilon=3.0):
    """Build adjacency dict {index: {neighbours}} using a spatial grid.

    Two drawings are connected if any of their endpoints are within
    *epsilon* PDF points of each other.
    """
    grid = SpatialGrid(cell_size=max(2 * epsilon, 1.0))
    adjacency = defaultdict(set)

    for idx in indices:
        adjacency[idx]  # ensure node exists even if isolated
        for px, py in _extract_endpoints(drawings[idx]):
            neighbours = grid.query(px, py, epsilon, exclude_index=idx)
            for n_idx in neighbours:
                adjacency[idx].add(n_idx)
                adjacency[n_idx].add(idx)
            grid.insert(px, py, idx)

    return adjacency


def _connected_components(adjacency):
    """BFS connected components.  Returns list of sets, largest first."""
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
                           bridge_eps=15.0, max_bridge_gap=150.0):
    """Merge disconnected components when an intermediate vector bridges them.

    For each non-excluded drawing, check if its endpoints are near (within
    *bridge_eps*) two different components whose bounding boxes are within
    *max_bridge_gap*.  Returns the merged component list (sorted by size,
    largest first).
    """
    if len(all_comps) <= 1:
        return all_comps

    comp_grid = SpatialGrid(cell_size=max(2 * bridge_eps, 1.0))
    comp_bboxes = {}
    for comp_id, comp in enumerate(all_comps):
        xs, ys = [], []
        for idx in comp:
            for px, py in _extract_endpoints(drawings[idx]):
                comp_grid.insert(px, py, comp_id)
                xs.append(px)
                ys.append(py)
        if xs:
            comp_bboxes[comp_id] = (min(xs), min(ys),
                                    max(xs), max(ys))

    merges = set()
    for i, d in enumerate(drawings):
        if i in exclude_set:
            continue
        endpoints = _extract_endpoints(d)
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
                        ba = comp_bboxes[ca]
                        bb = comp_bboxes[cb]
                        dx = max(0, max(ba[0], bb[0])
                                 - min(ba[2], bb[2]))
                        dy = max(0, max(ba[1], bb[1])
                                 - min(ba[3], bb[3]))
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

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in merges:
        _union(a, b)

    merged_groups = defaultdict(set)
    for comp_id, comp in enumerate(all_comps):
        merged_groups[_find(comp_id)].update(comp)

    return sorted(merged_groups.values(), key=len, reverse=True)


def auto_detect_fingerprints(drawings, thresholds=None):
    """Detect ALL potential wall fingerprints from the drawing set.
    Returns list of fingerprint dicts ranked by score."""
    if thresholds is None:
        thresholds = compute_thresholds(DEFAULT_PT_PER_INCH)
    fps = []

    # Strategy A: hatching
    hatch_candidates = [d for d in drawings
        if d.get("color") and d["color"] != (0.0, 0.0, 0.0)
        and len(d["items"]) >= 3 and 0 < (d.get("width") or 0) < 0.2]
    if len(hatch_candidates) >= 10:
        ref = max(hatch_candidates, key=lambda d: len(d["items"]))
        angle = _dominant_angle(ref["items"])
        hatch = {"color": ref["color"], "width": ref["width"],
                 "angle": angle if angle is not None else 45}
        hatch_rects = [d["rect"] for d in hatch_candidates]
        border_ws = Counter()
        for d in drawings:
            c = d.get("color"); w = d.get("width") or 0
            if c != (0.0, 0.0, 0.0) or w <= 0.3: continue
            for hr in hatch_rects[:50]:
                if d["rect"].intersects(hr + (-5, -5, 5, 5)):
                    border_ws[round(w, 1)] += 1; break
        if border_ws:
            ew = border_ws.most_common(1)[0][0]
            sec = [w for w in sorted(border_ws.keys()) if w != ew]
            fps.append({"edge_color": (0,0,0), "edge_width": ew,
                "secondary_edge_widths": sec, "fill_color": None,
                "hatch": hatch, "wall_style": "hatched",
                "_score": len(hatch_candidates),
                "_label": f"hatched (border w={ew})"})

    # Strategy B: fills by color
    wall_fills = [d for d in drawings
        if d.get("fill") and d["fill"] != (1.0, 1.0, 1.0)
        and _is_wall_shaped(d["rect"], thresholds) and _is_wall_fill(d, thresholds)]
    if wall_fills:
        color_groups = defaultdict(list)
        for d in wall_fills:
            color_groups[d["fill"]].append(d)
        for color, fills in sorted(color_groups.items(), key=lambda x: -len(x[1])):
            if len(fills) < 3: continue
            r,g,b = color
            fps.append({"edge_color": None, "edge_width": None,
                "secondary_edge_widths": [], "fill_color": color,
                "hatch": None, "wall_style": "solid_fill",
                "_score": len(fills),
                "_label": f"fill rgb({r:.2f},{g:.2f},{b:.2f}) x{len(fills)}"})

    fps.sort(key=lambda f: -f.get("_score", 0))
    return fps


def find_all_walls(drawings, fp, thresholds=None):
    """Match the fingerprint across all drawings using connectivity filtering.

    Returns dict with keys:
        edges     — list of (index, drawing) for wall edge/border lines
        hatches   — list of (index, drawing) for hatching patterns
        fills     — list of (index, drawing) for filled wall rects
        secondary_indices — set of indices from secondary-width discovery
        components — list of significant connected components (sets of indices)
    """
    if thresholds is None:
        thresholds = compute_thresholds(DEFAULT_PT_PER_INCH)

    ec = fp["edge_color"]
    ew = fp["edge_width"]
    wall_indices = set()
    secondary_indices = set()

    if ec is not None and fp.get("hatch"):
        # ============================================================
        # PATH 1a: Hatching-first (hatched walls)
        # ============================================================
        # Hatching is a 100%-precise signal: detect it first, cluster
        # into wall regions, then find border strokes at the edges.
        hc = fp["hatch"]["color"]
        hw = fp["hatch"]["width"]
        ha = fp["hatch"]["angle"]

        # --- Step 1: find all hatching drawings ---
        hatch_indices = []
        for i, d in enumerate(drawings):
            c, w = d.get("color"), d.get("width")
            if not c or not w:
                continue
            if not all(abs(a - b) < 0.05 for a, b in zip(c, hc)):
                continue
            if abs(w - hw) > 0.02:
                continue
            if len(d["items"]) < 1:
                continue
            dom = _dominant_angle(d["items"])
            if dom is not None and not _angle_close(dom, ha):
                continue
            hatch_indices.append(i)

        print(f"  Step 1 (hatch): {len(hatch_indices)} hatching drawings")

        # --- Step 2: cluster hatching into wall regions ---
        hatch_adj = _build_connectivity_graph(
            drawings, hatch_indices, epsilon=thresholds["hatch_cluster_eps"])
        hatch_comps = _connected_components(hatch_adj)
        # Keep clusters >= 2 hatching drawings (single stray hatch = noise)
        hatch_regions = [c for c in hatch_comps if len(c) >= 2]

        print(f"  Step 2 (cluster): {len(hatch_comps)} clusters, "
              f"{len(hatch_regions)} wall regions "
              f"({sum(len(c) for c in hatch_regions)} drawings)")

        # Build SpatialGrid indexed by hatching drawing index for
        # individual hatching rect intersection checks.
        HATCH_PAD = thresholds["hatch_pad"]
        hatch_grid = SpatialGrid(cell_size=max(2 * HATCH_PAD + 10, 6.0))
        all_hatch_set = set()
        for comp in hatch_regions:
            for hi in comp:
                all_hatch_set.add(hi)
                hr = drawings[hi]["rect"]
                cx = (hr.x0 + hr.x1) / 2
                cy = (hr.y0 + hr.y1) / 2
                hatch_grid.insert(cx, cy, hi)

        # --- Step 3: find border strokes that overlap hatching regions ---
        # Use rect intersection against individual hatching drawing rects
        # (not bounding boxes) to reject dimension lines, text, and window
        # strokes that merely terminate near a wall but don't overlap its
        # hatching.
        pad = (-HATCH_PAD, -HATCH_PAD, HATCH_PAD, HATCH_PAD)
        for i, d in enumerate(drawings):
            c, w = d.get("color"), d.get("width")
            if not c or not w:
                continue
            if not all(abs(a - b) < 0.05 for a, b in zip(c, ec)):
                continue
            if abs(w - ew) > 0.15:
                continue
            dr = d["rect"]
            dr_cx = (dr.x0 + dr.x1) / 2
            dr_cy = (dr.y0 + dr.y1) / 2
            search_r = max(dr.width, dr.height) / 2 + HATCH_PAD + 10
            nearby = hatch_grid.query(dr_cx, dr_cy, search_r)
            for hi in nearby:
                if dr.intersects(drawings[hi]["rect"] + pad):
                    wall_indices.add(i)
                    break

        print(f"  Step 3 (borders): {len(wall_indices)} border strokes "
              f"overlapping hatching")

        # --- Step 4: secondary width strokes overlapping hatching ---
        # Only accept widths identified in the fingerprint, not any w>=0.3
        known_sec = fp.get("secondary_edge_widths") or []
        if known_sec:
            for i, d in enumerate(drawings):
                if i in wall_indices:
                    continue
                c, w = d.get("color"), d.get("width")
                if not c or not w:
                    continue
                if not all(abs(a - b) < 0.05 for a, b in zip(c, ec)):
                    continue
                if not any(abs(w - sw) < 0.15 for sw in known_sec):
                    continue
                dr = d["rect"]
                dr_cx = (dr.x0 + dr.x1) / 2
                dr_cy = (dr.y0 + dr.y1) / 2
                search_r = max(dr.width, dr.height) / 2 + HATCH_PAD + 10
                nearby = hatch_grid.query(dr_cx, dr_cy, search_r)
                for hi in nearby:
                    if dr.intersects(drawings[hi]["rect"] + pad):
                        wall_indices.add(i)
                        secondary_indices.add(i)
                        break

        if secondary_indices:
            sec_widths = {round(drawings[i].get("width", 0), 1)
                          for i in secondary_indices}
            print(f"  Step 4 (secondary): +{len(secondary_indices)} "
                  f"secondary-width strokes (widths: {sorted(sec_widths)})")

    elif ec is not None:
        # ============================================================
        # PATH 1b: Stroke-only matching (outline/thick_stroke, no hatch)
        # ============================================================

        # --- Phase A: stroke-class match ---
        matched = []
        for i, d in enumerate(drawings):
            c, w = d.get("color"), d.get("width")
            if not c or not w:
                continue
            if not all(abs(a - b) < 0.05 for a, b in zip(c, ec)):
                continue
            if abs(w - ew) > 0.15:
                continue
            matched.append(i)

        print(f"  Phase A: {len(matched)} drawings match edge stroke class "
              f"(color~{ec}, width~{ew})")

        # --- Phase B: connectivity graph ---
        adjacency = _build_connectivity_graph(
            drawings, matched, epsilon=thresholds["stroke_eps"])

        # --- Phase C: component filter ---
        components = _connected_components(adjacency)
        significant = [c for c in components if len(c) >= 3]
        small_comps = [c for c in components if 1 <= len(c) < 3]
        for c in significant:
            wall_indices.update(c)

        print(f"  Phase B/C: {len(components)} total components, "
              f"{len(significant)} significant (>=3 drawings), "
              f"{len(wall_indices)} wall drawings")

        # --- Phase C2: gap-bridging for strokes ---
        all_comps = significant + small_comps
        matched_set = set(matched)
        comps_merged = _gap_bridge_components(
            drawings, all_comps, matched_set,
            bridge_eps=thresholds["bridge_eps"],
            max_bridge_gap=thresholds["max_bridge_gap"])

        if len(comps_merged) < len(all_comps):
            rescued = set()
            for c in comps_merged:
                if len(c) >= 3 and not c.issubset(wall_indices):
                    rescued.update(c - wall_indices)
            if rescued:
                wall_indices.update(rescued)
            print(f"  Phase C2 bridging: {len(all_comps)} -> "
                  f"{len(comps_merged)} components "
                  f"({len(all_comps) - len(comps_merged)} bridges, "
                  f"+{len(rescued)} rescued strokes)")

        # --- Phase D: secondary width discovery ---
        if wall_indices:
            wall_grid = SpatialGrid(cell_size=6.0)
            for idx in wall_indices:
                for px, py in _extract_endpoints(drawings[idx]):
                    wall_grid.insert(px, py, idx)

            candidates = set()
            for i, d in enumerate(drawings):
                if i in wall_indices:
                    continue
                c, w = d.get("color"), d.get("width")
                if not c or not w:
                    continue
                if not all(abs(a - b) < 0.05 for a, b in zip(c, ec)):
                    continue
                if abs(w - ew) < 0.05:
                    continue
                if w < 0.3:
                    continue
                for px, py in _extract_endpoints(d):
                    if wall_grid.query(px, py, thresholds["stroke_eps"]):
                        candidates.add(i)
                        break

            if candidates:
                combined = wall_indices | candidates
                adj2 = _build_connectivity_graph(
                    drawings, combined, epsilon=thresholds["stroke_eps"])
                comps2 = _connected_components(adj2)
                for comp in comps2:
                    if comp & wall_indices:
                        secondary_indices.update(comp - wall_indices)

            if secondary_indices:
                wall_indices.update(secondary_indices)
                sec_widths = {round(drawings[i].get("width", 0), 1)
                              for i in secondary_indices}
                print(f"  Phase D: +{len(secondary_indices)} secondary-width "
                      f"drawings (widths: {sorted(sec_widths)})")

    else:
        # ============================================================
        # PATH 2: Fill-based matching (solid_fill walls)
        # ============================================================
        fc = fp["fill_color"]

        # -- Pass 1: find all wall-shaped fills -------------------------
        candidates = []
        cand_set = set()
        for i, d in enumerate(drawings):
            if (d.get("fill") == fc and _is_wall_shaped(d["rect"], thresholds)
                    and _is_wall_fill(d, thresholds)):
                candidates.append(i)
                cand_set.add(i)

        print(f"  Pass 1 (fill): {len(candidates)} wall-shaped fills")

        # Build connectivity among candidates
        adj1 = _build_connectivity_graph(
            drawings, candidates, epsilon=thresholds["fill_eps"])
        comps1 = _connected_components(adj1)
        print(f"  Pass 1 components: {[len(c) for c in comps1[:10]]}")

        # -- Pass 2: gap-bridging via intermediate vectors ---------------
        comps_final = _gap_bridge_components(
            drawings, comps1, cand_set,
            bridge_eps=thresholds["bridge_eps"],
            max_bridge_gap=thresholds["max_bridge_gap"])

        if len(comps_final) < len(comps1):
            print(f"  Pass 2 bridging: {len(comps1)} -> "
                  f"{len(comps_final)} components "
                  f"({len(comps1) - len(comps_final)} bridges)")

        # -- Pass 3: filter — keep merged components >= 2 --------------
        for c in comps_final:
            if len(c) >= 2:
                wall_indices.update(c)
        print(f"  Pass 3 filtered: {len(wall_indices)} fills in "
              f"{sum(1 for c in comps_final if len(c) >= 2)} "
              f"components (>= 2)")

        # -- Pass 4: expand to adjacent same-color fills ----------------
        # Fill expansion can safely use a wider radius because
        # fills are filtered by color + _is_wall_fill (structural shape).
        if wall_indices:
            wall_grid = SpatialGrid(
                cell_size=max(2 * thresholds["fill_eps"], 1.0))
            for idx in wall_indices:
                for px, py in _extract_endpoints(drawings[idx]):
                    wall_grid.insert(px, py, idx)

            expansion = set()
            for i, d in enumerate(drawings):
                if i in wall_indices:
                    continue
                if d.get("fill") != fc:
                    continue
                if not _is_wall_fill(d, thresholds):
                    continue
                for px, py in _extract_endpoints(d):
                    if wall_grid.query(px, py, thresholds["fill_expand_eps"]):
                        expansion.add(i)
                        break

            if expansion:
                wall_indices.update(expansion)
                print(f"  Pass 4 expanded: +{len(expansion)} adjacent fills")

        # -- Pass 5: multi-color expansion (iterative) ----------------
        # Discover wall-shaped fills of OTHER colors adjacent to
        # confirmed walls.  Iterates because color A walls may touch
        # color B walls which touch color C walls.
        pass5_total = 0
        for mc_iter in range(5):
            mc_grid = SpatialGrid(
                cell_size=max(2 * thresholds["mc_expand_eps"], 1.0))
            for idx in wall_indices:
                for px, py in _extract_endpoints(drawings[idx]):
                    mc_grid.insert(px, py, idx)

            mc_expansion = set()
            for i, d in enumerate(drawings):
                if i in wall_indices:
                    continue
                f = d.get("fill")
                if not f or f == (1.0, 1.0, 1.0):
                    continue
                if f == fc:
                    continue  # same color handled by Pass 1+4
                if not _is_wall_shaped(d["rect"], thresholds):
                    continue
                if not _is_wall_fill(d, thresholds):
                    continue
                for px, py in _extract_endpoints(d):
                    if mc_grid.query(px, py, thresholds["mc_expand_eps"]):
                        mc_expansion.add(i)
                        break

            if not mc_expansion:
                break
            wall_indices.update(mc_expansion)
            pass5_total += len(mc_expansion)

        if pass5_total:
            print(f"  Pass 5 multi-color: +{pass5_total} fills "
                  f"({mc_iter + 1} iterations)")

        print(f"  Total fill walls: {len(wall_indices)}")

    # --- Phase E: collect interior drawings (hatching) ---
    hatches = []
    if fp["hatch"]:
        # PATH 1a already identified hatching in Step 1 — use directly
        if 'hatch_indices' in dir() and hatch_indices:
            for i in hatch_indices:
                if i not in wall_indices:
                    hatches.append((i, drawings[i]))
        else:
            # Fallback: scan for hatching near wall borders
            hc = fp["hatch"]["color"]
            hw = fp["hatch"]["width"]
            ha = fp["hatch"]["angle"]
            wall_rects = [drawings[i]["rect"] for i in wall_indices]

            for i, d in enumerate(drawings):
                if i in wall_indices:
                    continue
                c, w = d.get("color"), d.get("width")
                if not c or not w:
                    continue
                if not all(abs(a - b) < 0.05 for a, b in zip(c, hc)):
                    continue
                if abs(w - hw) > 0.02:
                    continue
                if len(d["items"]) < 1:
                    continue
                dom = _dominant_angle(d["items"])
                if dom is not None and not _angle_close(dom, ha):
                    continue
                for wr in wall_rects:
                    if d["rect"].intersects(wr + (-3, -3, 3, 3)):
                        hatches.append((i, d))
                        break

    # Separate fills from edges in output
    fills = []
    edges_only = []
    for i, d in [(i, drawings[i]) for i in wall_indices]:
        if d.get("fill") and d["fill"] != (1.0, 1.0, 1.0):
            fills.append((i, d))
        else:
            edges_only.append((i, d))

    print(f"  Phase E: {len(hatches)} hatch drawings, {len(fills)} fill drawings")

    return {
        "edges": edges_only,
        "hatches": hatches,
        "fills": fills,
        "secondary_indices": secondary_indices,
        "components": [],
    }


# ---------------------------------------------------------------------------
# Step 5 — Overlay rendering
# ---------------------------------------------------------------------------

def generate_overlay(pdf_path, wall_result, output_path="wall_overlay.png",
                     dpi=200):
    """Re-draw detected wall vectors on the PDF in colour, then render.

    Colour coding:
      - Wall edges (primary + secondary): red
      - Hatching: red, thin
    """
    doc = fitz.open(pdf_path)
    page = doc[0]

    secondary = wall_result["secondary_indices"]

    # --- Primary edges (red) ---
    primary_by_w = defaultdict(list)
    for idx, d in wall_result["edges"]:
        if idx in secondary:
            continue
        w = d.get("width") or 0.6
        for item in d["items"]:
            primary_by_w[w].append(item)

    for w, items in primary_by_w.items():
        s = page.new_shape()
        for item in items:
            if item[0] == "l":
                s.draw_line(item[1], item[2])
            elif item[0] == "re":
                s.draw_rect(item[1])
            elif item[0] == "qu":
                s.draw_quad(item[1])
        s.finish(color=(1, 0, 0), width=w)
        s.commit()

    # --- Secondary-width edges (red, same as primary) ---
    secondary_by_w = defaultdict(list)
    for idx, d in wall_result["edges"]:
        if idx not in secondary:
            continue
        w = d.get("width") or 0.6
        for item in d["items"]:
            secondary_by_w[w].append(item)

    for w, items in secondary_by_w.items():
        s = page.new_shape()
        for item in items:
            if item[0] == "l":
                s.draw_line(item[1], item[2])
            elif item[0] == "re":
                s.draw_rect(item[1])
            elif item[0] == "qu":
                s.draw_quad(item[1])
        s.finish(color=(1, 0, 0), width=w)
        s.commit()

    # --- Hatching (red, thin) ---
    if wall_result["hatches"]:
        s = page.new_shape()
        for _, d in wall_result["hatches"]:
            for item in d["items"]:
                if item[0] == "l":
                    s.draw_line(item[1], item[2])
                elif item[0] == "re":
                    s.draw_rect(item[1])
                elif item[0] == "qu":
                    s.draw_quad(item[1])
        s.finish(color=(1, 0, 0), width=0.5)
        s.commit()

    # --- Filled wall rects (red fill with red outline) ---
    for _, d in wall_result.get("fills", []):
        s = page.new_shape()
        for item in d["items"]:
            if item[0] == "re":
                s.draw_rect(item[1])
            elif item[0] == "l":
                s.draw_line(item[1], item[2])
            elif item[0] == "qu":
                s.draw_quad(item[1])
        s.finish(color=(1, 0, 0), fill=(1, 0, 0), width=0.5)
        s.commit()

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    pix.save(output_path)
    doc.close()
    return output_path


# ---------------------------------------------------------------------------
# VLM-free fallback fingerprint
# ---------------------------------------------------------------------------

def _fallback_fingerprint(drawings):
    """Auto-detect wall pattern without VLM by analysing the full drawing set.

    Tries in order:
      1. If there are filled black rectangles, assume they are walls.
      2. Otherwise, find the thickest common stroke class (width >= 0.3).
    """
    # Strategy 1: wall-shaped filled rectangles
    wall_fills = [d for d in drawings
                  if d.get("fill") and d["fill"] != (1.0, 1.0, 1.0)
                  and _is_wall_shaped(d["rect"])]
    if len(wall_fills) >= 5:
        fill_colors = Counter(d["fill"] for d in wall_fills)
        fill_color = fill_colors.most_common(1)[0][0]
        print(f"  Fallback: found {len(wall_fills)} wall-shaped filled rects")
        return {
            "edge_color": None,
            "edge_width": None,
            "secondary_edge_widths": [],
            "fill_color": fill_color,
            "hatch": None,
            "wall_style": "solid_fill",
        }

    # Strategy 2: thickest common stroke class
    width_counts = Counter()
    for d in drawings:
        w = d.get("width")
        c = d.get("color")
        if w and c and w >= 0.3:
            width_counts[round(w, 1)] += 1

    if not width_counts:
        return None

    # Pick the most common thick stroke class
    edge_width = width_counts.most_common(1)[0][0]
    print(f"  Fallback: using most common thick stroke width={edge_width} "
          f"({width_counts[edge_width]} drawings)")
    return {
        "edge_color": (0.0, 0.0, 0.0),
        "edge_width": edge_width,
        "secondary_edge_widths": [],
        "fill_color": None,
        "hatch": None,
        "wall_style": "outline",
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_pdf(pdf_path, tag=None, pt_per_inch=None):
    """Run the full wall-detection pipeline on a single PDF.

    Uses a select-merge architecture: auto-detect multiple fingerprints,
    run each through find_all_walls, evaluate with VLM, merge approved results.

    *tag* is used to name output files.  If None, derived from the filename.
    *pt_per_inch* overrides the scale lookup from KNOWN_SCALES.
    """
    if tag is None:
        tag = os.path.splitext(os.path.basename(pdf_path))[0].replace(" ", "_")

    # Scale lookup
    basename = os.path.basename(pdf_path)
    if pt_per_inch is None:
        pt_per_inch = KNOWN_SCALES.get(basename, DEFAULT_PT_PER_INCH)
    thresholds = compute_thresholds(pt_per_inch)

    # Output directory
    out_dir = f"output/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"  WALL DETECTION — {basename}")
    print(f"  Scale: {pt_per_inch} pt/inch")
    print("=" * 60)

    # -- Load ---------------------------------------------------------------
    print("\n[1] Loading PDF and extracting vectors...")
    doc = fitz.open(pdf_path)
    page = doc[0]
    drawings = page.get_drawings()
    print(f"  Page: {page.rect}  rotation={page.rotation}")
    print(f"  Total vector drawings: {len(drawings)}")

    # -- Auto-detect fingerprints -------------------------------------------
    print("\n[2] Auto-detecting wall fingerprints...")
    fps = auto_detect_fingerprints(drawings, thresholds)
    print(f"  Found {len(fps)} fingerprint candidates")
    for i, fp_item in enumerate(fps[:5]):
        print(f"    {i+1}. {fp_item.get('_label', '?')} "
              f"(score={fp_item.get('_score', 0)})")

    # -- Run each fingerprint -----------------------------------------------
    print("\n[3] Running wall detection for each fingerprint...")
    individual_results = []
    for i, fp in enumerate(fps[:5]):
        label = fp.pop("_label", f"fp{i+1}")
        fp.pop("_score", None)
        print(f"\n  --- Fingerprint {i+1}: {label} ---")
        result = find_all_walls(drawings, fp, thresholds)
        n = len(result["edges"]) + len(result["fills"]) + len(result["hatches"])
        overlay = f"{out_dir}/fp{i+1}_overlay.png"
        generate_overlay(pdf_path, result, output_path=overlay)
        print(f"  {n} wall drawings detected, overlay: {overlay}")
        individual_results.append((label, n, result, overlay))

    if not individual_results:
        print("  No fingerprints detected. Pipeline cannot continue.")
        doc.close()
        return

    # -- VLM evaluates each -------------------------------------------------
    print("\n[4] VLM evaluation of each fingerprint overlay...")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    approved = []
    for i, (label, count, result, overlay_path) in enumerate(individual_results):
        img = open(overlay_path, "rb").read()
        prompt = (f"This is a floor plan with walls highlighted in RED "
                  f"(pattern: '{label}', {count} walls).\n\n"
                  f"Score 1-10, should this be included?\n"
                  f'Return JSON: {{"score": <1-10>, "include": <true/false>, '
                  f'"reason": "<1 sentence>"}}')
        try:
            eval_result = _vlm_call(client, img, prompt, max_tokens=256)
        except Exception as e:
            print(f"    VLM eval failed for fp{i+1}: {e}")
            eval_result = {"score": 0, "include": False,
                           "reason": f"VLM eval error: {e}"}
        # Save eval
        with open(f"{out_dir}/fp{i+1}_eval.json", "w") as f:
            json.dump(eval_result, f, indent=2)
        include = eval_result.get("include", False)
        score = eval_result.get("score", 0)
        reason = eval_result.get("reason", "")
        print(f"    fp{i+1} ({label}): score={score}, "
              f"include={include}, reason={reason}")
        if include:
            approved.append(result)

    if not approved:
        # Fallback: use top-scoring
        print("  No fingerprints approved by VLM — using top result as fallback")
        approved = [individual_results[0][2]] if individual_results else []

    # -- Merge approved results ---------------------------------------------
    print(f"\n[5] Merging {len(approved)} approved result(s)...")
    merged_edge_idx = set()
    merged_edges = []
    merged_hatch_idx = set()
    merged_hatches = []
    merged_fill_idx = set()
    merged_fills = []
    merged_secondary = set()

    for result in approved:
        for idx, d in result["edges"]:
            if idx not in merged_edge_idx:
                merged_edge_idx.add(idx)
                merged_edges.append((idx, d))
        for idx, d in result["hatches"]:
            if idx not in merged_hatch_idx:
                merged_hatch_idx.add(idx)
                merged_hatches.append((idx, d))
        for idx, d in result["fills"]:
            if idx not in merged_fill_idx:
                merged_fill_idx.add(idx)
                merged_fills.append((idx, d))
        merged_secondary.update(result["secondary_indices"])

    merged = {
        "edges": merged_edges,
        "hatches": merged_hatches,
        "fills": merged_fills,
        "secondary_indices": merged_secondary,
        "components": [],
    }

    n_edges = len(merged["edges"])
    n_hatches = len(merged["hatches"])
    n_fills = len(merged["fills"])
    total_wall = n_edges + n_hatches + n_fills
    print(f"  Merged: {n_edges} edges, {n_hatches} hatches, {n_fills} fills "
          f"({total_wall} total)")

    # -- Generate final overlay ---------------------------------------------
    print("\n[6] Generating final overlay...")
    final_path = f"{out_dir}/final_overlay.png"
    generate_overlay(pdf_path, merged, output_path=final_path)
    print(f"  Saved to: {final_path}")

    # -- Save summary.json --------------------------------------------------
    summary = {
        "pdf": pdf_path,
        "basename": basename,
        "pt_per_inch": pt_per_inch,
        "thresholds": thresholds,
        "fingerprints_detected": len(fps),
        "fingerprints_evaluated": len(individual_results),
        "fingerprints_approved": len(approved),
        "total_wall_drawings": total_wall,
        "edge_drawings": n_edges,
        "hatch_drawings": n_hatches,
        "fill_drawings": n_fills,
        "total_drawings_in_pdf": len(drawings),
        "wall_coverage_pct": round(
            total_wall / max(len(drawings), 1) * 100, 1),
    }
    summary_path = f"{out_dir}/summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")

    # -- Console summary ----------------------------------------------------
    print("\n" + "-" * 60)
    print("  SUMMARY")
    print("-" * 60)
    print(f"  Scale (pt/inch)         : {pt_per_inch}")
    print(f"  Fingerprints detected   : {len(fps)}")
    print(f"  Fingerprints approved   : {len(approved)}")
    print(f"  Total drawings in PDF   : {len(drawings)}")
    print(f"  Wall edge drawings      : {n_edges}")
    print(f"  Wall hatch drawings     : {n_hatches}")
    print(f"  Wall fill drawings      : {n_fills}")
    print(f"  Total wall drawings     : {total_wall}")
    print(f"  Wall vector coverage    : "
          f"{total_wall / max(len(drawings), 1) * 100:.1f}%")
    if merged_secondary:
        print(f"  Secondary-width drawings: {len(merged_secondary)}")

    doc.close()
    print("\nDone!\n")


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
            print(f"  ERROR processing {pdf_path}: {e}\n")


if __name__ == "__main__":
    main()
