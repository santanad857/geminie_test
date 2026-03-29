#!/usr/bin/env python3
"""
Step 1: Exhaustive Pattern Extraction from Vector PDFs
======================================================

Extracts all geometric primitives from a PDF page, explodes multi-item
CAD paths into atomic primitives, groups them by quantized visual style,
and generates validation overlays.

Core function: fitz.Page.get_drawings()
"""

import json
import math
import os
import sys
from collections import defaultdict
from typing import Any

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Known scales (pt_per_inch) -- will be replaced by auto-detection later
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
# Quantization parameters
# ---------------------------------------------------------------------------

COLOR_DECIMALS = 2   # round RGB components to this many decimal places
WIDTH_DECIMALS = 2   # round stroke widths to this many decimal places


# ═══════════════════════════════════════════════════════════════════════════
# Quantization helpers
# ═══════════════════════════════════════════════════════════════════════════

def quantize_color(rgb: tuple | None, decimals: int = COLOR_DECIMALS) -> tuple | None:
    """Round RGB floats to absorb CAD floating-point drift."""
    if rgb is None:
        return None
    return tuple(round(float(c), decimals) for c in rgb)


def quantize_width(width: float, decimals: int = WIDTH_DECIMALS) -> float:
    """Round stroke width to absorb floating-point drift."""
    return round(float(width), decimals)


def make_stroke_key(color: tuple | None, width: float) -> str:
    """Build a stroke-only style key.  Example: stroke_rgb(0.0,0.0,0.0)_width_1.00"""
    if color is not None:
        r, g, b = quantize_color(color)
    else:
        r, g, b = "none", "none", "none"
    w = quantize_width(width)
    return f"stroke_rgb({r},{g},{b})_width_{w:.{WIDTH_DECIMALS}f}"


def make_fill_key(fill_color: tuple | None) -> str:
    """Build a fill-only style key.  Example: fill_rgb(0.8,0.8,0.8)"""
    if fill_color is not None:
        r, g, b = quantize_color(fill_color)
    else:
        r, g, b = "none", "none", "none"
    return f"fill_rgb({r},{g},{b})"


# ═══════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════

def point_to_list(pt) -> list[float]:
    """fitz.Point -> [x, y]"""
    return [float(pt.x), float(pt.y)]


def rect_to_corners(rect) -> list[list[float]]:
    """fitz.Rect -> 4 corners [TL, TR, BR, BL] (clockwise)."""
    return [
        [float(rect.x0), float(rect.y0)],
        [float(rect.x1), float(rect.y0)],
        [float(rect.x1), float(rect.y1)],
        [float(rect.x0), float(rect.y1)],
    ]


def quad_to_corners(quad) -> list[list[float]]:
    """fitz.Quad -> 4 corners [UL, UR, LR, LL] (clockwise)."""
    return [
        point_to_list(quad.ul),
        point_to_list(quad.ur),
        point_to_list(quad.lr),
        point_to_list(quad.ll),
    ]


def segment_length(p1: list[float], p2: list[float]) -> float:
    """Euclidean distance between two 2-D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def polygon_perimeter(corners: list[list[float]]) -> float:
    """Sum of edge lengths for a closed polygon."""
    n = len(corners)
    return sum(
        segment_length(corners[i], corners[(i + 1) % n]) for i in range(n)
    )


# ═══════════════════════════════════════════════════════════════════════════
# Length computation (raw + output) -- used for conservation tests
# ═══════════════════════════════════════════════════════════════════════════

def raw_item_length(item: tuple) -> float:
    """Length contribution of a single raw drawing item from get_drawings()."""
    kind = item[0]
    if kind == "l":
        p1, p2 = item[1], item[2]
        return math.hypot(float(p2.x - p1.x), float(p2.y - p1.y))
    elif kind == "re":
        rect = item[1]
        w = abs(float(rect.x1 - rect.x0))
        h = abs(float(rect.y1 - rect.y0))
        return 2.0 * (w + h)
    elif kind == "qu":
        corners = quad_to_corners(item[1])
        return polygon_perimeter(corners)
    elif kind == "c":
        pts = [point_to_list(item[i]) for i in range(1, 5)]
        return sum(segment_length(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    return 0.0


def compute_raw_total_length(drawings: list[dict]) -> float:
    """
    Total geometric length from raw get_drawings() output.

    For 'fs' paths, re/qu items are emitted into both the stroke and fill
    buckets, so their length is counted twice to match the output.
    """
    total = 0.0
    for path in drawings:
        path_type = path.get("type", "s")
        is_fs = path_type == "fs"
        for item in path["items"]:
            length = raw_item_length(item)
            if is_fs and item[0] in ("re", "qu"):
                total += 2.0 * length  # emitted in both stroke and fill
            else:
                total += length
    return total


def primitive_length(prim: dict) -> float:
    """Length contribution of an extracted primitive."""
    t = prim["type"]
    pts = prim["points"]
    if t == "line":
        return segment_length(pts[0], pts[1])
    elif t == "polygon":
        return polygon_perimeter(pts)
    elif t == "curve":
        return sum(segment_length(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Core: Path Explosion
# ═══════════════════════════════════════════════════════════════════════════

def explode_path(path: dict, path_index: int) -> dict[str, list[dict]]:
    """
    Explode a single path into **separate** stroke and fill emissions.

    Returns {"stroke": [...], "fill": [...]}.

    A primitive is EITHER a stroke OR a fill, never both.  For 'fs' paths
    (which have both), re/qu items are emitted into both buckets (stroke
    gets 4 boundary lines, fill gets 1 polygon).  l/c items go to the
    stroke bucket only (they have visible stroke; individually they do not
    define filled regions).

    Primitive schema:
        {"type": "line"|"polygon"|"curve",
         "points": [[x,y], ...],
         "original_path_index": int}
    """
    stroke_prims: list[dict] = []
    fill_prims: list[dict] = []
    path_type = path.get("type", "s")
    has_stroke = "s" in path_type
    has_fill = "f" in path_type

    for item in path["items"]:
        kind = item[0]

        if kind == "l":
            prim = {
                "type": "line",
                "points": [point_to_list(item[1]), point_to_list(item[2])],
                "original_path_index": path_index,
            }
            # Lines go to stroke if available, else fill (boundary segment)
            if has_stroke:
                stroke_prims.append(prim)
            else:
                fill_prims.append(prim)

        elif kind == "re":
            corners = rect_to_corners(item[1])
            if has_stroke:
                for i in range(4):
                    stroke_prims.append({
                        "type": "line",
                        "points": [corners[i], corners[(i + 1) % 4]],
                        "original_path_index": path_index,
                    })
            if has_fill:
                fill_prims.append({
                    "type": "polygon",
                    "points": corners,
                    "original_path_index": path_index,
                })

        elif kind == "qu":
            corners = quad_to_corners(item[1])
            if has_stroke:
                for i in range(4):
                    stroke_prims.append({
                        "type": "line",
                        "points": [corners[i], corners[(i + 1) % 4]],
                        "original_path_index": path_index,
                    })
            if has_fill:
                fill_prims.append({
                    "type": "polygon",
                    "points": corners,
                    "original_path_index": path_index,
                })

        elif kind == "c":
            prim = {
                "type": "curve",
                "points": [point_to_list(item[i]) for i in range(1, 5)],
                "original_path_index": path_index,
            }
            if has_stroke:
                stroke_prims.append(prim)
            else:
                fill_prims.append(prim)

    return {"stroke": stroke_prims, "fill": fill_prims}


# ═══════════════════════════════════════════════════════════════════════════
# Main extraction entry point
# ═══════════════════════════════════════════════════════════════════════════

def extract_patterns(pdf_path: str, page_number: int = 0) -> dict[str, Any]:
    """
    Ingest a single PDF page and return every geometric primitive grouped
    by quantized visual style.

    Returns dict with keys:
        "grouped"      : {style_key: [primitive, ...]}  (largest first)
        "raw_drawings" : original get_drawings() list
        "page_width"   : float (points)
        "page_height"  : float (points)
    """
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    drawings = page.get_drawings()

    grouped: dict[str, list[dict]] = defaultdict(list)

    for idx, path in enumerate(drawings):
        color = path.get("color")
        fill_color = path.get("fill")
        width = path.get("width", 0.0)

        emissions = explode_path(path, idx)

        if emissions["stroke"]:
            key = make_stroke_key(color, width)
            grouped[key].extend(emissions["stroke"])

        if emissions["fill"]:
            key = make_fill_key(fill_color)
            grouped[key].extend(emissions["fill"])

    # Sort groups by primitive count (largest first)
    grouped = dict(sorted(grouped.items(), key=lambda kv: -len(kv[1])))

    result = {
        "grouped": grouped,
        "raw_drawings": drawings,
        "page_width": float(page.rect.width),
        "page_height": float(page.rect.height),
    }
    doc.close()
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════════════

def patterns_to_json(grouped: dict) -> dict:
    """Return a JSON-serializable copy of the grouped primitives."""
    return {key: prims for key, prims in grouped.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Visualization: Faded-background + neon overlays
# ═══════════════════════════════════════════════════════════════════════════

OVERLAY_PALETTE = [
    (0, 1, 1),       # cyan
    (1, 0, 1),       # magenta
    (0, 1, 0),       # green
    (1, 0.5, 0),     # orange
    (1, 1, 0),       # yellow
    (0.5, 0, 1),     # purple
    (1, 0, 0),       # red
    (0, 0.5, 1),     # sky blue
    (1, 0.5, 0.5),   # salmon
    (0.5, 1, 0.5),   # light green
]


def _draw_primitives_on_shape(shape, prims: list[dict], color: tuple, line_width: float):
    """Draw a list of primitives onto a fitz.Shape."""
    for prim in prims:
        pts = prim["points"]
        ptype = prim["type"]

        if ptype == "line":
            shape.draw_line(
                fitz.Point(pts[0][0], pts[0][1]),
                fitz.Point(pts[1][0], pts[1][1]),
            )
            shape.finish(color=color, width=line_width)

        elif ptype == "polygon" and len(pts) >= 3:
            fitz_pts = [fitz.Point(p[0], p[1]) for p in pts]
            shape.draw_polyline(fitz_pts + [fitz_pts[0]])
            shape.finish(color=color, fill=color, width=0.5, fill_opacity=0.4)

        elif ptype == "curve" and len(pts) == 4:
            shape.draw_bezier(
                fitz.Point(pts[0][0], pts[0][1]),
                fitz.Point(pts[1][0], pts[1][1]),
                fitz.Point(pts[2][0], pts[2][1]),
                fitz.Point(pts[3][0], pts[3][1]),
            )
            shape.finish(color=color, width=line_width)


def generate_overlays(
    pdf_path: str,
    page_number: int,
    grouped: dict[str, list[dict]],
    output_dir: str,
    top_n: int = 10,
    line_width: float = 2.0,
    render_scale: float = 2.0,
):
    """
    For each of the top_n largest style groups, render a PNG:
      - Full original page faded to ~30% (white overlay at 70% opacity)
      - Group's primitives drawn in a high-contrast neon color on top
    """
    os.makedirs(output_dir, exist_ok=True)
    mat = fitz.Matrix(render_scale, render_scale)

    sorted_groups = sorted(grouped.items(), key=lambda kv: -len(kv[1]))[:top_n]

    for rank, (style_key, prims) in enumerate(sorted_groups):
        # Fresh document copy per overlay
        odoc = fitz.open(pdf_path)
        opage = odoc[page_number]

        # Step 1: fade background with semi-transparent white rect
        bg = opage.new_shape()
        bg.draw_rect(opage.rect)
        bg.finish(fill=(1, 1, 1), fill_opacity=0.7)
        bg.commit()

        # Step 2: draw the group's primitives in neon color
        color = OVERLAY_PALETTE[rank % len(OVERLAY_PALETTE)]
        fg = opage.new_shape()
        _draw_primitives_on_shape(fg, prims, color, line_width)
        fg.commit()

        # Render and save
        pix = opage.get_pixmap(matrix=mat, alpha=False)
        safe_key = (
            style_key.replace("(", "")
            .replace(")", "")
            .replace(",", "_")
            .replace(" ", "")
        )[:80]
        filename = f"pattern_{rank + 1:02d}__{safe_key}.png"
        filepath = os.path.join(output_dir, filename)
        pix.save(filepath)
        odoc.close()

        print(f"  [{rank + 1}/{len(sorted_groups)}] {filename}  ({len(prims):,} primitives)")


# ═══════════════════════════════════════════════════════════════════════════
# Summary report
# ═══════════════════════════════════════════════════════════════════════════

def generate_summary(grouped: dict, raw_drawings: list) -> dict:
    """Generate a JSON-serializable summary of the extraction."""
    raw_total = compute_raw_total_length(raw_drawings)
    output_total = sum(
        primitive_length(p) for prims in grouped.values() for p in prims
    )
    total_primitives = sum(len(prims) for prims in grouped.values())

    groups_info = []
    for key, prims in grouped.items():
        type_counts: dict[str, int] = defaultdict(int)
        for p in prims:
            type_counts[p["type"]] += 1
        groups_info.append({
            "style_key": key,
            "count": len(prims),
            "types": dict(type_counts),
        })

    return {
        "total_raw_paths": len(raw_drawings),
        "total_style_groups": len(grouped),
        "total_primitives": total_primitives,
        "raw_total_length_pt": round(raw_total, 4),
        "output_total_length_pt": round(output_total, 4),
        "length_conservation_error": round(abs(raw_total - output_total), 6),
        "groups": groups_info,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def run(
    pdf_path: str,
    page_number: int = 0,
    output_dir: str | None = None,
    top_n: int = 10,
):
    """Run the full Step-1 pipeline on a single PDF page."""
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    if output_dir is None:
        output_dir = os.path.join("output", basename.replace(" ", "_"), "step1")

    print(f"=== {pdf_path} (page {page_number}) ===")
    result = extract_patterns(pdf_path, page_number)
    grouped = result["grouped"]
    raw_drawings = result["raw_drawings"]

    total_prims = sum(len(v) for v in grouped.values())
    print(f"  Raw paths:        {len(raw_drawings):,}")
    print(f"  Style groups:     {len(grouped):,}")
    print(f"  Total primitives: {total_prims:,}")

    # JSON summary
    os.makedirs(output_dir, exist_ok=True)
    summary = generate_summary(grouped, raw_drawings)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Conservation err: {summary['length_conservation_error']}")

    # Full pattern data
    data_path = os.path.join(output_dir, "patterns.json")
    with open(data_path, "w") as f:
        json.dump(patterns_to_json(grouped), f, indent=2)

    # Overlays
    print(f"  Generating top-{top_n} overlays...")
    generate_overlays(pdf_path, page_number, grouped, output_dir, top_n=top_n)

    print(f"  Output: {output_dir}/")
    return result


if __name__ == "__main__":
    pdf_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "example_plans"
    )

    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        paths = sorted(
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.endswith(".pdf")
        )

    for pdf_path in paths:
        run(pdf_path)
        print()
