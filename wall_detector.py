"""
Wall Detection Pipeline for Architectural 2D Floor Plans (Division 9 Takeoffs).

Deterministic vector-geometry pipeline with optional VLM micro-validation.
Extracts structural walls from vector-based PDF floor plans and highlights them.

Usage:
    python wall_detector.py input.pdf [output.pdf] [--no-vlm] [--debug]
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import fitz  # PyMuPDF
import numpy as np

# ---------------------------------------------------------------------------
# Constants / tunables
# ---------------------------------------------------------------------------

# Render scale factor (PDF points → pixels).  3× gives ~216 DPI.
RENDER_SCALE: int = 4

# Rotation detection: tolerance (degrees) for snapping to 0°/90°
ROTATION_SNAP_TOL: float = 5.0

# Collinear merge: max gap (PDF pts) between segment endpoints to merge
COLLINEAR_GAP_TOL: float = 3.0
# Collinear merge: max perpendicular offset (PDF pts) for "same axis"
COLLINEAR_AXIS_TOL: float = 1.5

# Semantic filtering
MIN_LINE_LENGTH_PT: float = 4.0          # Drop wall-boundary segments shorter than this
MIN_HATCH_LENGTH_PT: float = 0.5         # Much lower threshold for hatching strokes
TEXT_BBOX_PADDING_PT: float = 2.0        # Padding around text boxes for exclusion
WALL_WIDTH_RANGE: tuple[float, float] = (0.10, 1.0)  # Acceptable stroke widths

# Hatching detection (adaptive — not tied to a specific color)
HATCHING_ANGLE_RANGE: tuple[float, float] = (25.0, 65.0)  # Degrees from horizontal

# Parallel-pair detection (primary, hatching-independent method)
PAIR_GAP_MIN_PT: float = 2.0   # Min gap between parallel H/V lines to be a wall pair
PAIR_GAP_MAX_PT: float = 18.0  # Max gap (thick exterior walls)
PAIR_OVERLAP_MIN_PT: float = 20.0  # Min overlapping run to qualify as a wall pair
PAIR_MIN_LINE_LEN_PT: float = 15.0  # Each line in a pair must be at least this long
PAIR_BORDER_MARGIN_PT: float = 30.0  # Ignore lines within this margin of the page edge

# Morphology
HATCH_RENDER_THICKNESS_PX: int = 3      # Render hatching strokes at this px width
HATCH_CLOSE_KERNEL: int = 15           # Close gaps between adjacent hatch strokes
MORPH_WALL_THICKEN: int = 3            # Light thickening (pixels) for hatching bands
MORPH_SMOOTH_KERNEL: int = 7           # Isotropic smoothing kernel

# Contour filtering
MIN_CONTOUR_AREA_PT2: float = 30.0      # Min area in PDF-pt² to keep a contour
MIN_CONTOUR_LENGTH_PT: float = 10.0     # Min perimeter
MAX_CONTOUR_ASPECT: float = 500.0       # Max aspect ratio (walls can be very elongated)

# Highlight appearance
HIGHLIGHT_COLOR: tuple[float, float, float] = (1.0, 1.0, 0.0)  # Yellow (R, G, B 0-1)
HIGHLIGHT_OPACITY: float = 0.30

# VLM
VLM_CROP_PAD_PT: float = 10.0  # Padding around contour bbox for the VLM crop
VLM_MODEL: str = "gemini-2.5-flash-lite"
VLM_MAX_RETRIES: int = 2

logger = logging.getLogger("wall_detector")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LineSegment:
    """A single straight line in PDF-point coordinates."""
    x0: float
    y0: float
    x1: float
    y1: float
    width: float = 0.0
    color: tuple[float, ...] = (0.0, 0.0, 0.0)
    is_fill: bool = False

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def angle_deg(self) -> float:
        """Angle in [0, 180) degrees from the positive X-axis."""
        return math.degrees(math.atan2(abs(self.y1 - self.y0), abs(self.x1 - self.x0)))

    @property
    def is_horizontal(self) -> bool:
        return self.angle_deg < ROTATION_SNAP_TOL

    @property
    def is_vertical(self) -> bool:
        return self.angle_deg > (90.0 - ROTATION_SNAP_TOL)

    @property
    def is_axis_aligned(self) -> bool:
        return self.is_horizontal or self.is_vertical

    @property
    def is_diagonal_hatching(self) -> bool:
        a = self.angle_deg
        return HATCHING_ANGLE_RANGE[0] <= a <= HATCHING_ANGLE_RANGE[1]


@dataclass
class TextBlock:
    """A text region extracted from the PDF."""
    x0: float
    y0: float
    x1: float
    y1: float
    text: str = ""

    @property
    def padded_rect(self) -> tuple[float, float, float, float]:
        p = TEXT_BBOX_PADDING_PT
        return (self.x0 - p, self.y0 - p, self.x1 + p, self.y1 + p)


@dataclass
class WallContour:
    """A detected wall region as a polygon in PDF coordinates."""
    polygon_pdf: np.ndarray          # shape (N, 2) in PDF-point space
    bbox_pdf: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    area_pdf: float
    validated: Optional[bool] = None  # None = not yet validated, True/False after VLM


# ---------------------------------------------------------------------------
# Step 1: Vector Extraction & Normalization
# ---------------------------------------------------------------------------

def extract_vectors(page: fitz.Page) -> tuple[list[LineSegment], list[TextBlock]]:
    """Extract all vector line segments and text blocks from a PDF page."""
    drawings = page.get_drawings()
    segments: list[LineSegment] = []

    for d in drawings:
        color = d.get("color")
        fill = d.get("fill")
        width = d.get("width") or 0.0
        is_fill = d.get("type") == "f"

        for item in d.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                seg = LineSegment(
                    x0=p1.x, y0=p1.y,
                    x1=p2.x, y1=p2.y,
                    width=width,
                    color=color if color else (fill if fill else (0.0, 0.0, 0.0)),
                    is_fill=is_fill,
                )
                segments.append(seg)
            elif item[0] == "re":
                # Decompose rectangle into 4 line segments
                r = item[1]
                corners = [
                    (r.x0, r.y0), (r.x1, r.y0),
                    (r.x1, r.y1), (r.x0, r.y1),
                ]
                effective_color = color if color else (fill if fill else (0.0, 0.0, 0.0))
                for j in range(4):
                    c0, c1 = corners[j], corners[(j + 1) % 4]
                    seg = LineSegment(
                        x0=c0[0], y0=c0[1],
                        x1=c1[0], y1=c1[1],
                        width=width,
                        color=effective_color,
                        is_fill=is_fill,
                    )
                    segments.append(seg)

    # Text blocks
    raw_blocks = page.get_text("blocks")
    text_blocks: list[TextBlock] = []
    for b in raw_blocks:
        # b = (x0, y0, x1, y1, text, block_no, block_type)
        if len(b) >= 5:
            text_blocks.append(TextBlock(
                x0=b[0], y0=b[1], x1=b[2], y1=b[3],
                text=str(b[4]).strip(),
            ))

    logger.info(
        "Extracted %d line segments, %d text blocks from page",
        len(segments), len(text_blocks),
    )
    return segments, text_blocks


def detect_rotation_angle(segments: list[LineSegment]) -> float:
    """Detect dominant rotation from axis-aligned segments.

    Returns the correction angle in degrees to apply so that the longest
    structural lines become strictly horizontal / vertical.
    """
    # Collect angles of the longest segments (likely structural)
    long_segs = sorted(segments, key=lambda s: s.length, reverse=True)[:500]

    angle_bins: dict[int, float] = {}  # rounded-angle → total length
    for seg in long_segs:
        if seg.length < 10:
            continue
        a = math.degrees(math.atan2(seg.y1 - seg.y0, seg.x1 - seg.x0)) % 180
        bin_key = round(a)
        angle_bins[bin_key] = angle_bins.get(bin_key, 0.0) + seg.length

    if not angle_bins:
        return 0.0

    dominant_angle = max(angle_bins, key=lambda k: angle_bins[k])

    # Snap: if within tolerance of 0° or 90°, no rotation needed
    if dominant_angle <= ROTATION_SNAP_TOL or dominant_angle >= (180 - ROTATION_SNAP_TOL):
        return 0.0
    if abs(dominant_angle - 90) <= ROTATION_SNAP_TOL:
        return 0.0

    # Otherwise return the offset from nearest axis
    nearest = min([0, 90, 180], key=lambda ax: abs(dominant_angle - ax))
    correction = -(dominant_angle - nearest)
    logger.info("Detected rotation offset %.1f° (dominant angle %d°)", correction, dominant_angle)
    return correction


def rotate_segments(
    segments: list[LineSegment],
    angle_deg: float,
    cx: float,
    cy: float,
) -> list[LineSegment]:
    """Rotate all segments around (cx, cy) by angle_deg."""
    if abs(angle_deg) < 0.01:
        return segments

    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    rotated: list[LineSegment] = []
    for s in segments:
        dx0, dy0 = s.x0 - cx, s.y0 - cy
        dx1, dy1 = s.x1 - cx, s.y1 - cy
        rotated.append(LineSegment(
            x0=cx + dx0 * cos_a - dy0 * sin_a,
            y0=cy + dx0 * sin_a + dy0 * cos_a,
            x1=cx + dx1 * cos_a - dy1 * sin_a,
            y1=cy + dx1 * sin_a + dy1 * cos_a,
            width=s.width,
            color=s.color,
            is_fill=s.is_fill,
        ))
    return rotated


def merge_collinear_segments(segments: list[LineSegment]) -> list[LineSegment]:
    """Merge collinear horizontal/vertical segments that share an axis within tolerance.

    Groups segments by their fixed-axis coordinate, then merges segments whose
    variable-axis ranges overlap or are separated by a tiny gap.
    """
    h_groups: dict[int, list[LineSegment]] = {}  # quantized Y → list
    v_groups: dict[int, list[LineSegment]] = {}  # quantized X → list
    others: list[LineSegment] = []

    quant = COLLINEAR_AXIS_TOL

    for seg in segments:
        if seg.is_horizontal and seg.length >= MIN_LINE_LENGTH_PT:
            key = round(((seg.y0 + seg.y1) / 2) / quant)
            h_groups.setdefault(key, []).append(seg)
        elif seg.is_vertical and seg.length >= MIN_LINE_LENGTH_PT:
            key = round(((seg.x0 + seg.x1) / 2) / quant)
            v_groups.setdefault(key, []).append(seg)
        else:
            others.append(seg)

    merged: list[LineSegment] = list(others)

    # Merge horizontal groups
    for _key, group in h_groups.items():
        merged.extend(_merge_1d(group, axis="h"))

    # Merge vertical groups
    for _key, group in v_groups.items():
        merged.extend(_merge_1d(group, axis="v"))

    logger.info(
        "Collinear merge: %d segments → %d segments",
        len(segments), len(merged),
    )
    return merged


def _merge_1d(group: list[LineSegment], axis: str) -> list[LineSegment]:
    """Merge a group of segments along one axis.

    For horizontal segments: merge along X, fixed Y.
    For vertical segments: merge along Y, fixed X.
    """
    if not group:
        return []

    if axis == "h":
        intervals = [(min(s.x0, s.x1), max(s.x0, s.x1), s) for s in group]
    else:
        intervals = [(min(s.y0, s.y1), max(s.y0, s.y1), s) for s in group]

    intervals.sort(key=lambda t: t[0])

    result: list[LineSegment] = []
    cur_lo, cur_hi, cur_seg = intervals[0]

    for lo, hi, seg in intervals[1:]:
        if lo <= cur_hi + COLLINEAR_GAP_TOL:
            cur_hi = max(cur_hi, hi)
            # Keep the wider/darker segment's metadata
            if seg.width >= cur_seg.width:
                cur_seg = seg
        else:
            result.append(_make_merged_seg(cur_lo, cur_hi, cur_seg, axis))
            cur_lo, cur_hi, cur_seg = lo, hi, seg

    result.append(_make_merged_seg(cur_lo, cur_hi, cur_seg, axis))
    return result


def _make_merged_seg(lo: float, hi: float, template: LineSegment, axis: str) -> LineSegment:
    """Create a merged segment from an interval and a template segment."""
    if axis == "h":
        y = (template.y0 + template.y1) / 2
        return LineSegment(x0=lo, y0=y, x1=hi, y1=y,
                           width=template.width, color=template.color,
                           is_fill=template.is_fill)
    else:
        x = (template.x0 + template.x1) / 2
        return LineSegment(x0=x, y0=lo, x1=x, y1=hi,
                           width=template.width, color=template.color,
                           is_fill=template.is_fill)


# ---------------------------------------------------------------------------
# Step 2: Semantic Filtering
# ---------------------------------------------------------------------------

def _rect_intersects(
    ax0: float, ay0: float, ax1: float, ay1: float,
    bx0: float, by0: float, bx1: float, by1: float,
) -> bool:
    """Check if two axis-aligned rectangles overlap."""
    return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0


def _seg_bbox(seg: LineSegment) -> tuple[float, float, float, float]:
    return (
        min(seg.x0, seg.x1), min(seg.y0, seg.y1),
        max(seg.x0, seg.x1), max(seg.y0, seg.y1),
    )


def filter_segments(
    segments: list[LineSegment],
    text_blocks: list[TextBlock],
) -> tuple[list[LineSegment], list[LineSegment]]:
    """Apply semantic filters.  Returns (wall_candidates, hatching_lines).

    Hatching detection is adaptive: any diagonal lines that appear in a
    consistent, dense pattern are treated as hatching, regardless of color.
    This avoids hardcoding a specific hatching color.
    """
    text_rects = [tb.padded_rect for tb in text_blocks]

    wall_candidates: list[LineSegment] = []
    all_diagonals: list[LineSegment] = []

    for seg in segments:
        if seg.width > WALL_WIDTH_RANGE[1]:
            continue

        if seg.is_diagonal_hatching and seg.length >= MIN_HATCH_LENGTH_PT:
            all_diagonals.append(seg)
        elif seg.is_axis_aligned and _is_dark(seg.color):
            if seg.length < MIN_LINE_LENGTH_PT:
                continue
            if seg.width < WALL_WIDTH_RANGE[0]:
                continue
            sb = _seg_bbox(seg)
            near_text = any(
                _rect_intersects(sb[0], sb[1], sb[2], sb[3], *tr)
                for tr in text_rects
            )
            if not near_text:
                wall_candidates.append(seg)

    # --- Adaptive hatching detection ---
    # Diagonal lines from many sources (text glyphs, dimension ticks, hatching).
    # True wall-hatching has a distinctive signature: many short strokes at a
    # consistent angle, clustered spatially.  We detect the dominant diagonal
    # color (by segment count) and accept it as hatching.  This works whether
    # the hatching is gray, black, blue, or any color.
    hatching_lines = _identify_hatching(all_diagonals)

    logger.info(
        "Semantic filter: %d wall boundary candidates, %d hatching lines "
        "(from %d diagonal candidates)",
        len(wall_candidates), len(hatching_lines), len(all_diagonals),
    )
    return wall_candidates, hatching_lines


def _identify_hatching(diagonals: list[LineSegment]) -> list[LineSegment]:
    """Identify which diagonal lines are wall hatching vs text/noise.

    Hatching lines cluster by color and are spatially dense.  Text glyphs
    produce diagonals too, but in many different colors or scattered locations.
    We find the color(s) whose diagonals form dense spatial clusters.
    """
    if not diagonals:
        return []

    # Group diagonals by quantized color
    from collections import defaultdict
    color_groups: dict[tuple[int, ...], list[LineSegment]] = defaultdict(list)
    for seg in diagonals:
        # Quantize color to 10% buckets to group similar shades
        ckey = tuple(int(c * 10) for c in (seg.color or (0, 0, 0)))
        color_groups[ckey].append(seg)

    # Rank color groups by segment count.  Hatching is mass-produced by CAD —
    # it will be the dominant diagonal color by a wide margin.
    ranked = sorted(color_groups.items(), key=lambda kv: -len(kv[1]))

    hatching: list[LineSegment] = []
    for ckey, segs in ranked:
        # A color group must have a significant number of segments to be hatching
        # (text glyphs produce at most a few hundred; hatching produces thousands)
        if len(segs) < 200:
            break
        # Verify spatial density: hatching segments cluster tightly.
        # Check that the median segment length is short (hatching strokes are tiny).
        lengths = sorted(s.length for s in segs)
        median_len = lengths[len(lengths) // 2]
        if median_len < 2.0 or median_len > 15.0:
            # Too short = text glyph fragments; too long = not hatching
            continue
        hatching.extend(segs)
        color_sample = tuple(round(c, 2) for c in (segs[0].color or (0,)))
        logger.info(
            "Hatching detected: color≈%s, %d segments, median_length=%.1fpt",
            color_sample, len(segs), median_len,
        )

    return hatching


def _grayscale(color: tuple[float, ...] | None) -> float:
    """Convert color tuple to a single grayscale value [0, 1]."""
    if color is None:
        return 0.0
    if len(color) == 1:
        return color[0]
    if len(color) >= 3:
        return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    return color[0]


def _is_dark(color: tuple[float, ...] | None) -> bool:
    """Check if color is dark enough to be a structural line."""
    if color is None:
        return False
    return _grayscale(color) < 0.45


# ---------------------------------------------------------------------------
# Step 3a: Parallel-pair detection (hatching-independent)
# ---------------------------------------------------------------------------

def detect_parallel_pairs(
    wall_candidates: list[LineSegment],
    page_width: float = 0.0,
    page_height: float = 0.0,
) -> list[tuple[float, float, float, float]]:
    """Detect parallel H/V line pairs that form wall cross-sections.

    This is the primary, hatching-independent wall detection method.
    Returns a list of filled-rectangle regions (x0, y0, x1, y1) in PDF coords.
    """
    margin = PAIR_BORDER_MARGIN_PT

    # Separate into horizontal and vertical, filtering short lines and borders
    h_lines: list[tuple[float, float, float]] = []  # (min_var, max_var, fixed_axis)
    v_lines: list[tuple[float, float, float]] = []

    for seg in wall_candidates:
        if seg.length < PAIR_MIN_LINE_LEN_PT:
            continue
        # Skip lines at the page border (frame/title block lines)
        cx = (seg.x0 + seg.x1) / 2
        cy = (seg.y0 + seg.y1) / 2
        if page_width > 0 and (cx < margin or cx > page_width - margin):
            continue
        if page_height > 0 and (cy < margin or cy > page_height - margin):
            continue

        if seg.is_horizontal:
            h_lines.append((min(seg.x0, seg.x1), max(seg.x0, seg.x1),
                            (seg.y0 + seg.y1) / 2))
        elif seg.is_vertical:
            v_lines.append((min(seg.y0, seg.y1), max(seg.y0, seg.y1),
                            (seg.x0 + seg.x1) / 2))

    raw_rects: list[tuple[float, float, float, float, str]] = []  # + axis tag

    # Find horizontal parallel pairs (same X-run, different Y)
    h_lines.sort(key=lambda l: l[2])  # sort by Y
    for r in _find_pairs(h_lines, axis="h"):
        raw_rects.append((*r, "h"))

    # Find vertical parallel pairs (same Y-run, different X)
    v_lines.sort(key=lambda l: l[2])  # sort by X
    for r in _find_pairs(v_lines, axis="v"):
        raw_rects.append((*r, "v"))

    logger.info("Parallel-pair raw candidates: %d", len(raw_rects))

    # ---- Junction connectivity filter ----
    # Walls connect to other walls at T/L-junctions.  Border/dimension pairs
    # are isolated.  Keep only pairs that connect perpendicularly to at least
    # one other pair (their endpoint region overlaps another pair's body).
    connected = _filter_by_junctions(raw_rects)

    logger.info("Parallel-pair after junction filter: %d wall rectangles", len(connected))
    return connected


def _rects_connect(
    r1: tuple[float, float, float, float],
    r2: tuple[float, float, float, float],
    tol: float = 5.0,
) -> bool:
    """Check if two rectangles touch or overlap within tolerance."""
    x0a, y0a, x1a, y1a = r1
    x0b, y0b, x1b, y1b = r2
    return (x0a - tol <= x1b and x1a + tol >= x0b and
            y0a - tol <= y1b and y1a + tol >= y0b)


def _filter_by_junctions(
    rects: list[tuple[float, float, float, float, str]],
) -> list[tuple[float, float, float, float]]:
    """Keep only pair-rectangles that connect perpendicularly to another pair.

    A horizontal pair must touch at least one vertical pair (and vice versa)
    to be considered part of the wall network.
    """
    h_rects = [(r[0], r[1], r[2], r[3]) for r in rects if r[4] == "h"]
    v_rects = [(r[0], r[1], r[2], r[3]) for r in rects if r[4] == "v"]

    # For each H rect, check if any V rect connects to it
    connected_h: set[int] = set()
    connected_v: set[int] = set()

    for i, hr in enumerate(h_rects):
        for j, vr in enumerate(v_rects):
            if _rects_connect(hr, vr):
                connected_h.add(i)
                connected_v.add(j)

    result = [h_rects[i] for i in connected_h]
    result.extend(v_rects[j] for j in connected_v)
    return result


def _find_pairs(
    lines: list[tuple[float, float, float]],
    axis: str,
) -> list[tuple[float, float, float, float]]:
    """Find nearest-neighbor parallel line pairs with wall-like gap.

    For each line, only pair it with its closest parallel neighbor (not every
    line in range).  This prevents border lines from pairing with distant
    wall lines, and dimension lines from cross-pairing with walls.
    """
    rects: list[tuple[float, float, float, float]] = []
    used: set[int] = set()  # track already-paired lines

    # Lines are sorted by fixed-axis coordinate
    for i in range(len(lines)):
        if i in used:
            continue
        lo_i, hi_i, fixed_i = lines[i]

        # Find the nearest neighbor in the fixed-axis direction
        best_j: int | None = None
        best_gap = float("inf")

        for j in range(i + 1, len(lines)):
            if j in used:
                continue
            lo_j, hi_j, fixed_j = lines[j]
            gap = fixed_j - fixed_i  # lines sorted, so always >= 0
            if gap < PAIR_GAP_MIN_PT:
                continue
            if gap > PAIR_GAP_MAX_PT:
                break

            # Check overlap
            overlap = min(hi_i, hi_j) - max(lo_i, lo_j)
            if overlap < PAIR_OVERLAP_MIN_PT:
                continue

            if gap < best_gap:
                best_gap = gap
                best_j = j

        if best_j is not None:
            lo_j, hi_j, fixed_j = lines[best_j]
            used.add(i)
            used.add(best_j)

            overlap_lo = max(lo_i, lo_j)
            overlap_hi = min(hi_i, hi_j)
            f_lo = min(fixed_i, fixed_j)
            f_hi = max(fixed_i, fixed_j)
            if axis == "h":
                rects.append((overlap_lo, f_lo, overlap_hi, f_hi))
            else:
                rects.append((f_lo, overlap_lo, f_hi, overlap_hi))

    return rects


# ---------------------------------------------------------------------------
# Step 3b: Rasterization & Morphology
# ---------------------------------------------------------------------------

def rasterize_and_extract_contours(
    wall_candidates: list[LineSegment],
    hatching_lines: list[LineSegment],
    page_width: float,
    page_height: float,
    debug_dir: Optional[str] = None,
) -> list[np.ndarray]:
    """Build wall regions from two independent signals and extract contours.

    Signal A — Hatching fill (when present):
      Render hatching, close into solid bands. This already IS the wall footprint.
      Just lightly thicken to match wall width. No bloated zone/masking chain.

    Signal B — Parallel-pair detection (always works):
      Find pairs of parallel H/V lines with wall-like gap. Fill the rectangle
      between each pair. Works on plans with no hatching at all.

    The two signals are unioned, then contours are extracted.
    """
    scale = RENDER_SCALE
    cw = int(page_width * scale)
    ch = int(page_height * scale)

    # ---- Signal A: Hatching fill ----
    canvas_hatch = np.zeros((ch, cw), dtype=np.uint8)
    if hatching_lines:
        for seg in hatching_lines:
            pt1 = (int(seg.x0 * scale), int(seg.y0 * scale))
            pt2 = (int(seg.x1 * scale), int(seg.y1 * scale))
            cv2.line(canvas_hatch, pt1, pt2, 255, HATCH_RENDER_THICKNESS_PX)

        # Close gaps between adjacent hatch strokes to form solid bands
        k_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (HATCH_CLOSE_KERNEL, HATCH_CLOSE_KERNEL),
        )
        hatch_filled = cv2.morphologyEx(canvas_hatch, cv2.MORPH_CLOSE, k_close, iterations=2)

        # Light thickening so the band matches actual wall width
        if MORPH_WALL_THICKEN > 0:
            k_thick = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (MORPH_WALL_THICKEN * 2 + 1, MORPH_WALL_THICKEN * 2 + 1),
            )
            hatch_filled = cv2.dilate(hatch_filled, k_thick, iterations=1)
    else:
        hatch_filled = canvas_hatch

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "debug_01_hatching_raw.png"), canvas_hatch)
        cv2.imwrite(os.path.join(debug_dir, "debug_02_hatch_filled.png"), hatch_filled)

    # ---- Signal B: Parallel-pair rectangles ----
    pair_rects = detect_parallel_pairs(wall_candidates, page_width, page_height)
    canvas_pairs = np.zeros((ch, cw), dtype=np.uint8)
    for x0, y0, x1, y1 in pair_rects:
        cv2.rectangle(
            canvas_pairs,
            (int(x0 * scale), int(y0 * scale)),
            (int(x1 * scale), int(y1 * scale)),
            255, -1,  # filled
        )

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "debug_03_parallel_pairs_raw.png"), canvas_pairs)

    # ---- Combine signals intelligently ----
    has_hatching = np.count_nonzero(hatch_filled) > 0

    if has_hatching:
        # Hatching is the trusted primary signal — use it directly.
        # Only accept parallel-pair rects that OVERLAP with hatching regions
        # (these reinforce/extend hatching, e.g. thick exterior walls).
        # Reject pairs that don't touch any hatching (border, dimension, title).
        hatch_proximity = cv2.dilate(
            hatch_filled,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)),
            iterations=2,
        )
        pairs_near_hatch = cv2.bitwise_and(canvas_pairs, hatch_proximity)
        combined = cv2.bitwise_or(hatch_filled, pairs_near_hatch)
        logger.info("Hatching present — pairs filtered by hatching proximity")
    else:
        # No hatching found — fall back entirely to parallel pairs
        combined = canvas_pairs
        logger.info("No hatching found — using parallel pairs only")

    # Light smoothing to clean jagged edges
    k_smooth = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_SMOOTH_KERNEL, MORPH_SMOOTH_KERNEL),
    )
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_smooth, iterations=1)

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "debug_04_combined_final.png"), combined)

    # ---- Contour extraction ----
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logger.info("Extracted %d raw contours from morphology", len(contours))
    return list(contours)


# ---------------------------------------------------------------------------
# Step 3b: Contour filtering
# ---------------------------------------------------------------------------

def filter_contours(
    contours: list[np.ndarray],
    page_width: float,
    page_height: float,
) -> list[WallContour]:
    """Filter contours by area, aspect ratio, and position. Convert to PDF coords."""
    scale = RENDER_SCALE
    results: list[WallContour] = []

    min_area_px = MIN_CONTOUR_AREA_PT2 * (scale ** 2)
    min_perim_px = MIN_CONTOUR_LENGTH_PT * scale

    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < min_area_px:
            continue

        perim_px = cv2.arcLength(cnt, closed=True)
        if perim_px < min_perim_px:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / min(w, h)
        if aspect > MAX_CONTOUR_ASPECT:
            continue

        # Convert contour to PDF coordinates
        poly_pdf = cnt.reshape(-1, 2).astype(np.float64) / scale
        bbox_pdf = (x / scale, y / scale, (x + w) / scale, (y + h) / scale)
        area_pdf = area_px / (scale ** 2)

        results.append(WallContour(
            polygon_pdf=poly_pdf,
            bbox_pdf=bbox_pdf,
            area_pdf=area_pdf,
        ))

    logger.info("Contour filter: %d → %d wall regions", len(contours), len(results))
    return results


# ---------------------------------------------------------------------------
# Step 4: Micro-VLM Validation
# ---------------------------------------------------------------------------

def validate_with_vlm(
    wall_contours: list[WallContour],
    page: fitz.Page,
    api_key: Optional[str] = None,
) -> list[WallContour]:
    """Send each contour crop to Gemini for WALL/NOISE binary classification.

    Requires GOOGLE_API_KEY or GEMINI_API_KEY env var (or api_key arg).
    Falls back to accepting all contours if the API is unavailable.
    Uses the modern ``google.genai`` SDK.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        logger.warning("No Gemini API key found — skipping VLM validation (accepting all contours)")
        for wc in wall_contours:
            wc.validated = True
        return wall_contours

    try:
        from google import genai
        from google.genai import types as genai_types
        from PIL import Image
        import io

        client = genai.Client(api_key=key)
    except Exception as e:
        logger.warning("Failed to initialize Gemini client: %s — skipping VLM validation", e)
        for wc in wall_contours:
            wc.validated = True
        return wall_contours

    prompt = (
        "You are a construction estimator reviewing a cropped section of an "
        "architectural floor plan. Does this image crop primarily show a "
        "structural wall or partition (including hatched or filled wall sections), "
        "or is it a dimension line, cabinetry, fixture, appliance, text, "
        "or other noise? Reply ONLY with the single word 'WALL' or 'NOISE'."
    )

    validated: list[WallContour] = []

    for i, wc in enumerate(wall_contours):
        bx0, by0, bx1, by1 = wc.bbox_pdf
        # Pad the crop
        clip = fitz.Rect(
            bx0 - VLM_CROP_PAD_PT,
            by0 - VLM_CROP_PAD_PT,
            bx1 + VLM_CROP_PAD_PT,
            by1 + VLM_CROP_PAD_PT,
        )
        # Render at 2× for VLM clarity
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        img_bytes = pix.tobytes("png")

        label = _call_vlm(client, prompt, img_bytes)
        wc.validated = (label == "WALL")

        logger.info(
            "VLM contour %d/%d (area=%.0f): %s",
            i + 1, len(wall_contours), wc.area_pdf, label,
        )

        # Rate-limit courtesy delay (free tier: 10 req/min)
        if i < len(wall_contours) - 1:
            import time
            time.sleep(1.0)

        if wc.validated:
            validated.append(wc)

    logger.info(
        "VLM validation: %d/%d classified as WALL",
        len(validated), len(wall_contours),
    )
    return validated


def _call_vlm(client: object, prompt: str, img_bytes: bytes) -> str:
    """Call Gemini with an image and return 'WALL' or 'NOISE'."""
    from PIL import Image
    import io
    import time

    img = Image.open(io.BytesIO(img_bytes))

    for attempt in range(VLM_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(  # type: ignore[union-attr]
                model=VLM_MODEL,
                contents=[prompt, img],
            )
            text = response.text.strip().upper()
            if "WALL" in text:
                return "WALL"
            return "NOISE"
        except Exception as e:
            if attempt < VLM_MAX_RETRIES:
                logger.debug("VLM call attempt %d failed: %s — retrying", attempt + 1, e)
                time.sleep(2.0 * (attempt + 1))
            else:
                logger.warning("VLM call failed after retries: %s — defaulting to WALL", e)
                return "WALL"
    return "WALL"


# ---------------------------------------------------------------------------
# Step 5: Output Generation
# ---------------------------------------------------------------------------

def highlight_walls_on_pdf(
    input_path: str,
    output_path: str,
    wall_contours: list[WallContour],
    page_index: int = 0,
) -> None:
    """Draw semi-transparent highlights over validated wall contours on the PDF."""
    doc = fitz.open(input_path)
    page = doc[page_index]

    r, g, b = HIGHLIGHT_COLOR
    fill_color = (r, g, b)

    for wc in wall_contours:
        if wc.validated is False:
            continue

        poly = wc.polygon_pdf
        if len(poly) < 3:
            continue

        # Build a fitz shape and draw the filled polygon
        shape = page.new_shape()
        shape.draw_polyline([fitz.Point(pt[0], pt[1]) for pt in poly])
        shape.finish(
            fill=fill_color,
            fill_opacity=HIGHLIGHT_OPACITY,
            color=None,  # No outline stroke
            closePath=True,
        )
        shape.commit()

    doc.save(output_path)
    doc.close()
    logger.info("Saved highlighted PDF to %s (%d wall regions)", output_path, len(wall_contours))


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_pdf: str,
    output_pdf: Optional[str] = None,
    use_vlm: bool = True,
    debug: bool = False,
    page_index: int = 0,
) -> list[WallContour]:
    """Execute the full wall-detection pipeline.

    Args:
        input_pdf:  Path to the input PDF floor plan.
        output_pdf: Path for the highlighted output PDF. Defaults to <input>_walls.pdf.
        use_vlm:    Whether to run Gemini VLM validation on detected regions.
        debug:      Whether to save intermediate debug images.
        page_index: Which page to process (0-indexed).

    Returns:
        List of detected WallContour objects.
    """
    input_path = Path(input_pdf)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

    if output_pdf is None:
        output_pdf = str(input_path.with_name(input_path.stem + "_walls.pdf"))

    debug_dir: Optional[str] = None
    if debug:
        debug_dir = str(input_path.parent / "debug_output")
        os.makedirs(debug_dir, exist_ok=True)
        logger.info("Debug images will be saved to %s", debug_dir)

    # --- Open PDF ---
    doc = fitz.open(input_pdf)
    if page_index >= len(doc):
        raise ValueError(f"Page index {page_index} out of range (document has {len(doc)} pages)")
    page = doc[page_index]

    page_w = page.rect.width
    page_h = page.rect.height
    logger.info(
        "Processing page %d: %.0f × %.0f pt, rotation=%d°",
        page_index, page_w, page_h, page.rotation,
    )

    # --- Step 1: Extract vectors ---
    segments, text_blocks = extract_vectors(page)

    # Detect and correct rotation
    rotation_correction = detect_rotation_angle(segments)
    if abs(rotation_correction) > 0.01:
        cx, cy = page_w / 2, page_h / 2
        segments = rotate_segments(segments, rotation_correction, cx, cy)
        logger.info("Applied rotation correction: %.1f°", rotation_correction)

    # Collinear de-fragmentation
    segments = merge_collinear_segments(segments)

    # --- Step 2: Semantic filtering ---
    wall_candidates, hatching_lines = filter_segments(segments, text_blocks)

    if not hatching_lines and not wall_candidates:
        logger.warning("No wall candidates or hatching found — trying broader filters")
        # Fallback: accept all dark axis-aligned lines with reasonable width
        wall_candidates = [
            s for s in segments
            if s.is_axis_aligned
            and _is_dark(s.color)
            and s.length >= MIN_LINE_LENGTH_PT
            and WALL_WIDTH_RANGE[0] <= s.width <= WALL_WIDTH_RANGE[1]
        ]
        # And treat all diagonal dark lines as hatching
        hatching_lines = [
            s for s in segments
            if s.is_diagonal_hatching and _is_dark(s.color) and s.length >= MIN_LINE_LENGTH_PT
        ]
        logger.info("Fallback: %d wall candidates, %d hatching", len(wall_candidates), len(hatching_lines))

    # --- Step 3: Rasterize & morphology ---
    raw_contours = rasterize_and_extract_contours(
        wall_candidates, hatching_lines,
        page_w, page_h,
        debug_dir=debug_dir,
    )

    wall_contours = filter_contours(raw_contours, page_w, page_h)

    if debug_dir:
        _save_debug_contours(wall_contours, page_w, page_h, debug_dir)

    # --- Step 4: VLM validation ---
    if use_vlm and wall_contours:
        wall_contours = validate_with_vlm(wall_contours, page)
    else:
        for wc in wall_contours:
            wc.validated = True

    doc.close()

    # --- Step 5: Output ---
    if wall_contours:
        highlight_walls_on_pdf(input_pdf, output_pdf, wall_contours, page_index)
    else:
        logger.warning("No wall contours detected — output PDF not generated")

    return wall_contours


def _save_debug_contours(
    contours: list[WallContour],
    page_w: float,
    page_h: float,
    debug_dir: str,
) -> None:
    """Save a debug visualization of filtered contours."""
    scale = RENDER_SCALE
    canvas = np.zeros((int(page_h * scale), int(page_w * scale), 3), dtype=np.uint8)

    for i, wc in enumerate(contours):
        pts = (wc.polygon_pdf * scale).astype(np.int32)
        color = (
            int(50 + (i * 47) % 206),
            int(50 + (i * 83) % 206),
            int(50 + (i * 131) % 206),
        )
        cv2.drawContours(canvas, [pts], -1, color, 2)
        # Label with index
        cx = int(wc.bbox_pdf[0] * scale)
        cy = int(wc.bbox_pdf[1] * scale)
        cv2.putText(canvas, str(i), (cx, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(os.path.join(debug_dir, "debug_05_contours.png"), canvas)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and highlight structural walls in a PDF floor plan.",
    )
    parser.add_argument("input_pdf", help="Path to the input PDF floor plan")
    parser.add_argument("output_pdf", nargs="?", default=None,
                        help="Path for the output highlighted PDF (default: <input>_walls.pdf)")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Skip Gemini VLM validation")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate debug images")
    parser.add_argument("--page", type=int, default=0,
                        help="Page index to process (0-based, default: 0)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    contours = run_pipeline(
        input_pdf=args.input_pdf,
        output_pdf=args.output_pdf,
        use_vlm=not args.no_vlm,
        debug=args.debug,
        page_index=args.page,
    )

    print(f"\nDetected {len(contours)} wall region(s).")
    if args.output_pdf:
        print(f"Output saved to: {args.output_pdf}")
    else:
        stem = Path(args.input_pdf).stem
        print(f"Output saved to: {stem}_walls.pdf")


if __name__ == "__main__":
    main()
