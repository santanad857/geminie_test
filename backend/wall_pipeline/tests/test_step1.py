"""
Test suite for Step 1: Exhaustive Pattern Extraction
=====================================================

Validates data conservation, primitive correctness, and quantization
integrity across all example PDFs.
"""

import math
import os
import re
import sys

# Allow importing from the pipeline directory (parent of tests/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from step1_extract import (
    COLOR_DECIMALS,
    WIDTH_DECIMALS,
    compute_raw_total_length,
    explode_path,
    extract_patterns,
    make_fill_key,
    make_stroke_key,
    polygon_perimeter,
    primitive_length,
    quantize_color,
    quantize_width,
    segment_length,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "example_plans")
EXAMPLE_PDFS = sorted(
    os.path.join(EXAMPLE_DIR, f)
    for f in os.listdir(EXAMPLE_DIR)
    if f.endswith(".pdf")
)


@pytest.fixture(params=EXAMPLE_PDFS, ids=[os.path.basename(p) for p in EXAMPLE_PDFS])
def extraction(request):
    """Run extract_patterns on each example PDF (page 0)."""
    return extract_patterns(request.param, page_number=0)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Length Conservation
# ═══════════════════════════════════════════════════════════════════════════

class TestLengthConservation:
    """
    The total cumulative length of all raw drawing items must equal the
    total length of all extracted primitives, within floating-point
    tolerance.  Data cannot be gained or lost.
    """

    TOLERANCE = 1e-4  # points

    def test_total_length_matches(self, extraction):
        raw_drawings = extraction["raw_drawings"]
        grouped = extraction["grouped"]

        raw_total = compute_raw_total_length(raw_drawings)
        output_total = sum(
            primitive_length(p)
            for prims in grouped.values()
            for p in prims
        )

        assert abs(raw_total - output_total) < self.TOLERANCE, (
            f"Conservation failed: raw={raw_total:.6f}, "
            f"output={output_total:.6f}, diff={abs(raw_total - output_total):.8f}"
        )

    def test_primitive_count_ge_raw_items(self, extraction):
        """
        Primitive count must be >= raw item count because rects in
        stroke-only paths explode into 4 lines each.
        """
        raw_item_count = sum(
            len(p["items"]) for p in extraction["raw_drawings"]
        )
        output_count = sum(
            len(prims) for prims in extraction["grouped"].values()
        )
        assert output_count >= raw_item_count, (
            f"Fewer output primitives ({output_count}) than raw items ({raw_item_count})"
        )

    def test_no_empty_groups(self, extraction):
        """Every style group must contain at least one primitive."""
        for key, prims in extraction["grouped"].items():
            assert len(prims) > 0, f"Empty group: {key}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Explode Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestExplodeValidation:
    """
    Verify that path explosion produces well-formed atomic primitives.
    """

    def test_line_has_exactly_two_points(self, extraction):
        for key, prims in extraction["grouped"].items():
            for p in prims:
                if p["type"] == "line":
                    assert len(p["points"]) == 2, (
                        f"Line in '{key}' has {len(p['points'])} points (expected 2)"
                    )

    def test_polygon_has_at_least_three_points(self, extraction):
        for key, prims in extraction["grouped"].items():
            for p in prims:
                if p["type"] == "polygon":
                    assert len(p["points"]) >= 3, (
                        f"Polygon in '{key}' has {len(p['points'])} points (expected >= 3)"
                    )

    def test_polygon_not_disjoint(self, extraction):
        """
        Every polygon must form a single connected ring: non-zero
        perimeter and no consecutive duplicate vertices.
        """
        for key, prims in extraction["grouped"].items():
            for p in prims:
                if p["type"] != "polygon":
                    continue
                pts = p["points"]
                perim = polygon_perimeter(pts)
                assert perim > 0, f"Degenerate polygon (zero perimeter) in '{key}'"
                # No consecutive duplicate points
                for i in range(len(pts)):
                    j = (i + 1) % len(pts)
                    dist = segment_length(pts[i], pts[j])
                    assert dist > 0, (
                        f"Polygon in '{key}' has duplicate consecutive "
                        f"vertices at index {i},{j}: {pts[i]}"
                    )

    def test_curve_has_four_control_points(self, extraction):
        for key, prims in extraction["grouped"].items():
            for p in prims:
                if p["type"] == "curve":
                    assert len(p["points"]) == 4, (
                        f"Curve in '{key}' has {len(p['points'])} points (expected 4)"
                    )

    def test_every_primitive_has_path_index(self, extraction):
        for key, prims in extraction["grouped"].items():
            for p in prims:
                assert "original_path_index" in p, (
                    f"Missing original_path_index in '{key}'"
                )
                assert isinstance(p["original_path_index"], int)

    def test_primitive_types_are_valid(self, extraction):
        valid = {"line", "polygon", "curve"}
        for key, prims in extraction["grouped"].items():
            for p in prims:
                assert p["type"] in valid, (
                    f"Invalid type '{p['type']}' in '{key}'"
                )

    def test_all_points_are_finite(self, extraction):
        for key, prims in extraction["grouped"].items():
            for p in prims:
                for pt in p["points"]:
                    assert len(pt) == 2, f"Point has {len(pt)} coords in '{key}'"
                    assert math.isfinite(pt[0]) and math.isfinite(pt[1]), (
                        f"Non-finite point {pt} in '{key}'"
                    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Quantization Checks
# ═══════════════════════════════════════════════════════════════════════════

class TestQuantization:
    """
    Validate that style keys use properly quantized values so that
    near-identical CAD styles are merged, not fractured.
    """

    def _extract_widths(self, grouped: dict) -> list[float]:
        """Parse all stroke widths from style keys."""
        widths = set()
        for key in grouped:
            # Match patterns like width_1.00 or width_0.12
            for m in re.finditer(r"width_([\d.]+)", key):
                widths.add(float(m.group(1)))
        return sorted(widths)

    def test_no_near_duplicate_widths(self, extraction):
        """
        With WIDTH_DECIMALS=2, the minimum gap between two distinct
        quantized widths is 0.01.  No two widths should be closer.
        """
        widths = self._extract_widths(extraction["grouped"])
        min_gap = 10 ** -WIDTH_DECIMALS  # 0.01 for decimals=2
        for i in range(len(widths) - 1):
            diff = widths[i + 1] - widths[i]
            assert diff >= min_gap - 1e-9, (
                f"Widths {widths[i]} and {widths[i+1]} differ by only "
                f"{diff:.6f} (min gap {min_gap})"
            )

    def test_width_values_are_rounded(self, extraction):
        """Every width in a style key must match WIDTH_DECIMALS precision."""
        widths = self._extract_widths(extraction["grouped"])
        for w in widths:
            assert w == round(w, WIDTH_DECIMALS), (
                f"Width {w} is not rounded to {WIDTH_DECIMALS} decimals"
            )

    def _extract_colors(self, grouped: dict) -> list[tuple]:
        """Parse all RGB tuples from style keys."""
        colors = set()
        for key in grouped:
            for m in re.finditer(r"rgb\(([\d.]+),([\d.]+),([\d.]+)\)", key):
                colors.add((float(m.group(1)), float(m.group(2)), float(m.group(3))))
        return sorted(colors)

    def test_color_values_are_rounded(self, extraction):
        """Every color component must match COLOR_DECIMALS precision."""
        colors = self._extract_colors(extraction["grouped"])
        assert len(colors) > 0, "No colors parsed from style keys"
        for rgb in colors:
            for c in rgb:
                assert c == round(c, COLOR_DECIMALS), (
                    f"Color component {c} not rounded to {COLOR_DECIMALS} decimals"
                )


# ═══════════════════════════════════════════════════════════════════════════
# 4. Unit tests for helper functions
# ═══════════════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_quantize_color_none(self):
        assert quantize_color(None) is None

    def test_quantize_color_rounds(self):
        assert quantize_color((0.12345, 0.99999, 0.50001)) == (0.12, 1.0, 0.5)

    def test_quantize_width_rounds(self):
        assert quantize_width(1.0001) == 1.0
        assert quantize_width(0.9998) == 1.0
        assert quantize_width(0.005) == 0.01  # rounds up

    def test_make_stroke_key(self):
        key = make_stroke_key((0.0, 0.0, 0.0), 1.0)
        assert key == "stroke_rgb(0.0,0.0,0.0)_width_1.00"

    def test_make_fill_key(self):
        key = make_fill_key((0.8, 0.8, 0.8))
        assert key == "fill_rgb(0.8,0.8,0.8)"

    def test_segment_length(self):
        assert segment_length([0, 0], [3, 4]) == 5.0

    def test_polygon_perimeter_square(self):
        square = [[0, 0], [10, 0], [10, 10], [0, 10]]
        assert abs(polygon_perimeter(square) - 40.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════
# 5. Stroke / Fill Decoupling
# ═══════════════════════════════════════════════════════════════════════════

class TestStrokeFillDecoupling:
    """
    Verify that no style key mixes stroke and fill properties, and that
    stroke-only buckets contain only lines/curves while fill-only buckets
    contain polygons (and boundary lines from fill-only paths).
    """

    def test_no_compound_keys(self, extraction):
        """Every key must be purely stroke_... or fill_..., never both."""
        for key in extraction["grouped"]:
            has_stroke = key.startswith("stroke_")
            has_fill = key.startswith("fill_")
            assert has_stroke or has_fill, f"Unexpected key format: {key}"
            assert not (has_stroke and "fill_" in key), (
                f"Compound stroke+fill key found: {key}"
            )

    def test_stroke_buckets_have_no_polygons(self, extraction):
        """Stroke buckets should only contain lines and curves."""
        for key, prims in extraction["grouped"].items():
            if not key.startswith("stroke_"):
                continue
            for p in prims:
                assert p["type"] in ("line", "curve"), (
                    f"Stroke bucket '{key}' contains a {p['type']}"
                )

    def test_fill_buckets_have_no_mixed_rendering(self, extraction):
        """
        Fill buckets may contain polygons and boundary lines/curves,
        but every polygon must have >= 3 points (basic sanity).
        """
        for key, prims in extraction["grouped"].items():
            if not key.startswith("fill_"):
                continue
            for p in prims:
                if p["type"] == "polygon":
                    assert len(p["points"]) >= 3
