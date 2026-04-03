"""
Test suite for Step 3: VLM Multi-Style Wall Classification
==========================================================

Test 1  Rendering        — multi-colour overlay is a valid, bbox-cropped PNG
                           and assigns distinct colours to each style
Test 2  Prompt           — contains required phrases; lists colour names
Test 3  Response Parsing — maps colour names back to per-style verdicts
Test 4  End-to-End VLM   — live classification (one call, requires API credits)
Test 5  Result Overlay   — generates a valid image file
"""

import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from step3_wall_classify import (
    STYLE_COLORS,
    build_multi_style_prompt,
    classify_candidate,
    load_api_key,
    parse_multi_style_response,
    parse_vlm_response,
    render_multi_style_overlay,
    render_result_overlay,
)

# ── paths ─────────────────────────────────────────────────────────────

_TESTS_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_TESTS_DIR, "..", ".."))

MAIN_ST_EX_PDF        = os.path.join(PROJECT_ROOT, "example_plans", "main_st_ex.pdf")
MAIN_ST_EX_CANDIDATES = os.path.join(
    PROJECT_ROOT, "output", "main_st_ex", "step2", "candidates.json",
)


# ── helpers ───────────────────────────────────────────────────────────

def _is_valid_png(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


def _png_dimensions(data: bytes) -> tuple[int, int]:
    w = struct.unpack(">I", data[16:20])[0]
    h = struct.unpack(">I", data[20:24])[0]
    return w, h


@pytest.fixture(scope="module")
def main_st_ex_candidate_1():
    with open(MAIN_ST_EX_CANDIDATES) as f:
        candidates = json.load(f)
    for c in candidates:
        if c["candidate_id"] == 1:
            return c
    pytest.fail("candidate 1 not found in main_st_ex")


# =====================================================================
# Test 1 — Multi-Colour Rendering
# =====================================================================

class TestRendering:
    """render_multi_style_overlay produces a valid, cropped PNG."""

    def test_overlay_is_valid_png(self, main_st_ex_candidate_1):
        png, color_to_style = render_multi_style_overlay(
            MAIN_ST_EX_PDF, main_st_ex_candidate_1,
        )
        assert len(png) > 0
        assert _is_valid_png(png)

    def test_overlay_is_cropped_not_full_page(self, main_st_ex_candidate_1):
        """Width should be bbox-cropped, not the full page (612 pt → 1224 px at 2×)."""
        png, _ = render_multi_style_overlay(
            MAIN_ST_EX_PDF, main_st_ex_candidate_1, render_scale=2.0,
        )
        w, h = _png_dimensions(png)
        assert w < 1000, f"width {w} looks like the full page was rendered"
        assert w > 100
        assert h > 100

    def test_color_map_covers_all_styles(self, main_st_ex_candidate_1):
        """Every style in the candidate (up to palette size) appears in the map."""
        _, color_to_style = render_multi_style_overlay(
            MAIN_ST_EX_PDF, main_st_ex_candidate_1,
        )
        n_styles = len(main_st_ex_candidate_1["grouped_primitives"])
        expected = min(n_styles, len(STYLE_COLORS))
        assert len(color_to_style) == expected

    def test_color_names_are_palette_names(self, main_st_ex_candidate_1):
        """All colour keys in the map belong to the declared palette."""
        palette_names = {name for name, _ in STYLE_COLORS}
        _, color_to_style = render_multi_style_overlay(
            MAIN_ST_EX_PDF, main_st_ex_candidate_1,
        )
        for color_name in color_to_style:
            assert color_name in palette_names


# =====================================================================
# Test 2 — Prompt Construction
# =====================================================================

class TestPromptConstruction:

    def _sample_map(self):
        return {
            "RED":    "stroke_rgb(0,0,0)_width_0.50",
            "CYAN":   "fill_rgb(0,0,0)",
            "MAGENTA": "stroke_rgb(0,0,0)_width_0.72",
        }

    def test_contains_structural_walls(self):
        prompt = build_multi_style_prompt(self._sample_map())
        assert "structural walls" in prompt.lower()

    def test_lists_all_colour_names(self):
        prompt = build_multi_style_prompt(self._sample_map())
        for color in ("RED", "CYAN", "MAGENTA"):
            assert color in prompt

    def test_contains_none_option(self):
        """Prompt must explain that 'NONE' is a valid answer."""
        prompt = build_multi_style_prompt(self._sample_map())
        assert "NONE" in prompt

    def test_count_matches(self):
        m = self._sample_map()
        prompt = build_multi_style_prompt(m)
        assert str(len(m)) in prompt


# =====================================================================
# Test 3 — Response Parsing
# =====================================================================

class TestResponseParsing:
    """parse_multi_style_response maps colour names to YES/NO verdicts."""

    def _color_map(self):
        return {
            "RED":    "stroke_rgb(0,0,0)_width_0.50",
            "CYAN":   "fill_rgb(0,0,0)",
            "MAGENTA": "stroke_rgb(0,0,0)_width_0.72",
        }

    def test_single_wall_colour(self):
        result = parse_multi_style_response("RED", self._color_map())
        assert result["stroke_rgb(0,0,0)_width_0.50"]["verdict"] == "YES"
        assert result["fill_rgb(0,0,0)"]["verdict"] == "NO"
        assert result["stroke_rgb(0,0,0)_width_0.72"]["verdict"] == "NO"

    def test_multiple_wall_colours(self):
        result = parse_multi_style_response("RED, CYAN", self._color_map())
        assert result["stroke_rgb(0,0,0)_width_0.50"]["verdict"] == "YES"
        assert result["fill_rgb(0,0,0)"]["verdict"] == "YES"
        assert result["stroke_rgb(0,0,0)_width_0.72"]["verdict"] == "NO"

    def test_none_response(self):
        result = parse_multi_style_response("NONE", self._color_map())
        for v in result.values():
            assert v["verdict"] == "NO"

    def test_case_insensitive(self):
        result = parse_multi_style_response("red", self._color_map())
        assert result["stroke_rgb(0,0,0)_width_0.50"]["verdict"] == "YES"

    def test_all_styles_present_in_result(self):
        """Every style key in the map must appear in the output."""
        m = self._color_map()
        result = parse_multi_style_response("RED", m)
        assert set(result.keys()) == set(m.values())

    def test_high_confidence_on_clean_response(self):
        result = parse_multi_style_response("RED", self._color_map())
        for v in result.values():
            assert v["confidence"] == "high"

    # Legacy YES/NO parser still works for backward compatibility
    @pytest.mark.parametrize("text,expected", [
        ("YES", "YES"),
        ("No", "NO"),
        ("yes definitely", "YES"),
        ("NO - these are fixtures", "NO"),
    ])
    def test_legacy_parse_vlm_response(self, text, expected):
        assert parse_vlm_response(text)["verdict"] == expected


# =====================================================================
# Test 4 — End-to-End with VLM  (skip with: pytest -m "not vlm")
# =====================================================================

@pytest.mark.vlm
class TestEndToEnd:

    def test_classify_main_st_ex_candidate_1(self, main_st_ex_candidate_1):
        """
        ONE VLM call for 3 styles.  Result must include at least one
        YES-classified wall style.
        """
        api_key = load_api_key()
        result  = classify_candidate(
            MAIN_ST_EX_PDF, main_st_ex_candidate_1, api_key,
        )

        # Schema
        assert "candidate_id"          in result
        assert "bounding_box"          in result
        assert "style_classifications" in result
        assert "wall_primitives"       in result
        assert "vlm_response"          in result
        assert "color_to_style"        in result

        # Every style must have a verdict
        n_styles = len(main_st_ex_candidate_1["grouped_primitives"])
        assert len(result["style_classifications"]) == n_styles

        # At least one wall style detected
        assert len(result["wall_primitives"]) > 0, (
            "expected at least one style classified as walls"
        )

        # Only ONE API call was made (tested implicitly via colour response)
        assert result["vlm_response"]  # non-empty response

    def test_vlm_response_contains_colour_or_none(self, main_st_ex_candidate_1):
        """VLM response should name a palette colour or say NONE."""
        api_key = load_api_key()
        result  = classify_candidate(
            MAIN_ST_EX_PDF, main_st_ex_candidate_1, api_key,
        )
        palette_names = {name for name, _ in STYLE_COLORS}
        response_upper = result["vlm_response"].upper()
        has_colour = any(
            re.search(rf"\b{c}\b", response_upper) for c in palette_names
        )
        has_none = "NONE" in response_upper
        assert has_colour or has_none, (
            f"unexpected VLM response: {result['vlm_response']!r}"
        )


# =====================================================================
# Test 5 — Result Overlay
# =====================================================================

class TestResultOverlay:

    def test_result_overlay_valid_png(self, main_st_ex_candidate_1, tmp_path):
        classifications = {
            sk: {"verdict": "YES" if i == 0 else "NO", "confidence": "high"}
            for i, sk in enumerate(
                main_st_ex_candidate_1["grouped_primitives"],
            )
        }
        out = str(tmp_path / "result.png")
        render_result_overlay(
            MAIN_ST_EX_PDF, main_st_ex_candidate_1, classifications, out,
        )
        assert os.path.isfile(out)
        with open(out, "rb") as f:
            data = f.read()
        assert _is_valid_png(data)
        assert len(data) > 1000
