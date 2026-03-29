"""
Test suite for Step 3: VLM Style-Layer Classification
=====================================================

Test 1  Rendering — overlay is a valid, bbox-cropped PNG
Test 2  Prompt Construction — contains required phrases
Test 3  Response Parsing — extracts YES/NO from varied formats
Test 4  End-to-End VLM — live classification (requires API credits)
Test 5  Result Overlay — generates a valid image file
"""

import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from step3_classify import (
    build_vlm_prompt,
    classify_candidate,
    load_api_key,
    parse_vlm_response,
    render_result_overlay,
    render_style_overlay,
)

# ── paths ─────────────────────────────────────────────────────────────

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_TESTS_DIR, "..", "..", ".."))

MAIN_ST_EX_PDF = os.path.join(PROJECT_ROOT, "example_plans", "main_st_ex.pdf")
MAIN_ST_EX_CANDIDATES = os.path.join(
    PROJECT_ROOT, "output", "main_st_ex", "step2", "candidates.json",
)


# ── helpers ───────────────────────────────────────────────────────────

def _is_valid_png(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Width and height from the IHDR chunk."""
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
# Test 1 — Rendering
# =====================================================================

class TestRendering:
    """Overlay for main_st_ex candidate 1, style width 0.50."""

    TARGET_STYLE = "stroke_rgb(0.0,0.0,0.0)_width_0.50"

    def test_overlay_is_valid_png(self, main_st_ex_candidate_1):
        png = render_style_overlay(
            MAIN_ST_EX_PDF, main_st_ex_candidate_1, self.TARGET_STYLE,
        )
        assert len(png) > 0
        assert _is_valid_png(png)

    def test_overlay_is_cropped_not_full_page(self, main_st_ex_candidate_1):
        """Image width should match the bbox crop, not the full page."""
        png = render_style_overlay(
            MAIN_ST_EX_PDF, main_st_ex_candidate_1, self.TARGET_STYLE,
            render_scale=2.0,
        )
        w, h = _png_dimensions(png)
        # bbox width ~235 pt + 60 padding → ~590 px at 2×
        # full page ~612 pt → ~1224 px at 2×
        assert w < 1000, f"width {w} looks like the full page"
        assert w > 100
        assert h > 100


# =====================================================================
# Test 2 — Prompt Construction
# =====================================================================

class TestPromptConstruction:

    def test_contains_structural_walls(self):
        assert "structural walls" in build_vlm_prompt()

    def test_ends_with_yes_no(self):
        assert build_vlm_prompt().rstrip().endswith("'YES' or 'NO'.")


# =====================================================================
# Test 3 — Response Parsing
# =====================================================================

class TestResponseParsing:

    @pytest.mark.parametrize("text,expected", [
        ("YES", "YES"),
        ("No", "NO"),
        ("yes definitely", "YES"),
        ("NO - these are fixtures", "NO"),
    ])
    def test_verdict(self, text, expected):
        assert parse_vlm_response(text)["verdict"] == expected

    def test_clean_response_high_confidence(self):
        assert parse_vlm_response("YES")["confidence"] == "high"
        assert parse_vlm_response("NO")["confidence"] == "high"

    def test_verbose_response_not_high(self):
        assert parse_vlm_response("yes definitely")["confidence"] != "high"


# =====================================================================
# Test 4 — End-to-End with VLM  (skip: pytest -m "not vlm")
# =====================================================================

@pytest.mark.vlm
class TestEndToEnd:

    def test_classify_main_st_ex_candidate_1(self, main_st_ex_candidate_1):
        """3 styles → 3 VLM calls.  At least one must be walls."""
        api_key = load_api_key()
        result = classify_candidate(
            MAIN_ST_EX_PDF, main_st_ex_candidate_1, api_key,
        )

        assert "candidate_id" in result
        assert "bounding_box" in result
        assert "style_classifications" in result
        assert "wall_primitives" in result
        assert len(result["style_classifications"]) == 3
        assert len(result["wall_primitives"]) > 0, (
            "expected at least one style classified as walls"
        )


# =====================================================================
# Test 5 — Result Overlay
# =====================================================================

class TestResultOverlay:

    def test_result_overlay_valid_png(self, main_st_ex_candidate_1, tmp_path):
        # mock classifications — first style YES, rest NO
        classifications = {}
        for i, sk in enumerate(
            main_st_ex_candidate_1["grouped_primitives"],
        ):
            classifications[sk] = {
                "verdict": "YES" if i == 0 else "NO",
                "confidence": "high",
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
