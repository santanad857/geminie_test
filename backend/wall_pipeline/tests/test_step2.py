"""
Test suite for Step 2: Two-Phase Spatial Clustering
====================================================

Phase 2.1 — intra-style clustering + noise filtering
Phase 2.2 — inter-style compositing into multi-style candidates
"""

import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from step2_cluster import (
    cluster_candidates,
    phase_2_1_intra_style,
    phase_2_2_inter_style,
    primitive_length,
    primitive_to_shapely,
)

PPI = 72.0  # standard PDF points-per-inch for test geometry


# ── helpers ──────────────────────────────────────────────────────────

def _line(x1, y1, x2, y2, idx=0):
    return {"type": "line", "points": [[x1, y1], [x2, y2]],
            "original_path_index": idx}


def _polygon(corners, idx=0):
    return {"type": "polygon", "points": corners,
            "original_path_index": idx}


# =====================================================================
# Phase 2.1  —  Intra-Style
# =====================================================================

class TestPhase21Isolation:
    """Different styles must NEVER merge inside Phase 2.1."""

    def test_different_styles_stay_separate(self):
        """Two parallel lines, different styles, 0.5*PPI apart."""
        prims = {
            "style_A": [_line(0, 0, 500, 0, 0)],
            "style_B": [_line(0, 0.5 * PPI, 500, 0.5 * PPI, 1)],
        }
        nets = phase_2_1_intra_style(prims, PPI, min_network_length=0)
        assert len(nets) == 2
        assert {n["style_key"] for n in nets} == {"style_A", "style_B"}

    def test_same_style_snaps(self):
        """Two collinear lines of the SAME style within snap_tol merge."""
        gap = 0.5 * PPI  # well within snap_tol (1.5*PPI = 108)
        prims = {
            "style_A": [
                _line(0, 0, 200, 0, 0),
                _line(200 + gap, 0, 400, 0, 1),
            ]
        }
        nets = phase_2_1_intra_style(prims, PPI, min_network_length=0)
        assert len(nets) == 1
        assert len(nets[0]["primitives"]) == 2


class TestPhase21Filtering:
    """Short connected components must be discarded."""

    def test_short_component_dropped(self):
        """Component of length 10*PPI is below threshold 36*PPI."""
        short_len = 10.0 * PPI   # 720 pt
        long_len = 50.0 * PPI    # 3600 pt
        threshold = 36.0 * PPI   # 2592 pt
        prims = {
            "style_A": [
                _line(0, 0, short_len, 0, 0),               # short, isolated
                _line(0, 5000, long_len, 5000, 1),           # long, isolated
            ]
        }
        nets = phase_2_1_intra_style(prims, PPI)
        assert len(nets) == 1
        assert nets[0]["total_length"] >= threshold

    def test_connected_short_lines_survive_if_combined_length_passes(self):
        """Many short lines that snap together can collectively exceed the
        threshold and therefore survive filtering."""
        # 40 lines, each 100 pt long, spaced 10pt apart (within snap_tol=108)
        prims = {"s": [_line(i * 110, 0, i * 110 + 100, 0, i)
                        for i in range(40)]}
        # Total length ≈ 40 * 100 = 4000 pt > 2592 threshold
        nets = phase_2_1_intra_style(prims, PPI)
        assert len(nets) == 1


# =====================================================================
# Phase 2.2  —  Inter-Style Compositing
# =====================================================================

class TestPhase22Compositing:
    """Structurally related networks from different styles merge."""

    def test_overlapping_polygon_and_lines_merge(self):
        """Lines from style_A + enclosing polygon from style_B -> 1 composite."""
        prims = {
            "style_A": [
                _line(0, 0, 500, 0, 0),
                _line(0, 100, 500, 100, 1),
            ],
            "style_B": [
                _polygon([[  -5, -5], [505, -5],
                          [505, 105], [  -5, 105]], 2),
            ],
        }
        nets = phase_2_1_intra_style(prims, PPI, min_network_length=0)
        assert len(nets) == 2

        composites = phase_2_2_inter_style(nets, PPI)
        assert len(composites) == 1
        gp = composites[0]["grouped_primitives"]
        assert "style_A" in gp and "style_B" in gp

    def test_nearby_parallel_styles_merge(self):
        """Two different-style line networks within wall_tol -> 1 composite."""
        sep = 6.0 * PPI  # 432 pt, well within wall_tol (14*72=1008)
        prims = {
            "stroke_black": [_line(0, 0, 800, 0, 0)],
            "stroke_grey":  [_line(0, sep, 800, sep, 1)],
        }
        nets = phase_2_1_intra_style(prims, PPI, min_network_length=0)
        composites = phase_2_2_inter_style(nets, PPI)
        assert len(composites) == 1


class TestPhase22Isolation:
    """Distant networks must NOT merge in Phase 2.2."""

    def test_distant_networks_stay_separate(self):
        gap = 24.0 * PPI  # 1728 pt >> wall_tol (1008)
        prims = {
            "style_A": [_line(0, 0, 500, 0, 0)],
            "style_B": [_line(0, gap, 500, gap, 1)],
        }
        nets = phase_2_1_intra_style(prims, PPI, min_network_length=0)
        composites = phase_2_2_inter_style(nets, PPI)
        assert len(composites) == 2


# =====================================================================
# Full pipeline  (cluster_candidates)
# =====================================================================

class TestFullPipeline:

    def test_end_to_end(self):
        """Wall pair (style A) + hatching (style B) inside -> 1 candidate."""
        sep = 6.0 * PPI
        # 30 hatch lines spaced 100 pt apart -> snap together -> total
        # length ~12 000 pt, well above the 2 592 pt threshold.
        hatches = [_line(x, 10, x, sep - 10, i + 2)
                   for i, x in enumerate(range(50, 3000, 100))]
        prims = {
            "outline": [
                _line(0, 0, 3000, 0, 0),
                _line(0, sep, 3000, sep, 1),
            ],
            "hatch": hatches,
        }
        cands = cluster_candidates(prims, PPI)
        assert len(cands) >= 1
        biggest = cands[0]
        assert "outline" in biggest["grouped_primitives"]
        assert "hatch" in biggest["grouped_primitives"]


# =====================================================================
# Output schema
# =====================================================================

class TestOutputSchema:

    def test_required_keys(self):
        prims = {"s": [_line(0, 0, 3000, 0)]}
        for c in cluster_candidates(prims, PPI):
            assert {"candidate_id", "bounding_box",
                    "grouped_primitives"} <= set(c)
            assert isinstance(c["candidate_id"], int)
            assert len(c["bounding_box"]) == 4
            assert isinstance(c["grouped_primitives"], dict)

    def test_empty_input(self):
        assert cluster_candidates({}, PPI) == []

    def test_all_filtered_returns_empty(self):
        """A single 1-pt line can't survive the length filter."""
        prims = {"s": [_line(0, 0, 1, 0)]}
        assert cluster_candidates(prims, PPI) == []

    def test_candidates_sorted_largest_first(self):
        prims = {
            "big":   [_line(0, 0, 5000, 0, 0)],
            "small": [_line(0, 9000, 3000, 9000, 1)],
        }
        cands = cluster_candidates(prims, PPI)
        sizes = [sum(len(v) for v in c["grouped_primitives"].values())
                 for c in cands]
        assert sizes == sorted(sizes, reverse=True)


# =====================================================================
# Helpers
# =====================================================================

class TestHelpers:

    def test_primitive_length_line(self):
        assert abs(primitive_length(_line(0, 0, 3, 4)) - 5.0) < 1e-9

    def test_primitive_length_polygon(self):
        p = _polygon([[0, 0], [10, 0], [10, 10], [0, 10]])
        assert abs(primitive_length(p) - 40.0) < 1e-9

    def test_shapely_line(self):
        g = primitive_to_shapely(_line(0, 0, 10, 0))
        assert g is not None and abs(g.length - 10) < 1e-9

    def test_shapely_polygon(self):
        g = primitive_to_shapely(
            _polygon([[0, 0], [10, 0], [10, 10], [0, 10]]))
        assert g is not None and abs(g.area - 100) < 1e-9

    def test_degenerate_returns_none(self):
        assert primitive_to_shapely(_line(5, 5, 5, 5)) is None


# =====================================================================
# Performance
# =====================================================================

class TestPerformance:

    def test_10k_lines_under_5s(self):
        rng = random.Random(42)
        lines = []
        for i in range(10_000):
            x, y = rng.uniform(0, 10_000), rng.uniform(0, 10_000)
            a = rng.uniform(0, 2 * math.pi)
            L = rng.uniform(10, 200)
            lines.append(_line(x, y, x + L * math.cos(a),
                               y + L * math.sin(a), i))
        prims = {"s": lines}
        t0 = time.perf_counter()
        cands = cluster_candidates(prims, PPI)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"took {elapsed:.2f}s"
