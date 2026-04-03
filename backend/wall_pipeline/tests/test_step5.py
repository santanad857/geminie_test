"""
Test suite for Step 5: Wall Polygon Reconstruction
===================================================

Test 1  Segment Decomposition — polyline edges → individual straight segments
Test 2  Parallel Pairing      — two parallel segments → one wall rectangle
Test 3  Single-Line Offset    — unpaired segment → offset rectangle
Test 4  Polygon Passthrough   — existing polygons pass through unchanged
Test 5  Linear Footage        — correct LF measurement per wall
Test 6  Full Pipeline         — reconstruct_walls end-to-end
Test 7  Output Schema         — correct structure & invariants
Test 8  Integration           — hardcoded confirmed-good plans (real data)
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from shapely.geometry import Polygon

from step5_reconstruct import (
    decompose_edges_to_segments,
    pair_parallel_segments,
    build_wall_polygon,
    offset_segment_to_wall,
    reconstruct_walls,
    load_consolidated,
    HARDCODED_GOOD,
)

PPI = 72.0


# ── helpers ──────────────────────────────────────────────────────────

def _edge(coords):
    """Build a Step 4-style edge dict."""
    length = 0
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        length += math.hypot(dx, dy)
    return {"coords": coords, "length": length}


def _poly(coords):
    """Build a Step 4-style polygon dict."""
    p = Polygon(coords)
    return {"coords": coords, "area": p.area}


def _consolidated(edges=None, polygons=None, cid=1):
    """Minimal Step 4 consolidated result."""
    return {
        "candidate_id": cid,
        "bounding_box": [0, 0, 5000, 5000],
        "edges": edges or [],
        "polygons": polygons or [],
        "stats": {},
    }


# =====================================================================
# Test 1 — Segment Decomposition
# =====================================================================

class TestDecomposition:

    def test_two_point_edge(self):
        """A simple 2-point edge yields one segment."""
        edges = [_edge([[0, 0], [100, 0]])]
        segs = decompose_edges_to_segments(edges)
        assert len(segs) == 1
        assert segs[0]["start"] == [0, 0]
        assert segs[0]["end"] == [100, 0]

    def test_multi_point_edge(self):
        """A 4-point L-shaped edge yields 3 segments."""
        edges = [_edge([[0, 0], [100, 0], [100, 50], [100, 150]])]
        segs = decompose_edges_to_segments(edges)
        assert len(segs) == 3

    def test_segment_has_required_keys(self):
        edges = [_edge([[0, 0], [100, 0]])]
        segs = decompose_edges_to_segments(edges)
        seg = segs[0]
        for key in ("start", "end", "length", "angle", "midpoint", "edge_idx"):
            assert key in seg, f"missing key: {key}"

    def test_segment_length(self):
        edges = [_edge([[0, 0], [3, 4]])]
        segs = decompose_edges_to_segments(edges)
        assert abs(segs[0]["length"] - 5.0) < 0.01

    def test_segment_angle_horizontal(self):
        edges = [_edge([[0, 0], [100, 0]])]
        segs = decompose_edges_to_segments(edges)
        assert abs(segs[0]["angle"]) < 0.01  # ~0 radians

    def test_segment_angle_vertical(self):
        edges = [_edge([[0, 0], [0, 100]])]
        segs = decompose_edges_to_segments(edges)
        assert abs(segs[0]["angle"] - math.pi / 2) < 0.01

    def test_empty_edges(self):
        assert decompose_edges_to_segments([]) == []

    def test_degenerate_segment_dropped(self):
        """Zero-length segment should be excluded."""
        edges = [_edge([[5, 5], [5, 5], [100, 5]])]
        segs = decompose_edges_to_segments(edges)
        assert len(segs) == 1
        assert segs[0]["length"] > 0


# =====================================================================
# Test 2 — Parallel Pairing
# =====================================================================

class TestParallelPairing:

    def test_horizontal_pair(self):
        """Two parallel horizontal lines 10pt apart → paired."""
        segs = decompose_edges_to_segments([
            _edge([[0, 0], [100, 0]]),
            _edge([[0, 10], [100, 10]]),
        ])
        pairs, unpaired = pair_parallel_segments(
            segs, max_thickness=20.0, min_overlap=0.3,
        )
        assert len(pairs) == 1
        assert len(unpaired) == 0

    def test_vertical_pair(self):
        """Two parallel vertical lines 8pt apart → paired."""
        segs = decompose_edges_to_segments([
            _edge([[0, 0], [0, 100]]),
            _edge([[8, 0], [8, 100]]),
        ])
        pairs, unpaired = pair_parallel_segments(
            segs, max_thickness=20.0, min_overlap=0.3,
        )
        assert len(pairs) == 1

    def test_perpendicular_not_paired(self):
        """Horizontal and vertical lines → not paired."""
        segs = decompose_edges_to_segments([
            _edge([[0, 0], [100, 0]]),
            _edge([[50, -10], [50, 10]]),
        ])
        _, unpaired = pair_parallel_segments(
            segs, max_thickness=20.0, min_overlap=0.3,
        )
        assert len(unpaired) == 2

    def test_parallel_but_too_far(self):
        """Parallel lines beyond max_thickness → not paired."""
        segs = decompose_edges_to_segments([
            _edge([[0, 0], [100, 0]]),
            _edge([[0, 50], [100, 50]]),
        ])
        _, unpaired = pair_parallel_segments(
            segs, max_thickness=20.0, min_overlap=0.3,
        )
        assert len(unpaired) == 2

    def test_parallel_no_overlap(self):
        """Parallel lines at same distance but no longitudinal overlap → not paired."""
        segs = decompose_edges_to_segments([
            _edge([[0, 0], [100, 0]]),
            _edge([[200, 10], [300, 10]]),
        ])
        _, unpaired = pair_parallel_segments(
            segs, max_thickness=20.0, min_overlap=0.3,
        )
        assert len(unpaired) == 2

    def test_partial_overlap_paired(self):
        """Parallel lines with >30% overlap → paired."""
        segs = decompose_edges_to_segments([
            _edge([[0, 0], [100, 0]]),
            _edge([[60, 10], [200, 10]]),
        ])
        pairs, _ = pair_parallel_segments(
            segs, max_thickness=20.0, min_overlap=0.3,
        )
        assert len(pairs) == 1

    def test_pair_contains_thickness(self):
        """Paired result includes the wall thickness."""
        segs = decompose_edges_to_segments([
            _edge([[0, 0], [100, 0]]),
            _edge([[0, 12], [100, 12]]),
        ])
        pairs, _ = pair_parallel_segments(
            segs, max_thickness=20.0, min_overlap=0.3,
        )
        assert abs(pairs[0]["thickness"] - 12.0) < 0.5


# =====================================================================
# Test 3 — Wall Polygon Construction
# =====================================================================

class TestBuildWallPolygon:

    def test_paired_rectangle(self):
        """Two horizontal segments → valid rectangle polygon."""
        seg_a = {"start": [0, 0], "end": [100, 0], "length": 100,
                 "angle": 0.0, "midpoint": [50, 0], "edge_idx": 0}
        seg_b = {"start": [0, 10], "end": [100, 10], "length": 100,
                 "angle": 0.0, "midpoint": [50, 10], "edge_idx": 1}
        poly = build_wall_polygon(seg_a, seg_b)
        assert poly.is_valid
        assert abs(poly.area - 1000.0) < 10  # 100 × 10

    def test_offset_rectangle(self):
        """Single segment offset by thickness → valid rectangle."""
        seg = {"start": [0, 0], "end": [100, 0], "length": 100,
               "angle": 0.0, "midpoint": [50, 0], "edge_idx": 0}
        poly = offset_segment_to_wall(seg, thickness=8.0)
        assert poly.is_valid
        assert abs(poly.area - 800.0) < 10  # 100 × 8

    def test_vertical_offset(self):
        """Vertical segment offset → rectangle with correct area."""
        seg = {"start": [0, 0], "end": [0, 100], "length": 100,
               "angle": math.pi / 2, "midpoint": [0, 50], "edge_idx": 0}
        poly = offset_segment_to_wall(seg, thickness=8.0)
        assert poly.is_valid
        assert abs(poly.area - 800.0) < 10

    def test_diagonal_offset(self):
        """45° diagonal segment → valid rectangle."""
        seg = {"start": [0, 0], "end": [100, 100],
               "length": math.hypot(100, 100),
               "angle": math.pi / 4, "midpoint": [50, 50], "edge_idx": 0}
        poly = offset_segment_to_wall(seg, thickness=10.0)
        assert poly.is_valid
        expected_area = math.hypot(100, 100) * 10
        assert abs(poly.area - expected_area) < 20


# =====================================================================
# Test 4 — Linear Footage
# =====================================================================

class TestLinearFootage:

    def test_simple_wall_lf(self):
        """100pt wall at 72ppi = 100/72 inches."""
        consolidated = _consolidated(edges=[
            _edge([[0, 0], [100, 0]]),
            _edge([[0, 10], [100, 10]]),
        ])
        result = reconstruct_walls(consolidated, pt_per_inch=PPI)
        total_lf_pt = result["total_lf_pt"]
        assert total_lf_pt > 90  # roughly 100pt

    def test_polygon_wall_lf(self):
        """Polygon wall: LF = longer dimension."""
        consolidated = _consolidated(polygons=[
            _poly([[0, 0], [200, 0], [200, 10], [0, 10]]),
        ])
        result = reconstruct_walls(consolidated, pt_per_inch=PPI)
        assert result["total_lf_pt"] > 150  # ~200pt long wall

    def test_lf_conversion_to_inches(self):
        """total_lf_inches = total_lf_pt / pt_per_inch."""
        consolidated = _consolidated(edges=[
            _edge([[0, 0], [72, 0]]),
            _edge([[0, 10], [72, 10]]),
        ])
        result = reconstruct_walls(consolidated, pt_per_inch=PPI)
        # 72pt at 72ppi = 1 inch
        assert abs(result["total_lf_inches"] - result["total_lf_pt"] / PPI) < 0.01


# =====================================================================
# Test 5 — Polygon Passthrough
# =====================================================================

class TestPolygonPassthrough:

    def test_polygon_becomes_wall(self):
        """Existing polygon from Step 4 → wall entry with source='polygon'."""
        consolidated = _consolidated(polygons=[
            _poly([[0, 0], [100, 0], [100, 10], [0, 10]]),
        ])
        result = reconstruct_walls(consolidated, pt_per_inch=PPI)
        polygon_walls = [w for w in result["walls"] if w["source"] == "polygon"]
        assert len(polygon_walls) == 1

    def test_polygon_area_preserved(self):
        consolidated = _consolidated(polygons=[
            _poly([[0, 0], [100, 0], [100, 10], [0, 10]]),
        ])
        result = reconstruct_walls(consolidated, pt_per_inch=PPI)
        wall = [w for w in result["walls"] if w["source"] == "polygon"][0]
        poly = Polygon(wall["polygon"])
        assert abs(poly.area - 1000.0) < 10


# =====================================================================
# Test 6 — Full Pipeline (reconstruct_walls)
# =====================================================================

class TestReconstructWalls:

    def test_paired_walls(self):
        """Two parallel edges → one paired wall."""
        consolidated = _consolidated(edges=[
            _edge([[0, 0], [200, 0]]),
            _edge([[0, 10], [200, 10]]),
        ])
        result = reconstruct_walls(consolidated, pt_per_inch=PPI)
        paired = [w for w in result["walls"] if w["source"] == "paired"]
        assert len(paired) >= 1

    def test_unpaired_gets_offset(self):
        """Single edge with no partner → offset wall."""
        consolidated = _consolidated(edges=[
            _edge([[0, 0], [200, 0]]),
        ])
        result = reconstruct_walls(consolidated, pt_per_inch=PPI)
        offset = [w for w in result["walls"] if w["source"] == "offset"]
        assert len(offset) >= 1

    def test_mixed_paired_and_offset(self):
        """Pair + lonely edge → one paired, one offset."""
        consolidated = _consolidated(edges=[
            _edge([[0, 0], [200, 0]]),
            _edge([[0, 10], [200, 10]]),
            _edge([[0, 500], [200, 500]]),  # far away, no partner
        ])
        result = reconstruct_walls(consolidated, pt_per_inch=PPI)
        paired = [w for w in result["walls"] if w["source"] == "paired"]
        offset = [w for w in result["walls"] if w["source"] == "offset"]
        assert len(paired) >= 1
        assert len(offset) >= 1

    def test_empty_input(self):
        consolidated = _consolidated()
        result = reconstruct_walls(consolidated, pt_per_inch=PPI)
        assert result["walls"] == []
        assert result["total_lf_pt"] == 0

    def test_mixed_edges_and_polygons(self):
        consolidated = _consolidated(
            edges=[
                _edge([[0, 0], [200, 0]]),
                _edge([[0, 10], [200, 10]]),
            ],
            polygons=[
                _poly([[500, 0], [600, 0], [600, 10], [500, 10]]),
            ],
        )
        result = reconstruct_walls(consolidated, pt_per_inch=PPI)
        sources = {w["source"] for w in result["walls"]}
        assert "paired" in sources or "offset" in sources
        assert "polygon" in sources


# =====================================================================
# Test 7 — Output Schema
# =====================================================================

class TestOutputSchema:

    @pytest.fixture
    def sample_result(self):
        consolidated = _consolidated(
            edges=[
                _edge([[0, 0], [200, 0]]),
                _edge([[0, 10], [200, 10]]),
            ],
            polygons=[
                _poly([[500, 0], [600, 0], [600, 10], [500, 10]]),
            ],
        )
        return reconstruct_walls(consolidated, pt_per_inch=PPI)

    def test_top_level_keys(self, sample_result):
        for key in ("candidate_id", "bounding_box", "walls",
                     "total_lf_pt", "total_lf_inches", "stats"):
            assert key in sample_result

    def test_wall_entry_keys(self, sample_result):
        for wall in sample_result["walls"]:
            for key in ("polygon", "thickness", "length_pt",
                        "length_inches", "source"):
                assert key in wall, f"missing key: {key}"

    def test_wall_polygon_is_valid(self, sample_result):
        for wall in sample_result["walls"]:
            poly = Polygon(wall["polygon"])
            assert poly.is_valid
            assert poly.area > 0

    def test_stats_keys(self, sample_result):
        for key in ("paired_count", "offset_count",
                     "polygon_count", "total_wall_count"):
            assert key in sample_result["stats"]

    def test_wall_count_consistency(self, sample_result):
        s = sample_result["stats"]
        assert (s["paired_count"] + s["offset_count"] +
                s["polygon_count"]) == s["total_wall_count"]
        assert s["total_wall_count"] == len(sample_result["walls"])

    def test_json_serializable(self, sample_result):
        dumped = json.dumps(sample_result)
        loaded = json.loads(dumped)
        assert loaded["candidate_id"] == sample_result["candidate_id"]


# =====================================================================
# Test 8 — Integration with hardcoded plans
# =====================================================================

@pytest.mark.integration
class TestIntegration:

    @pytest.fixture(scope="class")
    def results(self):
        out = {}
        for plan, cid in HARDCODED_GOOD.items():
            try:
                consolidated = load_consolidated(plan, cid)
            except FileNotFoundError:
                pytest.skip(f"step4 data not found for {plan}")
            out[plan] = reconstruct_walls(consolidated, pt_per_inch=PPI)
        return out

    def test_test2_has_walls(self, results):
        r = results["2026-03-27_Test-2_v4"]
        assert len(r["walls"]) > 0

    def test_test2_has_paired_walls(self, results):
        """Line-based plan should have paired walls."""
        r = results["2026-03-27_Test-2_v4"]
        paired = [w for w in r["walls"] if w["source"] == "paired"]
        assert len(paired) > 0

    def test_test2_positive_lf(self, results):
        assert results["2026-03-27_Test-2_v4"]["total_lf_pt"] > 0

    def test_fmoc_has_walls(self, results):
        r = results["2026.01.29_-_FMOC_A_-_50%_CD_copy"]
        assert len(r["walls"]) > 0

    def test_main_st_ex_has_polygon_walls(self, results):
        r = results["main_st_ex"]
        polygon_walls = [w for w in r["walls"] if w["source"] == "polygon"]
        assert len(polygon_walls) > 0

    def test_main_st_ex_positive_lf(self, results):
        assert results["main_st_ex"]["total_lf_pt"] > 0

    def test_all_walls_valid_polygons(self, results):
        for plan, r in results.items():
            for wall in r["walls"]:
                poly = Polygon(wall["polygon"])
                assert poly.is_valid, f"invalid wall polygon in {plan}"
                assert poly.area > 0, f"zero-area wall in {plan}"

    def test_all_json_serializable(self, results):
        for plan, r in results.items():
            json.dumps(r)
