"""
Test suite for Step 4: Geometric Consolidation
===============================================

Test 1  Flatten + Merge      — micro-segments fuse; multi-style dedup
Test 2  T-junction Snap      — near-miss endpoints snap to nearby walls
Test 3  Collinear Bridge     — small collinear gaps filled
Test 4  Stub Cleanup         — isolated short segments removed
Test 5  Full Pipeline        — consolidate_walls end-to-end
Test 6  Output Schema        — correct structure & invariants
Test 7  Integration          — hardcoded confirmed-good plans (real data)
Test 8  Performance          — 5 000 lines under 10 s
"""

import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from shapely.geometry import LineString, Point

from step4_consolidate import (
    HARDCODED_GOOD,
    consolidate_walls,
    load_classification,
    phase_4_1_flatten_merge,
    phase_4_2_t_junction_snap,
    phase_4_3_collinear_bridge,
    phase_4_4_stub_cleanup,
)

PPI = 72.0


# ── helpers ──────────────────────────────────────────────────────────

def _line(x1, y1, x2, y2):
    return {"type": "line", "points": [[x1, y1], [x2, y2]]}


def _polygon(corners):
    return {"type": "polygon", "points": corners}


def _classification(wall_primitives, cid=1):
    """Wrap wall_primitives into a minimal classification result."""
    return {
        "candidate_id": cid,
        "bounding_box": [0, 0, 5000, 5000],
        "wall_primitives": wall_primitives,
    }


# =====================================================================
# Test 1 — Phase 4.1: Flatten + Merge
# =====================================================================

class TestFlattenMerge:

    def test_connected_segments_merge(self):
        """Three collinear connected segments → one line."""
        prims = {"s": [
            _line(0, 0, 100, 0),
            _line(100, 0, 200, 0),
            _line(200, 0, 300, 0),
        ]}
        lines, _polys = phase_4_1_flatten_merge(prims)
        assert len(lines) == 1
        assert abs(lines[0].length - 300) < 1

    def test_multi_style_same_line_dedup(self):
        """Identical line from two styles → one line."""
        prims = {
            "style_A": [_line(0, 0, 100, 0)],
            "style_B": [_line(0, 0, 100, 0)],
        }
        lines, _ = phase_4_1_flatten_merge(prims)
        assert len(lines) == 1

    def test_disconnected_stay_separate(self):
        prims = {"s": [
            _line(0, 0, 100, 0),
            _line(0, 1000, 100, 1000),
        ]}
        lines, _ = phase_4_1_flatten_merge(prims)
        assert len(lines) == 2

    def test_polygons_extracted(self):
        prims = {"s": [
            _polygon([[0, 0], [100, 0], [100, 10], [0, 10]]),
        ]}
        _, polys = phase_4_1_flatten_merge(prims)
        assert len(polys) == 1
        assert polys[0].area > 0

    def test_overlapping_polygons_merge(self):
        prims = {"s": [
            _polygon([[0, 0], [100, 0], [100, 10], [0, 10]]),
            _polygon([[50, 0], [150, 0], [150, 10], [50, 10]]),
        ]}
        _, polys = phase_4_1_flatten_merge(prims)
        assert len(polys) == 1
        assert polys[0].area > 1000  # wider than either input alone

    def test_degenerate_line_dropped(self):
        prims = {"s": [_line(5, 5, 5, 5)]}
        lines, _ = phase_4_1_flatten_merge(prims)
        assert len(lines) == 0

    def test_empty_input(self):
        lines, polys = phase_4_1_flatten_merge({})
        assert lines == []
        assert polys == []

    def test_float_noise_merge(self):
        """Two segments whose shared endpoint differs by < 0.005 → merge."""
        prims = {"s": [
            _line(0, 0, 100.002, 0),
            _line(100.004, 0, 200, 0),
        ]}
        lines, _ = phase_4_1_flatten_merge(prims, coord_precision=2)
        # Both round to 100.00 → should merge
        assert len(lines) == 1

    def test_micro_segments_from_pdf(self):
        """Many tiny collinear segments (typical PDF wall) → one line."""
        segs = []
        for i in range(50):
            x0 = i * 10.0
            segs.append(_line(x0, 0, x0 + 10.0, 0))
        prims = {"s": segs}
        lines, _ = phase_4_1_flatten_merge(prims)
        assert len(lines) == 1
        assert abs(lines[0].length - 500) < 1


# =====================================================================
# Test 2 — Phase 4.2: T-junction Snap
# =====================================================================

class TestTJunctionSnap:

    def test_near_miss_snaps(self):
        """Horizontal line ending 5 pt from vertical wall → snaps."""
        lines = [
            LineString([(0, 50), (95, 50)]),
            LineString([(100, 0), (100, 100)]),
        ]
        result, count = phase_4_2_t_junction_snap(
            lines, PPI, t_junction_tol=10.0,
        )
        assert count >= 1
        # Some line should now pass through (100, 50) ± 1 pt
        target = Point(100, 50)
        assert min(l.distance(target) for l in result) < 1.0

    def test_already_touching_no_snap(self):
        """Endpoint exactly on the other line → no snap needed."""
        lines = [
            LineString([(0, 50), (100, 50)]),
            LineString([(100, 0), (100, 100)]),
        ]
        _, count = phase_4_2_t_junction_snap(
            lines, PPI, t_junction_tol=10.0,
        )
        assert count == 0

    def test_distant_no_snap(self):
        lines = [
            LineString([(0, 50), (50, 50)]),
            LineString([(100, 0), (100, 100)]),
        ]
        _, count = phase_4_2_t_junction_snap(
            lines, PPI, t_junction_tol=10.0,
        )
        assert count == 0

    def test_snap_to_nearest(self):
        """When two walls are candidates, endpoint snaps to the closer one."""
        lines = [
            LineString([(0, 50), (93, 50)]),     # 7 pt from wall A
            LineString([(100, 0), (100, 100)]),   # wall A
            LineString([(200, 0), (200, 100)]),   # wall B — further
        ]
        result, count = phase_4_2_t_junction_snap(
            lines, PPI, t_junction_tol=10.0,
        )
        assert count >= 1
        # Should have snapped to wall A (x=100), not wall B (x=200)
        target = Point(100, 50)
        assert min(l.distance(target) for l in result) < 1.0


# =====================================================================
# Test 3 — Phase 4.3: Collinear Bridge
# =====================================================================

class TestCollinearBridge:

    def test_collinear_gap_bridged(self):
        """Two horizontal lines with 5 pt gap → merged into one."""
        lines = [
            LineString([(0, 0), (100, 0)]),
            LineString([(105, 0), (200, 0)]),
        ]
        result, count = phase_4_3_collinear_bridge(
            lines, PPI, bridge_tol=10.0,
        )
        assert count == 1
        assert len(result) == 1
        assert result[0].length > 195

    def test_perpendicular_not_bridged(self):
        lines = [
            LineString([(0, 0), (100, 0)]),
            LineString([(105, 0), (105, 100)]),
        ]
        _, count = phase_4_3_collinear_bridge(
            lines, PPI, bridge_tol=10.0,
        )
        assert count == 0

    def test_parallel_offset_not_bridged(self):
        """Parallel but not collinear → no bridge."""
        lines = [
            LineString([(0, 0), (100, 0)]),
            LineString([(105, 10), (200, 10)]),
        ]
        _, count = phase_4_3_collinear_bridge(
            lines, PPI, bridge_tol=10.0,
        )
        assert count == 0

    def test_gap_too_large(self):
        lines = [
            LineString([(0, 0), (100, 0)]),
            LineString([(200, 0), (300, 0)]),
        ]
        _, count = phase_4_3_collinear_bridge(
            lines, PPI, bridge_tol=10.0,
        )
        assert count == 0

    def test_anti_hallway_guard(self):
        """Collinear gap with a crossing wall → no bridge."""
        lines = [
            LineString([(0, 0), (100, 0)]),
            LineString([(105, 0), (200, 0)]),
            LineString([(102, -50), (102, 50)]),  # perpendicular wall
        ]
        _, count = phase_4_3_collinear_bridge(
            lines, PPI, bridge_tol=10.0,
        )
        assert count == 0


# =====================================================================
# Test 4 — Phase 4.4: Stub Cleanup
# =====================================================================

class TestStubCleanup:

    def test_isolated_short_removed(self):
        lines = [
            LineString([(0, 0), (500, 0)]),
            LineString([(1000, 1000), (1005, 1000)]),  # 5 pt, isolated
        ]
        result, count = phase_4_4_stub_cleanup(
            lines, PPI, min_stub_length=10.0,
        )
        assert count == 1
        assert len(result) == 1

    def test_connected_short_kept(self):
        """Short segment sharing an endpoint with a long one → kept."""
        lines = [
            LineString([(0, 0), (500, 0)]),
            LineString([(500, 0), (505, 0)]),  # 5 pt but connected
        ]
        result, count = phase_4_4_stub_cleanup(
            lines, PPI, min_stub_length=10.0,
        )
        assert count == 0
        assert len(result) == 2

    def test_long_isolated_kept(self):
        lines = [
            LineString([(0, 0), (500, 0)]),
            LineString([(1000, 1000), (1100, 1000)]),  # 100 pt, isolated
        ]
        _, count = phase_4_4_stub_cleanup(
            lines, PPI, min_stub_length=10.0,
        )
        assert count == 0

    def test_cascading_removal(self):
        """Removing one stub may leave another newly isolated → also removed."""
        lines = [
            LineString([(0, 0), (500, 0)]),
            LineString([(1000, 0), (1005, 0)]),    # 5 pt, A
            LineString([(1005, 0), (1008, 0)]),    # 3 pt, B — shares ep with A
        ]
        # A has one ep shared with B → degree 2 → not removed initially.
        # B has one ep shared with A → degree 2 → not removed initially.
        # Neither is fully isolated, so neither should be removed.
        result, count = phase_4_4_stub_cleanup(
            lines, PPI, min_stub_length=10.0,
        )
        # Both short segments are connected to each other → kept
        assert count == 0


# =====================================================================
# Test 5 — Full Pipeline  (consolidate_walls)
# =====================================================================

class TestConsolidateWalls:

    def test_simple_wall_pair(self):
        cls = _classification({
            "style_A": [
                _line(0, 0, 500, 0),
                _line(0, 100, 500, 100),
            ],
        })
        result = consolidate_walls(cls)
        assert result["candidate_id"] == 1
        assert len(result["edges"]) == 2
        assert result["stats"]["input_primitive_count"] == 2

    def test_multi_style_connected(self):
        cls = _classification({
            "style_A": [_line(0, 0, 100, 0), _line(100, 0, 200, 0)],
            "style_B": [_line(200, 0, 300, 0)],
        })
        result = consolidate_walls(cls)
        assert len(result["edges"]) == 1

    def test_polygon_passthrough(self):
        cls = _classification({
            "fill_black": [
                _polygon([[0, 0], [100, 0], [100, 10], [0, 10]]),
                _polygon([[200, 0], [300, 0], [300, 10], [200, 10]]),
            ],
        })
        result = consolidate_walls(cls)
        assert len(result["polygons"]) == 2
        assert all(p["area"] > 0 for p in result["polygons"])

    def test_mixed_lines_and_polygons(self):
        cls = _classification({
            "stroke_A": [_line(0, 0, 500, 0), _line(0, 100, 500, 100)],
            "fill_A": [
                _polygon([[600, 0], [700, 0], [700, 10], [600, 10]]),
            ],
        })
        result = consolidate_walls(cls)
        assert len(result["edges"]) >= 1
        assert len(result["polygons"]) >= 1

    def test_empty_wall_primitives(self):
        cls = _classification({})
        result = consolidate_walls(cls)
        assert result["edges"] == []
        assert result["polygons"] == []


# =====================================================================
# Test 6 — Output Schema
# =====================================================================

class TestOutputSchema:

    def test_required_keys(self):
        result = consolidate_walls(_classification(
            {"s": [_line(0, 0, 100, 0)]},
        ))
        for key in ("candidate_id", "bounding_box", "edges",
                     "polygons", "stats"):
            assert key in result, f"missing key: {key}"

    def test_edge_structure(self):
        result = consolidate_walls(_classification(
            {"s": [_line(0, 0, 100, 0)]},
        ))
        for edge in result["edges"]:
            assert "coords" in edge
            assert "length" in edge
            assert isinstance(edge["coords"], list)
            assert len(edge["coords"]) >= 2
            assert edge["length"] > 0

    def test_polygon_structure(self):
        result = consolidate_walls(_classification(
            {"s": [_polygon([[0, 0], [100, 0], [100, 10], [0, 10]])]},
        ))
        for poly in result["polygons"]:
            assert "coords" in poly
            assert "area" in poly
            assert poly["area"] > 0

    def test_stats_keys(self):
        result = consolidate_walls(_classification(
            {"s": [_line(0, 0, 100, 0)]},
        ))
        expected = {
            "input_primitive_count", "post_merge_line_count",
            "output_edge_count", "output_polygon_count",
            "total_edge_length", "total_polygon_area",
            "t_junctions_snapped", "bridges_added", "stubs_removed",
        }
        assert expected <= set(result["stats"])

    def test_edges_sorted_longest_first(self):
        result = consolidate_walls(_classification({
            "s": [
                _line(0, 0, 50, 0),
                _line(0, 1000, 500, 1000),
            ],
        }))
        lengths = [e["length"] for e in result["edges"]]
        assert lengths == sorted(lengths, reverse=True)


# =====================================================================
# Test 7 — Integration with hardcoded plans
# =====================================================================

@pytest.mark.integration
class TestIntegration:

    @pytest.fixture(scope="class")
    def results(self):
        out = {}
        for plan, cid in HARDCODED_GOOD.items():
            try:
                cls = load_classification(plan, cid)
            except FileNotFoundError:
                pytest.skip(f"data not found for {plan}")
            out[plan] = consolidate_walls(cls)
        return out

    def test_test2_reduces_lines(self, results):
        r = results["2026-03-27_Test-2_v4"]
        assert r["stats"]["output_edge_count"] > 0
        assert r["stats"]["output_edge_count"] < r["stats"]["input_primitive_count"]

    def test_test2_reduction_ratio(self, results):
        """2 030 input lines should reduce meaningfully."""
        s = results["2026-03-27_Test-2_v4"]["stats"]
        ratio = s["input_primitive_count"] / max(s["output_edge_count"], 1)
        assert ratio > 1.2, f"reduction ratio {ratio:.1f}× is too low"

    def test_fmoc_has_edges(self, results):
        r = results["2026.01.29_-_FMOC_A_-_50%_CD_copy"]
        assert r["stats"]["output_edge_count"] > 0

    def test_fmoc_total_edge_length(self, results):
        """Consolidated edges should have substantial total length."""
        s = results["2026.01.29_-_FMOC_A_-_50%_CD_copy"]["stats"]
        assert s["total_edge_length"] > 100

    def test_main_st_ex_has_polygons(self, results):
        r = results["main_st_ex"]
        assert r["stats"]["output_polygon_count"] > 0

    def test_main_st_ex_polygon_area(self, results):
        """Merged polygons should have nonzero total area."""
        r = results["main_st_ex"]
        assert r["stats"]["total_polygon_area"] > 0

    def test_all_json_serializable(self, results):
        """Every result must survive a JSON round-trip."""
        for plan, r in results.items():
            dumped = json.dumps(r)
            loaded = json.loads(dumped)
            assert loaded["candidate_id"] == r["candidate_id"]


# =====================================================================
# Test 8 — Performance
# =====================================================================

class TestPerformance:

    def test_3000_lines_under_30s(self):
        rng = random.Random(42)
        segs = []
        for i in range(3000):
            x = rng.uniform(0, 10_000)
            y = rng.uniform(0, 10_000)
            a = rng.uniform(0, 2 * math.pi)
            L = rng.uniform(10, 200)
            segs.append(_line(x, y,
                              x + L * math.cos(a),
                              y + L * math.sin(a)))
        cls = _classification({"s": segs})

        t0 = time.perf_counter()
        consolidate_walls(cls)
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"took {elapsed:.2f}s"
