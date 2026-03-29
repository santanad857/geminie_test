#!/usr/bin/env python3
"""
Step 2 -- Two-Phase Spatial Clustering
======================================

Phase 2.1  Intra-Style Clustering (noise reduction)
    For *each* style_key independently, build a snap-tolerance connectivity
    graph, extract connected components, and discard any component whose
    total geometric length is below a 3-foot structural minimum.

Phase 2.2  Inter-Style Compositing (semantic merging)
    Build a *secondary* graph whose nodes are the purified networks from
    Phase 2.1.  Connect two networks when their geometries are within
    wall-thickness tolerance (proximity) or when one encloses the other
    (containment).  Connected components become multi-style Candidate
    Composites.

No global blending is ever performed — primitives of different styles
are never placed in the same connectivity graph.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import shapely as _shp
from scipy.spatial import cKDTree
from shapely import STRtree
from shapely.geometry import LineString, Polygon as ShapelyPolygon
from shapely.ops import unary_union


# ── geometry helpers ─────────────────────────────────────────────────

def linearize_bezier(
    control_points: list[list[float]],
    num_segments: int = 8,
) -> list[list[float]]:
    """Approximate a cubic Bezier as a polyline."""
    p0, p1, p2, p3 = (np.asarray(c) for c in control_points)
    ts = np.linspace(0.0, 1.0, num_segments + 1)
    return (
        np.outer((1 - ts) ** 3, p0)
        + np.outer(3 * (1 - ts) ** 2 * ts, p1)
        + np.outer(3 * (1 - ts) * ts ** 2, p2)
        + np.outer(ts ** 3, p3)
    ).tolist()


def primitive_to_shapely(prim: dict):
    """Convert a Step-1 primitive to a Shapely geometry (or *None*)."""
    pts = prim["points"]
    kind = prim["type"]
    if kind == "line":
        if len(pts) < 2 or pts[0] == pts[1]:
            return None
        return LineString(pts)
    if kind == "polygon":
        if len(pts) < 3:
            return None
        try:
            poly = ShapelyPolygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            return poly if not poly.is_empty else None
        except Exception:
            return None
    if kind == "curve":
        if len(pts) == 4:
            return LineString(linearize_bezier(pts))
        return LineString(pts) if len(pts) >= 2 else None
    return None


def primitive_length(prim: dict) -> float:
    """Total geometric length of a single primitive."""
    pts = prim["points"]
    kind = prim["type"]
    if kind == "line":
        return math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    if kind == "polygon":
        n = len(pts)
        return sum(
            math.hypot(pts[(i + 1) % n][0] - pts[i][0],
                       pts[(i + 1) % n][1] - pts[i][1])
            for i in range(n))
    if kind == "curve":
        return sum(
            math.hypot(pts[i + 1][0] - pts[i][0],
                       pts[i + 1][1] - pts[i][1])
            for i in range(len(pts) - 1))
    return 0.0


def _bbox_from_geoms(geoms) -> list[float]:
    """Fast bounding box without unary_union."""
    bounds = np.array([g.bounds for g in geoms])
    return [float(bounds[:, 0].min()), float(bounds[:, 1].min()),
            float(bounds[:, 2].max()), float(bounds[:, 3].max())]


# ── Phase 2.1: Intra-Style Clustering ───────────────────────────────

def phase_2_1_intra_style(
    primitives_dict: dict[str, list[dict]],
    pt_per_inch: float,
    *,
    snap_tol: float | None = None,
    min_network_length: float | None = None,
) -> list[dict]:
    """
    Cluster primitives *within each style group* and drop noise.

    Returns a list of purified-network dicts::

        {"network_id": int, "style_key": str,
         "primitives": [...], "total_length": float,
         "_geoms": [shapely, ...]}          # internal, not serialised
    """
    if snap_tol is None:
        snap_tol = 1.5 * pt_per_inch
    if min_network_length is None:
        min_network_length = 36.0 * pt_per_inch

    purified: list[dict] = []
    nid = 0

    for style_key, prims in primitives_dict.items():
        # convert to Shapely, drop failures
        entries: list[tuple[dict, Any]] = []
        for prim in prims:
            geom = primitive_to_shapely(prim)
            if geom is not None and not geom.is_empty:
                entries.append((prim, geom))
        if not entries:
            continue

        n = len(entries)
        geoms = [g for _, g in entries]
        geoms_arr = np.array(geoms, dtype=object)

        # ---- Rule A: snap within this style -------------------------
        tree = STRtree(geoms)
        a_l, a_r = tree.query(geoms_arr, predicate="dwithin",
                              distance=snap_tol)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        upper = a_l < a_r
        if upper.any():
            G.add_edges_from(zip(a_l[upper].tolist(),
                                 a_r[upper].tolist()))

        # ---- extract components, apply length filter ----------------
        for comp in nx.connected_components(G):
            comp_prims = [entries[i][0] for i in comp]
            comp_geoms = [entries[i][1] for i in comp]
            total_len  = sum(primitive_length(p) for p in comp_prims)

            if total_len >= min_network_length:
                nid += 1
                purified.append({
                    "network_id":   nid,
                    "style_key":    style_key,
                    "primitives":   comp_prims,
                    "total_length": total_len,
                    "_geoms":       comp_geoms,
                })

    return purified


# ── Phase 2.2: Inter-Style Compositing ──────────────────────────────

def phase_2_2_inter_style(
    purified_networks: list[dict],
    pt_per_inch: float,
    *,
    wall_tol: float | None = None,
) -> list[dict[str, Any]]:
    """
    Merge structurally related networks (from *different* styles) into
    multi-style Candidate Composites.

    Rule B: two networks are related when **any** primitive of one is
    within *wall_tol* of **any** primitive of the other.  This covers
    both proximity **and** enclosure (a line inside a polygon has
    distance 0 from that polygon, which is always < wall_tol).
    """
    if wall_tol is None:
        wall_tol = 14.0 * pt_per_inch
    K = len(purified_networks)
    if K == 0:
        return []

    # ---- build per-network STRtrees and bounding boxes --------------
    trees: list[STRtree] = []
    bboxes: list[tuple[float, ...]] = []
    for net in purified_networks:
        gs = net["_geoms"]
        trees.append(STRtree(gs))
        b = np.array([g.bounds for g in gs])
        bboxes.append((b[:, 0].min(), b[:, 1].min(),
                        b[:, 2].max(), b[:, 3].max()))

    # ---- secondary graph (nodes = networks) -------------------------
    G = nx.Graph()
    G.add_nodes_from(range(K))

    for i in range(K):
        bi = bboxes[i]
        for j in range(i + 1, K):
            bj = bboxes[j]
            # coarse: expanded bounding-boxes must overlap
            if (bi[0] - wall_tol > bj[2] or bi[2] + wall_tol < bj[0]
                    or bi[1] - wall_tol > bj[3]
                    or bi[3] + wall_tol < bj[1]):
                continue
            # detailed: any primitive of j within wall_tol of tree i?
            arr_j = np.array(purified_networks[j]["_geoms"], dtype=object)
            hits_l, _ = trees[i].query(arr_j, predicate="dwithin",
                                       distance=wall_tol)
            if len(hits_l) > 0:
                G.add_edge(i, j)

    # ---- extract composites -----------------------------------------
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    composites: list[dict[str, Any]] = []
    for cid, comp in enumerate(components, start=1):
        grouped: dict[str, list[dict]] = defaultdict(list)
        comp_geoms: list = []
        for net_idx in comp:
            net = purified_networks[net_idx]
            grouped[net["style_key"]].extend(net["primitives"])
            comp_geoms.extend(net["_geoms"])

        composites.append({
            "candidate_id":       cid,
            "bounding_box":       _bbox_from_geoms(comp_geoms),
            "grouped_primitives": dict(grouped),
        })

    return composites


# ── combined pipeline ────────────────────────────────────────────────

def cluster_candidates(
    primitives_dict: dict[str, list[dict]],
    pt_per_inch: float,
) -> list[dict[str, Any]]:
    """Phase 2.1 -> Phase 2.2.  Returns JSON-serialisable candidates."""
    networks   = phase_2_1_intra_style(primitives_dict, pt_per_inch)
    composites = phase_2_2_inter_style(networks, pt_per_inch)
    return composites


# ── visualisation (fitz / PyMuPDF) ───────────────────────────────────

NEON_PALETTE = [
    (1, 0, 0.25),    # neon red
    (0, 1, 1),       # cyan
    (1, 0, 1),       # magenta
    (0, 1, 0),       # green
    (1, 0.5, 0),     # orange
    (1, 1, 0),       # yellow
    (0.5, 0, 1),     # purple
    (0, 0.5, 1),     # sky blue
    (1, 0.5, 0.5),   # salmon
    (0.5, 1, 0.5),   # light green
]


def _draw_prims_on_shape(shape, prims, color, line_width):
    """Draw a list of Step-1 primitives onto a fitz.Shape."""
    import fitz
    for prim in prims:
        pts = prim["points"]
        kind = prim["type"]
        if kind == "line" and len(pts) >= 2:
            shape.draw_line(fitz.Point(pts[0]), fitz.Point(pts[1]))
            shape.finish(color=color, width=line_width)
        elif kind == "polygon" and len(pts) >= 3:
            fp = [fitz.Point(p) for p in pts]
            shape.draw_polyline(fp + [fp[0]])
            shape.finish(color=color, fill=color, width=0.5,
                         fill_opacity=0.35)
        elif kind == "curve" and len(pts) == 4:
            shape.draw_bezier(*(fitz.Point(p) for p in pts))
            shape.finish(color=color, width=line_width)


def generate_candidate_overlay(
    pdf_path: str,
    candidate: dict,
    output_path: str,
    all_primitives: dict[str, list[dict]] | None = None,
    page_number: int = 0,
    render_scale: float = 2.0,
) -> None:
    """
    Single-candidate overlay.  Each style within the composite gets its
    own neon colour so you can visually confirm Phase 2.2 merging.
    """
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    # fade background
    bg = page.new_shape()
    bg.draw_rect(page.rect)
    bg.finish(fill=(1, 1, 1), fill_opacity=0.70)
    bg.commit()

    # context: all primitives in faint grey
    if all_primitives:
        ctx = page.new_shape()
        for prims in all_primitives.values():
            _draw_prims_on_shape(ctx, prims, (0.72, 0.72, 0.72), 0.25)
        ctx.commit()

    # candidate: one neon colour per style
    for rank, (style_key, prims) in enumerate(
            candidate["grouped_primitives"].items()):
        color = NEON_PALETTE[rank % len(NEON_PALETTE)]
        fg = page.new_shape()
        _draw_prims_on_shape(fg, prims, color, 2.0)
        fg.commit()

    # bounding box
    bb = candidate["bounding_box"]
    bx = page.new_shape()
    bx.draw_rect(fitz.Rect(bb[0], bb[1], bb[2], bb[3]))
    bx.finish(color=(0, 1, 1), width=1.0, dashes="[4] 0")
    bx.commit()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale),
                    alpha=False).save(output_path)
    doc.close()


def generate_summary_overlay(
    pdf_path: str,
    candidates: list[dict],
    output_path: str,
    all_primitives: dict[str, list[dict]] | None = None,
    page_number: int = 0,
    top_n: int = 10,
    render_scale: float = 2.0,
) -> None:
    """Top-N candidates colour-coded on a single image."""
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    bg = page.new_shape()
    bg.draw_rect(page.rect)
    bg.finish(fill=(1, 1, 1), fill_opacity=0.70)
    bg.commit()

    if all_primitives:
        ctx = page.new_shape()
        for prims in all_primitives.values():
            _draw_prims_on_shape(ctx, prims, (0.78, 0.78, 0.78), 0.2)
        ctx.commit()

    for rank, cand in enumerate(candidates[:top_n]):
        color = NEON_PALETTE[rank % len(NEON_PALETTE)]
        fg = page.new_shape()
        for prims in cand["grouped_primitives"].values():
            _draw_prims_on_shape(fg, prims, color, 1.8)
        fg.commit()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale),
                    alpha=False).save(output_path)
    doc.close()
    print(f"  summary overlay -> {output_path}")


# ── CLI entry point ──────────────────────────────────────────────────

KNOWN_SCALES: dict[str, float] = {
    "352 AA copy 2.pdf":      0.75,
    "second_floor_352.pdf":   0.75,
    "custom_floor_plan.pdf":  1.5,
    "main_st_ex.pdf":         0.6,
    "main_st_ex2.pdf":        0.6,
}
DEFAULT_PT_PER_INCH = 1.0


def run(
    patterns_json_path: str,
    pdf_path: str | None = None,
    output_dir: str | None = None,
    pt_per_inch: float | None = None,
    top_n: int = 10,
) -> list[dict]:
    """Load Step-1 JSON -> Phase 2.1 -> Phase 2.2 -> overlays."""
    with open(patterns_json_path) as fh:
        data = json.load(fh)

    primitives_dict = (data["grouped"] if isinstance(data, dict)
                       and "grouped" in data else data)

    if pt_per_inch is None:
        parent = os.path.basename(
            os.path.dirname(patterns_json_path)).lower()
        for pdf_name, scale in KNOWN_SCALES.items():
            tag = pdf_name.replace(".pdf", "").replace(" ", "_").lower()
            if tag in parent:
                pt_per_inch = scale
                break
        if pt_per_inch is None:
            pt_per_inch = DEFAULT_PT_PER_INCH

    candidates = cluster_candidates(primitives_dict, pt_per_inch)

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(patterns_json_path), "step2")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "candidates.json"), "w") as fh:
        json.dump(candidates, fh, indent=2)

    if pdf_path and os.path.isfile(pdf_path):
        generate_summary_overlay(
            pdf_path, candidates,
            os.path.join(output_dir, "summary_overlay.png"),
            all_primitives=primitives_dict, top_n=top_n)
        for c in candidates[:top_n]:
            generate_candidate_overlay(
                pdf_path, c,
                os.path.join(
                    output_dir,
                    f"candidate_{c['candidate_id']}_overlay.png"),
                all_primitives=primitives_dict)

    print(f"Step 2 complete: {len(candidates)} candidates -> {output_dir}")
    return candidates


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python step2_cluster.py <patterns.json> "
              "[pdf_path] [output_dir] [pt_per_inch]")
        sys.exit(1)
    run(
        sys.argv[1],
        pdf_path=sys.argv[2] if len(sys.argv) > 2 else None,
        output_dir=sys.argv[3] if len(sys.argv) > 3 else None,
        pt_per_inch=float(sys.argv[4]) if len(sys.argv) > 4 else None,
    )
