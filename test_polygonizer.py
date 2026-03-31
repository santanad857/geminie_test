#!/usr/bin/env python3
"""
Test the new room_polygonizer on real floor plan data.

Uses room_pipeline's existing wall extraction + door sealing (Steps 0-3),
then feeds the result into room_polygonizer.extract_rooms() (replacing Step 4).

Outputs:
    output/polygonizer_overlay.pdf  — colored room fills on the original plan
    output/polygonizer_debug.png    — dark-background debug visualisation
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import fitz
import numpy as np
from shapely.geometry import LineString

from pipelines.room_pipeline import (
    WallSegment,
    ExposedEndpoint,
    RoomPolygon,
    extract_wall_segments_for_room_pipeline,
    find_exposed_endpoints,
    cast_rays_and_find_doors,
    compute_max_door_width,
    load_env,
    _PALETTE,
    _polygon_area,
    _centroid,
)
from pipelines.room_polygonizer import extract_rooms


def wall_segments_to_linestrings(
    segments: list[WallSegment],
) -> list[LineString]:
    """Convert WallSegment list to Shapely LineString list."""
    lines = []
    for seg in segments:
        length = math.hypot(seg.p2[0] - seg.p1[0], seg.p2[1] - seg.p1[1])
        if length > 0.5:
            lines.append(LineString([seg.p1, seg.p2]))
    return lines


def polys_to_room_polygons(polys, scale_ratio):
    """Convert shapely Polygons to RoomPolygon dataclass instances."""
    sqpt_to_sqft = (scale_ratio / 72.0) ** 2 / 144.0
    rooms = []
    for i, poly in enumerate(polys):
        verts = list(poly.exterior.coords[:-1])
        area_sqpt = poly.area
        area_sqft = area_sqpt * sqpt_to_sqft
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        rooms.append(RoomPolygon(
            room_id=i + 1,
            name=f"ROOM-{i + 1}",
            vertices=verts,
            area_sqpt=area_sqpt,
            area_sqft=area_sqft,
            color=_PALETTE[i % len(_PALETTE)],
            centroid=(cx, cy),
        ))
    return rooms


def render_overlay_pdf(pdf_path, rooms, door_segs, output_path, page_index=0):
    """Draw colored room polygons and door rays on the PDF."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]

    for room in rooms:
        if len(room.vertices) < 3:
            continue
        pts = [fitz.Point(v[0], v[1]) for v in room.vertices]
        shape = page.new_shape()
        shape.draw_polyline(pts + [pts[0]])
        r, g, b = room.color
        shape.finish(
            color=(r * 0.6, g * 0.6, b * 0.6),
            fill=(r, g, b),
            fill_opacity=0.30,
            width=0.5,
        )
        shape.commit()

        # Label.
        cx, cy = room.centroid
        label = f"{room.name}\n{room.area_sqft:.0f} ft²"
        half_w = max(40, len(room.name) * 3.5)
        rect = fitz.Rect(cx - half_w, cy - 10, cx + half_w, cy + 10)
        page.insert_textbox(rect, label, fontsize=6, fontname="helv",
                            color=(0, 0, 0), align=fitz.TEXT_ALIGN_CENTER)

    for ds in door_segs:
        shape = page.new_shape()
        shape.draw_line(fitz.Point(ds.p1[0], ds.p1[1]),
                        fitz.Point(ds.p2[0], ds.p2[1]))
        shape.finish(color=(0.1, 0.3, 0.9), width=0.8, dashes="[4 2]")
        shape.commit()

    doc.save(output_path)
    doc.close()
    return output_path


def render_debug_png(pdf_path, wall_segments, door_segments, exposed,
                     rooms, output_path, page_index=0, dpi=150):
    """Debug image: dark bg, white walls, green door rays, red endpoints,
    colored room fills."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    rot_mat = page.rotation_matrix
    scale = dpi / 72.0
    w = int(page.rect.width * scale) + 1
    h = int(page.rect.height * scale) + 1
    doc.close()
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)

    def _pt(x, y):
        rp = fitz.Point(x, y) * rot_mat
        return (int(rp.x * scale), int(rp.y * scale))

    # Room fills.
    for room in rooms:
        pts = np.array([_pt(v[0], v[1]) for v in room.vertices], dtype=np.int32)
        col = tuple(int(c * 180) for c in reversed(room.color))
        cv2.fillPoly(canvas, [pts], col)

    # Wall segments (white).
    for seg in wall_segments:
        cv2.line(canvas, _pt(*seg.p1), _pt(*seg.p2), (255, 255, 255), 1)

    # Door segments (green).
    for seg in door_segments:
        cv2.line(canvas, _pt(*seg.p1), _pt(*seg.p2), (0, 220, 0), 2)

    # Exposed endpoints (red dots).
    for ep in exposed:
        cv2.circle(canvas, _pt(*ep.point), 4, (0, 0, 255), -1)

    cv2.imwrite(output_path, canvas)
    return output_path


def main():
    load_env()
    os.makedirs("output", exist_ok=True)

    pdf_path = "./data/352 AA copy 2.pdf"
    scale_ratio = 96.0
    max_door_ft = 20.0
    epsilon = 3.0

    max_dw = compute_max_door_width(scale_ratio, max_door_ft)

    print("=" * 62)
    print("   ROOM POLYGONIZER TEST — 352 AA copy 2")
    print("=" * 62)
    print(f"  Scale: {scale_ratio}  |  Max door: {max_dw:.1f} pt ({max_door_ft} ft)")

    # ── Step 0: Wall segments ────────────────────────────────────────
    print("\n[Step 0] Loading wall vectors ...")
    segments, raw_segments, wall_source = extract_wall_segments_for_room_pipeline(
        pdf_path, page_index=0, wall_vector_mode="auto",
    )
    print(f"  Source: {wall_source}")
    print(f"  Centerlines: {len(segments)} | Raw: {len(raw_segments)}")

    # ── Step 1: Exposed endpoints ────────────────────────────────────
    print("\n[Step 1] Finding exposed endpoints ...")
    exposed = find_exposed_endpoints(segments, epsilon)

    # ── Step 2: Ray casting ──────────────────────────────────────────
    print("\n[Step 2] Casting rays to seal doorways ...")
    door_segs = cast_rays_and_find_doors(exposed, segments, max_dw)

    # ── Step 3: Convert to LineStrings ───────────────────────────────
    print("\n[Step 3] Converting to LineStrings ...")
    all_segs = segments + door_segs
    wall_lines = wall_segments_to_linestrings(all_segs)
    print(f"  Total LineStrings: {len(wall_lines)}")

    # ── Step 4: room_polygonizer ─────────────────────────────────────
    print("\n[Step 4] Running room_polygonizer.extract_rooms() ...")
    room_polys = extract_rooms(
        wall_lines,
        max_gap_tolerance=48.0,
        min_room_area_sqft=20.0,
        scale_ratio=scale_ratio,
    )

    # ── Convert to RoomPolygon for rendering ─────────────────────────
    rooms = polys_to_room_polygons(room_polys, scale_ratio)

    # ── Results table ────────────────────────────────────────────────
    sqpt_to_sqft = (scale_ratio / 72.0) ** 2 / 144.0
    print()
    print("=" * 62)
    print(f"  RESULTS: {len(rooms)} room polygons")
    print("=" * 62)
    print(f"  {'#':<4s}  {'AREA (ft²)':>10s}  {'VALID':>6s}  {'VERTICES':>8s}  BOUNDS")
    print("-" * 62)

    total_sqft = 0.0
    for rm in rooms:
        total_sqft += rm.area_sqft
        bx0, by0, bx1, by1 = room_polys[rm.room_id - 1].bounds
        print(
            f"  {rm.room_id:<4d}  {rm.area_sqft:>10.1f}  "
            f"{'True':>6s}  {len(rm.vertices):>8d}  "
            f"({bx0:.0f},{by0:.0f})-({bx1:.0f},{by1:.0f})"
        )

    print("-" * 62)
    print(f"  TOTAL: {total_sqft:.1f} ft²")
    print("=" * 62)

    # ── Render visual outputs ────────────────────────────────────────
    print("\n  Generating visual outputs ...")

    overlay_path = render_overlay_pdf(
        pdf_path, rooms, door_segs,
        "output/polygonizer_overlay.pdf",
    )
    print(f"  PDF overlay → {overlay_path}")

    debug_path = render_debug_png(
        pdf_path, segments, door_segs, exposed, rooms,
        "output/polygonizer_debug.png",
    )
    print(f"  Debug PNG   → {debug_path}")

    # Also render overlay as PNG for quick viewing.
    doc = fitz.open(overlay_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    pix.save("output/polygonizer_overlay.png")
    doc.close()
    print(f"  Overlay PNG → output/polygonizer_overlay.png")

    print("\n  Done.\n")
    return rooms


if __name__ == "__main__":
    rooms = main()
