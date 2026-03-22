"""
Main driver — extract wall geometries from an architectural PDF
and produce a visual overlay PDF for verification.

Usage:
    python main.py <pdf_path> [page_num]

If no arguments are provided, uses the default sample plan.
Outputs:
    walls_overlay.pdf  — original plan with red wall highlights
"""

import sys
from wall_extractor import (
    extract_walls,
    render_overlay,
    min_bounding_thickness,
    max_bounding_length,
    SCALE_FACTOR,
)


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "./352 AA copy 2.pdf"
    page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    walls = extract_walls(pdf_path, page_num)

    # ---- Print summary ----
    print(f"\n{'=' * 60}")
    print(f" RESULTS: {len(walls)} wall polygon(s)")
    print(f"{'=' * 60}")

    for idx, wall in enumerate(walls):
        try:
            t_pts = min_bounding_thickness(wall)
            l_pts = max_bounding_length(wall)
        except Exception:
            t_pts = l_pts = 0.0

        real_t = t_pts / SCALE_FACTOR if SCALE_FACTOR else 0
        real_l = l_pts / SCALE_FACTOR if SCALE_FACTOR else 0
        b = wall.bounds

        print(
            f"  Wall {idx + 1:3d}:  "
            f"{real_t:5.1f}\" x {real_l / 12:5.1f}'   "
            f"bounds=({b[0]:.0f},{b[1]:.0f})–({b[2]:.0f},{b[3]:.0f})   "
            f"area={wall.area:.0f} pts²"
        )

    # ---- Generate visual overlay ----
    out = render_overlay(pdf_path, walls, page_num=page_num)
    print(f"\n  Overlay written to: {out}")


if __name__ == "__main__":
    main()
