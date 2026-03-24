#!/usr/bin/env python3
"""
Wall Detection Pipeline for Architectural Floor Plans
======================================================

Strategy:
  1. Extract all vector paths from the PDF via PyMuPDF.
  2. Use a single VLM call (Claude) to identify ONE wall segment (seed).
  3. Extract a deterministic "fingerprint" of wall vectors from that seed region
     (hatch color/width/angle, border width).
  4. Match the fingerprint across every vector in the document — pure geometry,
     zero probabilistic inference.
  5. Render an overlay image highlighting all detected walls.

The VLM is used exactly once to bootstrap the process.  Everything after
that is deterministic pattern matching on exact vector properties.

Usage:
    python wall_pipeline.py [path_to_pdf]
"""

import base64
import json
import math
import os
import re
import sys
from collections import Counter

import anthropic
import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_env():
    """Load key=value pairs from .env beside this script."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def render_page(page, dpi=150):
    """Render a page to PNG bytes.  Returns (png_bytes, width_px, height_px)."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png"), pix.width, pix.height


# ---------------------------------------------------------------------------
# Step 2 — VLM seed locator
# ---------------------------------------------------------------------------

def vlm_identify_wall(page, dpi=150):
    """Ask Claude to locate one wall.  Returns a fitz.Rect in PDF-point space."""

    img_bytes, w_px, h_px = render_page(page, dpi)
    img_b64 = base64.b64encode(img_bytes).decode()

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = (
        f"This is an architectural floor plan image ({w_px} x {h_px} pixels).\n\n"
        "Identify ONE long, straight wall segment in the MAIN floor plan. Choose a wall that is clearly visible, at least ~100 pixels long, "
        "and away from door or window openings.\n\n"
        "Return ONLY a JSON object with the bounding box in pixel coordinates:\n"
        '{"x1": <left>, "y1": <top>, "x2": <right>, "y2": <bottom>}\n\n'
        "Return ONLY the JSON object, nothing else."
    )

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    text = resp.content[0].text
    m = re.search(r"\{[^}]+\}", text)
    if not m:
        raise ValueError(f"VLM did not return JSON: {text}")

    bbox = json.loads(m.group())

    # Clamp to image bounds
    bbox["x1"] = max(0, min(bbox["x1"], w_px))
    bbox["y1"] = max(0, min(bbox["y1"], h_px))
    bbox["x2"] = max(0, min(bbox["x2"], w_px))
    bbox["y2"] = max(0, min(bbox["y2"], h_px))

    print(f"  VLM pixel bbox: ({bbox['x1']}, {bbox['y1']}) -> ({bbox['x2']}, {bbox['y2']})")

    # Pixel → rotated page-point coordinates, then derotate to mediabox
    # coordinates (which get_drawings() uses).  get_pixmap() renders the
    # rotated page, so VLM pixel coords are in page.rect space.
    # get_drawings() returns mediabox coordinates.  We must derotate.
    s = 72.0 / dpi
    rotated_rect = fitz.Rect(bbox["x1"] * s, bbox["y1"] * s,
                             bbox["x2"] * s, bbox["y2"] * s)
    derot = page.derotation_matrix
    return rotated_rect * derot


# ---------------------------------------------------------------------------
# Step 3 — Fingerprint extraction
# ---------------------------------------------------------------------------

def _dominant_angle(items):
    """Return the dominant line angle (0-180°) from a drawing's items list."""
    angles = []
    for item in items:
        if item[0] == "l":
            dx = item[2].x - item[1].x
            dy = item[2].y - item[1].y
            if math.hypot(dx, dy) > 0.05:
                angles.append(math.degrees(math.atan2(dy, dx)) % 180)
    if not angles:
        return None
    bins = Counter(round(a / 5) * 5 for a in angles)
    return bins.most_common(1)[0][0]


def extract_fingerprint(drawings, seed_rect):
    """Derive the wall vector signature from drawings inside *seed_rect*."""

    expanded = seed_rect + (-15, -15, 15, 15)
    region = [d for d in drawings if d["rect"].intersects(expanded)]
    print(f"  {len(region)} drawings in seed region")

    # --- hatch pattern (non-black, thin, many lines, angled) ---------------
    hatch_candidates = [
        d for d in region
        if d.get("color")
        and d["color"] != (0.0, 0.0, 0.0)
        and len(d["items"]) >= 3
        and 0 < (d.get("width") or 0) < 0.2
    ]

    if not hatch_candidates:
        raise ValueError(
            "No hatch pattern found in seed region. "
            "The VLM may have pointed to a non-wall area."
        )

    ref = max(hatch_candidates, key=lambda d: len(d["items"]))
    angle = _dominant_angle(ref["items"])

    # --- border widths (all distinct black line widths in region > 0.3) -----
    border_ws = Counter()
    for d in region:
        if d.get("color") == (0.0, 0.0, 0.0) and (d.get("width") or 0) > 0.3:
            border_ws[round(d["width"], 1)] += 1

    # Collect ALL significant border widths (different plan areas may use
    # different widths for the same wall borders).
    border_widths = sorted(border_ws.keys()) if border_ws else [0.6]

    return {
        "hatch_color": ref["color"],
        "hatch_width": ref["width"],
        "hatch_angle": angle if angle is not None else 45,
        "border_widths": border_widths,
    }


# ---------------------------------------------------------------------------
# Step 4 — Deterministic global matching
# ---------------------------------------------------------------------------

def _angle_close(a, b, tol=10):
    """Check if two angles (0-180°) are within *tol* degrees."""
    diff = abs(a - b)
    return diff <= tol or (180 - diff) <= tol


def find_all_walls(drawings, fp):
    """Return (hatches, borders) — lists of (index, drawing) tuples."""

    hc = fp["hatch_color"]
    hw = fp["hatch_width"]
    ha = fp["hatch_angle"]

    # Phase 1: every drawing whose colour / width / angle matches the hatch
    hatches = []
    for i, d in enumerate(drawings):
        c, w = d.get("color"), d.get("width")
        if not c or not w:
            continue
        if not all(abs(a - b) < 0.05 for a, b in zip(c, hc)):
            continue
        if abs(w - hw) > 0.02:
            continue
        if len(d["items"]) < 1:
            continue
        dom = _dominant_angle(d["items"])
        if dom is None or not _angle_close(dom, ha):
            continue
        hatches.append((i, d))

    # Phase 2: black lines spatially adjacent to any hatch region.
    # Accept any border width that was observed in the fingerprint region,
    # plus a generous tolerance (wall borders vary across the plan).
    hatch_rects = [d["rect"] for _, d in hatches]
    border_ws = fp["border_widths"]
    bw_min = min(border_ws) - 0.2
    bw_max = max(border_ws) + 0.2

    borders = []
    for i, d in enumerate(drawings):
        if d.get("color") != (0.0, 0.0, 0.0):
            continue
        w = d.get("width")
        if not w or w < bw_min or w > bw_max:
            continue
        for hr in hatch_rects:
            if d["rect"].intersects(hr + (-3, -3, 3, 3)):
                borders.append((i, d))
                break

    return hatches, borders


# ---------------------------------------------------------------------------
# Step 5 — Overlay rendering
# ---------------------------------------------------------------------------

def generate_overlay(pdf_path, hatches, borders, output_path="wall_overlay.png", dpi=200):
    """Re-draw the exact detected wall vectors in red on the PDF, then render.

    Every line segment from the matched drawings is redrawn at its precise
    coordinates — no bounding-box approximation, no padding.
    """

    doc = fitz.open(pdf_path)
    page = doc[0]

    # --- hatch vectors (redraw each line segment in red) -------------------
    shape = page.new_shape()
    for _, d in hatches:
        for item in d["items"]:
            if item[0] == "l":
                shape.draw_line(item[1], item[2])
            elif item[0] == "re":
                shape.draw_rect(item[1])
            elif item[0] == "qu":
                shape.draw_quad(item[1])
    shape.finish(color=(1, 0, 0), width=0.5)
    shape.commit()

    # --- border vectors (preserve original stroke width per drawing) -------
    from collections import defaultdict
    by_width = defaultdict(list)
    for _, d in borders:
        w = d.get("width") or 0.6
        for item in d["items"]:
            by_width[w].append(item)

    for w, items in by_width.items():
        s = page.new_shape()
        for item in items:
            if item[0] == "l":
                s.draw_line(item[1], item[2])
            elif item[0] == "re":
                s.draw_rect(item[1])
            elif item[0] == "qu":
                s.draw_quad(item[1])
        s.finish(color=(1, 0, 0), width=w)
        s.commit()

    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    pix.save(output_path)

    doc.close()
    return output_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    load_env()

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "./352 AA copy 2.pdf"

    print("=" * 60)
    print("  WALL DETECTION PIPELINE")
    print("=" * 60)

    # -- Step 1: load -------------------------------------------------------
    print("\n[1/5] Loading PDF and extracting vectors...")
    doc = fitz.open(pdf_path)
    page = doc[0]
    drawings = page.get_drawings()
    print(f"  Page: {page.rect}  rotation={page.rotation}")
    print(f"  Total vector drawings: {len(drawings)}")

    # -- Steps 2-3: VLM seed + fingerprint (with retry) ----------------------
    MAX_VLM_ATTEMPTS = 3
    fp = None
    for attempt in range(1, MAX_VLM_ATTEMPTS + 1):
        print(f"\n[2/5] Asking VLM to identify one wall segment (attempt {attempt})...")
        try:
            seed_rect = vlm_identify_wall(page, dpi=150)
            print(f"  Seed rect (PDF pts): {seed_rect}")

            print("\n[3/5] Extracting wall fingerprint from seed region...")
            fp = extract_fingerprint(drawings, seed_rect)
            print(f"  Hatch colour : {tuple(round(c, 4) for c in fp['hatch_color'])}")
            print(f"  Hatch width  : {fp['hatch_width']:.4f}")
            print(f"  Hatch angle  : {fp['hatch_angle']}°")
            print(f"  Border widths: {fp['border_widths']}")
            break  # success
        except ValueError as e:
            print(f"  {e}")
            if attempt == MAX_VLM_ATTEMPTS:
                raise RuntimeError(
                    f"Could not extract a wall fingerprint after {MAX_VLM_ATTEMPTS} "
                    "VLM attempts. The VLM consistently pointed to non-wall areas."
                )

    # -- Step 4: global match -----------------------------------------------
    print("\n[4/5] Matching fingerprint across all vectors...")
    hatches, borders = find_all_walls(drawings, fp)
    print(f"  Hatch regions found  : {len(hatches)}")
    print(f"  Border elements found: {len(borders)}")

    # -- Step 5: overlay ----------------------------------------------------
    print("\n[5/5] Generating overlay image...")
    out = generate_overlay(pdf_path, hatches, borders)
    print(f"  Saved to: {out}")

    # -- Summary ------------------------------------------------------------
    h_items = sum(len(d["items"]) for _, d in hatches)
    b_items = sum(len(d["items"]) for _, d in borders)
    print("\n" + "-" * 60)
    print("  SUMMARY")
    print("-" * 60)
    print(f"  Total drawings in PDF     : {len(drawings)}")
    print(f"  Wall hatch drawings       : {len(hatches)}  ({h_items} line segments)")
    print(f"  Wall border drawings      : {len(borders)}  ({b_items} line segments)")
    print(f"  Total wall drawings       : {len(hatches) + len(borders)}")
    print(f"  Wall vector coverage      : "
          f"{(len(hatches) + len(borders)) / len(drawings) * 100:.1f}%")

    doc.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
