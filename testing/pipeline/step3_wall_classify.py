#!/usr/bin/env python3
"""
Step 3 -- VLM Multi-Style Wall Classification
=============================================

For each Candidate Composite from Step 2, render ALL style layers
simultaneously in distinct neon colours onto a single overlay image and
send it to a Vision-Language Model in ONE API call per candidate.

The VLM is shown every style at once and asked "which colour(s) are
structural walls?" — giving it the comparison context it needs to
distinguish wall outlines from dimension lines, stairs, fixtures, etc.

This replaces the previous per-style sequential approach that suffered from:
  - N API calls per candidate (expensive, slow)
  - Crop-dependent inconsistency (same style got different verdicts
    at different zoom levels)
  - Unanswerable YES/NO questions for mixed-content styles

Rendering strategy:
    1. Faded PDF page (~30 % opacity via 70 % white overlay)
    2. Each style layer drawn in a distinct named neon colour
    3. Cropped to candidate bounding box with padding
    4. One VLM call per candidate; response lists which colour(s) = walls
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

import fitz  # PyMuPDF
import openai

# ── constants ─────────────────────────────────────────────────────────

GROUND_TRUTH_CANDIDATES: dict[str, list[int]] = {
    "352_AA_copy_2":     [2, 3, 4, 5],
    "custom_floor_plan": [1],
    "main_st_ex":        [1],
    "main_st_ex2":       [1, 2],
    "second_floor_352":  [2, 4],
    "2026.01.29_-_FMOC_A_-_50%_CD_copy": [1],
}

KNOWN_SCALES: dict[str, float] = {
    "352 AA copy 2.pdf":     0.75,
    "second_floor_352.pdf":  0.75,
    "custom_floor_plan.pdf": 1.5,
    "main_st_ex.pdf":        0.6,
    "main_st_ex2.pdf":       0.6,
}

# Named colours the VLM can reliably identify, paired with fitz RGB tuples.
# Order matters: most visually distinct colours first.
STYLE_COLORS: list[tuple[str, tuple[float, float, float]]] = [
    ("RED",     (1.0, 0.0, 0.0)),
    ("CYAN",    (0.0, 1.0, 1.0)),
    ("MAGENTA", (1.0, 0.0, 1.0)),
    ("GREEN",   (0.0, 0.9, 0.0)),
    ("ORANGE",  (1.0, 0.5, 0.0)),
    ("YELLOW",  (1.0, 1.0, 0.0)),
    ("BLUE",    (0.2, 0.4, 1.0)),
    ("PINK",    (1.0, 0.5, 0.75)),
    ("LIME",    (0.5, 1.0, 0.0)),
    ("PURPLE",  (0.6, 0.0, 1.0)),
]

CONTEXT_GREY  = (0.72, 0.72, 0.72)
RESULT_GREEN  = (0.0, 0.9, 0.0)
RESULT_FAINT_RED = (1.0, 0.5, 0.5)
BBOX_PADDING_PT = 30.0

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))


# ── rendering helpers ─────────────────────────────────────────────────

def _is_fill_style(style_key: str) -> bool:
    """True when the style_key came from a fill bucket (not a stroke)."""
    return style_key.startswith("fill_")


def _draw_prims_on_shape(
    shape,
    prims: list[dict],
    color: tuple,
    line_width: float,
    outline_only: bool = False,
) -> None:
    """Draw Step-1 primitives onto a fitz.Shape.

    When *outline_only* is True, polygons are drawn as unfilled outlines
    (used for fill-based styles so they don't occlude stroke-based styles).
    """
    for prim in prims:
        pts  = prim["points"]
        kind = prim["type"]
        if kind == "line" and len(pts) >= 2:
            shape.draw_line(fitz.Point(pts[0]), fitz.Point(pts[1]))
            shape.finish(color=color, width=line_width)
        elif kind == "polygon" and len(pts) >= 3:
            fp = [fitz.Point(p) for p in pts]
            shape.draw_polyline(fp + [fp[0]])
            if outline_only:
                shape.finish(color=color, width=line_width)
            else:
                shape.finish(color=color, fill=color,
                             width=0.5, fill_opacity=0.35)
        elif kind == "curve" and len(pts) == 4:
            shape.draw_bezier(*(fitz.Point(p) for p in pts))
            shape.finish(color=color, width=line_width)


def render_multi_style_overlay(
    pdf_path: str,
    candidate: dict,
    page_number: int = 0,
    render_scale: float = 2.0,
    padding: float = BBOX_PADDING_PT,
) -> tuple[bytes, dict[str, str]]:
    """
    Render all style layers simultaneously, each in a distinct named
    neon colour, cropped to the candidate bounding box.

    Returns
    -------
    png_bytes : bytes
        The overlay image ready for base64 encoding.
    color_to_style : dict[str, str]
        Maps colour name (e.g. "RED") to style_key so the VLM response
        can be decoded back to style keys.
    """
    doc  = fitz.open(pdf_path)
    page = doc[page_number]

    # 1. Fade background
    bg = page.new_shape()
    bg.draw_rect(page.rect)
    bg.finish(fill=(1, 1, 1), fill_opacity=0.70)
    bg.commit()

    # 2. Partition styles: strokes first (full weight), fills last (outline only).
    #    Within each group sort by descending prim count.
    #    Fills drawn last so they never occlude stroke-based styles.
    all_styles = sorted(
        candidate["grouped_primitives"].items(),
        key=lambda kv: -len(kv[1]),
    )
    strokes = [(sk, p) for sk, p in all_styles if not _is_fill_style(sk)]
    fills   = [(sk, p) for sk, p in all_styles if     _is_fill_style(sk)]
    ordered = (strokes + fills)[:len(STYLE_COLORS)]

    color_to_style: dict[str, str] = {}
    for (style_key, prims), (color_name, color_rgb) in zip(ordered, STYLE_COLORS):
        is_fill = _is_fill_style(style_key)
        weight  = 0.6 if is_fill else 2.0
        shape   = page.new_shape()
        _draw_prims_on_shape(shape, prims, color_rgb, weight,
                             outline_only=is_fill)
        shape.commit()
        color_to_style[color_name] = style_key

    # 3. Crop to candidate bbox with padding
    bb   = candidate["bounding_box"]
    clip = fitz.Rect(
        bb[0] - padding, bb[1] - padding,
        bb[2] + padding, bb[3] + padding,
    ) * page.rotation_matrix & page.rect

    pixmap    = page.get_pixmap(
        matrix=fitz.Matrix(render_scale, render_scale),
        clip=clip,
        alpha=False,
    )
    png_bytes = pixmap.tobytes("png")
    doc.close()
    return png_bytes, color_to_style


def render_result_overlay(
    pdf_path: str,
    candidate: dict,
    style_classifications: dict[str, dict],
    output_path: str,
    page_number: int = 0,
    render_scale: float = 2.0,
    padding: float = BBOX_PADDING_PT,
) -> None:
    """
    Result overlay: YES-classified styles in neon green, NO in faint red.
    Cropped to candidate bbox.  Saved to *output_path*.
    """
    doc  = fitz.open(pdf_path)
    page = doc[page_number]

    bg = page.new_shape()
    bg.draw_rect(page.rect)
    bg.finish(fill=(1, 1, 1), fill_opacity=0.70)
    bg.commit()

    for style_key, prims in candidate["grouped_primitives"].items():
        verdict = style_classifications.get(style_key, {}).get("verdict", "NO")
        color   = RESULT_GREEN if verdict == "YES" else RESULT_FAINT_RED
        width   = 2.5          if verdict == "YES" else 0.5
        shape   = page.new_shape()
        _draw_prims_on_shape(shape, prims, color, width)
        shape.commit()

    bb   = candidate["bounding_box"]
    clip = fitz.Rect(
        bb[0] - padding, bb[1] - padding,
        bb[2] + padding, bb[3] + padding,
    ) * page.rotation_matrix & page.rect

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    page.get_pixmap(
        matrix=fitz.Matrix(render_scale, render_scale),
        clip=clip,
        alpha=False,
    ).save(output_path)
    doc.close()


# ── VLM integration ──────────────────────────────────────────────────

def build_multi_style_prompt(color_to_style: dict[str, str]) -> str:
    """
    Build the VLM prompt that lists colour→style and asks which
    colours represent structural walls.
    """
    style_lines = "\n".join(
        f"  - {color}: {style}"
        for color, style in color_to_style.items()
    )
    return (
        f"This architectural floor plan has {len(color_to_style)} drawing "
        f"style layers, each highlighted in a different colour:\n"
        f"{style_lines}\n\n"
        "Which of these colours represent STRUCTURAL WALLS? "
        "Structural walls form continuous room boundaries, appear as "
        "parallel pairs or outlines, and span significant distances "
        "across the plan.\n\n"
        "NOT walls: dimension lines, text underlines, arrows, door swings, "
        "furniture, fixtures, appliances, stairs, hatching fill patterns, "
        "or decorative elements.\n\n"
        "Respond with ONLY the colour names that represent walls, "
        "comma-separated (e.g. 'RED, CYAN'). "
        "If none represent walls, respond with 'NONE'."
    )


def parse_multi_style_response(
    response_text: str,
    color_to_style: dict[str, str],
) -> dict[str, dict]:
    """
    Parse the VLM response (a list of colour names) into per-style
    verdict dicts compatible with the existing output schema.

    Returns
    -------
    dict[style_key, {"verdict": "YES"|"NO", "confidence": str}]
        Covers every style in *color_to_style*.
    """
    text = response_text.strip().upper()

    # Identify which palette colour names appear in the response
    wall_colors: set[str] = set()
    for color_name in color_to_style:
        # Match whole words to avoid "BLUE" matching "BLUEBELL" etc.
        import re
        if re.search(rf"\b{color_name}\b", text):
            wall_colors.add(color_name)

    # Confidence: high if we got a clean list or explicit NONE
    confidence = (
        "high"
        if (wall_colors or "NONE" in text)
        else "low"
    )

    return {
        style_key: {
            "verdict":    "YES" if color_name in wall_colors else "NO",
            "confidence": confidence,
        }
        for color_name, style_key in color_to_style.items()
    }


# Keep the single-response parser for test compatibility
def parse_vlm_response(response_text: str) -> dict[str, str]:
    """
    Legacy per-style YES/NO parser (used only in tests).
    """
    text = response_text.strip().upper()
    if text in ("YES", "NO"):
        return {"verdict": text, "confidence": "high"}
    if text.startswith("YES"):
        return {"verdict": "YES", "confidence": "medium"}
    if text.startswith("NO"):
        return {"verdict": "NO", "confidence": "medium"}
    if "YES" in text:
        return {"verdict": "YES", "confidence": "low"}
    if "NO" in text:
        return {"verdict": "NO", "confidence": "low"}
    return {"verdict": "NO", "confidence": "low"}


# ── classification orchestrator ───────────────────────────────────────

def classify_candidate(
    pdf_path: str,
    candidate: dict,
    api_key: str,
    page_number: int = 0,
    output_dir: str | None = None,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """
    Classify all style layers in a single candidate with ONE VLM call.

    Renders every style in a distinct named colour, sends the image to
    the VLM, and asks which colours represent structural walls.

    Returns the same schema as the previous per-style implementation::

        {"candidate_id": int,
         "bounding_box": [...],
         "style_classifications": {style_key: {"verdict", "confidence"}},
         "wall_primitives": {style_key: [prims]},   # YES styles only
         "vlm_response": str,
         "color_to_style": {color_name: style_key}
        }
    """
    cid = candidate["candidate_id"]

    # Render multi-colour overlay
    png, color_to_style = render_multi_style_overlay(
        pdf_path, candidate, page_number=page_number,
    )

    # Persist overlay for inspection
    if output_dir:
        cand_dir = os.path.join(output_dir, f"candidate_{cid}")
        os.makedirs(cand_dir, exist_ok=True)
        with open(
            os.path.join(cand_dir, "multi_style_overlay.png"), "wb",
        ) as fh:
            fh.write(png)

    # One VLM call
    prompt = build_multi_style_prompt(color_to_style)
    b64    = base64.b64encode(png).decode("utf-8")
    client = openai.OpenAI(api_key=api_key)
    resp   = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text",      "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                }},
            ],
        }],
        max_tokens=50,
    )
    response_text = resp.choices[0].message.content
    print(f"  candidate {cid} VLM -> {response_text!r}")

    # Parse colour names back to style verdicts
    classifications = parse_multi_style_response(response_text, color_to_style)

    # Styles that exceeded the palette cap get a default NO
    for sk in candidate["grouped_primitives"]:
        if sk not in classifications:
            classifications[sk] = {"verdict": "NO", "confidence": "low"}

    # Print per-style summary
    for sk, res in classifications.items():
        print(f"    {res['verdict']:3s} ({res['confidence']:6s}) | {sk}")

    wall_prims = {
        sk: candidate["grouped_primitives"][sk]
        for sk, res in classifications.items()
        if res["verdict"] == "YES"
    }

    return {
        "candidate_id":        cid,
        "bounding_box":        candidate["bounding_box"],
        "style_classifications": classifications,
        "wall_primitives":     wall_prims,
        "vlm_response":        response_text,
        "color_to_style":      color_to_style,
    }


# ── top-level runner ──────────────────────────────────────────────────

def load_api_key() -> str:
    """Load OPENAI_API_KEY from the project .env file."""
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found in environment / .env")
    return key


def run(
    plan_name: str,
    candidates_json_path: str | None = None,
    pdf_path: str | None = None,
    pt_per_inch: float | None = None,
    page_number: int = 0,
    output_dir: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Run Step 3 classification on a plan's ground-truth candidates."""
    if candidates_json_path is None:
        candidates_json_path = os.path.join(
            PROJECT_ROOT, "output", plan_name, "step2", "candidates.json",
        )

    if pdf_path is None:
        for pdf_name, scale in KNOWN_SCALES.items():
            tag = pdf_name.replace(".pdf", "").replace(" ", "_").lower()
            if tag == plan_name.lower():
                pdf_path = os.path.join(
                    PROJECT_ROOT, "example_plans", pdf_name,
                )
                if pt_per_inch is None:
                    pt_per_inch = scale
                break

    if output_dir is None:
        output_dir = os.path.join(
            PROJECT_ROOT, "output", plan_name, "step3",
        )
    os.makedirs(output_dir, exist_ok=True)

    if api_key is None:
        api_key = load_api_key()

    with open(candidates_json_path) as fh:
        all_candidates = json.load(fh)

    target_ids = GROUND_TRUTH_CANDIDATES.get(plan_name, [])
    candidates = [c for c in all_candidates if c["candidate_id"] in target_ids]
    if not candidates:
        print(f"No matching candidates for plan '{plan_name}'")
        return []

    print(
        f"Step 3: classifying {len(candidates)} candidate(s) "
        f"for '{plan_name}' ({len(candidates)} VLM call(s) total)"
    )

    results: list[dict] = []
    for cand in candidates:
        result = classify_candidate(
            pdf_path, cand, api_key,
            page_number=page_number, output_dir=output_dir,
        )
        results.append(result)

        # Result overlay (green YES / faint-red NO)
        if pdf_path:
            render_result_overlay(
                pdf_path, cand, result["style_classifications"],
                os.path.join(
                    output_dir,
                    f"candidate_{cand['candidate_id']}",
                    "result_overlay.png",
                ),
                page_number=page_number,
            )

    with open(os.path.join(output_dir, "classifications.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"Step 3 complete: {len(results)} candidates -> {output_dir}")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python step3_wall_classify.py <plan_name> "
            "[candidates.json] [pdf_path]",
        )
        sys.exit(1)
    run(
        sys.argv[1],
        candidates_json_path=sys.argv[2] if len(sys.argv) > 2 else None,
        pdf_path=sys.argv[3]             if len(sys.argv) > 3 else None,
    )
