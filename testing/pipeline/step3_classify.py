#!/usr/bin/env python3
"""
Step 3 -- VLM Style-Layer Classification
=========================================

For each Candidate Composite from Step 2, render each style layer as an
isolated overlay image and classify it as structural walls (YES) or not
(NO) using a Vision-Language Model (GPT-4o via OpenAI API).

Rendering strategy ("Red Line"):
    1. Faded PDF page (~30 % opacity via 70 % white overlay)
    2. All candidate primitives in faint grey (context)
    3. Current style layer's primitives in thick neon red
    4. Cropped to candidate bounding box with padding
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
}

KNOWN_SCALES: dict[str, float] = {
    "352 AA copy 2.pdf":     0.75,
    "second_floor_352.pdf":  0.75,
    "custom_floor_plan.pdf": 1.5,
    "main_st_ex.pdf":        0.6,
    "main_st_ex2.pdf":       0.6,
}

VLM_PROMPT = (
    "Examine this architectural floor plan. The red highlighted lines "
    "represent a single drawing style layer. Do these red lines represent "
    "structural walls (exterior walls, interior partition walls, or wall "
    "outlines)? Consider: walls are typically long, straight, parallel "
    "pairs forming room boundaries. Ignore: furniture, fixtures, "
    "appliances, dimension lines, text, door swings, section markers. "
    "Answer with ONLY 'YES' or 'NO'."
)

NEON_RED = (1, 0, 0.25)
CONTEXT_GREY = (0.72, 0.72, 0.72)
RESULT_GREEN = (0, 1, 0)
RESULT_FAINT_RED = (1, 0.5, 0.5)
BBOX_PADDING_PT = 30.0

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))


# ── rendering helpers ─────────────────────────────────────────────────

def _draw_prims_on_shape(
    shape, prims: list[dict], color: tuple, line_width: float,
) -> None:
    """Draw Step-1 primitives onto a fitz.Shape."""
    for prim in prims:
        pts = prim["points"]
        kind = prim["type"]
        if kind == "line" and len(pts) >= 2:
            shape.draw_line(fitz.Point(pts[0]), fitz.Point(pts[1]))
            shape.finish(color=color, width=line_width)
        elif kind == "polygon" and len(pts) >= 3:
            fp = [fitz.Point(p) for p in pts]
            shape.draw_polyline(fp + [fp[0]])
            shape.finish(
                color=color, fill=color, width=0.5, fill_opacity=0.35,
            )
        elif kind == "curve" and len(pts) == 4:
            shape.draw_bezier(*(fitz.Point(p) for p in pts))
            shape.finish(color=color, width=line_width)


def render_style_overlay(
    pdf_path: str,
    candidate: dict,
    style_key: str,
    page_number: int = 0,
    render_scale: float = 2.0,
    padding: float = BBOX_PADDING_PT,
) -> bytes:
    """
    Render a single style layer highlighted in neon red over the faded
    PDF, cropped to the candidate bounding box.  Returns PNG bytes.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    # 1. Fade background
    bg = page.new_shape()
    bg.draw_rect(page.rect)
    bg.finish(fill=(1, 1, 1), fill_opacity=0.70)
    bg.commit()

    # 2. All candidate primitives in faint grey (context)
    ctx = page.new_shape()
    for prims in candidate["grouped_primitives"].values():
        _draw_prims_on_shape(ctx, prims, CONTEXT_GREY, 0.25)
    ctx.commit()

    # 3. Current style layer in thick neon red
    fg = page.new_shape()
    _draw_prims_on_shape(
        fg,
        candidate["grouped_primitives"].get(style_key, []),
        NEON_RED,
        2.5,
    )
    fg.commit()

    # 4. Crop to candidate bbox with padding
    #    Shape coords are in unrotated (mediabox) space, but get_pixmap
    #    clip is in rotated (page.rect) space — transform via rotation_matrix.
    bb = candidate["bounding_box"]
    clip = fitz.Rect(
        bb[0] - padding, bb[1] - padding,
        bb[2] + padding, bb[3] + padding,
    ) * page.rotation_matrix & page.rect

    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(render_scale, render_scale),
        clip=clip,
        alpha=False,
    )
    png_bytes = pixmap.tobytes("png")
    doc.close()
    return png_bytes


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
    Result overlay: YES-classified styles in neon green, NO styles in
    faint red.  Cropped to candidate bbox.  Saved to *output_path*.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    bg = page.new_shape()
    bg.draw_rect(page.rect)
    bg.finish(fill=(1, 1, 1), fill_opacity=0.70)
    bg.commit()

    for style_key, prims in candidate["grouped_primitives"].items():
        verdict = style_classifications.get(
            style_key, {},
        ).get("verdict", "NO")
        color = RESULT_GREEN if verdict == "YES" else RESULT_FAINT_RED
        width = 2.5 if verdict == "YES" else 0.5
        shape = page.new_shape()
        _draw_prims_on_shape(shape, prims, color, width)
        shape.commit()

    bb = candidate["bounding_box"]
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

def build_vlm_prompt() -> str:
    """Return the classification prompt for the VLM."""
    return VLM_PROMPT


def parse_vlm_response(response_text: str) -> dict[str, str]:
    """
    Extract YES/NO verdict and confidence from VLM response text.

    Confidence levels:
        high   — clean "YES" or "NO"
        medium — starts with YES/NO but has extra text
        low    — YES/NO found elsewhere, or unparseable
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


def classify_style_with_vlm(
    png_bytes: bytes,
    api_key: str,
    model: str = "gpt-4o",
) -> dict[str, str]:
    """
    Send a style-layer overlay to the VLM and return the parsed verdict.
    The image is base64-encoded in memory — never written to disk first.
    """
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_vlm_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                        },
                    },
                ],
            }
        ],
        max_tokens=10,
    )
    return parse_vlm_response(resp.choices[0].message.content)


# ── classification orchestrator ───────────────────────────────────────

def classify_candidate(
    pdf_path: str,
    candidate: dict,
    api_key: str,
    page_number: int = 0,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """
    Classify every style layer within a single candidate via the VLM.

    Returns::

        {"candidate_id": int,
         "bounding_box": [...],
         "style_classifications": {style_key: {verdict, confidence}},
         "wall_primitives": {style_key: [prims]}   # YES only
        }
    """
    cid = candidate["candidate_id"]
    classifications: dict[str, dict] = {}
    wall_prims: dict[str, list[dict]] = {}

    for idx, (style_key, prims) in enumerate(
        candidate["grouped_primitives"].items(),
    ):
        # render in memory
        png = render_style_overlay(
            pdf_path, candidate, style_key, page_number=page_number,
        )

        # optionally persist the overlay for inspection
        if output_dir:
            cand_dir = os.path.join(output_dir, f"candidate_{cid}")
            os.makedirs(cand_dir, exist_ok=True)
            with open(
                os.path.join(cand_dir, f"style_{idx}.png"), "wb",
            ) as f:
                f.write(png)

        result = classify_style_with_vlm(png, api_key)
        classifications[style_key] = result
        if result["verdict"] == "YES":
            wall_prims[style_key] = prims

        print(
            f"  candidate {cid} | {style_key} "
            f"-> {result['verdict']} ({result['confidence']})"
        )

    return {
        "candidate_id": cid,
        "bounding_box": candidate["bounding_box"],
        "style_classifications": classifications,
        "wall_primitives": wall_prims,
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
    """
    Run Step 3 classification on a plan's ground-truth candidates.
    """
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

    with open(candidates_json_path) as f:
        all_candidates = json.load(f)

    target_ids = GROUND_TRUTH_CANDIDATES.get(plan_name, [])
    candidates = [
        c for c in all_candidates if c["candidate_id"] in target_ids
    ]
    if not candidates:
        print(f"No matching candidates for plan '{plan_name}'")
        return []

    print(
        f"Step 3: classifying {len(candidates)} candidate(s) "
        f"for '{plan_name}'"
    )

    results: list[dict] = []
    for cand in candidates:
        result = classify_candidate(
            pdf_path, cand, api_key,
            page_number=page_number, output_dir=output_dir,
        )
        results.append(result)

        # generate the combined result overlay
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

    with open(os.path.join(output_dir, "classifications.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Step 3 complete: {len(results)} candidates -> {output_dir}")
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python step3_classify.py <plan_name> "
            "[candidates.json] [pdf_path]",
        )
        sys.exit(1)
    run(
        sys.argv[1],
        candidates_json_path=sys.argv[2] if len(sys.argv) > 2 else None,
        pdf_path=sys.argv[3] if len(sys.argv) > 3 else None,
    )
