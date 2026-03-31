"""
Cantena Wall Detection UI — FastAPI backend
=============================================
Serves the web UI and wraps the wall_pipeline for async processing
with Server-Sent Events (SSE) progress reporting.
"""

import asyncio
import json
import math
import os
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path

import fitz
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Cantena Wall Detection")

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# In-memory job store
jobs: dict[str, dict] = {}


def _save_job_metadata(job_id: str):
    """Persist job metadata to disk as JSON."""
    job = jobs.get(job_id)
    if not job:
        return
    meta = {
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "result": job.get("result"),
        "page_info": job.get("page_info"),
        "created_at": job.get("created_at"),
    }
    meta_path = RESULTS_DIR / job_id / "job.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _load_history():
    """Load completed jobs from disk on startup."""
    if not RESULTS_DIR.exists():
        return
    for job_dir in RESULTS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        meta_path = job_dir / "job.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            job_id = meta["id"]
            if job_id not in jobs:
                jobs[job_id] = {
                    **meta,
                    "pdf_path": "",
                    "progress": [],
                    "error": None,
                }
                # Resolve pdf_path from directory contents
                for p in job_dir.iterdir():
                    if p.suffix.lower() == ".pdf":
                        jobs[job_id]["pdf_path"] = str(p)
                        break
        except (json.JSONDecodeError, KeyError):
            continue


_load_history()


# ---------------------------------------------------------------------------
# Serve the SPA
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return FileResponse("static/index.html", media_type="text/html")


# Mount static assets
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_dir / file.filename
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Render first page as preview
    doc = fitz.open(str(pdf_path))
    page = doc[0]
    mat = fitz.Matrix(150 / 72, 150 / 72)
    pix = page.get_pixmap(matrix=mat)
    preview_path = job_dir / "preview.png"
    pix.save(str(preview_path))

    page_info = {
        "width": page.rect.width,
        "height": page.rect.height,
        "drawings": len(page.get_drawings()),
    }
    doc.close()

    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "pdf_path": str(pdf_path),
        "status": "uploaded",
        "progress": [],
        "result": None,
        "page_info": page_info,
        "created_at": time.time(),
    }

    return {
        "job_id": job_id,
        "filename": file.filename,
        "page_info": page_info,
    }


# ---------------------------------------------------------------------------
# Pipeline runner (runs in thread to not block event loop)
# ---------------------------------------------------------------------------

def _run_pipeline(job_id: str):
    """Execute the wall pipeline synchronously, updating job progress."""
    job = jobs[job_id]
    job["status"] = "processing"
    pdf_path = job["pdf_path"]
    job_dir = RESULTS_DIR / job_id

    tag = os.path.splitext(os.path.basename(pdf_path))[0].replace(" ", "_")

    def add_progress(step: int, total: int, label: str, detail: str = ""):
        job["progress"].append({
            "step": step,
            "total": total,
            "label": label,
            "detail": detail,
            "timestamp": time.time(),
        })

    try:
        # Import pipeline modules
        from pipelines.wall_pipeline import (
            load_env, vlm_identify_wall, _refine_seeds,
            save_debug_vlm_seed, extract_fingerprint,
            find_all_walls, generate_overlay, _fallback_fingerprint,
        )

        load_env()

        # Step 1: Load PDF
        add_progress(1, 5, "Loading PDF", "Extracting vector drawings...")
        doc = fitz.open(pdf_path)
        page = doc[0]
        drawings = page.get_drawings()
        add_progress(1, 5, "Loading PDF",
                     f"Found {len(drawings)} vector drawings")

        # Step 2: VLM Seed Identification
        MAX_VLM_ATTEMPTS = 3
        fp = None
        seeds = None

        for attempt in range(1, MAX_VLM_ATTEMPTS + 1):
            add_progress(2, 5, "Wall Analysis",
                         f"Attempt {attempt}/{MAX_VLM_ATTEMPTS} — analyzing floor plan...")
            try:
                seeds = vlm_identify_wall(page, dpi=150)
                seeds = _refine_seeds(seeds, drawings)

                debug_path = str(job_dir / "debug_vlm_seed.png")
                save_debug_vlm_seed(pdf_path, seeds, output_path=debug_path)

                # Step 3: Fingerprint
                add_progress(3, 5, "Extracting Wall Pattern",
                             "Analyzing wall characteristics...")
                for si, (seed_rect, vlm_hints) in enumerate(seeds):
                    try:
                        fp = extract_fingerprint(drawings, seed_rect, vlm_hints)
                        add_progress(3, 5, "Extracting Wall Pattern",
                                     f"Detected style: {fp['wall_style']}")
                        break
                    except ValueError:
                        if si == len(seeds) - 1:
                            raise
                break
            except ValueError:
                if attempt == MAX_VLM_ATTEMPTS:
                    add_progress(2, 5, "Wall Analysis",
                                 "Using fallback detection...")
                    fp = _fallback_fingerprint(drawings)
                    if not fp:
                        raise RuntimeError("Could not detect wall pattern")

        # Step 4: Global match
        add_progress(4, 5, "Detecting All Walls",
                     "Matching pattern across entire document...")
        wall_result = find_all_walls(drawings, fp)
        n_edges = len(wall_result["edges"])
        n_hatches = len(wall_result["hatches"])
        n_fills = len(wall_result.get("fills", []))
        total_wall = n_edges + n_hatches + n_fills
        add_progress(4, 5, "Detecting All Walls",
                     f"Found {total_wall} wall elements in {len(wall_result['components'])} groups")

        # Step 5: Generate overlay
        add_progress(5, 5, "Generating Overlay",
                     "Rendering wall detection overlay...")
        overlay_path = str(job_dir / "wall_overlay.png")
        generate_overlay(pdf_path, wall_result, output_path=overlay_path, dpi=200)

        # Also render the original at matching DPI for comparison
        original_render = str(job_dir / "original.png")
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = page.get_pixmap(matrix=mat)
        pix.save(original_render)

        doc.close()

        # Build summary
        e_items = sum(len(d["items"]) for _, d in wall_result["edges"])
        h_items = sum(len(d["items"]) for _, d in wall_result["hatches"])
        f_items = sum(len(d["items"]) for _, d in wall_result.get("fills", []))

        summary = {
            "wall_style": fp["wall_style"],
            "total_drawings": len(drawings),
            "edge_drawings": n_edges,
            "edge_segments": e_items,
            "hatch_drawings": n_hatches,
            "hatch_segments": h_items,
            "fill_drawings": n_fills,
            "fill_items": f_items,
            "total_wall_drawings": total_wall,
            "coverage_pct": round(total_wall / len(drawings) * 100, 1) if drawings else 0,
            "components": len(wall_result["components"]),
        }

        job["result"] = summary
        job["status"] = "completed"
        add_progress(5, 5, "Complete", "Wall detection finished successfully.")
        _save_job_metadata(job_id)

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        add_progress(0, 5, "Error", str(e))


# ---------------------------------------------------------------------------
# Process endpoint — kicks off pipeline
# ---------------------------------------------------------------------------

@app.post("/api/process/{job_id}")
async def process_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] not in ("uploaded", "error"):
        raise HTTPException(400, f"Job is already {job['status']}")

    job["status"] = "processing"
    job["progress"] = []
    job["result"] = None

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_pipeline, job_id)

    return {"status": "processing"}


# ---------------------------------------------------------------------------
# SSE progress stream
# ---------------------------------------------------------------------------

@app.get("/api/progress/{job_id}")
async def progress_stream(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        last_idx = 0
        while True:
            job = jobs[job_id]

            # Send any new progress entries
            while last_idx < len(job["progress"]):
                entry = job["progress"][last_idx]
                data = json.dumps(entry)
                yield f"data: {data}\n\n"
                last_idx += 1

            if job["status"] in ("completed", "error"):
                final = {
                    "status": job["status"],
                    "result": job.get("result"),
                    "error": job.get("error"),
                }
                yield f"event: done\ndata: {json.dumps(final)}\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Result endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    return {
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error"),
        "page_info": job.get("page_info"),
    }


@app.get("/api/files/{job_id}/{filename}")
async def serve_file(job_id: str, filename: str):
    # Allow serving files for jobs on disk even if not in memory
    file_path = RESULTS_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(file_path))


# ---------------------------------------------------------------------------
# History endpoints
# ---------------------------------------------------------------------------

@app.get("/api/history")
async def list_history():
    """Return all completed jobs, newest first."""
    completed = []
    for job in jobs.values():
        if job["status"] == "completed":
            completed.append({
                "id": job["id"],
                "filename": job["filename"],
                "status": job["status"],
                "result": job.get("result"),
                "page_info": job.get("page_info"),
                "created_at": job.get("created_at"),
            })
    completed.sort(key=lambda j: j.get("created_at") or 0, reverse=True)
    return completed


@app.delete("/api/history/{job_id}")
async def delete_history_item(job_id: str):
    """Delete a job from history and disk."""
    job_dir = RESULTS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    jobs.pop(job_id, None)
    return {"deleted": job_id}
