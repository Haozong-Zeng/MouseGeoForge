import os
import shutil
import subprocess
import sys
import json
import threading
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List

try:  # tomllib is stdlib on 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

app = FastAPI(title="VGGT Calibration GUI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

WORKSPACE_DIR = Path(__file__).parent
DATA_DIR = WORKSPACE_DIR / "gui_data"
VIDEOS_DIR = DATA_DIR / "videos"
FRAMES_DIR = DATA_DIR / "frames"
VGGT_OUTPUTS_DIR = DATA_DIR / "vggt_outputs"
PRIOR_DIR = DATA_DIR / "vggt_prior"
REFINED_DIR = DATA_DIR / "refined"

# Ensure directories exist
for d in [VIDEOS_DIR, FRAMES_DIR, PRIOR_DIR, REFINED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))


# ---------------------------------------------------------------------------
# Background pipeline job state
# ---------------------------------------------------------------------------
JOB_LOCK = threading.Lock()
JOB = {
    "state": "idle",      # idle | running | done | error
    "stage": "",          # human-readable current stage
    "log": [],            # list[str] of log lines
    "stats": {},          # aggregate stats once done
    "message": "",        # error message if any
}


def _reset_job():
    with JOB_LOCK:
        JOB["state"] = "running"
        JOB["stage"] = "Starting"
        JOB["log"] = []
        JOB["stats"] = {}
        JOB["message"] = ""


def _log(line: str):
    line = line.rstrip("\n")
    if not line:
        return
    with JOB_LOCK:
        JOB["log"].append(line)
    print(line, flush=True)


def _set_stage(stage: str):
    with JOB_LOCK:
        JOB["stage"] = stage
    _log(f"=== {stage} ===")


def _run_streamed(cmd: list, label: str) -> str:
    """Run a subprocess, streaming combined stdout/stderr into the job log.

    Returns the full captured output. Raises RuntimeError on non-zero exit.
    """
    _log(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(WORKSPACE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured_lines.append(line)
        _log(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed (exit code {proc.returncode}). See log for details.")
    return "".join(captured_lines)


def _infer_cameras():
    """Return the ordered list of camera names inferred from the uploaded videos."""
    from vggt_calibration_pipeline.extract_frames import discover_videos
    videos = discover_videos(VIDEOS_DIR)
    return [v.camera for v in videos]


def _pipeline_worker(max_frames: int, keep_best_percent: float):
    try:
        from vggt_calibration_pipeline.extract_frames import discover_videos, common_frame_count

        videos = discover_videos(VIDEOS_DIR)
        camera_names = [v.camera for v in videos]
        total_frames = common_frame_count(videos)
        _log(f"Discovered {len(camera_names)} camera(s): {', '.join(camera_names)}")
        _log(f"Common synchronized frame count: {total_frames}")

        if max_frames <= 0 or max_frames >= total_frames:
            frames_spec = "all"
            n_planned = total_frames
        else:
            step = max(1, total_frames // max_frames)
            frames_spec = f"0:{total_frames}:{step}"
            n_planned = len(range(0, total_frames, step))
        _log(f"Frame sampling spec: {frames_spec} (~{n_planned} frames)")

        # Clean stale outputs from any previous run
        for d in (FRAMES_DIR, VGGT_OUTPUTS_DIR, PRIOR_DIR, REFINED_DIR):
            if d.exists():
                shutil.rmtree(d)
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)

        # Step 1: Extract frames (for the annotation canvas)
        _set_stage("Step 1/3 — Extracting frames")
        _run_streamed([
            sys.executable, "-m", "vggt_calibration_pipeline.extract_frames",
            "--video-dir", str(VIDEOS_DIR),
            "--output-dir", str(FRAMES_DIR),
            "--camera-names", *camera_names,
            "--frames", frames_spec,
        ], "Frame extraction")

        # Step 2: VGGT inference (GPU)
        _set_stage("Step 2/3 — VGGT inference (GPU)")
        _run_streamed([
            sys.executable, "-m", "vggt_calibration_pipeline.run_pipeline",
            "--dataset", str(DATA_DIR),
            "--video-dir", str(VIDEOS_DIR),
            "--output-root", str(DATA_DIR),
            "--output-name", VGGT_OUTPUTS_DIR.name,
            "--camera-names", *camera_names,
            "--frames", frames_spec,
        ], "VGGT inference")

        # Step 3: Aggregate robust rig prior
        _set_stage("Step 3/3 — Aggregating rig prior")
        agg_out = _run_streamed([
            sys.executable, "-m", "vggt_calibration_pipeline.aggregate_vggt_priors",
            str(VGGT_OUTPUTS_DIR),
            "--output-dir", str(PRIOR_DIR),
            "--keep-best-percent", str(keep_best_percent),
        ], "Prior aggregation")

        stats = {}
        try:
            json_start = agg_out.find("{")
            json_end = agg_out.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                stats = json.loads(agg_out[json_start:json_end])
        except Exception as e:
            _log(f"Warning: failed to parse aggregate stats: {e}")

        with JOB_LOCK:
            JOB["stats"] = stats
            JOB["state"] = "done"
            JOB["stage"] = "Done"
        _log("Pipeline finished successfully.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        with JOB_LOCK:
            JOB["state"] = "error"
            JOB["message"] = str(e)
        _log(f"ERROR: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def read_root():
    return FileResponse(str(static_dir / "index.html"))


def _resolve_checkpoint() -> Path:
    """Checkpoint location, honoring VGGT_OMEGA_CHECKPOINT (same default as run_pipeline)."""
    env = os.environ.get("VGGT_OMEGA_CHECKPOINT")
    if env:
        p = Path(env)
        return p if p.is_absolute() else (WORKSPACE_DIR / p)
    return WORKSPACE_DIR / "checkpoints" / "vggt_omega_1b_512.pt"


@app.get("/api/health")
def health():
    cuda = False
    device = None
    torch_error = None
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            device = torch.cuda.get_device_name(0)
    except Exception as e:
        torch_error = f"{type(e).__name__}: {e}"

    checkpoint = _resolve_checkpoint()
    return {
        "cuda": cuda,
        "device": device,
        "torch_error": torch_error,
        "python": sys.executable,
        "checkpoint_path": str(checkpoint),
        "checkpoint_present": checkpoint.is_file(),
    }


@app.post("/api/upload_videos")
async def upload_videos(files: List[UploadFile] = File(...)):
    # Clean previous videos
    if VIDEOS_DIR.exists():
        shutil.rmtree(VIDEOS_DIR)
    VIDEOS_DIR.mkdir(parents=True)

    saved_files = []
    for file in files:
        # Guard against path traversal in client-supplied filenames
        safe_name = Path(file.filename).name
        file_path = VIDEOS_DIR / safe_name
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(safe_name)

    # Try to infer cameras / frame counts so the UI can give early feedback
    cameras, frame_count, warning = [], None, None
    try:
        from vggt_calibration_pipeline.extract_frames import discover_videos, common_frame_count
        videos = discover_videos(VIDEOS_DIR)
        cameras = [v.camera for v in videos]
        counts = {v.camera: v.frame_count for v in videos}
        frame_count = common_frame_count(videos)
        if len(set(counts.values())) > 1:
            warning = (
                "Videos have different frame counts "
                f"({counts}); only the common prefix of {frame_count} frames will be used. "
                "Make sure the videos are synchronized."
            )
    except Exception as e:
        warning = f"Could not analyze videos: {e}"

    return {
        "status": "success",
        "files": saved_files,
        "cameras": cameras,
        "frame_count": frame_count,
        "warning": warning,
    }


@app.post("/api/run_pipeline")
def run_pipeline(max_frames: int = 100, keep_best_percent: float = 100.0):
    with JOB_LOCK:
        if JOB["state"] == "running":
            return JSONResponse(status_code=409, content={"status": "error", "message": "A pipeline run is already in progress."})
    has_videos = VIDEOS_DIR.exists() and any(VIDEOS_DIR.iterdir())
    if not has_videos:
        return JSONResponse(status_code=400, content={"status": "error", "message": "No videos uploaded."})

    _reset_job()
    thread = threading.Thread(
        target=_pipeline_worker, args=(max_frames, keep_best_percent), daemon=True
    )
    thread.start()
    return {"status": "started"}


@app.get("/api/pipeline_status")
def pipeline_status():
    with JOB_LOCK:
        return {
            "state": JOB["state"],
            "stage": JOB["stage"],
            "log": JOB["log"],
            "stats": JOB["stats"],
            "message": JOB["message"],
        }


@app.get("/api/frames_data")
def get_frames_data():
    # Return structure of frames and cameras for annotation UI
    if not FRAMES_DIR.exists():
        return {"frames": []}

    frames = []
    for frame_dir in sorted(FRAMES_DIR.iterdir()):
        if frame_dir.is_dir() and frame_dir.name.startswith("frame_"):
            images = {}
            for img in sorted(frame_dir.glob("*.png")):
                # infer camera from filename e.g. pA.png -> A
                cam_name = img.stem[1:] if img.stem.startswith("p") else img.stem
                images[cam_name] = f"/data/frames/{frame_dir.name}/{img.name}"

            frames.append({
                "id": frame_dir.name,
                "index": int(frame_dir.name.split("_")[1]),
                "images": images,
            })

    return {"frames": frames}


@app.post("/api/refine")
async def refine_calibration(request: Request):
    try:
        points_data = await request.json()

        # ---- Validation -------------------------------------------------
        errors = []
        points = points_data.get("points", [])
        distances = points_data.get("distances", [])
        triangulable = [
            p for p in points if len(p.get("observations", {})) >= 2
        ]
        if len(triangulable) < 2:
            errors.append("Need at least 2 points each clicked in ≥2 views.")
        if not distances:
            errors.append("Need at least 1 known distance to set the metric scale.")
        if errors:
            return JSONResponse(status_code=400, content={"status": "error", "message": " ".join(errors)})

        prior_toml = PRIOR_DIR / "calibration_prior.toml"
        prior_stats_file = PRIOR_DIR / "prior_stats.json"
        if not prior_toml.is_file() or not prior_stats_file.is_file():
            return JSONResponse(status_code=400, content={"status": "error", "message": "Rig prior not found. Run the VGGT pipeline (step 2) first."})

        points_file = DATA_DIR / "selected_points.json"
        with open(points_file, "w") as f:
            json.dump(points_data, f, indent=2)

        REFINED_DIR.mkdir(parents=True, exist_ok=True)
        refine_cmd = [
            sys.executable, "-m", "vggt_calibration_pipeline.refine_metric_calibration",
            str(DATA_DIR),
            "--calibration", str(prior_toml),
            "--prior-stats", str(prior_stats_file),
            "--points", str(points_file),
            "--rotation-prior-deg", "3",
            "--translation-prior-mm", "25",
            "--refine-intrinsics", "--intrinsics-mode", "f",
            "--sigma-refit-iters", "2",
            "--drop-outlier-obs-z", "4.0",
            "--output-dir", str(REFINED_DIR),
        ]
        print("Running Refinement:", " ".join(refine_cmd))
        proc = subprocess.run(
            refine_cmd, cwd=str(WORKSPACE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if proc.returncode != 0:
            return JSONResponse(status_code=500, content={"status": "error", "message": "Refinement failed:\n" + (proc.stdout or "")[-2000:]})

        summary_file = REFINED_DIR / "summary.json"
        summary = {}
        prior_stats = {}
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
        if prior_stats_file.exists():
            with open(prior_stats_file) as f:
                prior_stats = json.load(f)

        has_holdout = any(d.get("holdout") for d in distances)
        return {
            "status": "success",
            "summary": summary,
            "prior_stats": prior_stats,
            "has_holdout": has_holdout,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.get("/api/export")
def export_calibration(fmt: str = "anipose"):
    """Download the metric calibration as Anipose TOML or a generic JSON."""
    metric_toml = REFINED_DIR / "calibration_metric.toml"
    if not metric_toml.is_file():
        return JSONResponse(status_code=404, content={"status": "error", "message": "No metric calibration found. Run refinement first."})

    if fmt == "anipose":
        return FileResponse(str(metric_toml), media_type="application/toml", filename="calibration.toml")

    if fmt == "json":
        if tomllib is None:
            return JSONResponse(status_code=500, content={"status": "error", "message": "tomllib unavailable (needs Python 3.11+)."})
        data = tomllib.loads(metric_toml.read_text())
        cameras = []
        for key, cam in data.items():
            if not key.startswith("cam_") or not isinstance(cam, dict):
                continue
            cameras.append({
                "name": cam.get("name", key),
                "size": cam.get("size"),
                "matrix": cam.get("matrix"),
                "distortions": cam.get("distortions"),
                "rotation_rvec": cam.get("rotation"),
                "translation": cam.get("translation"),
            })
        out_path = REFINED_DIR / "calibration_metric.json"
        out_path.write_text(json.dumps({"cameras": cameras}, indent=2))
        return FileResponse(str(out_path), media_type="application/json", filename="calibration.json")

    return JSONResponse(status_code=400, content={"status": "error", "message": f"Unknown format: {fmt}"})


app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
