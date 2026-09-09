# Checkerboard-free Calibration GUI

A desktop/web app that turns **N synchronized videos** (one per camera) into a
**metric, Anipose-compatible calibration** plus a **trust report** — no
checkerboard required. It wraps the pipeline described in
[`vggt_calibration_pipeline/PIPELINE_SPEC_FOR_GUI.md`](vggt_calibration_pipeline/PIPELINE_SPEC_FOR_GUI.md).

The only hard requirement on the input is that the videos are **synchronized**
(frame *k* shows the same instant in every view). Number of cameras, resolution,
and lens are all handled.

## Requirements

- Python 3.11+ (uses stdlib `tomllib`)
- A CUDA GPU + the VGGT-Omega checkpoint at `checkpoints/vggt_omega_1b_512.pt`
  (only the inference step needs the GPU; everything else is CPU and fast)

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
# or, with autoreload during development:
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open <http://localhost:8000>. A red banner appears at the top if no CUDA GPU
or no checkpoint is detected.

## Workflow (4 steps)

1. **Upload videos** — drop in N synchronized videos. Camera names are inferred
   from filenames (`pawfilt_camA.mp4` → `A`, `pB.mp4` → `B`, …). The app reports
   the detected cameras and warns if the videos have mismatched frame counts.
2. **VGGT processing** — extracts frames, runs VGGT inference on the GPU, and
   aggregates a robust rig prior. Runs as a background job; the log streams live
   into the terminal pane. *Max frames* controls how many evenly-spaced frames are
   sampled (more = stronger prior, diminishing returns); *% best frames* trims the
   least-consistent frames from the prior.
3. **Manual annotation** — click ≥~6 stationary landmarks, each in ≥2 views, and
   add a few known distances (mm). **Reserve 1–2 distances as hold-out** — that is
   the real accuracy test. Landmarks are stationary across the video, so you only
   annotate them once.
4. **Trust report** — a 3-tier gate:
   - *Tier 1* reprojection RMSE — gross-error filter (`< ~5 px`). Low is necessary,
     **not** a pass on its own.
   - *Tier 2* VGGT prior rotation spread — high ⇒ hard rig, prior less reliable.
   - *Tier 3* **held-out distance error** — the out-of-sample number that actually
     predicts accuracy (validated ρ=0.73 vs ρ≈0 for reprojection). Lead with this.

   Export the result as an Anipose `calibration.toml` (drop-in) or a generic JSON
   (`K, R(rvec), t, distortions` per camera).

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/health` | CUDA + checkpoint availability |
| POST | `/api/upload_videos` | Upload videos; returns inferred cameras / frame count |
| POST | `/api/run_pipeline?max_frames=&keep_best_percent=` | Start the background pipeline |
| GET  | `/api/pipeline_status` | Poll job state + streamed log + aggregate stats |
| GET  | `/api/frames_data` | Frame/camera structure for the annotation canvas |
| POST | `/api/refine` | Run metric bundle adjustment from `selected_points.json` |
| GET  | `/api/export?fmt=anipose\|json` | Download the metric calibration |

Working files are written under `gui_data/` (`videos/`, `frames/`,
`vggt_outputs/`, `vggt_prior/`, `refined/`). Each pipeline run clears the previous
outputs.
