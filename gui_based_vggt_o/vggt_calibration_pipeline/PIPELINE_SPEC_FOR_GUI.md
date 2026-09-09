# Pipeline Spec — Checkerboard-free Calibration GUI

**Audience:** the engineer packaging this prototype into a GUI application.
**Goal:** a desktop app where a user drops in **N synchronized videos** (one per
camera) and gets back a **metric, Anipose-compatible calibration** plus a
**trust report** — with no checkerboard.

> **The only hard requirement on the user's data is that the videos are
> synchronized** (frame *k* shows the same instant in every view). Everything
> else (number of cameras, resolution, lens) is handled. The prototype was
> validated on 3 views but the pipeline is view-count-agnostic — see §"N views".

---

## 0. End-to-end data flow

```
 N synchronized videos  (camA.mp4, camB.mp4, …, camN.mp4)
        │
   [1] extract & sync frames ───────────────► frames/frame_XXXX/p{cam}.png
        │
   [2] VGGT inference per frame ────────────► per-frame extrinsics + intrinsics
        │
   [3] robust multi-frame aggregation ──────► calibration_prior.toml + prior_stats.json
        │                                       (the "rig prior", unitless scale)
   [4] manual annotation (GUI) ─────────────► selected_points.json
        │      click ≥~6 stationary landmarks in ≥2 views each
        │      + a few known distances (mm); hold 1–2 distances OUT
        │
   [5] metric bundle adjustment ────────────► calibration_metric.toml  (METRIC)
        │                                       + points_3d.csv + reports
        │
   [6] quality report (3-tier gate) ────────► trust the calibration? (no GT needed)
        │
   [7] export ──────────────────────────────► Anipose calibration.toml (drop-in)
```

Each numbered stage already exists as a script/module (below). The GUI is mostly
**orchestration + an annotation canvas + a results/QA view**.

---

## 1. Frame extraction & sync — `extract_frames.py`
- `discover_videos(dir)`: finds the video files, **infers camera name from the
  filename** (`camA.mp4`→`A`, `pB.mp4`→`B`, etc. via `infer_camera_name`). Order
  is the camera order everywhere downstream.
- `common_frame_count(videos)`: the number of frames present in *all* views →
  enforces the synchronized-length assumption. If counts differ, take the common
  prefix (and warn the user — this is the place to surface "are these synced?").
- Extracts a chosen set of frames to `frames/frame_XXXX/p{cam}.png`.
- **GUI need:** a drop-zone for N files, a per-camera name editor (default =
  inferred name), and a frame-range / frame-count selector (default: a few hundred
  evenly-spaced frames — more frames = better prior, diminishing returns).

## 2. VGGT inference — `run_pipeline.py` (`predict_cameras_batch`)
- Loads VGGT-Omega (`load_model`), runs each frame's N images as one batch →
  per-frame extrinsics + pose-encoding → per-frame `calibration.toml`.
- **GPU step.** Needs the model weights + CUDA. This is the only heavy dependency;
  budget a progress bar over frames.
- Camera A (first) pinned to identity (gauge).

## 3. Robust rig prior — `aggregate_vggt_priors.py`
- Input: the per-frame calibrations. Output: `calibration_prior.toml` (the
  aggregated rig) + `prior_stats.json` (empirical per-camera/per-pair sigmas,
  inlier-frame count, **per-camera rotation_sigma_deg** — used by the QA gate) +
  `frame_deviations.csv`.
- Gauge-invariant (Umeyama/Procrustes), robust frame trimming. **Scale is
  unitless here** (~100× off true) — metric scale comes from stage 5.
- Runs in <1 s; no GPU.

## 4. Manual annotation (the GUI's core) — `point_selector.py` exists as a prototype
This is where the colleague will invest most UI effort. Contract = `selected_points.json`:

```jsonc
{
  "schema_version": 3,
  "distance_units": "mm",
  "cameras": ["A","B","C", ...],            // N cameras, any count
  "points": [
    { "id": "P001", "label": "...", "sigma_px": 6.0,
      "observations": {                      // a point may be seen in a SUBSET of views
        "A": {"x": 489.3, "y": 381.9, "source_x": 940.9, "source_y": 734.5},
        "B": {"x": ...,    "y": ...}
      } }
  ],
  "distances": [
    { "id": "D001", "point_a": "P001", "point_b": "P002",
      "distance_mm": 4.17, "sigma_mm": 0.5,
      "holdout": false, "use_for_refinement": true }   // set holdout=true to reserve for QA
  ]
}
```
- **Landmarks are STATIONARY across frames** (annotate on one frame; they hold for
  the video). They do not need to be physical markers — any identifiable fixed
  point works (rig corners, tape, anatomical landmarks on a still subject).
- A point must be clicked in **≥2 views** to be triangulable. More views = better.
- `sigma_px` defaults to 6 (human click error ≈ 5–7 px). Don't expose unless advanced.
- **GUI annotation canvas requirements:**
  - Show all N views (grid/tabs); click to drop point Pk in the current view;
    same Pk gets clicked in the other views (correspondence by point id).
  - Visual epipolar/aid optional (NOT required; explicitly out of scope per prior
    decision).
  - Distance editor: pick two point ids, type the real distance (mm), a checkbox
    "reserve for accuracy check (hold-out)".
  - Validation: warn if a point has <2 views; warn if <1 distance.
- **Guidance to bake into the UI** (from the study):
  - ~6–8 points is enough for good 3D *shape*.
  - **More points/distances mainly improves metric *scale* reliability** — encourage
    users who care about absolute mm to click more and add several distances.
  - Tell users to reserve **1–2 known distances as hold-out** — that's the trust test.

## 5. Metric bundle adjustment — `refine_metric_calibration.py`
Recommended invocation (the validated `agg_loose_f` recipe):
```
refine_metric_calibration.py <DATASET> \
  --calibration prior/calibration_prior.toml --prior-stats prior/prior_stats.json \
  --points selected_points.json \
  --rotation-prior-deg 3 --translation-prior-mm 25 \
  --refine-intrinsics --intrinsics-mode f \
  --sigma-refit-iters 2 --drop-outlier-obs-z 4.0 \
  --output-dir refined/
```
Outputs: `calibration_metric.toml` (the metric calibration), `points_3d.csv`,
`reprojection_report.csv`, `distance_report.csv`, `summary.json` (has
`reprojection_px`, `distance_residual_mm` incl. `refined_holdout`, `scale_to_mm`,
`num_dropped_observations`). Runs in ~4 s, no GPU.

## 6. Quality report — the trust gate (3 tiers)
Read from `summary.json` + `prior_stats.json` and present plainly:

| Tier | Signal | Source | Rule shown to user |
|------|--------|--------|--------------------|
| 1 gross-error filter | click reprojection RMSE | `summary.reprojection_px.refined.rmse` | `> ~5 px` ⇒ likely a bad/mislabeled click — re-click. **Low value is NOT a pass.** |
| 2 rig-quality | VGGT prior rotation spread | mean `prior_stats.cameras[*].rotation_sigma_deg` | High ⇒ hard rig, prior unreliable. |
| 3 **acceptance test** | **held-out distance error** | `summary.distance_residual_mm.refined_holdout` (needs ≥1 `holdout:true` distance) | This is the real number. Small held-out error ⇒ trust the metric calibration. |

**Why this matters (validated, n=24):** reprojection error does **not** predict true
accuracy (Spearman ρ≈0 for shape), because BA already minimized it — it's in-sample.
A held-out known distance is out-of-sample and strongly predicts true accuracy
(ρ=0.73). Example failure that reprojection misses: a rig with 0.87 px reprojection
that was actually 63% wrong in scale. **So the GUI must lead with the held-out
distance check, not reprojection.** If the user reserved no hold-out distance, prompt
them to add one.

## 7. Export
- `calibration_metric.toml` is already written in Anipose TOML layout
  (`write_anipose_toml` in `pipeline_utils.py`) → drop into an Anipose project's
  `calibration.toml`. Also expose a generic JSON (K, R, t, dist per camera) for
  non-Anipose users.

---

## N views (the generalization)
Nothing in the pipeline is hard-coded to 3 cameras:
- Camera set = whatever videos are dropped in; names inferred from filenames and
  threaded through as a list everywhere (`camera_names`).
- VGGT consumes an arbitrary number of images per frame in one batch.
- Prior aggregation, BA, `selected_points.json`, and the TOML writer are all keyed
  by camera name / iterate over the camera list.
- **The only component that must be written N-view-aware in the GUI is the
  annotation canvas** (display N panes, correspondence-by-id across all of them).
- Practical notes for the GUI: more views → easier triangulation and a stronger
  prior, but also more clicking; GPU memory for VGGT grows with N×resolution
  (batch per frame) — chunk if needed.

## Packaging checklist
- [ ] Bundle/point to VGGT-Omega weights; verify CUDA at startup, fall back with a
      clear message if no GPU (stage 2 is the only GPU stage).
- [ ] Wrap stages 1–3, 5–7 as a background job with a progress bar (stage 2 is the
      long one).
- [ ] Annotation canvas (stage 4) — the main new UI; emits `selected_points.json` v3.
- [ ] Results view = the 3-tier quality report, held-out distance front and center.
- [ ] One-click export to Anipose `calibration.toml`.
- [ ] Python deps: torch (CUDA), opencv, numpy, scipy, tomli/tomllib. Refinement &
      aggregation are pure-CPU and fast.
