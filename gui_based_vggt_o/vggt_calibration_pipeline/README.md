# VGGT Manual Metric Calibration Pipeline

This directory contains the current prototype for turning VGGT/VGGT-Omega
multi-camera frame outputs into an Anipose-compatible calibration with local
metric scale. It is intended for manual point annotation, metric refinement,
and 3D pose triangulation tests.

This README is written as a handoff for optimization. It describes what exists,
what was tested, how to reproduce the current numbers, and where improvement is
most likely to matter.

## Current Status

- The tested dataset is `videos_6yca`.
- VGGT frame outputs are in `/media/tony/backup_inte/vggt_paper/data_set`.
- Analysis and comparison scripts are in
  `/media/tony/backup_inte/vggt_paper/analysis_final`.
- Only one `videos_6yca` frame currently has manual annotations:
  `/media/tony/backup_inte/vggt_paper/data_set/videos_6yca/frame_000051`.
- The current manually annotated file is schema version 2, but the point
  selector now writes schema version 3 when the file is resaved.
- The current recommended configuration (`agg_loose_f`: aggregated VGGT prior,
  loose pose priors, per-camera focal gain refinement, IRLS + outlier drop)
  beats the previous best runs on every metric simultaneously:
  - reprojection RMSE: `0.277 px` (previous best `0.481 px`)
  - fit distance RMSE: `0.475 mm` (previous best `0.763 mm`)
  - leave-one-distance-out RMSE: `0.740 mm` (previous best `0.979 mm`)
  - pose shape error: `0.499 mm` (previous best `0.541 mm`)
- Aggregating VGGT calibrations over all 405 frames (no manual points at all)
  already improves pose shape from `1.502 mm` to `1.087 mm`.

Interpretation: the pipeline improves the original VGGT calibration for local
3D pose shape on `frame_000051`, and the aggregated multi-frame VGGT prior
plus restricted focal refinement now removes most of the previous tension
between low reprojection error and good pose shape. The remaining open
question is robustness across frames, users, and imperfect annotations.

## Important Files

```text
vggt_calibration_pipeline/
  point_selector.py            manual annotation GUI
  refine_metric_calibration.py metric bundle adjustment (single or multi frame)
  aggregate_vggt_priors.py     robust multi-frame VGGT rig prior + empirical sigmas
  auto_refine.py               config grid + LOO-based selection driver
  run_pipeline.py              older batch VGGT frame calibration path
  extract_frames.py
  frame_quality.py
  pipeline_utils.py

/media/tony/backup_inte/vggt_paper/data_set/
  videos_6yca/
    selected_points.json       video-level manual annotation from selector
    vggt_prior/
      calibration_prior.toml   aggregated rig prior (405 frames)
      prior_stats.json         empirical per-camera/per-pair sigmas
      frame_deviations.csv     per-frame deviation from the aggregate
    frame_000051/
      images.npy
      calibration.toml

/media/tony/backup_inte/vggt_paper/analysis_final/
  anipose_eval.py
  compare_calibrations.py
```

## Environment

Run commands from the repository root:

```bash
cd /home/tony/vggt-omega
```

The refinement scripts use Python packages already used by this repo:

```text
numpy
scipy
opencv-python / cv2
toml
```

Anipose triangulation uses the separate `anipose` conda environment:

```bash
conda run -n anipose anipose --config config.toml triangulate
```

## Recommended Workflow

1. Aggregate the per-frame VGGT calibrations into a rig prior (once per
   dataset; the rig is static, so every frame is an independent noisy
   measurement of the same geometry):

```bash
python vggt_calibration_pipeline/aggregate_vggt_priors.py \
  /media/tony/backup_inte/vggt_paper/data_set/videos_6yca
```

2. Annotate one or more frames with `point_selector.py`. The annotated points
   are stationary across frames, so the same point id in different frames is
   the same physical point.

3. Run the configuration grid and selection:

```bash
python -m vggt_calibration_pipeline.auto_refine \
  /media/tony/backup_inte/vggt_paper/data_set/videos_6yca/frame_000051 \
  --calibration /media/tony/backup_inte/vggt_paper/data_set/videos_6yca/vggt_prior/calibration_prior.toml \
  --prior-stats /media/tony/backup_inte/vggt_paper/data_set/videos_6yca/vggt_prior/prior_stats.json \
  --output-dir /tmp/auto_refine_51
```

The winner is copied to `OUTPUT_DIR/best/calibration_metric.toml`, the full
table to `OUTPUT_DIR/comparison.csv`.

Or run the recommended configuration directly:

```bash
python vggt_calibration_pipeline/refine_metric_calibration.py \
  /media/tony/backup_inte/vggt_paper/data_set/videos_6yca/frame_000051 \
  --calibration /media/tony/backup_inte/vggt_paper/data_set/videos_6yca/vggt_prior/calibration_prior.toml \
  --rotation-prior-deg 3 --translation-prior-mm 25 \
  --refine-intrinsics --intrinsics-mode f \
  --sigma-refit-iters 2 --drop-outlier-obs-z 4.0 \
  --loo \
  --output-dir /tmp/refit_agg_loose_f
```

## Aggregated VGGT Prior

`aggregate_vggt_priors.py` loads every `frame_*/calibration.toml` in a
dataset, normalizes per-frame scale (VGGT scale is arbitrary per inference;
mean baseline spread on `videos_6yca` is ±4.6%), robustly trims outlier frames
(some frames have a camera rotated by >100 degrees), and writes:

- `calibration_prior.toml`: chordal-mean rotations, median centers, median
  intrinsics over inlier frames.
- `prior_stats.json`: empirical per-camera rotation/center sigmas, per-pair
  relative rotation/baseline sigmas, and focal sigma percentages, usable as
  calibrated prior strengths via `--prior-stats`.

Measured spread on `videos_6yca` (405 frames, 214 inliers): rotation sigma
0.7-1.2 deg, center sigma ~1% of baseline, focal sigma 4-8.6%. The
single-frame `frame_000051` calibration has camera C focal +6.6% off the
405-frame median, so the aggregated prior also fixes a real intrinsics bias.

Important empirical caveat: feeding the *tight* empirical pose sigmas into the
refinement gives the best leave-one-distance-out numbers but **worse pose
shape**, because the solution stays near the VGGT rig shape instead of
following the manual clicks. The manual clicks are the higher-quality signal
for rig shape. Use the aggregated prior **mean** with loose pose sigmas
(`--rotation-prior-deg 3 --translation-prior-mm 25`) for pose-shape work.

## Data Layout Expected By The Refiner

New selector outputs are saved at the video dataset root:

```text
DATASET_DIR/
  selected_points.json
  frame_000051/
    images.npy
    calibration.toml
```

The annotation file is universal for the video. The JSON records which
`frame_XXXXXX` was displayed when the clicks were made, but the point IDs
represent the same physical points across the video.

The old per-frame layout is still accepted:

```text
FRAME_DIR/
  images.npy
  calibration.toml
  selected_points.json
```

Each manually selected point must be observed by at least two cameras to be
triangulated. Points do not need to be visible in all cameras.

The refiner accepts either a dataset directory containing
`selected_points.json` or multiple frame directories. Cameras and points are
shared across frames (static rig, stationary points), so multi-frame
annotations multiply the constraints on the same camera parameters:

```bash
python vggt_calibration_pipeline/refine_metric_calibration.py DATASET_DIR [options]
```

```bash
python vggt_calibration_pipeline/refine_metric_calibration.py FRAME_A FRAME_B ... [options]
```

Duplicate distances (same point pair annotated in several frames) are
deduplicated; conflicting targets produce a warning and keep the first.

## Select Manual Points

Start the GUI:

```bash
python vggt_calibration_pipeline/point_selector.py \
  /media/tony/backup_inte/vggt_paper/data_set/videos_6yca
```

Optional form:

```bash
python vggt_calibration_pipeline/point_selector.py DATASET_DIR \
  --frame frame_000051 \
  --port 8765 \
  --no-browser
```

By default, saving writes:

```text
DATASET_DIR/selected_points.json
```

Current GUI features:

- per-camera point selection
- points visible in any subset of cameras
- multiple distance constraints
- selectable units: `mm`, `cm`, `m`
- per-image zoom
- wheel zoom
- pan while zoomed
- per-image contrast adjustment
- hidden previous labels by default so old labels do not block new clicks
- optional display of all labels
- per-point click uncertainty in pixels
- per-distance measurement uncertainty
- fit distances and validation-only holdout distances
- 3D preview of points connected by distance edges

The browser server defaults to port `8765`.

## Annotation JSON

The older `frame_000051/selected_points.json` file is schema version 2 and
does not contain per-point or per-distance uncertainty fields. The refiner is
backward-compatible with that file and uses command-line defaults.

When the GUI is used now, it saves schema version 3 as
`DATASET_DIR/selected_points.json`. The important fields are:

```json
{
  "points": [
    {
      "id": "P001",
      "label": "P001",
      "sigma_px": 2.0,
      "observations": {
        "A": {"x": 486.65, "y": 384.08, "source_x": 935.87, "source_y": 738.62},
        "B": {"x": 554.43, "y": 314.33, "source_x": 1066.22, "source_y": 604.49}
      }
    }
  ],
  "distances": [
    {
      "id": "D001",
      "point_a": "P001",
      "point_b": "P002",
      "value": 4.17,
      "unit": "mm",
      "distance_mm": 4.17,
      "sigma_mm": 0.25,
      "use_for_refinement": true,
      "holdout": false
    }
  ],
  "templates": [
    {
      "id": "T001",
      "sigma_mm": 0.3,
      "points": {"P005": [0.0, 0.0, 0.0], "P006": [3.0, 0.0, 0.0], "P009": [3.0, 18.6, 0.0]}
    }
  ]
}
```

Use larger `sigma_px` for ambiguous image clicks. Use larger `sigma_mm` for
less reliable physical measurements. Mark some distances as holdouts to test
metric accuracy without fitting those distances.

`templates` is an optional refiner extension (not produced by the GUI yet):
when several annotated points lie on a rigid object with known geometry, give
their known 3D coordinates in any object frame (mm). The refiner fits a free
6-DOF pose per template and constrains the points to the template shape,
which is much stronger than pairwise distances. Templates need at least three
reconstructable points.

## Run Metric Refinement

Default weighted robust run with fixed intrinsics and fixed distortion
(unchanged legacy behavior, reproduces the original results):

```bash
python vggt_calibration_pipeline/refine_metric_calibration.py \
  /media/tony/backup_inte/vggt_paper/data_set/videos_6yca/frame_000051
```

Useful options:

```text
--calibration PATH            prior/init calibration (default: first FRAME_DIR/calibration.toml)
--prior-stats PATH            empirical sigmas from aggregate_vggt_priors.py
--prior-mode legacy|absolute|relative
--free-scale                  prior rig scale becomes a free variable
--prior-sigma-scale X         multiply all pose prior sigmas
--triangulation pixel|normalized
--origin A
--reprojection-sigma-px 2.0
--distance-sigma-mm 1.0
--rotation-prior-deg 3.0
--translation-prior-mm 25.0
--refine-intrinsics
--intrinsics-mode fxfy_pp|fxfy|f|f_shared
--refine-distortion
--distortion-mode k1|k1k2|full
--intrinsics-prior-percent 10.0
--principal-point-prior-px 50.0
--distortion-prior 0.05
--sigma-refit-iters N         IRLS: rescale click sigmas from robust residuals, refit
--drop-outlier-obs-z Z        drop observations above Z normalized residual, refit
--loo                         built-in leave-one-distance-out cross-validation
--template-sigma-mm 0.5
--loss soft_l1
--loss-f-scale 1.0
--outlier-z-threshold 3.0
--max-nfev 5000
```

Prior modes:

- `legacy` (default): rvec/tvec difference priors toward the scaled prior
  calibration. Bit-compatible with the original script.
- `absolute`: geodesic rotation priors plus camera-center priors; with
  `--free-scale` the prior scale floats so scale error in the initial metric
  estimate is not frozen into the prior anchor.
- `relative`: gauge-invariant priors on pairwise relative rotations and
  baseline vectors. Constrains rig shape only; best with `--prior-stats`.

Intrinsics modes (with `--refine-intrinsics`):

- `fxfy_pp`: fx, fy, cx, cy per camera (original behavior; overfits on
  sparse clicks).
- `fxfy`: fx, fy per camera, principal point fixed.
- `f`: one multiplicative focal gain per camera (recommended).
- `f_shared`: one global focal gain.

Default behavior keeps intrinsics and distortion fixed. Intrinsics and
distortion only move when the corresponding flags are passed.

## What The Refiner Optimizes

The objective contains:

- 2D reprojection residuals, scaled by `sigma_px`
- known-distance residuals, scaled by `sigma_mm`
- rigid-template residuals, scaled by `sigma_mm` (optional)
- camera pose priors toward the (scaled) prior calibration, in the selected
  prior mode
- optional focal-length and principal-point priors
- optional distortion coefficient priors

The default loss is `soft_l1`. The gauge is fixed by keeping one origin camera
fixed (default: first camera in the TOML, `A` for the tested data).

Outputs are in millimeters when the entered distances are in millimeters.

## Refiner Outputs

Each refinement output directory contains:

```text
calibration_metric.toml
points_3d.csv                 now includes posterior sigma_x/y/z_mm columns
distance_report.csv           now includes predicted_sigma_mm
validation_report.csv
reprojection_report.csv       now includes a frame column
intrinsics_report.csv
annotation_outliers.csv       includes dropped observations when --drop-outlier-obs-z is used
camera_positions.csv
camera_pairwise_distances.csv
loo_report.csv                with --loo
summary.json
```

The posterior sigmas come from the Gauss-Newton covariance at the solution
(`pinv(J^T J)` of the sigma-normalized residuals): use them as relative
confidence indicators, and as an identifiability check when intrinsics or
distortion parameters are freed.

Use `validation_report.csv`, `loo_report.csv`, or new holdout annotations to
judge metric generalization. Reprojection error alone is not enough — and on
this dataset **LOO alone is not a reliable proxy for pose shape either** (see
Tested Results).

## Automatic Configuration Selection

```bash
python -m vggt_calibration_pipeline.auto_refine FRAME_DIR [FRAME_DIR ...] \
  --calibration .../vggt_prior/calibration_prior.toml \
  --prior-stats .../vggt_prior/prior_stats.json \
  --output-dir OUT
```

Runs a fixed grid of configurations with `--loo`, writes `OUT/comparison.csv`
and `OUT/selection.json`, and copies the winner to `OUT/best/`.

Selection semantics: LOO RMSE reliably detects overfit *within* the
loose-prior family (it flags `fxfy`, `f_shared`, and loose-`k1` overfit), but
configurations with tight VGGT priors minimize LOO while inheriting VGGT's
rig-shape bias. By default only the shape-validated loose-prior family is
eligible for selection; the tight-prior relative-mode family is run for the
comparison table. Pass `--selection loo` when only local metric distances
matter (then `relative_fs_f_irls_drop` style configs win).

## Current Manual Points In Frame 000051

The current annotation contains 10 points and 7 distances:

```text
D001 P001-P002   4.17 mm
D002 P001-P003  50.00 mm
D003 P003-P004   4.17 mm
D004 P005-P006   3.00 mm
D005 P007-P008   3.00 mm
D006 P005-P009  18.83 mm
D007 P007-P010  18.83 mm
```

## Tested Results On Frame 000051

Pose shape error (`pa_mpjpe_mean`, Procrustes-aligned MPJPE in mm against the
Anipose reference pose) is the headline metric. Comparison CSV:

```text
/media/tony/backup_inte/vggt_paper/data_set/videos_6yca/metric_refinement_anipose_v2/anipose_pose3d_comparison_v2.csv
```

```text
configuration                       reproj_px  fit_mm  LOO_mm  shape_mm
original_vggt (frame 51 only)           n/a     n/a     n/a      1.502
vggt_prior_only (405-frame aggregate)   n/a     n/a     n/a      1.087
weighted_robust_default (old best)    2.482   0.763   0.990     0.541
plain_refine_intr_dist (old low-px)   0.481   0.843   0.979     0.863
agg_loose                             2.471   0.728   0.959     0.540
agg_loose_f_noirls                    1.751   0.711   0.807     0.486
agg_loose_f  (recommended)            0.277   0.475   0.740     0.499
relative_fs_f_irls_drop (best LOO)    0.648   0.436   0.458     0.817
best_existing_videos_6yca               n/a     n/a     n/a      0.475
```

Key findings:

- `agg_loose_f` = aggregated prior mean + loose pose priors (3 deg / 25 mm) +
  per-camera focal gain (`--intrinsics-mode f`) + IRLS + outlier drop. It
  dominates the old default on all four metrics and nearly matches the best
  single frame out of all 405 (`0.499` vs `0.475`).
- The focal refinement is real signal, not overfit: the checkerboard
  reference focals are A 1571 / B 1423 / C 1290 while VGGT predicts
  1543 / 1595 / 1592 (B +12%, C +23% too long). The `f` mode moves them in the
  correct direction. `fxfy_pp`, `fxfy`, and loose-`k1` modes overfit (LOO and
  shape both degrade).
- Tight empirical prior sigmas minimize LOO (`0.458 mm`) but pull shape toward
  the VGGT rig (`0.817 mm`): LOO and shape disagree *across* prior families.
  The manual clicks, not the VGGT prior, carry the shape information.
- The 405-frame aggregate alone (no manual points) improves shape 28% over the
  single-frame calibration. The empirical cross-frame spread is rotation
  0.7-1.2 deg, centers ~1% of baseline, focals 4-8.6%.
- A hybrid using the trusted Anipose intrinsics/distortion with VGGT
  extrinsics priors was tested and is NOT recommended: reference intrinsics
  are inconsistent with VGGT extrinsics (principal points ~120 px off center,
  strong distortion), giving `1.490 mm` shape unless the pose priors are
  loosened heavily (then `0.669 mm` LOO but still poor shape).

## Reproducing The Old Baselines

The legacy path is bit-compatible. Note the saved "plain" experiment rows were
run with `--reprojection-sigma-px 1.0` (the README previously omitted this):

```bash
# old default (weighted robust, fixed intrinsics): 2.482 px / 0.541 mm shape
python vggt_calibration_pipeline/refine_metric_calibration.py \
  /media/tony/backup_inte/vggt_paper/data_set/videos_6yca/frame_000051 \
  --output-dir /tmp/vggt_metric_default_frame_000051

# old low-reprojection run: 0.481 px / 0.863 mm shape
python vggt_calibration_pipeline/refine_metric_calibration.py \
  /media/tony/backup_inte/vggt_paper/data_set/videos_6yca/frame_000051 \
  --output-dir /tmp/vggt_metric_plain_intr_dist_frame_000051 \
  --loss linear --refine-intrinsics --refine-distortion --reprojection-sigma-px 1.0
```

## Reprojection Error Units

The refiner reports reprojection error in pixels because it compares clicked
2D image coordinates to projected 3D coordinates in image space.

Do not compare pixel reprojection RMSE directly to millimeter checkerboard
calibration errors. For this pipeline, report four separate metrics:

```text
2D fit quality:        reprojection RMSE in pixels
metric fit:            fit distance RMSE in millimeters
metric validation:     LOO / holdout distance error in millimeters
3D pose quality:       pose shape error in millimeters
```

Low reprojection error can coexist with worse 3D shape because intrinsics,
distortion, depth, camera baselines, camera poses, and 3D point positions can
trade off while still matching sparse 2D clicks.

## Anipose Pose Triangulation Test

Refined `calibration_metric.toml` files are copied into Anipose project
templates and triangulated. Current run directories:

```text
/media/tony/backup_inte/vggt_paper/data_set/videos_6yca/metric_refinement_anipose_v2/anipose_runs
/media/tony/backup_inte/vggt_paper/data_set/videos_6yca/metric_refinement_anipose_initial/anipose_runs   (older runs)
```

The template used for `frame_000051` is:

```text
/media/tony/backup_inte/vggt_paper/data_set/videos_6yca/frame_000051/anipose
```

To build and run one triangulation:

```bash
run=NEW_RUN_DIR
mkdir -p $run/videos/{calibration,pose-2d,pose-3d,videos-raw}
cp TEMPLATE/config.toml $run/
cp TEMPLATE/videos/pose-2d/*.h5 $run/videos/pose-2d/
cp REFINEMENT_DIR/calibration_metric.toml $run/videos/calibration/calibration.toml
cd $run && conda run -n anipose anipose --config config.toml triangulate
```

## Pose Comparison

```bash
python /media/tony/backup_inte/vggt_paper/analysis_final/anipose_eval.py compare \
  --output-root .../anipose_runs \
  --gt-pose3d /media/tony/backup_inte/vggt_paper/data_set/videos_6yca/pose-3d/pawfilt.csv \
  --output-csv .../anipose_pose3d_comparison.csv
```

The `pa_mpjpe_mean` column is the pose shape error in millimeters.

## Ground Truth Location

The Anipose checkerboard reference results are copied next to the dataset
(byte-identical to the originals in
`/media/tony/phd_thesis/FU_R2L2/FU_R2L2_2024-06-29_1/Trial_31/anipose_base/videos`),
so comparisons do not depend on the phd_thesis drive being mounted:

```text
/media/tony/backup_inte/vggt_paper/data_set/videos_6yca/
  calibration/calibration.toml   reference calibration (GT for geometry comparison)
  pose-2d/pawfilt_cam{A,B,C}.h5  DLC 2D detections (shared by all triangulation runs)
  pose-3d/pawfilt.csv            reference triangulated pose (GT for shape comparison)
```

All Anipose projects share the same `config.toml` (a copy lives in each frame
folder, e.g. `frame_000051/anipose/config.toml`):
`board_type = "checkerboard"`, `board_size = [11, 8]`,
`board_square_side_length = 1.5 mm`. The reference calibration is therefore
in millimeters, and the scale agreement of the refined calibrations
(0.2-1.3% across camera baselines, paw cloud, and manual point cloud) is a
genuine mm-vs-mm validation: the tape-measure-derived scale and the
checkerboard-derived scale agree at the 1% level.

## Comparison To The Anipose Reference

Two further comparisons were run against the checkerboard reference
(`FU_R2L2_2024-06-29_1/Trial_31/anipose_base`, local copy above). Outputs:

```text
/media/tony/backup_inte/vggt_paper/data_set/videos_6yca/metric_refinement_anipose_v2/calibration_geometry/
  geometry_comparison.csv      camera geometry vs reference (compare_calibrations.py)
  predictions/                 the compared calibration TOMLs
```

### Camera geometry (compare_calibrations.py)

```text
calibration                  ATE_scaled_mm  ATE_raw_mm  scale_to_gt  rot_deg  fx_err_pct
vggt_prior_only                    9.02       (raw)       97.76        5.60     12.4
relative_fs_f_irls_drop            9.07        13.22       1.0997      6.38     12.0
legacy_aggprior                   10.71        18.33       0.8773      4.40     12.4
original_frame51                  11.28       (raw)       98.63        7.58     16.2
old_plain_intr_dist               14.04        14.84       0.9603      5.59     20.4
agg_loose                         16.67        23.83       0.8612      8.79     12.4
agg_loose_f_noirls                17.65        19.22       0.9418      7.71      9.1
agg_loose_f                       18.67        19.58       1.0093      4.10     12.1
old_weighted_robust_default       19.65        28.84       0.8317      9.57     16.2
```

Takeaways:

- The 405-frame aggregate alone has the best global camera-geometry agreement
  (9.02 mm scaled ATE vs 11.28 mm for the single frame): aggregation improves
  global rig geometry, not just shape.
- `agg_loose_f` lands the metric scale within 0.9% of the reference
  (`scale_to_gt 1.0093`) and has the best rotation agreement of all refined
  calibrations (4.10 deg), but mid-pack ATE. Camera-geometry ATE and paw
  shape error do NOT rank configurations the same way (the paw workspace is a
  small central volume; the reference also models strong lens distortion that
  all VGGT-derived calibrations approximate with a pinhole), so treat ATE as
  a secondary diagnostic, consistent with the earlier finding that this
  method is a local metric-shape refinement, not a global rig-pose estimator.

### Triangulated points vs the reference

Per-frame paw scale ratio (ours/GT, Frobenius norm of the centered point
cloud; tests metric scale on independent data, which `pa_mpjpe` cannot
because it removes scale per frame):

```text
run                        scale_median   scale_err
agg_loose_f                   1.0125        1.25%
old_plain_intr_dist           0.9879        1.21%
old_weighted_robust_default   1.0337        3.37%
agg_loose                     1.0362        3.62%
agg_loose_f_noirls            1.0377        3.77%
```

Triangulating the 23 manual clicks with the reference calibration itself
gives the reference's own noise floor on the 7 tape-measured distances:

```text
GT-calibration distance RMSE:   0.788 mm   (mean abs 0.601 mm)
GT reprojection RMSE on clicks: 5.97 px
agg_loose_f fit / LOO:          0.475 / 0.740 mm
```

The reference shows the same systematic negative bias on the small 3-4.17 mm
distances (D003 -1.65, D004 -0.67, D005 -0.92 mm) as every refinement, so
that bias most likely lives in the annotations or tape measurements, not in
any calibration. Our LOO error (0.740 mm) is at the reference noise floor;
chasing lower LOO on this annotation set is likely chasing noise.

Manual-point clouds, ours vs GT-triangulated (rigid alignment):

```text
agg_loose_f          0.765 mm RMSE, scale ratio 1.0019
agg_loose_f_noirls   0.748 mm RMSE, scale ratio 0.9868
```

### Known gaps in the reference comparison

- The reference is a single calibration without replicates; its own
  uncertainty is unknown beyond the 0.788 mm distance residuals above.
- Pose comparisons share the same DLC 2D detections between reference and
  candidates, which isolates calibration effects but leaves absolute accuracy
  against physical reality unmeasured.
- Camera-geometry ranks against all 405 per-frame calibrations (the old
  296/405-style ranks) were not recomputed for the new candidates.

## Optimization Targets

Done since the last handoff:

- [x] Aggregated multi-frame VGGT prior with empirical sigmas
      (`aggregate_vggt_priors.py`).
- [x] Multi-frame refinement with shared cameras and shared stationary points.
- [x] Gauge-invariant relative-pose priors and free prior scale.
- [x] Restricted intrinsics parametrizations (`f` mode fixes the
      low-reprojection-vs-shape tradeoff).
- [x] Automatic leave-one-distance-out evaluation (`--loo`).
- [x] Auto-downweighting (IRLS) and outlier dropping with refit.
- [x] Uncertainty propagation to point/distance confidence values.
- [x] Rigid template constraints.
- [x] Config grid + selection + one-file report (`auto_refine.py`).
- [x] Hybrid Anipose-intrinsics mode tested (not recommended; see results).

Most useful next improvements:

- Annotate 2-3 more frames of `videos_6yca` and run the multi-frame bundle:
  shared cameras + stationary points means constraints stack with no new
  camera unknowns. This is the highest-value validation of robustness.
- Validate `agg_loose_f` on other datasets/rigs to check the loose-prior
  recipe generalizes.
- Add GUI support for writing `templates` (rigid known-geometry objects) and
  encourage denser distance graphs, including long-baseline distances.
- Investigate a shape-faithful no-GT selection metric; LOO ranks within the
  loose-prior family but disagrees with shape across prior families.
- Improve the GUI schema migration so opening an old schema v2 file and saving
  writes explicit `sigma_px`, `sigma_mm`, and holdout fields.

## Known Limitations

- Current quantitative testing is one manually annotated frame.
- Sparse manual points overfit unrestricted intrinsics/distortion; only the
  single-gain `f` mode is validated to help shape.
- A low reprojection error does not guarantee the best 3D pose shape, and a
  low LOO error does not either when the pose priors are tight.
- Manual points constrain local metric shape better than global camera rig
  geometry; treat this as local metric-shape refinement, not a global
  rig-pose estimator.
- The method requires at least two camera observations per 3D point
  (observations of the same point from different frames share cameras, so at
  least two *distinct* cameras are required across all frames).
- Metric scale depends on the correctness of the entered distances.
- The existing `run_batch.py` in the repo has local hardcoded paths and should
  be checked before using it for production runs.

## Current Bottom Line

The manual-point refinement is useful and now has a validated recipe. On
`videos_6yca/frame_000051`, `agg_loose_f` improves 3D pose shape from
`1.502 mm` (raw VGGT) to `0.499 mm`, with `0.277 px` manual reprojection,
`0.475 mm` fit distance RMSE, and `0.740 mm` leave-one-distance-out error —
better than both previous best modes on every metric at once. The next
optimization should validate the recipe across more annotated frames (the
multi-frame machinery is already in place) and across rigs.
