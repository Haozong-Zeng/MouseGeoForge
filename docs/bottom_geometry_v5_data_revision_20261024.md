# Bottom-geometry v5 data revision: shared asymmetric environments (DRAFT)

Status 2026-10-24: **draft, not registered.** Nothing has been rendered or trained.
The user chose this route (data revision before L0/P0/P1) after E0. Registration
requires the user's review of the defaults below and the workstation inputs listed
under "Prerequisites". The 8 sealed rigs stay sealed; Water Reaching, DANNCE2,
GR2/UR2/UR3 and protected TEST remain excluded.

## Why: E0 evidence (zero training, 2026-10-24)

Release model, fixed v4 evaluator, Slurm 3755863, outputs in
`output/v4_E0_h100_20261024/` (data disk). Bottom star-pair rotation error, median:

| Condition | Unseen synthetic rigs (72 pairs, 12 rigs) | Free-motion replay (13) | Known real rig (188) |
| --- | ---: | ---: | ---: |
| Original images | 132.5° | 52.0° | 161.3° |
| Gray background (GT body mask kept) | **20.4°** (11/12 rigs lower) | **7.5°** (13/13 < 15°) | not run (no admitted masks) |
| Blurred background | 127.7° | 102.3° | not run |
| v3 EMA arms A0/A1-GT/A1-SI/A2, original | 128–134° | 52–58° | 152–161° |

Gray also improves non-bottom views on unseen rigs (star rotation 91.0° → 15.0°;
raw joint coverage32 0.2 → 0.7), while blur degrades them. The v3 arms are no
better than release under the corrected evaluator, so the v3 failure is not a
scoring artifact. TRAIN-probe baseline for L0 (release, 48 probes): 146.7°, 0% < 15°.

Visual inspection of a TRAIN render (`render_shard0/train_rig00/*/frame_000049`):
the crossed bottom view sees the mouse against an almost featureless bright
gray backdrop, while side views see the cylinder, lamp and lab room. The real
bottom camera (e.g. AZL3 camA) is grayscale and looks up through a glass floor
disk at dark rig structure, the cylinder rim and reflections. Gray-background
success is a distribution-changing diagnostic: it supports a background
hypothesis but does not by itself distinguish symmetry from "background
content inconsistent across views" or from synthetic-backdrop unrealism.

## Hypothesis to register

The release model's bottom-camera failure on crossed renders is driven by
background content that all cameras do not share consistently. Rendering each
scene inside one shared, asymmetric 3D environment that every camera sees
(including the bottom camera, through the floor) makes bottom pose learnable on
TRAIN (L0) and transferable to held-out rigs **and** held-out environments.

## Design (defaults for review)

1. **Environments.** Seeded, procedurally generated layouts built from Blender
   primitives and procedural materials only (no new external assets/licences).
   Each layout = transparent floor disk with an off-centre rim feature and
   scratches visible from both sides; cylinder wall; 3–8 fixtures from a fixed
   library (rods, clamps, boxes, cables, lamp heads, housings) placed
   asymmetrically in 3D above **and** below the floor; a procedural room shell.
   Geometry the motion touches (floor, ClimbWall wall) is kept from the source
   scene. Every generated parameter is recorded per layout.
2. **Lighting.** Seeded per scene: 1–3 area/spot lights (position, size,
   intensity), colour temperature 3500–6500 K, environment strength; identical for
   all cameras of the scene.
3. **Forbidden shortcuts** (plan doc): no camera IDs, no pose-encoded colours, no
   per-view independent background changes, nothing correlated with camera role.
   Fixtures are never placed at, or aimed from, render-camera positions; the
   render cameras' own equipment stays hidden in RGB (v3 policy).
4. **Splits.** Keep the v3 rig geometry and rig splits unchanged (48 TRAIN /
   12 DEV / 8 sealed). Layouts: 24 TRAIN / 6 DEV / 6 sealed.
   - TRAIN: each TRAIN rig × 2 TRAIN layouts.
   - DEV primary: each DEV rig × 1 DEV layout (unseen rig and unseen environment).
   - DEV attribution: each DEV rig × 1 TRAIN layout (unseen rig, seen environment).
   - Sealed: each sealed rig × 1 sealed layout; files only, never scored without a
     separately reviewed frozen policy.
5. **Motions/frames.** ClimbWall + MoSeq First200, frames 49/97/145, 3 views,
   384 px, as v3. Scenes: TRAIN 576, DEV 144, sealed 48 → 768 scenes / 2,304
   views (v3: 408 / 1,224). CRIM13 stays free-motion replay only.
6. **Truth and admission.** Unchanged v3 policy: Cycles RGB, primary-ray
   camera-Z skin truth, glass/hair excluded from truth; visibility rule ≥16 body
   pixels at 96 px and ≥100 at 384 px, max 32 seeded resamples, same for all splits.
7. **Deferred, separately registered factors.** Grayscale/IR appearance (the real
   rig is monochrome); visible housings of the *other* cameras (real images show
   them); mixing the original scene environments back into TRAIN.

## Sequence after rendering

1. Render QA (truth round-trip, visibility, inventory), freeze a v5 manifest and
   data pins; no change to v4 evaluator semantics.
2. **E0-v5 (zero training):** release on v5 DEV primary and attribution sets,
   original and gray. The release model must still fail on original v5 DEV, or
   the revision has changed the problem rather than the data.
3. **L0 on v5 TRAIN** (300 updates, P0-only, v4 criteria: relative-rotation
   component −30%, TRAIN bottom median −20%) with a learnability receipt whose
   metric definitions are frozen before the run.
4. **P0/P1** on identical v5 data, as registered in the v4 plan; teacher gains are
   not attributed to the data change.

## Prerequisites

**Rendering happens off the cluster** (user instruction 2026-10-24); the source
scenes and their dependencies stay on the workstation:
`BlenderPyMouseProjects/Cylinder/PyMouseStudio-BrownFur-ClimbWall.blend`
(sha256 `e9d2a718…`) and
`BlenderPyMouseProjects/Cylinder/PyMouseStudio-fur-SceneOnly-MoSeqPose-Sample10-First200-mm.blend`
(sha256 `d99102c9…`).

Needed on the cluster, uploaded from the workstation:

1. **Source files (small; commit to Git).** The final v3 render script
   `BlenderScripts/render_bottom_geometry_v3_cycle_20261018.py`
   (sha256 `0dd33a68df7607d3062b0cbe84416490ade51b4d480259f722784b91ebe70448`),
   absent from Git although `tools/bottom_geometry_v3_data_20261018.py` checks its
   hash; the scripts that generated `render_design.json` and the visibility
   prescreen; later the v5 render/environment scripts, whose hash the v5
   manifest builder must check the same way.
2. **v5 renders, in the v3 layout.** Per scene `record.json`, `rgb_{0,1,2}.png`,
   `truth_{0,1,2}.npz`; per shard `completion.json` (records, rejected,
   config/script sha256, Blender version), inventories, progress and
   `rejected_configurations/`; `render_config_shard*.json`, `render_design.json`
   with every environment and lighting parameter, and the visibility receipt.
   v3 was 1.54 GB for 408 scenes, so the default 768 scenes are about 3 GB.

Built on the cluster from those uploads (source only, CPU jobs): the v5 manifest
and data pins, E0-v5 evaluation, the L0 learnability receipt, and L0/P0/P1
configs. All outputs and checkpoints go to the data disk via the repo's
`output/` link.
