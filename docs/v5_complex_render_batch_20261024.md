# V5 complex-background rendering batch (2026-10-24)

**2026-10-25 update:** all768 scenes/2304 views completed and passed raw file QA.
The user subsequently requested the canonical `dataset/BlenderPyMouse` layout.
The render output below is retained as the immutable source archive; use the
[BlenderPyMouse export](v5_blenderpymouse_export_20261025.md) for the six canonical
motion/split datasets, standard JPEG/PFM/camera files and Core23 annotations.

The user approved the workstation Blender scene and explicitly requested rendering:
“可以，开始渲染，注意符合合成数据的数据格式和文件夹安排”. This authorizes
local data production; it does not start geometry/keypoint/segmentation training.

The rendered dataset is separate from the head-first experiment sample pools. Do not
add new renders to any already-running F/U pair. The old draft's requirement that a
release model must continue to fail is superseded by the head-first route: report
zero-training results honestly; never choose environments to preserve a failure.

## Fixed design and provenance

- Output: `output/bottom_geometry_v5_complex_20261024/`.
- Exactly 768 scenes / 2304 RGB views and matching truth files, at 384px, RGB8,
  Cycles32 samples, denoising, max10 / transmission8 bounces.
- Original v3 48 TRAIN / 12 DEV / 8 sealed camera configurations, unchanged.
- Original admitted ClimbWall and MoSeq sources; frames49/97/145, unchanged.
- 24 TRAIN / 6 DEV / 6 sealed environment layouts. TRAIN uses two layouts per rig;
  DEV has unseen-layout primary and seen-layout attribution; sealed has one layout
  per rig. Counts: TRAIN576, DEV-primary72, DEV-attribution72, sealed48.
- The approved review project is a pinned local template. Only its procedural
  environment collection and world are appended; its mouse and cameras are not
  imported. New layouts vary the global room orientation, fixture groups, panel
  offsets and material gain from deterministic camera-independent seeds.
- The source mouse, animation, contact glass/cylinder and other apparatus remain.
  Original white room shell/top cover and original light objects are hidden as in
  the approved preview; camera equipment follows v3 hiding rules.
- Three physical soft spot lights are seeded once per scene and shared across all
  three cameras. All positions, powers, colors, approximate color temperatures,
  soft radii, world strength and random seeds are recorded. The reference TRAIN
  scene retains the approved layout and lighting.
- Sources, template, design, scripts and per-record payloads are SHA256 pinned.
  Source `.blend` files are never saved. Original data and review files are retained.

## Directory and payload contract

```text
render_design.json
render_config_shard0.json                 # TRAIN576
render_config_shard1.json                 # DEV144 + sealed48
render_shard0|1/<rig>/<environment>/<climb|moseq>/frame_XXXXXX/
    record.json
    rgb_0.png rgb_1.png rgb_2.png
    truth_0.npz truth_1.npz truth_2.npz
render_shard0|1/completion.json
environments/<environment>/environment_parameters.json
environments/<environment>/visibility_receipt.json
environments/<environment>/<source>_inventory.json
lighting/<environment>/<rig>/<motion>/frame_XXXXXX.json
rejected_configurations/<environment>/attempt_XX.json
visibility_prescreen.json
progress.json controller_status.json
manifest.json data_file_pins.json completion.json  # written after final CPU QA
engineering_smoke/                       # excluded from formal data
```

The environment directory is the only added hierarchy level over v3; this avoids
collisions when one rig/motion/frame is rendered in two environments. Paths inside
records are repository-relative POSIX paths. Preserve that prefix on transfer or
use explicit path remapping; do not flatten the tree.

`record.json` keeps the v3 fields: rig/split/dev_tier, source/producer hashes,
frame/kind/group/motion_id, three views, keypoints=null, admitted_motion_split,
apparatus_ablation=false and scene path. V5 adds environment ID/split, DEV condition,
layout/visibility/lighting paths and hashes, and per-payload hashes. `split` retains
v3 spelling `train/val/sealed`. Each view keeps id/image/truth/K/E/physical_role,
depth=null, mask=null, foreground_pixels, projection_closure_max_px and passed.

Each `truth_i.npz` has exactly `depth,body,valid,K,E`. Depth is camera Z, in the same
millimetre-like world coordinates as E translations; invalid depth is zero.
`body` and `valid` are boolean. K uses integer array-index pixel centres
(principal point side/2-0.5); E is 4x4 OpenCV world-to-camera. Glass and hair are
excluded from geometric truth. This is visible opaque **skin**, not a fur mask or
amodal segmentation label. New opaque cables are converted to meshes so they are
included in the existing mesh-only occlusion policy. RGB retains glass and hair.

## Admission, execution and recovery

The WSL controller holds the existing exclusive GPU flock and invokes one native
Windows Blender renderer at a time. An actual two-motion/three-view engineering
smoke must pass independent NumPy/Pillow validation before production starts.
No training, model inference, sealed scoring, Slurm submission or uploads run.

Before each environment is rendered, the same primary-ray rule is checked at96px
for every assigned rig × both source motions × all three frames × three views:
at least16 body pixels and closure<.003px. If any fail, reject the whole shared
environment candidate and retain its receipt; use at most32 deterministic
resamples. Rigs and frames are never changed and hard scenes are never dropped.
The final384px check requires at least100 body pixels and closure<.003px; failure
stops the batch rather than silently selecting another view or partial scene.

RGB/truth files are atomically renamed; `record.json` commits a complete scene
last. Resume accepts only identical code/design/source/layout/lighting pins and
verifies existing payload hashes. Per-shard completion requires every assigned
scene. The final controller validates all expected scene identities, splits,
payload hashes, image sizes, truth schema, valid/background values, K/E and
pixel-centre camera-Z round trips before publishing the manifest and final receipt.

The job has a finite48-hour deadline and20GiB payload cap. `STOP` in this output
directory requests a cooperative stop. On a stuck child/deadline, native process
termination is restricted to the owned Blender PID with the matching unique job
token and script command. Historical STOP files/queues are not touched.

Implementation: [controller](../tools/run_v5_complex_render_20261024.py),
[contract](../tools/v5_render_contract_20261024.py),
[renderer](../BlenderScripts/render_v5_complex_20261024.py),
[format tests](../tests/test_v5_render_contract_20261024.py).
