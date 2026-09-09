# VGGT Synthetic Supervision Roadmap

Last updated: 2026-07-23

## Goal

Improve VGGT-Omega camera and dense geometry finetuning with BlenderPyMouse
supervision that transfers to real water-reaching recordings, especially for
the palms, fingers, face, mouth, cup, and spout.

## Current State

- RGB, camera intrinsics/extrinsics, and depth are available per rendered view.
- The finetuning loader already supports optional body and semantic masks.
- The renderer has a legacy `vertex-keypoint-map` hook, but current annotations
  are empty when no map is supplied.
- The legacy map stores `vertex_id -> bone_name` and uses nearby surface
  vertices to decide whether an anatomical keypoint is visible.
- VGGT-Omega currently predicts camera encoding and dense depth. It has no
  keypoint prediction head, so keypoints must first be converted into depth,
  reprojection, crop, or multiview-consistency supervision.

## Immediate Tasks

- [x] Add configurable frame stepping to the Blender renderer.
- [x] Build a first rest-pose hand `vertex-keypoint-map`.
- [x] Export a JSON audit with topology signature, selected vertices, ring
  coverage, and rest-pose landmark coordinates.
- [x] Produce a debug Blender file with colored support vertices.
- [x] Visually inspect every finger root/PIP/DIP support ring in the debug file.
- [x] Add bilateral elbow and nose-tip surface visibility supports.
- [x] Add rectangular VGGT-Omega training inputs with correctly scaled intrinsics.
- [ ] Adjust exceptional rings manually if a support set crosses onto an
  adjacent finger or the palm.
- [ ] Make the renderer reject a map whose topology signature does not match
  `MS_Mouse`.
- [ ] Replace the hard-coded hand landmark list with a canonical landmark
  configuration shared by Blender, Anipose, and the VGGT loader.

## Vertex Map V2 Artifacts

- generator: `blender/build_vertex_keypoint_map.py`
- legacy renderer map:
  `blender/keypoint_mapping/mouse_hand_vertex_keypoint_map.pkl`
- topology and selection audit:
  `blender/keypoint_mapping/mouse_hand_vertex_keypoint_map.audit.json`
- rest-pose visual inspection file:
  `blender/keypoint_mapping/mouse_hand_vertex_keypoint_map_debug.blend`
- rendered closeups: `output/vertex_keypoint_map_debug/`

V2 contains 29 landmarks and 16 unique surface support vertices per landmark,
for 464 mapped vertices:

- 26 bilateral palm/finger landmarks use joint-plane support rings.
- `Wrist_L` and `Wrist_R` are the Blender-rig names whose bone heads coincide
  with the left and right elbows. They use tight local joint patches from the
  forearm and scapula vertex groups because a full plane ring intersects the
  torso.
- `HeadEnd_M` uses a visible nose-cap patch anchored 0.27 of the rest-bone
  length from its head toward its tail.

The annotation coordinate and visibility support are intentionally distinct:
elbow annotation remains at the `Wrist_L/R` bone head, while its mesh support
lies on the nearby visible surface. The nose annotation uses the same 0.27
surface anchor as its visibility patch.

The topology hash, rest-position hash, vertex count (9984), and polygon count
(10000) match across the base bilateral-hand IK file, the V19 shoulder-follow
file, the BrownFur FScamC mapping file, and the WhiteFur FScamC mapping file.
The same V2 map is therefore reusable across those checked projects.

## Existing BlenderPyMouse Keypoint Backfill

Completed on 2026-07-23 for all five existing training datasets derived from
the matching projects under `blender/predefined_animation`:

- 935 frame scenes
- 4,675 rendered views
- 29 landmarks per view
- 135,575 landmark/view instances
- 126,355 surface-visible instances (93.20%)

Each `frame_*` directory now contains:

- `keypoints.npz`: compact training arrays for world, camera, UV, visibility,
  support vertices, intrinsics, and extrinsics.
- `keypoints.json`: human-readable 3D and per-view annotations.
- populated `annotations/########.json` keypoint entries.
- populated keypoints in `scene_meta.json`.

The root `metadata.json` records the keypoint format, source blend, map, names,
visibility tolerances, and coverage.

Implementation and validation:

- backfill: `blender/annotate_existing_blenderpymouse_keypoints.py`
- projection visualization:
  `tools/visualize_blenderpymouse_keypoint_annotations.py`
- structural/reprojection audit:
  `tools/audit_blenderpymouse_keypoint_annotations.py`
- QA contact sheet: `output/blenderpymouse_keypoint_qa/contact_sheet.png`
- audit report: `output/blenderpymouse_keypoint_qa/audit.json`

The independent audit found zero structural errors. Reprojecting
`joint_world_xyz` with the stored camera intrinsics/extrinsics reproduced the
stored UV coordinates with a maximum error of 0.0000892 pixels. Twenty
stratified overlays across fur colors, actions, frames, and cameras were
visually checked without detecting an axis flip, camera-index mismatch, or
left/right swap.

## Canonical Landmark Data

Use real-data names wherever a direct correspondence exists:

- palms: `l_mid_A`, `r_mid_A`
- fingers: `l1_joint_A`, `l1_mid_A`, `l1_tip_A` through side/finger 4
- forelimbs: left/right elbow, mapped in Blender to `Wrist_L`, `Wrist_R`
- face: nose tip, mapped in Blender to `HeadEnd_M`
- interaction: `mouth_A`, `spout_tip_A`
- fixed rig anchors: `point1`, `point2`, `point3`

For each landmark, retain two distinct 3D positions:

1. `joint_xyz`: the anatomical bone head/tail used for pose and reprojection.
2. `surface_xyz`: a mesh surface anchor used for visible depth supervision.

Never use an internal bone joint depth as visible-surface depth.

Recommended per-frame arrays:

```text
names                 [K]
joint_world_xyz       [K, 3]
surface_world_xyz     [K, 3]
camera_xyz            [V, K, 3]
uv                    [V, K, 2]
surface_depth         [V, K]
in_frame              [V, K]
surface_visible       [V, K]
occluded              [V, K]
supervise_depth       [V, K]
part_id               [K]
importance            [K]
```

Coordinate conventions must be explicit:

- `uv`: original image pixels, top-left origin.
- camera coordinates: OpenCV camera-from-world, positive Z forward.
- depth target: camera Z, not Euclidean ray distance.
- any dataset scale normalization must also divide camera translation,
  `joint_xyz`, `surface_xyz`, and landmark depth by the same scale.

## Training Tasks

### Stage 1: Existing Heads

- [x] Load optional landmark arrays without requiring every scene to have them.
- [x] Add Gaussian keypoint-region weights to the dense log-depth loss.
- [x] Add bilinear sparse surface-depth loss at visible landmarks.
- [x] Add GT-world-keypoint reprojection loss for the predicted camera.
- [ ] Add multiview 3D consistency after backprojecting predicted landmark
  depths into the canonical reference camera.
- [ ] Log palm, fingertip, mouth, body, and scene losses separately.

Implemented in `vggt-omega/training/finetune_blendedmvs.py`.

The loader aligns the selected NPZ view IDs, rescales UV coordinates for the
training image size, transforms Blender-world landmarks into the same
first-camera canonical gauge used by the camera labels, and divides landmark
coordinates by the same median-depth or median-baseline scale as camera
translation and depth.

New options, all backward-compatible with zero default loss weight:

```text
--load-keypoints / --no-load-keypoints
--keypoint-region-weight
--keypoint-region-sigma
--keypoint-sparse-depth-weight
--keypoint-reprojection-weight
```

Initial balanced pilot weights based on the pretrained checkpoint loss
magnitudes:

```text
--keypoint-region-weight 0.1
--keypoint-sparse-depth-weight 0.25
--keypoint-reprojection-weight 0.02
```

Validation metrics are logged even when their training weight is zero:

- `keypoint_sparse_depth_mae`
- `keypoint_reprojection_error_px`
- labeled/in-frame/visible/valid point counts

Numerical test: `tools/test_vggt_keypoint_supervision.py`. At 624 x 416, the
canonicalized/scaled GT points reproject within 0.000153 pixels, and all three
new losses produce finite non-zero gradients. A no-keypoint BlendedMVS sample
also remains loadable with zero-length landmark tensors.

A two-step optimizer smoke run also passed with the proposed weights,
`train_mode=light_heads`, and 624 x 416 inputs:

- run directory: `output/vggt_keypoint_supervision_smoke/`
- trainable parameters: 2,123,817
- active terms: camera, dense depth, keypoint-region depth, keypoint sparse
  depth, and keypoint reprojection
- lightweight checkpoint:
  `output/vggt_keypoint_supervision_smoke/keypoint_light_heads_latest.pt`

The pretrained checkpoint's synthetic keypoint camera reprojection error was
roughly 220-358 pixels on the two sampled training batches and 973 pixels on
the held-out smoke frame, despite much smaller pose-encoding losses. This
supports keeping direct geometric reprojection as both a validation metric and
an auxiliary camera-training objective. These smoke values are not a convergence
result and should not be used to select a production checkpoint.

### Stage 2: Masks And Crops

- [ ] Render a binary mouse mask.
- [ ] Render part masks for head, ears, torso, upper arm, forearm, palm,
  fingers, tail, cup, and spout.
- [ ] Generate synchronized multiview hand and face crops.
- [ ] Update intrinsics after cropping instead of treating crops as full images.
- [ ] Balance full-scene, hand-crop, and face-crop samples.

### Stage 3: Optional New Heads

- [ ] Evaluate an auxiliary keypoint heatmap head.
- [ ] Add confidence-weighted tail spline fitting and ordered curve loss; use
  visible-fragment supervision and view-level robust rejection for occluded
  side/oblique views. See [tail curve supervision design](tail_curve_supervision_design_20260825.md).
- [ ] Evaluate a cross-view landmark descriptor/correspondence head.
- [ ] Only keep an auxiliary head if it improves held-out real fixed-rig 3D
  error, reprojection error, and palm-depth error.

## Resolution Plan

The model uses patch size 16 and dynamic RoPE/UV position encoding, so higher
resolution is architecturally supported when both dimensions are divisible by
16. The loader now accepts `--image-width` and `--image-height`; leaving both at
zero preserves the existing `--image-size` square behavior.

| Input | Patches/view | Relative tokens vs 256 square | Comment |
| --- | ---: | ---: | --- |
| 256 x 256 | 256 | 1.00x | Current low-cost finetuning |
| 384 x 256 | 384 | 1.50x | Preserves the rendered 3:2 aspect ratio |
| 512 x 336 | 672 | 2.63x | Near-512 width without square distortion |
| 624 x 416 | 1014 | 3.96x | Official balanced preprocessing for 3:2 images |
| 768 x 512 | 1536 | 6.00x | Higher-resolution ablation |
| 960 x 640 | 2400 | 9.38x | Fits current head-only setup but exceeds native scale |

Attention cost can grow approximately with the square of token count. Relative
to 256 square, 384 square has 2.25x tokens and potentially about 5.1x attention
cost; 512 square has 4x tokens and potentially about 16x attention cost.

The current workstation has an NVIDIA RTX PRO 6000 Blackwell Max-Q with 97,887
MiB VRAM. A local one-step benchmark reproduced the current expensive graph
layout: one three-view synthetic batch plus one batch of eight single-view real
images, frozen aggregator, full dense head trainable (32,330,784 parameters),
and one AdamW update.

| Input | Peak allocated VRAM | Increment above loaded model |
| --- | ---: | ---: |
| 256 x 256 | 5.04 GiB | 0.78 GiB |
| 384 x 256 | 5.42 GiB | 1.16 GiB |
| 512 x 336 | 6.27 GiB | 2.01 GiB |
| 624 x 416 | 7.27 GiB | 3.01 GiB |
| 768 x 512 | 8.80 GiB | 4.54 GiB |
| 960 x 640 | 11.34 GiB | 7.08 GiB |

These numbers are specific to the currently frozen aggregator. They must be
remeasured before unfreezing substantial transformer depth, because activation
and optimizer-state memory will increase sharply.

Recommended sequence:

1. Compare 512 x 336 and 624 x 416. Start with 624 x 416 because it matches the
   official `VGGT-Omega-1B-512` balanced preprocessing scale for 3:2 images.
2. Test 768 x 512 only after 624 x 416 improves held-out real palm-depth and
   fixed-rig geometry metrics. Do not choose 960 x 640 merely because it fits.
3. Use gradient accumulation and activation checkpointing if the unfrozen
   network depth increases.
4. Compare native-scale full frames against synchronized high-resolution
   hand/face crops. Crops are likely more efficient for finger and nose detail.
5. If full-frame resolution beyond the native scale is consistently useful,
   evaluate an HD-VGGT-style low-resolution global branch plus lightweight
   high-resolution refinement instead of indefinitely enlarging global tokens.
6. Select resolution using held-out real metrics, not synthetic loss alone.

Primary references:

- VGGT-Omega paper: https://arxiv.org/abs/2605.15195
- official VGGT-Omega code and runtime table:
  https://github.com/facebookresearch/vggt-omega
- HD-VGGT high-resolution architecture:
  https://arxiv.org/abs/2603.27222

## Validation Matrix

- baseline: current checkpoint and 256 square
- masks only
- keypoint-weighted depth
- keypoint depth plus reprojection
- 512 x 336 full-frame
- 624 x 416 full-frame
- 624 x 416 plus hand/face crops
- 768 x 512 full-frame ablation

For every experiment report:

- fixed-rig Anipose 3D RMSE after one video-level similarity alignment
- 2D reprojection error
- real palm-depth MAE
- camera rotation/translation error
- synthetic and real depth loss by semantic region
- peak GPU memory and steps per second
