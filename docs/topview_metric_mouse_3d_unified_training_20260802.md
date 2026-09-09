# Metric Top-View Mouse 3D and Unified Training Feasibility

Date: 2026-08-02

## Decision

The project is feasible if the output contract distinguishes three cases:

1. RGB only, unknown camera and unknown scene scale: predict plausible 3D pose
   and shape in a canonical, scale-normalized frame. Absolute size cannot be
   guaranteed from geometry alone.
2. Known arena geometry or a one-time camera calibration: predict metric mouse
   location and size, with the arena floor defining the world frame and scale.
3. Multiview, RGB-D, or calibrated synthetic supervision: provide metric 3D
   ground truth used to teach the monocular model its height and pose priors.

For open-field and home-cage data, prioritize case 2. Detecting or manually
recording arena geometry once per fixed-camera trial is more reliable than
asking a network to infer physical scale independently in every frame.

## Why Scale Needs an External Anchor

For a perspective camera, uniformly scaling the mouse, scene translation, and
camera distance produces the same 2D projection. A monocular network can learn
that adult mice are usually within a certain size range, but that is a
statistical prior rather than an observable metric measurement.

A known floor plane changes the problem:

- A visible rectangular arena with known corner coordinates defines a metric
  image-to-floor homography.
- If intrinsics are known, the homography can be decomposed into camera pose
  relative to the floor.
- With unknown intrinsics, the floor still gives metric XY for contact points;
  full camera recovery needs assumptions about focal length/principal point or
  additional calibration marks.
- Mouse height above the floor is still not determined by the plane alone and
  must come from learned 3D priors, metric depth, multiview labels, or RGB-D.

For a permanent setup, save a calibration image of an empty arena or a
ChArUco/Aruco target. Do not require the arena boundary to remain visible in
every animal frame.

## Evidence and Reusable Data

- SuperAnimal-TopViewMouse merges 13 overhead mouse datasets into a superset
  keypoint vocabulary. It uses missing-keypoint gradient masking and memory
  replay to avoid forgetting across differently annotated datasets:
  <https://www.nature.com/articles/s41467-024-48792-2>
- The released Mouse Pose Analysis Dataset contains 80 CT scans, 5,460
  annotated home-cage frames, and 108 annotated multiview frames. The CT scans
  are particularly useful for learning realistic variation in mouse shape:
  <https://github.com/google/mouse-pose-analysis-dataset>
- Optical Mouse demonstrates top-view monocular 3D pose using a kinematic
  model and multiview-derived pose prior. Reported per-joint errors are below
  10 mm after registering inferred pose to multiview ground truth, so this is
  evidence for plausible relative 3D rather than guaranteed raw metric scale:
  <https://www.nature.com/articles/s41598-023-40738-w>
- DANNCE demonstrates that calibrated multiview recordings can provide strong
  volumetric 3D supervision for animals across environments:
  <https://www.nature.com/articles/s41592-021-01106-6>
- CalMS21/MABe contributes large top-view social-mouse pose sequences, useful
  for appearance, occlusion, temporal, and multi-mouse representation learning,
  but it does not by itself provide metric monocular 3D:
  <https://data.caltech.edu/records/s0vdx-0k302>
- Generic models such as Depth Pro and Metric3D predict metric monocular depth
  and camera/focal information, but their zero-shot metric estimates should be
  treated as priors until validated on small furry animals and cage imagery:
  <https://machinelearning.apple.com/research/depth-pro>
  <https://openaccess.thecvf.com/content/ICCV2023/html/Yin_Metric3D_Towards_Zero-shot_Metric_3D_Prediction_from_A_Single_Image_ICCV_2023_paper.html>

## Recommended Model

### Trial-Level Geometry

Estimate or load once per video:

- camera intrinsics and distortion;
- camera-to-arena pose or floor homography;
- arena dimensions and geometry type;
- one metric scale value;
- optional individual mouse identity and shape code.

These values must be shared across all frames in a trial. A per-frame camera or
scale head would reintroduce the same temporal inconsistency previously seen
in VGGT camera prediction.

### Frame-Level Mouse State

Predict per frame:

- mouse instance mask and visible silhouette;
- canonical keypoints or surface anchors;
- global mouse translation and heading in arena coordinates;
- articulated pose and above-floor height;
- confidence and occlusion for every output.

The model should consume a camera ray map derived from intrinsics rather than
only resized image coordinates. Random crop, resize, and rotation augmentation
must update intrinsics and arena projections exactly.

### Shape and Size

Separate three variables that are often conflated:

- `canonical_shape`: proportions learned from CT scans and Blender variants;
- `animal_metric_scale`: constant for an individual or trial;
- `frame_pose`: articulation and soft-tissue deformation.

This prevents a crouched posture from being interpreted as a smaller mouse and
prevents camera distance changes from modifying estimated body size.

## Losses

Use labels only when the relevant field is available:

- arena corner/edge reprojection and floor-plane loss;
- metric multiview 3D keypoint loss without scale alignment;
- Procrustes-aligned 3D loss as a separate pose-quality diagnostic;
- body silhouette and part-mask losses;
- visible surface-depth and normal losses;
- constant bone length and trial-level body-size consistency;
- paw/floor contact and floor-penetration penalties;
- fixed-camera and trial-scale temporal consistency;
- multiview reprojection and calibrated triangulation;
- left/right and instance identity consistency for social data.

Do not optimize only scale-aligned MPJPE. It can improve while absolute mouse
size and camera distance remain wrong.

## Unifying With Water Reaching

Joint training is feasible, but use a shared core plus domain-specific geometry
and detail branches.

Shared components:

- image/VGGT backbone;
- mouse detector, mask, canonical body landmarks, and shape prior;
- camera-ray encoding, depth features, and temporal encoder;
- fur/color/lighting representation;
- canonical normalized mouse coordinate system.

Open-field/home-cage components:

- arena floor and boundary head;
- metric XY location, heading, and coarse whole-body 3D;
- multi-mouse detection, identity, and occlusion handling.

Water-reaching components:

- fixed-rig/cup/spout geometry;
- head-fix constraints;
- high-resolution forelimb, palm, wrist, and finger head;
- interaction and contact constraints.

The top-view datasets should not directly dominate the water-reaching decoder.
They contain many more easy torso frames and can bias the shared model toward
flat poses and weak hand detail. Use:

- task-balanced sampling rather than frame-count-proportional sampling;
- apparent mouse-size bins so wide and close shots are balanced;
- task/view tokens or small domain adapters;
- memory replay when adding a new domain;
- separate high-resolution hand crops for Water Reaching;
- per-loss availability masks, already supported by the current trainer.

## Fit to the Current Codebase

The current `finetune_blendedmvs.py` already supports optional camera, depth,
and keypoint availability, canonical keypoint discovery, real multiview
geometry, and fixed-rig supervision. The least disruptive extension is:

1. Add `arena.json` per trial with dimensions, floor coordinates, calibration,
   and confidence.
2. Add a trial sampler that returns several frames from the same fixed camera.
3. Add a shared trial token for camera, arena pose, metric scale, and animal
   shape.
4. Add open-field/home-cage dataset adapters while retaining the existing
   Water Reaching losses.
5. Report metric and scale-aligned metrics separately.

## First Experiments

### P0: Geometry-Only Top-View Baseline

- Select fixed-camera open-field videos with known arena dimensions.
- Estimate one homography per video from an empty-arena frame or four corners.
- Run SuperAnimal-TopViewMouse for 2D keypoints and masks.
- Fit the existing mouse skeleton/mesh with constant animal scale over the
  entire video.
- Measure body-length consistency and floor penetration before training a new
  network.

This establishes how much 3D is recoverable from arena geometry plus existing
2D models.

### P1: Blender Domain Randomization

- Build open-field and simple home-cage Blender scenes in metric units.
- Randomize focal length, sensor size, camera height, tilt, crop, arena size,
  mouse size, fur, lighting, bedding, and occluders.
- Sample top views heavily, with a smaller tail of oblique and side views.
- Export masks, metric depth, camera rays, floor plane, keypoints, and mesh.

### P2: Monocular Student From Multiview Teachers

- Use DANNCE/Anipose/calibrated rigs to produce metric 3D pseudo-labels.
- Train on one view at a time while retaining multiview reprojection during
  training.
- Hold out entire cameras, distances, arenas, and animals for evaluation.

### P3: Unified Training

- Mix top-view synthetic, top-view real, home-cage CT/multiview data, and
  Water Reaching.
- Freeze most of the backbone initially; train trial geometry and domain
  adapters first.
- Unfreeze upper backbone blocks only after each domain has a stable baseline.

## Evaluation

Report all of the following:

- camera rotation, camera height, and focal-length error;
- arena corner reprojection and floor-coordinate error in mm;
- raw metric MPJPE with no similarity alignment;
- rigid-aligned and scale-aligned MPJPE as diagnostics;
- mouse nose-to-tail/body-length MAE in mm;
- body-length coefficient of variation over one trial;
- silhouette IoU and depth error on the mouse only;
- floor penetration and paw-contact error;
- performance by apparent mouse size, view angle, camera model, coat color,
  cage type, and occlusion level;
- Water Reaching palm/finger metrics before and after joint training.

The decisive test is cross-setup: train without one camera/arena/lab and test
on that held-out setup. Random frame splits would overestimate generalization.

## Practical Recommendation

Start with known-arena top views and a fixed camera. This version has a clear
metric solution and can reuse large public 2D datasets without pretending they
contain absolute depth. General arbitrary-camera RGB should be offered as a
plausible canonical 3D mode with uncertainty unless arena calibration,
intrinsics, a depth sensor, or multiview evidence is available.
