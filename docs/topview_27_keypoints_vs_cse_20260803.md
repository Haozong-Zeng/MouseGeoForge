# TopViewMouse-27 vs Mouse CSE

Date: 2026-08-03

## Decision

Do not choose one representation for every task.

- Use the SuperAnimal TopViewMouse 27-keypoint vocabulary as the primary real
  image supervision and public-dataset interoperability layer for open field
  and home cage.
- Use a mouse CSE/surface head as an auxiliary representation for mesh fitting,
  silhouette alignment, arbitrary views, and dense whole-body geometry.
- Keep task-specific sparse heads for paws, elbows, wrists, palms, and fingers.

For the first top-view model, the 27-keypoint head should be mandatory and CSE
optional. For the final whole-body 3D model, use both.

## What the 27 Keypoints Represent

The downloaded TopViewMouse-5K vocabulary contains:

- nose, ears, ear tips, eyes, neck, and head midpoint;
- mid-back and several rear-body centerline points;
- mouse center;
- bilateral shoulders, mid-sides, and hips;
- tail base, five intermediate tail points, and tail end.

It contains no paws, elbows, wrists, palms, digits, knees, or ankles. It is a
strong top-view shape and heading representation, but not a complete articulated
mouse skeleton.

## Local Label Coverage Audit

TopViewMouse-5K contains 5,737 annotated instances. The annotation flag `-1`
means that the source dataset does not define that keypoint.

| Landmark group | Approximate definition coverage |
| --- | ---: |
| nose, tail base | 100% |
| mouse center | 81.4% |
| ears, tail end | 77.5-77.7% |
| hips | 43.3% |
| neck | 36.8% |
| body sides | 27.9% |
| intermediate tail points | 11.8-20.5% |
| eyes, ear tips, shoulders | 3.9% |

Across all point slots:

- visible: 46,105;
- defined but not visible: 3,156;
- undefined by the source dataset: 105,638.

Therefore the useful precedent from SuperAnimal is not merely the selected 27
names. It is the combination of a keypoint superset, gradient masking, dataset
conversion, and memory replay.

## Comparison

| Criterion | TopViewMouse-27 | Mouse CSE |
| --- | --- | --- |
| Existing real labels | Strong | Almost none for mice |
| Open-field zero/few-shot use | Strong | Requires new training |
| Top-view heading and body outline | Strong | Strong after training |
| Articulated paws and fingers | Absent | Dense in principle, weak on thin/occluded structures |
| Mesh registration | Sparse and underconstrained | Direct dense image-to-mesh correspondence |
| Arbitrary camera views | Limited by top-view training | Better representation if view-diverse training exists |
| Hidden/underside geometry | Pose prior required | Still requires a pose/shape prior |
| Symmetry ambiguity | Semantic left/right labels help | Needs side/part and temporal constraints |
| Annotation and data conversion | Relatively easy | Exact dense real labels are expensive |
| Metric size | Does not solve scale | Does not solve scale |
| Multi-mouse association | Mature top-down/bottom-up methods | Requires instance-aware surface maps |

Neither keypoints nor CSE recover metric size from an uncalibrated monocular
image. Arena dimensions, camera calibration, RGB-D, multiview geometry, or a
statistical body-size prior are still required.

## Literature Interpretation

SuperAnimal demonstrates that differently labeled mouse datasets can be merged
into a keypoint superset using gradient masks and memory replay:
<https://www.nature.com/articles/s41467-024-48792-2>

CSE predicts a canonical mesh descriptor for each foreground pixel and was
demonstrated on deformable animal categories:
<https://papers.nips.cc/paper/2020/hash/c81e728d9d4c2f636f067f89cc14862c-Abstract.html>

Animal Avatars reports that CSE gives denser and stronger template-fitting
constraints than sparse semantic keypoints, but explicitly retains sparse
keypoints because paws and other thin structures have less accurate CSE:
<https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10197.pdf>

This matches the Water Reaching observations: torso and silhouette benefit from
dense constraints, while fingers and contact interactions need semantic local
targets and kinematic constraints.

## Recommended Architecture

Use one shared image/VGGT backbone with four output families:

1. `topview27_head`
   - 27 heatmaps, visibility, and uncertainty;
   - initialized by distillation from SuperAnimal-TopViewMouse;
   - trained with `-1` gradient masking and source-balanced replay.
2. `surface_head`
   - foreground probability plus a 16D CSE descriptor;
   - trained densely on Blender triangle IDs and barycentric coordinates;
   - trained on real images with masks, temporal consistency, and sparse anchor
     constraints.
3. `task_keypoint_head`
   - configurable query set for open-field paws or Water Reaching forelimbs,
     palms, wrists, and fingers;
   - supervised only when each dataset defines the query.
4. `geometry_head`
   - camera, arena plane, depth, mouse shape/scale, and trial-level consistency.

The heads should share features, not force every semantic point to be derived
from CSE. Some top-view labels such as `mouse_center` are image/shape semantics,
not unique anatomical surface vertices.

## Training Curriculum

### Stage A: Real Top-View Foundation

- Train or distill the 27-keypoint head on TopViewMouse-5K.
- Preserve source-dataset sampling and all undefined-label masks.
- Add real body masks when available.
- Evaluate by held-out source dataset, not random-image split only.

### Stage B: Synthetic Surface Learning

- Export exact Blender face ID plus barycentric coordinates.
- Train CSE, part/side classification, silhouette, depth, and camera heads.
- Randomize top-view camera height, focal length, tilt, mouse size, fur, arena,
  bedding, and occluders.
- Use a fixed, versioned canonical mesh codebook. Do not recompute spectral
  eigenvectors independently for each run.

### Stage C: Mixed Real and Synthetic

- Synthetic samples supervise the full surface and task-specific skeleton.
- TopViewMouse samples supervise available 27 points and appearance.
- Real masks supervise foreground without pretending to provide correspondence.
- Multiview and calibrated arena data supervise metric geometry.
- Balance source datasets and apparent mouse size so top-view torso frames do
  not dominate close-up Water Reaching hands.

### Stage D: Task Refinement

- Open field: add paws and floor contact if a source provides them.
- Home cage: add instance identity, occlusion, and temporal constraints.
- Water Reaching: retain the high-resolution hand/finger head and rig/contact
  losses.

## Existing Mouse CSE Pilot

The local representation-only pilot used the 9,984-vertex `MS_Mouse` mesh and
a 16D graph-spectral codebook.

- Two tested frames, three views each;
- count-weighted sparse-query mean: 5.005 px for spectral CSE;
- canonical XYZ baseline: 5.893 px;
- mean three-view visible mesh coverage: 46.71%;
- no obvious left/right paw color swap in the inspected correspondence maps.

This confirms that the mesh can support a useful dense coordinate system. It
does not yet show that a network can predict CSE on real furry top-view images.

## Go/No-Go Rule

Use the 27-keypoint model as the initial production baseline. Promote CSE to a
required whole-body output only after a learned model passes all of:

- held-out synthetic surface-query median below 6 px;
- left/right visible-paw swap rate below 1%;
- real top-view mask-aligned correspondence is temporally stable;
- adding CSE improves raw metric 3D or mesh reprojection, not only aligned pose;
- Water Reaching palm/finger errors do not regress.

The expected final answer is therefore `27 + CSE + task queries`, not `27 or
CSE`.
