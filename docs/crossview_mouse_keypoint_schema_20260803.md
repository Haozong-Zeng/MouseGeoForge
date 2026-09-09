# Cross-View Mouse Full-Body Keypoint Schema

Date: 2026-08-03

## Decision

Use a view-invariant anatomical core for top, side, bottom, and oblique views.
Move silhouette-dependent points such as body side edges out of the mandatory
core and retain them as optional view-specific auxiliary supervision.

The proposed full-body core has 24 points. It intentionally omits finger and
toe detail, but includes the four paws and the major limb joints.

## Canonical Body Frame

- Origin: `spine_lower` or a pelvis-centered root derived from it.
- Positive X: caudal-to-cranial, from tail base toward nose.
- Positive Y: anatomical left.
- Positive Z: ventral-to-dorsal, from belly toward back.

Left/right names always refer to the animal, never to image left/right. Camera
projection determines how those labels appear in top, side, or bottom images.

## Core-24

| ID | Name | Role |
| ---: | --- | --- |
| 0 | `nose` | head direction and cranial endpoint |
| 1 | `left_ear` | head orientation |
| 2 | `right_ear` | head orientation |
| 3 | `neck` | head-to-torso connection |
| 4 | `spine_upper` | shoulder-girdle center |
| 5 | `spine_mid` | torso centerline |
| 6 | `spine_lower` | pelvis/lower-torso center |
| 7 | `tail_base` | tail root and body root reference |
| 8 | `left_shoulder` | left forelimb root |
| 9 | `left_elbow` | left forelimb articulation |
| 10 | `left_forepaw` | left forepaw/wrist surface target |
| 11 | `right_shoulder` | right forelimb root |
| 12 | `right_elbow` | right forelimb articulation |
| 13 | `right_forepaw` | right forepaw/wrist surface target |
| 14 | `left_hip` | left hindlimb root |
| 15 | `left_knee` | left hindlimb articulation |
| 16 | `left_hindpaw` | left hindpaw/ankle surface target |
| 17 | `right_hip` | right hindlimb root |
| 18 | `right_knee` | right hindlimb articulation |
| 19 | `right_hindpaw` | right hindpaw/ankle surface target |
| 20 | `tail_q1` | proximal tail curvature |
| 21 | `tail_mid` | middle tail curvature |
| 22 | `tail_q3` | distal tail curvature |
| 23 | `tail_tip` | tail endpoint |

The 20/22-point Mouse Pose Analysis Dataset provides a close public precedent:
it includes nose, neck, ears, shoulders, forepaws, hips, hindpaws, spine,
knees, elbows, and tail landmarks, plus calibrated top and side multiview
examples:
<https://google.github.io/mouse-pose-analysis-dataset/>

## Derived Points

Do not spend output channels on points that can be computed robustly:

- `head_center`: weighted mean of nose and ears;
- `mouse_center`: spine-center interpolation;
- `pelvis_center`: spine lower plus bilateral hip midpoint;
- body heading: tail base to neck/nose;
- shoulder width: distance between shoulders;
- hip width: distance between hips.

Derived points can still be compared with source annotations, but should not
define independent, potentially inconsistent degrees of freedom.

## Auxiliary Labels

Keep existing labels as optional losses when a dataset provides them.

Top-view auxiliary:

- left/right body side edges;
- ear tips and eyes;
- extra dorsal centerline points;
- extra tail subdivisions;
- source-specific mouse center.

Side-view auxiliary:

- dorsal and ventral silhouette anchors;
- mouth/chin when relevant;
- visible-side paw contact and rearing height.

Bottom-view auxiliary:

- paw contact masks or paw-surface centers;
- ventral torso mask;
- genital/abdomen landmarks only for experiments that need them.

Water Reaching auxiliary:

- palms, wrists, finger joints, fingertips, mouth, and spout/contact points.

Body side edges are better supervised by a body mask, contour distance loss,
or surface/CSE head than by pretending they are invariant anatomical joints.

## Mapping From SuperAnimal TopViewMouse-27

Direct mappings:

| TopViewMouse-27 | Core-24 |
| --- | --- |
| `nose` | `nose` |
| `left_ear`, `right_ear` | same |
| `neck` | `neck` |
| `mid_back` | `spine_mid` |
| `tail_base` | `tail_base` |
| `left_shoulder`, `right_shoulder` | same |
| `left_hip`, `right_hip` | same |
| `tail1` ... `tail5`, `tail_end` | resample to q1/mid/q3/tip |

Dataset-specific or derived mappings:

| TopViewMouse-27 | Treatment |
| --- | --- |
| `mouse_center` | auxiliary or derived from spine |
| `mid_backend*` | map to spine lower/tail transition only after visual audit |
| `left_midside`, `right_midside` | top-view contour auxiliary |
| eyes and ear tips | head-detail auxiliary |
| `head_midpoint` | derived head center or auxiliary |

New Core-24 points absent from TopViewMouse-27:

- bilateral elbows and forepaws;
- bilateral knees and hindpaws;
- explicit spine upper/lower where no audited mapping exists.

These points can be supervised by the Mouse Pose Analysis Dataset, calibrated
multiview data, Blender, or dataset-specific labels. They remain gradient-masked
for SuperAnimal samples that do not define them.

## View Handling

One synchronized 3D skeleton should project into every camera. Do not maintain
different anatomical naming conventions for top, side, and bottom views.

Per point and view, store independently:

- `defined_by_dataset`;
- `in_frame`;
- `surface_visible`;
- `occluded_but_annotated`;
- `confidence`;
- `joint_world_xyz` and `surface_world_xyz`.

An internal elbow joint and its visible fur-surface proxy are not the same 3D
point. Preserve both, as already done in the Water Reaching annotation format.

For bottom view, anatomical left/right may appear mirrored relative to top
view. Horizontal flips must swap left/right labels after projection; view type
must never redefine the anatomy.

## Model Structure

Use a shared backbone with:

1. a fixed Core-24 heatmap/query head;
2. per-point visibility and uncertainty;
3. optional source/view-specific query tokens;
4. a mask/contour head for body width and side edges;
5. a CSE/surface head for dense mesh fitting;
6. a 3D lifting or fixed-rig triangulation module.

The Core-24 head is the common semantic skeleton. CSE remains complementary,
especially for torso shape, surface alignment, and mesh reconstruction.

## Training Strategy

- Distill common TopViewMouse-27 points into Core-24.
- Preserve all 27-source outputs during replay so useful top-view signals are
  not forgotten.
- Train new limbs and paws from public multiview/home-cage data and Blender.
- Randomize Blender cameras over dorsal, lateral, ventral, and oblique views.
- Balance view types and projected mouse sizes.
- Use synchronized multiview reprojection to teach occluded/far-side joints.
- Apply dataset-definition masks separately from visibility masks.
- Evaluate with cameras and view families held out, not random frames only.

## Evaluation

Report by view family:

- 2D PCK/OKS for each Core-24 group;
- left/right swap rate;
- raw metric and aligned 3D MPJPE;
- limb-length temporal consistency;
- paw-floor contact and penetration;
- tail curve error;
- mask IoU and contour distance;
- performance for visible and occluded points separately.

The first comparison should be:

1. original TopViewMouse-27;
2. Core-24 only;
3. Core-24 plus masks;
4. Core-24 plus masks and CSE.

This isolates whether removing contour landmarks helps cross-view transfer and
whether dense surface supervision adds geometry beyond the anatomical skeleton.
