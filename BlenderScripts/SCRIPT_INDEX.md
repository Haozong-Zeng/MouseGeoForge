# Blender script index

The scripts remain in one directory because Blender launches many of them as
standalone files and several import sibling helper modules directly. The main
maintained entry points are grouped below.

## BlenderPyMouse annotation and conversion

- `annotate_existing_blenderpymouse_keypoints.py` — backfill Blender rig keypoints
- `blender_water_reaching_rendering.py` — water-reaching dataset renderer
- `convert_cylinder_renders_to_blenderpymouse_core23.py` — Cylinder Core23 conversion
- `convert_openfield_orbbec_to_blenderpymouse_core23.py` — Openfield conversion
- `augment_moseq_blendedmvs_core23.py` — single-mouse Core23 augmentation
- `augment_dual_mouse_blendedmvs_core23.py` — per-instance dual-mouse augmentation

## Homecage pipelines

- `run_homecage_blackfur_core23_full_pipeline.py` — black-mouse Homecage pipeline
- `run_homecage_dual_mouse_diverse200_render.py` — dual-mouse RGB/depth rendering
- `audit_homecage_dual_dataset_scene.py` — source scene audit
- `validate_dual_homecage_blenderpymouse.py` — rendered dataset validation
- `finalize_homecage_selective_side_depth.py` — selective side-wall depth policy

## Geometry and animation helpers

- `geometry_first_dual_mouse.py` — geometry-first two-mouse construction
- `procedural_dual_mouse_motion.py` — procedural dual-mouse motion utilities
- `flexible_tail_solver.py` — tail geometry correction
- `build_vertex_keypoint_map.py` — mesh support-vertex mapping
- `frontlimb_quality_check.py` — forelimb quality metrics

## Auditing and inspection

- `audit_blend_external_dependencies.py` — packed/external resource audit
- `audit_dual_mouse_mesh_overlap.py` — inter-mouse overlap audit
- `audit_dual_mouse_paw_stance.py` — paw stance audit
- `audit_pose_dynamics.py` — animation dynamics audit
- scripts beginning with `inspect_`, `diagnose_`, or `render_*_review` — targeted QA tools

Historical experiment-local scripts remain under `backup/blender/` and are not
dependencies of the active pipelines above.
