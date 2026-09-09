# MJX / Track-MJX to Blender Feasibility Notes

Date: 2026-06-09

## Downloaded assets

- Model/checkpoint repo:
  `track-mjx/model_checkpoints/mouse-reach-mjx-neurips`
  - 51 files, about 155 MB.
  - Contains `checkpoints/joints_only`, `checkpoints/physics-aware`, and rollout HDF5 files.
- MIMIC-MJX mouse arm data:
  `track-mjx/data/mouse_arm`
  - 46 trial HDF5 files, about 2 MB total.

The `track-mjx` repository convention is to keep data under `data/...` and model outputs under
`model_checkpoints/...`, so these paths match the local project layout.

## Relevant local evidence

- The NeurIPS mouse-arm paper describes a pipeline:
  1. multi-camera pose estimation,
  2. STAC-MJX registration to a MuJoCo forelimb model,
  3. Track-MJX imitation learning in MJX,
  4. physics-aware regularization using control/energy costs.
- The downloaded `mouse_arm` HDF5 clips contain:
  - `qpos`: `(100, 4)` with joints `sh_elv`, `sh_extension`, `sh_rotation`, `elbow_joint`
  - `qvel`: `(100, 4)`
  - `kp_names`: `Shoulder`, `Elbow`, `Wrist`
  - `marker_sites`: `(100, 3, 3)`
  - `xpos` / `xquat` for 9 bodies including `scapula`, `humerus`, `ulna`, `radius`, `wrist_body`
- The downloaded rollout files contain:
  - `qposes_ref`: `(46, 200, 4)`
  - `qposes_rollout`: `(46, 120, 4)`
  - `actions`: `(46, 119, 9)` for 9 muscle-like actuators
  - latent and decoder activations.
- Quick comparison of rollout files:
  - physics-aware qpos RMSE vs reference: about `0.221`
  - joint-only qpos RMSE vs reference: about `0.201`
  - physics-aware action magnitude is much lower and smoother than joint-only.
  - This matches the paper's warning: physical regularization improves naturalistic control but can trade off exact joint tracking.

## Fit to the Blender problem

This can help the Blender project, but as a forelimb physical prior first, not as a full replacement
for the Blender animation pipeline.

Reasons:

- The MJX asset currently covers a right forelimb model with 4 DoF and a wrist end effector.
- Our Blender rig is a full mouse with mesh deformation, fingers, cup, water pipe, tail, cranial window,
  and camera-specific visual targets.
- The MJX mouse-arm model has useful shoulder/elbow/wrist dynamics, but it does not directly model
  Blender fingers, cup contact, water pipe grasping, or whole-body/tail placement.
- The exact visual problems we have been fixing are mostly:
  - front-limb collapse,
  - cup/paw near-contact or penetration,
  - water-pipe reach,
  - mesh volume preservation.
  MJX can strongly help the first item and partly help smoothness, but the Blender mesh/collision checks
  should remain the authority for the other items.

## Recommended integration path

1. Keep the current Blender migration as the source of whole-body placement.
2. Build a small MJX prior module that reads:
   - `track-mjx/data/mouse_arm/*.h5`
   - `track-mjx/model_checkpoints/mouse-reach-mjx-neurips/rollouts/rollouts_full_mouse_arm_physics_aware.h5`
3. Convert each Blender forelimb pose to a compact shoulder-elbow-wrist feature:
   - shoulder to elbow vector,
   - elbow to wrist vector,
   - reach ratio,
   - elbow angle,
   - shoulder-frame azimuth/elevation/twist.
4. Learn or fit a nearest-neighbor / PCA manifold prior from the MJX `qpos`, `marker_sites`, and
   physics-aware rollout.
5. Add this prior to `source_mesh_match_refine.py` as a soft score:
   - source mesh matching remains the data term,
   - arm volume remains a hard/strong term,
   - cup/water-pipe checks remain hard gates,
   - MJX prior penalizes unnatural shoulder/elbow geometry and high-frequency jitter.
6. For collapsed intervals, search candidate poses in this order:
   - current IK candidate,
   - interpolation from nearby non-collapsed frames,
   - nearest MJX-manifold forelimb pose aligned to the current Blender shoulder/wrist target,
   - local roll/palm refinement with Blender collision gates.

## Practical next prototype

Create `blender/mjx_forelimb_prior.py` with:

- HDF5 loader for MJX mouse-arm clips and physics-aware rollouts.
- Feature extraction from Blender `Scapula_R` / `Wrist_R` and mirrored left-arm support.
- A prior score:
  `nearest_mjx_feature_distance + velocity_smoothness + elbow_range_penalty`.
- A report mode that evaluates current r10 frames and flags where MJX prior disagrees with Blender
  QC.

Then add an optional argument to `source_mesh_match_refine.py`:

`--mjx-prior-json blender/quick_checks/mjx_forelimb_prior.json`

This keeps the first integration lightweight: no JAX/MuJoCo install required, no GPU dependency,
and no full-policy rollout inside Blender. If that improves the remaining collapse/near-contact frames,
the next stage can run actual Track-MJX rollouts outside Blender and import qpos trajectories.

## Bottom line

Use MJX now as a physical regularizer and candidate-pose generator. Do not try to replace the current
Blender rig with MJX yet. The immediate win is a better forelimb manifold for preventing elbow/upper-arm
collapse while preserving the Blender-specific visual and collision constraints.
