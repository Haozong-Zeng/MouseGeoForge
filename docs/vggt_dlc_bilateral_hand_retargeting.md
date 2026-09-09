# VGGT/DLC Bilateral Hand Retargeting

## Goal

Retarget one fixed-camera water-reaching sequence to
`PyMouseStudio-BrowmFur-WaterReaching-IK-BilateralHand-Control-HairGN-Experiment`
without allowing per-frame camera or world-coordinate drift.

The solve is deliberately staged:

1. Align the complete video to the Blender world once.
2. Move each wrist/hand controller to the observed palm position.
3. Rotate each hand controller to reproduce palm orientation.
4. Solve finger spread and flexion as bind-relative local bone rotations.
5. Apply collision and temporal regularization before baking.

## Coordinate Systems

### VGGT/DLC reconstruction space

The fixed extrinsic file stores OpenCV world-to-camera matrices:

```text
x_camera = R_world_to_camera * X_vggt + t
C_camera = -R_world_to_camera^T * t
```

The same fixed extrinsics must be used for every frame. Per-frame similarity
alignment would make the reconstructed animal appear stable while silently
moving the cameras and apparatus.

### Blender camera space

Blender cameras look along local `-Z`, with local `+Y` up. Conversion to the
OpenCV camera convention is:

```text
D = diag(1, -1, -1)
R_cv_world_to_camera = D * R_blender_world_from_camera^T
t_cv = -R_cv_world_to_camera * C_blender
```

### Video-level world alignment

Match the three VGGT cameras to the three Blender experiment cameras, then
solve one orientation-guided similarity transform:

```text
X_blender = s * R_vggt_to_blender * X_vggt + t
```

For the current trial, the selected mapping is:

```text
VGGT A -> Blender Camera.003
VGGT B -> Blender Camera.001
VGGT C -> Blender Camera.002
```

The transform must have `det(R) = +1`; a negative determinant indicates a
reflection and usually swaps left/right semantics.

### Apparatus-anchor refinement

Camera calibration can leave a small world-space residual. Refine it once per
video using static landmarks such as the spout tip and three apparatus points:

1. Compute each landmark's robust temporal median after camera alignment.
2. Match those medians to Blender empties placed on the corresponding physical
   apparatus locations.
3. Solve a rigid correction, preferably with fixed scale.
4. Apply the same correction to every keypoint and every frame.

Do not align independently to the mouse body, wrists, or hands. Those are the
motion being reconstructed.

## Palm Controller Solve

The DLC palm center is a skin point, while `CTRL_Hand_L/R` is located near the
hand bone center. Directly assigning the controller to `l_mid_A/r_mid_A` would
therefore introduce a systematic offset.

Construct a palm frame from the four knuckles and palm center:

```text
across = normalize(joint_4 - joint_1)
forward = normalize(mean(joints) - palm_center), orthogonalized to across
normal = normalize(cross(across, forward))
R_palm = [across, forward, normal]
```

At a manually valid bind frame:

```text
o_position = R_palm_bind^T * (P_controller_bind - P_palm_bind)
R_offset = R_palm_bind^T * R_controller_bind
```

At each frame:

```text
P_controller = P_palm + R_palm * o_position
R_controller = R_palm * R_offset
```

This preserves the initial skin-to-bone offset and controller-axis convention.
It also lets the hand cube continue to control both wrist position and palm
turnover.

The bind frame must have valid observations for both hands. Automatic bind
selection ranks common valid frames by distance to the manually initialized
controllers; it should still be visually checked.

## Finger Descriptor Solve

For finger `i`, use three observed points:

```text
proximal = normalize(mid_i - joint_i)
distal = normalize(tip_i - mid_i)
```

Express the proximal direction in the palm frame:

```text
v = R_palm^T * proximal
spread = atan2(v.x, v.y)
base_flex = atan2(v.z, sqrt(v.x^2 + v.y^2))
pip_flex = signed_angle(proximal, distal, palm_across_axis)
```

Use descriptor deltas relative to the bind frame, not absolute Euler angles.
Absolute angles mix camera, palm, rest-bone, and mirrored-rig conventions.

### Automatic local-axis calibration

Do not hardcode left/right signs. At the bind frame, perturb local X and Z of
each finger bone by a small angle and measure the resulting model descriptor
change:

```text
J_base = d(model_spread, model_base_flex) / d(local_x, local_z)
delta_local_xz = inverse(J_base) * delta_observed_spread_flex
```

For the second phalanx, choose the local axis and sign with the largest stable
derivative of model PIP flexion. Keep local Y close to zero because the rig
uses it as a tightly limited twist degree of freedom.

This numerical calibration automatically handles:

- mirrored left/right rest bones;
- different bone-roll conventions;
- fingers whose useful X/Z axes are swapped;
- controller orientation offsets.

The current rig limits suggest local Z is usually the large flexion degree of
freedom, local X is a smaller secondary bend/spread degree of freedom, and
local Y is twist. The Jacobian check remains authoritative.

## Missing Data And Temporal Control

Apply quality gates before Blender keyframing:

- use points seen by at least two cameras;
- reject low DLC/VGGT confidence;
- reject degenerate palm planes;
- unwrap quaternion signs;
- reject implausible frame-to-frame position or rotation jumps;
- interpolate only short gaps;
- leave long gaps unconstrained or blend toward a pose prior.

Smooth positions and rotations separately. Average rotations as unit
quaternions, never by averaging Euler angles or raw matrix entries.

## Collision-Aware Order

For each frame:

1. Disable fingertip IK targets.
2. Move and rotate the hand controller.
3. Correct wrist, palm, mouth, cup, and self-collision.
4. Initialize fingertip targets at the evaluated bone tails.
5. Apply finger spread and flexion targets.
6. Correct finger-to-mouth, finger-to-palm, and inter-finger collision.
7. Bake only after constraints and collision checks pass.

This parent-before-child order prevents fingertips from dragging the wrist or
forcing a finger into reverse-joint bending.

## Current Prototype

`drive_bilateral_hand_controllers_from_vggt_dlc.py` implements stages 1-3 and
adds viewport-only keypoint markers. Finger bones are intentionally unchanged
until local-axis Jacobians and bind-relative angle limits are calibrated.
