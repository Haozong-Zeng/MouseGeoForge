# Cloud handoff: part-balanced depth + joint depth/keypoint encoder

Owner-requested route/code update; this is not a report that training ran.
Read [the route](part_balanced_depth_joint_route_20261028.md) and
[four-arm design](../configs/experiments/part_balanced_depth_joint_20261028.json).

## Meaning and source boundary

"Keypoint semantic loss" means small-part spatial depth weighting, preventing
the torso's pixel area from overwhelming nose/paws/tail. It does not mean merely
adding a keypoint head or setting surface depth equal to internal joint-Z.
Joint training is a separate factor in D0/D1/J0/J1.

At upstream `h100-branch` commit `b7721cd`, the report's depth route files are
absent from Git: `depth_adapt_20261026.py`, `own_rig_hull_20261026.py`, their new
tests and dated running plan. They are reported uncommitted on the cluster.
Do not reset/checkout over that work. Preserve a reviewed source-only snapshot
first (no payloads/weights/logs/secrets), then integrate the new independent
files and reconcile README/history-link hunks. The PR is based on h100-branch;
cloud real head/data adapters remain pending, not falsely claimed integrated.

## Integration required

1. Freeze the actual final cloud light/heavy decision and parent sampler/loss.
   Use the release patch encoder and depth H/KP H0 heads fitted on that SAME
   encoder; verify release/head hashes. Do not silently mix UX-adapted heads.
2. Keep original global SILog, mask BCE, metric anchor, plane and source weights.
   Add `focus_loss` on same-gauge SmoothL1 log residuals, following the route.
   strength0 is exact OFF; strength is already included, not multiplied twice.
   The4x cap is on pixel error coefficients, not all network gradients.
3. Join only admitted TRAIN visible points/part support to depth, with identical
   recorded crop/resize transforms. Gate unknown/occluded/depth-invalid support.
   No current-student mask selection, DEV points or gauge-only landmarks.
4. Wrap current depth decoder as `depth_head(four_taps,rgb)` and KP head as
   `keypoint_head(last_tap,rgb)`. Verify tap normalization/state keys and parent
   output parity independently. Adapt APIs without changing the underlying heads.
5. One optimizer over unique selected shared encoder+heads; depth and KP TRAIN
   microbatches accumulate before one step. Keep12000 depth exposures, flat1e-4,
   protected ramp and all masked-bias exclusions. Audit both encoder gradients.
6. Preserve raw/EMA/both heads/Adam/RNG/samplers/source identities with provided
   checkpoint helpers or demonstrate equivalent cloud state. EMA-only is not a
   complete resume. Do not save a partially failed optimizer update.

Without reliable local depth, larger small-part weights cannot invent toe/tail
geometry; report support fractions, cap saturation and difficult retained cases.

## Commands and explicit cloud adapter

```bash
python -m unittest discover -s tests -p 'test_part_balanced_depth.py' -v
python -m unittest discover -s tests -p 'test_shared_patch_joint_training.py' -v
python tools/preflight_part_balanced_joint_20261028.py --check-design
```

Add a binding in the cloud source, e.g. `tools/cloud_joint_binding.py`:

```
build(config, arm, seed, device) -> {
  model: SharedPatchTwoHeads with independently checked current heads,
  optimizer: precisely the selected trainable parameters, unique,
  depth_batch: {images: ..., split: "train", ...original fields...},
  keypoint_batch: {images: ..., split: "train", ...known/domain fields...},
  depth_loss: callable(head_output,batch) -> parent objective + OFF/ON focus,
  keypoint_loss: callable(head_output,batch) -> original masked KP scalar,
  reference_depth: independent original parent forward callable(rgb),
  reference_keypoints: independent original parent forward callable(rgb),
  identity: {code_sha256,data_sha256,release_sha256,heads_sha256,recipe_sha256}
}
```

Reference callables must NOT call the new wrapper under test. Then:

```bash
python tools/preflight_part_balanced_joint_20261028.py \
  --factory tools.cloud_joint_binding:build --arm J0 --seed 17 --device cuda \
  --output output/part_balanced_joint_20261028/preflight_J0_seed17.json
# Repeat J1 on the same frozen TRAIN support and fresh identical initialization.
```

The CLI checks parent parity, both task encoder gradients and one discarded
optimizer update. It writes no endpoint and dispatches no Slurm job. Formal12k
scheduling remains in the existing cloud finite trainer. Checkpoint helpers
require five actual64-character hashes and both complete sampler states.
`execution_enabled=false` means publication alone starts nothing.

## Evidence requested back

- Source commit, actual baseline/DONE/endpoint identities, unchanged DEV/sealed
  separation including hull items and ROI joins.
- TRAIN ROI support by source/view/part; visibility/transforms, depth-valid
  fractions, cap saturation, task norms and independent head-output parity.
- Fixed EMA D0/D1/J0/J1, per-part median/P90/mm/AbsRel with complete denominators,
  whole-body raw/aligned gauges and retention; paired seeds17/29/43 for a winner.
- Joint Pano bottom-paw PCK>=97.5 AND own DEV bottom aligned AbsRel<=.016, plus
  original domain/view keypoint and synthetic20% retention guards.
- Saved/reloaded endpoints, cost, pointcloud review and explicit limitations.

Do not train on the205 own gauge-only moments. Pano S03/S11 and own AZR2L2/FR2/FR3
stay DEV. S04/S05, GR2/UR2/UR3, Water Reaching, DANNCE2, eight sealed synthetic
rigs and protected TEST stay excluded. No old automation/renderer is restarted.
