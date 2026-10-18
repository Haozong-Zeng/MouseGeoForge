# MouseGeoForge

## Current experiment route (2026-10-18)

The [v3 scale diagnosis and staged experiment plan](docs/bottom_geometry_v3_reconciled_plan_20261018.md)
now defines the next bottom-view geometry comparison. Zero-weight-update tests
on all71 real validation groups separate fixed training-fitted scale from
validation-label oracle alignment. Scale explains part of the v2 gains, while
head drift, quaternion conditioning and limited rig coverage remain concerns.
The completed v2 run failed unseen-rig pose/retention gates; no model is promoted.

Next: freeze scale/evaluation rules and numerical safeguards; compare baseline,
native-GT and shared-scale objectives, then retention; finally run matched
frozen/last-two-block encoder pairs with seeds17/29/43. Planned synthetic rig
counts are48/12/8 for training/development/sealed confirmation, not generated data.
The [full admitted training-pool and H100 strategy](docs/data_scale_and_current_strategy_20261018.md)
remains in force; local1500/4000-update pilots are not full-data convergence budgets.
A frozen released-encoder external2D baseline remains an independent route.
Adapted geometry-to-2D/3D promotion stays gated. No new training, rendering or
automatic next arm has started. See [scale evidence](docs/experiments/bottom_geometry_zero_train_scale_20261018.json)
and the [frozen v2 results](docs/experiments/bottom_geometry_v2_final_20261018.json).

## Current research and distribution status (2026-10-06)

The latest [local-evidence and instance-grouping experiments](docs/local_evidence_instance_grouping_plan_20261005.md)
remain exploratory. Fixed development-set evaluation did not establish a robust
overall winner: duplicate detections, instance/foot assignment and cross-source
generalization remain unresolved. The experiment automation was stopped by the
user; no new checkpoint was promoted to the default model.

This repository is source-only. Data, model weights, rendered assets, virtual
environments and experiment outputs must remain outside Git. Do not use
`git add -f` to bypass the exclusions. Original annotations and final Blender
projects remain in `dataset/` and `BlenderPyMouseProjects/` locally.

Local distribution preparation (no commit, push or upload):

```bash
python tools/prepare_source_distribution.py --output backup/release-preflight --snapshot
python tools/package_local_assets.py plan --output backup/asset-packages --part-gib 16
python tools/package_local_assets.py run --output backup/asset-packages --workers 2
```

The asset packer retains originals and writes independently extractable
`.tar.gz` parts, per-file SHA-256 manifests, archive checksums, and progress JSON.
The 16 GiB target is **uncompressed source size**, not an upload-service limit;
a single larger file is kept intact. It excludes existing `backup/`, environments
and general third-party source trees; selected pretrained weights are included.
Existing DeepLabCut download ZIPs are inventoried without redundant recompression.
To restore, extract all parts for a dataset into a separate empty directory,
verify the manifests, then move the restored tree as needed. Never extract over
live data without reviewing conflicts. Resume an interrupted job by rerunning
the same `run` command; source changes fail verification rather than silently
changing the snapshot. Do not modify datasets while packaging is active.

On 2026-10-06, 92 untracked historical Blender appearance probes/review scripts
with no references in Git-eligible source/docs, plus old `tmp/` and `Log/`, were
moved reversibly to `backup/workspace_cleanup_20261006/`. The exact move mapping
is stored there as `move_plan.json`. Reusable conversion/training code, experiment
records, effective annotations, weights and final Blender scenes were retained.
Historical probes can be restored to their recorded paths if needed. Downloaded
projects and environments were not relocated because local workflows reference
them. No source, dataset or checkpoint was deleted.

Review third-party licenses and dataset redistribution rights before sharing
the asset packages; local packaging does not imply permission to publish them.
See [third-party notices](THIRD_PARTY_NOTICES.md). Before a GitHub push, review the
diff and untracked source list, run relevant tests and inspect the preflight
report. The heuristic secret scan is not an exhaustive security audit.

## Depth–pose coupling pilot

The controlled spatial J-off/J-on pilot and its safeguards are documented in
[the experiment protocol](docs/depth_pose_pilot_protocol.md). It uses complete
synthetic labels, source-group-held-out validation, paired seeds, and frozen RGB
feature caching. It is not yet the production temporal/partial-label trainer.
The [300/1000-step results](docs/depth_pose_pilot_results.md) do not demonstrate a
stable joint benefit; coupling stays off by default while pose localization is investigated.
The [8-scene pose diagnostic](docs/pose_overfit_diagnostic.md) identifies local
heatmap supervision as a major bottleneck in this setting. Disabling that auxiliary
loss passes the training-only overfit gate. The corrected 5000-step baseline has
completed and passed checkpoint reload verification: train/held-out NME is
0.064/0.178, with a substantial generalization gap and weaker dual-mouse fitting.
It is not production-ready or a new paired coupling ablation.
The [resolution-only 512 control](docs/pose_resolution_control.md) is complete:
held-out single-mouse NME improved (0.147 to 0.110), but dual-mouse NME barely
changed (0.209 to 0.206) and detection-aware PCK declined. Higher resolution alone
did not resolve dual-mouse failures within the fixed 5000-update budget.

The [paired convergence control](docs/pose_convergence_control.md) is complete:
15000 updates with hold/cosine LR reach 97% dual-mouse training PCK, but held-out
dual-mouse PCK is only 21.40% / 9.31% at 256 / 512. Training poses are now fitted;
generalization and instance localization, rather than training fit alone, remain
the next priority. No additional training or architecture intervention was launched.

A [diverse-data control](docs/pose_diverse_data_control.md) expands training from
64 to 400 scenes across four complete-Core23 Blender datasets, with unchanged
32-scene validation and the same 512/15000-step optimizer protocol. Real partial
labels and incompatible finger landmarks are not silently treated as Core23.
The run and independent checkpoint reload are complete: held-out dual-mouse NME
improved from 0.286 to 0.102 and detection-aware PCK from 9.31% to 58.02%.
Single-mouse PCK changed only slightly (40.63% to 41.03%); limb contraction and
depth-background errors remain. This is one synthetic-only seed, not a coupling
ablation or evidence of real-animal generalization.

A [768 resolution-only follow-up](docs/pose_diverse_resolution_768_control.md)
keeps the same diverse400 data, seed, update order, losses and 15000-step schedule.
It recomputes RGB features from source images; no 512 checkpoint continuation.
The run, independent reload and fixed-frame visual review are complete: 768 improves
single-mouse PCK (41.03% to 49.73%) and dense depth/masks, but worsens dual-mouse
NME (0.102 to 0.122) and PCK (58.02% to 53.60%). Limb contraction and tail errors
remain. One seed is not a universal resolution ranking. The user has provisionally
selected 768 for subsequent development; retain 512 as the paired reference.
This choice does not assert 768 superiority or launch additional training.

```bash
python tools/run_depth_pose_ablation.py --prepare-only
python tools/run_depth_pose_ablation.py
python tools/analyze_depth_pose_ablation.py
python tools/verify_depth_pose_ablation.py
```

The audited Blender subsets require `pfm_flip_y=True`; do not apply that setting
blindly to real datasets. Outputs/checkpoints remain local under `output/`.

Research workspace for synthetic mouse data generation, real multiview dataset
conversion, and instance-centric 2D/3D mouse pose estimation with VGGT-Omega.

## Repository scope

This is a **source-only research snapshot**, not a distribution of datasets,
trained models, Blender assets, or an already integrated temporal training system.

| Directory | Purpose |
| --- | --- |
| `vggt-omega/` | Upstream model source plus local multi-instance heads, losses, loaders and training tools |
| `mousegeoforge/`, `tests/` | New composition models, shared adapters, optional depth-pose bridge and tests |
| `BlenderScripts/` | Blender scene construction, animation, rendering and annotation tools |
| `tools/`, `scripts/` | Dataset conversion, validation and workflow utilities |
| `configs/` | Workflow configuration and proposed experiment specifications |
| `docs/` | Data contracts, implementation status and research designs |
| `gui_based_vggt_o/` | Local calibration/inference GUI source |

Local-only directories include `dataset/`, `BlenderPyMouseProjects/`, `backup/`,
`output/`, environments, downloaded third-party projects and checkpoints. They
are intentionally excluded from Git. Some historical scripts/configurations use
machine-specific paths; inspect their arguments before running them. No script
should be assumed portable or safe to run as a batch without reviewing its scope.

## Starting points

- [Multi-instance architecture and implementation status](docs/multi_mouse_instance_pose_architecture.md)
- [Temporal architecture and controlled ablations](docs/temporal_architecture_ablation_design.md)
- [Depth-pose research and current implementation](docs/depth_pose_joint_research_and_plan.md)
- [Occlusion-aware labels, robust fusion and 768 calibration audit](docs/occlusion_aware_pose_v1.md)
- [Held-out uncertainty calibration and Blender evaluation diagnostics](docs/occlusion_aware_pose_v2.md)
- [Decoupled pixel-space robust fusion: implementation and controlled evaluation](docs/occlusion_aware_pose_v3.md)
- [Two/four-view context and geometry controls, with camera-label audit](docs/occlusion_aware_pose_v4.md)
- [Unified 1–N view input: interfaces, sampling, scale contract and training plan](docs/variable_view_pose_design.md)
- [Bounded automatic variable-view adaptation run and recovery ledger](docs/variable_view_auto_run.md)
- [Variable-view adaptation results: gains, retention failures and visual audit](docs/variable_view_adaptation_v1_results.md)
- [Variable-view replay V2: single-mouse and fixed-pair retention control](docs/variable_view_replay_v2.md)
- [Variable-view V3: matched-budget lower-learning-rate replay control](docs/variable_view_lr_v3.md)
- [Variable-view V4: final fresh-seed replication and stopping criteria](docs/variable_view_seed_v4.md)
- [Variable-view final summary: replication limits and tail-specific next direction](docs/variable_view_final_summary.md)
- [Tail point-versus-curve supervision: matched-budget pilot and error audit](docs/tail_supervision_v1.md)
- [Cylinder bottom-view audit: missing validation coverage and frozen synthetic controls](docs/cylinder_bottom_view_audit_20260910.md)
- [Cylinder physical-view balance V1: matched-budget training and calibrated validation](docs/cylinder_view_balance_v1.md)
- [Cylinder pose-head-only isolation: frozen depth and camera-combination coverage audit](docs/cylinder_pose_only_v1.md)
- [Cylinder combination coverage V2: decoupled pose, camera subsets and reference order](docs/cylinder_combinations_v2.md)
- [Cylinder localization audit: coarse-window coverage, local heatmaps and frozen temperature diagnostics](docs/cylinder_localization_audit_v1.md)
- [Cylinder coarse-body auxiliary control](docs/cylinder_coarse_aux_v1.md) and [autonomous cycle ledger](docs/cylinder_auto_cycle.md)
- [Cylinder local-grid gradient audit: indirect opposition versus actual direction reversal](docs/cylinder_grid_gradient_audit_v1.md)
- [Cylinder matched-camera train/development gap](docs/cylinder_pose_gap_v1.md) and [train-source mixture: 14/17 gates with retention/depth tradeoffs](docs/cylinder_source_mix_v1.md)
- [Cylinder retention and geometry decomposition: box failures, tail drift and common-support 3D](docs/cylinder_retention_geometry_v1.md)
- [Cylinder body epipolar supervision: fixed-budget preflight and control](docs/cylinder_epipolar_v1.md)
- [Cylinder dual replay retention preflight: teacher quality gates and rejected training proposal](docs/cylinder_replay_retention_v1.md)
- [Cylinder replay coverage audit: same-source nonreplayed poses and body localization regression](docs/cylinder_replay_coverage_v1.md)
- [Cylinder task-gradient audit](docs/cylinder_task_gradients_v1.md) and [conditional local-temperature control](docs/cylinder_adaptive_temperature_v1.md)
- [Five-hour autonomous cycle log](docs/cylinder_five_hour_cycle_20260911.md)
- [Derived repository organization decision](docs/derived_repository_structure_research.md)
- [DANNCE 25 fps mask/depth delivery](docs/dannce_25fps_full_pseudo_depth.md)
- [Temporal indexing and source groups](docs/temporal_dataset_delivery.md)
- [Alternative research routes](docs/multi_mouse_experiment_routes_backlog.md)
- [Blender script organization](BlenderScripts/README.md)
- [Repository maintenance](docs/repository_maintenance.md)

## Environment

Use Python 3.10+ and install a PyTorch/CUDA build appropriate for your GPU first.
The existing local environment is **not** included. A starting point for the
model and basic geometry tooling is:

```bash
python -m venv .venv
source .venv/bin/activate
# Install the appropriate torch + torchvision build for your hardware.
pip install -e ./vggt-omega
pip install -e .
pip install scipy opencv-python matplotlib tqdm
PYTHONPATH=vggt-omega python -m unittest discover -s vggt-omega/training -p test_temporal_window_index.py
python tools/validate_repository_snapshot.py
```

Dataset preparation needs separately acquired data and, for some workflows,
Blender, SAM2, FFmpeg or other dependencies. There is no universal one-command
installation for every historical tool. Download checkpoints from their original
providers and follow their terms. Do not add datasets or model weights to Git.

## Current temporal status

Reliable Blender animations have `temporal_index.jsonl`; DANNCE sessions have
`frames.jsonl` and a working dedicated temporal loader. The source-group/window
index is implemented. The default Blender training entrypoint still consumes
independent multiview poses. The temporal adapter and experiment matrix in the
design document are **proposed**, not implemented or benchmarked.

## Depth-pose integration status

`mousegeoforge.models.DepthPoseModel` is implemented with a shared trainable token
adapter and optional `local_depth_features` coupling. Both task heads remain active
when coupling is `off`. The legacy model defaults are unchanged. The wrapper is
frame-level `[B,V,3,H,W]`; it rejects a time axis rather than silently mixing time
into views. It does not yet implement temporal attention or surface-to-joint offsets.

```python
from vggt_omega.models import VGGTOmega
from mousegeoforge.models import DepthPoseModel

spatial = VGGTOmega(enable_camera=False, enable_depth=True,
                    enable_multi_instance_pose=True, num_keypoints=23)
# Load and verify your upstream encoder/depth checkpoint before wrapping.
model = DepthPoseModel.from_spatial_model(spatial, coupling="off")
# Switch to coupling="local_depth_features" for explicit depth-to-pose fusion.
```

The main planned experiment is temporal × explicit depth-pose coupling × tail
curve supervision (8 variants). See `configs/experiments/temporal_depth_tail_factorial_v1.json`.
This matrix is a specification, not an executable full trainer. The older temporal
module/loss matrix remains a secondary diagnostic. Two-step real-asset GPU smoke
checks verify integration only; no comparative accuracy or generalization claim
has been established.

## Licenses

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). VGGT-Omega material retains
its upstream FAIR Noncommercial Research License in `vggt-omega/LICENSE`.
This repository does not relicense upstream material or redistribute source
datasets. No blanket MIT/Apache license is asserted for this mixed-source snapshot.
