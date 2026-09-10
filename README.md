# MouseGeoForge

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
Results are pending independent reload and fixed-frame visual acceptance.

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
