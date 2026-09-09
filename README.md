# MouseGeoForge

Research workspace for synthetic mouse data generation, real multiview dataset
conversion, and instance-centric 2D/3D mouse pose estimation with VGGT-Omega.

## Repository scope

This is a **source-only research snapshot**, not a distribution of datasets,
trained models, Blender assets, or an already integrated temporal training system.

| Directory | Purpose |
| --- | --- |
| `vggt-omega/` | Upstream model source plus local multi-instance heads, losses, loaders and training tools |
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
pip install scipy opencv-python matplotlib tqdm
PYTHONPATH=vggt-omega python -m unittest discover -s vggt-omega/training -p test_temporal_window_index.py
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

## Licenses

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). VGGT-Omega material retains
its upstream FAIR Noncommercial Research License in `vggt-omega/LICENSE`.
This repository does not relicense upstream material or redistribute source
datasets. No blanket MIT/Apache license is asserted for this mixed-source snapshot.
