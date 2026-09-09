# DeepLabCut audit for style-transferred synthetic views

This workflow evaluates whether A-to-B style transfer makes Blender images easier for
the existing real-data DeepLabCut models without silently changing pose.

## Imported projects

The Google Drive ZIP files merge into:

- `DeepLabCut/wr_cama-tf-2024-07-13`
- `DeepLabCut/wr_camb-tf-2024-07-11`
- `DeepLabCut/wr_camc-tf-2024-07-12`

The original project configs are preserved. Portable PyTorch configs are generated as
`config.local-pytorch.yaml` inside all three project roots. CamC was recovered from
2,475 labeled frames in 148 usable trial directories and fine-tuned from the CamB
HRNet-W18 checkpoint. Its selected checkpoint is epoch 15:

`DeepLabCut/wr_camc-tf-2024-07-12/dlc-models-pytorch/iteration-0/wr_camcJul12-trainset90shuffle1/train/snapshot-best-015.pt`

The imported CamC `config.yaml` contains stale CamA/B video paths. Exact image hashing
and visual checks confirmed that the files in its `labeled-data` directory are
independent CamC side-view images. Use `config.local-pytorch.yaml`; do not repair the
source config in place.

## Semantic mapping

The DLC labels and Blender keypoints share 27 scored landmarks:

- DLC digit 1 -> Blender Pinky
- DLC digit 2 -> Blender Ring
- DLC digit 3 -> Blender Middle
- DLC digit 4 -> Blender Index
- DLC `tip`, `mid`, `joint` -> Blender finger bones 3, 2, 1
- DLC `r_mid` / `l_mid` -> Blender `Fingers1_R` / `Fingers1_L`
- DLC `snout` -> Blender `HeadEnd_M`

DLC `mouth_tip` and `spout_tip` have no Blender counterpart. Blender wrists have no
DLC counterpart. Only Blender keypoints marked visible by the depth test are scored.

## Environment

```bash
bash tools/setup_dlc_mousegeoforge_env.sh
python3 tools/prepare_dlc_local_configs.py --dlc-root DeepLabCut
```

## Prepare raw, styled, and GT-matched images

Run this with the CycleGAN-Turbo environment:

```bash
img2img-turbo/.venv-mousegeoforge/bin/python \
  tools/prepare_dlc_style_transfer_audit.py \
  --dataset dataset/BlenderPyMouse/PyMouseStudio-BrowmFur-WaterReaching-FrameToFrame-FGcamB \
  --checkpoint img2img-turbo/output/cyclegan_turbo/mousegeoforge_curated_v1/selected/model_best_step9000.pkl \
  --output output/dlc_style_transfer_audit \
  --sample-count 12 \
  --views 0 1 2 \
  --device cuda:0 \
  --use-fp16 \
  --overwrite
```

The translated image is resized back to the original 1200 x 800 size, so Blender GT
pixel coordinates remain in the same image coordinate system.

## Run DeepLabCut and score

```bash
DeepLabCut/.venv-mousegeoforge/bin/python \
  tools/run_dlc_style_transfer_audit.py \
  --manifest output/dlc_style_transfer_audit/manifest.csv \
  --output output/dlc_style_transfer_audit_results \
  --config A=DeepLabCut/wr_cama-tf-2024-07-13/config.local-pytorch.yaml \
  --config B=DeepLabCut/wr_camb-tf-2024-07-11/config.local-pytorch.yaml \
  --config C=DeepLabCut/wr_camc-tf-2024-07-12/config.local-pytorch.yaml \
  --device cuda:0 \
  --pcutoff 0.1 \
  --seed 42 \
  --overwrite
```

Use `--snapshot C=2` to explicitly evaluate the regular epoch-20 checkpoint. The
default `C=-1` resolves to the selected `snapshot-best-015.pt`.

## Outputs

- `summary.csv`: per-camera raw versus styled coverage, likelihood, pixel error, and PCK
- `per_keypoint_metrics.csv`: one row per frame, condition, and shared keypoint
- `report.json`: configs, mapping, camera readiness, and summary metrics
- `triangulation_metrics.csv`: weighted-DLT 3D errors for AB, AC, BC, and ABC
- `triangulation_summary.csv`: 3D/PCK/reprojection summaries per view combination
- `outliers_2d.csv` / `outliers_3d.csv`: high-error cases for semantic inspection
- `contact_sheets/`: raw, styled, and GT-versus-DLC overlays for visual inspection
- `predictions/`: native DeepLabCut H5/CSV results and labeled images

In overlays, green dots are Blender GT, red crosses are DLC predictions, and yellow
segments show their displacement.

## Six-frame pilot

The deterministic six-frame pilot used frames 1, 51, 101, 150, 200, and 250.

| View | Condition | Median error | PCK@20 | PCK@40 | Mean likelihood |
| --- | --- | ---: | ---: | ---: | ---: |
| CamA | raw | 29.94 px | 33.1% | 58.4% | 0.563 |
| CamA | styled | 20.90 px | 44.8% | 60.4% | 0.542 |
| CamB | raw | 19.78 px | 50.6% | 73.5% | 0.288 |
| CamB | styled | 12.91 px | 72.8% | 86.4% | 0.340 |

CamB improves clearly after style transfer. CamA improves in median error and PCK but
still has catastrophic outliers where hand landmarks are assigned near the snout; its
mean error and RMSE therefore worsen. Report medians, PCK, and outlier overlays together
instead of selecting a style model by mean error alone.

## Twenty-four-frame three-view audit

The expanded audit uniformly samples 24 animation frames (72 camera images). With the
selected CamC epoch-15 checkpoint:

| View | Raw median | Styled median | Raw PCK@20 | Styled PCK@20 |
| --- | ---: | ---: | ---: | ---: |
| CamA | 36.42 px | 25.37 px | 29.8% | 39.9% |
| CamB | 18.92 px | 12.05 px | 51.3% | 69.3% |
| CamC | 15.65 px | 13.62 px | 56.9% | 64.7% |

For BC triangulation, style transfer improves median 3D error from 1.34 to 0.95 Blender
world units, PCK@2 from 69.1% to 76.1%, and median reprojection RMSE from 11.55 to
6.37 px. ABC also improves strongly, but CamA remains the limiting view because a
small number of high-confidence finger detections jump to the snout. BC is therefore
the most reliable default pair until CamA is retrained or a camera-reliability/outlier
filter is added.
