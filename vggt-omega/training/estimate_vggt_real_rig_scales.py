"""Estimate one fixed VGGT scene-unit scale per real camera rig trial."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.finetune_blendedmvs import (  # noqa: E402
    RealPoseDataset,
    discover_real_trials,
    load_checkpoint_chain,
)
from training.geometry_coupled_losses import (  # noqa: E402
    median_pairwise_camera_baseline,
)
from vggt_omega.models.vggt_omega import VGGTOmega  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", default="A,B,C")
    parser.add_argument("--image-width", type=int, default=624)
    parser.add_argument("--image-height", type=int, default=416)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=40)
    parser.add_argument("--frame-count", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mad-multiplier", type=float, default=3.5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def robust_values(values: list[float], mad_multiplier: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (array > 1e-6)]
    if array.size < 3:
        return array
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    threshold = mad_multiplier * max(1.4826 * mad, 1e-6)
    kept = array[np.abs(array - median) <= threshold]
    return kept if kept.size >= 3 else array


def main() -> None:
    args = parse_args()
    cameras = [item.strip() for item in args.cameras.split(",") if item.strip()]
    roots = sorted(
        path
        for path in args.root.iterdir()
        if path.is_dir()
        and (path / "calibration" / "calibration.toml").is_file()
        and all(
            (path / "videos-raw" / f"pawfilt_cam{camera}.mp4").is_file()
            for camera in cameras
        )
    )
    trials = discover_real_trials(",".join(str(path) for path in roots), cameras)
    dataset = RealPoseDataset(
        trials=trials,
        cameras=cameras,
        image_width=args.image_width,
        image_height=args.image_height,
        frames_text="",
        frame_start=args.frame_start,
        frame_step=args.frame_step,
        frame_count=args.frame_count,
        canonicalize_camera=True,
        scale_mode="median_baseline",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = VGGTOmega().to(device)
    load_checkpoint_chain(model, args.checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    predicted_by_trial: dict[str, list[float]] = defaultdict(list)
    physical_by_trial: dict[str, list[float]] = defaultdict(list)
    first_rotation_by_trial: dict[str, list[float]] = defaultdict(list)
    first_translation_by_trial: dict[str, list[float]] = defaultdict(list)
    identity = torch.eye(3, device=device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=images.is_cuda,
            ):
                predictions = model(images)
            extrinsics, _ = encoding_to_camera(
                predictions["pose_enc"].float(),
                images.shape[-2:],
                build_intrinsics=False,
            )
            predicted_baseline = median_pairwise_camera_baseline(extrinsics)[0]
            first_rotation_error = torch.linalg.matrix_norm(
                extrinsics[0, 0, :3, :3] - identity
            )
            first_translation_error = torch.linalg.vector_norm(
                extrinsics[0, 0, :3, 3]
            )
            trial = str(batch["scene"][0])
            predicted_by_trial[trial].append(float(predicted_baseline.cpu()))
            physical_by_trial[trial].append(float(batch["scale"][0]))
            first_rotation_by_trial[trial].append(float(first_rotation_error.cpu()))
            first_translation_by_trial[trial].append(float(first_translation_error.cpu()))

    rows = []
    for trial in sorted(predicted_by_trial):
        robust = robust_values(
            predicted_by_trial[trial],
            args.mad_multiplier,
        )
        predicted_baseline = float(np.median(robust))
        physical_baseline = float(np.median(physical_by_trial[trial]))
        rows.append(
            {
                "trial": trial,
                "frames": len(predicted_by_trial[trial]),
                "inlier_frames": int(robust.size),
                "predicted_normalized_baseline": predicted_baseline,
                "physical_baseline": physical_baseline,
                "target_scale": physical_baseline / predicted_baseline,
                "first_rotation_error_median": float(
                    np.median(first_rotation_by_trial[trial])
                ),
                "first_translation_error_median": float(
                    np.median(first_translation_by_trial[trial])
                ),
            }
        )

    ratios = np.asarray(
        [row["predicted_normalized_baseline"] for row in rows],
        dtype=np.float64,
    )
    payload = {
        "method": (
            "Per-trial fixed scale from the robust median ABC baseline predicted "
            "by the frozen original VGGT-Omega checkpoint."
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "image_shape": [args.image_height, args.image_width],
        "summary": {
            "trials": len(rows),
            "frames": int(sum(row["frames"] for row in rows)),
            "predicted_baseline_min": float(ratios.min()),
            "predicted_baseline_median": float(np.median(ratios)),
            "predicted_baseline_max": float(ratios.max()),
            "predicted_baseline_cv": float(ratios.std() / ratios.mean()),
        },
        "trials": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
