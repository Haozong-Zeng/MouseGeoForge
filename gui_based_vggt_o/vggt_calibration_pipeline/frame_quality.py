#!/usr/bin/env python3
"""Per-frame quality scoring using gradient-based attribution.

Computes d(pose_enc)/d(patch_embed_output) for each frame set and
derives a scalar quality score: the fraction of pose-gradient mass
that falls on bright, structured image patches (not dark/featureless ones).

Why this matters
----------------
High combined_score → the model anchored its pose prediction on visible,
textured content rather than dark borders or noise → likely a more reliable
calibration frame.

Metrics saved per frame set:
  quality_score   fraction of gradient mass on bright patches (> threshold)
  mean_grad       mean gradient magnitude (overall model sensitivity to the frame)
  combined_score  quality_score * log(mean_grad + 1)  — main ranking signal
  cam_X_quality   per-camera quality scores

Usage (standalone):
    python vggt_calibration_pipeline/frame_quality.py \\
        --frames-dir vggt_calibration_pipeline/runs/sample_videos/frames \\
        --checkpoint checkpoints/vggt_omega_1b_512.pt \\
        --camera-names A B C \\
        --output vggt_calibration_pipeline/runs/sample_videos/quality_scores.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from vggt_calibration_pipeline.pipeline_utils import load_model  # noqa: E402
from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: E402

BRIGHTNESS_THRESHOLD = 15.0  # out of 255; patches darker than this are treated as featureless


def _brightness_mask(images_tensor: torch.Tensor, patch_h: int, patch_w: int) -> np.ndarray:
    """Float32 [F, patch_h*patch_w] mask: 1 if patch mean brightness >= threshold, else 0.

    images_tensor: [F, 3, H, W], values in [0, 1] (TF.ToTensor convention).
    """
    imgs = images_tensor.detach().float().cpu().numpy()  # [F, 3, H, W]
    imgs = np.transpose(imgs, (0, 2, 3, 1)) * 255.0      # [F, H, W, 3] in [0, 255]
    F, H, W, _ = imgs.shape
    ph = H // patch_h
    pw = W // patch_w
    gray = imgs.mean(axis=3)  # [F, H, W]
    blocks = gray[:, : patch_h * ph, : patch_w * pw].reshape(F, patch_h, ph, patch_w, pw)
    patch_mean = blocks.mean(axis=(2, 4))  # [F, patch_h, patch_w]
    return (patch_mean >= BRIGHTNESS_THRESHOLD).astype(np.float32).reshape(F, patch_h * patch_w)


def compute_quality_score(
    model: torch.nn.Module,
    images_tensor: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    """Gradient quality score for one frame set (multiple camera views).

    images_tensor: [F, 3, H, W] on device — one set of camera views.

    Returns dict with keys:
        quality_score      float in [0, 1]
        mean_grad          float >= 0
        combined_score     float >= 0  (main ranking signal)
        per_camera_quality list[float], one entry per camera
    """
    F, _, H, W = images_tensor.shape
    patch_h, patch_w = H // 16, W // 16

    pe_store: dict = {}

    def _hook(module, inp, output):
        t = output["x_norm_patchtokens"] if isinstance(output, dict) else output
        t.retain_grad()
        pe_store["out"] = t

    handle = model.aggregator.patch_embed.register_forward_hook(_hook)

    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()

    try:
        with amp_ctx, torch.enable_grad():
            preds = model(images_tensor)
            pose_enc = preds["pose_enc"]
            img_hw = preds["images"].shape[-2:]
            ext, intr = encoding_to_camera(pose_enc, img_hw)
            scalar = ext.float().pow(2).sum() + intr.float().pow(2).sum()
            scalar.backward()
    finally:
        handle.remove()

    out_tensor = pe_store.get("out")
    if out_tensor is None or out_tensor.grad is None:
        del pe_store
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"quality_score": 0.0, "mean_grad": 0.0, "combined_score": 0.0, "per_camera_quality": [0.0] * F}

    importance = out_tensor.grad.detach().float().norm(dim=-1).cpu().numpy()  # [F, P]
    brightness = _brightness_mask(images_tensor, patch_h, patch_w)            # [F, P]

    per_cam: list[float] = []
    for f in range(F):
        total = float(importance[f].sum()) + 1e-12
        bright = float((importance[f] * brightness[f]).sum())
        per_cam.append(bright / total)

    quality_score = float(np.mean(per_cam))
    mean_grad = float(importance.mean())
    combined_score = quality_score * math.log(mean_grad + 1.0)

    del pe_store, preds, scalar
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "quality_score": quality_score,
        "mean_grad": mean_grad,
        "combined_score": combined_score,
        "per_camera_quality": per_cam,
    }


def score_frame_directory(
    frames_dir: Path,
    frame_indices: Sequence[int],
    camera_names: Sequence[str],
    model: torch.nn.Module,
    device: torch.device,
    model_resolution: int = 512,
    preprocess_mode: str = "balanced",
    verbose: bool = True,
) -> list[dict[str, object]]:
    """Score every frame set in frames_dir; return list of per-frame records."""
    records: list[dict[str, object]] = []
    total = len(frame_indices)

    for i, frame_idx in enumerate(frame_indices):
        frame_dir = frames_dir / f"frame_{frame_idx:06d}"
        image_paths = [str(frame_dir / f"p{name}.png") for name in camera_names]
        missing = [p for p in image_paths if not Path(p).is_file()]
        if missing:
            if verbose:
                print(f"  frame {frame_idx}: SKIP — missing {missing}", flush=True)
            continue

        images_tensor = load_and_preprocess_images(
            image_paths, mode=preprocess_mode, image_resolution=model_resolution
        ).to(device)

        scores = compute_quality_score(model, images_tensor, device)

        record: dict[str, object] = {
            "frame": frame_idx,
            "quality_score": scores["quality_score"],
            "mean_grad": scores["mean_grad"],
            "combined_score": scores["combined_score"],
        }
        for j, name in enumerate(camera_names):
            per_cam = scores["per_camera_quality"]
            record[f"cam_{name}_quality"] = per_cam[j] if j < len(per_cam) else 0.0

        records.append(record)

        if verbose and (i == 0 or i == total - 1 or (i + 1) % 10 == 0):
            print(
                f"  [{i + 1}/{total}] frame {frame_idx}: "
                f"quality={scores['quality_score']:.3f}  "
                f"mean_grad={scores['mean_grad']:.0f}  "
                f"combined={scores['combined_score']:.3f}",
                flush=True,
            )

    return records


def write_quality_csv(records: list[dict], output_path: Path, camera_names: Sequence[str]) -> None:
    fieldnames = (
        ["frame", "quality_score", "mean_grad", "combined_score"]
        + [f"cam_{n}_quality" for n in camera_names]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def rank_frames(
    records: list[dict],
    camera_names: Sequence[str],
    rank_camera: str | None = None,
    skip_warmup_fraction: float = 0.0,
    top_n: int | None = None,
) -> list[dict]:
    """Sort records by a composite score and return the top_n.

    rank_camera: camera name to use as primary signal (e.g. 'C').
                 If None, uses mean quality_score across all cameras.
    skip_warmup_fraction: drop the first fraction of frames (by frame index)
                          before ranking. Default 0 = keep all.
    top_n: return only this many frames. Default = all remaining after filtering.

    The composite score is:
        primary_quality + 0.10 * (frame / max_frame)

    The 0.10 position weight accounts for the empirical finding that later
    frames in a recording tend to produce better calibrations (animal settled,
    more representative pose). Do not over-tune this weight on one dataset.
    """
    if not records:
        return []

    max_frame = max(r["frame"] for r in records)
    min_warmup_frame = int(max_frame * skip_warmup_fraction)

    def _score(r: dict) -> float:
        if rank_camera and f"cam_{rank_camera}_quality" in r:
            primary = float(r[f"cam_{rank_camera}_quality"])
        else:
            primary = float(r["quality_score"])
        position = r["frame"] / max(max_frame, 1)
        return primary + 0.10 * position

    filtered = [r for r in records if r["frame"] >= min_warmup_frame]
    ranked = sorted(filtered, key=_score, reverse=True)
    return ranked[:top_n] if top_n is not None else ranked


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory containing frame_XXXXXX subdirectories.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera-names", nargs="+", default=["A", "B", "C"])
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV output path. Default: <frames-dir>/../quality_scores.csv",
    )
    parser.add_argument(
        "--frame-indices",
        nargs="*",
        type=int,
        help="Specific frame indices to score. Default: all extracted.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-resolution", type=int, default=512)
    parser.add_argument("--preprocess-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument(
        "--select",
        type=int,
        default=None,
        metavar="N",
        help="After scoring, print the top N frame indices ranked by quality (useful for piping into downstream scripts).",
    )
    parser.add_argument(
        "--rank-camera",
        default=None,
        metavar="NAME",
        help="Camera name to use as the primary ranking signal (e.g. C). Default: average across cameras.",
    )
    parser.add_argument(
        "--skip-warmup",
        type=float,
        default=0.0,
        metavar="FRACTION",
        help="Skip the first FRACTION of frames (by index) before ranking. "
             "E.g. 0.12 skips the first 12%% of the video. Default: 0.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; pass --device cpu")

    output = args.output or (args.frames_dir.parent / "quality_scores.csv")

    if args.frame_indices:
        frame_indices = list(args.frame_indices)
    else:
        frame_dirs = sorted(args.frames_dir.glob("frame_??????"))
        frame_indices = [int(d.name.split("_")[1]) for d in frame_dirs]
        if not frame_indices:
            raise FileNotFoundError(f"No frame_XXXXXX directories in {args.frames_dir}")

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    model.eval()

    print(f"Scoring {len(frame_indices)} frame sets...")
    records = score_frame_directory(
        frames_dir=args.frames_dir,
        frame_indices=frame_indices,
        camera_names=args.camera_names,
        model=model,
        device=device,
        model_resolution=args.model_resolution,
        preprocess_mode=args.preprocess_mode,
    )

    write_quality_csv(records, output, args.camera_names)
    print(f"\nWrote {len(records)} quality scores → {output}")

    if records:
        scores = [r["quality_score"] for r in records]
        best = records[int(np.argmax(scores))]
        worst = records[int(np.argmin(scores))]
        print(f"Best  frame: {best['frame']}  quality={best['quality_score']:.3f}  combined={best['combined_score']:.3f}")
        print(f"Worst frame: {worst['frame']}  quality={worst['quality_score']:.3f}  combined={worst['combined_score']:.3f}")

    if args.select is not None:
        top = rank_frames(
            records,
            camera_names=args.camera_names,
            rank_camera=args.rank_camera,
            skip_warmup_fraction=args.skip_warmup,
            top_n=args.select,
        )
        print(f"\nTop {len(top)} frames (rank_camera={args.rank_camera or 'mean'}, skip_warmup={args.skip_warmup}):")
        for r in top:
            cam_str = ""
            if args.rank_camera:
                cam_str = f"  cam_{args.rank_camera}={r.get(f'cam_{args.rank_camera}_quality', 0):.3f}"
            print(f"  frame {r['frame']:6d}  quality={r['quality_score']:.3f}{cam_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
