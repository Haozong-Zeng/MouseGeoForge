#!/usr/bin/env python3
"""Evaluate BlenderPyMouse VGGT-Omega adapter checkpoints and select the best."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "vggt-omega" / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from finetune_blendedmvs import (  # noqa: E402
    aggregator_is_frozen,
    configure_trainable,
    evaluate,
    evaluate_real_pose,
    load_model,
    make_loader,
    make_real_val_loader,
)


CHECKPOINT_RE = re.compile(r"_epoch(?P<epoch>\d+)_step(?P<step>\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "blenderpymouse_vggt_omega_all" / "run_20260712_134802",
    )
    parser.add_argument("--blendedmvs-root", default=str(PROJECT_ROOT / "dataset" / "BlenderPyMouse"))
    parser.add_argument(
        "--base-checkpoint",
        default=str(PROJECT_ROOT / "vggt-omega" / "vggt_omega_1b_512.pt"),
        help="Original VGGT-Omega checkpoint used only when --include-base is set.",
    )
    parser.add_argument(
        "--real-val-roots",
        default=str(PROJECT_ROOT / "water-reaching-dataset" / "FJ_L2_2024-06-24_1_Trial_7"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-views", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--samples-per-scene", type=int, default=2)
    parser.add_argument("--val-scenes", type=int, default=1)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--real-val-cameras", default="A,B,C")
    parser.add_argument("--real-val-frames", default="")
    parser.add_argument("--real-val-frame-start", type=int, default=0)
    parser.add_argument("--real-val-frame-step", type=int, default=10)
    parser.add_argument("--real-val-frame-count", type=int, default=32)
    parser.add_argument("--real-val-batches", type=int, default=4)
    parser.add_argument("--real-val-batch-size", type=int, default=1)
    parser.add_argument("--real-val-scale-mode", choices=["raw", "median_baseline"], default="median_baseline")
    parser.add_argument("--train-mode", default="light_heads")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--copy-best", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def checkpoint_epoch_step(path: Path) -> tuple[int | None, int | None]:
    match = CHECKPOINT_RE.search(path.name)
    if not match:
        return None, None
    return int(match.group("epoch")), int(match.group("step"))


def build_eval_args(args: argparse.Namespace, checkpoint: str) -> argparse.Namespace:
    return argparse.Namespace(
        blendedmvs_root=args.blendedmvs_root,
        checkpoint=checkpoint,
        adapter_checkpoint="",
        output="",
        checkpoint_dir="",
        log_csv="",
        device=args.device,
        image_size=args.image_size,
        num_views=args.num_views,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=0,
        max_steps=0,
        samples_per_scene=args.samples_per_scene,
        max_scenes=0,
        val_scenes=args.val_scenes,
        val_batches=args.val_batches,
        val_every=0,
        real_val_roots=args.real_val_roots,
        real_val_cameras=args.real_val_cameras,
        real_val_frames=args.real_val_frames,
        real_val_frame_start=args.real_val_frame_start,
        real_val_frame_step=args.real_val_frame_step,
        real_val_frame_count=args.real_val_frame_count,
        real_val_batches=args.real_val_batches,
        real_val_batch_size=args.real_val_batch_size,
        real_val_scale_mode=args.real_val_scale_mode,
        lr=1e-5,
        aggregator_lr_mult=0.1,
        weight_decay=0.05,
        camera_weight=1.0,
        translation_weight=1.0,
        rotation_weight=1.0,
        fov_weight=1.0,
        depth_weight=0.2,
        body_mask_dirs="body_masks,mouse_masks,masks",
        semantic_mask_dirs="semantic_masks,part_masks",
        body_depth_balance=0.0,
        body_depth_fraction=0.6,
        semantic_label_weights="",
        depth_gradient_weight=0.05,
        depth_gradient_scales="1,2,4",
        grad_clip=1.0,
        log_every=1,
        seed=0,
        top_srcs=8,
        train_mode=args.train_mode,
        unfreeze_last_blocks=1,
        train_camera_register_tokens=False,
        canonicalize_camera=True,
        pfm_flip_y=True,
        use_masked_images=False,
        scale_mode="median_depth",
        save_full=False,
        dry_run="none",
    )


def score_checkpoint(
    checkpoint: Path,
    eval_args: argparse.Namespace,
    val_loader,
    real_loader,
    device: torch.device,
) -> dict[str, float | int | str | None]:
    model = load_model(eval_args, device)
    configure_trainable(model, eval_args)
    frozen_aggregator = aggregator_is_frozen(model)
    val_scores = evaluate(model, val_loader, device, eval_args, frozen_aggregator, eval_args.val_batches)
    real_scores = evaluate_real_pose(model, real_loader, device, frozen_aggregator, eval_args.real_val_batches)
    epoch, step = checkpoint_epoch_step(checkpoint)
    row: dict[str, float | int | str | None] = {
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "epoch": epoch,
        "step": step,
    }
    row.update(val_scores)
    row.update(real_scores)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def copy_best(run_dir: Path, rows: list[dict[str, float | int | str | None]], key: str, output_name: str) -> Path | None:
    candidates = [row for row in rows if row.get(key) is not None]
    if not candidates:
        return None
    best = min(candidates, key=lambda row: float(row[key]))  # type: ignore[arg-type]
    src = Path(str(best["checkpoint"]))
    dst = run_dir / output_name
    shutil.copy2(src, dst)
    return dst


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pt"))
    if args.include_base:
        checkpoints = [Path(args.base_checkpoint).resolve(), *checkpoints]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {run_dir / 'checkpoints'}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    loader_args = build_eval_args(args, checkpoint=str(checkpoints[0]))
    val_loader = make_loader(loader_args, "val")
    real_loader = make_real_val_loader(loader_args)

    rows = []
    for idx, checkpoint in enumerate(checkpoints, start=1):
        eval_args = build_eval_args(args, checkpoint=str(checkpoint))
        print(f"[{idx}/{len(checkpoints)}] scoring {checkpoint.name}", flush=True)
        rows.append(score_checkpoint(checkpoint, eval_args, val_loader, real_loader, device))

    output_csv = (args.output_csv or run_dir / "checkpoint_scores.csv").resolve()
    output_json = (args.output_json or run_dir / "checkpoint_scores.json").resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "checkpoint_name",
        "epoch",
        "step",
        "real_val_loss_camera",
        "real_val_loss_cam_t",
        "real_val_loss_cam_q",
        "real_val_loss_cam_fov",
        "val_loss",
        "val_loss_cam_t",
        "val_loss_cam_q",
        "val_loss_cam_fov",
        "val_loss_depth",
        "checkpoint",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best_real = min(rows, key=lambda row: float(row.get("real_val_loss_camera", float("inf"))))
    best_synth = min(rows, key=lambda row: float(row.get("val_loss", float("inf"))))
    payload = {
        "run_dir": str(run_dir),
        "score_count": len(rows),
        "best_real_val": best_real,
        "best_synthetic_val": best_synth,
        "scores": rows,
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    copied = {}
    if args.copy_best:
        copied["best_real_val"] = str(copy_best(run_dir, rows, "real_val_loss_camera", "best_real_val.pt"))
        copied["best_synthetic_val"] = str(copy_best(run_dir, rows, "val_loss", "best_synthetic_val.pt"))
    print(json.dumps({**payload, "copied": copied}, indent=2), flush=True)


if __name__ == "__main__":
    main()
