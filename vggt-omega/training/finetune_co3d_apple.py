#!/usr/bin/env python3
"""Minimal VGGT-Omega fine-tuning entrypoint for raw CO3D apple downloads.

This is intentionally small and explicit: it reads the CO3D category zip layout
directly, converts raw CO3D/PyTorch3D cameras to OpenCV camera-from-world
matrices, and supervises VGGT-Omega camera pose encoding plus masked depth.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding


def read_jgz(path: Path) -> Any:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def read_co3d_depth(path: Path, scale_adjustment: float) -> np.ndarray:
    with Image.open(path) as depth_pil:
        raw = np.array(depth_pil, dtype=np.uint16)
    depth = np.frombuffer(raw.tobytes(), dtype=np.float16).reshape(raw.shape)
    return depth.astype(np.float32) * float(scale_adjustment)


def co3d_camera_to_opencv(viewpoint: dict[str, Any], image_size_wh: tuple[int, int], out_hw: tuple[int, int]):
    width, height = image_size_wh
    out_h, out_w = out_hw
    min_side = float(min(width, height))

    focal = viewpoint["focal_length"]
    pp = viewpoint["principal_point"]
    fx = float(focal[0]) * min_side / 2.0
    fy = float(focal[1]) * min_side / 2.0
    cx = width / 2.0 - float(pp[0]) * min_side / 2.0
    cy = height / 2.0 - float(pp[1]) * min_side / 2.0

    sx = out_w / float(width)
    sy = out_h / float(height)
    intri = np.array(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    r_p3d = np.asarray(viewpoint["R"], dtype=np.float32)
    t_p3d = np.asarray(viewpoint["T"], dtype=np.float32)
    p3d_to_cv = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    # CO3D/PyTorch3D stores R for row-vector right multiplication:
    # X_cam = X_world @ R + T. Convert to column-vector OpenCV first.
    r_cv = p3d_to_cv @ r_p3d.T
    t_cv = p3d_to_cv @ t_p3d
    extri = np.concatenate([r_cv, t_cv[:, None]], axis=1).astype(np.float32)
    return extri, intri


class RawCo3dAppleDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        num_views: int,
        image_size: int,
        min_frames: int = 2,
        seed: int = 0,
    ) -> None:
        self.root = root
        self.category_root = root / "apple"
        self.split = split
        self.num_views = num_views
        self.image_size = image_size
        self.rng = random.Random(seed)

        frames = read_jgz(self.category_root / "frame_annotations.jgz")
        by_path = {f["image"]["path"]: f for f in frames}
        set_lists = json.loads((self.category_root / "set_lists.json").read_text())
        split_keys = ["train_known", "train_unseen"] if split == "train" else ["test_known", "test_unseen"]

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for key in split_keys:
            for seq_name, _frame_no, image_path in set_lists[key]:
                anno = by_path.get(image_path)
                if anno is not None:
                    grouped[seq_name].append(anno)

        self.sequences = []
        for seq_name, seq_frames in grouped.items():
            seq_frames = sorted(seq_frames, key=lambda f: f["frame_number"])
            if len(seq_frames) >= max(min_frames, num_views):
                self.sequences.append((seq_name, seq_frames))
        if not self.sequences:
            raise RuntimeError(f"No CO3D apple sequences found under {root} for split={split}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        seq_name, frames = self.sequences[idx]
        if len(frames) == self.num_views:
            chosen = frames
        else:
            ids = sorted(self.rng.sample(range(len(frames)), self.num_views))
            chosen = [frames[i] for i in ids]

        images, depths, masks, extrinsics, intrinsics = [], [], [], [], []
        out_hw = (self.image_size, self.image_size)
        for anno in chosen:
            image_path = self.root / anno["image"]["path"]
            depth_path = self.root / anno["depth"]["path"]
            mask_path = self.root / anno["depth"]["mask_path"]

            with Image.open(image_path) as img:
                img = img.convert("RGB")
                width, height = img.size
                img_resized = img.resize((self.image_size, self.image_size), Image.BICUBIC)
                image = np.asarray(img_resized, dtype=np.float32) / 255.0

            depth = read_co3d_depth(depth_path, anno["depth"].get("scale_adjustment", 1.0))
            with Image.open(mask_path) as mask_img:
                mask = np.asarray(mask_img.resize((self.image_size, self.image_size), Image.NEAREST)) > 0
            depth = np.asarray(
                Image.fromarray(depth).resize((self.image_size, self.image_size), Image.NEAREST),
                dtype=np.float32,
            ).copy()
            depth[~mask] = 0.0

            extri, intri = co3d_camera_to_opencv(anno["viewpoint"], (width, height), out_hw)
            images.append(torch.from_numpy(image).permute(2, 0, 1))
            depths.append(torch.from_numpy(depth))
            masks.append(torch.from_numpy(mask))
            extrinsics.append(torch.from_numpy(extri))
            intrinsics.append(torch.from_numpy(intri))

        return {
            "seq_name": seq_name,
            "images": torch.stack(images).float(),
            "depths": torch.stack(depths).float(),
            "depth_masks": torch.stack(masks).bool(),
            "extrinsics": torch.stack(extrinsics).float(),
            "intrinsics": torch.stack(intrinsics).float(),
        }


def freeze_by_name(model: torch.nn.Module, patterns: list[str]) -> None:
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad_(False)


def camera_loss(pred_pose: torch.Tensor, target_pose: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_t, target_t = pred_pose[..., :3], target_pose[..., :3]
    pred_q = F.normalize(pred_pose[..., 3:7], dim=-1)
    target_q = F.normalize(target_pose[..., 3:7], dim=-1)
    pred_fov, target_fov = pred_pose[..., 7:], target_pose[..., 7:]

    loss_t = F.smooth_l1_loss(pred_t, target_t)
    loss_q = torch.minimum((pred_q - target_q).abs(), (pred_q + target_q).abs()).mean()
    loss_fov = F.smooth_l1_loss(pred_fov, target_fov)
    return {"loss_cam_t": loss_t, "loss_cam_q": loss_q, "loss_cam_fov": loss_fov}


def depth_loss(pred_depth: torch.Tensor, target_depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if pred_depth.ndim == 5:
        pred_depth = pred_depth[..., 0]
    valid = mask & torch.isfinite(target_depth) & (target_depth > 1e-6)
    if valid.sum() < 16:
        return pred_depth.mean() * 0.0
    pred = torch.log(pred_depth.clamp_min(1e-6))[valid]
    target = torch.log(target_depth.clamp_min(1e-6))[valid]
    return F.smooth_l1_loss(pred, target)


def make_loader(args: argparse.Namespace, split: str) -> DataLoader:
    dataset = RawCo3dAppleDataset(
        root=Path(args.co3d_root),
        split=split,
        num_views=args.num_views,
        image_size=args.image_size,
        seed=args.seed,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    loader = make_loader(args, "train")
    first = next(iter(loader))
    print(
        "dataset_ok",
        f"sequences={len(loader.dataset)}",
        f"images={tuple(first['images'].shape)}",
        f"depths={tuple(first['depths'].shape)}",
        f"valid_depth_px={int(((first['depths'] > 1e-6) & first['depth_masks']).sum())}",
    )
    if args.dry_run == "data":
        return

    model = VGGTOmega().to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        print(f"loaded_checkpoint={args.checkpoint}")
    if args.freeze_aggregator:
        freeze_by_name(model, ["aggregator"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable_params={trainable:,}")

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            depths = batch["depths"].to(device, non_blocking=True)
            masks = batch["depth_masks"].to(device, non_blocking=True)
            extrinsics = batch["extrinsics"].to(device, non_blocking=True)
            intrinsics = batch["intrinsics"].to(device, non_blocking=True)

            target_pose = extri_intri_to_pose_encoding(extrinsics, intrinsics, images.shape[-2:])
            pred = model(images)
            cam_losses = camera_loss(pred["pose_enc"], target_pose)
            loss_depth = depth_loss(pred["depth"], depths, masks)
            loss = (
                args.camera_weight * sum(cam_losses.values())
                + args.depth_weight * loss_depth
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            if step % args.log_every == 0:
                parts = [f"step={step}", f"loss={loss.item():.4f}", f"loss_depth={loss_depth.item():.4f}"]
                parts.extend(f"{k}={v.item():.4f}" for k, v in cam_losses.items())
                print(" ".join(parts), flush=True)
            if args.dry_run == "model":
                print("model_dry_run_ok", flush=True)
                return
            step += 1
            if args.max_steps and step >= args.max_steps:
                save_checkpoint(args, model, step)
                return
    save_checkpoint(args, model, step)


def save_checkpoint(args: argparse.Namespace, model: torch.nn.Module, step: int) -> None:
    if not args.output:
        return
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, output)
    print(f"saved={output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--co3d-root", default=str(PROJECT_ROOT / "dataset" / "co3d_apple"))
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "vggt_omega_1b_512.pt"))
    parser.add_argument("--output", default=str(REPO_ROOT / "checkpoints" / "vggt_omega_co3d_apple_smoke.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--camera-weight", type=float, default=1.0)
    parser.add_argument("--depth-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze-aggregator", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", choices=["none", "data", "model"], default="none")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
