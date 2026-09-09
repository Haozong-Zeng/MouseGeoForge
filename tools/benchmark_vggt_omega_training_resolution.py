#!/usr/bin/env python3
"""Measure VGGT-Omega training memory for a specific rectangular resolution."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT / "vggt-omega"
TRAINING_SCRIPT = REPO_ROOT / "training" / "finetune_blendedmvs.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location("mgf_finetune_blendedmvs", TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--real-batch-size", type=int, default=8)
    parser.add_argument("--train-mode", choices=["light_heads", "dense", "heads"], default="dense")
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "vggt_omega_1b_512.pt"))
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def gib(value: int) -> float:
    return value / (1024**3)


def dense_surrogate_loss(predictions: dict[str, torch.Tensor]) -> torch.Tensor:
    depth = predictions["depth"].float()
    confidence = predictions["depth_conf"].float()
    return depth.square().mean() + 1e-3 * confidence.square().mean()


def main() -> None:
    args = parse_args()
    if args.width % 16 or args.height % 16:
        raise ValueError("Width and height must both be divisible by the patch size 16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    training = load_training_module()
    device = torch.device("cuda")
    model_args = SimpleNamespace(
        checkpoint=args.checkpoint,
        adapter_checkpoint="",
        train_mode=args.train_mode,
        train_camera_register_tokens=False,
        unfreeze_last_blocks=0,
    )
    model = training.load_model(model_args, device)
    trainable_names = training.configure_trainable(model, model_args)
    frozen_aggregator = training.aggregator_is_frozen(model)
    model.train()

    optimizer_args = SimpleNamespace(
        lr=args.lr,
        aggregator_lr_mult=0.1,
        weight_decay=0.01,
    )
    optimizer = training.make_optimizer(model, trainable_names, optimizer_args)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    try:
        synthetic_images = torch.rand(
            args.batch_size,
            args.views,
            3,
            args.height,
            args.width,
            device=device,
        )
        synthetic_predictions = training.forward_for_training(
            model,
            synthetic_images,
            frozen_aggregator,
        )
        loss = dense_surrogate_loss(synthetic_predictions)

        if args.real_batch_size > 0:
            real_images = torch.rand(
                args.real_batch_size,
                1,
                3,
                args.height,
                args.width,
                device=device,
            )
            real_predictions = training.forward_for_training(
                model,
                real_images,
                frozen_aggregator,
            )
            loss = loss + dense_surrogate_loss(real_predictions)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        status = "ok"
        error = ""
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.synchronize()
        status = "oom"
        error = str(exc)

    elapsed = time.perf_counter() - started
    result = {
        "status": status,
        "error": error,
        "gpu": torch.cuda.get_device_name(),
        "width": args.width,
        "height": args.height,
        "patches_per_view": (args.width // 16) * (args.height // 16),
        "synthetic_batch_size": args.batch_size,
        "synthetic_views": args.views,
        "real_batch_size": args.real_batch_size,
        "train_mode": args.train_mode,
        "aggregator_frozen": frozen_aggregator,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "baseline_allocated_gib": round(gib(baseline_allocated), 3),
        "baseline_reserved_gib": round(gib(baseline_reserved), 3),
        "peak_allocated_gib": round(gib(torch.cuda.max_memory_allocated()), 3),
        "peak_reserved_gib": round(gib(torch.cuda.max_memory_reserved()), 3),
        "incremental_peak_allocated_gib": round(
            gib(torch.cuda.max_memory_allocated() - baseline_allocated),
            3,
        ),
        "elapsed_seconds": round(elapsed, 3),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if status != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
